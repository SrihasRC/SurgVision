# Router: live MJPEG streaming with YOLO overlay + voice command support

import cv2
import uuid
import shutil
import threading
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Query, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import DEFAULT_MODEL, DEFAULT_CONF, DEFAULT_DEVICE
from src.services.yolo_service import model_manager

router = APIRouter(prefix="/stream", tags=["Streaming"])

# ─── Session state ───────────────────────────────────────────────

# Cancel events per session
_sessions: dict[str, threading.Event] = {}

# Per-session display config (updated by voice commands)
_session_configs: dict[str, dict] = {}

# Keyword → class name mapping for fuzzy voice matching
CLASS_KEYWORDS: dict[str, str] = {
    "artery": "External Iliac Artery",
    "iliac artery": "External Iliac Artery",
    "external iliac artery": "External Iliac Artery",
    "vein": "External Iliac Vein",
    "iliac vein": "External Iliac Vein",
    "external iliac vein": "External Iliac Vein",
    "instrument": "Instrument",
    "tool": "Instrument",
    "nerve": "Obturator Nerve",
    "obturator": "Obturator Nerve",
    "obturator nerve": "Obturator Nerve",
    "ovary": "Ovary",
    "ureter": "Ureter",
    "uterine": "Uterine Artery",
    "uterine artery": "Uterine Artery",
    "uterus": "uterus",
}


def _default_config() -> dict:
    return {
        "show_masks": True,
        "show_labels": True,
        "hidden_classes": set(),  # class names to HIDE
    }


# ─── MJPEG generator ─────────────────────────────────────────────


def _generate_mjpeg_frames(
    video_path: str,
    model_name: str,
    conf: float,
    session_id: str,
):
    """Generator that yields MJPEG frames from a video file with YOLO overlay."""
    cancel_event = _sessions.get(session_id)
    model = model_manager.get_model(model_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        error_frame = _create_text_frame("Failed to open video file.", 640, 480)
        _, buffer = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        _cleanup_session(session_id, video_path)
        return

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while True:
            if cancel_event and cancel_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Read current session config
            cfg = _session_configs.get(session_id, _default_config())

            # Run inference
            results = model.predict(
                frame,
                conf=conf,
                device=DEFAULT_DEVICE,
                verbose=False,
                retina_masks=True,
            )[0]

            # Determine which detections to show
            hidden = cfg.get("hidden_classes", set())
            show_masks = cfg.get("show_masks", True)
            show_labels = cfg.get("show_labels", True)

            if show_masks and results.masks is not None:
                # Filter: build mask indices to keep
                keep_indices = []
                for i, box in enumerate(results.boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name not in hidden:
                        keep_indices.append(i)

                if keep_indices and len(keep_indices) < len(results.boxes):
                    # Partial filter — plot only kept detections
                    import torch

                    filtered = results[torch.tensor(keep_indices)]
                    annotated = filtered.plot(
                        boxes=False, labels=False, conf=False
                    )
                    if show_labels:
                        _draw_labels(annotated, filtered, model.names)
                elif keep_indices:
                    # Show all (nothing hidden)
                    annotated = results.plot(
                        boxes=False, labels=False, conf=False
                    )
                    if show_labels:
                        _draw_labels(annotated, results, model.names)
                else:
                    # All hidden — show raw frame
                    annotated = frame.copy()
            else:
                # Masks off — show raw frame
                annotated = frame.copy()

            # Progress bar
            h, w = annotated.shape[:2]
            progress = frame_idx / max(total_frames, 1)
            bar_y = h - 6
            cv2.rectangle(annotated, (0, bar_y), (w, h), (30, 30, 30), -1)
            cv2.rectangle(
                annotated, (0, bar_y), (int(w * progress), h), (0, 200, 180), -1
            )

            _, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()
        _cleanup_session(session_id, video_path)


def _draw_labels(frame, results, names):
    """Draw class name labels at bounding box center."""
    if results.masks is None:
        return
    for i, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        class_name = names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        lx = (x1 + x2) // 2
        ly = (y1 + y2) // 2
        (tw, th), _ = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), (0, 0, 0), -1
        )
        cv2.putText(
            frame, class_name, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )


def _cleanup_session(session_id: str, video_path: str):
    _sessions.pop(session_id, None)
    _session_configs.pop(session_id, None)
    try:
        Path(video_path).unlink(missing_ok=True)
    except Exception:
        pass


def _create_text_frame(text: str, width: int, height: int):
    import numpy as np

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
    cv2.putText(
        frame, text, ((width - tw) // 2, (height + th) // 2),
        font, 0.6, (200, 200, 200), 1,
    )
    return frame


# ─── Endpoints ────────────────────────────────────────────────────

# Stores temp video path per session for the feed endpoint
_session_videos: dict[str, str] = {}
_session_model: dict[str, tuple[str, float]] = {}


@router.post("/create", summary="Upload video and create a streaming session")
async def create_stream(
    file: UploadFile = File(...),
    model_name: str = Query(default=DEFAULT_MODEL),
    conf: float = Query(default=DEFAULT_CONF, ge=0.05, le=0.95),
):
    """
    Upload a video file and receive a session_id + feed URL.
    Use GET /stream/feed/{session_id} to consume the MJPEG stream.
    """
    suffix = Path(file.filename or "video.mp4").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()

    session_id = str(uuid.uuid4())
    _sessions[session_id] = threading.Event()
    _session_configs[session_id] = _default_config()
    _session_videos[session_id] = tmp.name
    _session_model[session_id] = (model_name, conf)

    return {
        "session_id": session_id,
        "feed_url": f"/stream/feed/{session_id}",
    }


@router.get("/feed/{session_id}", summary="MJPEG stream for a session")
def feed_stream(session_id: str):
    """Serve the MJPEG stream for a previously created session."""
    video_path = _session_videos.get(session_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Session not found")

    model_name, conf = _session_model.get(session_id, (DEFAULT_MODEL, DEFAULT_CONF))

    return StreamingResponse(
        _generate_mjpeg_frames(video_path, model_name, conf, session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Keep the old /start endpoint for backward compat with /live page
@router.post("/start", summary="Upload video and start MJPEG stream (legacy)")
async def start_stream(
    file: UploadFile = File(...),
    model_name: str = Query(default=DEFAULT_MODEL),
    conf: float = Query(default=DEFAULT_CONF, ge=0.05, le=0.95),
):
    suffix = Path(file.filename or "video.mp4").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()

    session_id = str(uuid.uuid4())
    _sessions[session_id] = threading.Event()
    _session_configs[session_id] = _default_config()

    return StreamingResponse(
        _generate_mjpeg_frames(tmp.name, model_name, conf, session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Session-Id": session_id},
    )


@router.post("/stop/{session_id}", summary="Stop an active stream")
def stop_stream(session_id: str):
    event = _sessions.get(session_id)
    if event:
        event.set()
        return {"status": "stopped", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found or already ended")


# ─── Voice command endpoint ──────────────────────────────────────


class VoiceCommand(BaseModel):
    command: str  # "show_masks", "hide_masks", "show_labels", "hide_labels",
    #              "show_class", "hide_class", "show_all", "stop"
    class_name: Optional[str] = None  # for show_class / hide_class


def _resolve_class(keyword: Optional[str]) -> Optional[str]:
    """Fuzzy-match a keyword to a model class name."""
    if not keyword:
        return None
    kw = keyword.strip().lower()
    # Direct keyword match
    if kw in CLASS_KEYWORDS:
        return CLASS_KEYWORDS[kw]
    # Partial match — find the best keyword that the spoken text contains
    for key, class_name in CLASS_KEYWORDS.items():
        if key in kw or kw in key:
            return class_name
    return None


@router.post(
    "/command/{session_id}",
    summary="Send a voice command to an active stream",
)
def send_command(session_id: str, cmd: VoiceCommand):
    """
    Update the display config for an active streaming session.
    Returns the confirmation text to be spoken via TTS.
    """
    cfg = _session_configs.get(session_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Session not found")

    confirmation = ""

    if cmd.command == "show_masks":
        cfg["show_masks"] = True
        confirmation = "Masks visible"

    elif cmd.command == "hide_masks":
        cfg["show_masks"] = False
        confirmation = "Masks hidden"

    elif cmd.command == "show_labels":
        cfg["show_labels"] = True
        confirmation = "Labels visible"

    elif cmd.command == "hide_labels":
        cfg["show_labels"] = False
        confirmation = "Labels hidden"

    elif cmd.command == "show_class":
        resolved = _resolve_class(cmd.class_name)
        if resolved:
            cfg["hidden_classes"].discard(resolved)
            confirmation = f"Showing {resolved}"
        else:
            confirmation = f"Unknown class: {cmd.class_name}"

    elif cmd.command == "hide_class":
        resolved = _resolve_class(cmd.class_name)
        if resolved:
            cfg["hidden_classes"].add(resolved)
            confirmation = f"Hiding {resolved}"
        else:
            confirmation = f"Unknown class: {cmd.class_name}"

    elif cmd.command == "show_all":
        cfg["hidden_classes"] = set()
        cfg["show_masks"] = True
        cfg["show_labels"] = True
        confirmation = "Showing all classes"

    elif cmd.command == "stop":
        event = _sessions.get(session_id)
        if event:
            event.set()
        confirmation = "Inference stopped"

    else:
        confirmation = f"Unknown command: {cmd.command}"

    return {
        "status": "ok",
        "confirmation": confirmation,
        "config": {
            "show_masks": cfg["show_masks"],
            "show_labels": cfg["show_labels"],
            "hidden_classes": list(cfg["hidden_classes"]),
        },
    }


@router.get("/sessions", summary="List active streaming sessions")
def list_sessions():
    return {"active_sessions": list(_sessions.keys())}
