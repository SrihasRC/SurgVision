# Router: live MJPEG streaming from uploaded video with YOLO overlay

import cv2
import uuid
import shutil
import threading
import tempfile
from pathlib import Path
from fastapi import APIRouter, Query, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from src.config import DEFAULT_MODEL, DEFAULT_CONF, DEFAULT_DEVICE
from src.services.yolo_service import model_manager

router = APIRouter(prefix="/stream", tags=["Streaming"])

# Active streaming sessions — maps session_id -> cancel event
_sessions: dict[str, threading.Event] = {}


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
            # Check if cancelled
            if cancel_event and cancel_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run inference
            results = model.predict(
                frame,
                conf=conf,
                device=DEFAULT_DEVICE,
                verbose=False,
                retina_masks=True,
            )[0]

            annotated = results.plot()

            # Add progress bar to frame
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


def _cleanup_session(session_id: str, video_path: str):
    """Remove session and temp file."""
    _sessions.pop(session_id, None)
    try:
        Path(video_path).unlink(missing_ok=True)
    except Exception:
        pass


def _create_text_frame(text: str, width: int, height: int):
    """Create a frame with centered text."""
    import numpy as np

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, (200, 200, 200), thickness)
    return frame


@router.post("/start", summary="Upload video and start MJPEG stream")
async def start_stream(
    file: UploadFile = File(...),
    model_name: str = Query(default=DEFAULT_MODEL),
    conf: float = Query(default=DEFAULT_CONF, ge=0.05, le=0.95),
):
    """
    Upload a video file and receive an MJPEG stream with YOLO overlay.
    Returns a session_id in headers for cancellation.
    """
    # Save uploaded file to temp path
    suffix = Path(file.filename or "video.mp4").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()

    session_id = str(uuid.uuid4())
    _sessions[session_id] = threading.Event()

    return StreamingResponse(
        _generate_mjpeg_frames(tmp.name, model_name, conf, session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Session-Id": session_id},
    )


@router.post("/stop/{session_id}", summary="Stop an active stream")
def stop_stream(session_id: str):
    """Signal a streaming session to stop."""
    event = _sessions.get(session_id)
    if event:
        event.set()
        return {"status": "stopped", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found or already ended")


@router.get("/sessions", summary="List active streaming sessions")
def list_sessions():
    """Return IDs of currently active streaming sessions."""
    return {"active_sessions": list(_sessions.keys())}
