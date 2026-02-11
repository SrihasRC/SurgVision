# Router: live MJPEG video streaming with YOLO overlay

import cv2
import time
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from src.config import DEFAULT_MODEL, DEFAULT_CONF, DEFAULT_DEVICE
from src.services.yolo_service import model_manager

router = APIRouter(prefix="/stream", tags=["Streaming"])


def _generate_mjpeg_frames(
    model_name: str,
    conf: float,
    source: int = 0,
):
    """Generator that yields MJPEG frames from a video source with YOLO overlay."""
    model = model_manager.get_model(model_name)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        # Yield a single error frame
        error_frame = _create_text_frame(
            "No video source available. Check webcam connection.", 640, 480
        )
        _, buffer = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video or stop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            # Run inference
            results = model.predict(
                frame,
                conf=conf,
                device=DEFAULT_DEVICE,
                verbose=False,
                retina_masks=True,
            )[0]

            annotated = results.plot()

            _, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()


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


@router.get("/live", summary="Live MJPEG stream with YOLO overlay")
def live_stream(
    model_name: str = Query(default=DEFAULT_MODEL, description="Model name"),
    conf: float = Query(default=DEFAULT_CONF, ge=0.05, le=0.95),
    source: int = Query(default=0, description="Video source index (0 = webcam)"),
):
    """
    Returns an MJPEG stream from the specified video source
    with real-time YOLO segmentation overlay.
    Use as <img src="/stream/live?model_name=26n"> in the frontend.
    """
    return StreamingResponse(
        _generate_mjpeg_frames(model_name, conf, source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
