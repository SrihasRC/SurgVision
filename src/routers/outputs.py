# Router: serve processed video files from server_output/

import os
import cv2
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from src.config import PROJECT_ROOT

router = APIRouter(prefix="/outputs", tags=["Outputs"])

# Folder where GPU server outputs are stored
SERVER_OUTPUT_DIR = PROJECT_ROOT / "server_output"


def _get_video_files() -> list[dict]:
    """List video files in the server_output directory."""
    if not SERVER_OUTPUT_DIR.exists():
        SERVER_OUTPUT_DIR.mkdir(exist_ok=True)
        return []

    extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    files = []
    for f in sorted(SERVER_OUTPUT_DIR.iterdir()):
        if f.suffix.lower() in extensions and f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": str(
                    __import__("datetime").datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                ),
                "thumbnail_url": f"/outputs/{f.name}/thumbnail",
                "video_url": f"/outputs/{f.name}",
            })
    return files


@router.get("/", summary="List output video files")
def list_output_files():
    """Returns metadata for all video files in server_output/."""
    return _get_video_files()


@router.get("/{filename}", summary="Serve a video file")
def serve_video(filename: str):
    """Stream a video file from server_output/."""
    file_path = SERVER_OUTPUT_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    media_types = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    media_type = media_types.get(file_path.suffix.lower(), "video/mp4")
    return FileResponse(str(file_path), media_type=media_type, filename=filename)


@router.get("/{filename}/thumbnail", summary="Get video thumbnail")
def get_thumbnail(filename: str):
    """Extract and return the first frame of a video as a JPEG thumbnail."""
    file_path = SERVER_OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    cap = cv2.VideoCapture(str(file_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read video frame")

    # Resize for thumbnail
    h, w = frame.shape[:2]
    thumb_w = 640
    thumb_h = int(thumb_w * h / w)
    thumb = cv2.resize(frame, (thumb_w, thumb_h))

    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
    )
