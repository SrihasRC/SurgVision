# Router: serve processed video files from server_output/ with subfolder support

import datetime
import cv2
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from src.config import PROJECT_ROOT

router = APIRouter(prefix="/outputs", tags=["Outputs"])

SERVER_OUTPUT_DIR = PROJECT_ROOT / "server_output"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def _safe_resolve(subpath: str) -> Path:
    """Resolve subpath under SERVER_OUTPUT_DIR, preventing path traversal."""
    resolved = (SERVER_OUTPUT_DIR / subpath).resolve()
    if not str(resolved).startswith(str(SERVER_OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    return resolved


@router.get("/", summary="List folders and files in server_output")
def list_output_contents(path: str = Query(default="", description="Subfolder path")):
    """
    Lists subfolders and video files at the given path within server_output/.
    Returns { folders: [...], files: [...], current_path: "..." }
    """
    if not SERVER_OUTPUT_DIR.exists():
        SERVER_OUTPUT_DIR.mkdir(exist_ok=True)

    target = _safe_resolve(path)
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {path}")

    folders = []
    files = []

    for item in sorted(target.iterdir()):
        rel = str(item.relative_to(SERVER_OUTPUT_DIR))
        if item.is_dir():
            # Count videos inside (non-recursive, just immediate)
            video_count = sum(
                1 for f in item.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            child_folders = sum(1 for f in item.iterdir() if f.is_dir())
            folders.append({
                "name": item.name,
                "path": rel,
                "video_count": video_count,
                "subfolder_count": child_folders,
            })
        elif item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
            stat = item.stat()
            files.append({
                "name": item.name,
                "path": rel,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "thumbnail_url": f"/outputs/thumbnail/{rel}",
                "video_url": f"/outputs/file/{rel}",
            })

    return {
        "current_path": path or "",
        "folders": folders,
        "files": files,
    }


@router.get("/file/{filepath:path}", summary="Serve a video file")
def serve_video(filepath: str):
    """Stream a video file from server_output/."""
    file_path = _safe_resolve(filepath)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Not found: {filepath}")

    media_types = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    media_type = media_types.get(file_path.suffix.lower(), "video/mp4")
    return FileResponse(str(file_path), media_type=media_type, filename=file_path.name)


@router.get("/thumbnail/{filepath:path}", summary="Get video thumbnail")
def get_thumbnail(filepath: str):
    """Extract first frame of a video as JPEG thumbnail."""
    file_path = _safe_resolve(filepath)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {filepath}")

    cap = cv2.VideoCapture(str(file_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read video frame")

    h, w = frame.shape[:2]
    thumb_w = 640
    thumb_h = int(thumb_w * h / w)
    thumb = cv2.resize(frame, (thumb_w, thumb_h))

    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
