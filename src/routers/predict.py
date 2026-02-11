# Router: prediction endpoints (image, video, distances)

import cv2
import base64
import time
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse

from src.config import (
    DEFAULT_MODEL, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_DEVICE,
    UPLOAD_DIR, OUTPUT_DIR,
)
from src.services.yolo_service import model_manager, predict_image, predict_video
from src.services.distance_service import (
    calculate_frame_distances, draw_distances_on_frame,
)
from src.models import PredictionResponse, DistancePredictionResponse, VideoResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into an OpenCV image."""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


@router.post("/image", response_model=PredictionResponse, summary="Predict on an image")
async def predict_on_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    model_name: str = Query(default=DEFAULT_MODEL, description="Model name"),
    conf: float = Query(default=DEFAULT_CONF, ge=0.01, le=1.0, description="Confidence threshold"),
    iou: float = Query(default=DEFAULT_IOU, ge=0.01, le=1.0, description="IOU threshold"),
    annotate: bool = Query(default=True, description="Return annotated image"),
):
    """
    Upload an image and get YOLO detection/segmentation results.
    Returns bounding boxes, class names, confidences, and optionally
    a base64-encoded annotated image.
    """
    try:
        image = _read_image_from_upload(file)
        result = predict_image(
            image, model_name, conf=conf, iou=iou,
            device=DEFAULT_DEVICE, return_annotated=annotate,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/image/distances",
    response_model=DistancePredictionResponse,
    summary="Predict with distance calculation",
)
async def predict_with_distances(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    model_name: str = Query(default=DEFAULT_MODEL, description="Model name"),
    conf: float = Query(default=DEFAULT_CONF, ge=0.01, le=1.0),
    iou: float = Query(default=DEFAULT_IOU, ge=0.01, le=1.0),
    topk: int = Query(default=7, ge=1, le=20, description="Top K closest structures"),
    min_organ_conf: float = Query(default=0.20, ge=0.01, le=1.0),
    min_mask_area: int = Query(default=100, ge=0),
):
    """
    Upload an image and get instrument-to-organ distance measurements.
    Returns detections + distance measurements + annotated image with lines.
    Requires a segmentation model.
    """
    try:
        image = _read_image_from_upload(file)
        model = model_manager.get_model(model_name)

        t0 = time.time()
        results = model.predict(
            image, conf=conf, iou=iou,
            device=DEFAULT_DEVICE, verbose=False, retina_masks=True,
        )[0]
        inference_ms = (time.time() - t0) * 1000

        h, w = image.shape[:2]
        detections = []
        distances = []

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            # Build detections list
            for i in range(len(results.boxes)):
                det = {
                    "class_name": model.names[int(classes[i])],
                    "class_id": int(classes[i]),
                    "confidence": round(float(confs[i]), 4),
                    "bbox": [round(float(x), 2) for x in results.boxes.xyxy[i].tolist()],
                }
                detections.append(det)

            # Calculate distances
            distances = calculate_frame_distances(
                masks, classes, confs, model.names,
                min_mask_area=min_mask_area,
                min_organ_conf=min_organ_conf,
                topk=topk,
            )

        # Draw annotations
        annotated = results.plot()
        if distances:
            annotated = draw_distances_on_frame(annotated, distances)

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buffer).decode("utf-8")

        return {
            "model_name": model_name,
            "image_width": w,
            "image_height": h,
            "detections": detections,
            "distances": distances,
            "annotated_image_base64": b64_image,
            "inference_time_ms": round(inference_ms, 2),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/video", response_model=VideoResponse, summary="Process a video")
async def predict_on_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, etc.)"),
    model_name: str = Query(default=DEFAULT_MODEL, description="Model name"),
    conf: float = Query(default=DEFAULT_CONF, ge=0.01, le=1.0),
    iou: float = Query(default=DEFAULT_IOU, ge=0.01, le=1.0),
):
    """
    Upload a video and get a processed video with YOLO annotations.
    Returns the output video file path and processing stats.
    """
    try:
        # Save uploaded video
        input_path = str(UPLOAD_DIR / file.filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        output_filename = f"{Path(file.filename).stem}_predicted.mp4"
        output_path = str(OUTPUT_DIR / output_filename)

        result = predict_video(
            input_path, output_path, model_name,
            conf=conf, iou=iou, device=DEFAULT_DEVICE,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/video/download/{filename}", summary="Download processed video")
async def download_video(filename: str):
    """Download a previously processed video by filename."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        filename=filename,
    )
