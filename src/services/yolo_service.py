# YOLO Service: model loading, caching, and inference

import time
import base64
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.config import MODEL_REGISTRY, DEFAULT_DEVICE, DEFAULT_IMG_SIZE


class ModelManager:
    """Loads and caches YOLO models for reuse across requests."""

    def __init__(self):
        self._cache: dict[str, YOLO] = {}

    def get_model(self, model_name: str) -> YOLO:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        if model_name not in self._cache:
            info = MODEL_REGISTRY[model_name]
            print(f"📦 Loading model: {info.name} ({info.path})")
            self._cache[model_name] = YOLO(info.path)
        return self._cache[model_name]

    def get_model_info(self, model_name: str) -> dict:
        model = self.get_model(model_name)
        info = MODEL_REGISTRY[model_name]
        return {
            "name": info.name,
            "path": info.path,
            "size_mb": info.size_mb,
            "directory": info.directory,
            "class_names": model.names,
            "task": model.task if hasattr(model, "task") else None,
        }

    def list_models(self) -> list[dict]:
        return [
            {
                "name": m.name,
                "path": m.path,
                "size_mb": m.size_mb,
                "directory": m.directory,
            }
            for m in MODEL_REGISTRY.values()
        ]

    def preload(self, model_name: str):
        """Preload a model into cache (called at startup)."""
        try:
            self.get_model(model_name)
            print(f"✅ Preloaded model: {model_name}")
        except Exception as e:
            print(f"⚠️  Failed to preload {model_name}: {e}")


# Singleton
model_manager = ModelManager()


def predict_image(
    image: np.ndarray,
    model_name: str,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = DEFAULT_DEVICE,
    return_annotated: bool = True,
) -> dict:
    """
    Run YOLO inference on a single image.
    Returns detections and optionally a base64 annotated image.
    """
    model = model_manager.get_model(model_name)

    t0 = time.time()
    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        retina_masks=True,
    )[0]
    inference_ms = (time.time() - t0) * 1000

    h, w = image.shape[:2]
    detections = []

    if results.boxes is not None:
        boxes = results.boxes
        for i in range(len(boxes)):
            det = {
                "class_name": model.names[int(boxes.cls[i])],
                "class_id": int(boxes.cls[i]),
                "confidence": round(float(boxes.conf[i]), 4),
                "bbox": [round(float(x), 2) for x in boxes.xyxy[i].tolist()],
            }
            # Add mask polygon if segmentation model
            if results.masks is not None:
                mask = results.masks.data[i].cpu().numpy()
                contours, _ = cv2.findContours(
                    (mask * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    # Use largest contour
                    largest = max(contours, key=cv2.contourArea)
                    # Simplify polygon
                    epsilon = 0.01 * cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, epsilon, True)
                    det["mask_polygon"] = approx.reshape(-1, 2).tolist()

            detections.append(det)

    response = {
        "model_name": model_name,
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "inference_time_ms": round(inference_ms, 2),
    }

    if return_annotated:
        annotated = results.plot()
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        response["annotated_image_base64"] = base64.b64encode(buffer).decode("utf-8")

    return response


def predict_video(
    video_path: str,
    output_path: str,
    model_name: str,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """Process a video file frame-by-frame and save annotated output."""
    model = model_manager.get_model(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    t0 = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                frame,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
                retina_masks=True,
            )[0]

            annotated = results.plot()
            writer.write(annotated)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    return {
        "model_name": model_name,
        "total_frames": frame_count,
        "output_path": output_path,
        "processing_time_s": round(time.time() - t0, 2),
    }
