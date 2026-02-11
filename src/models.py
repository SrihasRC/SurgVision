# Pydantic schemas for request/response models

from pydantic import BaseModel, Field
from typing import Optional


# ---- Response models ----

class HealthResponse(BaseModel):
    status: str = "ok"
    models_available: int
    default_model: Optional[str]


class ModelInfoResponse(BaseModel):
    name: str
    path: str
    size_mb: float
    directory: str
    class_names: Optional[dict[int, str]] = None
    task: Optional[str] = None


class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: list[float] = Field(description="[x1, y1, x2, y2]")
    mask_polygon: Optional[list[list[float]]] = Field(
        default=None, description="Polygon points [[x,y], ...] for segmentation"
    )


class DistanceMeasurement(BaseModel):
    organ: str
    distance_px: float
    status: str = Field(description="SAFE, CAUTION, or DANGER")
    instrument_point: list[int] = Field(description="[x, y]")
    organ_point: list[int] = Field(description="[x, y]")
    organ_confidence: float


class PredictionResponse(BaseModel):
    model_name: str
    image_width: int
    image_height: int
    detections: list[Detection]
    annotated_image_base64: Optional[str] = Field(
        default=None, description="Base64-encoded annotated image (JPEG)"
    )
    inference_time_ms: float


class DistancePredictionResponse(BaseModel):
    model_name: str
    image_width: int
    image_height: int
    detections: list[Detection]
    distances: list[DistanceMeasurement]
    annotated_image_base64: Optional[str] = Field(
        default=None, description="Base64-encoded annotated image with distance lines"
    )
    inference_time_ms: float


class VideoResponse(BaseModel):
    model_name: str
    total_frames: int
    output_path: str
    processing_time_s: float
