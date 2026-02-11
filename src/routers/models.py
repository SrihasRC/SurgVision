# Router: model management endpoints

from fastapi import APIRouter, HTTPException

from src.services.yolo_service import model_manager
from src.models import ModelInfoResponse

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/", summary="List all available YOLO models")
def list_models():
    """Returns a list of all discovered .pt model files."""
    return model_manager.list_models()


@router.get("/{model_name}/info", response_model=ModelInfoResponse, summary="Get model details")
def get_model_info(model_name: str):
    """Load a model and return its class names, task type, and metadata."""
    try:
        info = model_manager.get_model_info(model_name)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
