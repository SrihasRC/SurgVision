# Config: auto-discovers .pt models and provides settings for the API

import os
from pathlib import Path
from dataclasses import dataclass, field

# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to scan for .pt model files
MODEL_SEARCH_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "medhack_yolo",
    PROJECT_ROOT / "medh_v2",
    PROJECT_ROOT / "medhack_yolov11",
]

# Upload directory for temporary files
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Default inference parameters
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_DEVICE = "0"  # GPU 0; use "cpu" for CPU-only
DEFAULT_IMG_SIZE = 640

# Distance calculation settings
DISTANCE_THRESHOLDS = {"safe": 50, "caution": 30, "danger": 30}
INSTRUMENT_KEYWORDS = ["instrument", "tool", "forceps", "grasper", "scissors"]


@dataclass
class ModelInfo:
    name: str
    path: str
    size_mb: float
    directory: str


def discover_models() -> dict[str, ModelInfo]:
    """Scan project directories for .pt model files."""
    models: dict[str, ModelInfo] = {}
    seen_paths: set[str] = set()

    for search_dir in MODEL_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for pt_file in search_dir.glob("*.pt"):
            abs_path = str(pt_file.resolve())
            if abs_path in seen_paths:
                continue
            seen_paths.add(abs_path)

            name = pt_file.stem
            # Disambiguate duplicate names from different dirs
            if name in models:
                parent = pt_file.parent.name
                name = f"{parent}_{name}"

            models[name] = ModelInfo(
                name=name,
                path=abs_path,
                size_mb=round(pt_file.stat().st_size / (1024 * 1024), 2),
                directory=str(pt_file.parent),
            )

    return models


# Discover on import
MODEL_REGISTRY = discover_models()
DEFAULT_MODEL = "26n" if "26n" in MODEL_REGISTRY else next(iter(MODEL_REGISTRY), None)
