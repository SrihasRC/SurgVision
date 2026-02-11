# FastAPI main entry point

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODEL_REGISTRY, DEFAULT_MODEL
from src.routers import predict, models, outputs, stream, tts
from src.services.yolo_service import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload the default model at startup."""
    print("\n" + "=" * 60)
    print("  YOLO Inference API — Starting up")
    print("=" * 60)
    print(f"\n📂 Discovered {len(MODEL_REGISTRY)} models:")
    for name, info in MODEL_REGISTRY.items():
        print(f"   • {name} ({info.size_mb} MB) — {info.path}")

    if DEFAULT_MODEL:
        model_manager.preload(DEFAULT_MODEL)
    else:
        print("⚠️  No models found! Place .pt files in the project directory.")

    print("\n🚀 API ready!\n")
    yield
    print("\n👋 Shutting down YOLO API")


app = FastAPI(
    title="YOLO Inference API",
    description=(
        "FastAPI backend for YOLO segmentation and detection models. "
        "Supports image prediction, video processing, and surgical "
        "instrument distance calculation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

# Include routers
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(outputs.router)
app.include_router(stream.router)
app.include_router(tts.router)


@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "models_available": len(MODEL_REGISTRY),
        "default_model": DEFAULT_MODEL,
    }
