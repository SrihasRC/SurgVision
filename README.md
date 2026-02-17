# AI-Powered Surgical Assistance Platform

An intelligent surgical video analysis system that provides real-time segmentation, distance monitoring, and voice-enabled assistance for surgical procedures. Built to enhance surgical safety through computer vision and smart alerting.

## 🎯 Overview

This platform uses state-of-the-art YOLOv11 segmentation models to identify and track critical anatomical structures and surgical instruments in real-time during procedures. The system calculates precise distances between instruments and sensitive organs, providing visual and voice alerts to help prevent unintended tissue damage.

## ✨ Key Features

### 🔍 Real-Time Segmentation
- **Multi-structure detection**: Identifies organs, vessels, nerves, and surgical instruments
- **YOLOv11-powered**: Leverages advanced instance segmentation for precise boundary detection
- **High-accuracy tracking**: Trained on specialized surgical datasets for medical applications
- **Multiple model support**: Switch between different model weights for varied surgical procedures

### 📏 Safety Distance Monitoring
- **Automatic distance calculation**: Real-time measurement between instruments and organs
- **Three-tier alert system**:
  - 🟢 **SAFE**: Adequate clearance (>60px)
  - 🟡 **CAUTION**: Moderate proximity (30-60px)
  - 🔴 **DANGER**: Critical proximity (<30px)
- **Visual overlays**: Color-coded masks and distance lines rendered directly on video
- **Boundary-to-boundary precision**: Calculates minimum Euclidean distance between segmentation contours

### 🎙️ Voice-Enabled Assistance
- **Event-based alerts**: Non-intrusive announcements for structure detection and disappearance
  - "Ureter identified"
  - "External iliac artery in view"
  - "Ureter no longer visible"
- **Hands-free voice commands**:
  - `"hide overlays"` / `"show masks"` — Toggle all visual overlays
  - `"show ureter"` / `"show nerve"` — Filter specific structures
  - `"show vessels"` — Display arteries and veins only
  - `"pause AI"` / `"resume AI"` — Control detection
  - `"what structures are visible?"` — On-demand status query
- **Configurable reannouncement**: Smart timeout prevents alert fatigue
- **Offline TTS**: Uses Piper neural TTS for low-latency, privacy-preserving speech

### 🖥️ Modern Web Dashboard
- **Next.js frontend**: Responsive, type-safe UI built with TypeScript and shadcn/ui
- **Live video streaming**: Real-time inference visualization via WebRTC/HTTP streams
- **Model management**: Browse, switch, and inspect available YOLO models
- **Output gallery**: Review saved predictions, videos, and analysis sessions
- **Health monitoring**: Backend connection status and system diagnostics
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation

### 🎥 Video Processing
- **Batch video analysis**: Process pre-recorded surgical videos with full annotation
- **Frame-by-frame inference**: Extract predictions with timestamps and metadata
- **Smoothing algorithms**: Temporal filtering to reduce detection jitter
- **Export capabilities**: Save annotated videos and JSON outputs for review

## 🏗️ Architecture

### Backend (FastAPI)
- **Model management**: Dynamic loading of YOLO `.pt` weights with automatic discovery
- **Inference service**: Optimized YOLO inference pipeline with GPU acceleration support
- **Distance calculation**: Geometric analysis using OpenCV and SciPy
- **Video streaming**: Session-based streaming for live camera feeds
- **TTS integration**: Piper ONNX models for voice synthesis

### Frontend (Next.js 15)
- **Modern React**: App router with server/client components
- **shadcn/ui**: Accessible, customizable UI components
- **Type-safe API client**: Strongly-typed HTTP client for backend communication
- **Responsive design**: Mobile-first layout with Tailwind CSS

### AI/ML Stack
- **YOLOv11 Segmentation**: Instance segmentation trained for surgical instruments and anatomy
- **Vosk Speech Recognition**: Offline voice command processing
- **Piper Neural TTS**: High-quality text-to-speech synthesis
- **OpenCV & NumPy**: Image processing and geometric calculations

## 📂 Project Structure

```
├── frontend/              # Next.js web dashboard
│   ├── app/              # App router pages
│   │   ├── live/         # Live video streaming
│   │   ├── analysis/     # Batch video analysis
│   │   ├── models/       # Model management
│   │   └── outputs/      # Results gallery
│   ├── components/       # React components & UI
│   └── lib/              # API client & utilities
│
├── src/                  # FastAPI backend
│   ├── routers/          # API endpoints
│   │   ├── predict.py    # Image inference
│   │   ├── stream.py     # Video streaming
│   │   ├── models.py     # Model management
│   │   ├── outputs.py    # Results retrieval
│   │   └── tts.py        # Text-to-speech
│   └── services/         # Core logic
│       ├── yolo_service.py       # YOLO inference
│       └── distance_service.py   # Distance calculation
│
├── medhack_yolo/         # Voice-enabled inference & training
│   ├── surgical_assistant_voice.py    # Voice-enabled CLI with commands
│   ├── inference_distance_live.py     # Live inference w/ distance
│   ├── inference_video_server.py      # Video streaming server
│   ├── train_yolov11_seg.py           # Model training script
│   └── VOICE_FEATURES_README.md       # Voice features documentation
│
├── medh_v2/              # Production inference scripts
│   ├── inference_distance_v2.py       # Distance calculation (smooth)
│   ├── inference_distance_nosmooth.py # Distance calculation (raw)
│   ├── inference_video.py             # Video batch processing
│   └── slow_vid.py                    # Video speed adjustment
│
└── outputs/              # Saved predictions & videos
```

## 🎯 Use Cases

### 🏥 Surgical Training
- Train junior surgeons on anatomical identification
- Demonstrate safe instrument handling techniques
- Review recorded procedures with AI annotations

### 🛡️ Intraoperative Assistance
- Real-time alerts during live surgery (with proper validation and approval)
- Assist in identifying critical structures during complex dissections
- Reduce cognitive load through voice announcements

### 📊 Post-Operative Analysis
- Review surgical videos with automated segmentation
- Analyze near-miss incidents with distance data
- Generate training materials from annotated recordings

## 🔬 Technical Highlights

- **Custom-trained models**: YOLOv11 segmentation trained on surgical datasets (Roboflow + Kaggle)
- **Production-ready API**: CORS-enabled FastAPI with structured error handling
- **Efficient inference**: Model caching and preloading for sub-second predictions
- **Extensible design**: Pluggable model registry for easy addition of new weights
- **Type safety**: End-to-end TypeScript in frontend, Pydantic models in backend

## 📊 Supported Anatomical Structures

The system can detect and track various surgical structures depending on the loaded model:
- Ureters
- Ovaries
- Uterus
- External iliac artery & vein
- Uterine artery
- Obturator nerve
- Surgical instruments (graspers, scissors, electrocautery)

## 🚨 Disclaimer

This software is a research prototype and educational tool. It is **not** a medical device and has **not** been approved by any regulatory body (FDA, CE, etc.). It should **never** be used as the sole basis for clinical decision-making. All surgical decisions must be made by qualified medical professionals based on their training, judgment, and established clinical protocols.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This is a research and educational project. Please consult with appropriate institutional review boards (IRBs) and obtain necessary approvals before using with actual patient data or in clinical settings.

---

**Built for medical innovation** • **Powered by AI** • **Designed for safety**
