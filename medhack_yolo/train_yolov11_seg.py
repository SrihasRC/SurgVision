"""
YOLOv11 Segmentation Training Script for Surgical Organ Detection
Optimized for NVIDIA MX550 + CPU with medical-specific augmentations
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

# Hardware configuration
def setup_device():
    """Configure optimal device settings for MX550 GPU"""
    if torch.cuda.is_available():
        device = 0  # Use GPU
        # Clear cache before starting
        torch.cuda.empty_cache()
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Set memory allocation strategy for fragmentation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        device = 'cpu'
        print("⚠ Running on CPU")
    return device

def train_yolov11_segmentation(
    data_yaml='medhack_yolov11/data.yaml',
    model_size='n',  # nano - best for MX550/CPU
    epochs=100,
    batch_size=8,  # Optimized for MX550 2GB VRAM
    img_size=640,
    project='runs/segment',
    name='surgical_organs',
    resume=False,
    pretrained=True
):
    """
    Train YOLOv11 segmentation model for surgical organ detection
    
    Args:
        data_yaml: Path to dataset yaml file
        model_size: Model size - 'n'(nano), 's'(small), 'm'(medium), 'l'(large), 'x'(xlarge)
                   For MX550/CPU: Use 'n' (nano) - fastest inference
        epochs: Number of training epochs
        batch_size: Batch size (8 for MX550, reduce if OOM)
        img_size: Input image size
        project: Project directory
        name: Experiment name
        resume: Resume from last checkpoint
        pretrained: Use pretrained weights
    """
    
    # Setup device
    device = setup_device()
    
    # Model selection - YOLOv11n-seg is optimal for your hardware
    model_name = f'yolo11{model_size}-seg.pt'
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Model: YOLOv11{model_size.upper()}-seg (Segmentation)")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    if pretrained:
        print(f"Loading pretrained {model_name}...")
        model = YOLO(model_name)
    else:
        print(f"Creating new {model_name} model...")
        model = YOLO(model_name.replace('.pt', '.yaml'))
    
    # Training hyperparameters optimized for medical imaging
    training_args = {
        # Basic training parameters
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        'resume': resume,
        'verbose': True,
        
        # Optimization parameters
        'optimizer': 'AdamW',  # Better for medical images
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Data augmentation for surgical/medical images
        'hsv_h': 0.010,  # Hue augmentation (reduced for medical)
        'hsv_s': 0.5,    # Saturation augmentation
        'hsv_v': 0.3,    # Value/brightness augmentation
        'degrees': 5.0,  # Rotation (±5 degrees - conservative for organs)
        'translate': 0.1,  # Translation (10% - minimal shift)
        'scale': 0.3,    # Scaling (±30% - account for zoom)
        'shear': 0.0,    # No shear (organs don't shear in surgery)
        'perspective': 0.0,  # No perspective (endoscopic view is consistent)
        'flipud': 0.0,   # No vertical flip (organs have orientation)
        'fliplr': 0.5,   # Horizontal flip 50% (left/right surgery views)
        'mosaic': 0.8,   # Mosaic augmentation probability
        'mixup': 0.1,    # Mixup augmentation probability (light)
        'copy_paste': 0.2,  # Copy-paste augmentation (useful for organs)
        # Performance optimizations
        'amp': True,  # Automatic Mixed Precision (reduces memory usage)
        'half': False,  # Don't use FP16 during training (stability)
        'workers': 4,  # Reduced data loading workers (save memory)
        'cache': False,  # Don't cache (save memory)
        'close_mosaic': 10,  # Disable mosaic last 10 epochs
        'pretrained': True,  # Use pretrained weights for better convergence
        # Performance optimizations
        'amp': True,  # Automatic Mixed Precision
        'half': False,  # Don't use FP16 during training (stability)
        'workers': 8,  # Data loading workers
        'cache': False,  # Don't cache (save memory)
        'close_mosaic': 10,  # Disable mosaic last 10 epochs
        
        # Validation & Saving
        'val': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'patience': 30,  # Early stopping patience
        'plots': True,  # Generate training plots
        'rect': False,  # Rectangular training
        'cos_lr': True,  # Cosine learning rate scheduler
        'overlap_mask': True,  # Masks can overlap (organs can overlap visually)
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,  # No dropout in YOLO architecture
        
        # Loss weights (tuned for segmentation)
        'box': 7.5,  # Box loss weight
        'cls': 0.5,  # Classification loss weight
        'dfl': 1.5,  # Distribution focal loss weight
        'pose': 12.0,  # Pose loss weight (not used in seg)
        'kobj': 1.0,  # Keypoint obj loss weight (not used in seg)
        
        # Labels & names
        'single_cls': False,  # Multi-class detection
        'label_smoothing': 0.0,  # No label smoothing
        
        # Advanced options
        'nbs': 64,  # Nominal batch size
        'seed': 42,  # Random seed for reproducibility
        'deterministic': False,  # Not fully deterministic (faster)
        'multi_scale': True,  # Multi-scale training
    }
    
    # Start training
    print("\n🚀 Starting training...\n")
    results = model.train(**training_args)
    
    # Print results summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best model saved to: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"Last model saved to: {Path(project) / name / 'weights' / 'last.pt'}")
    print(f"{'='*60}\n")
    
    # Validate best model
    print("\n📊 Validating best model...\n")
    best_model = YOLO(Path(project) / name / 'weights' / 'best.pt')
    metrics = best_model.val()
    
    print(f"\n{'='*60}")
    print(f"Validation Metrics:")
    print(f"{'='*60}")
    print(f"mAP50: {metrics.seg.map50:.4f}")
    print(f"mAP50-95: {metrics.seg.map:.4f}")
    print(f"{'='*60}\n")
    
    return results, metrics

def export_model(model_path, formats=['onnx', 'torchscript']):
    """
    Export trained model to different formats for deployment
    
    Args:
        model_path: Path to trained model weights
        formats: List of export formats
    """
    print(f"\n📦 Exporting model from {model_path}...")
    model = YOLO(model_path)
    
    for fmt in formats:
        print(f"Exporting to {fmt.upper()}...")
        try:
            model.export(format=fmt, imgsz=640, half=False, int8=False)
            print(f"✓ {fmt.upper()} export successful")
        except Exception as e:
            print(f"✗ {fmt.upper()} export failed: {str(e)}")
    
    print("\n✓ Model export complete!\n")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'data_yaml': 'medhack_yolov11/data.yaml',
        'model_size': 'n',  # nano - optimal for MX550 (2GB VRAM) + CPU
        'epochs': 150,
        'batch_size': 4,  # Reduced for MX550 1.64GB VRAM (use 2 if still OOM)
        'img_size': 640,
        'project': 'runs/segment',
        'name': 'surgical_organs_v1',
        'resume': False,
    }
    
    # Print system info
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CPU Cores: {os.cpu_count()}")
    print("="*60 + "\n")
    
    # Train model
    results, metrics = train_yolov11_segmentation(**CONFIG)
    
    # Export model to ONNX for deployment
    best_model_path = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'best.pt'
    if best_model_path.exists():
        export_model(best_model_path, formats=['onnx'])
    
    print("\n✅ All done! You can now use your trained model for inference.")
    print(f"   Best model: {best_model_path}")
    print(f"   To use: model = YOLO('{best_model_path}')")
