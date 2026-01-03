"""
YOLOv11 Segmentation Training - Memory Optimized for MX550 (1.64GB VRAM)
Ultra-low memory usage configuration
"""

import os
import torch
from ultralytics import YOLO

# Set memory allocation strategy before anything else
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def check_gpu():
    """Check GPU availability and memory"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        total_memory = device.total_memory / 1024**3
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        
        # Clear memory
        clear_memory()
        
        # Check available memory
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = total_memory - reserved
        
        print(f"  Available: {free:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        
        return True
    else:
        print("\n⚠ No GPU found - will use CPU")
        return False

def train_memory_optimized():
    """Train with ultra-low memory settings"""
    
    print("="*60)
    print("Memory-Optimized YOLOv11 Segmentation Training")
    print("Configured for MX550 with 1.64GB VRAM")
    print("="*60)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Clear memory before starting
    clear_memory()
    
    # Load model
    print("\n📦 Loading YOLOv11n-seg model...")
    model = YOLO('yolo11n-seg.pt')
    
    # Ultra-low memory training configuration
    training_args = {
        # Basic settings
        'data': 'medhack_yolov11/data.yaml',
        'epochs': 150,
        'batch': 2,  # Very small batch for 1.64GB VRAM
        'imgsz': 512,  # Reduced image size (from 640)
        'device': 0 if has_gpu else 'cpu',
        'project': 'runs/segment',
        'name': 'surgical_organs_lowmem',
        'exist_ok': True,
        'verbose': False,  # Reduce output verbosity
        
        # Memory optimization
        'cache': False,  # Don't cache images
        'workers': 2,  # Minimal workers
        'amp': True,  # Mixed precision (reduces memory)
        'half': False,  # Keep FP32 for stability
        'rect': False,  # Rectangular training OFF
        'multi_scale': False,  # Disable multi-scale (save memory)
        
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # Conservative augmentation (medical imaging)
        'hsv_h': 0.010,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.3,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,  # Reduced from 0.8
        'mixup': 0.0,  # Disabled to save memory
        'copy_paste': 0.0,  # Disabled to save memory
        'auto_augment': None,  # Disabled to save memory
        'erasing': 0.0,  # Disabled to save memory
        
        # Validation & Saving
        'val': True,
        'save': True,
        'save_period': 20,  # Save less frequently
        'patience': 30,
        'plots': True,
        'cos_lr': True,
        
        # Segmentation
        'overlap_mask': True,
        'mask_ratio': 4,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Other
        'close_mosaic': 10,
        'seed': 42,
        'deterministic': False,
        'nbs': 64,
    }
    
    print(f"\n🚀 Starting Training...")
    print(f"   Batch Size: {training_args['batch']}")
    print(f"   Image Size: {training_args['imgsz']}")
    print(f"   Device: {'GPU' if has_gpu else 'CPU'}")
    print(f"   Memory Mode: Ultra-Low\n")
    
    # Train
    try:
        results = model.train(**training_args)
        print("\n✅ Training completed successfully!")
        print(f"Best model: runs/segment/surgical_organs_lowmem/weights/best.pt")
        
        # Validate
        print("\n📊 Validating best model...")
        clear_memory()
        best_model = YOLO('runs/segment/surgical_organs_lowmem/weights/best.pt')
        metrics = best_model.val()
        
        print(f"\n{'='*60}")
        print(f"Validation Metrics:")
        print(f"{'='*60}")
        print(f"Box mAP50: {metrics.box.map50:.4f}")
        print(f"Seg mAP50: {metrics.seg.map50:.4f}")
        print(f"{'='*60}\n")
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Still out of memory!")
        print("\nTry these options:")
        print("1. Reduce batch size to 1: batch=1")
        print("2. Reduce image size to 384: imgsz=384")
        print("3. Train on CPU (slower but no memory limit): device='cpu'")
        print("\nTo train on CPU, run: python train_yolov11_seg_cpu.py")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    # Clear any existing GPU memory
    clear_memory()
    
    # Run training
    train_memory_optimized()
