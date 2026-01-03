"""
YOLOv11 Segmentation Validation and Inference Script
For surgical organ detection and segmentation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import time

# Organ class names
CLASS_NAMES = [
    'External Iliac Artery',
    'External Iliac Vein', 
    'Obturator Nerve',
    'Ovary',
    'Ureter',
    'Uterine Artery',
    'uterus'
]

# Color palette for visualization (RGB format)
COLORS = [
    (255, 0, 0),      # Red - External Iliac Artery
    (0, 0, 255),      # Blue - External Iliac Vein
    (255, 255, 0),    # Yellow - Obturator Nerve
    (255, 0, 255),    # Magenta - Ovary
    (0, 255, 255),    # Cyan - Ureter
    (255, 128, 0),    # Orange - Uterine Artery
    (128, 0, 255),    # Purple - uterus
]

def validate_model(model_path, data_yaml='medhack_yolov11/data.yaml', imgsz=640, conf=0.25, iou=0.45):
    """
    Validate trained model on validation set
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset yaml
        imgsz: Image size for inference
        conf: Confidence threshold
        iou: IoU threshold for NMS
    """
    print(f"\n{'='*60}")
    print(f"Model Validation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Image Size: {imgsz}")
    print(f"Confidence: {conf}")
    print(f"IoU Threshold: {iou}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=True,
        plots=True
    )
    
    # Print detailed metrics
    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"{'='*60}")
    print(f"Box Metrics:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"\nSegmentation Metrics:")
    print(f"  mAP50:    {metrics.seg.map50:.4f}")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")
    print(f"  Precision: {metrics.seg.mp:.4f}")
    print(f"  Recall:    {metrics.seg.mr:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Segmentation mAP50:")
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(metrics.seg.ap50):
            print(f"  {class_name:25s}: {metrics.seg.ap50[i]:.4f}")
    
    print(f"{'='*60}\n")
    
    return metrics

def predict_single_image(model_path, image_path, output_dir='predictions', 
                         conf=0.25, iou=0.45, save_txt=True, save_conf=True):
    """
    Run prediction on a single image
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to input image
        output_dir: Directory to save predictions
        conf: Confidence threshold
        iou: IoU threshold for NMS
        save_txt: Save predictions to txt file
        save_conf: Save confidence scores
    """
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run prediction
    print(f"\n🔍 Predicting on: {image_path}")
    start_time = time.time()
    
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False
    )
    
    inference_time = time.time() - start_time
    
    # Process results
    result = results[0]
    img = cv2.imread(str(image_path))
    
    # Draw segmentation masks and boxes
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Create overlay
        overlay = img.copy()
        
        # Draw masks
        for i, (mask, box, score, cls) in enumerate(zip(masks, boxes, scores, classes)):
            # Resize mask to image size
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_bool = mask_resized > 0.5
            
            # Apply color mask
            color = COLORS[cls % len(COLORS)]
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5
            
            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{CLASS_NAMES[cls]}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save annotated image
        output_file = output_path / f"{Path(image_path).stem}_predicted.jpg"
        cv2.imwrite(str(output_file), overlay)
        
        print(f"✓ Detected {len(masks)} objects")
        print(f"✓ Inference time: {inference_time:.3f}s ({1/inference_time:.1f} FPS)")
        print(f"✓ Saved to: {output_file}")
        
        # Save txt annotations
        if save_txt:
            txt_file = output_path / f"{Path(image_path).stem}_predicted.txt"
            with open(txt_file, 'w') as f:
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    line = f"{cls} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n"
                    f.write(line)
            print(f"✓ Saved labels to: {txt_file}")
    else:
        print("✗ No objects detected")
    
    return results

def predict_batch(model_path, image_dir, output_dir='predictions', 
                  conf=0.25, iou=0.45, max_images=None):
    """
    Run predictions on a batch of images
    
    Args:
        model_path: Path to trained model weights
        image_dir: Directory containing images
        output_dir: Directory to save predictions
        conf: Confidence threshold
        iou: IoU threshold for NMS
        max_images: Maximum number of images to process (None = all)
    """
    # Load model
    model = YOLO(model_path)
    
    # Get image files
    image_path = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_path.glob(f'*{ext}')))
        image_files.extend(list(image_path.glob(f'*{ext.upper()}')))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n{'='*60}")
    print(f"Batch Prediction")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Input Directory: {image_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Total Images: {len(image_files)}")
    print(f"Confidence: {conf}")
    print(f"IoU Threshold: {iou}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process images
    total_time = 0
    total_detections = 0
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
        
        start_time = time.time()
        results = predict_single_image(
            model_path, 
            img_file, 
            output_dir, 
            conf=conf, 
            iou=iou
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if results[0].masks is not None:
            total_detections += len(results[0].masks)
    
    # Summary
    avg_time = total_time / len(image_files)
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete!")
    print(f"{'='*60}")
    print(f"Total Images: {len(image_files)}")
    print(f"Total Detections: {total_detections}")
    print(f"Average Time per Image: {avg_time:.3f}s")
    print(f"Average FPS: {1/avg_time:.1f}")
    print(f"{'='*60}\n")

def benchmark_model(model_path, imgsz=640, iterations=100):
    """
    Benchmark model inference speed
    
    Args:
        model_path: Path to trained model weights
        imgsz: Image size for inference
        iterations: Number of iterations for benchmarking
    """
    print(f"\n{'='*60}")
    print(f"Model Benchmarking")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Image Size: {imgsz}")
    print(f"Iterations: {iterations}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Create dummy input
    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = model.predict(dummy_img, verbose=False)
    
    # Benchmark
    print(f"Running {iterations} iterations...")
    times = []
    
    for i in range(iterations):
        start = time.time()
        _ = model.predict(dummy_img, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{iterations}")
    
    # Results
    times = np.array(times)
    print(f"\n{'='*60}")
    print(f"Benchmark Results:")
    print(f"{'='*60}")
    print(f"Mean Inference Time: {times.mean()*1000:.2f} ms")
    print(f"Std Deviation: {times.std()*1000:.2f} ms")
    print(f"Min Time: {times.min()*1000:.2f} ms")
    print(f"Max Time: {times.max()*1000:.2f} ms")
    print(f"Average FPS: {1/times.mean():.1f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'runs/segment/surgical_organs_v1/weights/best.pt'
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print(f"Please train the model first using train_yolov11_seg.py")
        exit(1)
    
    # Menu
    print("\n" + "="*60)
    print("YOLOv11 Segmentation - Inference & Validation")
    print("="*60)
    print("1. Validate model on validation set")
    print("2. Predict on single image")
    print("3. Predict on batch of images (test set)")
    print("4. Benchmark model speed")
    print("5. Exit")
    print("="*60)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Validate model
        validate_model(MODEL_PATH)
        
    elif choice == '2':
        # Single image prediction
        image_path = input("Enter image path: ").strip()
        if Path(image_path).exists():
            predict_single_image(MODEL_PATH, image_path)
        else:
            print(f"❌ Image not found: {image_path}")
            
    elif choice == '3':
        # Batch prediction
        image_dir = input("Enter image directory (default: medhack_yolov11/test/images): ").strip()
        if not image_dir:
            image_dir = 'medhack_yolov11/test/images'
        
        if Path(image_dir).exists():
            predict_batch(MODEL_PATH, image_dir, output_dir='predictions/test_results')
        else:
            print(f"❌ Directory not found: {image_dir}")
            
    elif choice == '4':
        # Benchmark
        benchmark_model(MODEL_PATH)
        
    elif choice == '5':
        print("\n👋 Goodbye!")
        
    else:
        print("\n❌ Invalid choice!")
