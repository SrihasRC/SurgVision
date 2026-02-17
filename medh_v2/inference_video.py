"""
YOLOv11 Segmentation - Video Inference with Streaming
Minimal script for real-time video processing
"""

from ultralytics import YOLO
import cv2
import time

# Configuration
MODEL_PATH = '26n.pt'  # Your trained model
VIDEO_PATH = "../chunk_01.mp4"  # Input video path (or 0 for webcam)
OUTPUT_PATH = '../test-3.mp4'  # Output video path
CONF_THRESHOLD = 0.40  # Confidence threshold

# Display options
SHOW_BOXES = False  # Show bounding boxes
SHOW_LABELS = True  # Show class labels
SHOW_CONFIDENCE = False  # Show confidence scores
SHOW_MASKS = True  # Show segmentation masks

# Load model
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video {VIDEO_PATH}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"\nProcessing video... Press 'q' to quit")

frame_count = 0
start_time = time.time()
try:
    while cap.isOpened():
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        
        # Get annotated frame - masks only without boxes
        annotated_frame = results[0].plot(
            boxes=False,
            labels=False,
            conf=False,
            masks=SHOW_MASKS
        )
        
        # Add labels manually on masks if enabled
        if SHOW_LABELS and results[0].masks is not None:
            for i, box in enumerate(results[0].boxes):
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])
                
                # Get box center position for label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_x = (x1 + x2) // 2
                label_y = (y1 + y2) // 2
                
                # Create label text
                if SHOW_CONFIDENCE:
                    label = f"{class_name} {conf:.2f}"
                else:
                    label = class_name
                
                # Draw label with background
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (label_x - 5, label_y - text_h - 5), 
                            (label_x + text_w + 5, label_y + 5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, label, (label_x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calculate and display FPS
        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame
        out.write(annotated_frame)
        
        # Display streaming (optional - comment out for faster processing)
        cv2.imshow('YOLOv11 Segmentation', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Progress
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Done! Output saved to: {OUTPUT_PATH}")
    print(f"   Processed {frame_count} frames")
