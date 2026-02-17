"""
YOLOv11 Segmentation - Surgical Distance Measurement
Calculates distances between instruments and critical anatomical structures
"""

from ultralytics import YOLO
import cv2
import numpy as np
from scipy.spatial import distance
from collections import defaultdict
import time

# Configuration
MODEL_PATH = '26n.pt'
VIDEO_PATH = "iv2.mp4"
OUTPUT_PATH = 'ov2_distances.mp4'
CONF_THRESHOLD = 0.40

# Class names
CLASS_NAMES = [
    'External Iliac Artery', 
    'External Iliac Vein', 
    'Instruments', 
    'Obturator Nerve', 
    'Ovary', 
    'Ureter', 
    'Uterine Artery', 
    'uterus'
]

# Display options
SHOW_MASKS = True
SHOW_DISTANCE_LINES = True
SHOW_CRITICAL_DISTANCES = True
SHOW_INSTRUMENT_TIP = True

# Distance thresholds (in pixels - you can calibrate to mm)
CRITICAL_DISTANCE = 50  # Red warning
WARNING_DISTANCE = 100  # Yellow warning
SAFE_DISTANCE = 150     # Green

# Pixel to mm conversion (calibrate based on your video)
PIXEL_TO_MM = 0.5  # Adjust this based on your actual calibration

# Define critical structures (everything except instruments)
CRITICAL_STRUCTURES = [
    'External Iliac Artery', 
    'External Iliac Vein', 
    'Obturator Nerve', 
    'Ovary', 
    'Ureter', 
    'Uterine Artery', 
    'uterus'
]

# Color coding for distances
COLOR_CRITICAL = (0, 0, 255)    # Red
COLOR_WARNING = (0, 165, 255)   # Orange
COLOR_SAFE = (0, 255, 0)        # Green
COLOR_INSTRUMENT = (255, 255, 0) # Cyan


def find_instrument_tip(mask, box):
    """
    Find the tip of the surgical instrument
    Uses the bottom-most point of the mask as the tip
    
    Args:
        mask: Binary mask of the instrument
        box: Bounding box coordinates
        
    Returns:
        (x, y) coordinates of the instrument tip
    """
    # Get mask points
    points = np.column_stack(np.where(mask > 0))
    
    if len(points) == 0:
        # Fallback to bounding box center bottom
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int(y2))
    
    # Find bottom-most point (highest y-value)
    bottom_idx = np.argmax(points[:, 0])
    tip_y, tip_x = points[bottom_idx]
    
    # Refine: find the average of bottom 5% points for stability
    bottom_threshold = tip_y - 5
    bottom_points = points[points[:, 0] >= bottom_threshold]
    
    if len(bottom_points) > 0:
        tip_y = int(np.mean(bottom_points[:, 0]))
        tip_x = int(np.mean(bottom_points[:, 1]))
    
    return (tip_x, tip_y)


def get_mask_contour(mask):
    """
    Extract contour points from mask
    
    Args:
        mask: Binary mask
        
    Returns:
        Array of contour points
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return np.array([])
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour.reshape(-1, 2)


def calculate_point_to_mask_distance(point, mask):
    """
    Calculate minimum distance from a point to a mask contour
    
    Args:
        point: (x, y) tuple
        mask: Binary mask
        
    Returns:
        (distance, closest_point) tuple
    """
    contour_points = get_mask_contour(mask)
    
    if len(contour_points) == 0:
        return float('inf'), None
    
    # Calculate distances to all contour points
    distances = distance.cdist([point], contour_points, 'euclidean')[0]
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    closest_point = tuple(contour_points[min_idx])
    
    return min_distance, closest_point


def get_distance_color(dist, critical=CRITICAL_DISTANCE, warning=WARNING_DISTANCE):
    """
    Get color based on distance threshold
    
    Args:
        dist: Distance value
        critical: Critical threshold
        warning: Warning threshold
        
    Returns:
        BGR color tuple
    """
    if dist < critical:
        return COLOR_CRITICAL
    elif dist < warning:
        return COLOR_WARNING
    else:
        return COLOR_SAFE


def draw_distance_info(frame, distances_dict, instrument_tips):
    """
    Draw distance information panel on frame
    
    Args:
        frame: Video frame
        distances_dict: Dictionary of distances
        instrument_tips: List of instrument tip positions
        
    Returns:
        Frame with distance panel
    """
    panel_height = 250
    panel_width = 400
    panel_x = frame.shape[1] - panel_width - 10
    panel_y = 10
    
    # Create semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "DISTANCE MONITOR", 
                (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw distances
    y_offset = panel_y + 55
    sorted_distances = sorted(distances_dict.items(), key=lambda x: x[1]['distance'])
    
    for i, (structure, info) in enumerate(sorted_distances[:6]):  # Show top 6 closest
        dist_px = info['distance']
        dist_mm = dist_px * PIXEL_TO_MM
        color = get_distance_color(dist_px)
        
        # Structure name
        text = f"{structure[:20]}"
        cv2.putText(frame, text, 
                    (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Distance value
        dist_text = f"{dist_mm:.1f}mm"
        cv2.putText(frame, dist_text, 
                    (panel_x + 250, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Status indicator
        if dist_px < CRITICAL_DISTANCE:
            status = "CRITICAL"
        elif dist_px < WARNING_DISTANCE:
            status = "WARNING"
        else:
            status = "SAFE"
        
        cv2.putText(frame, status, 
                    (panel_x + 310, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        y_offset += 30
    
    # Legend
    legend_y = panel_y + panel_height - 40
    cv2.putText(frame, f"Critical: <{CRITICAL_DISTANCE*PIXEL_TO_MM:.0f}mm", 
                (panel_x + 10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CRITICAL, 1)
    cv2.putText(frame, f"Warning: <{WARNING_DISTANCE*PIXEL_TO_MM:.0f}mm", 
                (panel_x + 10, legend_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WARNING, 1)
    
    return frame


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
print(f"Pixel to MM ratio: {PIXEL_TO_MM}")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"\nProcessing video with distance calculations...")
print(f"Critical distance: {CRITICAL_DISTANCE*PIXEL_TO_MM:.1f}mm")
print(f"Warning distance: {WARNING_DISTANCE*PIXEL_TO_MM:.1f}mm")

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
        
        # Get annotated frame with masks
        annotated_frame = results[0].plot(
            boxes=False,
            labels=True,
            conf=True,
            masks=SHOW_MASKS
        )
        
        # Process detections
        if results[0].masks is not None and results[0].boxes is not None:
            masks_data = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            # Separate instruments and critical structures
            instruments = []
            critical_structures = []
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Resize mask to frame size
                mask = masks_data[i]
                mask_resized = cv2.resize(mask, (width, height), 
                                         interpolation=cv2.INTER_LINEAR)
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                
                if class_name == 'Instruments':
                    tip = find_instrument_tip(mask_binary, xyxy)
                    instruments.append({
                        'tip': tip,
                        'mask': mask_binary,
                        'box': xyxy
                    })
                    
                    # Draw instrument tip
                    if SHOW_INSTRUMENT_TIP:
                        cv2.circle(annotated_frame, tip, 8, COLOR_INSTRUMENT, -1)
                        cv2.circle(annotated_frame, tip, 10, (0, 0, 0), 2)
                        cv2.putText(annotated_frame, "TIP", 
                                  (tip[0] - 15, tip[1] - 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  COLOR_INSTRUMENT, 2)
                
                elif class_name in CRITICAL_STRUCTURES:
                    critical_structures.append({
                        'name': class_name,
                        'mask': mask_binary,
                        'box': xyxy
                    })
            
            # Calculate distances
            distances_dict = {}
            
            for instrument in instruments:
                tip = instrument['tip']
                
                for structure in critical_structures:
                    dist, closest_point = calculate_point_to_mask_distance(
                        tip, structure['mask']
                    )
                    
                    # Store distance
                    key = structure['name']
                    if key not in distances_dict or dist < distances_dict[key]['distance']:
                        distances_dict[key] = {
                            'distance': dist,
                            'closest_point': closest_point,
                            'tip': tip
                        }
                    
                    # Draw distance line
                    if SHOW_DISTANCE_LINES and closest_point is not None:
                        color = get_distance_color(dist)
                        cv2.line(annotated_frame, tip, closest_point, color, 2)
                        
                        # Draw closest point
                        cv2.circle(annotated_frame, closest_point, 5, color, -1)
                        
                        # Draw distance label on line
                        mid_x = (tip[0] + closest_point[0]) // 2
                        mid_y = (tip[1] + closest_point[1]) // 2
                        dist_mm = dist * PIXEL_TO_MM
                        
                        cv2.putText(annotated_frame, f"{dist_mm:.1f}mm", 
                                  (mid_x + 5, mid_y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                                  color, 2)
            
            # Draw distance information panel
            if SHOW_CRITICAL_DISTANCES and distances_dict:
                annotated_frame = draw_distance_info(
                    annotated_frame, 
                    distances_dict, 
                    [inst['tip'] for inst in instruments]
                )
        
        # Calculate and display FPS
        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Write frame
        out.write(annotated_frame)
        
        # Display
        cv2.imshow('Surgical Distance Monitor', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Progress
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed
            eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"Avg FPS: {avg_fps:.1f} | ETA: {eta:.0f}s")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Done! Output saved to: {OUTPUT_PATH}")
    print(f"   Processed {frame_count} frames in {elapsed:.1f}s")
    print(f"   Average FPS: {frame_count/elapsed:.1f}")