"""
================================================================================
YOLOV11 SEGMENTATION - DISTANCE MONITORING (Server + Live)
================================================================================
Surgical instrument distance measurement from anatomical structures.
Compatible with Linux GPU servers (headless) and local display.

Classes: External Iliac Artery, External Iliac Vein, Instruments,
         Obturator Nerve, Ovary, Ureter, Uterine Artery, uterus
================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import defaultdict
import subprocess
import shutil
import tempfile
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_COLORS = {
    "External Iliac Artery": (0, 0, 255),      # Red
    "External Iliac Vein":   (255, 0, 255),     # Magenta
    "Instruments":           (0, 255, 255),     # Cyan/Yellow
    "Obturator Nerve":       (0, 165, 255),     # Orange
    "Ovary":                 (255, 0, 0),       # Blue
    "Ureter":                (0, 255, 0),       # Green
    "Uterine Artery":        (128, 0, 255),     # Purple
    "uterus":                (255, 0, 128),     # Pink
}

# Will be auto-detected from model — handles 'Instruments', 'Instrument', etc.
INSTRUMENT_CLASS = None  # Auto-detect

DISTANCE_THRESHOLDS = {'safe': 50, 'caution': 30, 'danger': 30}
DEFAULT_MODEL_PATH = '26n.pt'

# Temporal smoothing settings
MASK_PERSIST_FRAMES = 5       # Keep showing a mask for N frames after it disappears
DISTANCE_SMOOTH_ALPHA = 0.4   # EMA smoothing (0=full smooth, 1=no smooth, 0.3-0.5 is good)
MIN_DETECT_RATIO = 0.3        # Must be detected in 30% of recent frames to show

# ============================================================================
# DISTANCE UTILITIES
# ============================================================================

def get_mask_boundary_points(mask: np.ndarray, sample_rate: int = 5) -> np.ndarray:
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.array([])
    all_points = []
    for contour in contours:
        points = contour.reshape(-1, 2)[::sample_rate]
        all_points.append(points)
    return np.vstack(all_points) if all_points else np.array([])


def get_mask_area(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0))


def calculate_minimum_distance(mask1: np.ndarray, mask2: np.ndarray) -> tuple:
    points1 = get_mask_boundary_points(mask1)
    points2 = get_mask_boundary_points(mask2)
    if len(points1) == 0 or len(points2) == 0:
        return float('inf'), None, None
    distances = cdist(points1, points2, metric='euclidean')
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return (
        distances[min_idx],
        tuple(points1[min_idx[0]].astype(int)),
        tuple(points2[min_idx[1]].astype(int)),
    )


def get_distance_color(distance: float) -> tuple:
    if distance >= DISTANCE_THRESHOLDS['safe']:
        return (0, 255, 0)
    elif distance >= DISTANCE_THRESHOLDS['caution']:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


def get_distance_label(distance: float) -> str:
    if distance >= DISTANCE_THRESHOLDS['safe']:
        return "SAFE"
    elif distance >= DISTANCE_THRESHOLDS['caution']:
        return "CAUTION"
    else:
        return "DANGER"


# ============================================================================
# VIDEO ENCODING
# ============================================================================

def frames_to_video_ffmpeg(frames_dir: Path, output_path: Path, fps: int) -> bool:
    """Encode saved frames with FFmpeg."""
    print(f"\n🎬 Encoding with FFmpeg...")
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
    except Exception:
        print("   ⚠️  FFmpeg not found, skipping")
        return False

    frames_pattern = str(frames_dir / 'frame_%06d.png')
    output_abs = str(output_path.absolute())

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', frames_pattern,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_abs,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            print(f"   ✅ FFmpeg succeeded!")
            return True
        else:
            print(f"   ❌ FFmpeg failed (code {result.returncode})")
            if result.stderr:
                print(f"   stderr: {result.stderr[:300]}")
            return False
    except Exception as e:
        print(f"   ❌ FFmpeg error: {e}")
        return False


def frames_to_video_opencv(
    frames_dir: Path, output_path: Path, fps: int, width: int, height: int
) -> bool:
    """Fallback: try OpenCV VideoWriter."""
    print(f"\n📹 Trying OpenCV VideoWriter as fallback...")

    frame_files = sorted(frames_dir.glob('frame_*.png'))
    if not frame_files:
        print("   ❌ No frames found!")
        return False

    codecs = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
    ]

    for fourcc_str, ext in codecs:
        test_output = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(test_output), fourcc, fps, (width, height))

        if not writer.isOpened():
            continue

        print(f"   Trying {fourcc_str} codec...")
        success = True
        for frame_file in tqdm(
            frame_files[:100], desc=f"   Testing {fourcc_str}", ncols=80
        ):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                success = False
                break
            writer.write(frame)

        writer.release()

        if success and test_output.exists() and test_output.stat().st_size > 10000:
            print(f"   ✅ {fourcc_str} works! Encoding full video...")
            writer = cv2.VideoWriter(str(test_output), fourcc, fps, (width, height))
            for frame_file in tqdm(frame_files, desc="   Encoding", ncols=80):
                frame = cv2.imread(str(frame_file))
                writer.write(frame)
            writer.release()
            print(f"   ✅ Saved: {test_output}")
            return True
        else:
            if test_output.exists():
                test_output.unlink()

    print("   ❌ All OpenCV codecs failed")
    return False


# ============================================================================
# PROCESSOR
# ============================================================================

class DistanceMonitoringProcessor:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        line_thickness: int = 2,
        device: str = '0',
        show_all_distances: bool = True,
        pixel_to_mm: float = None,
        topk: int = 5,
        min_organ_conf: float = 0.20,
        min_mask_area: int = 100,
        only_warn: bool = False,
        use_panel: bool = True,
    ):
        print(f"📦 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.line_thickness = line_thickness
        self.device = device
        self.show_all_distances = show_all_distances
        self.pixel_to_mm = pixel_to_mm
        self.topk = topk
        self.min_organ_conf = min_organ_conf
        self.min_mask_area = min_mask_area
        self.only_warn = only_warn
        self.use_panel = use_panel
        self.class_names = self.model.names
        self.distance_stats = defaultdict(list)
        self.debug_frame_count = 0  # For debug logging

        # ---- Temporal smoothing state ----
        # Mask persistence: {class_name: {'mask': ..., 'conf': ..., 'age': frames_since_last_seen}}
        self.mask_buffer = {}
        # Distance smoothing: {organ_name: smoothed_distance}
        self.smoothed_distances = {}
        # Detection history: {class_name: deque of booleans (detected or not)}
        from collections import deque
        self.detection_history = defaultdict(lambda: deque(maxlen=15))

        # Auto-detect instrument class
        global INSTRUMENT_CLASS
        model_classes = list(self.class_names.values())
        print(f"\n=== Model Classes ===")
        for idx, name in self.class_names.items():
            print(f"  {idx}: '{name}'")

        # Try to find instrument class by matching common names
        instrument_keywords = ['instrument', 'tool', 'forceps', 'grasper', 'scissors']
        self.instrument_class = INSTRUMENT_CLASS  # Use global if already set
        if self.instrument_class is None:
            for cls_name in model_classes:
                if any(kw in cls_name.lower() for kw in instrument_keywords):
                    self.instrument_class = cls_name
                    break

        if self.instrument_class and self.instrument_class in model_classes:
            print(f"\n✅ Instrument class: '{self.instrument_class}'")
        else:
            print(f"\n⚠️  WARNING: Could not find instrument class!")
            print(f"   Searched for keywords: {instrument_keywords}")
            print(f"   Available classes: {model_classes}")
            print(f"   Set INSTRUMENT_CLASS manually at top of script")
            self.instrument_class = None

        # All non-instrument classes are organs to measure distance to
        self.organ_classes = [c for c in model_classes if c != self.instrument_class]
        print(f"   Organ classes: {self.organ_classes}")
        print(f"   Top-K: {topk} | Min organ conf: {min_organ_conf} | Min mask area: {min_mask_area}")
        print(f"   Smoothing: persist={MASK_PERSIST_FRAMES}f, alpha={DISTANCE_SMOOTH_ALPHA}, min_ratio={MIN_DETECT_RATIO}")

    def draw_segmentation_boundaries(self, frame, masks, classes, confidences):
        output = frame.copy()
        for mask, cls, conf in zip(masks, classes, confidences):
            class_name = self.class_names[int(cls)]
            color = CLASS_COLORS.get(class_name, (128, 128, 128))
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(output, contours, -1, color, self.line_thickness)
        return output

    def calculate_distances(self, masks, classes, confidences):
        if self.instrument_class is None:
            return []

        is_debug = self.debug_frame_count < 5  # Debug first 5 frames

        detections = {'instrument': [], 'organs': []}
        for mask, cls, conf in zip(masks, classes, confidences):
            class_name = self.class_names[int(cls)]
            area = get_mask_area(mask)

            if is_debug:
                print(f"   [DEBUG] Detected: '{class_name}' conf={conf:.3f} area={area}")

            if area < self.min_mask_area:
                if is_debug:
                    print(f"           → SKIPPED (area {area} < min {self.min_mask_area})")
                continue

            det = {'mask': mask, 'class': class_name, 'conf': float(conf)}
            if class_name == self.instrument_class:
                detections['instrument'].append(det)
                if is_debug:
                    print(f"           → INSTRUMENT ✓")
            elif conf >= self.min_organ_conf:
                detections['organs'].append(det)
                if is_debug:
                    print(f"           → ORGAN ✓")
            else:
                if is_debug:
                    print(f"           → SKIPPED (conf {conf:.3f} < min {self.min_organ_conf})")

        # Pick highest-confidence instrument detection
        if detections['instrument']:
            detections['instrument'] = [
                max(detections['instrument'], key=lambda d: d['conf'])
            ]

        if is_debug:
            print(f"   [DEBUG] Instruments found: {len(detections['instrument'])}, "
                  f"Organs found: {len(detections['organs'])}")

        if not detections['instrument'] or not detections['organs']:
            if is_debug:
                if not detections['instrument']:
                    print(f"   [DEBUG] ❌ No instrument detected! Cannot compute distances.")
                if not detections['organs']:
                    print(f"   [DEBUG] ❌ No organs detected! Cannot compute distances.")
            return []

        instrument_mask = detections['instrument'][0]['mask']
        best_per_organ = {}
        for organ_det in detections['organs']:
            min_dist, p1, p2 = calculate_minimum_distance(
                instrument_mask, organ_det['mask']
            )
            if p1 and p2:
                organ_name = organ_det['class']
                if (
                    organ_name not in best_per_organ
                    or min_dist < best_per_organ[organ_name]['distance']
                ):
                    best_per_organ[organ_name] = {
                        'organ': organ_name,
                        'distance': min_dist,
                        'pipe_point': p1,
                        'organ_point': p2,
                        'organ_conf': organ_det['conf'],
                    }
                    self.distance_stats[organ_name].append(min_dist)

        if is_debug and best_per_organ:
            print(f"   [DEBUG] Distances computed:")
            for name, d in best_per_organ.items():
                print(f"           {name}: {d['distance']:.1f}px")

        self.debug_frame_count += 1

        distances = sorted(best_per_organ.values(), key=lambda x: x['distance'])
        if self.only_warn:
            distances = [
                d for d in distances if get_distance_label(d['distance']) != "SAFE"
            ]
        return distances[: self.topk] if self.topk else distances

    def draw_distance_lines(self, frame, distances):
        for d in distances:
            color = get_distance_color(d['distance'])
            cv2.line(frame, d['pipe_point'], d['organ_point'], color, 2)
            cv2.circle(frame, d['pipe_point'], 5, (0, 255, 255), -1)
            cv2.circle(frame, d['organ_point'], 5, color, -1)

            # Distance label at midpoint
            mid_x = (d['pipe_point'][0] + d['organ_point'][0]) // 2
            mid_y = (d['pipe_point'][1] + d['organ_point'][1]) // 2
            if self.pixel_to_mm:
                val = f"{d['distance'] * self.pixel_to_mm:.1f}mm"
            else:
                val = f"{d['distance']:.0f}px"
            (tw, th), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                frame, (mid_x - 3, mid_y - th - 3), (mid_x + tw + 3, mid_y + 3),
                (0, 0, 0), -1,
            )
            cv2.putText(
                frame, val, (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
        return frame

    def draw_distance_panel(self, frame, distances):
        if not distances:
            return frame
        pw, ph = 380, 30 + 28 * len(distances) + 10
        x0, y0 = 10, 120
        cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (255, 255, 255), 2)
        cv2.putText(
            frame, "INSTRUMENT DISTANCES",
            (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        y = y0 + 50
        for i, d in enumerate(distances):
            dist = d['distance']
            status = get_distance_label(dist)
            color = get_distance_color(dist)
            val = (
                f"{dist * self.pixel_to_mm:.1f}mm"
                if self.pixel_to_mm
                else f"{dist:.0f}px"
            )
            cv2.putText(
                frame, f"{i+1}.", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, d['organ'], (x0 + 35, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, val, (x0 + 220, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.rectangle(
                frame, (x0 + pw - 85, y - 14), (x0 + pw - 10, y + 4), color, -1,
            )
            cv2.putText(
                frame, status, (x0 + pw - 80, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )
            y += 28
        return frame

    def draw_legend(self, frame):
        h, w = frame.shape[:2]
        num_classes = len(CLASS_COLORS)
        legend_h = 30 + (num_classes * 22) + 10
        legend_w = 220
        x0, y0 = w - legend_w - 10, 10

        cv2.rectangle(frame, (x0, y0), (x0 + legend_w, y0 + legend_h), (0, 0, 0), -1)
        cv2.rectangle(
            frame, (x0, y0), (x0 + legend_w, y0 + legend_h), (255, 255, 255), 2,
        )
        cv2.putText(
            frame, "CLASS LEGEND", (x0 + 10, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        y = y0 + 40
        for class_name, color in CLASS_COLORS.items():
            cv2.rectangle(frame, (x0 + 10, y - 12), (x0 + 28, y + 2), color, -1)
            cv2.putText(
                frame, class_name, (x0 + 35, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )
            y += 22

        # Distance status legend
        status_y0 = y0 + legend_h + 15
        status_h = 95
        cv2.rectangle(
            frame, (x0, status_y0), (x0 + legend_w, status_y0 + status_h),
            (0, 0, 0), -1,
        )
        cv2.rectangle(
            frame, (x0, status_y0), (x0 + legend_w, status_y0 + status_h),
            (255, 255, 255), 2,
        )
        cv2.putText(
            frame, "DISTANCE", (x0 + 10, status_y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        cv2.rectangle(
            frame, (x0 + 10, status_y0 + 28), (x0 + 28, status_y0 + 42),
            (0, 255, 0), -1,
        )
        cv2.putText(
            frame, f"SAFE >{DISTANCE_THRESHOLDS['safe']}px",
            (x0 + 35, status_y0 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

        cv2.rectangle(
            frame, (x0 + 10, status_y0 + 48), (x0 + 28, status_y0 + 62),
            (0, 255, 255), -1,
        )
        cv2.putText(
            frame,
            f"CAUTION {DISTANCE_THRESHOLDS['caution']}-{DISTANCE_THRESHOLDS['safe']}px",
            (x0 + 35, status_y0 + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

        cv2.rectangle(
            frame, (x0 + 10, status_y0 + 68), (x0 + 28, status_y0 + 82),
            (0, 0, 255), -1,
        )
        cv2.putText(
            frame, f"DANGER <{DISTANCE_THRESHOLDS['danger']}px",
            (x0 + 35, status_y0 + 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

        return frame

    def _update_mask_buffer(self, masks, classes, confs):
        """Update temporal mask buffer with current detections."""
        detected_this_frame = set()

        if masks is not None:
            for mask, cls, conf in zip(masks, classes, confs):
                class_name = self.class_names[int(cls)]
                detected_this_frame.add(class_name)
                # Always store latest detection (fresher = better)
                if class_name not in self.mask_buffer or conf > self.mask_buffer[class_name].get('conf', 0) * 0.7:
                    self.mask_buffer[class_name] = {
                        'mask': mask.copy(),
                        'cls': int(cls),
                        'conf': float(conf),
                        'age': 0,
                    }
                # Record detection
                self.detection_history[class_name].append(True)

        # Age out classes not detected this frame
        to_remove = []
        for class_name, buf in self.mask_buffer.items():
            if class_name not in detected_this_frame:
                buf['age'] += 1
                self.detection_history[class_name].append(False)
                if buf['age'] > MASK_PERSIST_FRAMES:
                    to_remove.append(class_name)
        for key in to_remove:
            del self.mask_buffer[key]

    def _get_stabilized_detections(self):
        """
        Return stabilized masks/classes/confs from the buffer.
        Filters out classes that appear too infrequently (noise).
        """
        stable_masks, stable_classes, stable_confs = [], [], []
        for class_name, buf in self.mask_buffer.items():
            history = self.detection_history[class_name]
            if len(history) < 2:
                detect_ratio = 1.0  # Give new detections benefit of doubt
            else:
                detect_ratio = sum(history) / len(history)

            # Only include if detected frequently enough (not random noise)
            if detect_ratio >= MIN_DETECT_RATIO:
                stable_masks.append(buf['mask'])
                stable_classes.append(buf['cls'])
                # Reduce displayed confidence for persisted (aged) detections
                conf = buf['conf'] * max(0.5, 1.0 - buf['age'] * 0.1)
                stable_confs.append(conf)

        if not stable_masks:
            return None, None, None
        return (
            np.array(stable_masks),
            np.array(stable_classes),
            np.array(stable_confs),
        )

    def _smooth_distances(self, raw_distances):
        """
        Apply exponential moving average to distance values.
        Prevents distances from jumping erratically between frames.
        """
        smoothed = []
        current_organs = set()
        for d in raw_distances:
            organ = d['organ']
            current_organs.add(organ)
            raw_dist = d['distance']

            if organ in self.smoothed_distances:
                # EMA: new = alpha * raw + (1 - alpha) * old
                d['distance'] = (
                    DISTANCE_SMOOTH_ALPHA * raw_dist
                    + (1 - DISTANCE_SMOOTH_ALPHA) * self.smoothed_distances[organ]
                )
            self.smoothed_distances[organ] = d['distance']
            smoothed.append(d)

        # Decay organs not seen this frame
        for organ in list(self.smoothed_distances.keys()):
            if organ not in current_organs:
                del self.smoothed_distances[organ]

        return smoothed

    def process_frame(self, frame):
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )[0]

        # Get raw detections
        raw_masks = results.masks.data.cpu().numpy() if results.masks is not None else None
        raw_classes = results.boxes.cls.cpu().numpy() if results.masks is not None else None
        raw_confs = results.boxes.conf.cpu().numpy() if results.masks is not None else None

        # Update temporal buffer with current raw detections
        self._update_mask_buffer(raw_masks, raw_classes, raw_confs)

        # Get stabilized detections from buffer
        stable_masks, stable_classes, stable_confs = self._get_stabilized_detections()

        if stable_masks is None:
            return self.draw_legend(frame)

        # Draw with stabilized detections
        output = self.draw_segmentation_boundaries(frame, stable_masks, stable_classes, stable_confs)
        raw_distances = self.calculate_distances(stable_masks, stable_classes, stable_confs)
        distances = self._smooth_distances(raw_distances)
        output = self.draw_distance_lines(output, distances)
        if self.use_panel:
            output = self.draw_distance_panel(output, distances)
        return self.draw_legend(output)

    def _encode_video(self, temp_dir, output_path, fps, width, height):
        """Encode frames from temp_dir into a video. Called on normal exit AND on interrupt."""
        frame_files = sorted(temp_dir.glob('frame_*.png'))
        if not frame_files:
            print(f"\n⚠️  No frames to encode!")
            return False

        print(f"\n🎬 Encoding {len(frame_files)} frames into video...")
        if frames_to_video_ffmpeg(temp_dir, output_path, fps):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"✅ Video saved: {output_path.absolute()}")
            return True
        elif frames_to_video_opencv(temp_dir, output_path, fps, width, height):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"✅ Video saved: {output_path.absolute()}")
            return True
        else:
            print(f"\n⚠️  Video encoding failed!")
            print(f"   Frames saved in: {temp_dir}")
            print(
                f"   Manual encode: ffmpeg -framerate {fps} -i \"{temp_dir}/frame_%06d.png\""
                f' -c:v libx264 -crf 23 "{output_path}"'
            )
            return False

    def process_video(
        self, input_video, output_video, show_preview=False, resize_width=None
    ):
        print(f"\n📹 Processing: {input_video}")
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {input_video}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w, h = (
            (resize_width, int(resize_width * orig_h / orig_w))
            if resize_width
            else (orig_w, orig_h)
        )

        print(
            f"   Size: {orig_w}x{orig_h} → {w}x{h} | FPS: {fps} | Frames: {total_frames}"
        )

        # Temp directory for frames
        temp_dir = Path(tempfile.gettempdir()) / f'surgical_seg_{Path(input_video).stem}'
        temp_dir.mkdir(exist_ok=True)
        print(f"   Temp frames: {temp_dir}")

        output_path = Path(output_video)
        final_width = w * 2
        final_height = h + 35

        # Process frames — wrapped in try/finally so video is ALWAYS encoded,
        # even if you cancel with Ctrl+C
        print(f"\n🎬 Processing frames...")
        frame_count = 0
        interrupted = False
        try:
            with tqdm(total=total_frames, desc="   Progress", ncols=100) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if resize_width:
                        frame = cv2.resize(frame, (w, h))

                    processed = self.process_frame(frame)
                    combined = np.hstack([frame, processed])

                    # Header bar
                    header_h = 35
                    final = np.zeros((h + header_h, w * 2, 3), dtype=np.uint8)
                    final[header_h:, :] = combined
                    cv2.putText(
                        final, "ORIGINAL", (w // 2 - 60, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        final, "DISTANCE MONITORING (SMOOTHED)", (w + w // 2 - 200, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        final,
                        f"Frame {frame_count + 1}/{total_frames}",
                        (10, h + header_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                    )

                    cv2.imwrite(str(temp_dir / f"frame_{frame_count:06d}.png"), final)

                    if show_preview:
                        display_h, display_w = final.shape[:2]
                        max_display_h, max_display_w = 900, 1600
                        scale_w = max_display_w / display_w if display_w > max_display_w else 1.0
                        scale_h = max_display_h / display_h if display_h > max_display_h else 1.0
                        scale = min(scale_w, scale_h)
                        if scale < 1.0:
                            display = cv2.resize(
                                final,
                                (int(display_w * scale), int(display_h * scale)),
                                interpolation=cv2.INTER_AREA,
                            )
                        else:
                            display = final
                        cv2.imshow('Surgical Distance Monitoring', display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print(f"\n⏹️  Stopped by user (q key)")
                            break

                    frame_count += 1
                    pbar.update(1)

        except KeyboardInterrupt:
            interrupted = True
            print(f"\n\n⏹️  Interrupted! Encoding {frame_count} frames that were processed...")

        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

            # ALWAYS encode whatever frames we have
            if frame_count > 0:
                if interrupted:
                    print(f"   Saving partial video ({frame_count}/{total_frames} frames)...")
                self._encode_video(temp_dir, output_path, fps, final_width, final_height)
            else:
                print(f"\n⚠️  No frames were processed, nothing to encode.")

            self.print_stats()

        return frame_count

    def print_stats(self):
        if not self.distance_stats:
            return
        print(f"\n📊 Distance Statistics:")
        for organ, dists in sorted(self.distance_stats.items()):
            if dists:
                unit = "mm" if self.pixel_to_mm else "px"
                factor = self.pixel_to_mm if self.pixel_to_mm else 1
                print(
                    f"   {organ:<25}: min={np.min(dists)*factor:.1f} "
                    f"avg={np.mean(dists)*factor:.1f} "
                    f"max={np.max(dists)*factor:.1f} {unit}"
                )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Surgical instrument distance monitoring with YOLOv11 segmentation"
    )
    parser.add_argument('input_video', help="Path to input video")
    parser.add_argument('-o', '--output', default=None, help="Output video path")
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument('-c', '--conf', type=float, default=0.25, help="Confidence threshold")
    parser.add_argument('--topk', type=int, default=7, help="Show top-K closest structures")
    parser.add_argument('--min-organ-conf', type=float, default=0.20, help="Min organ confidence")
    parser.add_argument('--min-mask-area', type=int, default=100, help="Min mask area (pixels)")
    parser.add_argument('--only-warn', action='store_true', help="Only show caution/danger")
    parser.add_argument('--show-preview', action='store_true', help="Show live CV2 preview (needs display)")
    parser.add_argument('-d', '--device', default='0', help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument('--pixel-to-mm', type=float, default=None,
                        help='Pixel to mm conversion factor (e.g. 0.2)')
    parser.add_argument('--resize', type=int, default=None, help="Resize width")
    args = parser.parse_args()

    if not args.output:
        # Save output in current working directory
        args.output = f"{Path(args.input_video).stem}_distances_smoothed.mp4"

    processor = DistanceMonitoringProcessor(
        model_path=args.model,
        conf_threshold=args.conf,
        topk=args.topk,
        min_organ_conf=args.min_organ_conf,
        min_mask_area=args.min_mask_area,
        only_warn=args.only_warn,
        device=args.device,
        pixel_to_mm=args.pixel_to_mm,
    )

    processor.process_video(
        args.input_video,
        args.output,
        show_preview=args.show_preview,
        resize_width=args.resize,
    )


if __name__ == '__main__':
    main()
