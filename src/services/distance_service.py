# Distance Service: instrument-to-organ distance calculation
# Reuses distance logic from inference_distance_v2.py

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from src.config import DISTANCE_THRESHOLDS, INSTRUMENT_KEYWORDS


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
        return float("inf"), None, None
    distances = cdist(points1, points2, metric="euclidean")
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return (
        float(distances[min_idx]),
        points1[min_idx[0]].astype(int).tolist(),
        points2[min_idx[1]].astype(int).tolist(),
    )


def get_distance_status(distance: float) -> str:
    if distance >= DISTANCE_THRESHOLDS["safe"]:
        return "SAFE"
    elif distance >= DISTANCE_THRESHOLDS["caution"]:
        return "CAUTION"
    return "DANGER"


def get_distance_color(distance: float) -> tuple:
    if distance >= DISTANCE_THRESHOLDS["safe"]:
        return (0, 255, 0)
    elif distance >= DISTANCE_THRESHOLDS["caution"]:
        return (0, 255, 255)
    return (0, 0, 255)


def detect_instrument_class(class_names: dict[int, str]) -> str | None:
    """Auto-detect which class is the instrument."""
    for cls_name in class_names.values():
        if any(kw in cls_name.lower() for kw in INSTRUMENT_KEYWORDS):
            return cls_name
    return None


def calculate_frame_distances(
    masks: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    class_names: dict[int, str],
    min_mask_area: int = 100,
    min_organ_conf: float = 0.20,
    topk: int = 7,
) -> list[dict]:
    """
    Given masks, classes, and confidences from a YOLO result,
    calculate distances between the instrument and all organs.
    """
    instrument_class = detect_instrument_class(class_names)
    if instrument_class is None:
        return []

    instrument_det = None
    organ_dets = []

    for mask, cls, conf in zip(masks, classes, confidences):
        class_name = class_names[int(cls)]
        if get_mask_area(mask) < min_mask_area:
            continue

        if class_name == instrument_class:
            if instrument_det is None or float(conf) > instrument_det["conf"]:
                instrument_det = {"mask": mask, "conf": float(conf)}
        elif float(conf) >= min_organ_conf:
            organ_dets.append({
                "mask": mask,
                "class": class_name,
                "conf": float(conf),
            })

    if instrument_det is None or not organ_dets:
        return []

    best_per_organ: dict[str, dict] = {}
    for organ in organ_dets:
        dist, p1, p2 = calculate_minimum_distance(instrument_det["mask"], organ["mask"])
        if p1 and p2:
            name = organ["class"]
            if name not in best_per_organ or dist < best_per_organ[name]["distance_px"]:
                best_per_organ[name] = {
                    "organ": name,
                    "distance_px": round(dist, 2),
                    "status": get_distance_status(dist),
                    "instrument_point": p1,
                    "organ_point": p2,
                    "organ_confidence": round(organ["conf"], 4),
                }

    distances = sorted(best_per_organ.values(), key=lambda x: x["distance_px"])
    return distances[:topk]


def draw_distances_on_frame(frame: np.ndarray, distances: list[dict]) -> np.ndarray:
    """Draw distance lines and labels on a frame."""
    output = frame.copy()
    for d in distances:
        color = get_distance_color(d["distance_px"])
        p1 = tuple(d["instrument_point"])
        p2 = tuple(d["organ_point"])

        cv2.line(output, p1, p2, color, 2)
        cv2.circle(output, p1, 5, (0, 255, 255), -1)
        cv2.circle(output, p2, 5, color, -1)

        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        label = f"{d['distance_px']:.0f}px {d['status']}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            output, (mid_x - 3, mid_y - th - 3), (mid_x + tw + 3, mid_y + 3),
            (0, 0, 0), -1,
        )
        cv2.putText(
            output, label, (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    return output
