"""
=============================================================================
utils/geometry.py — Shared Geometry Helpers
=============================================================================
"""

import numpy as np


def iou(box_a, box_b) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def euclidean_distance(p1, p2) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def clamp(value, low, high):
    return max(low, min(high, value))


def point_in_polygon(point, polygon: np.ndarray) -> bool:
    import cv2
    return cv2.pointPolygonTest(polygon.astype(np.int32), tuple(point), False) >= 0
