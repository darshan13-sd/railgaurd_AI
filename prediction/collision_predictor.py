"""
=============================================================================
prediction/collision_predictor.py — Trajectory-Based Collision Prediction
=============================================================================

Given an object's recent centroid history, linearly extrapolates its
future position a few frames ahead and checks whether that projected
path intersects the track ROI. This lets the system flag an object
that is *about to* enter the track, not just objects already inside it.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class CollisionPrediction:
    """Result of a collision projection for one object."""
    def __init__(self, will_enter_track: bool, projected_point: Optional[Tuple[int, int]], frames_ahead: int):
        self.will_enter_track = will_enter_track
        self.projected_point = projected_point
        self.frames_ahead = frames_ahead


def predict_collision(
    centroid_history: List[Tuple[int, int]],
    roi_polygon: Optional[np.ndarray],
    frame_width: int,
    frame_height: int,
    frames_ahead: int = 15,
) -> CollisionPrediction:
    """
    Linearly extrapolate the object's trajectory `frames_ahead` frames
    into the future and test whether the projected point falls inside
    the track ROI polygon.
    """
    if roi_polygon is None or len(centroid_history) < 2:
        return CollisionPrediction(False, None, frames_ahead)

    (x1, y1), (x2, y2) = centroid_history[-2], centroid_history[-1]
    vx, vy = (x2 - x1), (y2 - y1)

    projected_x = int(x2 + vx * frames_ahead)
    projected_y = int(y2 + vy * frames_ahead)

    # Clamp to frame bounds for sanity
    projected_x = max(0, min(frame_width - 1, projected_x))
    projected_y = max(0, min(frame_height - 1, projected_y))

    inside = cv2.pointPolygonTest(roi_polygon.astype(np.int32), (projected_x, projected_y), False) >= 0

    return CollisionPrediction(inside, (projected_x, projected_y), frames_ahead)
