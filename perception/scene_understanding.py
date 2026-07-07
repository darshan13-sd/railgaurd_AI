"""
=============================================================================
perception/scene_understanding.py — Scene State Aggregation
=============================================================================

Fuses the outputs of the other perception modules (track ROI, tracked
objects, depth map) into a single structured "SceneState" object that
the prediction stage consumes. This keeps prediction/ decoupled from
the details of how perception produces its data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import cv2
import numpy as np


@dataclass
class TrackedObject:
    """A single object in the current frame, enriched with ROI/depth context."""
    track_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    center: tuple
    on_track: bool
    depth_value: float          # 0..1, higher = nearer
    centroid_history: List[tuple] = field(default_factory=list)


@dataclass
class SceneState:
    """Aggregated understanding of the current frame."""
    frame_index: int
    roi_polygon: Optional[np.ndarray]
    objects: List[TrackedObject]


def build_scene_state(frame_index, roi_polygon, tracks: Dict, depth_map, frame_width, frame_height) -> SceneState:
    """
    Combine tracker output + ROI polygon + depth map into a SceneState.
    """
    roi_mask = None
    if roi_polygon is not None:
        roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_polygon.astype(np.int32)], 1)

    objects = []
    for tid, track in tracks.items():
        det = track.last_detection
        if det is None:
            continue

        on_track = False
        if roi_mask is not None:
            x1 = max(0, det.x1)
            y1 = max(0, det.y1)
            x2 = min(frame_width, det.x2)
            y2 = min(frame_height, det.y2)
            if x2 > x1 and y2 > y1:
                on_track = bool(roi_mask[y1:y2, x1:x2].sum() > 0)

        depth_value = 0.0
        if depth_map is not None:
            hh, ww = depth_map.shape[:2]
            cx = min(max(det.center[0], 0), ww - 1)
            cy = min(max(det.center[1], 0), hh - 1)
            depth_value = float(depth_map[cy, cx])

        objects.append(TrackedObject(
            track_id=tid,
            class_name=det.class_name,
            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
            confidence=det.confidence,
            center=det.center,
            on_track=on_track,
            depth_value=depth_value,
            centroid_history=list(track.history),
        ))

    return SceneState(frame_index=frame_index, roi_polygon=roi_polygon, objects=objects)
