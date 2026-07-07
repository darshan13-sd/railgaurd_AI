"""
=============================================================================
perception/object_tracker.py — Lightweight Multi-Object Tracker
=============================================================================

Assigns a persistent track ID to each detection across frames using
greedy centroid matching (nearest-neighbor within a distance threshold).
This gives the prediction stage a position history per object, which is
required for speed/closing-velocity estimation and collision prediction.

This is intentionally dependency-free (no external tracker library) so
the system stays easy to install; swap in DeepSORT/ByteTrack later by
replacing this module's `update()` implementation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from config import config as cfg
from perception.object_detector import Detection


@dataclass
class Track:
    """A tracked object across frames."""
    track_id: int
    class_name: str
    history: List[Tuple[int, int]] = field(default_factory=list)   # centroid history
    boxes_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    last_detection: Detection = None
    age: int = 0            # frames since last matched
    hits: int = 0           # total number of matches

    @property
    def centroid(self):
        return self.history[-1] if self.history else None


class ObjectTracker:
    """Greedy centroid tracker with track aging."""

    def __init__(self):
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        """
        Match new detections to existing tracks, create new tracks for
        unmatched detections, and age out stale tracks.

        Returns dict of {track_id: Track}, and each Detection gets a
        `.track_id` attribute set dynamically for downstream use.
        """
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Build candidate matches: (distance, track_id, det_idx)
        candidates = []
        for tid, track in self._tracks.items():
            if track.centroid is None:
                continue
            for di in unmatched_dets:
                det = detections[di]
                dist = _euclidean(track.centroid, det.center)
                if dist <= cfg.TRACKER_MAX_DISTANCE_PX:
                    candidates.append((dist, tid, di))

        candidates.sort(key=lambda c: c[0])

        used_dets = set()
        for dist, tid, di in candidates:
            if tid in matched_track_ids or di in used_dets:
                continue
            det = detections[di]
            track = self._tracks[tid]
            track.history.append(det.center)
            track.boxes_history.append((det.x1, det.y1, det.x2, det.y2))
            track.last_detection = det
            track.age = 0
            track.hits += 1
            det.track_id = tid
            matched_track_ids.add(tid)
            used_dets.add(di)

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            if di in used_dets:
                continue
            det = detections[di]
            tid = self._next_id
            self._next_id += 1
            track = Track(track_id=tid, class_name=det.class_name)
            track.history.append(det.center)
            track.boxes_history.append((det.x1, det.y1, det.x2, det.y2))
            track.last_detection = det
            track.hits = 1
            det.track_id = tid
            self._tracks[tid] = track

        # Age out unmatched tracks
        dead_ids = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track.age += 1
            if track.age > cfg.TRACKER_MAX_AGE:
                dead_ids.append(tid)

        for tid in dead_ids:
            del self._tracks[tid]

        # Cap history length for memory
        for track in self._tracks.values():
            if len(track.history) > 60:
                track.history = track.history[-60:]
                track.boxes_history = track.boxes_history[-60:]

        return self._tracks


def _euclidean(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
