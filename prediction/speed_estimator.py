"""
=============================================================================
prediction/speed_estimator.py — Object Speed & Closing-Velocity Estimation
=============================================================================

Estimates each tracked object's apparent screen-space speed (pixels/sec)
and depth-based closing velocity (how fast it's approaching the camera),
using centroid history and depth history. This feeds risk_engine.py's
Time-To-Collision (TTC) calculation.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from config import config as cfg


class SpeedEstimator:
    """Tracks per-object speed using an exponential moving average."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._speed_px_s: Dict[int, float] = defaultdict(float)
        self._depth_ema: Dict[int, float] = {}
        self._closing_speed: Dict[int, float] = defaultdict(float)
        self._prev_depth: Dict[int, float] = {}

    def update(self, track_id: int, centroid_history: List[Tuple[int, int]], depth_value: float):
        """
        Update speed estimates for one track.
          - screen speed: pixel distance between last two centroids * fps
          - closing speed: rate of depth increase (nearer = higher depth value)
        """
        # --- Screen-space speed ---
        if len(centroid_history) >= 2:
            (x1, y1), (x2, y2) = centroid_history[-2], centroid_history[-1]
            dist_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            instant_speed = dist_px * self.fps
            alpha = cfg.SPEED_SMOOTHING_ALPHA
            self._speed_px_s[track_id] = alpha * instant_speed + (1 - alpha) * self._speed_px_s.get(track_id, instant_speed)

        # --- Depth-based closing speed ---
        prev_depth = self._prev_depth.get(track_id)
        if prev_depth is not None:
            # depth_value near 1 == very close; increase in depth == closing in
            instant_closing = (depth_value - prev_depth) * self.fps
            alpha = cfg.SPEED_SMOOTHING_ALPHA
            self._closing_speed[track_id] = (
                alpha * instant_closing + (1 - alpha) * self._closing_speed.get(track_id, instant_closing)
            )
        self._prev_depth[track_id] = depth_value

    def get_screen_speed(self, track_id: int) -> float:
        """Pixels/second of apparent screen motion."""
        return self._speed_px_s.get(track_id, 0.0)

    def get_closing_speed(self, track_id: int) -> float:
        """Depth-units/second; positive means getting closer to the camera."""
        return self._closing_speed.get(track_id, 0.0)

    def estimate_time_to_collision(self, track_id: int, depth_value: float) -> float:
        """
        Rough Time-To-Collision in seconds, based on how quickly the
        object's depth value is approaching 1.0 (maximum proximity).
        Returns float('inf') if not closing in.
        """
        closing = self.get_closing_speed(track_id)
        if closing <= 1e-6:
            return float("inf")
        remaining = max(0.0, 1.0 - depth_value)
        return remaining / closing

    def cleanup(self, active_track_ids):
        """Remove state for tracks no longer active."""
        for store in (self._speed_px_s, self._depth_ema, self._closing_speed, self._prev_depth):
            for tid in list(store.keys()):
                if tid not in active_track_ids:
                    del store[tid]
