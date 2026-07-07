"""
=============================================================================
perception/depth_estimator.py — Monocular Depth Estimation
=============================================================================

Runs a Depth-Anything ONNX model to produce a per-pixel relative depth
map for the frame. Used by prediction/speed_estimator.py and
prediction/risk_engine.py to reason about how close an obstacle actually
is, instead of relying on bounding-box Y-position alone.

If the model or onnxruntime is unavailable, depth queries fall back to
a simple heuristic based on bounding-box vertical position (lower in
frame = closer), so the rest of the pipeline degrades gracefully.
"""

import cv2
import numpy as np

from config import config as cfg

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


class DepthEstimator:
    """Produces a normalized relative depth map for each frame."""

    def __init__(self):
        self.session = None
        self.enabled = cfg.DEPTH_ENABLED and _ONNX_AVAILABLE

        if self.enabled:
            try:
                self.session = ort.InferenceSession(
                    cfg.DEPTH_MODEL_PATH,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._input_name = self.session.get_inputs()[0].name
                print(f"[DepthEstimator] ONNX model loaded: {cfg.DEPTH_MODEL_PATH}")
            except Exception as e:
                print(f"[DepthEstimator] Could not load depth model ({e}); using heuristic fallback.")
                self.session = None
                self.enabled = False
        else:
            print("[DepthEstimator] Depth disabled or onnxruntime missing; using heuristic fallback.")

    def estimate(self, frame):
        """
        Returns a depth map the same size as the frame.
        Values are normalized 0 (far) .. 1 (near).
        """
        h, w = frame.shape[:2]

        if self.session is not None:
            try:
                return self._run_model(frame, h, w)
            except Exception as e:
                print(f"[DepthEstimator] Inference error: {e}; falling back to heuristic.")

        return self._heuristic_depth(h, w)

    def depth_at_box(self, depth_map, x1, y1, x2, y2):
        """Median depth value (0..1, higher = nearer) inside a bounding box."""
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = depth_map[y1:y2, x1:x2]
        return float(np.median(region))

    # ------------------------------------------------------------------
    def _run_model(self, frame, orig_h, orig_w):
        in_h, in_w = cfg.DEPTH_INPUT_SIZE
        resized = cv2.resize(frame, (in_w, in_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        outputs = self.session.run(None, {self._input_name: blob})
        depth = np.squeeze(outputs[0])

        # Normalize to 0..1 (near = 1)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def _heuristic_depth(self, h, w):
        """Fallback: assume 'near' increases linearly toward the bottom of frame."""
        gradient = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1)
        return np.repeat(gradient, w, axis=1)
