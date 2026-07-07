"""
=============================================================================
perception/track_segmenter.py — Rail/Track ROI Segmentation
=============================================================================

Runs an ONNX segmentation model (models/track_seg.onnx) to find the
railway track region, converts the predicted mask into a polygon ROI,
and applies temporal smoothing across frames to prevent flicker.

If the ONNX model can't be loaded, falls back to a static trapezoid ROI
so the rest of the pipeline can still run.
"""

import cv2
import numpy as np

from config import config as cfg

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


class TrackSegmenter:
    """Detects railway tracks and produces a smoothed ROI polygon."""

    def __init__(self, frame_width, frame_height):
        self.W = frame_width
        self.H = frame_height

        self._smoothed_polygon = None
        self._frame_count = 0
        self._default_polygon = self._build_default_polygon()
        self._last_valid_polygon = self._default_polygon.copy()

        self.session = None
        if _ONNX_AVAILABLE:
            try:
                self.session = ort.InferenceSession(
                    cfg.TRACK_SEG_MODEL_PATH,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._input_name = self.session.get_inputs()[0].name
                _, _, self._in_h, self._in_w = self.session.get_inputs()[0].shape
                print(f"[TrackSegmenter] ONNX model loaded: {cfg.TRACK_SEG_MODEL_PATH}")
            except Exception as e:
                print(f"[TrackSegmenter] Could not load ONNX model ({e}); using default ROI fallback.")
                self.session = None
        else:
            print("[TrackSegmenter] onnxruntime not installed; using default ROI fallback.")

    def detect(self, frame):
        """
        Process one frame. Returns:
          - roi_polygon: smoothed track ROI (Nx2 int32 array)
          - mask: raw binary segmentation mask (or None)
        """
        self._frame_count += 1
        raw_polygon = None
        mask = None

        run_now = (self._frame_count % cfg.TRACK_DETECTION_INTERVAL == 1) or (self._frame_count == 1)

        if run_now and self.session is not None:
            try:
                mask = self._run_segmentation(frame)
                raw_polygon = self._mask_to_polygon(mask)
            except Exception as e:
                print(f"[TrackSegmenter] Inference error: {e}")

        if raw_polygon is not None:
            roi_polygon = self._smooth_polygon(raw_polygon)
            self._last_valid_polygon = roi_polygon
        else:
            roi_polygon = self._last_valid_polygon

        return roi_polygon, mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_segmentation(self, frame):
        resized = cv2.resize(frame, (self._in_w, self._in_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        outputs = self.session.run(None, {self._input_name: blob})
        raw_mask = outputs[0]

        # Squeeze to 2D probability/class mask
        raw_mask = np.squeeze(raw_mask)
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[0]

        binary_mask = (raw_mask > cfg.TRACK_SEG_CONF_THRESHOLD).astype(np.uint8) * 255
        binary_mask = cv2.resize(binary_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return binary_mask

    def _mask_to_polygon(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return None

        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        return approx.reshape(-1, 2).astype(np.int32)

    def _build_default_polygon(self):
        return np.array([
            [int(self.W * 0.42), int(self.H * 0.35)],
            [int(self.W * 0.58), int(self.H * 0.35)],
            [int(self.W * 0.90), self.H - 1],
            [int(self.W * 0.10), self.H - 1],
        ], dtype=np.int32)

    def _smooth_polygon(self, raw_polygon):
        if self._smoothed_polygon is None or self._frame_count <= cfg.SMOOTHING_WARMUP_FRAMES:
            self._smoothed_polygon = raw_polygon.astype(np.float64)
            return raw_polygon

        if len(self._smoothed_polygon) != len(raw_polygon):
            self._smoothed_polygon = raw_polygon.astype(np.float64)
            return raw_polygon

        alpha = cfg.TEMPORAL_SMOOTHING_ALPHA
        raw = raw_polygon.astype(np.float64)
        prev = self._smoothed_polygon

        # Outlier rejection: clamp per-vertex movement
        delta = raw - prev
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        dist_safe = np.where(dist == 0, 1, dist)
        scale = np.minimum(1.0, cfg.MAX_VERTEX_JUMP_PX / dist_safe)
        clamped = prev + delta * scale

        self._smoothed_polygon = alpha * clamped + (1 - alpha) * prev
        return self._smoothed_polygon.astype(np.int32)
