"""
=============================================================================
app.py — RailGuard-AI Entry Point
=============================================================================

Orchestrates the complete pipeline end-to-end:

  perception:
    1. TrackSegmenter   — locates the rail ROI polygon
    2. DepthEstimator   — monocular relative depth map
    3. ObjectDetector   — dual YOLO detection (fallen tree + COCO), merged
    4. ObjectTracker    — assigns persistent IDs across frames
    5. scene_understanding.build_scene_state — fuses all of the above

  prediction:
    6. SpeedEstimator + risk_engine.assess_all — per-object risk scoring
    7. decision_engine.classify — DANGER / WARNING / SAFE labels

  alert:
    8. Notifier — voice alert + console log + SQLite event logging

  dashboard:
    9. visualization — draws the annotated HUD frame

Usage:
    python app.py

Configure everything via config/settings.yaml before running.
"""

import time
import sys

from config import config as cfg
from utils.video_io import open_capture, get_video_properties, open_writer

from perception.object_detector import ObjectDetector
from perception.track_segmenter import TrackSegmenter
from perception.depth_estimator import DepthEstimator
from perception.object_tracker import ObjectTracker
from perception.scene_understanding import build_scene_state

from prediction.speed_estimator import SpeedEstimator
from prediction.risk_engine import assess_all
from prediction.decision_engine import classify, split_by_label, get_track_status

from alert.voice_alert import VoiceAlert
from alert.event_logger import EventLogger
from alert.notifier import Notifier

from dashboard import visualization as viz


class RailGuardApp:
    """Wires together perception -> prediction -> alert -> visualization."""

    def __init__(self, width, height, total_frames, fps):
        self.W = width
        self.H = height
        self.total_frames = total_frames
        self.fps = fps
        self._frame_count = 0
        self._fps_buffer = []
        self._fps_display = 0.0

        print("=" * 70)
        print("  Initializing RailGuard-AI pipeline modules...")
        print("=" * 70)

        self.detector = ObjectDetector()
        self.segmenter = TrackSegmenter(width, height)
        self.depth_estimator = DepthEstimator()
        self.tracker = ObjectTracker()
        self.speed_estimator = SpeedEstimator(fps=fps)

        self.voice_alert = VoiceAlert()
        self.event_logger = EventLogger()
        self.notifier = Notifier(self.voice_alert, self.event_logger)

        print("  ✅ All modules ready.")
        print("=" * 70)

    def process_frame(self, frame):
        t_start = time.time()
        self._frame_count += 1

        # --- Perception ---
        roi_polygon, _mask = self.segmenter.detect(frame)
        depth_map = self.depth_estimator.estimate(frame) if cfg.DEPTH_ENABLED else None
        detections = self.detector.detect_all(frame)
        tracks = self.tracker.update(detections)
        scene = build_scene_state(self._frame_count, roi_polygon, tracks, depth_map, self.W, self.H)

        # --- Prediction ---
        risk_assessments = assess_all(scene.objects, self.H, self.W, roi_polygon, self.speed_estimator)
        decisions = classify(risk_assessments)
        danger, warning, safe = split_by_label(decisions)
        status_text, status_color = get_track_status(len(danger), len(warning))

        # --- Alert ---
        self.notifier.notify(self._frame_count, status_text, danger, warning)

        # --- Visualization ---
        annotated = frame.copy()
        annotated = viz.draw_roi(annotated, roi_polygon)
        annotated = viz.draw_decisions(annotated, danger, warning, safe)
        annotated = viz.draw_status_bar(annotated, status_text, status_color)

        fps = self._compute_fps(time.time() - t_start)
        annotated = viz.draw_info_overlay(
            annotated, fps, self._frame_count, self.total_frames,
            len(danger) + len(warning), len(safe)
        )
        annotated = viz.draw_alerts(annotated, danger, warning)

        if self._frame_count % cfg.LOG_INTERVAL == 0:
            print(
                f"  Frame {self._frame_count:>5}/{self.total_frames}  |  "
                f"FPS: {fps:>5.1f}  |  "
                f"Objects: {len(scene.objects):>2} "
                f"(Danger: {len(danger)}, Warning: {len(warning)}, Safe: {len(safe)})  |  "
                f"{status_text}"
            )

        return annotated

    def _compute_fps(self, elapsed):
        if elapsed > 0:
            self._fps_buffer.append(1.0 / elapsed)
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)
        if self._fps_buffer:
            self._fps_display = sum(self._fps_buffer) / len(self._fps_buffer)
        return self._fps_display


def main():
    print("\n" + "=" * 70)
    print("   🚂 RailGuard-AI — Railway Track Obstacle Detection System")
    print("=" * 70)

    print(f"\n📹 Input Video : {cfg.INPUT_VIDEO_PATH}")
    cap = open_capture(cfg.INPUT_VIDEO_PATH)
    width, height, fps, total_frames = get_video_properties(cap)
    print(f"   Resolution  : {width} x {height}")
    print(f"   FPS         : {fps:.1f}")
    print(f"   Total Frames: {total_frames}")

    print(f"\n💾 Output Video: {cfg.OUTPUT_VIDEO_PATH}")
    out = open_writer(cfg.OUTPUT_VIDEO_PATH, fps, width, height)

    app = RailGuardApp(width, height, total_frames, fps)

    print("\n" + "-" * 70)
    print("  Running inference...")
    print("-" * 70)

    t_total_start = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            annotated_frame = app.process_frame(frame)
            out.write(annotated_frame)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user.")

    cap.release()
    out.release()

    t_total = time.time() - t_total_start
    print("\n" + "=" * 70)
    print("  ✅ INFERENCE COMPLETE")
    print("=" * 70)
    print(f"  Frames Processed : {frame_count}")
    print(f"  Total Time       : {t_total:.1f}s")
    print(f"  Average FPS      : {frame_count / max(t_total, 0.001):.1f}")
    print(f"  Output Saved     : {cfg.OUTPUT_VIDEO_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
