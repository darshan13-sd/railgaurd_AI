"""
=============================================================================
dashboard/dashboard.py — Live Streamlit Dashboard
=============================================================================

Run with:
    streamlit run dashboard/dashboard.py

Shows:
  - Live annotated video feed (reads the same input as app.py, or an
    uploaded file)
  - Rolling event log pulled from the SQLite database
  - Current status banner

This dashboard runs its own lightweight instance of the pipeline; for
production, app.py and the dashboard should share a single pipeline
process communicating over a queue instead of duplicating inference.
"""

import time

import cv2
import streamlit as st

from config import config as cfg
from perception.object_detector import ObjectDetector
from perception.track_segmenter import TrackSegmenter
from perception.depth_estimator import DepthEstimator
from perception.object_tracker import ObjectTracker
from perception.scene_understanding import build_scene_state
from prediction.speed_estimator import SpeedEstimator
from prediction.risk_engine import assess_all
from prediction.decision_engine import classify, split_by_label, get_track_status
from dashboard import visualization as viz
from database.sqlite import Database


st.set_page_config(page_title="RailGuard-AI Dashboard", layout="wide")
st.title("🚂 RailGuard-AI — Live Obstacle Detection Dashboard")

col_video, col_events = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col_events:
    st.subheader("Recent Alert Events")
    events_placeholder = st.empty()

source = st.sidebar.text_input("Video source", value=cfg.INPUT_VIDEO_PATH)
run_button = st.sidebar.button("Start")

if run_button:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"Could not open video source: {source}")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        detector = ObjectDetector()
        segmenter = TrackSegmenter(width, height)
        depth_estimator = DepthEstimator()
        tracker = ObjectTracker()
        speed_estimator = SpeedEstimator(fps=fps)
        db = Database(cfg.DATABASE_PATH)
        db.init_schema()

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            roi_polygon, _ = segmenter.detect(frame)
            depth_map = depth_estimator.estimate(frame) if cfg.DEPTH_ENABLED else None
            detections = detector.detect_all(frame)
            tracks = tracker.update(detections)
            scene = build_scene_state(frame_index, roi_polygon, tracks, depth_map, width, height)

            risk_assessments = assess_all(scene.objects, height, width, roi_polygon, speed_estimator)
            decisions = classify(risk_assessments)
            danger, warning, safe = split_by_label(decisions)
            status_text, status_color = get_track_status(len(danger), len(warning))

            annotated = frame.copy()
            annotated = viz.draw_roi(annotated, roi_polygon)
            annotated = viz.draw_decisions(annotated, danger, warning, safe)
            annotated = viz.draw_status_bar(annotated, status_text, status_color)
            annotated = viz.draw_alerts(annotated, danger, warning)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)
            status_placeholder.markdown(f"### {status_text}")

            if frame_index % 10 == 0:
                events = db.fetch_recent_events(limit=15)
                events_placeholder.table(events)

        cap.release()
        st.success("Playback finished.")
else:
    st.info("Set a video source in the sidebar and click **Start**.")
