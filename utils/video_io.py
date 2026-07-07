"""
=============================================================================
utils/video_io.py — Video Capture / Writer Helpers
=============================================================================
"""

import sys
import cv2


def open_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[video_io] ERROR: could not open video source: {path}")
        sys.exit(1)
    return cap


def get_video_properties(cap: cv2.VideoCapture):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    return width, height, fps, total_frames


def open_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[video_io] ERROR: could not create output writer: {path}")
        sys.exit(1)
    return writer
