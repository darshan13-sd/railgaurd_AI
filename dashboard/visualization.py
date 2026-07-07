"""
=============================================================================
dashboard/visualization.py — Per-Frame HUD Drawing
=============================================================================

Draws the heads-up display onto each video frame: track ROI overlay,
bounding boxes color-coded by DANGER/WARNING/SAFE, status bar, FPS/frame
counters, and flashing alert banners. Consumed by app.py per frame, and
also used by dashboard.py to render preview frames in the live view.
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np

from config import config as cfg
from prediction.decision_engine import Decision


def draw_roi(frame: np.ndarray, polygon: Optional[np.ndarray]) -> np.ndarray:
    if polygon is None:
        return frame

    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], cfg.ROI_OVERLAY_COLOR)
    frame = cv2.addWeighted(overlay, cfg.ROI_OVERLAY_ALPHA, frame, 1 - cfg.ROI_OVERLAY_ALPHA, 0)
    cv2.polylines(frame, [polygon], isClosed=True,
                  color=cfg.ROI_BORDER_COLOR, thickness=cfg.ROI_BORDER_THICKNESS)
    return frame


def draw_decisions(frame: np.ndarray, danger: List[Decision], warning: List[Decision], safe: List[Decision]) -> np.ndarray:
    """Draw bounding boxes + labels for all classified objects."""
    for d in safe:
        _draw_single(frame, d, cfg.SAFE_BOX_COLOR, "SAFE")
    for d in warning:
        _draw_single(frame, d, cfg.WARNING_BOX_COLOR, "WARNING")
    for d in danger:
        _draw_single(frame, d, cfg.DANGER_BOX_COLOR, "DANGER")
    return frame


def _draw_single(frame: np.ndarray, decision: Decision, color: Tuple[int, int, int], tag: str) -> None:
    obj = decision.risk.obj
    x1, y1, x2, y2 = obj.x1, obj.y1, obj.x2, obj.y2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.BOX_THICKNESS)

    label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
    (lw, lh), _ = cv2.getTextSize(label, cfg.FONT, cfg.FONT_SCALE_SMALL, 1)
    label_y = max(y1 - 8, lh + 5)
    cv2.rectangle(frame, (x1, label_y - lh - 5), (x1 + lw + 6, label_y + 3), color, -1)
    cv2.putText(frame, label, (x1 + 3, label_y - 2),
                cfg.FONT, cfg.FONT_SCALE_SMALL, (255, 255, 255), 1, cv2.LINE_AA)

    ttc = decision.risk.ttc_seconds
    ttc_str = f"{ttc:.1f}s" if ttc != float("inf") else "--"
    sub_tag = f"[{tag}] risk={decision.risk.risk_score:.2f} ttc={ttc_str}"
    (tw, th), _ = cv2.getTextSize(sub_tag, cfg.FONT, cfg.FONT_SCALE_SMALL, 1)
    tag_y = min(y2 + th + 8, frame.shape[0] - 5)
    cv2.putText(frame, sub_tag, (x1, tag_y),
                cfg.FONT, cfg.FONT_SCALE_SMALL, color, 1, cv2.LINE_AA)

    cx, cy = obj.center
    cv2.circle(frame, (cx, cy), 4, color, -1)


def draw_status_bar(frame: np.ndarray, status_text: str, status_color: Tuple[int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(status_text, cfg.FONT, cfg.FONT_SCALE_LARGE, cfg.FONT_THICKNESS)

    bar_h = th + 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    tx = (w - tw) // 2
    ty = th + 12
    cv2.putText(frame, status_text, (tx, ty),
                cfg.FONT, cfg.FONT_SCALE_LARGE, status_color, cfg.FONT_THICKNESS, cv2.LINE_AA)
    return frame


def draw_info_overlay(frame: np.ndarray, fps: float, frame_number: int, total_frames: int,
                       at_risk_count: int, safe_count: int) -> np.ndarray:
    h, w = frame.shape[:2]
    y_offset = 60

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, y_offset),
                cfg.FONT, cfg.FONT_SCALE_MEDIUM, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (12, y_offset + 25),
                cfg.FONT, cfg.FONT_SCALE_MEDIUM, (200, 200, 200), 1, cv2.LINE_AA)

    summary_y = h - 20
    cv2.putText(frame, f"At Risk: {at_risk_count}  |  Safe: {safe_count}", (12, summary_y),
                cfg.FONT, cfg.FONT_SCALE_SMALL, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def draw_alerts(frame: np.ndarray, danger: List[Decision], warning: List[Decision]) -> np.ndarray:
    if not danger and not warning:
        return frame

    h, w = frame.shape[:2]
    border_thickness = 6

    if danger:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), cfg.DANGER_BOX_COLOR, border_thickness)
        alert_text = "OBSTACLE ON TRACK"
        bg_color, fg_color = (0, 0, 180), (255, 255, 255)
    else:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), cfg.WARNING_BOX_COLOR, border_thickness)
        alert_text = "OBSTACLE APPROACHING"
        bg_color, fg_color = cfg.WARNING_BOX_COLOR, (0, 0, 0)

    (tw, th), _ = cv2.getTextSize(alert_text, cfg.FONT, cfg.FONT_SCALE_LARGE, cfg.FONT_THICKNESS)
    tx = (w - tw) // 2
    ty = h - 40
    cv2.rectangle(frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), bg_color, -1)
    cv2.putText(frame, alert_text, (tx, ty),
                cfg.FONT, cfg.FONT_SCALE_LARGE, fg_color, cfg.FONT_THICKNESS, cv2.LINE_AA)
    return frame
