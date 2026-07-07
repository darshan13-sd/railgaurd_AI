"""
=============================================================================
config/config.py — Central Configuration Loader for RailGuard-AI
=============================================================================

Loads settings.yaml and exposes them as module-level constants, so the
rest of the codebase can simply do:

    from config import config as cfg
    cfg.YOLO_CONFIDENCE_THRESHOLD

Editing behavior no longer requires touching Python — just edit
config/settings.yaml.
"""

import os
import yaml
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(_THIS_DIR, "settings.yaml")

with open(_SETTINGS_PATH, "r") as f:
    _S = yaml.safe_load(f)

# ===========================================================================
# PATHS
# ===========================================================================
_ROOT = os.path.dirname(_THIS_DIR)

FALLEN_TREE_MODEL_PATH = os.path.join(_ROOT, _S["paths"]["fallen_tree_model"])
COCO_MODEL_PATH = _S["paths"]["coco_model"]
DEPTH_MODEL_PATH = os.path.join(_ROOT, _S["paths"]["depth_model"])
TRACK_SEG_MODEL_PATH = os.path.join(_ROOT, _S["paths"]["track_seg_model"])

_raw_input_video = _S["paths"]["input_video"]
# Allow input_video to be a webcam index (0, 1, ...) instead of a file path.
if isinstance(_raw_input_video, int) or (isinstance(_raw_input_video, str) and _raw_input_video.strip().isdigit()):
    INPUT_VIDEO_PATH = int(_raw_input_video)
else:
    INPUT_VIDEO_PATH = _raw_input_video

OUTPUT_VIDEO_PATH = _S["paths"]["output_video"]
DATABASE_PATH = os.path.join(_ROOT, _S["paths"]["database"])

# ===========================================================================
# YOLO DETECTION SETTINGS
# ===========================================================================
YOLO_CONFIDENCE_THRESHOLD = _S["yolo"]["confidence_threshold"]
YOLO_NMS_IOU_THRESHOLD = _S["yolo"]["nms_iou_threshold"]
YOLO_MAX_DETECTIONS = _S["yolo"]["max_detections"]
YOLO_DEVICE = _S["yolo"]["device"]
YOLO_IMG_SIZE = _S["yolo"]["img_size"]

COCO_CLASSES_OF_INTEREST = {int(k): v for k, v in _S["coco_classes_of_interest"].items()}

# ===========================================================================
# TRACK SEGMENTATION
# ===========================================================================
TRACK_SEG_CONF_THRESHOLD = _S["track_segmentation"]["conf_threshold"]
TRACK_DETECTION_INTERVAL = _S["track_segmentation"]["detection_interval"]

# ===========================================================================
# TEMPORAL SMOOTHING
# ===========================================================================
TEMPORAL_SMOOTHING_ALPHA = _S["smoothing"]["alpha"]
SMOOTHING_WARMUP_FRAMES = _S["smoothing"]["warmup_frames"]
MAX_VERTEX_JUMP_PX = _S["smoothing"]["max_vertex_jump_px"]

# ===========================================================================
# DETECTION MERGING
# ===========================================================================
CROSS_MODEL_NMS_IOU_THRESHOLD = _S["merging"]["cross_model_nms_iou_threshold"]

# ===========================================================================
# DEPTH / SPEED / RISK
# ===========================================================================
DEPTH_ENABLED = _S["depth"]["enabled"]
DEPTH_INPUT_SIZE = tuple(_S["depth"]["input_size"])

TRACKER_MAX_AGE = _S["tracking"]["max_age_frames"]
TRACKER_MAX_DISTANCE_PX = _S["tracking"]["max_match_distance_px"]

SPEED_SMOOTHING_ALPHA = _S["speed"]["smoothing_alpha"]

RISK_DANGER_THRESHOLD = _S["risk"]["danger_threshold"]
RISK_WARNING_THRESHOLD = _S["risk"]["warning_threshold"]
RISK_WEIGHT_ROI_OVERLAP = _S["risk"]["weight_roi_overlap"]
RISK_WEIGHT_PROXIMITY = _S["risk"]["weight_proximity"]
RISK_WEIGHT_CLOSING_SPEED = _S["risk"]["weight_closing_speed"]

TTC_DANGER_SECONDS = _S["risk"]["ttc_danger_seconds"]
TTC_WARNING_SECONDS = _S["risk"]["ttc_warning_seconds"]

DANGER_Y_THRESHOLD_FRACTION = _S["risk"]["danger_y_threshold_fraction"]

# ===========================================================================
# ALERTS
# ===========================================================================
VOICE_ALERTS_ENABLED = _S["alerts"]["voice_enabled"]
VOICE_ALERT_COOLDOWN_SEC = _S["alerts"]["voice_cooldown_sec"]
LOG_EVENTS_TO_DB = _S["alerts"]["log_events_to_db"]

# ===========================================================================
# VISUALIZATION
# ===========================================================================
ROI_OVERLAY_COLOR = tuple(_S["viz"]["roi_overlay_color"])
ROI_OVERLAY_ALPHA = _S["viz"]["roi_overlay_alpha"]
ROI_BORDER_COLOR = tuple(_S["viz"]["roi_border_color"])
ROI_BORDER_THICKNESS = _S["viz"]["roi_border_thickness"]

DANGER_BOX_COLOR = tuple(_S["viz"]["danger_box_color"])
WARNING_BOX_COLOR = tuple(_S["viz"]["warning_box_color"])
SAFE_BOX_COLOR = tuple(_S["viz"]["safe_box_color"])
BOX_THICKNESS = _S["viz"]["box_thickness"]

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE = _S["viz"]["font_scale_large"]
FONT_SCALE_MEDIUM = _S["viz"]["font_scale_medium"]
FONT_SCALE_SMALL = _S["viz"]["font_scale_small"]
FONT_THICKNESS = _S["viz"]["font_thickness"]

STATUS_CLEAR_COLOR = tuple(_S["viz"]["status_clear_color"])
STATUS_WARNING_COLOR = tuple(_S["viz"]["status_warning_color"])
STATUS_DANGER_COLOR = tuple(_S["viz"]["status_danger_color"])

# ===========================================================================
# PERFORMANCE / LOGGING
# ===========================================================================
TRACK_PROCESSING_SCALE = _S["performance"]["track_processing_scale"]
LOG_INTERVAL = _S["performance"]["log_interval"]
DEBUG_MODE = _S["performance"]["debug_mode"]
