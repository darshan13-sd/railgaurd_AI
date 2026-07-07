"""
=============================================================================
prediction/risk_engine.py — Weighted Risk Scoring
=============================================================================

Computes a single 0..1 risk score per tracked object by combining:
  - ROI overlap (is it on the track at all?)
  - Proximity (how far down/near in frame / how near in depth?)
  - Closing speed (is it approaching quickly?)

The weighted score plus Time-To-Collision (TTC) are then handed to
decision_engine.py, which turns them into DANGER / WARNING / SAFE
classifications and human-readable status text.
"""

from dataclasses import dataclass
from typing import List

from config import config as cfg
from perception.scene_understanding import TrackedObject
from prediction.speed_estimator import SpeedEstimator
from prediction.collision_predictor import predict_collision, CollisionPrediction


@dataclass
class RiskAssessment:
    track_id: int
    obj: TrackedObject
    risk_score: float
    ttc_seconds: float
    collision_prediction: CollisionPrediction


def compute_risk(
    obj: TrackedObject,
    frame_height: int,
    speed_estimator: SpeedEstimator,
    roi_polygon,
    frame_width: int,
) -> RiskAssessment:
    """Compute a weighted 0..1 risk score for a single tracked object."""

    # --- Component 1: ROI overlap ---
    roi_component = 1.0 if obj.on_track else 0.0

    # --- Component 2: Proximity (frame position + depth) ---
    y_frac = obj.y2 / max(1, frame_height) if hasattr(obj, "y2") else 0.0
    position_proximity = max(0.0, min(1.0, (y_frac - cfg.DANGER_Y_THRESHOLD_FRACTION) /
                                       (1.0 - cfg.DANGER_Y_THRESHOLD_FRACTION + 1e-6)))
    proximity_component = max(position_proximity, obj.depth_value)

    # --- Component 3: Closing speed ---
    speed_estimator.update(obj.track_id, obj.centroid_history, obj.depth_value)
    closing_speed = speed_estimator.get_closing_speed(obj.track_id)
    closing_component = max(0.0, min(1.0, closing_speed))  # already ~0..1 scale (depth units/sec)

    risk_score = (
        cfg.RISK_WEIGHT_ROI_OVERLAP * roi_component +
        cfg.RISK_WEIGHT_PROXIMITY * proximity_component +
        cfg.RISK_WEIGHT_CLOSING_SPEED * closing_component
    )
    risk_score = max(0.0, min(1.0, risk_score))

    ttc = speed_estimator.estimate_time_to_collision(obj.track_id, obj.depth_value)

    collision_pred = predict_collision(
        obj.centroid_history, roi_polygon, frame_width, frame_height
    )

    # If trajectory predicts entering the track soon, boost the score
    if collision_pred.will_enter_track and not obj.on_track:
        risk_score = max(risk_score, cfg.RISK_WARNING_THRESHOLD + 0.05)

    return RiskAssessment(
        track_id=obj.track_id,
        obj=obj,
        risk_score=risk_score,
        ttc_seconds=ttc,
        collision_prediction=collision_pred,
    )


def assess_all(objects: List[TrackedObject], frame_height, frame_width, roi_polygon, speed_estimator) -> List[RiskAssessment]:
    return [
        compute_risk(obj, frame_height, speed_estimator, roi_polygon, frame_width)
        for obj in objects
    ]
