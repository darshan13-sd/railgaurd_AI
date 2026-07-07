"""
=============================================================================
prediction/decision_engine.py — Final Classification & Track Status
=============================================================================

Converts each object's RiskAssessment (score + TTC) into a final
DANGER / WARNING / SAFE label, and rolls all objects up into an overall
track status message + color for the HUD.
"""

from typing import List, Tuple
from dataclasses import dataclass

from config import config as cfg
from prediction.risk_engine import RiskAssessment


@dataclass
class Decision:
    risk: RiskAssessment
    label: str   # "DANGER" | "WARNING" | "SAFE"


def classify(risk_assessments: List[RiskAssessment]) -> List[Decision]:
    """Assign a DANGER/WARNING/SAFE label to each risk assessment."""
    decisions = []
    for ra in risk_assessments:
        if ra.risk_score >= cfg.RISK_DANGER_THRESHOLD or ra.ttc_seconds <= cfg.TTC_DANGER_SECONDS:
            label = "DANGER"
        elif ra.risk_score >= cfg.RISK_WARNING_THRESHOLD or ra.ttc_seconds <= cfg.TTC_WARNING_SECONDS:
            label = "WARNING"
        else:
            label = "SAFE"
        decisions.append(Decision(risk=ra, label=label))
    return decisions


def split_by_label(decisions: List[Decision]) -> Tuple[List[Decision], List[Decision], List[Decision]]:
    danger = [d for d in decisions if d.label == "DANGER"]
    warning = [d for d in decisions if d.label == "WARNING"]
    safe = [d for d in decisions if d.label == "SAFE"]
    return danger, warning, safe


def get_track_status(danger_count: int, warning_count: int) -> Tuple[str, Tuple[int, int, int]]:
    """Overall track status message + BGR color for the HUD status bar."""
    if danger_count > 0:
        return f"DANGER - {danger_count} OBSTACLE(S) ON TRACK", cfg.STATUS_DANGER_COLOR
    elif warning_count > 0:
        return "WARNING - OBSTACLE APPROACHING TRACK", cfg.STATUS_WARNING_COLOR
    else:
        return "TRACK CLEAR", cfg.STATUS_CLEAR_COLOR
