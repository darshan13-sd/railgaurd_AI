"""
=============================================================================
alert/notifier.py — Notification Dispatch
=============================================================================

Central place to fan out alerts to whichever channels are enabled:
console/log output today; easy to extend to SMS, email, MQTT, a
signalling relay to the train's cab, etc. Keeping this separate from
voice_alert.py and event_logger.py means adding a new channel doesn't
require touching the pipeline.
"""

import time
from typing import List

from alert.voice_alert import VoiceAlert
from alert.event_logger import EventLogger


class Notifier:
    """Fan-out notifier for DANGER/WARNING events."""

    def __init__(self, voice_alert: VoiceAlert = None, event_logger: EventLogger = None):
        self.voice_alert = voice_alert or VoiceAlert()
        self.event_logger = event_logger or EventLogger()

    def notify(self, frame_index: int, status_text: str, danger_decisions: List, warning_decisions: List):
        """
        Dispatch a notification for the current frame's status.
        Called once per frame from the pipeline.
        """
        danger_count = len(danger_decisions)
        warning_count = len(warning_decisions)

        # Console log
        if danger_count > 0:
            print(f"[ALERT] Frame {frame_index}: DANGER — {danger_count} obstacle(s) on track.")
        elif warning_count > 0:
            print(f"[ALERT] Frame {frame_index}: WARNING — obstacle approaching track.")

        # Voice
        self.voice_alert.alert_for_status(danger_count, warning_count)

        # Persist to DB
        if danger_count > 0 or warning_count > 0:
            for d in danger_decisions:
                self.event_logger.log_event(
                    frame_index=frame_index,
                    track_id=d.risk.track_id,
                    class_name=d.risk.obj.class_name,
                    label="DANGER",
                    risk_score=d.risk.risk_score,
                    ttc_seconds=d.risk.ttc_seconds,
                    timestamp=time.time(),
                )
            for d in warning_decisions:
                self.event_logger.log_event(
                    frame_index=frame_index,
                    track_id=d.risk.track_id,
                    class_name=d.risk.obj.class_name,
                    label="WARNING",
                    risk_score=d.risk.risk_score,
                    ttc_seconds=d.risk.ttc_seconds,
                    timestamp=time.time(),
                )
