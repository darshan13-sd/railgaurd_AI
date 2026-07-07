"""
=============================================================================
alert/event_logger.py — Persist Alert Events to SQLite
=============================================================================

Thin wrapper around database/sqlite.py specifically for logging
DANGER/WARNING obstacle events, so they can later be reviewed on the
dashboard or exported for incident reports.
"""

from config import config as cfg
from database.sqlite import Database


class EventLogger:
    """Logs obstacle alert events to the events table."""

    def __init__(self, db: Database = None):
        self.enabled = cfg.LOG_EVENTS_TO_DB
        self.db = db or Database(cfg.DATABASE_PATH)
        if self.enabled:
            self.db.init_schema()

    def log_event(self, frame_index, track_id, class_name, label, risk_score, ttc_seconds, timestamp):
        if not self.enabled:
            return
        self.db.insert_event(
            frame_index=frame_index,
            track_id=track_id,
            class_name=class_name,
            label=label,
            risk_score=risk_score,
            ttc_seconds=ttc_seconds,
            timestamp=timestamp,
        )
