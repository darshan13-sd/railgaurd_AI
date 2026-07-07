"""
=============================================================================
database/sqlite.py — SQLite Persistence Layer
=============================================================================

Minimal wrapper around sqlite3 for storing detection/alert events.
The database file (events.db) is created automatically on first run —
it is intentionally NOT bundled with the source code.
"""

import os
import sqlite3
import threading


class Database:
    """Thread-safe-ish wrapper around a single sqlite3 connection."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)

    def init_schema(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_index INTEGER,
                    track_id INTEGER,
                    class_name TEXT,
                    label TEXT,
                    risk_score REAL,
                    ttc_seconds REAL,
                    timestamp REAL
                )
            """)
            self._conn.commit()

    def insert_event(self, frame_index, track_id, class_name, label, risk_score, ttc_seconds, timestamp):
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events
                    (frame_index, track_id, class_name, label, risk_score, ttc_seconds, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (frame_index, track_id, class_name, label, risk_score, ttc_seconds, timestamp),
            )
            self._conn.commit()

    def fetch_recent_events(self, limit: int = 100):
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            )
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            return [dict(zip(cols, row)) for row in rows]

    def close(self):
        with self._lock:
            self._conn.close()
