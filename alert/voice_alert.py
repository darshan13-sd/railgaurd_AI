"""
=============================================================================
alert/voice_alert.py — Spoken Voice Alerts
=============================================================================

Speaks alerts out loud using pyttsx3 (offline TTS, no internet required),
on a background thread so it never blocks the video pipeline. A cooldown
prevents the same alert from being spoken every single frame.
"""

import threading
import time

from config import config as cfg

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class VoiceAlert:
    """Non-blocking, rate-limited text-to-speech alerting."""

    def __init__(self):
        self.enabled = cfg.VOICE_ALERTS_ENABLED and _TTS_AVAILABLE
        self._last_spoken_at = 0.0
        self._last_message = None
        self._lock = threading.Lock()

        if self.enabled:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 175)
            except Exception as e:
                print(f"[VoiceAlert] Failed to init TTS engine ({e}); voice alerts disabled.")
                self.enabled = False
        else:
            if not _TTS_AVAILABLE:
                print("[VoiceAlert] pyttsx3 not installed; voice alerts disabled.")

    def speak(self, message: str, force: bool = False):
        """
        Speak `message` if the cooldown has elapsed (or `force=True`),
        without blocking the caller.
        """
        if not self.enabled:
            return

        now = time.time()
        if not force:
            if message == self._last_message and (now - self._last_spoken_at) < cfg.VOICE_ALERT_COOLDOWN_SEC:
                return

        self._last_spoken_at = now
        self._last_message = message

        thread = threading.Thread(target=self._speak_sync, args=(message,), daemon=True)
        thread.start()

    def _speak_sync(self, message: str):
        with self._lock:
            try:
                self._engine.say(message)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[VoiceAlert] TTS error: {e}")

    def alert_for_status(self, danger_count: int, warning_count: int):
        """Convenience wrapper: speak an appropriate message for the current status."""
        if danger_count > 0:
            self.speak(f"Danger. {danger_count} obstacle{'s' if danger_count != 1 else ''} on track.")
        elif warning_count > 0:
            self.speak("Warning. Obstacle approaching track.")
