"""Energy-based voice activity detection over a 16 kHz mono stream.

Deliberately dependency-free: it runs on 20 ms frames, tracks the room's noise
floor, and applies hysteresis so a single loud frame doesn't open a segment and a
brief pause between words doesn't close one. It is the only VAD in the system:
mlx-whisper has no built-in VAD filter, so a false trigger here reaches the model
(and the hallucination filter in asr.py is what catches the result).
"""

from __future__ import annotations

import numpy as np

from .config import FRAME_SAMPLES


class EnergyVad:
    def __init__(
        self,
        abs_threshold: float,
        noise_ratio: float,
        onset_frames: int,
    ) -> None:
        self.abs_threshold = abs_threshold
        self.noise_ratio = noise_ratio
        self.onset_frames = onset_frames

        self._noise_floor = abs_threshold
        self._consecutive_speech = 0
        self._speaking = False

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    def reset(self) -> None:
        self._consecutive_speech = 0
        self._speaking = False

    def is_speech(self, frame: np.ndarray) -> bool:
        """Classify one 20 ms frame and update internal state."""
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        threshold = max(self.abs_threshold, self._noise_floor * self.noise_ratio)
        loud = rms > threshold

        if loud:
            self._consecutive_speech += 1
        else:
            self._consecutive_speech = 0
            # Adapt the noise floor only while quiet, and let it rise slower than
            # it falls so a passing truck doesn't deafen us for the next minute.
            alpha = 0.05 if rms < self._noise_floor else 0.005
            self._noise_floor = (1 - alpha) * self._noise_floor + alpha * rms

        if self._speaking:
            self._speaking = loud
        else:
            self._speaking = self._consecutive_speech >= self.onset_frames

        return self._speaking


def frames(buf: np.ndarray) -> list[np.ndarray]:
    """Split a buffer into whole 20 ms frames, discarding any remainder."""
    n = len(buf) // FRAME_SAMPLES
    return [buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES] for i in range(n)]


def rms_level(buf: np.ndarray) -> float:
    if buf.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(buf), dtype=np.float64)))
