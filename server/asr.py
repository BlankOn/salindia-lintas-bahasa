"""Whisper on the M4 GPU via MLX.

Everything runs on one worker thread. MLX evaluates lazily against a shared
Metal command queue, so serialising model calls is both simpler and, on a single
GPU, no slower than fighting over it from several threads.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .gpu import BACKGROUND, URGENT, GpuGate

log = logging.getLogger(__name__)

# Whisper emits these over silence, music or room tone -- an artefact of its
# training data (YouTube captions), not of anything the speaker said. Matched
# after lowercasing and stripping punctuation.
#
# Caption boilerplate nobody says into a live interpreter: always dropped.
_ALWAYS_HALLUCINATED = {
    "terima kasih telah menonton",
    "terima kasih telah menonton video ini",
    "sampai jumpa di video selanjutnya",
    "jangan lupa like dan subscribe",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by the amaraorg community",
    "subtitle by",
    "you",
}
# Short phrases people genuinely say ("Thank you." ends many English turns), so
# they are only dropped when Whisper itself doubts there was speech at all.
_SUSPECT_IF_UNSURE = {
    "terima kasih",
    "terima kasih banyak",
    "thank you",
    "thank you very much",
    "thanks",
    "bye",
}
_SUSPECT_NO_SPEECH = 0.3

_NO_SPEECH_MAX = 0.6
_AVG_LOGPROB_MIN = -1.0


class AsrError(RuntimeError):
    """A speech-to-text failure with a message fit to show the user."""


@dataclass
class AsrResult:
    text: str
    language: str | None
    no_speech_prob: float
    avg_logprob: float
    elapsed: float

    @property
    def is_empty(self) -> bool:
        return not self.text


def _clean(text: str) -> str:
    text = text.strip()
    # Whisper sometimes wraps non-speech guesses in brackets: [Musik], (applause)
    text = re.sub(r"^[\[\(\*].*?[\]\)\*]$", "", text).strip()
    return text


def _looks_hallucinated(text: str, no_speech: float, logprob: float) -> bool:
    if not text:
        return True
    normalised = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if normalised in _ALWAYS_HALLUCINATED:
        return True
    if normalised in _SUSPECT_IF_UNSURE and no_speech > _SUSPECT_NO_SPEECH:
        return True
    if no_speech > _NO_SPEECH_MAX and logprob < _AVG_LOGPROB_MIN:
        return True
    # A handful of words repeated forever is the classic decoder loop.
    words = normalised.split()
    if len(words) >= 8 and len(set(words)) <= 2:
        return True
    return False


class WhisperEngine:
    """Serialised, async-friendly wrapper around mlx_whisper."""

    engine = "local"
    partials = True  # cheap here: re-decoding costs GPU time, not money
    auto_detect = False  # auto direction is offered with the OpenAI engine only

    def __init__(self, repo: str, task: str, gate: GpuGate | None = None) -> None:
        self.repo = repo
        self.task = task  # "transcribe" or "translate"
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.gate = gate or GpuGate()

    # -- blocking side, always on the single worker thread ------------------

    def _transcribe_blocking(
        self,
        audio: np.ndarray,
        *,
        language: str,
        initial_prompt: str | None,
        best_of: int,
    ) -> AsrResult:
        import mlx_whisper

        started = time.perf_counter()
        out = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            task=self.task,
            language=language,
            temperature=0.0 if best_of <= 1 else (0.0, 0.2, 0.4),
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=False,
            fp16=True,
        )
        elapsed = time.perf_counter() - started

        segments = out.get("segments") or []
        no_speech = max((s.get("no_speech_prob", 0.0) for s in segments), default=0.0)
        logprob = min((s.get("avg_logprob", 0.0) for s in segments), default=0.0)

        return AsrResult(
            text=_clean(out.get("text", "")),
            language=out.get("language"),
            no_speech_prob=float(no_speech),
            avg_logprob=float(logprob),
            elapsed=elapsed,
        )

    # -- async API ----------------------------------------------------------

    async def run(
        self,
        audio: np.ndarray,
        *,
        language: str = "id",
        initial_prompt: str | None = None,
        final: bool = False,
    ) -> AsrResult:
        """Decode ``audio``. ``language`` is the *spoken* language (``id`` or ``en``);
        pinning it avoids Whisper's per-call language detection, which is both
        slower and easily fooled by code-switched Indonesian."""
        loop = asyncio.get_running_loop()
        # Partials are cosmetic: anything the user is waiting on goes first.
        async with self.gate.hold(URGENT if final else BACKGROUND):
            result = await loop.run_in_executor(
                self._pool,
                lambda: self._transcribe_blocking(
                    audio,
                    language=language,
                    initial_prompt=initial_prompt,
                    best_of=2 if final else 1,
                ),
            )

        if _looks_hallucinated(result.text, result.no_speech_prob, result.avg_logprob):
            log.debug("dropped likely hallucination: %r", result.text)
            result.text = ""
        return result

    async def warmup(self) -> None:
        """Pull weights and let Metal compile its kernels before the first user."""
        log.info("loading ASR model %s (task=%s)...", self.repo, self.task)
        started = time.perf_counter()
        silence = np.zeros(16_000, dtype=np.float32)
        await self.run(silence)
        log.info("ASR ready in %.1fs", time.perf_counter() - started)

    async def aclose(self) -> None:
        pass

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
