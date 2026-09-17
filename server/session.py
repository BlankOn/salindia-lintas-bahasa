"""Per-connection streaming state machine.

The shape of it:

  audio frames -> VAD -> utterance buffer
      |-> partials:    re-decode the trailing window, one job at a time, dropped
      |                if the previous is still running (natural backpressure --
      |                stale audio is never queued)
      |-> speculation: after a short pause, start decoding the whole utterance
      |                before we're sure it has ended. If the speaker stays quiet,
      |                that result *is* the final and ASR is skipped entirely.
      `-> finals:      on trailing silence or a forced cut, start a pipeline
                       (ASR -> emit source -> translate -> emit translation).
                       Up to FINAL_WORKERS pipelines run at once, so sentence
                       N+1 can be in ASR while sentence N is being translated.

Two things keep that concurrency honest:

- Each model runs one job at a time (see WhisperEngine / Translator), because on
  a single GPU concurrent jobs just time-slice: three at once measured only
  2-7% more throughput and made the *first* result much later.
- Output is ordered by two turnstiles, one for source text and one for
  translations, so a quick short sentence can never overtake a long one.

Each utterance is stamped with the translation direction current when it
*started*, so switching direction mid-sentence takes effect from the next one.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from .asr import AsrError, AsrResult, WhisperEngine
from .config import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, Settings
from .langid import direction_for, resolve_language
from .translate import MAX_NOTES, STYLES, Translator
from .vad import EnergyVad, rms_level

log = logging.getLogger(__name__)

Sender = Callable[[dict], Awaitable[None]]

_LEVEL_EVERY_FRAMES = 5  # ~100 ms


class _Turnstile:
    """Lets holders of registered sequence numbers proceed strictly in order."""

    def __init__(self) -> None:
        self._order: deque[int] = deque()
        self._cond = asyncio.Condition()

    def register(self, seq: int) -> None:
        self._order.append(seq)

    async def wait(self, seq: int) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self._order and self._order[0] == seq)

    async def release(self, seq: int) -> None:
        async with self._cond:
            if self._order and self._order[0] == seq:
                self._order.popleft()
            else:  # released without ever reaching the front (error paths)
                try:
                    self._order.remove(seq)
                except ValueError:
                    pass
            self._cond.notify_all()


@dataclass
class _Speculation:
    seq: int
    speech_frames: int  # speech seen when the snapshot was taken
    task: asyncio.Task


class StreamSession:
    def __init__(
        self,
        send: Sender,
        asr: WhisperEngine,
        translators: dict[str, Translator],
        settings: Settings,
        directions: tuple[str, ...] | None = None,
        usage: "Usage | None" = None,
        on_usage: "Callable[[dict], Awaitable[None]] | None" = None,
    ) -> None:
        self.send = send
        # Only set for the paid engines; None means nothing is being spent.
        self.usage = usage
        # Called with each fresh snapshot so the talk record keeps up.
        self.on_usage = on_usage
        # What this connection may switch to; "auto" only when the engine can
        # detect the spoken language (see main.allowed_directions).
        self.directions = tuple(directions or settings.directions)
        self.asr = asr
        # direction -> Translator; empty in direct mode
        self.translators = translators
        self.cfg = settings

        self.vad = EnergyVad(
            settings.vad_abs_threshold,
            settings.vad_noise_ratio,
            settings.vad_onset_frames,
        )

        self._residual = np.zeros(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, settings.preroll_ms // FRAME_MS)
        )
        self._utt: list[np.ndarray] = []
        # Per frame of _utt: (rms, was_speech). Lets a forced cut find a quiet spot.
        self._utt_meta: list[tuple[float, bool]] = []
        self._utt_samples = 0
        self._silence_frames = 0
        self._speech_frames = 0
        self._active = False
        # Idle watchdog: the server's own clock, independent of the page.
        self._last_speech = time.monotonic()
        self._idle = False
        self._frame_count = 0

        self._seq = 0
        self._partial_inflight = False
        self._partial_at_samples = 0
        self._last_partial_text = ""

        # Never start in a direction this mode cannot produce: MODE=direct only
        # ever outputs English, so an en-id default lands on id-en instead.
        self.direction = settings.default_direction_for(settings.mode)
        self._utt_dir = self.direction

        self._spec: _Speculation | None = None
        self._slots = asyncio.Semaphore(max(1, settings.final_workers))
        self._pipelines: set[asyncio.Task] = set()
        self._src_turn = _Turnstile()
        self._dst_turn = _Turnstile()
        # Context is per direction: mixing en->id pairs into an id->en prompt
        # would teach the model the wrong output language.
        self._history: dict[str, list[tuple[str, str]]] = {}
        # Translation context from the presenter (see Translator / build_prompt).
        self.style = settings.id_style
        self.notes = ""
        # Auto mode: the last language heard, used when a sentence gives no clue.
        self._last_lang = "id" if self.direction == "id-en" else "en"
        self._partial_task: asyncio.Task | None = None
        self._closed = False

        # Counters for tests and logs.
        self.stats = {"speculation_hits": 0, "speculation_misses": 0}

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self._closed = True
        for task in (self._partial_task, *self._pipelines):
            if task:
                task.cancel()
        if self._spec:
            self._spec.task.cancel()

    def set_direction(self, direction: str) -> bool:
        """Switch direction for subsequent utterances. False if unsupported."""
        if direction not in self.directions:
            return False
        self.direction = direction
        return True

    def set_context(self, style: str | None, notes: str | None) -> dict:
        """Update the Indonesian style and presenter notes for later sentences."""
        style = style if style in STYLES else self.style
        notes = (notes if isinstance(notes, str) else self.notes).strip()[:MAX_NOTES]
        if style != self.style:
            # Earlier turns in the old register would pull the model back to it.
            self._history.pop("en-id", None)
        self.style, self.notes = style, notes
        return {"style": style, "notes": notes}

    # -- audio in -----------------------------------------------------------

    async def feed(self, pcm: np.ndarray) -> None:
        """Accept a chunk of float32 mono 16 kHz audio."""
        if self._closed:
            return
        # Idle-stopped: drop the audio on the floor. A page that keeps streaming
        # after the stop -- frozen, or killed mid-stream -- costs nothing.
        if self._idle:
            return

        self._residual = (
            pcm if self._residual.size == 0 else np.concatenate([self._residual, pcm])
        )
        n_frames = self._residual.size // FRAME_SAMPLES
        if n_frames == 0:
            return

        usable = self._residual[: n_frames * FRAME_SAMPLES]
        self._residual = self._residual[n_frames * FRAME_SAMPLES :]

        for i in range(n_frames):
            frame = usable[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            await self._handle_frame(frame)

    async def _maybe_idle_stop(self) -> None:
        """Stop listening after a stretch with no speech at all."""
        limit = self.cfg.idle_stop_s
        if limit <= 0 or self._idle:
            return
        if time.monotonic() - self._last_speech < limit:
            return
        if self._active:
            await self._finalise(forced=False)
        self._idle = True
        log.info("idle for %.0fs with no speech -- stopping", limit)
        await self.send({"type": "idle_stop", "after_s": round(limit)})

    def resume(self) -> bool:
        """Listen again after an idle stop. False if it was never stopped."""
        if not self._idle:
            return False
        self._idle = False
        self._last_speech = time.monotonic()
        log.info("listening again after an idle stop")
        return True

    async def _handle_frame(self, frame: np.ndarray) -> None:
        speaking = self.vad.is_speech(frame)
        self._frame_count += 1

        if self._frame_count % _LEVEL_EVERY_FRAMES == 0:
            await self.send(
                {
                    "type": "level",
                    "rms": round(rms_level(frame), 5),
                    "floor": round(self.vad.noise_floor, 5),
                    "speaking": speaking,
                }
            )

        if speaking:
            self._last_speech = time.monotonic()
            if not self._active:
                self._begin_utterance()
                await self.send(
                    {"type": "speech_start", "id": self._seq, "dir": self._utt_dir}
                )
            self._append(frame, speech=True)
            self._silence_frames = 0
            self._speech_frames += 1
        elif self._active:
            # Keep trailing silence in the buffer: Whisper decodes the tail of an
            # utterance better when it isn't cut flush against the last phoneme.
            self._append(frame, speech=False)
            self._silence_frames += 1
            silent_ms = self._silence_frames * FRAME_MS
            needed = self._silence_needed()
            if silent_ms >= needed:
                await self._finalise(forced=False)
                await self._maybe_idle_stop()
                return
            spec_ms = self.cfg.speculate_after_ms
            # ">=" not "==": frames are 20 ms, so an odd setting like 250 would
            # never match exactly. _speculate() ignores repeats for the same audio.
            if (
                0 < spec_ms < needed
                and silent_ms >= spec_ms
                # e.g. the OpenAI engine: a speculation the speaker talks through
                # is discarded, and there it was a paid request.
                and getattr(self.asr, "speculates", True)
            ):
                self._speculate()
        else:
            self._preroll.append(frame.copy())
            await self._maybe_idle_stop()
            if self._idle:
                return

        if (
            self._active
            and self._utt_samples >= self.cfg.max_utterance_s * SAMPLE_RATE
        ):
            await self._finalise(forced=True)
            return

        await self._maybe_partial()

    # -- utterance bookkeeping ---------------------------------------------

    def _begin_utterance(self) -> None:
        self._seq += 1
        self._utt_dir = self.direction
        self._active = True
        self._silence_frames = 0
        self._speech_frames = 0
        self._utt = list(self._preroll)
        self._utt_meta = [(rms_level(f), False) for f in self._utt]
        self._utt_samples = sum(len(f) for f in self._utt)
        self._preroll.clear()
        self._partial_at_samples = 0
        self._last_partial_text = ""

    def _append(self, frame: np.ndarray, *, speech: bool) -> None:
        self._utt.append(frame.copy())
        self._utt_meta.append((rms_level(frame), speech))
        self._utt_samples += len(frame)

    def _silence_needed(self) -> int:
        """Pause length that ends the current utterance.

        Long utterances make unreadable subtitles and slow finals, so past
        SOFT_MAX_S a short breath is enough to cut.
        """
        if self._utt_samples >= self.cfg.soft_max_s * SAMPLE_RATE:
            return min(self.cfg.short_silence_ms, self.cfg.silence_ms)
        return self.cfg.silence_ms

    def _quiet_cut(self) -> int:
        """Frame index for a forced cut: the quietest 100 ms of the last 2 s.

        Cutting wherever MAX_UTTERANCE_S lands tends to split a word, which
        garbles both halves.
        """
        n = len(self._utt_meta)
        span, win = 100, 5  # frames: 2 s search, 100 ms smoothing
        lo, hi = max(win, n - span), n - 2
        if hi <= lo:
            return n
        rms = np.array([r for r, _ in self._utt_meta], dtype=np.float32)
        smooth = np.convolve(rms, np.ones(win, dtype=np.float32) / win, mode="same")
        return lo + int(np.argmin(smooth[lo:hi])) + 1

    def _utterance_audio(self) -> np.ndarray:
        if not self._utt:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._utt)

    # -- partials -----------------------------------------------------------

    async def _maybe_partial(self) -> None:
        if not self._active or self._partial_inflight:
            return
        if not getattr(self.asr, "partials", True):
            return  # e.g. the OpenAI engine, where every partial is a paid call
        # Nothing new to show during a pause, and speculation owns that time.
        if self._silence_frames > 0:
            return
        new_samples = self._utt_samples - self._partial_at_samples
        if new_samples < self.cfg.partial_min_new_ms * SAMPLE_RATE / 1000:
            return
        # Don't compete with finals for the GPU: they are the results the user
        # is actually waiting on, and partials would slow them down measurably.
        if self._pipelines:
            return

        self._partial_inflight = True
        self._partial_at_samples = self._utt_samples
        audio = self._utterance_audio()
        window = int(self.cfg.partial_window_s * SAMPLE_RATE)
        if audio.size > window:
            audio = audio[-window:]
        seq = self._seq
        self._partial_task = asyncio.create_task(
            self._run_partial(seq, audio, self._utt_dir)
        )

    async def _run_partial(self, seq: int, audio: np.ndarray, direction: str) -> None:
        try:
            result = await self.asr.run(
                audio, language=_source_lang(direction), final=False
            )
            # The utterance ended while we were decoding -- the final supersedes this.
            if seq != self._seq or not self._active or self._closed:
                return
            if result.text and result.text != self._last_partial_text:
                self._last_partial_text = result.text
                await self.send(
                    {
                        "type": "partial",
                        "id": seq,
                        "dir": direction,
                        "text": result.text,
                        "ms": round(result.elapsed * 1000),
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("partial failed")
        finally:
            self._partial_inflight = False

    # -- speculation --------------------------------------------------------

    def _speculate(self) -> None:
        """Start the final-quality decode early, during the pause."""
        if self._spec and self._spec.seq == self._seq:
            if self._spec.speech_frames == self._speech_frames:
                return  # already decoding exactly this audio
            # Speech resumed after the last attempt; that result is stale.
            self._spec.task.add_done_callback(_swallow)
        audio = self._utterance_audio()
        self._spec = _Speculation(
            seq=self._seq,
            speech_frames=self._speech_frames,
            task=asyncio.create_task(
                self.asr.run(audio, language=_source_lang(self._utt_dir), final=True)
            ),
        )

    def _take_speculation(self, seq: int) -> asyncio.Task | None:
        spec, self._spec = self._spec, None
        if spec is None or spec.seq != seq:
            if spec:
                spec.task.add_done_callback(_swallow)
            return None
        if spec.speech_frames != self._speech_frames:
            # The speaker carried on after the snapshot; it misses the ending.
            spec.task.add_done_callback(_swallow)
            self.stats["speculation_misses"] += 1
            return None
        self.stats["speculation_hits"] += 1
        return spec.task

    # -- finals -------------------------------------------------------------

    async def _finalise(self, *, forced: bool) -> None:
        # A forced cut keeps everything after the quiet point for the next
        # utterance; a natural end takes the whole buffer.
        cut = self._quiet_cut() if forced else len(self._utt)
        head, tail = self._utt[:cut], self._utt[cut:]
        head_meta, tail_meta = self._utt_meta[:cut], self._utt_meta[cut:]

        audio = np.concatenate(head) if head else np.zeros(0, dtype=np.float32)
        seq = self._seq
        direction = self._utt_dir
        speech_ms = sum(sp for _, sp in head_meta) * FRAME_MS
        speculated = None if forced else self._take_speculation(seq)

        if forced:
            # Speaker is still going: the remainder opens the next utterance.
            self._seq += 1
            self._utt_dir = self.direction
            self._utt, self._utt_meta = tail, tail_meta
            self._utt_samples = sum(len(f) for f in tail)
            self._speech_frames = sum(sp for _, sp in tail_meta)
            trailing = 0
            for _, sp in reversed(tail_meta):
                if sp:
                    break
                trailing += 1
            self._silence_frames = trailing
            self._partial_at_samples = 0
            self._last_partial_text = ""
        else:
            self._active = False
            self._utt, self._utt_meta = [], []
            self._utt_samples = 0
            self._silence_frames = 0
            self._speech_frames = 0
            self._preroll.clear()
            await self.send({"type": "speech_end", "id": seq})

        if speech_ms < self.cfg.min_utterance_ms:
            if speculated:
                speculated.add_done_callback(_swallow)
            await self.send({"type": "discard", "id": seq})
            return

        self._src_turn.register(seq)
        self._dst_turn.register(seq)
        task = asyncio.create_task(
            self._pipeline(seq, audio, direction, speculated)
        )
        self._pipelines.add(task)
        task.add_done_callback(self._pipelines.discard)

    async def _pipeline(
        self,
        seq: int,
        audio: np.ndarray,
        direction: str,
        speculated: asyncio.Task | None,
    ) -> None:
        try:
            async with self._slots:
                await self._process_final(seq, audio, direction, speculated)
        except asyncio.CancelledError:
            raise
        except AsrError as exc:
            log.warning("final %d failed: %s", seq, exc)
            await self.send({"type": "error", "id": seq, "message": str(exc)})
            await self.send({"type": "discard", "id": seq})
        except Exception:
            log.exception("final failed")
            await self.send(
                {"type": "error", "id": seq, "message": "transcription failed"}
            )
            await self.send({"type": "discard", "id": seq})
        finally:
            await self._src_turn.release(seq)
            await self._dst_turn.release(seq)

    async def _process_final(
        self,
        seq: int,
        audio: np.ndarray,
        direction: str,
        speculated: asyncio.Task | None,
    ) -> None:
        result: AsrResult | None = None
        if speculated is not None:
            try:
                result = await speculated
            except Exception:
                log.exception("speculative decode failed; decoding again")
        if result is None:
            result = await self.asr.run(
                audio, language=_source_lang(direction), final=True
            )

        auto = direction == "auto"
        if auto and result.text:
            lang = resolve_language(result.text, result.language)
            if lang is None:
                lang = self._last_lang
                log.info("auto: no clear language in %r, assuming %s", result.text, lang)
            self._last_lang = lang
            direction = direction_for(lang)

        await self._src_turn.wait(seq)
        if not result.text:
            await self.send({"type": "discard", "id": seq})
            return

        if self.cfg.mode == "direct":
            # Whisper translated in the same pass; there is no source text.
            await self._dst_turn.wait(seq)
            await self.send(
                {
                    "type": "final",
                    "id": seq,
                    "dir": direction,
                    "auto": auto,
                    "src": None,
                    "dst": result.text,
                    "asr_ms": round(result.elapsed * 1000),
                    "speculated": speculated is not None,
                }
            )
            await self._report_usage()
            return

        # pipeline mode: ship the source text immediately, translate after
        await self.send(
            {
                "type": "final",
                "id": seq,
                "dir": direction,
                "auto": auto,
                "src": result.text,
                "dst": None,
                "asr_ms": round(result.elapsed * 1000),
                "speculated": speculated is not None,
            }
        )
        await self._src_turn.release(seq)

        # History is read when this sentence starts translating; with several
        # pipelines the previous sentence may still be in flight, which costs a
        # little context but never correctness.
        history = self._history.setdefault(direction, [])
        mt = await self.translators[direction].translate(
            result.text, list(history), direction, style=self.style, notes=self.notes
        )

        await self._dst_turn.wait(seq)
        if mt.text:
            history.append((result.text, mt.text))
            keep = self.cfg.mt_context_turns
            del history[: max(0, len(history) - keep)]
        await self.send(
            {
                "type": "translation",
                "id": seq,
                "dir": direction,
                "auto": auto,
                "dst": mt.text,
                "mt_ms": round(mt.elapsed * 1000),
            }
        )
        await self._report_usage()

    async def _report_usage(self) -> None:
        """Push the running estimate; the page shows it in the top bar."""
        if self.usage is None:
            return
        snap = self.usage.snapshot()
        await self.send({"type": "usage", **snap})
        if self.on_usage is not None:
            try:
                await self.on_usage(snap)
            except Exception:  # bookkeeping must never break a talk
                log.exception("failed to record usage")

    # -- stop ---------------------------------------------------------------

    async def flush(self) -> None:
        """Called when the client stops the mic: close out whatever is buffered."""
        if self._active:
            await self._finalise(forced=False)
        self.vad.reset()
        while self._pipelines:
            await asyncio.gather(*list(self._pipelines), return_exceptions=True)
        await self.send({"type": "idle"})


def _source_lang(direction: str) -> str | None:
    """Whisper language code for what is being *spoken*; None lets ASR detect it."""
    if direction == "auto":
        return None
    return direction.split("-", 1)[0]


def _swallow(task: asyncio.Task) -> None:
    """Retrieve an abandoned task's outcome so asyncio doesn't warn about it."""
    if not task.cancelled():
        task.exception()
