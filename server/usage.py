"""Running tally of what the OpenAI engines have cost this session.

An estimate, deliberately: the rates below are list prices at the time of
writing and OpenAI changes them, so every rate is overridable from the
environment and the UI labels the figure "est." Nothing here ever blocks or
throttles a request -- it only counts what already happened.

Speech is billed per minute of audio and text per token, so the two are counted
in their own units and only meet at the end, in ``usd``.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field


def _rates(env: str, default: dict[str, float]) -> dict[str, float]:
    """Defaults, overridable as ``model=rate,model=rate`` in the environment."""
    out = dict(default)
    for pair in os.environ.get(env, "").split(","):
        name, _, value = pair.partition("=")
        if name.strip() and value.strip():
            try:
                out[name.strip()] = float(value)
            except ValueError:
                continue
    return out


# USD per minute of audio.
ASR_PER_MIN = _rates(
    "PRICE_ASR_PER_MIN",
    {
        "whisper-1": 0.006,
        "gpt-4o-transcribe": 0.006,
        "gpt-4o-mini-transcribe": 0.003,
        "gpt-transcribe": 0.006,
    },
)

# USD per million tokens, (input, output).
MT_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}
for _name, _r in _rates("PRICE_MT_IN_PER_MTOK", {}).items():
    MT_PER_MTOK[_name] = (_r, MT_PER_MTOK.get(_name, (0.0, 0.0))[1])
for _name, _r in _rates("PRICE_MT_OUT_PER_MTOK", {}).items():
    MT_PER_MTOK[_name] = (MT_PER_MTOK.get(_name, (0.0, 0.0))[0], _r)


@dataclass
class Usage:
    """Shared by the speech and translation engines for one server run."""

    asr_model: str = ""
    mt_model: str = ""
    audio_s: float = 0.0
    asr_requests: int = 0
    in_tok: int = 0
    out_tok: int = 0
    mt_requests: int = 0
    # Set when the API reports cached input tokens, which bill at a discount.
    cached_tok: int = 0
    _priced: bool = field(default=True, repr=False)

    def seed(self, row: dict) -> None:
        """Carry a reconnected talk's existing totals forward.

        Without this the fresh per-connection tally would overwrite the row with
        a smaller number, and the talk would appear to have cost less than it
        did the moment the socket blipped.
        """
        self.audio_s = float(row.get("audio_s") or 0)
        self.asr_requests = int(row.get("asr_requests") or 0)
        self.in_tok = int(row.get("in_tok") or 0)
        self.out_tok = int(row.get("out_tok") or 0)
        self.mt_requests = int(row.get("mt_requests") or 0)
        self.asr_model = row.get("asr_model") or ""
        self.mt_model = row.get("mt_model") or ""

    def add_audio(self, model: str, seconds: float) -> None:
        self.asr_model = model
        self.audio_s += max(0.0, seconds)
        self.asr_requests += 1

    def add_tokens(self, model: str, prompt: int, completion: int, cached: int = 0) -> None:
        self.mt_model = model
        self.in_tok += max(0, prompt)
        self.out_tok += max(0, completion)
        self.cached_tok += max(0, cached)
        self.mt_requests += 1

    @property
    def asr_usd(self) -> float:
        rate = ASR_PER_MIN.get(self.asr_model)
        return 0.0 if rate is None else self.audio_s / 60.0 * rate

    @property
    def mt_usd(self) -> float:
        rates = MT_PER_MTOK.get(self.mt_model)
        if rates is None:
            return 0.0
        rate_in, rate_out = rates
        # Cached input tokens bill at half; they are a subset of prompt tokens.
        billed_in = (self.in_tok - self.cached_tok) + self.cached_tok * 0.5
        return billed_in / 1e6 * rate_in + self.out_tok / 1e6 * rate_out

    @property
    def usd(self) -> float:
        return self.asr_usd + self.mt_usd

    @property
    def known(self) -> bool:
        """False when no rate is known, so the UI can say so instead of $0.00."""
        if self.asr_requests and self.asr_model not in ASR_PER_MIN:
            return False
        if self.mt_requests and self.mt_model not in MT_PER_MTOK:
            return False
        return True

    def snapshot(self) -> dict:
        return {
            "usd": round(self.usd, 4),
            "asr_usd": round(self.asr_usd, 4),
            "mt_usd": round(self.mt_usd, 4),
            "audio_s": round(self.audio_s, 1),
            "asr_requests": self.asr_requests,
            "in_tok": self.in_tok,
            "out_tok": self.out_tok,
            "mt_requests": self.mt_requests,
            "known": self.known,
        }


# The talk being recorded on this connection. Set by the websocket handler, so
# every task it spawns inherits it and the engines -- which are shared across
# connections -- still attribute their spend to the right talk.
CURRENT: ContextVar["Usage | None"] = ContextVar("salindia_usage", default=None)

# Everything this process has spent, talks and warmups alike.
TOTAL = Usage()


def record_audio(model: str, seconds: float) -> None:
    TOTAL.add_audio(model, seconds)
    if (u := CURRENT.get()) is not None:
        u.add_audio(model, seconds)


def record_tokens(model: str, prompt: int, completion: int, cached: int = 0) -> None:
    TOTAL.add_tokens(model, prompt, completion, cached)
    if (u := CURRENT.get()) is not None:
        u.add_tokens(model, prompt, completion, cached)
