"""Indonesian <-> English text translation with a small MLX LLM.

Used only in MODE=pipeline, and only on finalised utterances -- partials stay in
the spoken language, which keeps the live latency budget entirely inside Whisper.

An LLM rather than a dedicated MT model because conversational Indonesian leans
hard on slang, particles (dong, sih, kan, nih) and Jakarta-informal forms that
older MarianMT/NLLB checkpoints handle poorly -- in both directions.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .gpu import URGENT, GpuGate

log = logging.getLogger(__name__)

ID_EN_PROMPT = """You are a live interpreter turning spoken Indonesian into natural English.

Rules:
- Output ONLY the English translation. No notes, no quotes, no romanisation.
- Match the speaker's register: keep casual Indonesian casual, formal formal.
- Render slang and particles (dong, sih, kan, nih, banget, deh, aja) as English
  tone and word choice, never as literal words.
- The input is speech-to-text, so it may lack punctuation or contain small
  transcription errors. Translate the intended meaning, not the literal string.
- Keep proper nouns, brand names and code-switched English words as they are.
- Keep every number, amount and currency exactly as spoken. Never convert
  currencies (Rp stays Rp, $ stays $).
- If the input is not meaningful speech, output nothing.

Examples:
Aduh, macet banget nih di jalan.
-> Ugh, the traffic out there is awful.

Menurut aku sih harganya kemahalan banget, mending cari vendor lain aja deh.
-> Honestly I think it's way overpriced. We're better off finding another vendor.

Tolong kirimkan laporan keuangan kuartal ketiga sebelum hari Jumat ya.
-> Please send over the Q3 financial report before Friday.

Gue lagi nggak enak badan nih, kayaknya nggak bisa dateng meeting besok.
-> I'm not feeling great, I don't think I can make tomorrow's meeting.

Udah dong, jangan dibahas lagi, kan udah selesai kemarin.
-> Come on, let's drop it, it was already settled yesterday."""


_EN_ID_HEAD = """You are a live interpreter turning spoken English into natural Indonesian.

Rules:
- Output ONLY the Indonesian translation. No notes, no quotes, no English gloss.
"""

# The register rule and the examples must agree: few-shot examples steer the
# model harder than a rule does, so each style gets its own set.
_REGISTER = {
    "formal": (
        "- Always use polite, standard Indonesian with saya and Anda (Bapak/Ibu when\n"
        "  addressing people), even when the English is casual. Never use aku, kamu,\n"
        "  gue, nggak, udah or other slang.\n"
    ),
    "casual": (
        "- Always use relaxed, everyday spoken Indonesian with aku and kamu\n"
        "  (nggak, udah, banget are fine), even when the English is formal.\n"
    ),
    "match": (
        "- Match the speaker's register. Casual English becomes everyday spoken\n"
        "  Indonesian (aku/kamu, nggak, udah, banget). Neutral or formal English becomes\n"
        "  standard Indonesian (saya/Anda).\n"
    ),
}

_EN_ID_TAIL = """- Write it the way an Indonesian speaker would actually say it, not as a stiff
  word-for-word rendering. Pick the meaning that fits the context.
- Use one pronoun set per sentence: never mix saya with aku.
- Keep tense and aspect: "not yet" / "still hasn't" is belum, "already" is sudah.
- Keep every number, amount and currency exactly as spoken. Never convert
  currencies ($ stays $, Rp stays Rp).
- Keep proper nouns, brand names, and tech/work terms Indonesians normally say in
  English (meeting, deadline, deploy, server, email) as they are.
- The input is speech-to-text, so it may lack punctuation or contain small
  transcription errors. Translate the intended meaning, not the literal string.
- If the input is not meaningful speech, output nothing.

Examples:
"""

_EN_ID_EXAMPLES = {
    "formal": """Hey, sorry I'm running late, the traffic is insane.
-> Mohon maaf, saya terlambat, macetnya parah sekali.

Could you please send me the updated proposal by Friday?
-> Mohon kirimkan proposal yang sudah diperbarui paling lambat hari Jumat.

Honestly, I don't think that's a good idea.
-> Sejujurnya, menurut saya itu bukan ide yang baik.

We pushed the fix to production last night and it looks stable.
-> Perbaikannya sudah kami rilis ke production tadi malam, dan sejauh ini terlihat stabil.

Good morning, everyone. Let's get started.
-> Selamat pagi, Bapak dan Ibu sekalian. Mari kita mulai.""",
    "casual": """Hey, sorry I'm running late, the traffic is insane.
-> Eh, sori aku telat, macetnya parah banget.

Could you please send me the updated proposal by Friday?
-> Tolong kirimin proposal yang udah diperbarui paling lambat Jumat ya.

Honestly, I don't think that's a good idea.
-> Jujur, menurutku itu bukan ide yang bagus.

We pushed the fix to production last night and it looks stable.
-> Fix-nya udah kami push ke production tadi malam, dan kelihatannya udah stabil.

Good morning, everyone. Let's get started.
-> Pagi, semuanya. Yuk kita mulai.""",
    "match": """Hey, sorry I'm running late, the traffic is insane.
-> Eh, sori aku telat, macetnya parah banget.

Could you please send me the updated proposal by Friday?
-> Boleh tolong kirimkan proposal yang sudah diperbarui paling lambat hari Jumat?

Honestly, I don't think that's a good idea.
-> Jujur, menurutku itu bukan ide yang bagus.

We pushed the fix to production last night and it looks stable.
-> Fix-nya udah kami push ke production tadi malam, dan kelihatannya udah stabil.

Good morning, everyone. Let's get started.
-> Selamat pagi, semuanya. Mari kita mulai.""",
}

DIRECTIONS = ("id-en", "en-id")
STYLES = tuple(_REGISTER)  # formal, casual, match
MAX_NOTES = 1500


def build_prompt(direction: str, style: str = "match", notes: str = "") -> str:
    """System prompt for one direction, Indonesian style and presenter notes.

    The fixed text comes first and the presenter's notes last, so the KV cache
    keeps reusing the long shared prefix when only the notes change.
    """
    if direction == "id-en":
        prompt = ID_EN_PROMPT  # English output: the Indonesian style doesn't apply
    else:
        style = style if style in _REGISTER else "match"
        prompt = _EN_ID_HEAD + _REGISTER[style] + _EN_ID_TAIL + _EN_ID_EXAMPLES[style]
    notes = (notes or "").strip()[:MAX_NOTES]
    if notes:
        prompt += (
            "\n\nContext from the presenter. Follow it; where it conflicts with the"
            " examples above, it wins:\n" + notes
        )
    return prompt


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class MtResult:
    text: str
    elapsed: float


def _clean(text: str) -> str:
    text = _THINK.sub("", text).strip()
    # Models occasionally wrap the answer in quotes or prefix it with a label.
    text = re.sub(
        r"^(english|indonesian|bahasa indonesia|translation|terjemahan)\s*[:\-]\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


class Translator:
    def __init__(
        self,
        repo: str,
        max_tokens: int,
        context_turns: int,
        *,
        pool: ThreadPoolExecutor | None = None,
        gate: GpuGate | None = None,
    ) -> None:
        self.repo = repo
        self.max_tokens = max_tokens
        self.context_turns = context_turns
        self._model = None
        self._tokenizer = None
        # One KV cache per direction, reused across calls so each fixed system
        # prompt is processed once instead of on every sentence (~430ms/call on
        # an M4). Separate caches mean toggling direction doesn't evict the other.
        self._caches: dict[str, object] = {}
        self._cached: dict[str, list[int]] = {}
        self.last_prompt_tokens = 0
        self.last_cached_tokens = 0
        # Translators share one pool, and every model shares the GPU gate.
        # Called from the worker thread with a short label for each slow step of
        # the first load. main.py points this at the state the page polls;
        # unset it is a no-op, so the class stays usable on its own.
        self.on_stage: Callable[[str], None] | None = None
        self._owns_pool = pool is None
        self._pool = pool or ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt")
        self.gate = gate or GpuGate()

    # -- blocking side ------------------------------------------------------

    def _stage(self, label: str) -> None:
        if self.on_stage is not None:
            self.on_stage(label)

    def _load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        log.info("loading translation model %s...", self.repo)
        self._stage("Downloading and preparing model…")
        started = time.perf_counter()
        self._model, self._tokenizer = load(self.repo)
        log.info("translator ready in %.1fs", time.perf_counter() - started)
        # Weights are in memory, but the caller still has a warmup sentence to
        # generate -- minutes on a CPU backend. Stop saying "downloading".
        self._stage("Warming up…")

    def _reset_cache(self, direction: str) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self._caches[direction] = make_prompt_cache(self._model)
        self._cached[direction] = []

    def _generate(self, messages: list[dict], direction: str) -> str:
        from mlx_lm.generate import stream_generate
        from mlx_lm.models.cache import trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        ids = list(
            self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        )
        if direction not in self._caches:
            self._reset_cache(direction)
        cache = self._caches[direction]
        cached = self._cached[direction]

        # Reuse however much of the cached prefix still matches -- in practice the
        # system prompt, and the conversation history when it hasn't rolled over.
        shared = 0
        for a, b in zip(cached, ids):
            if a != b:
                break
            shared += 1
        # Always leave at least one token for the model to actually run on.
        shared = min(shared, len(ids) - 1)

        stale = len(cached) - shared
        if stale > 0 and trim_prompt_cache(cache, stale) != stale:
            # This cache type can't be trimmed exactly; start over rather than
            # generate from a cache that no longer matches the prompt.
            self._reset_cache(direction)
            cache = self._caches[direction]
            shared = 0

        self.last_prompt_tokens = len(ids)
        self.last_cached_tokens = shared

        pieces: list[str] = []
        produced: list[int] = []
        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt=ids[shared:],
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.0),
            prompt_cache=cache,
        ):
            pieces.append(resp.text)
            produced.append(resp.token)

        self._cached[direction] = ids + produced
        return "".join(pieces)

    def _translate_blocking(
        self,
        text: str,
        history: list[tuple[str, str]],
        direction: str,
        style: str,
        notes: str,
    ) -> MtResult:
        self._load()
        started = time.perf_counter()

        messages: list[dict] = [
            {"role": "system", "content": build_prompt(direction, style, notes)}
        ]
        for src, dst in history[-self.context_turns :]:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": dst})
        messages.append({"role": "user", "content": text})

        raw = self._generate(messages, direction)
        log.debug(
            "mt prompt=%d tokens, %d reused from cache",
            self.last_prompt_tokens, self.last_cached_tokens,
        )
        return MtResult(text=_clean(raw), elapsed=time.perf_counter() - started)

    # -- async API ----------------------------------------------------------

    async def translate(
        self,
        text: str,
        history: list[tuple[str, str]] | None = None,
        direction: str = "id-en",
        style: str = "match",
        notes: str = "",
    ) -> MtResult:
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}")
        loop = asyncio.get_running_loop()
        async with self.gate.hold(URGENT):
            return await loop.run_in_executor(
                self._pool,
                lambda: self._translate_blocking(
                    text, history or [], direction, style, notes
                ),
            )

    async def warmup(
        self, directions: tuple[str, ...] = DIRECTIONS, style: str = "match"
    ) -> None:
        # Warm each direction so its system prompt is already in the cache.
        samples = {"id-en": "Halo, apa kabar?", "en-id": "Hello, how are you?"}
        for direction in directions:
            result = await self.translate(samples[direction], direction=direction, style=style)
            log.info(
                "translator warmup %s: %r (%.2fs)", direction, result.text, result.elapsed
            )

    def shutdown(self) -> None:
        if self._owns_pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
