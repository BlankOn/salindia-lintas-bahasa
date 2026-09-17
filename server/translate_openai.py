"""Text translation through the OpenAI chat API, as a drop-in for Translator.

Same ``translate()`` contract as the local translator, so the session doesn't
care which one it has. Picked automatically when the speech engine is OpenAI,
which makes the whole pipeline runnable on a machine with no GPU and no MLX --
the local translator needs Apple Silicon to be any use.

Differences from the MLX translator worth knowing:

- No GPU gate and no worker pool: the work happens on OpenAI's side, so calls
  overlap freely and the single-thread serialisation the local models need
  doesn't apply.
- No KV cache. The local translator keeps one per direction so the fixed system
  prompt is processed once; here the prompt is re-sent every call and the cost
  shows up as tokens rather than latency. Prompt caching on OpenAI's side makes
  the repeated prefix cheaper without anything to manage here.
- The prompts are shared with the local path (``build_prompt``), so both engines
  translate to the same rules.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import httpx

from .asr import AsrError
from .translate import DIRECTIONS, MtResult, _clean, build_prompt
from .usage import record_tokens

log = logging.getLogger(__name__)

# Chat models that make sense here: cheap and fast matters more than depth,
# since each call is one spoken sentence.
MODELS = ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1")
_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}

_WARMUP_SAMPLES = {"id-en": "Halo, apa kabar?", "en-id": "Hello, how are you?"}


def _api_message(resp: httpx.Response) -> str:
    try:
        return resp.json()["error"]["message"]
    except Exception:
        return resp.text[:200] or resp.reason_phrase


class OpenAITranslator:
    engine = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        max_tokens: int,
        context_turns: int,
        transport: httpx.AsyncBaseTransport | None = None,  # tests inject a fake
    ) -> None:
        self.model = model
        self.repo = f"openai/{model}"  # for logs and the UI
        self.max_tokens = max_tokens
        self.context_turns = context_turns
        # Present so main.py can drive the loading UI the same way for both
        # translators; there is no slow local load to report, so it stays quiet.
        self.on_stage: Callable[[str], None] | None = None
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=transport,
        )

    async def _request(self, path: str, payload: dict) -> httpx.Response:
        delay = 0.5
        for attempt in range(3):
            try:
                resp = await self._client.post(path, json=payload)
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise AsrError(f"cannot reach OpenAI ({type(exc).__name__})") from exc
            else:
                if resp.status_code < 400:
                    return resp
                if resp.status_code not in _RETRYABLE or attempt == 2:
                    raise AsrError(self._explain(resp))
                log.warning("OpenAI %s, retrying: %s", resp.status_code, _api_message(resp))
            await asyncio.sleep(delay)
            delay *= 3
        raise AsrError("OpenAI request failed")  # unreachable

    def _explain(self, resp: httpx.Response) -> str:
        detail = _api_message(resp)
        if resp.status_code == 401:
            return f"OpenAI rejected the API key ({detail})"
        if resp.status_code == 403:
            return f"this API key can't use {self.model} ({detail})"
        if resp.status_code == 404:
            return f"model {self.model} not found ({detail})"
        if resp.status_code == 429:
            return f"OpenAI rate limit or quota reached ({detail})"
        return f"OpenAI error {resp.status_code}: {detail}"

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
        started = time.perf_counter()

        messages: list[dict] = [
            {"role": "system", "content": build_prompt(direction, style, notes)}
        ]
        # Same shape as the local path: prior sentence pairs as turns, so the
        # model keeps pronouns and terminology consistent across a talk.
        for src, dst in (history or [])[-self.context_turns :]:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": dst})
        messages.append({"role": "user", "content": text})

        resp = await self._request(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                # Subtitles want the obvious reading, not a creative one.
                "temperature": 0.0,
            },
        )
        body = resp.json()
        u = body.get("usage") or {}
        record_tokens(
            self.model,
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        )
        try:
            raw = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise AsrError(f"unexpected OpenAI response ({type(exc).__name__})") from exc

        return MtResult(text=_clean(raw), elapsed=time.perf_counter() - started)

    async def warmup(
        self, directions: tuple[str, ...] = DIRECTIONS, style: str = "match"
    ) -> None:
        """One real call per direction: proves the key and model work.

        Cheap (a few tokens) and worth it -- a bad model name should surface in
        the engine dialog, not on the presenter's first sentence.
        """
        for direction in directions:
            result = await self.translate(
                _WARMUP_SAMPLES[direction], direction=direction, style=style
            )
            log.info(
                "translator warmup %s: %r (%.2fs)", direction, result.text, result.elapsed
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    def shutdown(self) -> None:
        # Mirrors Translator.shutdown(); the client is closed on the async path.
        pass
