"""Speech-to-text through the OpenAI audio API, as a drop-in for WhisperEngine.

Same ``run()`` contract as the local engine, so the session doesn't care which
one it has. Differences worth knowing:

- It never touches the GPU gate: the work happens on OpenAI's side, so an API
  call can overlap a local translation freely.
- Live partials are off by default (OPENAI_PARTIALS). The local engine re-decodes
  about once a second while you talk; against the API that is a paid request
  per second for text that is thrown away moments later.
- ``whisper-1`` returns ``verbose_json`` with per-segment ``no_speech_prob`` /
  ``avg_logprob``, which feeds the hallucination filter. The ``gpt-*-transcribe``
  models only return plain ``json``, so only the phrase blocklists apply there.
- ``/audio/translations`` (MODE=direct) exists for ``whisper-1`` only.

Written against the parameter definitions in the official openai-python SDK;
it uses httpx directly to avoid another dependency.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave

import httpx
import numpy as np

from .asr import AsrError, AsrResult, _clean, _looks_hallucinated

log = logging.getLogger(__name__)

MODELS = ("whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-transcribe")
_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def _wav_bytes(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _api_message(resp: httpx.Response) -> str:
    try:
        return resp.json()["error"]["message"]
    except Exception:
        return resp.text[:200] or resp.reason_phrase


class OpenAIWhisperEngine:
    engine = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        task: str,
        partials: bool,
        transport: httpx.AsyncBaseTransport | None = None,  # tests inject a fake
    ) -> None:
        if task == "translate" and model != "whisper-1":
            raise AsrError("OpenAI only offers direct translation with whisper-1")
        self.model = model
        self.repo = f"openai/{model}"  # for logs and the UI
        self.task = task
        self.partials = partials
        # Auto direction needs the API to detect the spoken language itself.
        self.auto_detect = task == "transcribe"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=transport,
        )

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        delay = 0.5
        for attempt in range(3):
            try:
                resp = await self._client.request(method, path, **kwargs)
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

    async def run(
        self,
        audio: np.ndarray,
        *,
        language: str | None = "id",
        initial_prompt: str | None = None,
        final: bool = False,
    ) -> AsrResult:
        verbose = self.model == "whisper-1"
        data = {
            "model": self.model,
            "response_format": "verbose_json" if verbose else "json",
            "temperature": "0",
        }
        if self.task == "transcribe":
            if language:  # None = let the API detect it (auto direction)
                data["language"] = language
            path = "/audio/transcriptions"
        else:
            path = "/audio/translations"
        if initial_prompt:
            data["prompt"] = initial_prompt
        files = {"file": ("speech.wav", _wav_bytes(audio), "audio/wav")}

        started = time.perf_counter()
        resp = await self._request("POST", path, data=data, files=files)
        elapsed = time.perf_counter() - started
        body = resp.json()

        segments = body.get("segments") or []
        no_speech = max((s.get("no_speech_prob", 0.0) for s in segments), default=0.0)
        logprob = min((s.get("avg_logprob", 0.0) for s in segments), default=0.0)
        result = AsrResult(
            text=_clean(body.get("text", "")),
            # whisper-1 reports e.g. "indonesian"; gpt-*-transcribe reports nothing.
            language=body.get("language") or language,
            no_speech_prob=float(no_speech),
            avg_logprob=float(logprob),
            elapsed=elapsed,
        )
        if _looks_hallucinated(result.text, result.no_speech_prob, result.avg_logprob):
            log.debug("dropped likely hallucination: %r", result.text)
            result.text = ""
        return result

    async def warmup(self) -> None:
        """Validate the key and model up front instead of on the first sentence."""
        started = time.perf_counter()
        await self._request("GET", f"/models/{self.model}")
        log.info("OpenAI %s reachable (%.1fs)", self.model, time.perf_counter() - started)

    async def aclose(self) -> None:
        await self._client.aclose()

    def shutdown(self) -> None:
        pass  # the HTTP client is closed via aclose()
