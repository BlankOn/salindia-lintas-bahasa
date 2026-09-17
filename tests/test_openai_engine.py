"""OpenAI engine against a fake HTTP transport: no key, no network, no cost.

Checks what we send (endpoint, form fields, WAV payload, auth header) and how
responses and failures come back, per the openai-python SDK's parameter docs.
"""

import asyncio
import io
import pathlib
import sys
import wave

import httpx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from server.asr import AsrError
from server.asr_openai import OpenAIWhisperEngine

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")


def parse_multipart(req: httpx.Request) -> tuple[dict, bytes]:
    """Tiny multipart reader, enough for our own requests."""
    boundary = req.headers["content-type"].split("boundary=")[1].encode()
    fields, file = {}, b""
    for part in req.content.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, body = part.split(b"\r\n\r\n", 1)
        body = body.rsplit(b"\r\n", 1)[0]
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        if b"filename=" in head:
            file = body
        else:
            fields[name] = body.decode()
    return fields, file


def engine(handler, *, model="whisper-1", task="transcribe"):
    return OpenAIWhisperEngine(
        api_key="sk-test", model=model, base_url="https://api.example/v1",
        task=task, partials=False, transport=httpx.MockTransport(handler),
    )


audio = (np.sin(np.linspace(0, 400, 16000)) * 0.3).astype(np.float32)


async def main():
    print("case: whisper-1 transcription request")
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        seen["fields"], seen["file"] = parse_multipart(req)
        return httpx.Response(200, json={
            "text": " Selamat pagi semuanya. ", "language": "indonesian",
            "segments": [{"no_speech_prob": 0.01, "avg_logprob": -0.2}],
        })

    eng = engine(handler)
    r = await eng.run(audio, language="id", final=True)
    check("posts to /audio/transcriptions", seen["url"] == "https://api.example/v1/audio/transcriptions", seen["url"])
    check("bearer auth", seen["auth"] == "Bearer sk-test")
    check("model / language / format / temperature",
          seen["fields"] == {"model": "whisper-1", "language": "id",
                             "response_format": "verbose_json", "temperature": "0"},
          str(seen["fields"]))
    with wave.open(io.BytesIO(seen["file"])) as w:
        check("file is 16 kHz mono 16-bit WAV of the right length",
              (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 16000))
    check("text cleaned", r.text == "Selamat pagi semuanya.", repr(r.text))
    check("partials disabled for the API", eng.partials is False)
    await eng.aclose()

    print("\ncase: gpt-4o-transcribe uses plain json, English language pinned")
    def handler_4o(req):
        seen["fields"], _ = parse_multipart(req)
        return httpx.Response(200, json={"text": "Have you read the contract?"})
    eng = engine(handler_4o, model="gpt-4o-transcribe")
    r = await eng.run(audio, language="en", final=True)
    check("json format + en", seen["fields"]["response_format"] == "json"
          and seen["fields"]["language"] == "en", str(seen["fields"]))
    check("text returned", r.text == "Have you read the contract?")
    await eng.aclose()

    print("\ncase: auto direction leaves language detection to the API")
    def handler_auto(req):
        seen["fields"], _ = parse_multipart(req)
        return httpx.Response(200, json={"text": "Hello there.", "language": "english",
                                         "segments": [{"no_speech_prob": 0.0, "avg_logprob": -0.1}]})
    eng = engine(handler_auto)
    r = await eng.run(audio, language=None, final=True)
    check("no language field sent", "language" not in seen["fields"], str(seen["fields"]))
    check("detected language returned", r.language == "english", repr(r.language))
    check("engine advertises auto-detect", eng.auto_detect is True)
    await eng.aclose()
    tr = engine(handler_auto, task="translate")
    check("direct-translate engine does not", tr.auto_detect is False)
    await tr.aclose()

    print("\ncase: hallucination filter still applies")
    def handler_ghost(req):
        return httpx.Response(200, json={
            "text": "Terima kasih.",
            "segments": [{"no_speech_prob": 0.8, "avg_logprob": -1.4}],
        })
    eng = engine(handler_ghost)
    r = await eng.run(audio, final=True)
    check("'Terima kasih' over silence dropped", r.text == "", repr(r.text))
    await eng.aclose()

    print("\ncase: direct mode uses /audio/translations without language")
    def handler_tr(req):
        seen["url"] = str(req.url)
        seen["fields"], _ = parse_multipart(req)
        return httpx.Response(200, json={"text": "Good morning, everyone."})
    eng = engine(handler_tr, task="translate")
    r = await eng.run(audio, language="id", final=True)
    check("translations endpoint", seen["url"].endswith("/audio/translations"), seen["url"])
    check("no language field", "language" not in seen["fields"], str(seen["fields"]))
    await eng.aclose()
    try:
        engine(handler_tr, model="gpt-4o-transcribe", task="translate")
        check("direct translate refused for non-whisper-1", False)
    except AsrError as exc:
        check("direct translate refused for non-whisper-1", True, str(exc))

    print("\ncase: errors")
    def unauthorized(req):
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})
    eng = engine(unauthorized)
    try:
        await eng.run(audio)
        check("401 raises", False)
    except AsrError as exc:
        check("401 -> readable key error", "rejected the API key" in str(exc), str(exc))
    await eng.aclose()

    calls = {"n": 0}
    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"text": "Halo."})
    eng = engine(flaky)
    r = await eng.run(audio)
    check("429 retried then succeeds", r.text == "Halo." and calls["n"] == 2, f"{calls['n']} calls")
    await eng.aclose()

    def down(req):
        raise httpx.ConnectError("no route", request=req)
    eng = engine(down)
    try:
        await eng.run(audio)
        check("network failure raises", False)
    except AsrError as exc:
        check("network failure -> readable error", "cannot reach OpenAI" in str(exc), str(exc))
    await eng.aclose()

    print("\ncase: warmup validates the model")
    def models(req):
        seen["url"] = str(req.url)
        if req.url.path.endswith("/models/whisper-1"):
            return httpx.Response(200, json={"id": "whisper-1"})
        return httpx.Response(404, json={"error": {"message": "The model does not exist"}})
    eng = engine(models)
    await eng.warmup()
    check("GET /models/whisper-1", seen["url"].endswith("/models/whisper-1"))
    await eng.aclose()
    eng = engine(models, model="gpt-transcribe")
    try:
        await eng.warmup()
        check("unknown model raises", False)
    except AsrError as exc:
        check("unknown model -> readable error", "not found" in str(exc), str(exc))
    await eng.aclose()

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
