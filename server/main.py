"""FastAPI app: static UI + a WebSocket that interprets live speech, ID <-> EN."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import logging
import pathlib

from urllib.parse import unquote

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .asr import AsrError, WhisperEngine
from .asr_openai import MODELS as OPENAI_MODELS
from .asr_openai import OpenAIWhisperEngine
from .config import settings
from .gpu import GpuGate
from .session import StreamSession
from .slides import SlideError, SlideStore
from .translate import Translator
from .translate_openai import OpenAITranslator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("salindia")

WEB_DIR = pathlib.Path(__file__).resolve().parent.parent / "web"


class State:
    # "choose" -> waiting for the user to pick a speech engine in the browser
    # "loading" -> engine picked, models warming up
    # "ready" / "failed"
    phase = "choose"
    engine: str | None = None
    asr: WhisperEngine | OpenAIWhisperEngine | None = None
    gate: GpuGate | None = None
    translators: dict[str, Translator] = {}
    mt_pool: ThreadPoolExecutor | None = None
    error: str | None = None
    # What the "loading" phase is doing right now, in words the page can show.
    # A first run spends minutes downloading weights; without this the UI has
    # nothing to say between "loading" and "ready".
    detail: str = ""
    warm_task: asyncio.Task | None = None

    @property
    def ready(self) -> bool:
        return self.phase == "ready"

    def directions(self) -> tuple[str, ...]:
        dirs = settings.directions
        if settings.mode == "pipeline" and getattr(self.asr, "auto_detect", False):
            dirs = (*dirs, "auto")
        return dirs

    def engine_info(self) -> dict:
        return {
            "directions": list(self.directions()),
            "phase": self.phase,
            "detail": self.detail,
            "engine": self.engine,
            "asr_model": getattr(self.asr, "repo", None),
            "error": self.error,
            "openai_key_configured": bool(settings.openai_api_key),
            "openai_models": list(OPENAI_MODELS),
            "openai_default_model": settings.openai_asr_model,
            "local_asr_model": settings.effective_asr_model,
            # What is actually loaded, which is not settings.mt_models: the
            # OpenAI engine never loads the configured MLX repos.
            "mt_models": {d: t.repo for d, t in self.translators.items()},
        }


def _set_detail(label: str) -> None:
    """Stage callback for the models; called from their worker threads."""
    state.detail = label


state = State()
slides = SlideStore()


async def _warmup() -> None:
    try:
        state.detail = (
            "Checking your OpenAI key…"
            if state.engine == "openai"
            else "Downloading and preparing model…"
        )
        await state.asr.warmup()
    except AsrError as exc:
        # Bad key, unknown model, no network: let the user fix it and retry.
        log.warning("speech engine check failed: %s", exc)
        await state.asr.aclose()
        state.asr, state.engine = None, None
        state.error = str(exc)
        state.detail = ""
        state.phase = "choose"
        return
    except Exception as exc:
        log.exception("ASR warmup failed")
        state.error = f"{type(exc).__name__}: {exc}"
        state.detail = ""
        state.phase = "failed"
        return

    try:
        # Only the direction a session starts in is warmed. The other direction
        # can be a different multi-GB repo (4B for id-en, 8B for en-id) that the
        # user may never switch to, so it downloads and loads on first use
        # instead -- one slower sentence beats a download nobody asked for.
        start = settings.default_direction
        translator = state.translators.get(start)
        if translator is not None:
            # The translator reports its own steps: the download/load, then the
            # warmup sentence. Both are slow enough to need separate labels.
            translator.on_stage = _set_detail
            state.detail = (
                "Checking the translation model…"
                if getattr(translator, "engine", "") == "openai"
                else "Downloading and preparing model…"
            )
            try:
                await translator.warmup((start,), style=settings.id_style)
            finally:
                translator.on_stage = None
        state.detail = ""
        state.phase = "ready"
        log.info(
            "ready (speech engine: %s, translator: %s)",
            state.asr.repo,
            translator.repo if translator is not None else "none",
        )
    except Exception as exc:  # surfaced to the UI rather than killing the server
        log.exception("model warmup failed")
        state.error = f"{type(exc).__name__}: {exc}"
        state.detail = ""
        state.phase = "failed"


def _build_translators(engine: str, api_key: str = "") -> None:
    """Translators for the chosen engine. MODE=direct needs none.

    With the OpenAI engine the translation also goes to the API, so nothing
    local is loaded at all -- the point of the pairing, since the MLX
    translator is only usable on Apple Silicon.
    """
    for old in {id(t): t for t in state.translators.values()}.values():
        if (aclose := getattr(old, "aclose", None)) is not None:
            asyncio.create_task(aclose())
    state.translators = {}
    if settings.mode != "pipeline":
        return

    if engine == "openai":
        shared = OpenAITranslator(
            api_key=api_key or settings.openai_api_key,
            model=settings.openai_mt_model,
            base_url=settings.openai_base_url,
            max_tokens=settings.mt_max_tokens,
            context_turns=settings.mt_context_turns,
        )
        state.translators = dict.fromkeys(settings.directions, shared)
    else:
        if state.mt_pool is None:
            state.mt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt")
        by_repo: dict[str, Translator] = {}
        for direction in settings.directions:
            repo = settings.mt_model_for(direction)
            if repo not in by_repo:
                by_repo[repo] = Translator(
                    repo,
                    settings.mt_max_tokens,
                    settings.mt_context_turns,
                    pool=state.mt_pool,
                    gate=state.gate,
                )
            state.translators[direction] = by_repo[repo]

    for direction, t in state.translators.items():
        log.info("translator %s: %s", direction, t.repo)


def start_engine(engine: str, api_key: str = "", model: str = "") -> None:
    """Build the chosen speech engine and warm everything up in the background."""
    task = "transcribe" if settings.mode == "pipeline" else "translate"
    if engine == "local":
        state.asr = WhisperEngine(settings.effective_asr_model, task, state.gate)
    else:
        key = api_key or settings.openai_api_key
        if not key:
            raise AsrError("an OpenAI API key is required")
        state.asr = OpenAIWhisperEngine(
            api_key=key,
            model=model or settings.openai_asr_model,
            base_url=settings.openai_base_url,
            task=task,
            partials=settings.openai_partials,
        )
    _build_translators(engine, api_key)
    state.engine = engine
    state.error = None
    state.detail = "Starting the speech engine…"
    state.phase = "loading"
    log.info("speech engine: %s (%s)", engine, state.asr.repo)
    state.warm_task = asyncio.create_task(_warmup())


async def lifespan(app: FastAPI):
    state.gate = GpuGate()  # shared by every local model in the process
    log.info("mode=%s", settings.mode)
    if settings.asr_engine:
        try:
            start_engine(settings.asr_engine)
        except AsrError as exc:
            state.error = str(exc)
            log.warning("ASR_ENGINE=%s: %s -- asking in the browser", settings.asr_engine, exc)
    else:
        log.info("waiting for a speech engine choice in the browser")
    try:
        yield
    finally:
        if state.warm_task:
            state.warm_task.cancel()
        if state.asr:
            state.asr.shutdown()
        for t in {id(t): t for t in state.translators.values()}.values():
            if (aclose := getattr(t, "aclose", None)) is not None:
                await aclose()
            await state.asr.aclose()
        if state.mt_pool:
            state.mt_pool.shutdown(wait=False, cancel_futures=True)
        slides.shutdown()


app = FastAPI(title="Salindia", lifespan=lifespan)


@app.middleware("http")
async def no_stale_ui(request, call_next):
    # The UI changes often and has no build step to fingerprint file names;
    # without this, browsers heuristically reuse an old app.js with a new page.
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "ready": state.ready,
            "error": state.error,
            "engine": state.engine_info(),
            "config": settings.public(),
        }
    )


@app.get("/api/engine")
async def get_engine() -> JSONResponse:
    return JSONResponse(state.engine_info())


@app.post("/api/engine")
async def choose_engine(request: Request) -> JSONResponse:
    body = await request.json()
    engine = body.get("engine")
    if engine not in ("local", "openai"):
        raise HTTPException(status_code=400, detail="engine must be 'local' or 'openai'")
    if state.phase in ("loading", "ready"):
        # Models are process-wide; switching means restarting the server.
        raise HTTPException(
            status_code=409,
            detail=f"speech engine already set to {state.engine}; restart the server to change it",
        )
    model = body.get("model") or ""
    if engine == "openai" and model and model not in OPENAI_MODELS:
        raise HTTPException(status_code=400, detail=f"unknown OpenAI model {model!r}")
    try:
        # The key only lives in this process's memory; it is never logged or stored.
        start_engine(engine, api_key=(body.get("api_key") or "").strip(), model=model)
    except AsrError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(state.engine_info())


@app.post("/api/slides")
async def upload_slides(request: Request) -> JSONResponse:
    # Raw PDF body (no multipart), so no extra form-parsing dependency.
    name = unquote(request.headers.get("x-filename", "slides.pdf"))[:200]
    data = await request.body()
    try:
        deck = await slides.add(data, name)
    except SlideError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(deck.meta())


@app.get("/api/slides/{deck_id}")
async def slides_meta(deck_id: str) -> JSONResponse:
    deck = slides.get(deck_id)
    if deck is None:
        raise HTTPException(status_code=404, detail="deck not loaded (server restarted?)")
    return JSONResponse(deck.meta())


@app.get("/api/slides/{deck_id}/{number}.jpg")
async def slide_image(deck_id: str, number: int, w: int = 1920) -> Response:
    try:
        data = await slides.page(deck_id, number, w)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="deck not loaded") from exc
    except SlideError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Deck ids are unique per upload, so a rendered page never changes.
    return Response(
        data,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400, immutable"},
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            try:
                await ws.send_text(json.dumps(payload))
            except (WebSocketDisconnect, RuntimeError):
                pass

    await send(
        {
            "type": "hello",
            "ready": state.ready,
            "config": settings.public(),
            "engine": state.engine_info(),
        }
    )

    # Hold the client at the door until a speech engine is chosen and every
    # model is loaded; otherwise the first utterance would block behind a
    # multi-gigabyte download. Tell the page each time the situation changes.
    last = None
    while not state.ready:
        now = (state.phase, state.error, state.detail)
        if now != last:
            last = now
            info = state.engine_info()
            if state.phase == "failed":
                await send({"type": "error", "message": f"model load failed: {state.error}"})
                await ws.close(code=1011)
                return
            await send(
                {"type": "choose" if state.phase == "choose" else "loading", "engine": info}
            )
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if msg["type"] == "websocket.disconnect":
            return

    await send(
        {"type": "ready", "config": settings.public(), "engine": state.engine_info()}
    )

    session = StreamSession(
        send, state.asr, state.translators, settings, directions=state.directions()
    )
    await session.start()

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                await session.feed(pcm)
            elif (text := message.get("text")) is not None:
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "stop":
                    await session.flush()
                elif msg.get("type") == "context":
                    ctx = session.set_context(msg.get("style"), msg.get("notes"))
                    await send({"type": "context", **ctx})
                elif msg.get("type") == "direction":
                    value = msg.get("value")
                    if session.set_direction(value):
                        await send({"type": "direction", "value": value})
                    else:
                        await send(
                            {
                                "type": "error",
                                "message": f"direction {value!r} is not available "
                                f"with the {state.engine} engine in {settings.mode} mode",
                            }
                        )
                elif msg.get("type") == "ping":
                    await send({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket loop failed")
    finally:
        await session.close()


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:  # pragma: no cover
    log.warning("web/ directory missing at %s -- UI will not be served", WEB_DIR)
