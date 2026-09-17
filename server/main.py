"""FastAPI app: static UI + a WebSocket that interprets live speech, ID <-> EN."""

from __future__ import annotations

import asyncio
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
import pathlib


import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .asr import AsrError, WhisperEngine
from .asr_openai import MODELS as OPENAI_MODELS
from .asr_openai import OpenAIWhisperEngine
from .config import MODE_IS_EXPLICIT, MODES, Settings, settings
from .gpu import GpuGate
from .session import StreamSession
from .translate import Translator
from .translate_openai import OpenAITranslator
from .db import Db
from .usage import CURRENT, Usage

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
    # MODE= only says where we start; the bar can switch it while we run, which
    # reloads the speech model and rebuilds the translators.
    mode: str = settings.mode
    # Remembered so a mode switch can rebuild the same engine without asking
    # again. Memory only, like the key the browser sent -- never logged or stored.
    api_key: str = ""
    asr_choice: str = ""
    asr: WhisperEngine | OpenAIWhisperEngine | None = None
    gate: GpuGate | None = None
    translators: dict[str, Translator] = {}
    mt_pool: ThreadPoolExecutor | None = None
    error: str | None = None
    db: Db | None = None
    # Open browser sockets, so a mode switch can send them round again.
    sockets: set[WebSocket] = set()
    # What the "loading" phase is doing right now, in words the page can show.
    # A first run spends minutes downloading weights; without this the UI has
    # nothing to say between "loading" and "ready".
    detail: str = ""
    warm_task: asyncio.Task | None = None

    @property
    def ready(self) -> bool:
        return self.phase == "ready"

    def directions(self) -> tuple[str, ...]:
        dirs = settings.directions_for(self.mode)
        if self.mode == "pipeline" and getattr(self.asr, "auto_detect", False):
            dirs = (*dirs, "auto")
        return dirs

    def config(self) -> "Settings":
        """Settings as they stand, with the live mode folded in."""
        return replace(
            settings,
            mode=self.mode,
            default_direction=settings.default_direction_for(self.mode),
        )

    def mode_options(self) -> dict[str, str]:
        """mode -> "" if it can be switched to right now, else why not."""
        out = {}
        for mode in MODES:
            reason = ""
            if mode == "direct" and self.engine == "openai":
                model = self.asr_choice or settings.openai_asr_model
                if model != "whisper-1":
                    reason = f"OpenAI only translates directly with whisper-1, not {model}"
            out[mode] = reason
        return out

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
            "local_asr_model": settings.asr_model_for(self.mode),
            # What is actually loaded, which is not settings.mt_models: the
            # OpenAI engine never loads the configured MLX repos.
            "mt_models": {d: t.repo for d, t in self.translators.items()},
            "mode": self.mode,
            "mode_options": self.mode_options(),
            # So the chooser can say which mode a given engine would land in.
            "mode_is_explicit": MODE_IS_EXPLICIT,
            "env_mode": settings.mode,
        }


def _set_detail(label: str) -> None:
    """Stage callback for the models; called from their worker threads."""
    state.detail = label


state = State()


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
        start = settings.default_direction_for(state.mode)
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
    if state.mode != "pipeline":
        return

    directions = settings.directions_for(state.mode)

    if engine == "openai":
        shared = OpenAITranslator(
            api_key=api_key or settings.openai_api_key,
            model=settings.openai_mt_model,
            base_url=settings.openai_base_url,
            max_tokens=settings.mt_max_tokens,
            context_turns=settings.mt_context_turns,
        )
        state.translators = dict.fromkeys(directions, shared)
    else:
        if state.mt_pool is None:
            state.mt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt")
        by_repo: dict[str, Translator] = {}
        for direction in directions:
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


def preferred_mode(engine: str, model: str = "") -> str:
    """Where a freshly chosen engine should start.

    The OpenAI engine goes direct when it can: ``/audio/translations`` does
    speech and translation in the one Whisper pass, which is both faster and
    cheaper than transcribing and then paying a chat model to translate. It only
    ever outputs English, so this costs en-id -- switchable in the bar. An
    explicit MODE= in the environment always wins.
    """
    if MODE_IS_EXPLICIT or engine != "openai":
        return settings.mode
    # /audio/translations exists for whisper-1 alone; the gpt-*-transcribe
    # models can only transcribe, so they stay on the pipeline.
    return "direct" if (model or settings.openai_asr_model) == "whisper-1" else "pipeline"


async def _drop_engine() -> None:
    """Let go of the loaded speech engine so another can take its place."""
    asr, state.asr = state.asr, None
    if asr is None:
        return
    asr.shutdown()
    try:
        await asr.aclose()
    except Exception:  # a failing close must not block the new engine
        log.exception("closing the old speech engine failed")


def start_engine(
    engine: str, api_key: str = "", model: str = "", mode: str | None = None
) -> None:
    """Build the chosen speech engine and warm everything up in the background."""
    if mode is not None:
        state.mode = mode
    task = "transcribe" if state.mode == "pipeline" else "translate"
    if engine == "local":
        state.asr = WhisperEngine(settings.asr_model_for(state.mode), task, state.gate)
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
            speculates=settings.openai_speculate,
        )
    _build_translators(engine, api_key)
    state.engine = engine
    state.api_key = api_key
    state.asr_choice = model
    state.error = None
    state.detail = "Starting the speech engine…"
    state.phase = "loading"
    log.info("speech engine: %s (%s), mode=%s", engine, state.asr.repo, state.mode)
    state.warm_task = asyncio.create_task(_warmup())


async def lifespan(app: FastAPI):
    state.gate = GpuGate()  # shared by every local model in the process
    state.db = Db(settings.db_path)
    log.info("mode=%s", state.mode)
    if settings.force_approach:
        try:
            start_engine(
                settings.force_approach, mode=preferred_mode(settings.force_approach)
            )
            log.info("FORCE_APPROACH=%s: skipping the chooser", settings.force_approach)
        except AsrError as exc:
            state.error = str(exc)
            log.warning(
                "FORCE_APPROACH=%s: %s -- asking in the browser",
                settings.force_approach, exc,
            )
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
        if state.db:
            state.db.close()


app = FastAPI(title="Salindia", lifespan=lifespan)


@app.middleware("http")
async def no_stale_ui(request, call_next):
    # The UI changes often and has no build step to fingerprint file names;
    # without this, browsers heuristically reuse an old app.js with a new page.
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


# ---------------------------------------------------------------- access gate
#
# A shared phrase, not per-user auth: it keeps the page off the open internet
# and nothing more. Tokens live in memory, so a restart signs everyone out.
_tokens: set[str] = set()
_failures: dict[str, int] = {}


def _client_ip(request_or_ws) -> str:
    fwd = request_or_ws.headers.get("x-forwarded-for", "").split(",")[0].strip()
    client = getattr(request_or_ws, "client", None)
    return fwd or (client.host if client else "?")


def _has_access(token: str | None) -> bool:
    if not settings.needs_passphrase:
        return True
    return bool(token) and token in _tokens


def require_access(request: Request) -> None:
    """FastAPI dependency: 401 unless the caller has a valid token."""
    if not _has_access(request.headers.get("x-salindia-access")):
        raise HTTPException(status_code=401, detail="passphrase required")


@app.get("/api/bootstrap")
async def bootstrap() -> JSONResponse:
    """Unauthenticated: only says whether a phrase is needed at all."""
    return JSONResponse({"needs_passphrase": settings.needs_passphrase})


@app.post("/api/access")
async def access(request: Request) -> JSONResponse:
    if not settings.needs_passphrase:
        return JSONResponse({"token": ""})
    ip = _client_ip(request)
    if _failures.get(ip, 0) >= 10:
        raise HTTPException(status_code=429, detail="too many attempts; restart the server")
    body = await request.json()
    given = str(body.get("passphrase") or "")
    # Constant-time, so a wrong phrase leaks nothing through timing.
    if not secrets.compare_digest(given, settings.access_passphrase):
        _failures[ip] = _failures.get(ip, 0) + 1
        await asyncio.sleep(0.5)  # blunt the brute-force rate
        log.warning("bad passphrase from %s (attempt %d)", ip, _failures[ip])
        raise HTTPException(status_code=401, detail="that passphrase is not right")
    _failures.pop(ip, None)
    token = secrets.token_urlsafe(32)
    _tokens.add(token)
    log.info("access granted to %s", ip)
    return JSONResponse({"token": token})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "ready": state.ready,
            "error": state.error,
            "engine": state.engine_info(),
            "config": settings.public(state.mode),
        }
    )


@app.get("/api/engine")
async def get_engine() -> JSONResponse:
    return JSONResponse(state.engine_info())


@app.post("/api/engine")
async def choose_engine(request: Request) -> JSONResponse:
    require_access(request)
    body = await request.json()
    engine = body.get("engine")
    if engine not in ("local", "openai"):
        raise HTTPException(status_code=400, detail="engine must be 'local' or 'openai'")
    if state.phase in ("loading", "ready"):
        if engine == state.engine:
            # A refreshed page picking what is already loaded: nothing to do.
            return JSONResponse(state.engine_info())
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
        start_engine(
            engine,
            api_key=(body.get("api_key") or "").strip(),
            model=model,
            mode=preferred_mode(engine, model),
        )
    except AsrError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(state.engine_info())




@app.post("/api/mode")
async def choose_mode(request: Request) -> JSONResponse:
    """Switch between pipeline and direct while the server runs.

    Both the speech model and the translators depend on it, so this reloads
    them: with the OpenAI engine that is a couple of API calls, but locally it
    means a different multi-GB Whisper repo. Open sockets are left to notice the
    reload and reconnect -- they resume the same talk.
    """
    require_access(request)
    body = await request.json()
    mode = body.get("mode")
    if mode not in MODES:
        raise HTTPException(
            status_code=400, detail=f"mode must be one of {', '.join(MODES)}"
        )
    if mode == state.mode and state.phase in ("loading", "ready"):
        return JSONResponse(state.engine_info())
    if (why := state.mode_options().get(mode)):
        raise HTTPException(status_code=409, detail=why)

    if state.engine is None:  # nothing loaded yet: just remember the choice
        state.mode = mode
        log.info("mode=%s (no engine chosen yet)", mode)
        return JSONResponse(state.engine_info())

    log.info("switching mode: %s -> %s", state.mode, mode)
    if state.warm_task:
        state.warm_task.cancel()
    # Shut the door before letting go of the models: a page reconnecting into
    # the gap would otherwise be waved through to an engine that is not there.
    state.phase, state.detail, state.error = "loading", "Switching mode…", None
    # The models these pages were talking to are going away. Send them round
    # again rather than leaving them on a closed engine; a reconnect resumes
    # the same talk, so mid-presentation this costs a sentence, not the talk.
    for sock in state.sockets.copy():
        state.sockets.discard(sock)  # its own handler may never get to
        try:
            await sock.close(code=4409)
        except Exception:
            pass
    await _drop_engine()
    try:
        start_engine(state.engine, api_key=state.api_key, model=state.asr_choice, mode=mode)
    except AsrError as exc:
        state.phase, state.error = "choose", str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(state.engine_info())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    token = ws.query_params.get("access") or ws.headers.get("x-salindia-access")
    if not _has_access(token):
        # Accept first, then close with an application code: a pre-accept close
        # is just a failed handshake, and the browser only ever sees 1006.
        await ws.accept()
        await ws.close(code=4401)
        return
    await ws.accept()
    state.sockets.add(ws)
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
            "config": settings.public(state.mode),
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
                state.sockets.discard(ws)
                return
            await send(
                {"type": "choose" if state.phase == "choose" else "loading", "engine": info}
            )
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if msg["type"] == "websocket.disconnect":
            state.sockets.discard(ws)
            return

    cfg = state.config()
    await send(
        {"type": "ready", "config": settings.public(cfg.mode), "engine": state.engine_info()}
    )

    # Every connection gets its own tally and its own row, so a refresh starts a
    # fresh talk rather than continuing the last one's numbers.
    usage = Usage() if state.engine == "openai" else None
    CURRENT.set(usage)

    # Wait for the presenter to name the talk. The page asks right after the
    # engine is settled; nothing is recorded, and no audio is accepted, until it
    # arrives -- so an abandoned tab never creates a row.
    talk_id = None
    while talk_id is None:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if msg["type"] == "websocket.disconnect":
            state.sockets.discard(ws)
            return
        text = msg.get("text")
        if not text:
            continue
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            continue
        if body.get("type") != "begin":
            continue
        # A page that dropped mid-talk sends the id it already had: continue
        # that row rather than splitting one talk's cost across two.
        if (prev := body.get("talk_id")) is not None:
            row = await state.db.resume_talk(int(prev))
            if row is not None:
                talk_id = int(prev)
                if usage is not None:
                    usage.seed(row)
                log.info("talk #%s resumed after a reconnect", talk_id)
                await send({"type": "begun", "talk_id": talk_id})
                continue
        client = ws.client
        talk_id = await state.db.start_talk(
            title=str(body.get("title") or "")[:200],
            # X-Forwarded-For only when something in front of us set it.
            ip=(ws.headers.get("x-forwarded-for", "").split(",")[0].strip()
                or (client.host if client else None)),
            user_agent=ws.headers.get("user-agent"),
            engine=state.engine,
            mode=cfg.mode,
            direction=cfg.default_direction,
            asr_model=getattr(state.asr, "repo", None),
            mt_model=next(
                (t.repo for t in state.translators.values()), None
            ),
        )
        log.info("talk #%s: %r", talk_id, body.get("title"))
        await send({"type": "begun", "talk_id": talk_id})

    session = StreamSession(
        send,
        state.asr,
        state.translators,
        cfg,
        directions=state.directions(),
        usage=usage,
        on_usage=(lambda snap: state.db.update_usage(talk_id, snap)),
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
                elif msg.get("type") == "resume":
                    # The page pressed Start again after an idle stop.
                    if session.resume():
                        await send({"type": "resumed"})
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
                                f"with the {state.engine} engine in {cfg.mode} mode",
                            }
                        )
                elif msg.get("type") == "rename":
                    # The presenter edited the title from the top bar; the row
                    # is already open, so only the label changes.
                    title = await state.db.rename_talk(talk_id, str(msg.get("title") or ""))
                    log.info("talk #%s retitled to %r", talk_id, title)
                    await send({"type": "renamed", "talk_id": talk_id, "title": title})
                elif msg.get("type") == "ping":
                    await send({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket loop failed")
    finally:
        state.sockets.discard(ws)
        await session.close()
        # Close the row even on a crash or an abrupt disconnect, so a talk that
        # ended badly still has its final numbers.
        try:
            await state.db.end_talk(talk_id, usage.snapshot() if usage else None)
        except Exception:
            log.exception("failed to close talk #%s", talk_id)


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:  # pragma: no cover
    log.warning("web/ directory missing at %s -- UI will not be served", WEB_DIR)
