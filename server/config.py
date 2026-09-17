"""Settings, read from the environment with defaults tuned for an M4 Mac."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


def _s(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _i(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _f(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


MODES = ("pipeline", "direct")

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


@dataclass(frozen=True)
class Settings:
    # "pipeline" = turbo transcribe (id) + LLM translate. "direct" = large-v3 translate.
    mode: str = _s("MODE", "pipeline")
    # Indonesian output register: formal (saya/Anda), casual (aku/kamu) or match
    # the speaker. The UI can change it per session.
    id_style: str = _s("ID_STYLE", "formal")
    # Direction a new connection starts in; the UI can switch it per session.
    # MODE=direct can only produce English, so en-id falls back there (see below).
    default_direction: str = _s("DIRECTION", "en-id")

    # Speech-to-text engine: "local" (MLX Whisper), "openai" (API), or empty to
    # let the user pick in the browser before anything is loaded.
    asr_engine: str = _s("ASR_ENGINE", "")
    # Skip the engine chooser entirely: "openai" or "local". The browser goes
    # straight to the talk title. ASR_ENGINE is the older name and still works.
    force_approach: str = _s("FORCE_APPROACH", _s("ASR_ENGINE", ""))

    # Where per-talk records go. Relative paths resolve next to the project.
    db_path: str = _s("DB_PATH", "salindia.sqlite3")

    # Gate the whole app behind a shared phrase. Empty = open to anyone who can
    # reach the port. This keeps the page off the open internet; it is not
    # per-user auth, and it is only as private as the phrase you share.
    access_passphrase: str = field(default=_s("ACCESS_PASSPHRASE", ""), repr=False)
    openai_api_key: str = field(default=_s("OPENAI_API_KEY", ""), repr=False)
    openai_base_url: str = _s("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_asr_model: str = _s("OPENAI_ASR_MODEL", "whisper-1")
    # Chat model used for translation when the speech engine is OpenAI. Keeps
    # the whole pipeline off the local GPU, which is the only way MODE=pipeline
    # runs anywhere but Apple Silicon.
    openai_mt_model: str = _s("OPENAI_MT_MODEL", "gpt-4o-mini")
    # Live partials cost one API request per refresh, so they're off by default.
    openai_partials: bool = _s("OPENAI_PARTIALS", "0") not in ("", "0", "false", "no")
    # A speculation the speaker talks through is thrown away; on the API that is
    # a paid request, and the head start it buys is small next to the round-trip.
    openai_speculate: bool = _s("OPENAI_SPECULATE", "0") not in ("", "0", "false", "no")

    asr_model: str = _s("ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
    asr_model_direct: str = _s("ASR_MODEL_DIRECT", "mlx-community/whisper-large-v3-mlx")
    # Translator per direction. EN->ID gets the bigger model: 4B was fine for
    # ID->EN but converted currencies and mixed saya/aku going the other way.
    # Pointing both at the same repo loads it once. (MT_MODEL is the old name.)
    mt_model_id_en: str = _s(
        "MT_MODEL_ID_EN", _s("MT_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
    )
    mt_model_en_id: str = _s("MT_MODEL_EN_ID", "mlx-community/Qwen3-8B-4bit")

    partial_window_s: float = _f("PARTIAL_WINDOW_S", 8.0)
    silence_ms: int = _i("SILENCE_MS", 650)
    # Subtitles want short chunks. Once an utterance is this long, a mere breath
    # (SHORT_SILENCE_MS) is enough to end it instead of a full SILENCE_MS pause.
    soft_max_s: float = _f("SOFT_MAX_S", 6.0)
    short_silence_ms: int = _i("SHORT_SILENCE_MS", 250)
    # Hard cap; the cut is placed at the quietest point of the last couple of
    # seconds rather than wherever the cap happens to land.
    max_utterance_s: float = _f("MAX_UTTERANCE_S", 12.0)
    min_utterance_ms: int = _i("MIN_UTTERANCE_MS", 350)

    # Audio kept from just before speech onset, so the first syllable isn't clipped.
    preroll_ms: int = _i("PREROLL_MS", 300)
    # Minimum new audio before a fresh partial is worth running.
    partial_min_new_ms: int = _i("PARTIAL_MIN_NEW_MS", 400)
    # Start the final decode after this much silence, before SILENCE_MS confirms
    # the utterance is over. 0 disables. Must be below SILENCE_MS to matter.
    speculate_after_ms: int = _i("SPECULATE_AFTER_MS", 250)
    # Sentences processed concurrently (each model still runs one job at a time).
    final_workers: int = _i("FINAL_WORKERS", 3)

    # Stop listening after this long with no speech. The browser stops its own
    # capture too, but the server keeps its own clock: a wedged or killed page
    # can leave the socket open and streaming silence, and with a paid engine
    # that is real money. 0 disables.
    idle_stop_s: float = _f("IDLE_STOP_S", 60.0)

    vad_abs_threshold: float = _f("VAD_ABS_THRESHOLD", 0.006)
    vad_noise_ratio: float = _f("VAD_NOISE_RATIO", 3.0)
    vad_onset_frames: int = _i("VAD_ONSET_FRAMES", 3)

    # How much prior context to hand the translator for pronoun/term consistency.
    mt_context_turns: int = _i("MT_CONTEXT_TURNS", 2)
    mt_max_tokens: int = _i("MT_MAX_TOKENS", 256)

    host: str = _s("HOST", "127.0.0.1")
    port: int = _i("PORT", 8000)

    @property
    def needs_passphrase(self) -> bool:
        return bool(self.access_passphrase)

    # -- mode-dependent views ------------------------------------------------
    #
    # The mode is switchable from the bar while the server runs, so everything
    # it decides is a function of a mode argument. The properties below are the
    # same answers for the mode this process started in.

    def directions_for(self, mode: str) -> tuple[str, ...]:
        # Whisper's built-in translate task only ever outputs English, so direct
        # mode can't do en-id; that needs the LLM stage.
        return ("id-en", "en-id") if mode == "pipeline" else ("id-en",)

    def asr_model_for(self, mode: str) -> str:
        return self.asr_model if mode == "pipeline" else self.asr_model_direct

    def default_direction_for(self, mode: str) -> str:
        """The configured default, or the nearest thing this mode can do."""
        dirs = self.directions_for(mode)
        return self.default_direction if self.default_direction in dirs else dirs[0]

    @property
    def directions(self) -> tuple[str, ...]:
        return self.directions_for(self.mode)

    def mt_model_for(self, direction: str) -> str:
        return {"id-en": self.mt_model_id_en, "en-id": self.mt_model_en_id}[direction]

    @property
    def effective_asr_model(self) -> str:
        return self.asr_model_for(self.mode)

    def public(self, mode: str | None = None) -> dict:
        """The subset worth showing in the UI, as of ``mode`` (default: ours)."""
        mode = mode or self.mode
        directions = self.directions_for(mode)
        d = asdict(self)
        d["mode"] = mode
        d["asr_model"] = self.asr_model_for(mode)
        d["mt_models"] = (
            {k: self.mt_model_for(k) for k in directions} if mode == "pipeline" else {}
        )
        d["directions"] = list(directions)
        d["default_direction"] = self.default_direction_for(mode)
        d["needs_passphrase"] = self.needs_passphrase
        # The dropdown in the bar offers these. MODE in the environment only
        # says where a fresh server starts, not what it will accept later.
        d["modes"] = list(MODES)
        return {
            k: d[k]
            for k in (
                "mode",
                "modes",
                "asr_model",
                "mt_models",
                "directions",
                "default_direction",
                "id_style",
                "silence_ms",
                "idle_stop_s",
                "force_approach",
                "needs_passphrase",
                "max_utterance_s",
            )
        }


settings = Settings()

if settings.mode not in MODES:
    raise SystemExit(f"MODE must be 'pipeline' or 'direct', got {settings.mode!r}")
if settings.id_style not in ("formal", "casual", "match"):
    raise SystemExit(f"ID_STYLE must be formal, casual or match, got {settings.id_style!r}")
if settings.asr_engine not in ("", "local", "openai"):
    raise SystemExit(f"ASR_ENGINE must be 'local', 'openai' or empty, got {settings.asr_engine!r}")
if settings.force_approach not in ("", "local", "openai"):
    raise SystemExit(
        f"FORCE_APPROACH must be 'local', 'openai' or empty, got {settings.force_approach!r}"
    )
if settings.default_direction not in ("id-en", "en-id"):
    raise SystemExit(
        f"DIRECTION must be id-en or en-id, got {settings.default_direction!r}"
    )
if os.environ.get("DIRECTION") and settings.default_direction not in settings.directions:
    # Both were asked for by name and they contradict each other. A default
    # nobody chose just falls back instead (see default_direction_for), and the
    # mode is switchable in the bar, so this only guards the starting pair.
    raise SystemExit(
        f"DIRECTION={settings.default_direction!r} is not available in "
        f"MODE={settings.mode} (choose from {', '.join(settings.directions)})"
    )
