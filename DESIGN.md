# Salindia — Design & Handoff

> Salindia is a local, real-time, two-way speech interpreter between Indonesian
> and English. You speak into a browser and read the translation as subtitles at
> the bottom of the screen. This document is for the next engineer or agent
> picking the project up. It covers what exists, why it is built this way, what
> was measured, and what is still open. The README covers usage; this covers
> reasoning.

## 1. Goal and constraints

- **Input:** live speech from a browser microphone, in Indonesian **or** English.
  A toggle chooses the direction: ID → EN or EN → ID.
- **Output:** a translation shown as subtitles at the bottom of the screen, as
  close to real time as possible. The page is meant to sit next to a slide
  deck, so **the Space key is never captured**.
- **Hardware:** MacBook Air M4, 24 GB, macOS 26, Python 3.13. There is no CUDA,
  so all inference runs on the Metal GPU through **MLX**.
- **Only open-weight models**, running locally. Audio never leaves the machine.
- The user chose a **Python backend** (not in-browser WebGPU) and **live
  streaming** (not record-then-translate).

## 2. Architecture

```
Browser (web/)                              Python (FastAPI, one process)
──────────────                              ─────────────────────────────
getUserMedia → AudioWorklet                 /ws WebSocket
  (recorder-worklet.js: 16 kHz Int16,         binary: Int16LE PCM, 16 kHz mono, 1024-sample chunks
   resamples if the browser ignores          text:   {"type":"stop"}
   the requested rate)                               {"type":"direction","value":"id-en"|"en-id"}
  → WebSocket ─────────────────────────────→        {"type":"ping"}
                                            StreamSession (session.py), one per connection
                                              EnergyVad (vad.py), 20 ms frames
                                              ├─ partials     → WhisperEngine  [BACKGROUND priority]
                                              ├─ speculation  → WhisperEngine  [URGENT]
                                              └─ finals: up to FINAL_WORKERS pipelines
                                                   ASR (or reuse speculation) → emit source
                                                   → Translator[direction]     → emit translation
                                              every model call passes through one GpuGate (gpu.py)
  ← JSON events ←──────────────────────────
app.js draws the last 2 cues as subtitles
```

### Files

| File | Responsibility |
|---|---|
| `server/config.py` | Every setting, read from environment variables (documented in `.env.example`). `Settings.directions` lists the directions the current mode supports. `Settings.public()` is what the UI receives. |
| `server/main.py` | FastAPI app, `/healthz`, `/ws`, and the static `web/` files. Builds a single `GpuGate`, the ASR engine, and one `Translator` per direction (reused when both directions use the same model repo). Warms the models in the background and keeps clients in a `loading` state until they are ready. |
| `server/gpu.py` | `GpuGate`: allows one model call at a time across the whole process, with a priority queue (`URGENT` before `BACKGROUND`) and correct handling of cancelled waiters. |
| `server/vad.py` | Dependency-free energy VAD: RMS against an adaptive noise floor, plus 3 frames of onset hysteresis. It is **the only VAD**, because mlx-whisper has no `vad_filter`. |
| `server/session.py` | The streaming state machine. It covers partials, speculation, parallel pipelines, the ordering turnstiles, and per-direction history. **Read this first.** |
| `server/langid.py` | Decides Indonesian vs English for auto direction (function words first, the API's label as a tiebreak). |
| `server/asr_openai.py` | `OpenAIWhisperEngine`: the OpenAI audio API behind the same interface (see §3.1b). |
| `server/asr.py` | `WhisperEngine`: wraps `mlx_whisper.transcribe` with the language pinned (`id`/`en`), and filters hallucinations. |
| `server/translate.py` | `Translator`: an `mlx_lm` LLM with one few-shot system prompt per direction and one reusable KV prompt cache per direction. |
| `web/` | Plain HTML/CSS/JS with no build step. The page shows subtitle cues, a direction toggle, and **Show original text** and **Show latency** checkboxes; the checkbox settings and the direction are saved in `localStorage`. |
| `tests/test_stream.py` | Tests the state machine with stubbed models. Fast and needs no model weights. **It must print ALL PASS.** |
| `tests/bench_real.py` | Real models on held-out sentences read by macOS TTS: `Damayanti` for Indonesian, `Samantha` for English. `MT_MODEL=` overrides the translator. |
| `tests/test_live_ws.py` | Full WebSocket path against a running server, with audio sent at real-time pace. Prints per-sentence latency. Arguments: `url direction gap-ms`. |

### WebSocket events (server → client)

| type | fields | meaning |
|---|---|---|
| `choose` | `engine` | No speech engine chosen yet (or the last attempt failed, see `engine.error`). The page shows the chooser. |
| `hello` | `ready`, `config`, `engine` | Sent on connect. `config` includes `mode`, `directions`, `default_direction`, `asr_model` and `mt_models`. |
| `loading` / `ready` | `config` | Model warmup status. On `ready`, the page sends its saved direction. |
| `direction` | `value` | Confirms a direction change. It applies from the **next** utterance. |
| `level` | `rms`, `floor`, `speaking` | Sent about every 100 ms; drives the level meter. |
| `speech_start` / `speech_end` | `id`, (`dir`) | VAD transitions. `id` is the utterance sequence number. |
| `partial` | `id`, `dir`, `text`, `ms` | In-progress text in the spoken language (in direct mode, already English). |
| `final` | `id`, `dir`, `src`, `dst`, `asr_ms`, `speculated` | Pipeline mode: `src` is set and `dst` is null. Direct mode: `dst` is set and `src` is null. |
| `context` | `style`, `notes` | Confirms the translation context for this session. |
| `translation` | `id`, `dir`, `dst`, `mt_ms` | Pipeline mode only. Always follows the `final` with the same `id`. |
| `discard` | `id` | The blip was too short, or Whisper returned nothing usable. |
| `idle` | | Sent after `stop`, once every pipeline has finished. |
| `error` | `message`, `id?` | |

`final` events are always emitted in `id` order, and so are `translation` events.

**Subtitle display (`app.js`)** shows **one cue at a time**; the user found two
cues confusing.
- Finished translations join a queue.
- Each translation stays on screen for
  `clamp(45 ms × chars, 1.8 s, 5 s)`. When several are waiting, that time is
  divided by the queue length, with a 0.9 s floor.
- The previous translation stays visible while the next sentence is being
  spoken. The in-progress (partial) cue is shown only before anything has been
  translated.
- Redraws are skipped when nothing changed, and the fade-in plays only when a
  new sentence takes over.

**Subtitle position.** The page has ▲ ▼ buttons in the bottom bar, and
Shift+↑ / Shift+↓ also work in fullscreen; plain arrows still change slides.
- They set `--subs-pos` on the stage: the gap below the subtitles as a % of the
  stage height, from 0 to 80 in steps of 4, default 4.
- The position is saved in `localStorage` (`salindia.subsPos`).

**Subtitle auto-hide.** A dropdown next to ▲ ▼ offers 2, 3, 5, 8, 10 or
15 s, or never; the default is 5 s, stored as `salindia.subsHide`.
- `scheduleHide()` runs whenever the displayed cue changes. It starts a timer
  only when the cue has a translation, so a line still waiting for one is never
  hidden.
- When the timer fires, `.subs.faded` sets opacity to 0; the next redraw
  removes the class.

**Slides (`server/slides.py`, `/api/slides`)**
- The raw PDF body is uploaded (no multipart), with the name in `X-Filename`.
  At most 3 decks are kept in memory.
- PDFium renders pages on its own thread (not through the GPU gate) to JPEG
  (q92, no chroma subsampling). JPEG encodes in about 9 ms, while PNG took
  about 1 s.
- Widths are rounded up to 160 px buckets, and rendered pages are cached per
  deck.
- In the UI:
  - **Open slides** button, or drag and drop a PDF.
  - Space / → / ↓ / PageDown / Enter / N for the next slide; Shift+Space /
    ← / ↑ / PageUp / Backspace / P for the previous one; Home / End; `F` for
    fullscreen on the stage, with subtitles overlaid and a live badge.
  - The deck id and page are saved, so a page reload during a talk restores the
    slide as long as the server hasn't restarted.
- Space is matched on `e.key === " "` **or** `e.code === "Space"`: the browser
  automation tool sends an empty `key`.

## 3. Key decisions

### 3.1 Two stages (`MODE=pipeline`, the default)

- **ASR:** `mlx-community/whisper-large-v3-turbo`, run with
  `task="transcribe"` and the spoken language pinned.
  - The language is pinned because auto-detection is slower and is fooled by
    code-switched Indonesian.
- **MT, one model per direction:**
  - **ID → EN:** `Qwen3-4B-Instruct-2507-4bit`. Quality is good and it takes
    about 0.9 s per sentence.
  - **EN → ID:** `Qwen3-8B-4bit`.
    - The 4B model converted **$2,350 into Rp2.350**, mixed *saya*/*aku*, and
      wrote *nggak* where *belum* was needed. Prompt rules did not fix this.
    - The 8B model keeps currencies, uses *belum* correctly, and gets register
      right. It takes about 1.7 s per sentence.
    - `enable_thinking=False` is passed to the chat template, because Qwen3-8B
      is a hybrid reasoning model.
  - **Rejected: `gemma-3-4b-it-4bit`.** On one sentence it broke character
    ("Okay, let's do this. Just give me the speech-to-text input."), and it
    used *Gue* for a formal thank-you.
- **Why not a single model?** Whisper's built-in `translate` task can only
  produce **English**, and turbo's translate quality is degraded.
  - `MODE=direct` (`whisper-large-v3-mlx`, translate task) remains available,
    but it supports **ID → EN only**. The UI disables EN → ID in that mode.
- **Why an LLM rather than NLLB or Marian?** Conversational Indonesian relies
  on slang, particles and code-switching.
- **Other models ruled out:**
  - SeamlessM4T v2: CC-BY-NC, so no commercial use.
  - Voxtral: no official Indonesian support.
  - faster-whisper: needs CUDA to be fast.
  - whisper.cpp + Core ML: the only route to the Apple Neural Engine. Still a
    possible future option.

### 3.1b Speech engine choice (local or OpenAI)

- **The server starts without loading anything.** `State.phase` moves
  `choose` → `loading` → `ready` (or `failed`). The WebSocket sends
  `choose` / `loading` / `ready` with an `engine` info block. The page shows a
  modal chooser that can't be dismissed, and posts the answer to
  `POST /api/engine` (`{engine, model?, api_key?}`). `GET /api/engine` returns
  the current state. `ASR_ENGINE=local|openai` skips the question. After an
  engine is chosen, changing it requires a server restart (the endpoint
  returns 409).
- **`server/asr_openai.py` (`OpenAIWhisperEngine`)** has the same `run()`
  contract as `WhisperEngine`.
  - It uses httpx directly (multipart, 16 kHz WAV), with no OpenAI SDK
    dependency. The parameters were checked against the official openai-python
    SDK source (`types/audio/*`).
  - `whisper-1` uses `verbose_json`, which returns segment `no_speech_prob` /
    `avg_logprob` for the hallucination filter. The `gpt-*-transcribe` models
    only support `json`.
  - `MODE=direct` uses `/audio/translations`, which exists for `whisper-1`
    only.
  - It does **not** go through the GPU gate.
  - `partials = False` by default (`OPENAI_PARTIALS`), because each live
    refresh would be a paid request. Speculation stays on and can waste a
    request when the speaker resumes.
  - Retries on 408/409/429/5xx and network errors (3 attempts, backoff
    0.5 s → 1.5 s). Errors become `AsrError` with readable messages, which the
    session sends as `error` events followed by a `discard`.
  - `warmup()` calls `GET /models/{model}` to validate the key and model. On
    failure the phase returns to `choose` with the error shown in the dialog.
- **Key handling.**
  - The key comes from `OPENAI_API_KEY` (`repr=False` on the settings field),
    or from the dialog. A dialog key lives only in the engine's HTTP client
    headers. It is never logged, and the page clears the input and never puts
    it in `localStorage`.
  - The server binds to 127.0.0.1 by default. Exposing it on a LAN would let
    anyone on that network use the key.
- **Not tested against the real API** (the user asked for this not to be
  done). `tests/test_openai_engine.py` covers the request shape, the formats,
  the filter, direct translation, 401/429/network errors and warmup, using
  `httpx.MockTransport`. The UI chooser was checked in a browser with
  `OPENAI_BASE_URL` pointed at a dead local port, which covered the error path
  and confirmed the key never appears in the log.
- `whisper-1` is Whisper **V2**. For Indonesian it may be *less* accurate than
  the local large-v3-turbo; the `gpt-4o-*-transcribe` models are the likely
  better API choice. This hasn't been measured.

### 3.1c Auto direction (OpenAI engine only)

- **Availability.** A third direction, `auto`, is offered only when the engine
  sets `auto_detect` (the OpenAI engine in transcribe mode) and `MODE=pipeline`.
  `State.directions()` builds the per-connection list, which is sent as
  `engine.directions` and enforced by `StreamSession.set_direction`. The user
  asked for this restriction: local Whisper could detect language too, but it
  isn't offered there.
- **Language hint.** An `auto` utterance is transcribed with `language=None`,
  so the `language` form field is omitted.
- **Resolving the language.** When the final arrives,
  `langid.resolve_language(text, api_language)` picks `id` or `en`:
  - Indonesian and English function words are counted in the transcript.
  - The API's label is used only to break a tie, because `whisper-1` often
    says *malay* for Indonesian and the `gpt-*-transcribe` models return no
    label.
  - With no evidence at all, the last language heard is kept.
- **After resolving.** The resolved direction selects the translator and its
  history. `final` / `translation` events carry the resolved `dir` plus
  `auto: true`, and the cue tag reads e.g. "AUTO · EN→ID".
- **UI.**
  - The Auto button is hidden unless the engine allows it.
  - The saved preference (`salindia.direction`) is kept even while Auto is
    unavailable, e.g. in the chooser phase, and it is re-applied on `ready`.
  - `T` cycles ID→EN → EN→ID → Auto.
- **Untested with real audio or the real API.** The session behaviour is
  covered by `test_stream.py` (a scripted fake engine, including the Malay
  mix-up), and the request shape by `test_openai_engine.py`.

### 3.1d Translation context (Indonesian style + presenter notes)

- **Style** (`formal`, `casual` or `match`; default `ID_STYLE=formal`, at the
  user's request for *saya* over *aku*).
  - `translate.build_prompt()` puts together the EN → ID prompt from a register
    rule plus a **matching set of examples**. The original examples were casual
    ("sori aku telat"), and few-shot examples outweigh rules, so a formal rule
    alone would lose to them.
  - ID → EN ignores the style.
- **Notes** (up to 1500 characters, both directions) are appended at the
  **end** of the system prompt, so the per-direction KV cache keeps reusing the
  long fixed prefix.
- **Per session.**
  - The page sends `{"type":"context","style","notes"}` on `ready` and on
    Save. The server acknowledges with a `context` event.
  - Changing the style drops the session's EN → ID history, because earlier
    turns in the old register would pull the model back to it.
  - The page stores the context in `localStorage` (`salindia.context`).
- **UI:** a **Context · saya/Anda** button opens a dialog with the style
  radios and a notes textarea. The button is highlighted when notes are set.
  Keyboard shortcuts are ignored while any dialog is open or a textarea has
  focus.
- **Unmeasured:** how well Qwen3-8B sticks to *saya* on real speech. Covered
  by unit tests only (prompt building, session plumbing).

### 3.2 Direction handling

- Each utterance is stamped with the direction that was active when it
  **started**. Switching in the middle of a sentence therefore affects the next
  sentence, which suits two people taking turns.
- Translator history is kept **per direction**. Mixing EN → ID pairs into an
  ID → EN prompt would teach the model the wrong output language.
- Each translator keeps **one KV prompt cache per direction**, so toggling does
  not evict the other direction's cached system prompt.

### 3.3 Streaming, speculation and parallel pipelines

- **Partials** re-decode the last `PARTIAL_WINDOW_S` (8 s) of audio.
  - Only one partial job runs at a time. If a job is still running, no new one
    is queued, which provides natural backpressure.
  - Partials are skipped while any final pipeline is in flight, and during
    pauses.
  - They run at `BACKGROUND` priority.
- **Speculation** (`SPECULATE_AFTER_MS`, default 250):
  - After 250 ms of silence, the final-quality decode of the whole utterance
    starts, before `SILENCE_MS` (650 ms) confirms that the sentence has ended.
  - If no speech arrives before the cut, that result becomes the final and ASR
    is skipped (`speculated: true` in the event; the UI shows "early start").
  - If speech resumes, the result is discarded, and a new speculation starts at
    the next pause.
  - The trigger uses `>=` because frames are 20 ms long: an `==` check against
    250 never fires. This was a real bug.
- **Pipelines** (`FINAL_WORKERS`, default 3; the user asked for 3):
  - Each finished utterance becomes its own task (ASR, then emit source, then
    translate, then emit translation).
  - A semaphore limits how many run at once.
  - Two `_Turnstile`s order the output: source lines in order, translations in
    order.
  - History is snapshotted when translation starts. With several pipelines in
    flight, the previous sentence may be missing from the context, which costs
    a little context but never correctness.
- **Adaptive segmentation for subtitles.** The user reported that long speech
  came out as one big chunk that was later split into parts that flashed past.
  Now:
  - After `SOFT_MAX_S` (6 s), a `SHORT_SILENCE_MS` (250 ms) breath is enough
    to end an utterance.
  - `MAX_UTTERANCE_S` was lowered from 22 to 12 s. A forced cut is placed at
    the quietest 100 ms of the last 2 s (`_quiet_cut`, using the per-frame
    `_utt_meta`), and the remainder carries over into the next utterance.
- **The GPU gate is essential.** It is what makes the parallel pipelines safe.
  See §4.
- Other details:
  - 300 ms of preroll is kept before each utterance.
  - Trailing silence is kept in the buffer.
  - `MAX_UTTERANCE_S` (22 s) forces a cut and opens the next utterance
    immediately.
  - Blips are gated on **speech frames**, not buffer length.

### 3.4 Hallucination filter (`asr.py`)

- `_ALWAYS_HALLUCINATED` is always dropped. It holds caption boilerplate such
  as "thank you for watching", "subtitles by the amara.org community" and
  "you".
- `_SUSPECT_IF_UNSURE` is dropped only when `no_speech_prob > 0.3`. It holds
  short phrases people genuinely say, such as "thank you", "terima kasih" and
  "bye". A blanket drop was fine for Indonesian-only input, but it would eat
  real English endings.
- Also dropped: results with high `no_speech_prob` together with low
  `avg_logprob`, and repeated-token decoder loops.
- Phrases are matched after lowercasing and **stripping punctuation**, so list
  entries must not contain punctuation (e.g. `amaraorg`).

## 4. Measurements (M4 Air, 24 GB)

| What | Result |
|---|---|
| Whisper turbo, 2–20 s of audio | ~0.9–1.05 s, **flat**. Whisper pads every input to 30 s. |
| Qwen3-4B, one sentence (warm cache, measured alone) | EN → ID 1.13 s, ID → EN 0.86 s |
| Qwen3-8B, one sentence (warm cache, measured alone) | EN → ID 1.71 s, ID → EN 1.49 s |
| Same model, 3 jobs in 3 threads vs one after another | Whisper 2845 vs 2910 ms; LLM 1797 vs 1931 ms. **Only 2–7% faster**, and the first result arrives much later. |
| KV prompt-cache reuse | Saves about 430 ms per sentence |
| Weights warmup (all three models) | About 16 s |

**End-to-end latency** is measured from the moment the voice stops to the
translation appearing (`test_live_ws.py`, 4 sentences, TTS voices). Note:
earlier versions of this document quoted "~1.1 s" measured from the *last audio
sample*, which is a different and flattering reference point.

| Config | ID → EN, 1.1 s gaps | ID → EN, 0.7 s gaps | EN → ID (8B), 1.1 s | EN → ID (8B), 0.7 s |
|---|---|---|---|---|
| 1 worker, no speculation, no gate (old behaviour) | 2.59 s | 2.94 s | 3.22 s | 3.80 s |
| 3 workers + speculation, **no gate** | 4.28 s | 3.40 s | **10.84 s** | **13.16 s** |
| 3 workers + speculation, **with gate** (current default) | **not measured yet** | | | |

The second row is why `gpu.py` exists. Whisper running at the same time as the
8B LLM slowed both of them dramatically. The user stopped the benchmark before
the gated configuration was measured, so **the current defaults are
unverified**. See §8.1.

**Quality** was checked on held-out sentences, none of which appear in the
few-shot prompts:

- **ID → EN:** ASR correct on 6/7 (*baca* was heard as *baka*); all 7
  translations natural.
- **EN → ID with 8B:** clearly better than 4B.
  - One remaining error: "that line is ridiculous" became *kalimat*
    ("sentence") instead of *antrean* ("queue").
  - Minor issues: *bantuan-mu* instead of *bantuanmu*, and "Halo, sore".

## 5. Lessons learned (bugs found and fixed)

1. **GPU contention.** A partial running during a translation made it take
   3207 ms instead of 644 ms. Different models running at the same time made
   latency 3–4x worse. The fix is the `GpuGate`, with finals and translations
   ahead of partials.
2. **The blip filter measured buffer length**, which includes preroll and
   trailing silence. It now counts speech frames.
3. **The system prompt was reprocessed on every call.** A prompt cache that
   finds the longest common token prefix and calls `trim_prompt_cache` fixed
   it. Few-shot examples are now almost free.
4. **Contaminated evaluation, twice.** First, benchmark sentences also appeared
   as few-shot examples. Later, a prompt rule quoted a benchmark sentence
   ("a line you wait in is an antrean"); that rule was removed. Keep benchmark
   sentences out of the prompts.
5. **A resampler off-by-one** in the worklet produced NaN at the end of a
   buffer.
6. **The speculation trigger used `==` on a 250 ms threshold with 20 ms
   frames**, so it never fired.
7. **Process hygiene:**
   - Each agent Bash call runs in a fresh shell, so `kill %1` does nothing.
     Kill by port instead.
   - The user's shell is **zsh**, which does not word-split an unquoted
     `$cfg`. A loop passing `env $cfg` gave the server
     `FINAL_WORKERS="1 SPECULATE_AFTER_MS=0"` and it crashed while reading the
     config. Run such loops under `bash`.
8. **A false claim in the docs:** an earlier `vad.py` docstring and this
   document said Whisper also ran Silero VAD. That is true of faster-whisper,
   not mlx-whisper.
9. **UI focus and Space:** a clicked button or checkbox keeps focus, and Space
   re-activates it. Space belongs to the presenter's slides, so the page has no
   Space shortcut and drops focus after every click. `T` switches direction.

10. **Stale cached `app.js`.** After the UI rewrite, the user's browser ran
    the new `index.html` with the old cached `app.js`. The direction buttons did
    nothing and no subtitles appeared, because the old script crashed on
    elements that no longer exist. Fixes:
    - a `Cache-Control: no-cache` middleware in `main.py`;
    - a `?v=N` suffix on the asset URLs in `index.html` (**bump it whenever
      `app.js` or `styles.css` changes**);
    - a `window.onerror` handler that shows script errors in the status line.

11. **Memory pressure from parallel dev servers.** A debug server (~9 GB of
    models) running alongside the user's `run.sh` (~9 GB) plus a test harness
    pushed about 4 GB into swap. Model calls became 5–14x slower, which looked
    like a hang. Run only one model-loading process at a time on this 24 GB
    machine.

## 6. Environment notes

- The user's Homebrew `ffmpeg` is broken (missing `libx265.215.dylib`). The app
  never calls ffmpeg, and the tests generate WAV files directly with
  `say -v <voice> -o x.wav --data-format=LEI16@16000 "..."`.
- Weights are cached in `~/.cache/huggingface`:
  - Used by the app: whisper-large-v3-turbo (~1.6 GB), Qwen3-4B-Instruct-2507-4bit
    (~2.4 GB), Qwen3-8B-4bit (~4.6 GB).
  - Downloaded but unused: gemma-3-4b-it-4bit (~2.6 GB), which can be deleted.
  - **Not downloaded:** whisper-large-v3-mlx, which direct mode needs.
- Memory use in pipeline mode is about 9 GB. On a 16 GB Mac, point both
  `MT_MODEL_*` variables at the 4B model.
- `mlx-lm` pulls in `transformers` and `torch`, so a large venv is expected.
- Browsers allow the microphone only on `localhost` or HTTPS.
- The Air has no fan, so sustained load may throttle it.
- `/tmp/salindia-serve.sh PORT [ENV=VAL ...]` is a throwaway helper written
  during development: it restarts a test server and waits for `ready`. It is
  not part of the repo.

## 7. How to verify changes

```bash
./.venv/bin/python tests/test_stream.py                               # always; must print ALL PASS
./.venv/bin/python tests/bench_real.py /tmp/salindia-audio both       # after any ASR/MT/prompt change
# end-to-end latency (start a server on 8123 first; run loops under bash):
./.venv/bin/python tests/test_live_ws.py ws://127.0.0.1:8123/ws id-en 1100
./.venv/bin/python tests/test_live_ws.py ws://127.0.0.1:8123/ws en-id 700
lsof -t -iTCP:8123 -sTCP:LISTEN | xargs kill
```

The subtitle UI was checked in a browser by injecting events: layout, the
checkboxes and the `T` shortcut all work. **Nobody has yet tested it with a
real human voice through the microphone.**

## 8. Open work

1. **Measure the current defaults** (gated, `FINAL_WORKERS=3`,
   `SPECULATE_AFTER_MS=250`) against `FINAL_WORKERS=1 SPECULATE_AFTER_MS=0`
   using `test_live_ws.py`, in both directions with 1100 ms and 700 ms gaps.
   If the defaults are not faster than the 2.6 s / 3.2 s baseline, change them.
   Separate the effect of speculation from the effect of workers.
2. **Try the OpenAI engine with a real key** (the user will do this). Compare
   `whisper-1` with `gpt-4o-transcribe` for Indonesian, and decide whether
   partials are worth enabling (`OPENAI_PARTIALS=1`).
3. **The user asked for no more benchmarks; they test by hand.** Don't run
   model-loading benchmarks or servers unless asked.
3. **Test with a real microphone** and tune `VAD_ABS_THRESHOLD` and
   `SILENCE_MS`. Echo cancellation and noise suppression interact with the
   energy VAD.
3. **Speculation hit rate on real speech.** A partial that is already running
   when the pause starts delays the speculative decode by up to ~1 s, because
   the GPU runs one job at a time. Consider not starting new partials once the
   utterance is long, or cancelling partials between frames.
4. **Replace the energy VAD with Silero VAD** (the `silero-vad` package, or its
   ONNX model).
5. **EN → ID quality:** word-sense errors ("line" → *kalimat*). Options: try
   `Qwen3-4B-Instruct-2507-8bit`, a larger model, or a glossary.
6. **Test `MODE=direct`.** Its code was updated for the new message fields, but
   it has never been run.
7. **Glossary / domain terms** for both Whisper's `initial_prompt` (supported
   by `WhisperEngine.run` but unused) and the translator prompts.
8. **Stream the translation token by token** into the subtitle.
9. **Presenter features:** subtitle font size and position controls, a
   pop-out or transparent overlay window, and auto-hiding cues after N seconds
   of silence.
10. **Text-to-speech output, export (SRT/TXT), packaging** (`pyproject`,
    pytest, CI on macOS arm64), and multi-user fairness.

## 9. Conventions

- No frontend build step. Keep `web/` as plain JS/HTML/CSS.
- Never bind Space in the UI.
- All configuration goes through `server/config.py` and environment variables.
  Document new settings in `.env.example`.
- Every model call goes through `GpuGate`. Never add a model call that bypasses
  it.
- Heavy imports (`mlx_whisper`, `mlx_lm`) happen lazily inside the worker
  threads.
- Comments explain **why**, not what. Match the existing density.
