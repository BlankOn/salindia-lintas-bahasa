const TARGET_RATE = 16000;
const LABEL = { "id-en": "ID→EN", "en-id": "EN→ID", auto: "AUTO" };
const CYCLE = ["id-en", "en-id", "auto"]; // order the T key steps through

const $ = (id) => document.getElementById(id);
const el = {
  dot: $("dot"), status: $("status"), brand: $("brand"), hint: $("hint"), subs: $("subs"),
  record: $("record"), recLabel: $("recLabel"),
  meterFill: $("meterFill"), meterThresh: $("meterThresh"),
  copy: $("copy"), clear: $("clear"), showSrc: $("showSrc"), showLat: $("showLat"),
  dirButtons: [...document.querySelectorAll(".seg button")],
  stage: $("stage"), slide: $("slide"), liveBadge: $("liveBadge"), liveDir: $("liveDir"),
  pdfInput: $("pdfInput"), openSlides: $("openSlides"), deckNav: $("deckNav"),
  prevSlide: $("prevSlide"), nextSlide: $("nextSlide"), pageLabel: $("pageLabel"),
  fullscreen: $("fullscreen"), closeSlides: $("closeSlides"),
  subsUp: $("subsUp"), subsDown: $("subsDown"), subsHide: $("subsHide"),
  contextBtn: $("contextBtn"), contextDialog: $("contextDialog"), contextForm: $("contextForm"),
  contextNotes: $("contextNotes"), contextCancel: $("contextCancel"),
  engineDialog: $("engineDialog"), engineForm: $("engineForm"), engineGo: $("engineGo"),
  engineError: $("engineError"), openaiFields: $("openaiFields"), openaiModel: $("openaiModel"),
  openaiKey: $("openaiKey"), keyRow: $("keyRow"), keyNote: $("keyNote"), localModel: $("localModel"),
};

let ws = null, audioCtx = null, node = null, stream = null;
let live = false, ready = false, config = {};
let direction = load("direction", "id-en");
let deck = null; // open slide deck: { id, name, pages, sizes }
let engine = {};  // speech engine state from the server (see /api/engine)
let loading = false;  // models are warming up; survives a dropped socket
let page = 1;
if (!(direction in LABEL)) direction = "id-en"; // e.g. a bad value saved by an older build
// What the user picked last. "auto" may be unavailable until the server says
// which engine is running, so it is kept separately from what is applied.
const preferredDirection = direction;

// Surface script errors in the status line instead of failing silently.
window.addEventListener("error", (e) => setStatus(`page error: ${e.message}`, true));

// Every utterance this session: id -> { dir, src, dst, partial, asrMs, mtMs, early, done }.
// Only the last few are drawn, but Copy uses all of them.
const cues = new Map();

// ------------------------------------------------------------- persistence

function load(key, fallback) {
  try { return localStorage.getItem("salindia." + key) ?? fallback; } catch { return fallback; }
}
function save(key, value) {
  try { localStorage.setItem("salindia." + key, value); } catch { /* private mode */ }
}

// ------------------------------------------------------------------ status

function setStatus(text, isError = false) {
  el.status.textContent = text;
  el.status.classList.toggle("err", isError);
}
function setDot(cls) { el.dot.className = "dot" + (cls ? " " + cls : ""); }

// --------------------------------------------------------------- subtitles

function cue(id) {
  if (!cues.has(id)) cues.set(id, { dir: direction, src: "", dst: "", partial: "", done: false });
  return cues.get(id);
}

// One cue on screen at a time, paced so people can read it. Finished
// translations join a queue; each stays up for a minimum time based on its
// length before the next replaces it, and the hold shrinks when a backlog
// builds so the subtitles never drift far behind the speaker. Before anything
// has been translated, the in-progress cue is shown instead.
const HOLD_MIN_MS = 1800, HOLD_MAX_MS = 5000, HOLD_PER_CHAR_MS = 45, HOLD_FLOOR_MS = 900;
const queued = [];        // cue ids with a translation, waiting to be shown
let showing = null;      // cue id on screen
let showingSince = 0;
let pumpTimer = 0;

function holdFor(id) {
  const c = cues.get(id);
  const chars = (c?.dst || "").length;
  let ms = Math.min(HOLD_MAX_MS, Math.max(HOLD_MIN_MS, chars * HOLD_PER_CHAR_MS));
  if (queued.length >= 2) ms = Math.max(HOLD_FLOOR_MS, ms / queued.length); // catch up
  return ms;
}

function enqueue(id) {
  if (id !== showing && !queued.includes(id)) {
    queued.push(id);
    queued.sort((a, b) => a - b);
  }
  pump();
}

function pump() {
  clearTimeout(pumpTimer);
  if (!queued.length) return;
  const wait = showing === null ? 0 : showingSince + holdFor(showing) - performance.now();
  if (wait > 0) { pumpTimer = setTimeout(pump, wait); return; }
  showing = queued.shift();
  showingSince = performance.now();
  render();
  if (queued.length) pumpTimer = setTimeout(pump, holdFor(showing));
}

function currentCue() {
  if (showing !== null && cues.has(showing)) return [showing];
  const ids = [...cues.keys()].sort((a, b) => a - b);
  return ids.length ? [ids[ids.length - 1]] : [];
}

let shown = "";       // signature of what's on screen, to skip no-op redraws
let shownId = null;   // cue id on screen; only a new sentence gets the fade-in

function render() {
  el.hint.classList.toggle("gone", cues.size > 0 || !!deck);
  const ids = currentCue();
  const sig = JSON.stringify(ids.map((id) => {
    const c = cues.get(id);
    return [id, c.src, c.partial, c.dst, c.asrMs, c.mtMs, config.mode];
  }));
  if (sig === shown) return;
  shown = sig;
  scheduleHide(ids.length ? cues.get(ids[0]) : null);
  el.subs.replaceChildren(...ids.map((id) => {
    const c = cues.get(id);
    const box = document.createElement("div");
    box.className = "cue" + (id !== shownId ? " enter" : "");
    shownId = id;

    const srcText = c.src || c.partial;
    if (srcText && config.mode !== "direct") {
      const src = document.createElement("div");
      src.className = "src" + (c.src ? "" : " live");
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = (c.auto ? "AUTO · " : "") + (LABEL[c.dir] || c.dir);
      src.append(tag, srcText);
      box.append(src);
    }

    const dst = document.createElement("div");
    // In direct mode partials are already in the target language.
    const dstText = c.dst || (config.mode === "direct" ? c.partial : "");
    dst.className = "dst" + (c.dst ? "" : " pending");
    dst.textContent = dstText;
    box.append(dst);

    if (c.asrMs != null) {
      const lat = document.createElement("div");
      lat.className = "lat";
      const parts = [`asr ${c.asrMs}ms`];
      if (c.mtMs != null) parts.push(`mt ${c.mtMs}ms`);
      else if (!c.dst) parts.push("translating");
      if (c.early) parts.push("early start");
      lat.textContent = parts.join(" · ");
      box.append(lat);
    }
    return box;
  }));
}

// --------------------------------------------------------------- direction

function allowedDirections() {
  return engine.directions || config.directions || [];
}

function applyDirection(value, { persist = true } = {}) {
  direction = value;
  if (persist) save("direction", value);
  el.liveDir.textContent = LABEL[value] || "";
  for (const b of el.dirButtons) b.setAttribute("aria-pressed", String(b.dataset.dir === value));
}

function requestDirection(value) {
  if (!allowedDirections().includes(value)) return;
  applyDirection(value); // optimistic; the server confirms or errors
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "direction", value }));
  }
  const name = value === "auto" ? "AUTO (detects ID or EN per sentence)" : LABEL[value];
  setStatus(live ? `switched to ${name} — applies from the next sentence` : `direction ${name}`);
}

function syncDirectionButtons() {
  const allowed = allowedDirections();
  for (const b of el.dirButtons) {
    const ok = allowed.includes(b.dataset.dir);
    // Auto only exists with an engine that detects language; hide it otherwise
    // rather than showing a button that can never be pressed.
    if (b.dataset.dir === "auto") b.hidden = !ok;
    b.disabled = !ok;
    b.title = ok ? ""
      : b.dataset.dir === "auto" ? "Auto-detect needs the OpenAI speech engine"
      : "Not available in direct mode (needs MODE=pipeline)";
  }
  // Restore the user's pick once it's allowed; fall back without forgetting it.
  const want = allowed.includes(preferredDirection) ? preferredDirection
    : allowed.includes(direction) ? direction
    : (config.default_direction || allowed[0]);
  applyDirection(want, { persist: want === preferredDirection });
}

// ------------------------------------------------------------ engine choice

// Which models are actually running. It used to sit in the top bar, but it is
// reference information, not a control -- it lives on the brand's tooltip now.
// The translators come from the engine when it reports them: with the OpenAI
// engine the configured MLX repos are never loaded, so config would be wrong.
function renderMeta() {
  const short = (r) => (r || "").split("/").pop();
  const asr = engine.asr_model
    ? `${engine.engine === "openai" ? "OpenAI " : ""}${short(engine.asr_model)}`
    : "not chosen";
  const mt = Object.entries(engine.mt_models || config.mt_models || {})
    .map(([d, r]) => `${LABEL[d]} ${short(r)}`).join(" · ");
  el.brand.title = [config.mode, `asr ${asr}`, mt].filter(Boolean).join("  ·  ");
}

function chosenEngine() {
  return el.engineForm.querySelector('input[name="engine"]:checked')?.value || "openai";
}

function syncEngineFields() {
  el.openaiFields.hidden = chosenEngine() !== "openai";
}

function openEngineDialog() {
  el.localModel.textContent = (engine.local_asr_model || "").split("/").pop() || "large-v3-turbo";
  const models = engine.openai_models || ["whisper-1"];
  el.openaiModel.replaceChildren(...models.map((m) => new Option(m, m)));
  el.openaiModel.value = load("openaiModel", engine.openai_default_model || models[0]);

  const hasEnvKey = !!engine.openai_key_configured;
  el.openaiKey.required = false;
  el.keyRow.hidden = hasEnvKey;
  el.keyNote.textContent = hasEnvKey
    ? "Using OPENAI_API_KEY from the server's .env."
    : "Kept in the server's memory only. It is never written to disk or to this browser.";

  const last = load("engine", "openai");
  const radio = el.engineForm.querySelector(`input[name="engine"][value="${last}"]`);
  if (radio) radio.checked = true;
  syncEngineFields();

  el.engineError.hidden = !engine.error;
  el.engineError.textContent = engine.error || "";
  el.engineGo.disabled = false;
  el.engineGo.textContent = "Continue";
  if (!el.engineDialog.open) el.engineDialog.showModal();
}

function closeEngineDialog() {
  if (el.engineDialog.open) el.engineDialog.close();
  el.openaiKey.value = ""; // don't leave the key sitting in the page
}

el.engineForm.addEventListener("change", syncEngineFields);
// The dialog is required: Esc must not dismiss it and leave the app stuck.
el.engineDialog.addEventListener("cancel", (e) => e.preventDefault());

el.engineForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const choice = chosenEngine();
  const body = { engine: choice };
  if (choice === "openai") {
    body.model = el.openaiModel.value;
    body.api_key = el.openaiKey.value.trim();
    if (!engine.openai_key_configured && !body.api_key) {
      el.engineError.hidden = false;
      el.engineError.textContent = "Enter an OpenAI API key, or set OPENAI_API_KEY in .env.";
      el.openaiKey.focus();
      return;
    }
    save("openaiModel", body.model);
  }
  save("engine", choice);
  el.engineGo.disabled = true;
  el.engineGo.textContent = choice === "openai" ? "Checking key…" : "Starting…";
  el.engineError.hidden = true;
  try {
    const res = await fetch("api/engine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    // The WebSocket reports "loading", then either "ready" or "choose" + error.
  } catch (err) {
    el.engineGo.disabled = false;
    el.engineGo.textContent = "Continue";
    el.engineError.hidden = false;
    el.engineError.textContent = err.message;
  }
});

// ---------------------------------------------------------------- websocket

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("connected, checking models…");
  ws.onclose = () => {
    ready = false;
    setDot("");
    el.record.disabled = true;
    // Models keep loading on the server across a dropped socket, so say that
    // rather than "Disconnected" -- the reconnect below picks the step back up.
    el.recLabel.textContent = loading ? "Loading models…" : "Disconnected";
    setStatus(loading
      ? "still loading models on the server — reconnecting…"
      : "connection closed — reconnecting…", !loading);
    if (live) stopCapture();
    setTimeout(connect, 2000);
  };
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
}

function handle(msg) {
  switch (msg.type) {
    case "hello": {
      config = msg.config || {};
      engine = msg.engine || {};
      renderMeta();
      renderContextButton();
      syncDirectionButtons();
      break;
    }

    case "choose":
      engine = msg.engine || engine;
      loading = false;
      openEngineDialog();
      setStatus("choose a speech engine to start");
      break;

    case "loading":
      engine = msg.engine || engine;
      closeEngineDialog();
      renderMeta();
      loading = true;
      setDot("loading");
      el.record.disabled = true;
      // The server names the step it is on; a first run sits on one of these
      // for minutes, so show it rather than a generic "loading".
      el.recLabel.textContent = engine.detail || "Loading models…";
      setStatus(engine.detail || "loading models — the first run downloads weights, this can take a few minutes…");
      break;

    case "ready":
      engine = msg.engine || engine;
      closeEngineDialog();
      renderMeta();
      syncDirectionButtons();
      ready = true;
      loading = false;
      setDot("ready");
      el.record.disabled = false;
      el.recLabel.textContent = "Start listening";
      // Tell this new connection which direction the page is showing.
      ws.send(JSON.stringify({ type: "direction", value: direction }));
      sendContext();
      setStatus(`ready · ${LABEL[direction]}`);
      break;

    case "direction":
      applyDirection(msg.value);
      break;

    case "context":
      context = { style: msg.style, notes: msg.notes };
      renderContextButton();
      break;

    case "level": {
      // RMS is small and perceived logarithmically; stretch it to be visible.
      el.meterFill.style.width = Math.min(100, (msg.rms / 0.08) * 100).toFixed(1) + "%";
      el.meterFill.classList.toggle("speaking", msg.speaking);
      const t = Math.min(100, (Math.max(0.006, msg.floor * 3) / 0.08) * 100);
      el.meterThresh.style.left = t.toFixed(1) + "%";
      break;
    }

    case "speech_start":
      cue(msg.id).dir = msg.dir;
      render();
      break;

    case "partial":
      if (!cue(msg.id).src) { cue(msg.id).partial = msg.text; render(); }
      break;

    case "final": {
      const c = cue(msg.id);
      c.dir = msg.dir;
      c.auto = !!msg.auto;
      c.asrMs = msg.asr_ms;
      c.early = !!msg.speculated;
      if (msg.src) c.src = msg.src;
      if (msg.dst) { c.dst = msg.dst; c.done = true; enqueue(msg.id); }
      render();
      break;
    }

    case "translation": {
      const c = cue(msg.id);
      c.dst = msg.dst;
      c.mtMs = msg.mt_ms;
      c.done = true;
      if (c.dst) enqueue(msg.id);
      render();
      break;
    }

    case "discard":
      cues.delete(msg.id);
      render();
      break;

    case "idle":
      setStatus(`stopped · ${LABEL[direction]}`);
      break;

    case "error":
      setStatus(msg.message, true);
      break;
  }
}

// ------------------------------------------------------------------ capture

async function startCapture() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    setStatus(`microphone unavailable: ${err.message}`, true);
    return;
  }

  audioCtx = new AudioContext({ sampleRate: TARGET_RATE });
  await audioCtx.resume();
  await audioCtx.audioWorklet.addModule("recorder-worklet.js");

  const source = audioCtx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(audioCtx, "pcm-worklet", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    processorOptions: { targetRate: TARGET_RATE },
  });
  node.port.onmessage = (ev) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(ev.data);
  };
  source.connect(node);

  live = true;
  el.liveBadge.hidden = false;
  setDot("live");
  el.record.classList.add("live");
  el.recLabel.textContent = "Stop";
  const note = audioCtx.sampleRate !== TARGET_RATE ? ` (resampling from ${audioCtx.sampleRate} Hz)` : "";
  setStatus(`listening · ${LABEL[direction]}${note}`);
}

function stopCapture() {
  live = false;
  el.liveBadge.hidden = true;
  el.record.classList.remove("live");
  el.recLabel.textContent = "Start listening";
  setDot(ready ? "ready" : "");
  el.meterFill.style.width = "0%";

  if (node) { node.port.onmessage = null; node.disconnect(); node = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
    setStatus("finishing the last sentence…");
  }
}

// ------------------------------------------------------------------- wiring

el.record.onclick = () => (live ? stopCapture() : startCapture());
for (const b of el.dirButtons) b.onclick = () => requestDirection(b.dataset.dir);

el.clear.onclick = () => {
  cues.clear();
  queued.length = 0;
  showing = null;
  clearTimeout(pumpTimer);
  render();
};

el.copy.onclick = async () => {
  const lines = [];
  for (const [, c] of [...cues].sort((a, b) => a[0] - b[0])) {
    if (!c.dst) continue;
    lines.push(el.showSrc.checked && c.src ? `${c.src}\n→ ${c.dst}\n` : c.dst);
  }
  try {
    await navigator.clipboard.writeText(lines.join("\n"));
    setStatus(`copied ${lines.length} line(s)`);
  } catch {
    setStatus("clipboard unavailable", true);
  }
};

function bindToggle(input, key, bodyClass, dflt = "1") {
  input.checked = load(key, dflt) === "1";
  const apply = () => {
    document.body.classList.toggle(bodyClass, !input.checked);
    save(key, input.checked ? "1" : "0");
  };
  input.onchange = () => { apply(); input.blur(); }; // keep Space for the slides
  apply();
}
bindToggle(el.showSrc, "showSrc", "hide-src", "0");
bindToggle(el.showLat, "showLat", "hide-lat", "0");

// Space is left alone on purpose: it belongs to the presenter's slides. A
// clicked button keeps focus, and Space on a focused button re-clicks it, so
// drop focus after every click.
document.addEventListener("click", (e) => {
  const b = e.target.closest?.("button");
  if (b) b.blur();
});

window.addEventListener("keydown", (e) => {
  const typing = (e.target instanceof HTMLInputElement && e.target.type !== "checkbox")
    || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement;
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if (document.querySelector("dialog[open]")) return; // an open dialog owns the keyboard

  // Shift+↑/↓ moves the subtitles; plain arrows stay with the slides.
  if (e.shiftKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
    e.preventDefault();
    moveSubs(e.key === "ArrowUp" ? SUBS_STEP : -SUBS_STEP);
    return;
  }

  if (deck) {
    const next = ["ArrowRight", "ArrowDown", "PageDown", "Enter", "n", "N"];
    const prev = ["ArrowLeft", "ArrowUp", "PageUp", "Backspace", "p", "P"];
    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      goTo(page + (e.shiftKey ? -1 : 1));
      return;
    }
    if (next.includes(e.key)) { e.preventDefault(); goTo(page + 1); return; }
    if (prev.includes(e.key)) { e.preventDefault(); goTo(page - 1); return; }
    if (e.key === "Home") { e.preventDefault(); goTo(1); return; }
    if (e.key === "End") { e.preventDefault(); goTo(deck.pages); return; }
  }
  if (e.key === "f" || e.key === "F") toggleFullscreen();
  else if (e.key === "t" || e.key === "T") {
    const order = CYCLE.filter((d) => allowedDirections().includes(d));
    requestDirection(order[(order.indexOf(direction) + 1) % order.length]);
  }
});

// ------------------------------------------------------ translation context

const STYLE_LABEL = { formal: "saya/Anda", casual: "aku/kamu", match: "match speaker" };
let context = (() => {
  try {
    const saved = JSON.parse(load("context", "null"));
    if (saved && saved.style in STYLE_LABEL) return { style: saved.style, notes: saved.notes || "" };
  } catch { /* ignore a corrupt value */ }
  return null; // not chosen yet: the server default (ID_STYLE) applies
})();

function renderContextButton() {
  const style = context?.style || config.id_style || "formal";
  el.contextBtn.textContent = `Context · ${STYLE_LABEL[style] || style}`;
  el.contextBtn.classList.toggle("active", !!context?.notes);
  el.contextBtn.title = context?.notes ? `Notes: ${context.notes}` : "Indonesian style and notes for the translator";
}

function sendContext() {
  if (!context || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "context", style: context.style, notes: context.notes }));
}

el.contextBtn.onclick = () => {
  const style = context?.style || config.id_style || "formal";
  const radio = el.contextForm.querySelector(`input[name="style"][value="${style}"]`);
  if (radio) radio.checked = true;
  el.contextNotes.value = context?.notes || "";
  el.contextDialog.showModal();
};
el.contextCancel.onclick = () => el.contextDialog.close();
el.contextForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const style = el.contextForm.querySelector('input[name="style"]:checked')?.value || "formal";
  context = { style, notes: el.contextNotes.value.trim().slice(0, 1500) };
  save("context", JSON.stringify(context));
  el.contextDialog.close();
  renderContextButton();
  sendContext();
  setStatus(`context saved: ${STYLE_LABEL[style]}${context.notes ? " + notes" : ""}. Applies from the next sentence.`);
});

// -------------------------------------------------------- subtitle position

// Gap below the subtitles as a % of the stage height, so the position holds
// across window sizes and fullscreen.
const SUBS_MIN = 0, SUBS_MAX = 80, SUBS_STEP = 4, SUBS_DEFAULT = 4;
let subsPos = Number(load("subsPos", SUBS_DEFAULT));
if (!Number.isFinite(subsPos)) subsPos = SUBS_DEFAULT;

function applySubsPos() {
  subsPos = Math.max(SUBS_MIN, Math.min(SUBS_MAX, subsPos));
  el.stage.style.setProperty("--subs-pos", String(subsPos));
  el.subsUp.disabled = subsPos >= SUBS_MAX;
  el.subsDown.disabled = subsPos <= SUBS_MIN;
  save("subsPos", String(subsPos));
}

function moveSubs(delta) {
  subsPos += delta;
  applySubsPos();
  setStatus(`subtitles at ${subsPos}% from the bottom`);
}

// Auto-hide: a finished subtitle fades after `hideAfter` seconds unless
// something new replaces it first. 0 = never.
let hideAfter = Number(load("subsHide", "5"));
if (!Number.isFinite(hideAfter) || hideAfter < 0) hideAfter = 5;
let hideTimer = 0;

function scheduleHide(cue) {
  clearTimeout(hideTimer);
  el.subs.classList.remove("faded");
  // Only finished lines time out; one still waiting for its translation stays.
  if (!cue?.dst || hideAfter <= 0) return;
  hideTimer = setTimeout(() => el.subs.classList.add("faded"), hideAfter * 1000);
}

el.subsHide.value = String(hideAfter);
if (el.subsHide.value !== String(hideAfter)) el.subsHide.value = "5"; // unknown saved value
el.subsHide.onchange = () => {
  hideAfter = Number(el.subsHide.value);
  save("subsHide", String(hideAfter));
  el.subsHide.blur(); // keep Space for the slides
  const current = currentCue()[0];
  scheduleHide(current === undefined ? null : cues.get(current)); // restart with the new length
  setStatus(hideAfter ? `subtitles hide after ${hideAfter}s` : "subtitles stay on screen");
};

el.subsUp.onclick = () => moveSubs(SUBS_STEP);
el.subsDown.onclick = () => moveSubs(-SUBS_STEP);
applySubsPos();

// ------------------------------------------------------------------- slides

function slideUrl(n) {
  // Ask for the pixels the stage really has, bucketed like the server does.
  // A hidden tab reports 0; fall back to the screen so slides aren't blurry.
  const css = el.stage.clientWidth || screen.width || 1920;
  const px = css * (window.devicePixelRatio || 1);
  const w = Math.min(3840, Math.max(320, Math.ceil(px / 160) * 160));
  return `api/slides/${deck.id}/${n}.jpg?w=${w}`;
}

function showDeck() {
  const on = !!deck;
  el.slide.hidden = !on;
  el.deckNav.hidden = !on;
  el.stage.classList.toggle("has-slides", on);
  el.openSlides.textContent = on ? "Change slides" : "Open slides";
  render();
}

function goTo(n) {
  if (!deck) return;
  page = Math.max(1, Math.min(deck.pages, n));
  el.slide.src = slideUrl(page);
  el.slide.alt = `${deck.name}, slide ${page} of ${deck.pages}`;
  el.pageLabel.textContent = `${page} / ${deck.pages}`;
  el.prevSlide.disabled = page <= 1;
  el.nextSlide.disabled = page >= deck.pages;
  save("deck", JSON.stringify({ id: deck.id, page }));
  // Warm the neighbours so a page turn is instant.
  for (const k of [page + 1, page - 1]) {
    if (k >= 1 && k <= deck.pages) new Image().src = slideUrl(k);
  }
}

async function openPdf(file) {
  if (!file) return;
  setStatus(`loading ${file.name}…`);
  try {
    const res = await fetch("api/slides", {
      method: "POST",
      headers: { "Content-Type": "application/pdf", "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    deck = body;
    showDeck();
    goTo(1);
    setStatus(`${deck.name} · ${deck.pages} slides — Space / → next, ← previous, F fullscreen`);
  } catch (err) {
    setStatus(`could not open slides: ${err.message}`, true);
  }
}

function closeDeck() {
  deck = null;
  el.slide.removeAttribute("src");
  save("deck", "");
  if (document.fullscreenElement) document.exitFullscreen();
  showDeck();
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else el.stage.requestFullscreen().catch((err) => setStatus(`fullscreen failed: ${err.message}`, true));
}

// A reload mid-talk shouldn't lose your place: the server still has the deck
// until it restarts.
async function restoreDeck() {
  let saved;
  try { saved = JSON.parse(load("deck", "") || "null"); } catch { saved = null; }
  if (!saved?.id) return;
  try {
    const res = await fetch(`api/slides/${saved.id}`);
    if (!res.ok) { save("deck", ""); return; }
    deck = await res.json();
    showDeck();
    goTo(saved.page || 1);
  } catch { /* server not up yet; ignore */ }
}

el.openSlides.onclick = () => el.pdfInput.click();
el.pdfInput.onchange = () => { openPdf(el.pdfInput.files[0]); el.pdfInput.value = ""; };
el.prevSlide.onclick = () => goTo(page - 1);
el.nextSlide.onclick = () => goTo(page + 1);
el.closeSlides.onclick = closeDeck;
el.fullscreen.onclick = toggleFullscreen;

// Dropping a PDF anywhere on the page opens it too.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = [...(e.dataTransfer?.files || [])].find((f) => f.type === "application/pdf" || f.name.endsWith(".pdf"));
  if (file) openPdf(file);
});

// Re-render at the new size after resizing or entering/leaving fullscreen.
let resizeTimer = 0;
const rerender = () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (deck) goTo(page); }, 150);
};
window.addEventListener("resize", rerender);
document.addEventListener("fullscreenchange", () => {
  el.fullscreen.textContent = document.fullscreenElement ? "Exit fullscreen" : "Fullscreen";
  rerender();
});

window.addEventListener("beforeunload", () => { if (live) stopCapture(); });

applyDirection(direction);
render();
restoreDeck();
connect();
