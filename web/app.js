const TARGET_RATE = 16000;
const LABEL = { "id-en": "ID→EN", "en-id": "EN→ID", auto: "AUTO" };
const CYCLE = ["id-en", "en-id", "auto"]; // order the T key steps through

const $ = (id) => document.getElementById(id);
const el = {
  dot: $("dot"), status: $("status"), brand: $("brand"), cost: $("cost"),
  hint: $("hint"), tip: $("tip"), subs: $("subs"),
  record: $("record"), recLabel: $("recLabel"),
  meterFill: $("meterFill"), meterThresh: $("meterThresh"),
  debug: $("debug"),
  // Both copies of the direction switch: the top bar and the live badge.
  dirButtons: [...document.querySelectorAll("[data-dir]")],
  mode: $("mode"),
  waiting: $("waiting"), stage: $("stage"), slide: $("slide"), liveBadge: $("liveBadge"), liveDir: $("liveDir"),
  pdfInput: $("pdfInput"), openSlides: $("openSlides"), deckNav: $("deckNav"),
  prevSlide: $("prevSlide"), nextSlide: $("nextSlide"), pageLabel: $("pageLabel"),
  fullscreen: $("fullscreen"), closeSlides: $("closeSlides"),
  subsUp: $("subsUp"), subsDown: $("subsDown"), subsHide: $("subsHide"),
  contextBtn: $("contextBtn"), contextDialog: $("contextDialog"), contextForm: $("contextForm"),
  contextNotes: $("contextNotes"), contextCancel: $("contextCancel"),
  engineDialog: $("engineDialog"), engineForm: $("engineForm"), engineGo: $("engineGo"),
  work: $("work"), workTitle: $("workTitle"),
  accessDialog: $("accessDialog"), accessForm: $("accessForm"),
  passphrase: $("passphrase"), accessError: $("accessError"), accessGo: $("accessGo"),
  titleChip: $("titleChip"), titleDialog: $("titleDialog"), titleHeading: $("titleHeading"), titleForm: $("titleForm"), talkTitle: $("talkTitle"),
  titleGo: $("titleGo"), titleError: $("titleError"),
  engineError: $("engineError"), openaiFields: $("openaiFields"), openaiModel: $("openaiModel"),
  modeNote: $("modeNote"),
  openaiKey: $("openaiKey"), keyRow: $("keyRow"), keyNote: $("keyNote"), localModel: $("localModel"),
};

let ws = null, audioCtx = null, node = null, stream = null;
let live = false, ready = false, config = {};
// Empty until "hello": the server's DIRECTION decides, and only a direction the
// presenter picked themselves (stored here) overrides it.
let direction = load("direction", "");
let deck = null; // open slide deck: { id, name, pages, sizes }
let engine = {};  // speech engine state from the server (see /api/engine)
let loading = false;  // models are warming up; survives a dropped socket
// Idle watchdog. The server runs the same clock and is the authority -- this
// one just stops the microphone promptly if the socket has gone quiet.
let idleTimer = null, lastSpeechAt = 0;
let page = 1;
if (!(direction in LABEL)) direction = "id-en"; // e.g. a bad value saved by an older build
// What the user picked last, read live rather than captured at load: only an
// explicit click writes it, so an empty value means "use the server's DIRECTION".
const userPick = () => load("direction", "");

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

// Per tab, and never on disk: this is where the answers a reload shouldn't ask
// twice live -- the access phrase and which talk this tab is recording. A fresh
// tab starts over, closing the browser forgets everything.
function sload(key, fallback) {
  try { return sessionStorage.getItem("salindia." + key) ?? fallback; } catch { return fallback; }
}
function ssave(key, value) {
  try { sessionStorage.setItem("salindia." + key, value); } catch { /* private mode */ }
}
function sdrop(key) {
  try { sessionStorage.removeItem("salindia." + key); } catch { /* private mode */ }
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

// The dots stand in for a subtitle: shown only while listening, and only when
// nothing is on screen -- including after auto-hide has faded the last one.
function updateWaiting() {
  const showing = el.subs.childElementCount > 0 && !el.subs.classList.contains("faded");
  el.waiting.hidden = !live || showing;
}

function render() {
  const idle = cues.size === 0 && !deck;
  el.hint.classList.toggle("gone", !idle);
  el.tip.classList.toggle("gone", !idle);
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
  updateWaiting();
}

// -------------------------------------------------------------------- mode
//
// pipeline = transcribe, then translate with a second model. direct = one
// Whisper pass that comes out in English, so it can only do ID->EN. Both the
// speech model and the translators hang off it, so switching reloads them and
// the server sends us round for a fresh socket.

const MODE_HELP = {
  pipeline: "Transcribe, then translate with a second model. Both directions.",
  direct: "One Whisper pass straight to English. Faster and cheaper, ID → EN only.",
};

let modeBusy = false;

function syncModeSelect() {
  const reasons = engine.mode_options || {};
  const modes = config.modes || ["pipeline", "direct"];
  for (const opt of el.mode.options) {
    const known = modes.includes(opt.value);
    // Blocked rather than hidden: the reason is the useful part, e.g. that
    // OpenAI only translates directly with whisper-1.
    opt.disabled = !known || !!reasons[opt.value];
    opt.title = reasons[opt.value] || MODE_HELP[opt.value] || "";
  }
  if (config.mode) el.mode.value = config.mode;
  el.mode.disabled = modeBusy || loading;
  el.mode.title = reasons[el.mode.value] || MODE_HELP[el.mode.value] || "";
  // Direct mode is one Whisper pass: there is no translator to prompt, so the
  // style and notes have nothing to act on.
  el.contextBtn.hidden = el.mode.value === "direct";
}

function modeBlocked(value) {
  return (engine.mode_options || {})[value] || "";
}

async function requestMode(value, why = "") {
  if (value === config.mode) return;
  modeBusy = true;
  syncModeSelect();
  setStatus(why || `switching to ${value} mode — reloading models…`);
  try {
    const res = await fetch("api/mode", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ mode: value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    engine = data;
    // The socket is being closed from the other end; the watchdog reconnects
    // and "hello" brings the new config back. Nothing to do but wait.
  } catch (err) {
    setStatus(`could not switch mode: ${err.message}`, true);
  } finally {
    modeBusy = false;
    syncModeSelect();  // snaps back to config.mode if the switch was refused
  }
}

el.mode.addEventListener("change", () => requestMode(el.mode.value));

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

// Which mode a direction needs. Direct mode is one Whisper pass and that pass
// only ever comes out in English, so anything into Indonesian needs the
// pipeline -- asking for EN → ID is asking for pipeline, and says so.
function modeFor(value) {
  return value === "id-en" ? config.mode : "pipeline";
}

// Everything a click (or T) can get to from here: what this mode allows, plus
// what it would allow after a mode switch we're able to make.
function reachableDirections() {
  const allowed = allowedDirections();
  return CYCLE.filter(
    (d) => allowed.includes(d) || (d !== "auto" && !modeBlocked(modeFor(d)))
  );
}

function requestDirection(value) {
  const want = modeFor(value);
  if (want !== config.mode && !modeBlocked(want)) {
    // applyDirection stores the pick, which matters here: the mode switch drops
    // the socket, and it is the stored pick the fresh connection comes back in.
    applyDirection(value);
    requestMode(want, `${LABEL[value]} needs ${want} mode — reloading models…`);
    return;
  }
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
    const dir = b.dataset.dir;
    const ok = allowed.includes(dir);
    // Auto is the one that can't be reached by switching mode: it needs an
    // engine that detects the language. Hide it rather than show a dead button.
    if (dir === "auto") b.hidden = !ok;
    // The other two are always offered. If the current mode can't do it, the
    // click changes the mode -- the direction is the choice people care about,
    // and making them work out which mode it needs first is the wrong way round.
    const reachable = reachableDirections().includes(dir);
    b.disabled = !reachable;
    b.title = ok ? ""
      : dir === "auto" ? "Auto-detect needs the OpenAI speech engine"
      : reachable ? `Switches to ${modeFor(dir)} mode — the models reload`
      : modeBlocked(modeFor(dir));
  }
  // The user's own pick wins once it's allowed; otherwise whatever is showing,
  // otherwise the server's default. Never persisted here -- storing a fallback
  // is what made DIRECTION stick to a stale value across reloads.
  const pick = userPick();
  const want = allowed.includes(pick) ? pick
    : allowed.includes(direction) ? direction
    : (allowed.includes(config.default_direction) ? config.default_direction : allowed[0]);
  if (want) applyDirection(want, { persist: false });
}

// Running spend, pushed by the server after each sentence. Only the paid
// engines send it; the local models cost nothing, so the badge stays hidden.
function renderCost(u) {
  if (!u) { el.cost.hidden = true; return; }
  el.cost.hidden = false;
  el.cost.textContent = u.known
    ? `Estimated cost ~$${u.usd.toFixed(3)}`
    : "Estimated cost unavailable";
  const mins = (u.audio_s / 60).toFixed(1);
  el.cost.title = u.known
    ? `Estimated this session — list prices, verify against OpenAI's billing.\n` +
      `speech    $${u.asr_usd.toFixed(3)}  (${mins} min over ${u.asr_requests} requests)\n` +
      `translate $${u.mt_usd.toFixed(3)}  (${u.in_tok} in / ${u.out_tok} out over ${u.mt_requests} requests)`
    : "No price on record for one of the chosen models — set PRICE_ASR_PER_MIN / PRICE_MT_*_PER_MTOK.";
}

// ---------------------------------------------------------- waiting dialog
//
// Only one of these is ever up: the spinner while the presenter has nothing to
// answer yet, then whichever question is actually answerable.

function closeDialogs(except) {
  for (const d of [el.accessDialog, el.engineDialog, el.titleDialog]) {
    if (d !== except && d.open) d.close();
  }
}

// Progress is a corner badge, not a dialog. It never covers the slides and it
// never steals focus, so a page turn still works while the server is away.
function showLoading(title) {
  el.workTitle.textContent = title;
  el.work.hidden = false;
}

function hideLoading() {
  el.work.hidden = true;
}

// ------------------------------------------------------------- access gate

// Both survive a dropped socket -- and a server restart, which invalidates the
// token but not the phrase, so the page can let itself back in without
// interrupting the talk. They survive a reload too, in this tab's session
// storage: re-typing the phrase because the page refreshed mid-talk is worse
// than keeping it out of reach of another tab. A phrase that has since changed
// on the server stops working on its own -- boot() checks it before using it.
let accessToken = sload("accessToken", "");
let accessPhrase = sload("accessPhrase", "");
// Set once a talk has begun. From then on every reconnection is silent: the
// presenter is mid-talk and a modal over their slides is worse than a gap.
let talkStarted = false;
// Remembered per tab, so a reload rejoins the same row instead of asking the
// presenter to name the talk again (see the "ready" handler).
let talkId = Number(sload("talkId", "")) || null;
// Was the mic live when the socket dropped? If so the presenter is still
// talking and expects it back without touching anything.
let resumeListening = false;

function authHeaders(extra = {}) {
  return accessToken ? { ...extra, "X-Salindia-Access": accessToken } : extra;
}

function openAccessDialog(message) {
  closeDialogs(el.accessDialog);
  el.accessError.hidden = !message;
  el.accessError.textContent = message || "";
  el.accessGo.disabled = false;
  el.accessGo.textContent = "Continue";
  el.passphrase.value = "";
  if (!el.accessDialog.open) el.accessDialog.showModal();
  el.passphrase.focus();
}

// Trades a phrase for a token, and remembers both. Throws with the server's
// own wording -- a wrong phrase and a changed one read the same from here.
async function authenticate(phrase) {
  const res = await fetch("api/access", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ passphrase: phrase }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  accessToken = data.token || "";
  accessPhrase = phrase;
  ssave("accessToken", accessToken);
  ssave("accessPhrase", accessPhrase);
}

function forgetAccess() {
  accessToken = "";
  accessPhrase = "";
  sdrop("accessToken");
  sdrop("accessPhrase");
}

el.accessDialog.addEventListener("cancel", (e) => e.preventDefault());
el.accessForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  el.accessGo.disabled = true;
  el.accessGo.textContent = "Checking…";
  el.accessError.hidden = true;
  try {
    await authenticate(el.passphrase.value);
    el.accessDialog.close();
    showLoading("Connecting…");
    connect();
  } catch (err) {
    el.accessGo.disabled = false;
    el.accessGo.textContent = "Continue";
    el.accessError.hidden = false;
    el.accessError.textContent = err.message;
    el.passphrase.focus();
    el.passphrase.select();
  }
});

// Reach the server before asking the presenter for anything. A passphrase box
// on a page that cannot talk to the backend just turns a server problem into a
// wrong-password one, so nothing is shown until /api/bootstrap answers.
// Retries forever with a capped backoff: reloading while the server restarts is
// a normal thing to do mid-talk.
async function boot() {
  showLoading("Contacting the server…");
  let info = null;
  for (let attempt = 0; info === null; attempt++) {
    try {
      const res = await fetch("api/bootstrap", { cache: "no-store" });
      if (!res.ok) throw new Error(res.statusText);
      info = await res.json();
    } catch {
      showLoading("Waiting for the server…");
      const wait = Math.min(5000, 400 * 2 ** Math.min(attempt, 4));
      await new Promise((r) => setTimeout(r, wait));
    }
  }
  // The passphrase has to come first: it is what opens the socket that would
  // tell us anything about the models.
  if (info.needs_passphrase) {
    // A reload is not a new visitor. Spend the phrase this tab already has --
    // which also replaces a token a server restart threw away. Only if the
    // server no longer accepts it (someone changed ACCESS_PASSPHRASE) is there
    // anything to ask about, and then the server's reason is worth showing.
    if (accessPhrase) {
      try {
        await authenticate(accessPhrase);
      } catch (err) {
        forgetAccess();
        openAccessDialog(err.message);
        return;
      }
    } else if (!accessToken) {
      openAccessDialog();
      return;
    }
  }
  showLoading("Connecting…");
  connect();
}

// -------------------------------------------------------------- talk title

let talkTitle = sload("talkTitle", "");

// The bar shows it once a talk is under way, and clicking it comes back here:
// a mistyped title should not mean losing the talk's record.
function renderTalkTitle() {
  el.titleChip.hidden = !talkStarted || !talkTitle;
  el.titleChip.textContent = talkTitle;
  el.titleChip.title = `${talkTitle} — click to rename this talk`;
}

function openTitleDialog() {
  closeDialogs(el.titleDialog);
  el.titleError.hidden = true;
  el.titleGo.disabled = false;
  el.titleHeading.textContent = talkStarted ? "Rename this talk" : "What are you presenting?";
  el.titleGo.textContent = talkStarted ? "Rename" : "Start";
  el.talkTitle.value = talkTitle || load("lastTitle", "");
  if (!el.titleDialog.open) el.titleDialog.showModal();
  el.talkTitle.focus();
  el.talkTitle.select();
}

function closeTitleDialog() {
  if (el.titleDialog.open) el.titleDialog.close();
}

el.titleDialog.addEventListener("cancel", (e) => e.preventDefault());
el.titleForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const title = el.talkTitle.value.trim();
  if (!title) {
    el.titleError.hidden = false;
    el.titleError.textContent = "Give the talk a title so its cost can be tracked.";
    el.talkTitle.focus();
    return;
  }
  talkTitle = title;
  save("lastTitle", title);
  ssave("talkTitle", title);
  el.titleGo.disabled = true;
  if (talkStarted) {
    // Already recording: this is a rename, not a new talk.
    renderTalkTitle();
    closeTitleDialog();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "rename", title }));
    }
    return;
  }
  ws.send(JSON.stringify({ type: "begin", title }));
});

el.titleChip.onclick = () => openTitleDialog();

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

// Which mode this choice will land in, said before it happens. Whatever the
// engine, that is the server's MODE -- pipeline unless its environment says
// otherwise -- so the note only has to say which one that is.
function plannedMode() {
  return engine.env_mode || "pipeline";
}

function syncEngineFields() {
  const choice = chosenEngine();
  el.openaiFields.hidden = choice !== "openai";
  el.modeNote.hidden = choice !== "openai";
  el.modeNote.textContent = plannedMode() === "direct"
    ? "Starts in direct mode: one Whisper pass straight to English, "
      + "ID → EN only. Switch to pipeline in the top bar for EN → ID."
    : "Starts in pipeline mode: transcribe, then translate with "
      + "a chat model. Both directions. Direct is in the top bar: "
      + "whisper-1 translates in the same pass there, cheaper and ID → EN only.";
}

function openEngineDialog() {
  closeDialogs(el.engineDialog);
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
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    engine = data;
    if (data.phase === "ready") {
      // Already loaded (a refresh): no "loading" will arrive over the socket.
      closeEngineDialog();
      renderMeta();
      openTitleDialog();
    }
    // Otherwise the WebSocket reports "loading", then "ready" or "choose" + error.
  } catch (err) {
    el.engineGo.disabled = false;
    el.engineGo.textContent = "Continue";
    el.engineError.hidden = false;
    el.engineError.textContent = err.message;
  }
});

// ---------------------------------------------------------------- websocket

// Reconnection is supervised rather than chained. The old version scheduled the
// next attempt from onclose, so any throw before that handler was attached --
// or a re-auth fetch that hung instead of failing -- left the page with no
// socket and no pending retry, dead until someone reloaded it. Now a watchdog
// owns the invariant "if we want a connection and haven't got one, make one".
let wantConnection = false;
let connecting = false;
let reconnectTimer = null;
let reconnectAttempt = 0;

function socketUsable() {
  return ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING);
}

function scheduleReconnect() {
  if (!wantConnection) return;
  clearTimeout(reconnectTimer);
  const wait = Math.min(5000, 500 * 2 ** Math.min(reconnectAttempt++, 3));
  reconnectTimer = setTimeout(connect, wait);
}

async function connect() {
  wantConnection = true;
  if (connecting || socketUsable()) return;
  connecting = true;
  try {
    // A restarted server forgets every token. Swap the remembered phrase for a
    // fresh one before opening the socket, so the reconnect just works.
    if (!accessToken && accessPhrase) {
      try {
        const res = await fetch("api/access", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ passphrase: accessPhrase }),
          // Never let an unreachable host stall the loop: a dropped packet can
          // leave fetch pending far longer than a refused connection does.
          signal: AbortSignal.timeout(5000),
        });
        if (res.ok) accessToken = (await res.json()).token;
      } catch { /* still down; the watchdog tries again */ }
    }
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    // Browsers can't set headers on a WebSocket, so the token rides the query.
    const q = accessToken ? `?access=${encodeURIComponent(accessToken)}` : "";
    ws = new WebSocket(`${proto}//${location.host}/ws${q}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      reconnectAttempt = 0;
      setStatus("connected, checking models…");
    };
    ws.onclose = async (ev) => {
      ready = false;
      setDot("");
      el.record.disabled = true;
      if (live) {
        resumeListening = true;
        stopCapture();
      }

      // Mid-talk, a dropped server is a status-bar matter. Never a dialog: the
      // slides are on a projector and a modal over them is the worst outcome.
      // 4409: the mode changed and the models are being reloaded. Expected,
      // and the reconnect is the point -- don't cry about an unreachable server.
      const reloading = ev.code === 4409;
      if (talkStarted) {
        closeDialogs(null);
        showLoading(reloading ? "Reloading models…" : "Reconnecting…");
        el.recLabel.textContent = reloading ? "Reloading models…" : "Reconnecting…";
        setStatus(
          reloading ? "mode changed — reloading models…" : "server unreachable — reconnecting…",
          !reloading,
        );
        // The token died with the server; the phrase didn't, so let ourselves
        // back in rather than asking again.
        if (ev.code === 4401 && accessPhrase) { accessToken = ""; sdrop("accessToken"); }
        scheduleReconnect();
        return;
      }

      // Before a talk has begun there is nothing to interrupt, so the setup
      // dialogs can do their normal thing.
      if (ev.code === 4401) {
        accessToken = "";
        sdrop("accessToken");
        if (!accessPhrase) { openAccessDialog(); return; }
      }
      const busy = loading || reloading;
      showLoading(busy ? "Loading models…" : "Reconnecting…");
      el.recLabel.textContent = busy ? "Loading models…" : "Disconnected";
      setStatus(busy ? "loading models on the server…" : "reconnecting…", !busy);
      scheduleReconnect();
    };
    ws.onmessage = (ev) => handle(JSON.parse(ev.data));
  } catch {
    // new WebSocket() can throw outright; without this the loop would stop.
    scheduleReconnect();
  } finally {
    connecting = false;
  }
}

// The watchdog: independent of any handler firing, and the thing that makes a
// stalled or never-started attempt recover on its own.
setInterval(() => {
  if (wantConnection && !connecting && !socketUsable()) connect();
}, 3000);

// Come back at once rather than waiting out a backoff: a hidden tab has its
// timers throttled to a minute or more, which mid-talk is an eternity.
window.addEventListener("online", () => { if (wantConnection) connect(); });
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && wantConnection && !socketUsable()) connect();
});

function handle(msg) {
  switch (msg.type) {
    case "hello": {
      config = msg.config || {};
      engine = msg.engine || {};
      if (!userPick()) applyDirection(config.default_direction || "en-id", { persist: false });
      renderMeta();
      renderCost(null);
      renderContextButton();
      syncDirectionButtons();
      syncModeSelect();
      break;
    }

    case "choose":
      engine = msg.engine || engine;
      loading = false;
      syncModeSelect();
      if (config.force_approach) {
        // The server is configured to pick; it only lands here if that failed.
        setStatus(engine.error || `FORCE_APPROACH=${config.force_approach} could not start`, true);
        break;
      }
      openEngineDialog();
      setStatus("choose a speech engine to start");
      break;

    case "loading":
      engine = msg.engine || engine;
      closeEngineDialog();
      renderMeta();
      loading = true;
      syncModeSelect();
      setDot("loading");
      el.record.disabled = true;
      // The server names the step it is on; a first run sits on one of these
      // for minutes, so show it rather than a generic "loading".
      // The step text lives in the dialog; a long label here stretches the bar.
      showLoading(engine.detail || "Loading models…");
      el.recLabel.textContent = "Loading models…";
      setStatus(engine.detail || "loading models — the first run downloads weights, this can take a few minutes…");
      break;

    case "ready":
      engine = msg.engine || engine;
      // The mode can have changed under us (the bar, or another page), and it
      // decides the directions and the models, so take the fresh copy.
      config = msg.config || config;
      closeEngineDialog();
      renderMeta();
      syncDirectionButtons();
      loading = false;
      syncModeSelect();
      renderContextButton();
      setDot("");
      hideLoading();
      if (talkStarted || talkId) {
        // Reconnected mid-talk, or reloaded: pick the same talk back up
        // silently. If the row has gone the server opens a new one under the
        // title we send along, so this never leaves the page stuck.
        ws.send(JSON.stringify({ type: "begin", title: talkTitle, talk_id: talkId }));
        break;
      }
      // Models can take audio now, so the questions are finally answerable.
      if (config.force_approach) {
        openTitleDialog();
      } else if (!el.titleDialog.open) {
        openEngineDialog();
      }
      break;

    case "begun":
      talkStarted = true;
      talkId = msg.talk_id ?? talkId;
      if (talkId) ssave("talkId", String(talkId));
      renderTalkTitle();
      closeTitleDialog();
      hideLoading();
      ready = true;
      renderCost(null);
      setDot("ready");
      el.record.disabled = false;
      el.recLabel.textContent = "Start listening";
      // Tell this new connection which direction the page is showing.
      ws.send(JSON.stringify({ type: "direction", value: direction }));
      sendContext();
      setStatus("ready");
      // The mic was open when the connection went: pick it straight back up.
      // getUserMedia is already granted for this page, so there is no prompt.
      if (resumeListening) {
        resumeListening = false;
        startCapture().catch(() => setStatus("could not resume the microphone — press Start listening", true));
      }
      break;

    case "usage":
      renderCost(msg);
      break;

    case "direction":
      applyDirection(msg.value);
      break;

    case "context":
      context = { style: msg.style, notes: msg.notes };
      renderContextButton();
      break;

    case "renamed":
      talkTitle = msg.title || talkTitle;
      ssave("talkTitle", talkTitle);
      renderTalkTitle();
      break;

    case "level": {
      // RMS is small and perceived logarithmically; stretch it to be visible.
      el.meterFill.style.width = Math.min(100, (msg.rms / 0.08) * 100).toFixed(1) + "%";
      el.meterFill.classList.toggle("speaking", msg.speaking);
      const t = Math.min(100, (Math.max(0.006, msg.floor * 3) / 0.08) * 100);
      el.meterThresh.style.left = t.toFixed(1) + "%";
      break;
    }

    case "idle_stop":
      // The server stopped listening; match it so the mic light goes out.
      if (live) stopCapture();
      setStatus(`stopped after ${msg.after_s}s with no speech — press Start listening to resume`);
      break;

    case "speech_start":
      noteSpeech();
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
      setStatus("stopped");
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
  updateWaiting();
  // Clears a server-side idle stop; harmless when there wasn't one.
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "resume" }));
  startIdleWatch();
  el.liveBadge.hidden = false;
  setDot("live");
  el.record.classList.add("live");
  el.recLabel.textContent = "Stop";
  const note = audioCtx.sampleRate !== TARGET_RATE ? ` (resampling from ${audioCtx.sampleRate} Hz)` : "";
  setStatus(`listening${note}`);
}

function stopCapture() {
  live = false;
  updateWaiting();
  stopIdleWatch();
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

// ------------------------------------------------------------------ idle stop

function noteSpeech() { lastSpeechAt = Date.now(); }

function startIdleWatch() {
  stopIdleWatch();
  noteSpeech();
  const limit = (config.idle_stop_s || 0) * 1000;
  if (limit <= 0) return;
  idleTimer = setInterval(() => {
    if (!live) return;
    if (Date.now() - lastSpeechAt < limit) return;
    stopCapture();
    setStatus(`stopped after ${Math.round(limit / 1000)}s with no speech — press Start listening to resume`);
  }, 1000);
}

function stopIdleWatch() {
  if (idleTimer) { clearInterval(idleTimer); idleTimer = null; }
}

// ------------------------------------------------------------------- wiring

el.record.onclick = () => (live ? stopCapture() : startCapture());
for (const b of el.dirButtons) b.onclick = () => requestDirection(b.dataset.dir);

// One switch for everything a presenter doesn't want on a projector: the
// original transcript and the per-stage timings.
function bindToggle(input, key, bodyClasses, dflt = "1") {
  input.checked = load(key, dflt) === "1";
  const apply = () => {
    for (const cls of bodyClasses) {
      document.body.classList.toggle(cls, !input.checked);
    }
    save(key, input.checked ? "1" : "0");
  };
  input.onchange = () => { apply(); input.blur(); }; // keep Space for the slides
  apply();
}
bindToggle(el.debug, "debug", ["hide-src", "hide-lat"], "0");

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
    const order = reachableDirections();
    if (order.length) requestDirection(order[(order.indexOf(direction) + 1) % order.length]);
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
  // Direct mode is a single Whisper pass with no translator behind it, so the
  // style and the notes have nothing to reach. Hide the button rather than
  // take notes that quietly go nowhere.
  el.contextBtn.hidden = config.mode === "direct";
  if (el.contextBtn.hidden && el.contextDialog.open) el.contextDialog.close();

  // The current style lives in the dialog and the tooltip; the button keeps a
  // fixed width so the bar doesn't reflow when the style changes.
  const style = context?.style || config.id_style || "formal";
  el.contextBtn.classList.toggle("active", !!context?.notes);
  const summary = STYLE_LABEL[style] || style;
  el.contextBtn.title = context?.notes
    ? `${summary} · notes: ${context.notes}`
    : `${summary} — Indonesian style and notes for the translator`;
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
  updateWaiting();
  // Only finished lines time out; one still waiting for its translation stays.
  if (!cue?.dst || hideAfter <= 0) return;
  hideTimer = setTimeout(() => {
    el.subs.classList.add("faded");
    updateWaiting();
  }, hideAfter * 1000);
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
//
// Rendered in the browser with a vendored pdf.js. The PDF never leaves this
// machine, and a page turn costs a canvas draw instead of an upload, a
// server-side render and a fetch -- which is the difference between a slide
// appearing with the key press and appearing a beat later.

const PDF_LIB = "vendor/pdf.min.mjs";
const PDF_WORKER = "vendor/pdf.worker.min.mjs";
const PRERENDER = 2;        // neighbours kept ready on each side
const PAGE_CACHE_MAX = 12;  // rendered bitmaps kept; a deck can be hundreds

let pdfjs = null;           // the library, imported on first use
let pdfDoc = null;          // the open PDFDocumentProxy
let pageCache = new Map();  // page number -> ImageBitmap
let renderToken = 0;        // cancels a draw the user has already moved past

async function pdfLib() {
  if (!pdfjs) {
    pdfjs = await import(`./${PDF_LIB}`);
    pdfjs.GlobalWorkerOptions.workerSrc = PDF_WORKER;
  }
  return pdfjs;
}

/** Pixels the stage really has, so the canvas is sharp without being wasteful. */
function stagePixels() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = (el.stage.clientWidth || screen.width || 1920) * dpr;
  const h = (el.stage.clientHeight || screen.height || 1080) * dpr;
  return { w, h };
}

async function renderToBitmap(n) {
  const cached = pageCache.get(n);
  if (cached) return cached;
  const page = await pdfDoc.getPage(n);
  const { w, h } = stagePixels();
  const base = page.getViewport({ scale: 1 });
  // Fit inside the stage, never upscale past what the stage can show.
  const scale = Math.min(w / base.width, h / base.height);
  const viewport = page.getViewport({ scale });
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.ceil(viewport.width));
  canvas.height = Math.max(1, Math.ceil(viewport.height));
  const ctx = canvas.getContext("2d", { alpha: false });
  // Paper is white. Plenty of decks (LaTeX, plain text) paint no background of
  // their own, and an opaque canvas starts black -- black text on black.
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({ canvasContext: ctx, viewport, background: "#ffffff" }).promise;
  const bitmap = await createImageBitmap(canvas);
  pageCache.set(n, bitmap);
  // Drop the least recently added once the cache is full.
  while (pageCache.size > PAGE_CACHE_MAX) {
    const [oldest] = pageCache.keys();
    pageCache.get(oldest)?.close?.();
    pageCache.delete(oldest);
  }
  return bitmap;
}

function paint(bitmap) {
  const c = el.slide;
  if (c.width !== bitmap.width || c.height !== bitmap.height) {
    c.width = bitmap.width;
    c.height = bitmap.height;
  }
  c.getContext("2d", { alpha: false }).drawImage(bitmap, 0, 0);
}

function showDeck() {
  const on = !!deck;
  el.slide.hidden = !on;
  el.deckNav.hidden = !on;
  el.stage.classList.toggle("has-slides", on);
  el.openSlides.textContent = on ? "Change slides" : "Open slides";
  render();
}

async function goTo(n) {
  if (!deck || !pdfDoc) return;
  page = Math.max(1, Math.min(deck.pages, n));
  el.slide.setAttribute("aria-label", `${deck.name}, slide ${page} of ${deck.pages}`);
  el.pageLabel.textContent = `${page} / ${deck.pages}`;
  el.prevSlide.disabled = page <= 1;
  el.nextSlide.disabled = page >= deck.pages;
  save("deck", JSON.stringify({ name: deck.name, page }));

  const token = ++renderToken;
  const wanted = page;
  try {
    const bitmap = await renderToBitmap(wanted);
    if (token !== renderToken) return;  // the user turned again mid-render
    paint(bitmap);
  } catch (err) {
    if (token === renderToken) setStatus(`could not draw slide ${wanted}: ${err.message}`, true);
    return;
  }
  // Warm the neighbours so the next turn is a straight blit.
  for (let d = 1; d <= PRERENDER; d++) {
    for (const k of [wanted + d, wanted - d]) {
      if (k >= 1 && k <= deck.pages && !pageCache.has(k)) {
        renderToBitmap(k).catch(() => {});
      }
    }
  }
}

async function loadPdfBytes(bytes, name) {
  const lib = await pdfLib();
  // pdf.js takes ownership of the buffer it is given, so hand it a copy and
  // keep the original for IndexedDB.
  pdfDoc = await lib.getDocument({ data: bytes.slice(0) }).promise;
  for (const b of pageCache.values()) b.close?.();
  pageCache = new Map();
  deck = { name, pages: pdfDoc.numPages };
}

async function openPdf(file) {
  if (!file) return;
  setStatus(`loading ${file.name}…`);
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    await loadPdfBytes(bytes, file.name);
    await saveDeckBytes(bytes, file.name);
    showDeck();
    await goTo(1);
    setStatus(`${deck.name} · ${deck.pages} slides — Space / → next, ← previous, F fullscreen`);
  } catch (err) {
    setStatus(`could not open slides: ${err.message}`, true);
  }
}

function closeDeck() {
  deck = null;
  pdfDoc?.destroy?.();
  pdfDoc = null;
  for (const b of pageCache.values()) b.close?.();
  pageCache = new Map();
  el.slide.getContext("2d")?.clearRect(0, 0, el.slide.width, el.slide.height);
  save("deck", "");
  clearDeckBytes();
  if (document.fullscreenElement) document.exitFullscreen();
  showDeck();
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else el.stage.requestFullscreen().catch((err) => setStatus(`fullscreen failed: ${err.message}`, true));
}

// -- the PDF itself, kept in this browser so a reload doesn't lose it --------
//
// A File can't be re-read after a refresh without asking the presenter to pick
// it again, and mid-talk that is the last thing they want to do.

const DECK_DB = "salindia-decks";

function deckStore(mode) {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DECK_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore("decks");
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const db = req.result;
      resolve(db.transaction("decks", mode).objectStore("decks"));
    };
  });
}

async function saveDeckBytes(bytes, name) {
  try {
    const store = await deckStore("readwrite");
    store.put({ bytes, name }, "current");
  } catch { /* private mode, or storage refused: the deck just won't survive */ }
}

async function clearDeckBytes() {
  try { (await deckStore("readwrite")).delete("current"); } catch { /* ignore */ }
}

async function restoreDeck() {
  let saved;
  try { saved = JSON.parse(load("deck", "") || "null"); } catch { saved = null; }
  if (!saved) return;
  try {
    const store = await deckStore("readonly");
    const rec = await new Promise((resolve, reject) => {
      const r = store.get("current");
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
    if (!rec?.bytes) { save("deck", ""); return; }
    await loadPdfBytes(rec.bytes, rec.name || saved.name || "slides.pdf");
    showDeck();
    await goTo(saved.page || 1);
  } catch {
    save("deck", "");  // unreadable: start clean rather than half-open
  }
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
  resizeTimer = setTimeout(() => {
    if (!deck) return;
    // Cached bitmaps were rasterised for the old stage; at a new size they
    // would be upscaled and soft, so throw them away and draw again.
    for (const b of pageCache.values()) b.close?.();
    pageCache = new Map();
    goTo(page);
  }, 150);
};
window.addEventListener("resize", rerender);
document.addEventListener("fullscreenchange", () => {
  el.fullscreen.textContent = document.fullscreenElement ? "Exit fullscreen" : "Fullscreen";
  rerender();
});

window.addEventListener("beforeunload", () => { if (live) stopCapture(); });

if (direction) applyDirection(direction, { persist: false });
render();
restoreDeck();
boot();
