"""Exercise the VAD + segmentation state machine with stubbed models.

Synthetic audio only -- this checks that speech is cut into utterances at the
right places and that events arrive in the right order, which is where the
streaming logic can actually go wrong. Model quality is a separate concern.
"""

import asyncio
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from server.asr import AsrResult, _looks_hallucinated
from server.config import SAMPLE_RATE, Settings
from server.session import StreamSession
from server.translate import MtResult

rng = np.random.default_rng(0)
SILENCE_TAIL = Settings().silence_ms / 1000  # trailing silence kept in a final


def silence(ms):
    return (rng.standard_normal(SAMPLE_RATE * ms // 1000) * 0.0008).astype(np.float32)


def speech(ms, amp=0.12):
    return (rng.standard_normal(SAMPLE_RATE * ms // 1000) * amp).astype(np.float32)


class FakeAsr:
    def __init__(self, delay=lambda seconds, final: 0.002, script=None):
        # script: optional list of (text, api_language) returned by successive
        # final decodes, to simulate what an auto-detecting engine hears.
        self.script = list(script or [])
        self.calls = []
        self.languages = []
        self.delay = delay
        self.timeline = []

    async def run(self, audio, *, language="id", initial_prompt=None, final=False):
        secs = len(audio) / SAMPLE_RATE
        self.timeline.append(("asr_start", round(secs, 1), final))
        await asyncio.sleep(self.delay(secs, final))
        self.timeline.append(("asr_end", round(secs, 1), final))
        self.calls.append(("final" if final else "partial", len(audio) / SAMPLE_RATE))
        self.languages.append(language)
        text, lang = f"kalimat {'akhir' if final else 'sementara'}", "id"
        if final and self.script:
            text, lang = self.script.pop(0)
        return AsrResult(
            text=text,
            language=lang,
            no_speech_prob=0.01,
            avg_logprob=-0.2,
            elapsed=0.01,
        )


class FakeTranslator:
    def __init__(self, delay=0.002):
        self.delay = delay
        self.directions = []
        self.contexts = []

    async def translate(self, text, history=None, direction="id-en", style="match", notes=""):
        await asyncio.sleep(self.delay)
        self.directions.append(direction)
        self.contexts.append((style, notes))
        return MtResult(
            text=f"{direction}[{text}] ctx={len(history or [])}", elapsed=0.01
        )


async def run_case(stream, cfg, switches=None, asr=None, translator=None, directions=None):
    """``switches`` maps a sample offset to a direction to switch to there."""
    events = []

    async def send(payload):
        events.append(payload)

    asr = asr or FakeAsr()
    translator = translator or FakeTranslator()
    session = StreamSession(
        send, asr, {"id-en": translator, "en-id": translator}, cfg, directions=directions
    )
    await session.start()

    # 64 ms chunks, matching what the worklet sends.
    chunk = 1024
    pending = sorted((switches or {}).items())
    for i in range(0, len(stream), chunk):
        while pending and pending[0][0] <= i:
            assert session.set_direction(pending.pop(0)[1])
        await session.feed(stream[i : i + chunk])
        # Pace roughly like real audio so inference tasks have room to finish;
        # feeding flat out would make every partial arrive after its utterance.
        await asyncio.sleep(0.004)
    await asyncio.sleep(0.1)
    await session.flush()
    await session.close()
    run_case.last_session = session
    return events, asr, translator


def kinds(events, *want):
    return [e for e in events if e["type"] in want]


async def main():
    cfg = Settings()
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")

    # --- two utterances separated by a clear pause -------------------------
    print("case: two utterances separated by 1s of silence")
    stream = np.concatenate([
        silence(500), speech(1800), silence(1000), speech(1500), silence(800)
    ])
    events, asr, _ = await run_case(stream, cfg)

    starts = kinds(events, "speech_start")
    finals = kinds(events, "final")
    trans = kinds(events, "translation")
    partials = kinds(events, "partial")

    check("two speech_start events", len(starts) == 2, f"got {len(starts)}")
    check("two finals", len(finals) == 2, f"got {len(finals)}")
    check("two translations", len(trans) == 2, f"got {len(trans)}")
    check("partials emitted during speech", len(partials) >= 2, f"got {len(partials)}")
    check("finals carry source text, translation comes later",
          all(f["src"] and f["dst"] is None for f in finals))
    check("default direction is id-en with Indonesian ASR",
          all(f["dir"] == "id-en" for f in finals) and set(asr.languages) == {"id"})
    check("final ids are ordered", [f["id"] for f in finals] == sorted(f["id"] for f in finals),
          str([f["id"] for f in finals]))
    check("translation ids match final ids",
          [t["id"] for t in trans] == [f["id"] for f in finals])
    check("translator received prior context on 2nd turn",
          "ctx=1" in trans[1]["dst"], trans[1]["dst"])
    check("idle emitted after flush", events[-1]["type"] == "idle")

    order = [e["type"] for e in events if e["type"] in
             ("speech_start", "speech_end", "final", "translation")]
    check("per-utterance event order",
          order == ["speech_start", "speech_end", "final", "translation"] * 2, str(order))

    finals_audio = [d for k, d in asr.calls if k == "final"]
    check("final audio ~ utterance length + preroll + trailing silence",
          all(1.8 <= d <= 3.6 for d in finals_audio),
          f"{[round(d,2) for d in finals_audio]}s")

    partial_audio = [d for k, d in asr.calls if k == "partial"]
    check("partials never exceed the window cap",
          all(d <= cfg.partial_window_s + 0.1 for d in partial_audio),
          f"max {max(partial_audio, default=0):.2f}s")

    # --- a blip too short to be worth decoding -----------------------------
    print("\ncase: 150ms blip is discarded")
    events, asr, _ = await run_case(np.concatenate([silence(400), speech(150), silence(900)]), cfg)
    check("no final for a sub-threshold blip", len(kinds(events, "final")) == 0)

    # --- a monologue longer than the forced-cut limit ----------------------
    print("\ncase: continuous speech past MAX_UTTERANCE_S is force-cut")
    short = Settings(max_utterance_s=3.0)
    events, asr, _ = await run_case(
        np.concatenate([silence(300), speech(9000), silence(900)]), short
    )
    finals = kinds(events, "final")
    check("long monologue split into several finals", len(finals) >= 3, f"got {len(finals)}")
    check("ids still unique and ordered",
          [f["id"] for f in finals] == sorted(set(f["id"] for f in finals)),
          str([f["id"] for f in finals]))
    check("no final exceeds the cap",
          all(d <= 3.3 for k, d in asr.calls if k == "final"),
          f"{[round(d,2) for k,d in asr.calls if k=='final']}")

    # --- adaptive segmentation ----------------------------------------------
    print("\ncase: a short breath only ends an utterance once it is long")
    events, asr, _ = await run_case(np.concatenate([
        silence(400), speech(4000), silence(300),   # 4 s in: breath ignored
        speech(3000), silence(300),                 # 7.3 s in: breath cuts
        speech(2000), silence(1000),
    ]), Settings(speculate_after_ms=0))
    finals = [round(d, 1) for k, d in asr.calls if k == "final"]
    check("two utterances, split at the second breath",
          len(kinds(events, "final")) == 2, f"final lengths {finals}s")
    check("first utterance holds both early chunks", finals and finals[0] > 7, str(finals))

    print("\ncase: forced cut lands in the quiet dip, not at the cap")
    events, asr, _ = await run_case(np.concatenate([
        silence(400), speech(11000), silence(120), speech(3000), silence(1000),
    ]), Settings(speculate_after_ms=0, soft_max_s=30, max_utterance_s=12))
    finals = [round(d, 2) for k, d in asr.calls if k == "final"]
    check("two finals", len(finals) == 2, str(finals))
    check("first cut at the dip (~11.3 s incl. preroll), not at 12 s",
          finals and 11.0 <= finals[0] <= 11.6, str(finals))
    check("nothing lost across the cut",
          abs(sum(finals) - (0.3 + 11 + 0.12 + 3 + SILENCE_TAIL)) < 0.3, str(finals))

    # --- switching direction ------------------------------------------------
    print("\ncase: direction switch applies from the next utterance")
    stream = np.concatenate([
        silence(500), speech(2000), silence(1000),   # utterance 1: id-en
        speech(1500), silence(1000),                 # utterance 2: en-id
        speech(1500), silence(900),                  # utterance 3: en-id
    ])
    sr = SAMPLE_RATE
    # Switch half-way through utterance 1: it must stay id-en.
    events, asr, translator = await run_case(
        stream, cfg, switches={int(1.5 * sr): "en-id"}
    )
    finals = kinds(events, "final")
    trans = kinds(events, "translation")
    check("three finals", len(finals) == 3, f"got {len(finals)}")
    check("mid-utterance switch doesn't change that utterance",
          [f["dir"] for f in finals] == ["id-en", "en-id", "en-id"],
          str([f["dir"] for f in finals]))
    check("translator called with matching directions",
          translator.directions == ["id-en", "en-id", "en-id"], str(translator.directions))
    check("ASR finals pinned to the spoken language",
          [lang for (k, _), lang in zip(asr.calls, asr.languages) if k == "final"]
          == ["id", "en", "en"])
    check("partials use the utterance's language too",
          all(e["dir"] in ("id-en", "en-id") for e in kinds(events, "partial")))
    check("history kept per direction (first en-id turn has no context)",
          "ctx=0" in trans[1]["dst"] and "ctx=1" in trans[2]["dst"],
          f'{trans[1]["dst"]} | {trans[2]["dst"]}')

    print("\ncase: unsupported direction is rejected")
    direct = Settings(mode="direct")
    session = StreamSession(lambda e: asyncio.sleep(0), FakeAsr(), {}, direct)
    check("direct mode refuses en-id", not session.set_direction("en-id"))
    check("direct mode keeps id-en", session.direction == "id-en")
    check("garbage direction refused", not StreamSession(
        lambda e: asyncio.sleep(0), FakeAsr(), {}, cfg).set_direction("fr-en"))

    # --- speculation ----------------------------------------------------------
    print("\ncase: speculative decode is reused as the final")
    events, asr, _ = await run_case(
        np.concatenate([silence(500), speech(1800), silence(1000)]), cfg
    )
    finals_asr = [c for c in asr.calls if c[0] == "final"]
    stats = run_case.last_session.stats
    check("one final", len(kinds(events, "final")) == 1)
    check("final ASR ran exactly once (speculation reused)",
          len(finals_asr) == 1, f"{len(finals_asr)} final-quality decodes")
    check("final marked as speculated", kinds(events, "final")[0]["speculated"])
    check("speculation hit counted", stats["speculation_hits"] == 1, str(stats))

    print("\ncase: speaker resumes after a short pause")
    events, asr, _ = await run_case(
        np.concatenate([silence(500), speech(1200), silence(400),
                        speech(1200), silence(1000)]), cfg
    )
    finals = kinds(events, "final")
    stats = run_case.last_session.stats
    check("pause shorter than SILENCE_MS stays one utterance", len(finals) == 1,
          f"got {len(finals)}")
    check("stale speculation discarded, fresh one used",
          stats["speculation_hits"] == 1 and len([c for c in asr.calls if c[0] == "final"]) == 2,
          f'{stats}, {len([c for c in asr.calls if c[0] == "final"])} final decodes')
    check("final covers both halves of the sentence",
          [d for k, d in asr.calls if k == "final"][-1] >= 3.0,
          f'{[d for k, d in asr.calls if k == "final"]}')

    print("\ncase: speculation disabled falls back to a normal decode")
    events, asr, _ = await run_case(
        np.concatenate([silence(500), speech(1800), silence(1000)]),
        Settings(speculate_after_ms=0),
    )
    check("still one final", len(kinds(events, "final")) == 1)
    check("not marked speculated", not kinds(events, "final")[0]["speculated"])

    # --- parallel pipelines --------------------------------------------------
    print("\ncase: parallel pipelines keep output in order")
    # First sentence is long and slow to decode, second short and fast, so the
    # second finishes ASR first if nothing enforces order.
    slow_first = FakeAsr(delay=lambda secs, final: 0.6 if (final and secs > 3) else 0.01)
    stream = np.concatenate([
        silence(400), speech(3500), silence(800),
        speech(700), silence(900),
    ])
    events, asr, _ = await run_case(
        stream, Settings(speculate_after_ms=0, final_workers=3),
        asr=slow_first, translator=FakeTranslator(delay=0.3),
    )
    finals = kinds(events, "final")
    trans = kinds(events, "translation")
    check("both sentences finalised", len(finals) == 2 and len(trans) == 2)
    check("source lines emitted in order", [f["id"] for f in finals] == [1, 2],
          str([f["id"] for f in finals]))
    check("translations emitted in order", [t["id"] for t in trans] == [1, 2],
          str([t["id"] for t in trans]))
    ends = [(i, e) for i, e in enumerate(asr.timeline) if e[0] == "asr_end" and e[2]]
    check("sentence 2 finished ASR before sentence 1 (really concurrent)",
          ends[0][1][1] < 3, str(ends))

    print("\ncase: FINAL_WORKERS=1 is strictly sequential")
    one = FakeAsr(delay=lambda secs, final: 0.2 if final else 0.01)
    events, asr, _ = await run_case(
        stream, Settings(speculate_after_ms=0, final_workers=1),
        asr=one, translator=FakeTranslator(delay=0.2),
    )
    finals_tl = [e for e in asr.timeline if e[2]]
    check("no overlapping final decodes",
          all(finals_tl[i][0] != finals_tl[i + 1][0] for i in range(len(finals_tl) - 1)),
          str(finals_tl))
    check("order preserved", [t["id"] for t in kinds(events, "translation")] == [1, 2])

    # --- auto direction -------------------------------------------------------
    print("\ncase: auto direction follows the language actually spoken")
    heard = FakeAsr(script=[
        ("Selamat pagi semuanya, kita mulai ya.", "malay"),        # Whisper's Malay mix-up
        ("Have you read the contract I sent you?", "english"),
        ("Oke.", None),                                            # no clue: keep last language
    ])
    stream = np.concatenate([
        silence(500), speech(1800), silence(1000),
        speech(1500), silence(1000),
        speech(700), silence(1000),
    ])
    events, asr, translator = await run_case(
        stream, Settings(speculate_after_ms=0), asr=heard,
        switches={0: "auto"}, directions=("id-en", "en-id", "auto"),
    )
    finals = kinds(events, "final")
    trans = kinds(events, "translation")
    check("three sentences", len(finals) == 3 and len(trans) == 3, f"{len(finals)}/{len(trans)}")
    check("ASR asked to detect the language (no hint)",
          set(asr.languages) == {None}, str(set(asr.languages)))
    check("directions resolved per sentence",
          [f["dir"] for f in finals] == ["id-en", "en-id", "en-id"], str([f["dir"] for f in finals]))
    check("events flagged as auto", all(f["auto"] for f in finals) and all(t["auto"] for t in trans))
    check("translator got the resolved directions",
          translator.directions == ["id-en", "en-id", "en-id"], str(translator.directions))
    check("speech_start reports auto", {e["dir"] for e in kinds(events, "speech_start")} == {"auto"})

    print("\ncase: auto is refused unless the engine offers it")
    plain = StreamSession(lambda e: asyncio.sleep(0), FakeAsr(), {}, cfg)
    check("local-engine session refuses auto", not plain.set_direction("auto"))
    offered = StreamSession(lambda e: asyncio.sleep(0), FakeAsr(), {}, cfg,
                            directions=("id-en", "en-id", "auto"))
    check("openai-engine session accepts auto", offered.set_direction("auto"))

    # --- translation context ----------------------------------------------------
    print("\ncase: presenter context reaches the translator")
    probe = StreamSession(lambda e: asyncio.sleep(0), FakeAsr(), {}, cfg)
    check("default style comes from settings", probe.style == cfg.id_style, probe.style)
    ctx = probe.set_context("casual", "  Quarterly results. CEO is Pak Budi.  ")
    check("style and trimmed notes stored",
          ctx == {"style": "casual", "notes": "Quarterly results. CEO is Pak Budi."}, str(ctx))
    check("unknown style ignored", probe.set_context("shouty", None)["style"] == "casual")
    check("notes capped", len(probe.set_context(None, "x" * 5000)["notes"]) == 1500)
    probe._history["en-id"] = [("hi", "hai")]
    probe._history["id-en"] = [("halo", "hello")]
    probe.set_context("formal", None)
    check("style change drops en-id history only",
          "en-id" not in probe._history and "id-en" in probe._history)

    tr = FakeTranslator()
    events = []
    async def collect(e):
        events.append(e)
    live = StreamSession(collect, FakeAsr(), {"id-en": tr, "en-id": tr}, cfg)
    await live.start()
    live.set_context("formal", "Use saya.")
    for chunk in np.array_split(np.concatenate([silence(400), speech(1500), silence(1000)]), 40):
        await live.feed(chunk)
        await asyncio.sleep(0.004)
    await live.flush()
    check("translator called with the session context",
          tr.contexts == [("formal", "Use saya.")], str(tr.contexts))

    # --- pure silence -------------------------------------------------------
    print("\ncase: silence produces nothing")
    events, asr, _ = await run_case(silence(3000), cfg)
    check("no speech detected in silence", len(kinds(events, "speech_start")) == 0)
    check("model never invoked", len(asr.calls) == 0)

    # --- hallucination filter ----------------------------------------------
    print("\ncase: hallucination filter")
    check("drops 'Terima kasih' over silence",
          _looks_hallucinated("Terima kasih.", 0.9, -1.5))
    check("drops 'Thank you for watching'",
          _looks_hallucinated("Thank you for watching!", 0.2, -0.3))
    check("drops decoder loops",
          _looks_hallucinated("ya ya ya ya ya ya ya ya ya", 0.1, -0.4))
    check("keeps real speech",
          not _looks_hallucinated("Saya mau pesan kopi susu satu ya", 0.02, -0.25))
    check("keeps a genuine short thanks with confident scores",
          not _looks_hallucinated("Terima kasih banyak atas bantuannya", 0.02, -0.2))
    check("keeps a confidently spoken English 'Thank you.'",
          not _looks_hallucinated("Thank you.", 0.05, -0.3))
    check("drops 'Thank you.' when Whisper doubts there was speech",
          _looks_hallucinated("Thank you.", 0.5, -0.4))
    check("drops Amara caption credit",
          _looks_hallucinated("Subtitles by the Amara.org community", 0.1, -0.3))

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
