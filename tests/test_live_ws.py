"""Smoke test + latency probe for a running server.

    python tests/test_live_ws.py [ws-url] [id-en|en-id] [gap-ms]

Streams TTS speech over the WebSocket at real-time pace, sentences separated by
`gap-ms` of room tone, and reports for each sentence how long after the speaker
went quiet its source text and translation appeared.
"""

import asyncio, json, subprocess, sys, time, wave
import numpy as np
import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8123/ws"
DIRECTION = sys.argv[2] if len(sys.argv) > 2 else "id-en"
GAP_MS = int(sys.argv[3]) if len(sys.argv) > 3 else 1100
TMP = "/tmp/ws_probe.wav"

LINES = {
    "id-en": ("Damayanti", [
        "Halo, selamat siang. Saya mau tanya soal jadwal pengiriman barangnya.",
        "Kalau bisa tolong dikirim minggu depan ya, soalnya kami butuh cepat.",
        "Oh iya, alamatnya masih sama seperti bulan lalu.",
        "Terima kasih banyak atas bantuannya.",
    ]),
    "en-id": ("Samantha", [
        "Hi, good afternoon. I wanted to ask about the delivery schedule.",
        "If possible, please ship it next week, because we need it soon.",
        "Oh, and the address is still the same as last month.",
        "Thank you so much for your help.",
    ]),
}


def pcm_for(text, voice):
    subprocess.run(["say", "-v", voice, "-o", TMP,
                    "--data-format=LEI16@16000", text], check=True)
    with wave.open(TMP) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def quiet(ms):
    n = 16000 * ms // 1000
    return (np.random.default_rng(1).standard_normal(n) * 20).astype("<i2")


async def main():
    voice, lines = LINES[DIRECTION]
    pieces, speech_ends = [quiet(600)], []
    cursor = len(pieces[0])
    for text in lines:
        pcm = pcm_for(text, voice)
        # TTS pads with near-silence; find where the voice really stops.
        loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) > 300)
        speech_ends.append((cursor + (loud[-1] if loud.size else len(pcm))) / 16000)
        pieces += [pcm, quiet(GAP_MS)]
        cursor += len(pcm) + 16000 * GAP_MS // 1000
    stream = np.concatenate(pieces)
    print(f"{DIRECTION}: streaming {len(stream)/16000:.1f}s, {len(lines)} sentences, "
          f"{GAP_MS}ms gaps\n")

    async with websockets.connect(URL, max_size=None) as ws:
        events, done = [], asyncio.Event()
        config = {}

        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "level":
                    continue
                if msg["type"] in ("hello", "ready"):
                    config.update(msg.get("config", {}))
                events.append((time.perf_counter(), msg))
                if msg["type"] == "idle":
                    done.set()
                    return

        task = asyncio.create_task(reader())
        t0 = time.perf_counter()
        while not any(m["type"] == "ready" for _, m in events):
            if time.perf_counter() - t0 > 300:
                raise SystemExit("server never became ready")
            await asyncio.sleep(0.2)
        await ws.send(json.dumps({"type": "direction", "value": DIRECTION}))
        await asyncio.sleep(0.2)
        if any(m["type"] == "error" for _, m in events):
            raise SystemExit([m for _, m in events if m["type"] == "error"])

        start = time.perf_counter()
        chunk = 1024  # 64 ms, same as the browser worklet
        for i in range(0, len(stream), chunk):
            await ws.send(stream[i:i + chunk].tobytes())
            target = start + (i + chunk) / 16000
            if (delay := target - time.perf_counter()) > 0:
                await asyncio.sleep(delay)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(done.wait(), timeout=120)
        task.cancel()

    finals = {m["id"]: (ts - start, m) for ts, m in events if m["type"] == "final"}
    trans = {m["id"]: (ts - start, m) for ts, m in events if m["type"] == "translation"}
    order = [m["id"] for _, m in events if m["type"] == "translation"]

    print(f"{'#':>2} {'src after':>9} {'dst after':>9}  spec  text")
    lat = []
    for n, (sid, (fts, f)) in enumerate(sorted(finals.items())):
        tts, t = trans.get(sid, (None, {}))
        quiet_at = speech_ends[n] if n < len(speech_ends) else fts
        d_src, d_dst = fts - quiet_at, (tts - quiet_at) if tts else float("nan")
        lat.append(d_dst)
        print(f"{n+1:>2} {d_src:8.2f}s {d_dst:8.2f}s  {'yes' if f.get('speculated') else ' no'}   "
              f"{f['src']}\n{'':30}-> {t.get('dst')}")
    errors = [m for _, m in events if m["type"] == "error"]
    ok = len(finals) == len(lines) and order == sorted(order) and not errors
    print(f"\nsentences: {len(finals)}/{len(lines)}   in order: {order == sorted(order)}"
          f"   errors: {len(errors)}")
    if lat:
        print(f"end-of-speech -> translation: mean {np.nanmean(lat):.2f}s, "
              f"max {np.nanmax(lat):.2f}s")
    sys.exit(0 if ok else 1)


asyncio.run(main())
