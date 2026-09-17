"""End-to-end check against the real models, using macOS TTS as the speaker.

    python tests/bench_real.py OUT_DIR [id-en|en-id|both]

Sentences are deliberately held out: none appear in the few-shot system prompts,
so the translations say something about the model rather than the examples.
"""

import asyncio, os, pathlib, subprocess, sys, time, wave
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from server.asr import WhisperEngine
from server.translate import Translator

OUT = pathlib.Path(sys.argv[1])
WHICH = sys.argv[2] if len(sys.argv) > 2 else "both"

VOICES = {"id-en": "Damayanti", "en-id": "Samantha"}

SENTENCES = {
    "id-en": [
        ("meeting",  "Rapatnya diundur jadi jam tiga sore ya, soalnya klien belum konfirmasi."),
        ("slang",    "Wah gila sih, antriannya panjang banget. Mendingan kita balik lagi besok pagi aja."),
        ("technical","Server-nya sempat down sekitar lima belas menit gara-gara ada masalah di database."),
        ("negation", "Saya belum sempat baca dokumennya, jadi belum bisa kasih keputusan sekarang."),
        ("numbers",  "Total biayanya dua juta tiga ratus lima puluh ribu rupiah, belum termasuk pajak."),
        ("question", "Kamu udah ngecek email dari tim legal belum? Katanya penting banget."),
        ("polite",   "Mohon maaf mengganggu waktunya, apakah Bapak berkenan menjadwalkan ulang pertemuan kita?"),
    ],
    "en-id": [
        ("meeting",  "The meeting got moved to three this afternoon because the client still hasn't confirmed."),
        ("casual",   "Dude, that line is ridiculous. Let's just come back tomorrow morning."),
        ("technical","The API was timing out for about ten minutes, so we rolled back the last release."),
        ("formal",   "I would like to thank all of you for attending today's meeting on such short notice."),
        ("numbers",  "The total comes to two thousand three hundred and fifty dollars, not including tax."),
        ("question", "Have you had a chance to look at the contract I sent you yesterday?"),
        ("feeling",  "I'm really sorry, I completely forgot about your birthday. Let me make it up to you."),
    ],
}


def synth(text: str, voice: str, path: pathlib.Path) -> np.ndarray:
    """macOS TTS straight to 16 kHz mono PCM -- no ffmpeg in the loop."""
    wav = path.with_suffix(".wav")
    subprocess.run(
        ["say", "-v", voice, "-o", str(wav), "--data-format=LEI16@16000", text],
        check=True)
    with wave.open(str(wav)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    asr = WhisperEngine("mlx-community/whisper-large-v3-turbo", "transcribe")
    mt_model = os.environ.get("MT_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
    print(f"MT model: {mt_model}")
    mt = Translator(mt_model, 256, 2)

    print("warming up (compiling Metal kernels)...")
    t = time.perf_counter(); await asr.warmup(); await mt.warmup()
    print(f"warmup took {time.perf_counter() - t:.1f}s")

    directions = list(SENTENCES) if WHICH == "both" else [WHICH]
    for direction in directions:
        src_lang = direction.split("-")[0]
        print(f"\n==== {direction}  (voice: {VOICES[direction]}) ====\n")
        history, rtfs = [], []
        for label, text in SENTENCES[direction]:
            audio = synth(text, VOICES[direction], OUT / f"{direction}-{label}")
            dur = len(audio) / 16000

            r = await asr.run(audio, language=src_lang, final=True)
            m = await mt.translate(r.text, history, direction)
            history = (history + [(r.text, m.text)])[-2:]

            total = r.elapsed + m.elapsed
            rtfs.append(dur / total)
            print(f"[{label}]  {dur:.1f}s audio")
            print(f"  said : {text}")
            print(f"  asr  : {r.text}")
            print(f"  out  : {m.text}")
            print(f"  time : asr {r.elapsed*1000:.0f}ms + mt {m.elapsed*1000:.0f}ms "
                  f"= {total*1000:.0f}ms  ({dur/total:.1f}x realtime)\n")
        print(f"{direction} mean throughput: {sum(rtfs)/len(rtfs):.1f}x realtime")

    asr.shutdown(); mt.shutdown()

asyncio.run(main())
