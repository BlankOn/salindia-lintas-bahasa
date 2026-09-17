// Captures mic audio and emits 16 kHz mono Int16 PCM chunks.
//
// We ask for an AudioContext at 16 kHz so the browser resamples for us, but
// Safari and some devices ignore that and hand back the hardware rate -- so the
// worklet resamples too if what it actually gets doesn't match.

const CHUNK = 1024; // ~64 ms at 16 kHz

class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions.targetRate;
    this.ratio = sampleRate / this.targetRate;
    this.needsResample = Math.abs(this.ratio - 1) > 1e-6;
    this.pos = 0; // fractional read position for the resampler
    this.tail = new Float32Array(0);
    this.out = new Float32Array(CHUNK);
    this.filled = 0;
  }

  resample(input) {
    // Prepend the previous block's last sample so interpolation stays continuous.
    const src = new Float32Array(this.tail.length + input.length);
    src.set(this.tail, 0);
    src.set(input, this.tail.length);

    const outLen = Math.max(0, Math.floor((src.length - 1 - this.pos) / this.ratio) + 1);
    const out = new Float32Array(outLen);
    let p = this.pos;
    for (let i = 0; i < outLen; i++) {
      const i0 = Math.floor(p);
      const frac = p - i0;
      // Guard the final sample: i0 + 1 can land one past the end when p is exact.
      const s1 = i0 + 1 < src.length ? src[i0 + 1] : src[i0];
      out[i] = src[i0] * (1 - frac) + s1 * frac;
      p += this.ratio;
    }
    const consumed = Math.floor(p);
    this.pos = p - consumed;
    this.tail = src.slice(Math.min(consumed, src.length - 1));
    return out;
  }

  push(samples) {
    let i = 0;
    while (i < samples.length) {
      const n = Math.min(CHUNK - this.filled, samples.length - i);
      this.out.set(samples.subarray(i, i + n), this.filled);
      this.filled += n;
      i += n;
      if (this.filled === CHUNK) {
        const pcm = new Int16Array(CHUNK);
        for (let k = 0; k < CHUNK; k++) {
          const s = Math.max(-1, Math.min(1, this.out[k]));
          pcm[k] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this.filled = 0;
      }
    }
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    this.push(this.needsResample ? this.resample(channel) : channel);
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
