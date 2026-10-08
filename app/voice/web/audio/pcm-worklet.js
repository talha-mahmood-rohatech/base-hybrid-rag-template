/* Mic -> 16kHz mono PCM16, on the audio thread.
 *
 * Deliberately dumb: it makes no speech/silence decision at all. That now
 * belongs to the server VAD (app/voice/vad). All this does is convert and post.
 *
 * The AudioContext is normally opened at 16kHz so the browser's own resampler
 * does the work; `resampleRatio` is the fallback for contexts that refuse the
 * requested rate and hand back the hardware rate instead.
 */

const FRAME = 512;        // matches the server VAD (app/voice/vad) frame_samples - one model frame
const CHUNK_FRAMES = 4;   // post every ~128ms: few enough messages, low latency

class PcmWorklet extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const { resampleRatio } = options.processorOptions || {};
        this.ratio = resampleRatio || 1;
        this.buf = new Int16Array(FRAME * CHUNK_FRAMES);
        this.n = 0;
        this.pos = 0;   // fractional read position, only used when resampling
        this.peak = 0;
    }

    // Linear interpolation. Only reached when the context wouldn't open at
    // 16kHz; quality here matters little since Silero is robust and this is
    // a fallback path, but decimating without it would alias badly.
    resample(input) {
        const out = [];
        while (this.pos < input.length) {
            const i = Math.floor(this.pos);
            const frac = this.pos - i;
            const a = input[i];
            const b = i + 1 < input.length ? input[i + 1] : a;
            out.push(a + (b - a) * frac);
            this.pos += this.ratio;
        }
        this.pos -= input.length;
        return out;
    }

    process(inputs) {
        const channel = inputs[0] && inputs[0][0];
        if (!channel) return true;

        const samples = this.ratio === 1 ? channel : this.resample(channel);

        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            if (s > this.peak) this.peak = s;
            this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;

            if (this.n === this.buf.length) {
                // Transfer the buffer rather than copy it - this runs on the
                // audio thread and must not allocate pressure or block.
                const out = this.buf.slice();
                this.port.postMessage({ pcm: out.buffer, peak: this.peak }, [out.buffer]);
                this.n = 0;
                this.peak = 0;
            }
        }
        return true;
    }
}

registerProcessor('pcm-worklet', PcmWorklet);
