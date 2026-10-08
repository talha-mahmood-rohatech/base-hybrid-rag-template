'use strict';

/* =========================================================================
 * MicStream - the browser half of the voice pipeline.
 *
 * Captures the mic and emits 16kHz mono PCM16 chunks. It holds no opinion
 * about speech, silence, or when a turn ends: that is the server VAD (app/voice/vad)'s job, and
 * keeping the decision in exactly one place is the point of the split. This
 * file should never grow a threshold.
 *
 *   const mic = new MicStream();
 *   mic.onChunk = (buf) => ws.send(buf);   // ArrayBuffer, PCM16LE @ 16kHz
 *   mic.onLevel = (v) => drawBars(v);      // 0..1, for the waveform only
 *   await mic.open();                      // once, at boot
 *   mic.resume();  // start sending
 *   mic.pause();   // stop sending (mic stays open, no permission re-prompt)
 * ===================================================================== */

const MIC_SAMPLE_RATE = 16000;   // the server VAD (app/voice/vad) speaks only this

class MicStream {
    constructor() {
        this.onChunk = null;
        this.onLevel = null;
        this._ctx = null;
        this._node = null;
        this._stream = null;
        this._streaming = false;
    }

    get ready() { return this._node !== null; }
    get streaming() { return this._streaming; }

    async open() {
        this._stream = await navigator.mediaDevices.getUserMedia({
            // echoCancellation is what keeps open speakers out of
            // the mic. Silero cannot help here - Axon's TTS *is* real speech,
            // so it would be classified as such; this has to be cancelled in
            // the capture path or not at all.
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });

        // Ask for 16kHz directly so the browser's high-quality resampler does
        // the conversion. Chrome/Edge honour this; if a browser hands back the
        // hardware rate anyway, the worklet resamples instead.
        const Ctx = window.AudioContext || window.webkitAudioContext;
        this._ctx = new Ctx({ sampleRate: MIC_SAMPLE_RATE });
        if (this._ctx.state === 'suspended') await this._ctx.resume();

        await this._ctx.audioWorklet.addModule('/voice/audio/pcm-worklet.js');

        const ratio = this._ctx.sampleRate / MIC_SAMPLE_RATE;
        if (Math.abs(ratio - 1) > 0.001) {
            console.warn(`🎚️ context opened at ${this._ctx.sampleRate}Hz, resampling in worklet`);
        }

        this._node = new AudioWorkletNode(this._ctx, 'pcm-worklet', {
            numberOfInputs: 1, numberOfOutputs: 0,
            processorOptions: { resampleRatio: ratio },
        });
        this._node.port.onmessage = (e) => {
            if (this.onLevel) this.onLevel(e.data.peak);
            // Dropped rather than queued while paused: this is audio from a
            // turn nobody asked for.
            if (this._streaming && this.onChunk) this.onChunk(e.data.pcm);
        };

        this._ctx.createMediaStreamSource(this._stream).connect(this._node);
    }

    resume() {
        if (!this._node) return;
        if (this._ctx.state === 'suspended') this._ctx.resume();
        this._streaming = true;
    }

    pause() {
        this._streaming = false;
    }
}
