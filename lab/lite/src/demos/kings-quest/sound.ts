import { byte, word } from "./resources.js";

export interface Note {
    start: number;
    duration: number;
    divisor: number;
    attenuation: number;
    noise: number;
}

export interface Sound {
    channels: Note[][];
    duration: number;
}

export function decodeSound(data: Uint8Array): Sound {
    const channels: Note[][] = [];
    let duration = 0;
    for (let channel = 0; channel < 4; channel++) {
        const notes: Note[] = [];
        let p = word(data, channel * 2);
        const end = channel < 3 ? word(data, (channel + 1) * 2) : data.length;
        if (p < 8 || end < p || end > data.length) {
            throw new Error("Invalid SOUND channel offset.");
        }
        let start = 0;
        while (p < end) {
            const ticks = word(data, p);
            if (ticks === 65535) {
                break;
            }
            if (p + 5 > end) {
                throw new Error("Truncated SOUND note.");
            }
            const length = ticks / 60;
            notes.push({
                start,
                duration: length,
                divisor: ((byte(data, p + 2) & 63) << 4) | (byte(data, p + 3) & 15),
                attenuation: byte(data, p + 4) & 15,
                noise: byte(data, p + 3) & 7,
            });
            start += length;
            p += 5;
        }
        duration = Math.max(duration, start);
        channels.push(notes);
    }
    return { channels, duration };
}

export interface AudioState {
    context: AudioContext;
    gain: GainNode;
    sources: AudioScheduledSourceNode[];
    muted: boolean;
}

export function noiseNotes(sound: Sound, note: Note): Note[] {
    if ((note.noise & 3) !== 3) {
        return [note];
    }
    const end = note.start + note.duration;
    const changes = sound.channels[2]!.filter((tone) => tone.start > note.start && tone.start < end);
    const starts = [note.start, ...changes.map((tone) => tone.start), end];
    return starts.slice(0, -1).map((start, index) => {
        const tone = sound.channels[2]!.filter((tone) => tone.start <= start).at(-1);
        return { ...note, start, duration: starts[index + 1]! - start, divisor: tone?.divisor ?? 0 };
    });
}

export function createAudio(): AudioState {
    const context = new AudioContext();
    const gain = context.createGain();
    gain.gain.value = 0.12;
    gain.connect(context.destination);
    return { context, gain, sources: [], muted: false };
}

export function stopSound(audio: AudioState): void {
    for (const source of audio.sources) {
        source.stop();
        source.disconnect();
    }
    audio.sources.length = 0;
}

export function playSound(audio: AudioState, sound: Sound): void {
    stopSound(audio);
    const base = audio.context.currentTime + 0.01;
    for (let channel = 0; channel < sound.channels.length; channel++) {
        const notes = channel === 3 ? sound.channels[channel]!.flatMap((note) => noiseNotes(sound, note)) : sound.channels[channel]!;
        for (const note of notes) {
            if (note.attenuation === 15 || note.duration === 0 || (channel < 3 && note.divisor === 0)) {
                continue;
            }
            const volume = audio.context.createGain();
            volume.gain.value = Math.pow(10, -note.attenuation / 10);
            volume.connect(audio.gain);
            let source: OscillatorNode | AudioBufferSourceNode;
            if (channel < 3) {
                const oscillator = audio.context.createOscillator();
                oscillator.type = "square";
                oscillator.frequency.value = Math.min(audio.context.sampleRate / 2, 111860 / note.divisor);
                source = oscillator;
            } else {
                const buffer = audio.context.createBuffer(1, audio.context.sampleRate, audio.context.sampleRate);
                const values = buffer.getChannelData(0);
                const mode = note.noise & 3;
                const frequency = mode === 3 ? 111860 / Math.max(1, note.divisor) : 1193180 / (512 << mode);
                let register = 0x4000;
                let phase = 0;
                for (let i = 0; i < values.length; i++) {
                    phase += frequency / audio.context.sampleRate;
                    if (phase >= 1) {
                        phase -= Math.floor(phase);
                        const feedback = note.noise & 4 ? (register ^ (register >> 1)) & 1 : register & 1;
                        register = (register >> 1) | (feedback << 14);
                    }
                    values[i] = register & 1 ? 1 : -1;
                }
                const noise = audio.context.createBufferSource();
                noise.buffer = buffer;
                noise.loop = true;
                source = noise;
            }
            source.connect(volume);
            source.start(base + note.start);
            source.stop(base + note.start + note.duration);
            audio.sources.push(source);
            source.onended = () => {
                const index = audio.sources.indexOf(source);
                if (index >= 0) {
                    audio.sources.splice(index, 1);
                }
                source.disconnect();
                volume.disconnect();
            };
        }
    }
}

export function setMuted(audio: AudioState, muted: boolean): void {
    audio.muted = muted;
    audio.gain.gain.value = muted ? 0 : 0.12;
}
