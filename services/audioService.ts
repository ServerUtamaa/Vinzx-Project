
import { GoogleGenAI, Modality } from "@google/genai";
import { VoiceGender } from "../types";

// --- SFX GENERATOR (Telemetri biar keren pas loading) ---
export const playTelemetrySound = () => {
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContext) return;

    const ctx = new AudioContext();
    if (ctx.state === 'suspended') ctx.resume();

    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    
    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.type = 'sine';
    osc.frequency.setValueAtTime(1200, t);
    osc.frequency.exponentialRampToValueAtTime(600, t + 0.1);
    
    gain.gain.setValueAtTime(0.05, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + 0.1);

    osc.start(t);
    osc.stop(t + 0.1);
};

// --- AUDIO DECODER UTILS (RAW PCM HANDLING) ---
// Gemini ngirim audio mentah (PCM), bukan MP3. Jadi harus didecode manual biar bisa bunyi di browser/APK.

function decodeBase64(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number = 24000, // Gemini Default Hz
  numChannels: number = 1
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      // Convert Int16 (-32768 to 32767) to Float32 (-1.0 to 1.0)
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- MAIN SERVICE CLASS ---
class GeminiAudioService {
    private audioCtx: AudioContext | null = null;

    private getContext(): AudioContext {
        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ 
                sampleRate: 24000 
            });
        }
        return this.audioCtx;
    }

    // 1. Generate Audio dari Google Gemini
    public async generate(text: string, gender: VoiceGender): Promise<string | null> {
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            
            // Konfigurasi Suara
            // MALE -> Fenrir (Deep, Authority, Bold)
            // FEMALE -> Kore (Soft, Elegant, Soothing)
            const voiceName = gender === 'MALE' ? 'Fenrir' : 'Kore';

            // Bersihkan teks biar bacanya enak (Gak baca simbol aneh)
            const cleanText = text
                .replace(/\*/g, '')
                .replace(/_/g, '')
                .replace(/-/g, ' ')
                .replace(/\//g, ' atau ')
                .replace(/RR:/g, 'Risk Reward ')
                .replace(/SL/g, 'Stop Loss ')
                .replace(/TP/g, 'Target Profit ');

            const response = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts",
                contents: { parts: [{ text: cleanText }] },
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: {
                        voiceConfig: {
                            prebuiltVoiceConfig: { voiceName: voiceName },
                        },
                    },
                },
            });

            const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            return base64Audio || null;

        } catch (error) {
            console.error("Gemini Audio Error:", error);
            return null;
        }
    }

    // 2. Play Audio Raw PCM
    public async play(base64Audio: string, onComplete?: () => void) {
        if (!base64Audio) return;

        try {
            const ctx = this.getContext();
            
            // PENTING BUAT APK: Resume context kalau suspended (karena autoplay policy)
            if (ctx.state === 'suspended') {
                await ctx.resume();
            }

            const pcmData = decodeBase64(base64Audio);
            const audioBuffer = await decodeAudioData(pcmData, ctx);
            
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);
            
            source.onended = () => {
                if (onComplete) onComplete();
            };

            source.start();

        } catch (e) {
            console.error("Playback Error:", e);
            if (onComplete) onComplete();
        }
    }

    public stop() {
        if (this.audioCtx) {
            this.audioCtx.close().then(() => {
                this.audioCtx = null;
            });
        }
    }
}

export const audioService = new GeminiAudioService();
