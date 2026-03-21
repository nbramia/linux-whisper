# Linux Whisper — Vision

## One-liner

Wispr Flow-quality voice dictation, running entirely locally on Linux.

## The Problem

Voice dictation on Linux is stuck in 2019. The options are:

- **GNOME/KDE built-in:** Basic, low accuracy, no formatting, no context awareness
- **Nerd-Scriptable:** `whisper.cpp` piped into `xdotool` — functional but fragile, no streaming, no text cleanup
- **Cloud-dependent:** Google/Azure STT APIs — good quality but sends all audio off-machine, adds latency, costs money, requires internet

Meanwhile, macOS users have Wispr Flow — press a hotkey, ramble naturally with filler words and self-corrections, release, and polished text appears at the cursor in under a second. It works in every app. It adapts its tone to whether you're in Slack or a code editor. It feels like magic.

There is no Linux equivalent. We're building it.

## What We're Building

A system tray application that:

1. **Listens for a global hotkey** (hold-to-talk or toggle mode)
2. **Captures audio** from any input device via PipeWire/PulseAudio
3. **Streams transcription in real-time** using state-of-the-art local models on CPU
4. **Cleans and formats the transcript** using a hybrid pipeline — deterministic encoder models handle filler removal and punctuation, a local LLM resolves self-corrections and polishes grammar
5. **Injects the polished text** at the cursor position in whatever application is focused
6. All of this happens **entirely on-device**, with **zero network calls**, in **under 1 second end-to-end**

## Design Principles

### 1. Local-First, Always
No cloud. No API keys. No internet required. Audio never leaves the machine. This is a privacy guarantee, not a fallback mode.

### 2. Latency Is the Feature
The difference between "useful" and "magical" is 500ms. Every architectural decision optimizes for time-to-text. We target:
- **< 300ms** for the first transcribed words to appear (streaming)
- **< 800ms** end-to-end from releasing the hotkey to polished text at cursor
- **Constant latency** — 5-second and 30-second utterances should feel equally responsive

### 3. Quality Over Quantity of Features
We ship fewer features, but each one is polished. The core loop — speak, release, text appears — must be flawless before we add anything else.

### 4. Works Everywhere
Text injection works in any application that accepts keyboard input: terminals, browsers, IDEs, Slack, email clients, LibreOffice. X11 and Wayland. GNOME, KDE, Hyprland, Sway.

### 5. Invisible When Not in Use
Minimal resource consumption when idle. No background CPU burn. The tray icon is the only visible footprint. Models stay warm in RAM (~3.4GB of 64GB) for instant response with negligible system impact.

### 6. CPU-Native by Design
Architected for high-core-count AMD CPUs with AVX-512, not as a GPU fallback. Every model chosen runs best on CPU. GPU acceleration (ROCm, NPU) is a future bonus, not a requirement.

## User Experience

### Core Flow

```
[User holds hotkey] → Tray icon changes to "recording" + subtle audio cue
[User speaks naturally] → Streaming transcription preview appears (optional overlay)
[User releases hotkey] → Audio cue, brief pause (~500ms)
[Polished text appears at cursor]
```

### What "Polished" Means

You say:
> "Hey um so I was thinking we should probably... actually no, let's definitely move the deployment to uh Friday. Friday afternoon. Can you update the ticket?"

You get:
> Let's move the deployment to Friday afternoon. Can you update the ticket?

Filler words gone. Self-correction resolved. Punctuation inferred. Professional tone maintained.

### How Polishing Works

Unlike Wispr Flow (which sends everything through a cloud LLM), we use a **hybrid pipeline** that's both faster and safer:

1. **Deterministic cleanup** (encoder models, ~10ms): A BERT token classifier removes filler words, repetitions, and false starts. An ELECTRA model adds punctuation and capitalization. These are sequence-labeling tasks — they **cannot hallucinate or add content**.
2. **Smart correction** (generative LLM, ~300ms): Only invoked when the transcript contains self-corrections or needs grammar repair. A local 4B model resolves "actually no, make it X" patterns while strictly preserving the speaker's words.

This split means 80%+ of dictations never touch a generative model at all.

### Context Awareness (v2)

The system detects the focused application and adjusts output style:
- **Terminal/IDE:** Terse, technical. No pleasantries.
- **Slack/Discord:** Conversational. Casual punctuation.
- **Email client:** Professional. Complete sentences.
- **Markdown editor:** Proper heading/list formatting.

### Command Mode (v2)

After dictating, hold the hotkey again and speak an editing command:
- "Make that more formal"
- "Translate to Spanish"
- "Turn that into a bullet list"
- "Actually, remove the last sentence"

## Target Hardware

### Primary Target
- **AMD Ryzen AI MAX+ 395** (or similar high-core-count CPU with AVX-512)
- 32+ GB unified RAM
- Integrated AMD GPU (ROCm-capable, optional acceleration)
- XDNA2 NPU (future acceleration path)

### Minimum Requirements
- Any x86_64 CPU with AVX2, 4+ cores
- 8GB RAM
- No GPU required

### With NVIDIA GPU (alternative path)
- Any NVIDIA GPU with 4GB+ VRAM enables faster-whisper GPU acceleration
- Not the primary target but supported via backend selection

### Memory Budget (Always-On)

| Component | RAM |
|-----------|-----|
| Moonshine v2 Medium (ONNX) | ~500MB |
| BERT disfluency classifier | ~110MB |
| ELECTRA punctuation model | ~60MB |
| Qwen3 4B Instruct (Q4_K_M) | ~2.5GB |
| Python runtime + overhead | ~200MB |
| **Total** | **~3.4GB** |

~5% of 64GB. Comfortable for continuous background operation alongside any workload.

## Non-Goals (For Now)

- **Mobile/tablet support** — desktop Linux only
- **Speaker diarization** — single-speaker dictation only
- **Real-time captioning** — we optimize for dictation, not live transcription of meetings
- **GUI settings panel** — config file is fine for v1
- **Plugin/extension system** — keep the surface area small
- **Windows/macOS** — Linux-only, no cross-platform abstractions

## Success Metrics

- End-to-end latency p95 < 800ms on target hardware (Ryzen AI MAX+ 395)
- Word error rate < 7% on native English dictation (Moonshine v2 Medium baseline: 6.65%)
- Filler word removal accuracy > 95% (BERT classifier)
- Self-correction resolution accuracy > 90% (LLM)
- Zero network calls during operation (verified by audit)
- Idle CPU usage < 0.5%, idle RAM ~3.4GB (models warm)
- Works on X11 and Wayland without user configuration

## Roadmap

### v0.1 — Core Loop
- Hold-to-talk hotkey via evdev
- Audio capture via PipeWire
- Transcription via Moonshine v2 (streaming)
- Raw text injection at cursor (no cleanup yet)
- System tray icon with recording state

### v0.2 — Polish Pipeline
- BERT disfluency removal (filler words, repetitions)
- ELECTRA punctuation and capitalization
- LLM self-correction resolution (Qwen3 4B)
- Audio feedback (start/stop cues)
- Configurable hotkey

### v0.3 — Performance & Alternatives
- whisper.cpp backend for highest-accuracy batch mode
- VAD-based auto-stop (release hotkey not required)
- Streaming transcription preview (optional floating overlay)
- ROCm iGPU acceleration (experimental)
- Latency profiling and optimization pass

### v1.0 — Release
- Context-aware formatting
- Personal dictionary / custom vocabulary
- Voice snippets (trigger phrases for canned text)
- Proper packaging (`.deb`, Flatpak, AUR)
- Documentation and onboarding

### v2.0 — Command Mode
- Voice editing commands on existing text
- Local LLM-powered text transformation
- Multi-language support
- XDNA2 NPU acceleration for STT
