# Linux Whisper — Architecture

## Hardware Profile

The primary development target is an AMD-based Linux workstation:

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen AI MAX+ 395 — 16 cores / 32 threads |
| ISA Extensions | AVX-512 (including BF16, VNNI), AVX2, SSE4.2 |
| RAM | 64GB unified (shared with iGPU) |
| GPU | Radeon 8060S (RDNA 3.5 iGPU, gfx1151) — ROCm driver loaded |
| NPU | XDNA2 (AMD Ryzen AI) — future acceleration target |
| GPU Compute | ROCm recognized (rocminfo/rocm-smi work), but PyTorch ROCm for gfx1151 is experimental |

**Key constraint:** No NVIDIA GPU. All CUDA-dependent tools (faster-whisper GPU mode, CTranslate2 CUDA, NeMo) are unavailable. The architecture is CPU-first, with ROCm and NPU as future acceleration paths.

**Key advantage:** AVX-512 with VNNI (Vector Neural Network Instructions) and BF16 make quantized CPU inference unusually fast. 64GB unified RAM eliminates memory pressure entirely.

## System Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Linux Whisper                                │
│                                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │  Input    │→│  Audio    │→│   STT    │→│  Polish Pipeline    │   │
│  │  Manager  │  │  Pipeline │  │  Engine  │  │  (hybrid 3-stage)  │   │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┬───────────┘   │
│       ↑                                               │               │
│  ┌──────────┐                                   ┌─────▼───────────┐   │
│  │  Hotkey   │                                   │  Text Injector  │   │
│  │  Daemon   │                                   │  (X11/Wayland)  │   │
│  └──────────┘                                   └─────────────────┘   │
│       ↑                                               │               │
│  ┌──────────┐                                   ┌─────▼───────────┐   │
│  │  System   │                                   │  Focused App    │   │
│  │  Tray     │                                   │  (any text      │   │
│  └──────────┘                                   │   field)        │   │
│                                                  └─────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

The end-to-end pipeline has 6 stages. Each stage has a latency budget:

| Stage | Component | Latency Budget | Runs On | Notes |
|-------|-----------|---------------|---------|-------|
| 1 | Hotkey detection | < 5ms | CPU (evdev) | Kernel-level input event |
| 2 | Audio capture + VAD | < 10ms | CPU | PipeWire stream, Silero VAD |
| 3 | Speech-to-text | < 300ms | CPU (AVX-512) | faster-whisper large-v3-turbo, INT8 batch |
| 4a | Disfluency removal | < 15ms | CPU | BERT token classifier |
| 4b | Punctuation + caps | < 15ms | CPU | ELECTRA-small classifier |
| 4c | Self-correction + grammar | < 350ms | CPU (AVX-512) | Qwen3 4B Q4_K_M, only when needed |
| 5 | Text injection | < 20ms | subprocess | ydotool/xdotool |
| **Total (simple)** | | **< 365ms** | | **No self-corrections detected** |
| **Total (complex)** | | **< 715ms** | | **Self-corrections present → LLM invoked** |

Stage 3 (STT) runs in batch mode after recording ends — faster-whisper processes the complete audio with INT8 quantization via CTranslate2. Stages 4a and 4b are fast encoder models that run in series on the final transcript. Stage 4c (generative LLM) is only invoked when the disfluency detector flags self-corrections in the transcript.

---

## Stage 1: Input Manager

### Hotkey Daemon

Captures global hotkeys regardless of focused application.

**Implementation:** `evdev` — reads directly from `/dev/input/event*` devices. This works on both X11 and Wayland without requiring root (user must be in the `input` group). No dependency on desktop environment.

**Why not alternatives:**
- `keyboard` (Python lib): Requires root on Linux
- `pynput`: X11-only, broken on Wayland
- `xbindkeys`: X11-only
- D-Bus global shortcuts portal: GNOME/KDE only, not universal

**Modes:**
- **Auto (default):** Automatically detects hold vs double-tap. Hold the key for longer than 300ms and it behaves as hold-to-talk (stops on release). Double-tap quickly and it enters toggle mode (stays recording until the next tap). Best of both worlds with zero configuration.
- **Hold:** Recording starts on key-down, stops on key-up. Simplest, most reliable.
- **Toggle:** First press starts recording, second press stops. Better for long dictation.
- **VAD-auto-stop:** Recording starts on key-down, stops automatically when silence is detected for N seconds. Best for hands-free.

**Default hotkey:** `Ctrl+Shift+E` (low conflict, ergonomic).

### State Machine

```
     ┌─────────┐  hotkey down   ┌───────────┐  hotkey up    ┌────────────┐
     │  IDLE   │──────────────→│ RECORDING │─────────────→│ PROCESSING │
     └─────────┘               └───────────┘              └──────┬─────┘
          ↑                                                       │
          └───────────────────────────────────────────────────────┘
                              text injected / error
```

---

## Stage 2: Audio Pipeline

### Capture

**Backend:** PipeWire (with PulseAudio compatibility fallback).

PipeWire is the default audio server on modern Linux (Fedora 34+, Ubuntu 22.10+, Arch). We use the PipeWire API directly via `sounddevice` (which wraps PortAudio, which supports PipeWire). Fallback to PulseAudio for older systems.

**Format:**
- Sample rate: 16kHz (native for all target STT models)
- Channels: Mono
- Bit depth: 16-bit signed integer (converted to float32 for model input)
- Buffer size: 512 samples (32ms at 16kHz) — balances latency vs. overhead

### Voice Activity Detection (VAD)

**Model:** Silero VAD v5

- ~1ms inference per 32ms audio chunk on CPU
- Detects speech onset within 50ms
- Detects speech offset within 200ms (configurable)
- Pre-trained, no fine-tuning needed
- Used for: auto-stop mode, trimming silence from start/end of recordings, filtering noise-only activations

**Ring Buffer:**
Audio is captured into a lock-free ring buffer. The VAD runs on every chunk. When the hotkey is released (or VAD auto-stop triggers), the buffer contents between the first and last speech frames are extracted and sent to Stage 3.

For streaming mode, audio chunks are forwarded to the STT engine in real-time as they pass VAD, without waiting for the recording to end.

### Audio Feedback

Start/stop cues are played via `sounddevice` output stream:
- **Recording start:** Short rising tone (50ms, 880Hz→1760Hz)
- **Recording stop:** Short falling tone (50ms, 1760Hz→880Hz)
- Generated programmatically — no audio file dependencies

---

## Stage 3: Speech-to-Text Engine

### Model Strategy

We support multiple STT backends selected at startup. The engine interface is abstract — all backends implement the same protocol. The default is optimized for our target hardware (CPU with AVX-512, no NVIDIA GPU).

#### Primary: faster-whisper large-v3-turbo (Default)

| Attribute | Value |
|-----------|-------|
| Parameters | 809M |
| Architecture | Whisper encoder-decoder, INT8 via CTranslate2 |
| Avg WER (Open ASR datasets) | 7.25% |
| Mode | Batch (processes complete audio after recording ends) |
| Runtime | CTranslate2 (INT8 quantization, AVX-512 optimized) |
| RAM | ~4GB (Q8) |
| Languages | 99 languages |

**Why faster-whisper large-v3-turbo:**
- **Best transcription quality on CPU.** Noticeably better output than smaller models in practice.
- **Built-in Silero VAD filter** handles silence trimming well, reducing noise-only segments.
- **CTranslate2 INT8** leverages AVX-512 VNNI for fast quantized inference on the target CPU.
- **No PyTorch dependency.** CTranslate2 is a standalone C++ inference engine.
- **Hot-swappable** from the system tray menu at runtime without restarting the application.

#### Alternative: Moonshine v2 Medium (Low-Latency Streaming)

| Attribute | Value |
|-----------|-------|
| Parameters | 244.9M |
| Architecture | Sliding-window streaming encoder |
| Avg WER | 6.65% |
| Streaming | Native — 80ms algorithmic lookahead |
| Runtime | ONNX Runtime (CPU) |
| RAM | ~500MB |

For users who prefer streaming output (words appear as you speak) or need lower memory usage. Designed for CPU/edge with 6.65% WER at 245M params.

#### Alternative: Moonshine v2 Tiny (Minimal)

| Attribute | Value |
|-----------|-------|
| Parameters | 33.6M |
| TTFT | ~50ms |
| WER | 12.01% (avg) |
| RAM | ~150MB |

For users who prioritize speed over accuracy (short commands, quick notes). The 50ms TTFT is near-imperceptible.

#### Fallback: whisper.cpp (Highest Accuracy)

For batch-mode transcription via GGML quantization.

| Model | Params | Avg WER | Quantization | RAM | CPU Performance |
|-------|--------|---------|-------------|-----|-----------------|
| large-v3-turbo Q8_0 | 809M | 7.25% | 8-bit GGML | ~4GB | Fast on AVX-512 |
| large-v3-turbo Q5_1 | 809M | ~7.3% | 5-bit GGML | ~2.5GB | Fastest |
| distil-large-v3.5 Q8_0 | 756M | 7.10% | 8-bit GGML | ~3.5GB | English only, best WER in Whisper family |

whisper.cpp has explicit AVX-512 optimization paths and runs well on this CPU. Not streaming, but fast enough for short utterances in batch mode.

#### Future: ROCm GPU Acceleration

The Radeon 8060S iGPU is visible to ROCm (gfx1151) and reports as a compute-capable agent. However:
- PyTorch is currently installed with CUDA 12.8 backend, not ROCm
- gfx1151 (RDNA 3.5) is a brand-new target — framework support is experimental
- If ROCm + PyTorch stabilizes for gfx1151, faster-whisper or Moonshine could potentially offload to the iGPU

This is a v0.3+ exploration item, not a launch dependency.

#### Future: XDNA2 NPU Acceleration

The Ryzen AI MAX+ 395 includes an XDNA2 Neural Processing Unit. AMD's Ryzen AI SDK is maturing but Linux support is still early. If/when ONNX Runtime gains XDNA2 execution provider support on Linux, Moonshine v2 (already ONNX) could run on the NPU with near-zero CPU impact.

### Benchmark Context: Open ASR Leaderboard (March 2026)

For reference, current top models and where our choices sit:

| Rank | Model | Avg WER | Params | CPU-Viable | Notes |
|------|-------|---------|--------|------------|-------|
| 1 | IBM Granite 4.0 1B Speech | 5.52% | ~2B | Possible but heavy | New #1, just released Mar 2026 |
| 2 | NVIDIA Canary-Qwen 2.5B | 5.63% | 2.5B | No (NeMo/CUDA) | |
| 5 | NVIDIA Canary-1B-Flash | 6.35% | 883M | No (NeMo/CUDA) | |
| 6 | NVIDIA Parakeet-TDT 0.6B v3 | 6.34% | 600M | No (NeMo/CUDA) | |
| — | Moonshine v2 Medium | 6.65% | 245M | Yes — designed for it | ← Our alternative (streaming) |
| 8 | Distil-Whisper v3.5 | 7.10% | 756M | Yes (faster-whisper) | ← Available option |
| 9 | Whisper large-v3 | 7.14% | 1.55B | Slow | |
| 10 | **Whisper large-v3-turbo** | **7.25%** | **809M** | **Yes (faster-whisper INT8)** | **← Our default** |

Our default (faster-whisper large-v3-turbo) offers the best practical quality on CPU with INT8 quantization. Moonshine v2 Medium remains available for users who want streaming output or lower memory usage.

### Engine Interface

```python
class STTEngine(Protocol):
    def start_stream(self) -> None:
        """Prepare for streaming audio input."""

    def feed_audio(self, chunk: np.ndarray) -> list[TranscriptSegment]:
        """Feed an audio chunk, return any new transcript segments."""

    def finalize(self) -> TranscriptResult:
        """Signal end of audio, return final transcript."""

    def reset(self) -> None:
        """Reset state for next utterance."""
```

All backends implement this interface. Streaming backends (Moonshine) emit partial results from `feed_audio()`. Batch backends (whisper.cpp) buffer internally and only return results from `finalize()`.

### Model Management

Models are downloaded from Hugging Face Hub on first use and cached in `~/.cache/linux-whisper/models/`. The app sets `HF_HUB_OFFLINE=1` after initial download to guarantee offline operation.

A CLI command handles model management:
```bash
linux-whisper models list          # Show available/downloaded models
linux-whisper models download <id> # Download a specific model
linux-whisper models default <id>  # Set the default model
```

---

## Stage 4: Polish Pipeline (Hybrid 3-Stage)

This is what separates dictation from transcription. Rather than sending everything through a single generative LLM (like Wispr Flow does with cloud Llama), we use a hybrid pipeline that's faster, more predictable, and cannot hallucinate for the most common operations.

### Why Hybrid?

| Approach | Pros | Cons |
|----------|------|------|
| Single LLM for everything | Simple architecture | Can hallucinate, add words, paraphrase; always pays LLM latency even for trivial cleanup |
| Encoder-only models | Deterministic, fast (~10ms), zero hallucination risk | Can't resolve self-corrections or rephrase |
| **Hybrid (our approach)** | **Fast path for 80%+ of cases; LLM only when genuinely needed** | **Slightly more complex pipeline** |

### Stage 4a: Disfluency Removal (BERT Token Classifier)

**Task:** Tag and remove filler words, repetitions, and false starts.

**Model:** Fine-tuned BERT-base for disfluency detection (~110M params, ~440MB)

- Sequence labeling (BIO tags), not generation — **cannot hallucinate or add content**
- Trained on Fisher/Switchboard corpora for English disfluency detection
- Identifies: filled pauses ("um", "uh"), discourse markers ("like", "you know", "I mean", "basically"), repetitions ("I I I think"), false starts ("we should go to the— let's stay here")
- F1 ~91% on Switchboard benchmark
- Inference: ~10ms on CPU for typical utterance length
- Can use a compressed 6-layer distilled variant (~1.3MB at INT8) with F1 ~88.4% if memory is critical

**Example:**
```
Input:  "um so I was I was thinking we should uh probably move it"
Output: "I was thinking we should probably move it"
Tags:   [RM][RM][KEEP][KEEP][RM][RM][KEEP][KEEP][KEEP][RM][KEEP][KEEP][KEEP]
```

### Stage 4b: Punctuation & Capitalization (ELECTRA-Small)

**Task:** Add punctuation (commas, periods, question marks) and fix capitalization.

**Model:** Two consecutive ELECTRA-small models (~14M params each, ~60MB total)

- Token classification: each token gets a punctuation label (NONE, COMMA, PERIOD, QUESTION) + capitalization label
- Latency of just 4 tokens — suitable for streaming output
- **Outperforms GPT services** on punctuation restoration benchmarks (Polacek et al., Interspeech 2023)
- No risk of content modification — only inserts punctuation characters and adjusts case
- Inference: ~5ms on CPU

**Alternative:** NVIDIA NeMo Punctuation DistilBERT (~207MB) — handles the same task with a single model. Slightly heavier but well-tested.

**Example:**
```
Input:  "i was thinking we should probably move it to friday can you update the ticket"
Output: "I was thinking we should probably move it to Friday. Can you update the ticket?"
```

### Stage 4c: Self-Correction Resolution & Grammar (Generative LLM)

**Task:** Resolve self-corrections ("at 2... actually 4") and fix grammar that encoder models can't handle.

**Trigger:** Only invoked when Stage 4a detects self-correction patterns (reparandum + repair spans). For simple dictation without self-corrections, this stage is **skipped entirely**, saving ~300ms.

**Model Selection:**

We evaluated sub-4B models on **IFEval** (instruction following) as the critical benchmark — the model must follow "do NOT paraphrase" instructions precisely.

| Model | IFEval | Q4_K_M Size | Est. Speed (AVX-512) | Notes |
|-------|--------|-------------|---------------------|-------|
| **Gemma 3 4B IT** | **90.2** | 3.4GB | ~40-60 tok/s | Best instruction following by far |
| Qwen3 4B Instruct | ~87.8 | 2.5GB | ~50-70 tok/s | Strong all-around, smaller footprint |
| Phi-4-mini (3.8B) | ~83.7 | 2.5GB | ~50-70 tok/s | Best reasoning per param |
| Llama 3.2 3B Instruct | 77.4 | 2.0GB | ~60-80 tok/s | Lighter but weaker on instruction following |
| Qwen 2.5 3B Instruct | 58.2 | 2.0GB | ~60-80 tok/s | Poor instruction following at this size |

**Primary: Qwen3 4B Instruct (Q4_K_M)**
- IFEval ~87.8 — second-best instruction following, critical for "do NOT paraphrase"
- 2.5GB Q4_K_M — 0.9GB lighter than Gemma 3 4B with only ~2.4% lower IFEval
- ~50-70 tok/s on AVX-512 — generates 20-50 cleanup tokens in 300-700ms
- Strong multilingual foundation for future language support
- Served via `llama-cpp-python` (GGUF format), kept warm in RAM

**Alternative: Gemma 3 4B IT (Q4_K_M)**
- IFEval 90.2 — absolute best instruction adherence in the sub-4B class
- 3.4GB Q4_K_M — heavier but worth it if instruction following proves critical in testing
- Swap-in replacement; same interface

**Why not larger models:** The LLM task is constrained — it resolves self-corrections and fixes grammar on a ~20-50 token transcript. A 4B model with a good system prompt handles this reliably. Larger models would blow the latency budget without meaningful quality gain.

### Prompt Architecture

```
System: You clean up dictated text. Preserve the speaker's exact words and meaning.

Rules:
- Resolve self-corrections: keep only the speaker's final intent
  Example: "at 2 no actually at 4" → "at 4"
- Fix grammar: subject-verb agreement, articles, tense
- Do NOT remove filler words (already handled)
- Do NOT add punctuation (already handled)
- Do NOT paraphrase, rephrase, or add content
- Do NOT add greetings, sign-offs, or pleasantries
- Output ONLY the cleaned text, nothing else

Input: {transcript_from_4b}
Output:
```

The prompt is deliberately narrow. Filler removal and punctuation are already done by stages 4a/4b, so the LLM only handles what encoder models can't: semantic self-correction resolution and grammar repair.

### Pipeline Flow

```
Raw STT transcript
    │
    ▼
┌─────────────────────────┐
│ 4a: BERT Disfluency     │  ~10ms, always runs
│     Remove fillers,     │
│     tag self-corrections│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 4b: ELECTRA Punctuation │  ~5ms, always runs
│     Add .,?! and caps   │
└───────────┬─────────────┘
            │
            ▼
        ┌───────────┐
        │ Self-corr  │──── No ──→ [Done: inject text]
        │ detected?  │
        └─────┬─────┘
              │ Yes
              ▼
┌─────────────────────────┐
│ 4c: Qwen3 4B LLM       │  ~300ms, conditional
│     Resolve corrections │
│     Fix grammar         │
└───────────┬─────────────┘
            │
            ▼
      [Done: inject text]
```

### Bypass Mode

For users who want raw transcription without any post-processing, a config flag (`polish.enabled: false`) skips Stage 4 entirely.

Individual stages can also be toggled:
```yaml
polish:
  disfluency: true   # 4a
  punctuation: true   # 4b
  llm: true           # 4c (auto-triggered or always-on)
  llm_always: false   # force LLM on every utterance, not just self-corrections
```

---

## Stage 5: Text Injection

### The Problem

Injecting text into an arbitrary focused application on Linux is surprisingly hard. The approach varies by display server.

### X11: xdotool

```bash
xdotool type --clearmodifiers -- "$text"
```

`xdotool type` synthesizes X11 key events. `--clearmodifiers` ensures held modifier keys (from the hotkey) don't interfere. Reliable, well-tested, works in virtually all X11 applications.

### Wayland: ydotool + wtype

Wayland's security model intentionally prevents applications from synthesizing input events in other windows. Two approaches:

**Option A: `ydotool`** (requires `ydotoold` daemon running as root or with uinput access)
- Creates a virtual input device via `/dev/uinput`
- Works across all Wayland compositors
- Requires user to be in the `input` group or `ydotoold` running as a service

**Option B: `wtype`** (wlroots compositors only: Sway, Hyprland, etc.)
- Uses `wlr-virtual-keyboard-unstable-v1` protocol
- No root/uinput needed
- Does NOT work on GNOME Wayland or KDE Wayland

**Option C: Clipboard injection fallback**
- Copy text to clipboard via `wl-copy`
- Simulate `Ctrl+V` via `ydotool`
- Works everywhere but clobbers the user's clipboard (mitigated by saving/restoring)

**Strategy:** Detect the display server and compositor at startup. Use `wtype` if available (wlroots), fall back to `ydotool`, with clipboard injection as a last resort.

### Input Method Framework (Future)

The most robust long-term solution is implementing an IBus or Fcitx5 input method. This integrates natively with the Linux input stack and works on both X11 and Wayland without hacks. However, it's significantly more complex to implement and is deferred to a later version.

---

## System Tray Integration

**Library:** `pystray` with AppIndicator backend (GNOME/Unity) or StatusNotifier backend (KDE).

**States:**
| State | Icon | Tooltip |
|-------|------|---------|
| Idle | Gray microphone | "Linux Whisper — Ready" |
| Recording | Red microphone (animated pulse) | "Recording..." |
| Processing | Yellow microphone | "Transcribing..." |
| Error | Red exclamation | Error description |

**Menu:**
- **Copy Last** — copies the most recent transcription to the clipboard
- **Model** — submenu to hot-swap STT model at runtime (persists to config)
- **Mode** — submenu to switch between auto (default — hold vs double-tap detection), hold, toggle, and VAD-auto modes (persists to config)
- Latency stats (last / avg / p95)
- Settings (opens config file)
- Quit

---

## Configuration

YAML config file at `~/.config/linux-whisper/config.yaml`:

```yaml
# Hotkey
hotkey: "ctrl+shift+e"
mode: "auto"  # auto | hold | toggle | vad-auto

# STT Engine
stt:
  backend: "faster-whisper"  # faster-whisper | moonshine | whisper-cpp
  model: "large-v3-turbo"  # large-v3-turbo | distil-large-v3.5 | medium.en | small.en | moonshine-medium | moonshine-tiny
  threads: 8  # CPU threads for inference (0 = auto)

# Polish Pipeline
polish:
  enabled: true
  disfluency: true       # 4a: BERT filler/repetition removal
  punctuation: true       # 4b: ELECTRA punctuation + capitalization
  llm: true               # 4c: Qwen3 4B self-correction + grammar
  llm_always: false       # true = run LLM on every utterance; false = only on self-corrections
  llm_backend: "llama-cpp"
  llm_model: "Qwen3-4B-Instruct-Q4_K_M"
  llm_threads: 8          # CPU threads for LLM inference (0 = auto)

# Audio
audio:
  sample_rate: 16000
  vad_threshold: 0.5
  silence_timeout: 2.0  # seconds, for VAD auto-stop
  feedback_sounds: true

# Text Injection
inject:
  method: "auto"  # auto | xdotool | ydotool | wtype | clipboard
  typing_delay: 0  # ms between keystrokes, 0 = instant

# UI
tray:
  enabled: true
  show_preview: false  # floating overlay with streaming transcript
```

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.12+ | Ecosystem (ML libs), rapid iteration, adequate perf with native extensions |
| Audio capture | `sounddevice` (PortAudio) | Cross-backend (PipeWire, PulseAudio, ALSA), well-maintained |
| VAD | Silero VAD v5 | ~1ms inference, best open-source VAD, CPU-native |
| STT (default) | faster-whisper (CTranslate2) | INT8 quantization, AVX-512 optimized, large-v3-turbo, 7.25% WER |
| STT (streaming) | Moonshine v2 (ONNX Runtime) | Native streaming, CPU-designed, 245M params, 6.65% WER |
| STT (batch alt) | `whisper.cpp` (via Python bindings) | AVX-512 optimized, GGML quantization |
| Disfluency | BERT token classifier (ONNX) | Deterministic filler removal, ~10ms, zero hallucination |
| Punctuation | ELECTRA-small (ONNX) | Token classification, ~5ms, outperforms GPT on this task |
| LLM | `llama-cpp-python` | GGUF quantized Qwen3 4B, AVX-512 optimized, ~50-70 tok/s |
| Text injection | `ydotool` / `xdotool` / `wtype` | Covers X11 + all major Wayland compositors |
| System tray | `pystray` | AppIndicator + StatusNotifier support |
| Hotkey | `evdev` | Kernel-level, works on X11 + Wayland, no root needed |
| Config | `PyYAML` | Simple, human-readable |
| Packaging | `uv` (deps) + systemd user service | Modern Python tooling, auto-start on login |

### Why Python?

The latency-critical paths (STT inference, LLM inference, audio capture) are all backed by native code (ONNX Runtime C++, llama.cpp C++, PortAudio C). Python is only the orchestration layer — reading configs, managing state, calling into native backends. The overhead is negligible (<5ms for the Python glue between stages).

If profiling reveals Python overhead is significant, the hotkey daemon and audio pipeline can be extracted to a Rust/C companion process communicating via Unix socket.

### Why Not PyTorch?

PyTorch is conspicuously absent from this stack. Reasons:

1. **Runtime overhead:** PyTorch's CUDA context alone is 500-1,200MB. Even on CPU, the framework initialization adds ~1.2GB RAM and 2-5 seconds to startup.
2. **We don't need it.** Moonshine v2 runs on ONNX Runtime. The BERT and ELECTRA models export to ONNX trivially. llama.cpp has its own inference engine. Silero VAD can run via ONNX too.
3. **Dependency hell.** PyTorch pulls in CUDA/ROCm libraries, numpy version constraints, and ~2GB of packages. ONNX Runtime + llama.cpp is ~200MB total.

The one exception: if we add ROCm GPU acceleration in v0.3+, PyTorch-ROCm may become necessary for some model backends. This would be an optional dependency, not a requirement.

---

## Memory Budget

STT and encoder models stay warm in RAM for instant response. The LLM (Qwen3 4B) is **lazy-loaded** — it remains unloaded until the disfluency detector first flags a self-correction, saving ~2.5GB idle RAM.

| Component | RAM (Resident) | Notes |
|-----------|---------------|-------|
| faster-whisper large-v3-turbo (CTranslate2 INT8) | ~4,000MB | Default STT model |
| BERT disfluency (ONNX) | ~110MB | Or ~1.3MB with INT8 distilled variant |
| ELECTRA punctuation (ONNX) | ~60MB | Two 14M-param models |
| Qwen3 4B Q4_K_M (llama.cpp) | ~2,500MB | **Lazy-loaded** — only when self-corrections detected |
| llama.cpp runtime overhead | ~100MB | Only when LLM is loaded |
| ONNX Runtime overhead | ~100MB | Shared across all ONNX models |
| Silero VAD | ~5MB | Tiny model |
| Python + app overhead | ~200MB | asyncio, evdev, sounddevice, pystray |
| **Total (idle, no LLM)** | **~4,475MB** | **~7% of 64GB** |
| **Total (LLM warm)** | **~7,075MB** | **After first self-correction triggers LLM load** |

### Comparison with Alternatives

| Configuration | Idle RAM | LLM-warm RAM | Trade-off |
|---------------|----------|-------------|-----------|
| Moonshine Tiny + no polish | ~350MB | — | Fastest, lowest quality |
| Moonshine Medium + encoder cleanup only | ~870MB | — | Low latency, streaming, no self-correction handling |
| **faster-whisper large-v3-turbo + polish (default)** | **~1,500MB** | **~4,000MB** | **Best quality. LLM lazy-loaded.** |
| faster-whisper large-v3-turbo + Gemma 3 4B | ~1,500MB | ~4,900MB | Best instruction following |

---

## Process Architecture

```
┌──────────────────────────────────────┐
│           Main Process               │
│    (Python asyncio event loop)       │
│                                      │
│  ├─ Hotkey listener (evdev)          │  ← dedicated thread
│  ├─ Audio capture (sounddevice)      │  ← callback thread (PortAudio)
│  ├─ VAD (Silero, ONNX)              │  ← runs in audio callback
│  ├─ System tray (pystray)            │  ← dedicated thread
│  │                                   │
│  ├─ STT inference (faster-whisper)    │  ← async task, CPU threads
│  ├─ Disfluency removal (BERT, ONNX)  │  ← async task, CPU (fast)
│  ├─ Punctuation (ELECTRA, ONNX)      │  ← async task, CPU (fast)
│  ├─ LLM inference (llama.cpp)        │  ← async task, CPU threads
│  └─ Text injection                   │  ← async task, subprocess
└──────────────────────────────────────┘
```

The main process uses `asyncio` for coordination. CPU-bound inference runs in thread pools. The audio callback thread is real-time priority and does minimal work (buffer copy + VAD check).

### Thread Allocation

With 16 cores / 32 threads available:

| Task | Threads | Notes |
|------|---------|-------|
| Audio callback | 1 | Real-time priority, minimal work |
| Hotkey listener | 1 | Blocks on evdev read |
| System tray | 1 | GTK/Qt event loop |
| STT inference | 8 | ONNX Runtime intra-op parallelism |
| LLM inference | 8 | llama.cpp thread pool |
| BERT + ELECTRA | 2 | Small models, fast enough single-threaded |
| asyncio event loop | 1 | Coordination only |
| **Total active during transcription** | **~14** | **Leaves headroom for system** |

STT and LLM never run simultaneously (they're sequential in the pipeline), so their thread pools don't compete. During idle, only the hotkey listener and audio callback threads are active.

### Concurrency Model

1. **Audio callback** (real-time thread): Copies audio to ring buffer. Runs Silero VAD (~1ms). Sets event flag on speech onset/offset.
2. **STT task** (async): Awaits complete audio from ring buffer, feeds to faster-whisper batch engine, collects final transcript.
3. **Polish pipeline** (async): Awaits final transcript from STT. Runs BERT disfluency → ELECTRA punctuation → (conditional) LLM correction. Sequential, fast.
4. **Injector task** (async): Awaits polished text, invokes text injection subprocess.

Tasks are chained via `asyncio.Queue` for backpressure-free handoff.

---

## Error Handling

| Failure | Recovery |
|---------|----------|
| Model not downloaded | Prompt user to run `linux-whisper models download` |
| Audio device not found | Show error in tray, list available devices |
| Hotkey conflict | Warn at startup, suggest alternative |
| Text injection fails | Fall back to clipboard mode, notify user |
| STT returns empty | Discard silently (noise-only activation) |
| LLM generates garbage | Fall back to encoder-only output (4a + 4b) for this utterance |
| LLM times out (>500ms) | Return encoder-only output, log warning |
| ONNX Runtime error | Fall back to whisper.cpp backend |
| Out of memory | Shouldn't happen with 64GB, but: unload LLM, switch to encoder-only mode |

---

## Security Considerations

- **Audio data:** Never written to disk unless `save_transcripts` is explicitly enabled. Ring buffer is overwritten on every utterance.
- **Clipboard:** When using clipboard injection, the original clipboard contents are saved and restored after injection. Clipboard is cleared of transcript text after a 5-second delay.
- **Input group:** The `evdev` hotkey listener requires the user to be in the `input` group. This grants read access to all input devices (keyboards, mice). Document this tradeoff clearly.
- **Model downloads:** Performed over HTTPS from Hugging Face Hub. SHA256 checksums verified. After download, `HF_HUB_OFFLINE=1` is enforced.
- **No telemetry.** No analytics. No phone-home. Ever.

---

## Testing Strategy

### Unit Tests
- VAD accuracy on synthetic audio (speech + silence + noise)
- BERT disfluency detection on annotated test set
- ELECTRA punctuation accuracy on unpunctuated transcripts
- LLM prompt formatting and output parsing (rejects hallucinated content)
- Config loading and validation
- State machine transitions
- Self-correction detection heuristic accuracy

### Integration Tests
- Full pipeline: audio file → STT → polish → text output (no injection)
- Latency benchmarks with regression detection (per-stage and end-to-end)
- Memory usage monitoring (ensure no leaks over 1000+ transcriptions)
- ONNX Runtime memory stability (CTranslate2 has known leak issues; verify ONNX doesn't)

### Manual Tests
- Text injection in: Firefox, Chrome, VS Code, terminal (kitty, alacritty), Slack (Electron), LibreOffice, Obsidian
- X11 and Wayland (GNOME, KDE, Sway, Hyprland)
- Hold-to-talk, toggle, VAD-auto modes
- Long dictation (2+ minutes continuous)
- Noisy environment (fan, music, keyboard)
- Edge cases: empty utterance, single word, numbers, URLs, code snippets
- Self-correction edge cases: multiple corrections, corrections at start/end, nested corrections

---

## Dependencies

```toml
[project]
name = "linux-whisper"
requires-python = ">=3.12"
dependencies = [
    # Core
    "sounddevice>=0.5",
    "numpy>=2.0",
    "pyyaml>=6.0",
    "pystray>=0.19",
    "Pillow>=10.0",

    # VAD + Encoder models
    "onnxruntime>=1.19",   # Runs Silero VAD, Moonshine, BERT, ELECTRA

    # STT
    "moonshine>=0.2",      # Moonshine v2 streaming

    # LLM
    "llama-cpp-python>=0.3",  # Qwen3 4B GGUF inference

    # Input
    "evdev>=1.7",
]

[project.optional-dependencies]
whisper = [
    "whispercpp>=0.1",    # whisper.cpp Python bindings
]
rocm = [
    "onnxruntime-rocm>=1.19",  # ROCm GPU acceleration for ONNX models
]
```

System packages (installed via package manager):
```bash
# Text injection
sudo apt install xdotool ydotool wtype wl-clipboard

# Audio
sudo apt install libportaudio2
```

Managed via `uv` with a `pyproject.toml`. Single source of truth.
