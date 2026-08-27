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

**Key constraint:** No NVIDIA GPU. All CUDA-dependent tools (faster-whisper GPU mode, CTranslate2 CUDA, NeMo) are unavailable.

**Key advantage:** ROCm 7.2 with gfx1151 support enables GPU-accelerated inference via ggml's HIP backend. Both whisper.cpp (STT) and llama.cpp (LLM) run on the Radeon 8060S iGPU. AVX-512 with VNNI and BF16 provide fast CPU fallback. 64GB unified RAM eliminates memory pressure.

## System Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Linux Whisper                                │
│                                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │  Input    │→│  Audio    │→│   STT    │→│  Polish Pipeline    │   │
│  │  Manager  │  │  Pipeline │  │  Engine  │  │  (hybrid 4-stage)  │   │
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

| Stage | Component | Latency (GPU) | Latency (CPU) | Notes |
|-------|-----------|--------------|---------------|-------|
| 1 | Hotkey detection | < 5ms | < 5ms | Kernel-level evdev input event |
| 2 | Audio capture + VAD + AGC | < 10ms | < 10ms | PipeWire stream, Silero VAD, auto gain control |
| 3 | Speech-to-text | **~285ms** | ~2.5s | whisper.cpp large-v3-turbo (GPU via ROCm HIP) |
| 4a | Disfluency removal | < 15ms | < 15ms | BERT token classifier / regex fallback |
| 4b | Punctuation + caps | < 15ms | < 15ms | ELECTRA-small classifier / rule-based |
| 4d | Number/date formatting | < 1ms | < 1ms | Rule-based spoken-form conversion |
| 4c | Self-correction + grammar | **~150ms** | ~370ms | Qwen3-4B-Instruct-2507 Q4_K_M (GPU), only when needed |
| -- | Focused app detection | < 10ms | < 10ms | xdotool/swaymsg/hyprctl subprocess |
| 5 | Text injection | < 20ms | < 20ms | ydotool/xdotool/wtype/clipboard |
| **Total (simple)** | | **~340ms** | ~2.6s | **No self-corrections detected** |
| **Total (complex)** | | **~490ms** | ~2.9s | **Self-corrections present → LLM invoked** |

Stage 3 (STT) runs in batch mode after recording ends. The default backend is **whisper.cpp large-v3-turbo** with ROCm GPU acceleration via ggml's HIP backend, running in a **separate subprocess** to avoid a shared-library conflict with onnxruntime (both link `libamdhip64`). The `WhisperGPUEngine` spawns a worker process that loads pywhispercpp, communicates via stdin/stdout pipes, and stays warm between transcriptions. On systems without ROCm, it falls back to CPU automatically.

Parakeet TDT v3 is also available and is faster, but loses badly on real dictation because it does not do inverse text normalisation — see "Parakeet TDT Backend" under Testing Strategy. Stages 4a, 4b, and 4d are fast encoder/rule-based models. Stage 4c (generative LLM) is only invoked when the disfluency detector flags self-corrections. Voice snippet matches bypass the entire polish pipeline.

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

**Model:** Silero VAD v6

**Window size is 576 samples (36ms at 16kHz), not 512.** v5 used 512; v6 requires 576. This is a trap rather than a detail: the v6 graph accepts a 512-sample window without raising, then returns ~0.001 for every window including unambiguous speech. Voice activity detection is silently dead while logs, tests, and hold-to-talk dictation all still look healthy — only VAD-driven auto-stop breaks. Measured on LibriSpeech test-clean: 0.0% of speech frames cross the 0.6 threshold at 512, versus 69-90% at 576.

The model declares its input dimension dynamically, so the window size cannot be read from the graph. `tests/benchmarks/run.py --suite vad` is the guard: it scores speech detection rate plus false-positive rates on silence and low-level noise, and the compare gate fails when the speech rate collapses.

The capture blocksize (`audio.buffer_size`, 512) and the VAD window are deliberately independent — an accumulator re-chunks capture blocks into VAD windows.

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
| 6 | NVIDIA Parakeet-TDT 0.6B v3 | 6.34% | 600M | Yes (INT8 ONNX via `onnx-asr`) | ← Available, not default |
| — | Moonshine v2 Medium | 6.65% | 245M | Yes — designed for it | ← Our alternative (streaming) |
| 8 | Distil-Whisper v3.5 | 7.10% | 756M | Yes (faster-whisper) | ← Available option |
| 9 | Whisper large-v3 | 7.14% | 1.55B | Slow | |
| 10 | Whisper large-v3-turbo | 7.25% | 809M | Yes (faster-whisper INT8) | ← Previous default, still supported |

The "CPU-Viable: No (NeMo/CUDA)" note against Parakeet was true when this table was written and is not any more — `onnx-asr` runs the INT8 ONNX export on plain ONNX Runtime with no NeMo and no CUDA.

Note what this table does **not** measure: every WER here comes from read-prose corpora containing no digits, symbols, or filenames. See "Parakeet TDT Backend" below for why that inverted the ranking on real dictation.

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

**Primary: Qwen3-4B-Instruct-2507 (Q4_K_M)**
- IFEval ~87.8 — second-best instruction following, critical for "do NOT paraphrase"
- 2.5GB Q4_K_M — 0.9GB lighter than Gemma 3 4B with only ~2.4% lower IFEval
- ~50-70 tok/s on AVX-512 — generates 20-50 cleanup tokens in 300-700ms
- Strong multilingual foundation for future language support
- Served via `llama-cpp-python` (GGUF format), kept warm in RAM
- **Instruct-only.** The original Qwen3-4B was a hybrid-reasoning checkpoint that needed a `/no_think` suffix appended to every system prompt to suppress reasoning traces. That workaround is gone.

**The prompt's "delete nothing else" rule is load-bearing.** Without it this checkpoint compresses past the self-correction — dropping qualifiers like "just" and leading subjects like "Let's" — which reads as a summary rather than a transcript. Benchmarked: adding the rule moved exact-match from 40.9% to 45.5%.

**Why not Qwen3.5-4B?** Not for the reason originally recorded. `llama-cpp-python` was rebuilt from source with HIP for gfx1151 (0.3.16 → 0.3.34, vendoring llama.cpp from July 2026), and Qwen3.5-4B now loads and runs on ROCm. It was then benchmarked against Qwen3-4B-Instruct-2507 on the polish suite and **lost on latency for no measurable quality gain**:

| | Qwen3-4B-Instruct-2507 | Qwen3.5-4B |
|---|---|---|
| polish WER | 0.1070 | 0.1070 |
| exact match | 62.5% | 62.5% |
| stage 4c p50 | **141ms** | 262ms |
| polish total p95 | **200ms** | 338ms |

31 of 32 fixtures produced byte-identical output. The one that differed favoured Qwen3.5 semantically — on `self-correction-quantity` it kept the repair ("fifteen") where 2507 kept the reparandum ("fifty") — but both scored the same WER because neither converts the number to a digit (see #32).

So the extra capability does not show on a task this narrow, and +138ms p95 is not worth one fixture. **2507 stays the default.** Qwen3.5-4B remains selectable via `polish.llm_model` if the polish task ever widens.

The rebuild was still worth doing: it is what makes Qwen3-ASR available for the STT fallback path, and it removes the version ceiling on every future model.

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
│ 4c: Qwen3-4B-Instr LLM │  ~150ms, conditional
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

## Recording Overlay

A floating pill (200x40, rounded, 16 animated audio-level bars) that appears on
recording start and disappears on stop — WisprFlow-style, so feedback lives
where you're looking instead of the top bar. Renders via GTK3 + Cairo, kept
isolated from asyncio and the real-time audio thread in its own daemon thread.

**Why GTK3, not GTK4 layer-shell.** GTK4 removed `gtk_window_move()` outright
and has no positioning API on Wayland without `wlr-layer-shell`, which Mutter
(GNOME's compositor) does not implement — that rules out a GNOME layer-shell
overlay at any GTK4 version. The one path that positions correctly, verified
on GNOME 46 / Mutter: **GTK3 running through XWayland**, with
`GDK_BACKEND=x11` forced only for the moment the overlay thread opens its
display connection (never mutated process-wide — the backend is chosen once
per process, so the override is restored immediately after), and
`Gtk.WindowType.POPUP` for an override-redirect window.

**Focus safety.** `Gtk.WindowType.POPUP` never takes input focus, and the
window additionally calls `set_accept_focus(False)`, `set_can_focus(False)`,
and `set_focus_on_map(False)`. This is load-bearing, not cosmetic: text
injection (Stage 5) delivers to whatever window has focus, so an overlay that
could steal focus would redirect dictated text into itself instead of the
target application.

**Positioning.** The target monitor is resolved from the **pointer**
(`Gdk.Display.get_monitor_at_point()`), not window focus — `xdotool
getactivewindow` fails on this desktop because the focused window is a native
Wayland surface, so X11 focus queries can't resolve it. `move()` is
re-asserted after `show_all()`, since the window manager can reposition the
window on map. `overlay.position` (`center`, `bottom-center`, `top-center`)
controls the vertical anchor; horizontal placement is always centered on the
monitor.

**Live levels.** `push_audio_level()` is called from the audio monitor loop
in `app.py` (the same `asyncio` loop that drives `set_speech_active()` for
the tray), pushing the peak amplitude of the last 100ms alongside the
existing RMS-based speech detection. It is never called from the sounddevice
callback — that thread runs at 32ms intervals and is a documented escalation
boundary.

**Failure mode.** If GTK 3.0 / PyGObject isn't installed, `_setup_overlay()`
logs a WARNING naming the reason and the app continues without the pill (the
system tray remains as a fallback indicator). This used to fail silently.

---

## System Tray Integration

The floating overlay above is the primary recording indicator; the tray
remains for the context menu, mode/model switching, and latency stats.

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
  llm: true               # 4c: Qwen3-4B-Instruct-2507 self-correction + grammar
  llm_always: false       # true = run LLM on every utterance; false = only on self-corrections
  llm_backend: "llama-cpp"
  llm_model: "Qwen3-4B-Instruct-2507-Q4_K_M"
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

# Recording overlay (floating pill with audio level bars)
overlay:
  enabled: true
  position: "center"  # center | bottom-center | top-center
```

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.12+ | Ecosystem (ML libs), rapid iteration, adequate perf with native extensions |
| Audio capture | `sounddevice` (PortAudio) | Cross-backend (PipeWire, PulseAudio, ALSA), well-maintained |
| VAD | Silero VAD v6 | ~0.1ms inference, best open-source VAD, CPU-native |
| STT (default) | faster-whisper (CTranslate2) | INT8 quantization, AVX-512 optimized, large-v3-turbo, 7.25% WER |
| STT (streaming) | Moonshine v2 (ONNX Runtime) | Native streaming, CPU-designed, 245M params, 6.65% WER |
| STT (batch alt) | `whisper.cpp` (via Python bindings) | AVX-512 optimized, GGML quantization |
| Disfluency | BERT token classifier (ONNX) | Deterministic filler removal, ~10ms, zero hallucination |
| Punctuation | ELECTRA-small (ONNX) | Token classification, ~5ms, outperforms GPT on this task |
| LLM | `llama-cpp-python` | GGUF quantized Qwen3-4B-Instruct-2507, AVX-512 optimized, ~50-70 tok/s |
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

STT and encoder models stay warm in RAM for instant response. The LLM (Qwen3-4B-Instruct-2507) is **lazy-loaded** — it remains unloaded until the disfluency detector first flags a self-correction, saving ~2.5GB idle RAM.

| Component | RAM (Resident) | Notes |
|-----------|---------------|-------|
| faster-whisper large-v3-turbo (CTranslate2 INT8) | ~4,000MB | Default STT model |
| BERT disfluency (ONNX) | ~110MB | Or ~1.3MB with INT8 distilled variant |
| ELECTRA punctuation (ONNX) | ~60MB | Two 14M-param models |
| Parakeet TDT 0.6B v3 INT8 (ONNX) | ~1,200MB | Default STT backend, resident in the main process |
| Qwen3-4B-Instruct-2507 Q4_K_M (llama.cpp) | ~2,500MB | **Lazy-loaded** — only when self-corrections detected |
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

### Parakeet TDT Backend (available, not default)

`parakeet` held the default slot briefly on LibriSpeech numbers and lost it on real dictation. The reversal is the most important measurement in this document, because it shows the corpus was answering a different question than the one that matters.

**On LibriSpeech (read audiobook prose):**

| | whisper.cpp large-v3-turbo | Parakeet TDT 0.6B v3 |
|---|---|---|
| test-clean WER | 1.48% | **0.54%** |
| test-other WER | 1.43% | 1.43% |
| latency p50 | 285ms | **191ms** |

**On 28 recorded dictation clips — real voice, the actual use case:**

| subset | Parakeet | whisper.cpp |
|--------|----------|-------------|
| all 28 clips | 49.3% | **21.5%** |
| 22 literal-reference clips | 45.3% | **17.4%** |
| — code / filenames (13) | 64.3% | **25.6%** |
| — plain prose (5) | 4.9% | **0.0%** |

whisper wins every subset, by 2.5x on technical material.

**LibriSpeech got it wrong because it contains no digits, symbols, filenames, or technical vocabulary.** It therefore never tests inverse text normalisation — and ITN is most of what dictation is:

| dictated | Parakeet | whisper.cpp |
|---|---|---|
| "twenty three minutes … forty dollars" | `twenty three minutes and cost forty dollars` | `23 minutes and cost $40` |
| "max underscore default underscore threads" | `max underscore default underscore threads` | `max_default_threads` |
| "zero point three point three four" | `zero point three point three four` | `0.3.34` |
| "server hyphen test dot s h" | *(empty)* | `server-test.sh` |
| "R O C M … G G M L" | `R O C M … G G M L` | `ROCM … GGML` |

Parakeet transcribes the spoken form; whisper renders the written form. For dictation only the written form is useful.

**Parakeet also returns empty transcripts on some clips** — two of 28 produced nothing at all, deterministically across repeated runs. That is silent data loss, not a degraded transcript.

Parakeet stays selectable via `stt.backend: parakeet` and is genuinely faster with a real edge on clean read prose. It runs INT8 ONNX on the **CPU** execution provider (the GPU path would need onnxruntime's ROCm EP, the same `libamdhip64` conflict that put `whisper_gpu` in a subprocess). Its default thread count is capped at 8 — `cpu_count()` oversubscribes it badly, costing ~80% latency.

**Licence:** CC-BY-4.0, not MIT. **Memory:** ~1.2GB resident in the main process, versus whisper.cpp's ~4GB in the GPU subprocess.

### Model Benchmarks

`tests/benchmarks/` scores the real stack — real models, real audio — and gates every model swap. It is deliberately outside the default pytest run, since CI must mock all models.

```bash
python -m tests.benchmarks.run --suite all --label candidate --out /tmp/cand.json
python -m tests.benchmarks.run --compare tests/benchmarks/baseline/current.json /tmp/cand.json
```

| Suite | Fixtures | Scores |
|-------|----------|--------|
| `stt` | LibriSpeech `test-clean`, or user clips via `--fixtures-dir` | WER, per-utterance latency, RTFx |
| `polish` | Text pairs, no audio needed | WER vs expected, exact match, punctuation F1, capitalisation, per-stage latency |

`--compare` exits non-zero on regression, so it works as a merge gate. Baselines are hardware-specific. See `tests/benchmarks/README.md`.

**Fallback behaviour matters when reading results.** Stages 4a and 4b degrade to regex and rule-based implementations when their ONNX models are absent from `~/.cache/linux-whisper/models/`. This is silent apart from an INFO log, and sub-millisecond stage latencies in a benchmark run are the tell. A polish benchmark taken without those models scores the fallbacks, not the models described above.

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
    "llama-cpp-python>=0.3",  # Qwen3-4B-Instruct-2507 GGUF inference

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
