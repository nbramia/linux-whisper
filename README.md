# Linux Whisper

Local voice dictation for Linux. Press a hotkey, speak naturally with filler words and self-corrections, release, and polished text appears at the cursor in under a second. Runs entirely on-device with zero network calls. Built for developers and power users on X11 and Wayland.

## How It Works

1. **Press the hotkey** (default: `Fn`). In auto mode, the system detects your intent: hold > 300ms for hold-to-talk, double-tap for toggle mode. A 750ms pre-roll buffer captures audio from *before* the keypress so the first syllable is never lost.
2. **Speak naturally.** Say "um", repeat yourself, change your mind mid-sentence. Silero VAD v5 monitors speech in real time. The tray icon and floating overlay show when speech is detected.
3. **Release the hotkey.** Audio goes through a 6-stage pipeline: STT transcription (whisper.cpp with GPU acceleration), then a four-stage polish pipeline that removes filler words, adds punctuation, formats numbers/dates, and conditionally invokes a local LLM to resolve self-corrections.
4. **Text appears at the cursor** in whatever application is focused, via xdotool (X11), wtype (wlroots), ydotool (any Wayland), or clipboard fallback.

## Features

- **Auto mode** (default): hold fn > 300ms for hold-to-talk, double-tap for toggle. Plus explicit hold, toggle, and VAD-auto-stop modes via kernel-level evdev (works on X11 and Wayland)
- **GPU-accelerated STT**: whisper.cpp with ROCm HIP on AMD GPUs. ~5-10x faster than CPU (0.3s vs 2.5s for 5s audio). Automatic CPU fallback when GPU is unavailable
- **GPU-accelerated LLM**: Qwen3 4B via llama-cpp with ROCm offload (~2x speedup)
- **Automatic gain control**: normalizes quiet/whispered speech before STT for consistent transcription quality
- **Voice snippets**: say a trigger phrase ("my email") to instantly expand configured text, bypassing the full pipeline
- **Context-aware tone**: detects the focused application (Slack, email, terminal, etc.) and adjusts LLM output tone accordingly
- **Number/date formatting**: spoken forms like "three hundred and fifty" or "march twenty second" are automatically converted to "350" and "March 22nd"
- **Multiple STT backends**: whisper.cpp (default, GPU), faster-whisper (CTranslate2, CPU), Moonshine v2 (ONNX, streaming)
- **Hot-swappable models** from the system tray menu with automatic config persistence
- **Four-stage polish pipeline**: BERT disfluency removal, ELECTRA/rule-based punctuation, spoken-form formatting, conditional LLM correction
- **Pre-roll buffer** (750ms) captures audio before the hotkey press
- **Text injection** auto-detects display server and compositor
- **System tray** with state icons, model/mode switcher, snippets menu, latency stats
- **Floating GTK4 pill overlay** with animated audio level bars
- **Full YAML configuration** with validation

## Architecture

```
Press hotkey         Capture audio        Transcribe            Polish text            Inject
   (evdev)          (sounddevice)       (whisper.cpp)      (BERT+rules+fmt+LLM)    (xdotool/wtype)
     |                   |                   |                    |                     |
     v                   v                   v                    v                     v
 +--------+   +------------------+   +---------------+   +------------------+   +------------+
 | Stage 1|-->| Stage 2          |-->| Stage 3       |-->| Stage 4a/4b/4d/4c|-->| Stage 5    |
 | Hotkey |   | Audio + VAD      |   | STT Engine    |   | Polish Pipeline  |   | Text       |
 | <5ms   |   | Ring buf+preroll |   | ~300ms (GPU)  |   | 4a: <15ms BERT   |   | Injection  |
 |        |   | Silero VAD <1ms  |   | ~2.5s  (CPU)  |   | 4b: <15ms rules  |   | <20ms      |
 +--------+   +------------------+   | AGC applied   |   | 4d: <1ms format  |   +------------+
                                     +---------------+   | 4c: ~200ms LLM*  |
                                                         +------------------+

 * Stage 4c only runs when self-corrections are detected.
 ** Snippet matches bypass the entire polish pipeline.

 With GPU (default):  ~350ms total (simple) / ~550ms (with LLM)
 CPU-only fallback:   ~2.6s total (simple) / ~2.9s (with LLM)
```

**Process isolation:** The GPU STT engine (whisper.cpp/pywhispercpp) runs in a separate subprocess to avoid a ROCm shared-library conflict with onnxruntime (used by Silero VAD, BERT, ELECTRA). The worker process loads the model once, stays warm between transcriptions, and communicates via stdin/stdout pipes. This is transparent — the `WhisperGPUEngine` implements the same `STTEngine` protocol as all other backends.

The polish pipeline uses a hybrid approach:

- **Stage 4a** -- BERT token classifier (or regex fallback) removes filler words ("um", "uh", "like"), repetitions, and false starts. Cannot hallucinate.
- **Stage 4b** -- ELECTRA classifier (or rule-based fallback) adds punctuation and fixes capitalization. Cannot hallucinate.
- **Stage 4d** -- Rule-based formatter converts spoken numbers, dates, times, currency, emails, and phone numbers to their written forms.
- **Stage 4c** -- Qwen3 4B LLM via llama-cpp resolves self-corrections ("at 2, actually 4" becomes "at 4"). Only invoked when Stage 4a flags self-correction patterns. Context-aware: adapts tone based on the focused application. Lazy-loaded to save ~2.5GB RAM when not needed. Has a 3-second timeout and a 2x length hallucination guard.

This split means 80%+ of dictations never touch a generative model.

## STT Models

| Model | Backend | Params | WER | Speed (GPU) | Speed (CPU) | Use Case |
|-------|---------|--------|-----|-------------|-------------|----------|
| **large-v3-turbo** | whisper.cpp | 809M | 7.25% | **~300ms/5s** | ~2.5s/5s | **Default.** Best quality. |
| distil-large-v3.5 | whisper.cpp / faster-whisper | 756M | 7.10% | ~250ms/5s | ~2s/5s | English-only, slightly better WER. |
| moonshine-medium | moonshine | 244.9M | 6.65% | N/A | ONNX streaming | Low latency, streaming capable. |
| moonshine-tiny | moonshine | 33.6M | 12.01% | N/A | ONNX streaming | Instant inference, lower accuracy. |

The default is whisper.cpp with `large-v3-turbo` and ROCm GPU acceleration. On systems without a supported GPU, it falls back to CPU automatically. Models are hot-swappable from the system tray menu at runtime.

## Installation

### Requirements

- Linux with X11 or Wayland
- Python >= 3.12
- For GPU acceleration: AMD GPU with ROCm support (tested on Radeon 8060S / gfx1151 / RDNA 3.5)

### System dependencies

```bash
# Debian/Ubuntu
sudo apt install libportaudio2 xdotool wl-clipboard

# For Wayland (wlroots compositors like Sway/Hyprland)
sudo apt install wtype

# For Wayland (GNOME/KDE)
sudo apt install ydotool
sudo systemctl enable --now ydotool
```

### Install from source

```bash
git clone https://github.com/nbramia/linux-whisper.git
cd linux-whisper
pip install -e ".[full]"
```

The `[full]` extra installs all optional dependencies. Install individual extras as needed:

```bash
pip install -e ".[audio,vad,hotkey,tray]"                # Core without ML backends
pip install -e ".[audio,vad,hotkey,tray,whisper]"         # Add whisper.cpp STT
pip install -e ".[audio,vad,hotkey,tray,whisper,llm]"     # Add LLM polish stage
```

### GPU acceleration (AMD ROCm)

The default `pywhispercpp` and `llama-cpp-python` packages include ROCm support. For optimal LLM performance, rebuild llama-cpp-python with HIP:

```bash
CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151" \
  pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Replace `gfx1151` with your GPU's architecture (check with `rocminfo`).

### Input group setup

The hotkey daemon reads from `/dev/input/event*` via evdev, which requires membership in the `input` group:

```bash
sudo usermod -aG input $USER
```

Log out and back in for the group change to take effect. Verify with:

```bash
linux-whisper listen-keys
```

### Initialize configuration

```bash
linux-whisper config init
```

This creates `~/.config/linux-whisper/config.yaml` with sensible defaults.

### Systemd user service (auto-start on login)

Create `~/.config/systemd/user/linux-whisper.service`:

```ini
[Unit]
Description=Linux Whisper voice dictation
After=graphical-session.target

[Service]
ExecStart=%h/.local/bin/linux-whisper run
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now linux-whisper
```

## Configuration

Config file: `~/.config/linux-whisper/config.yaml`

```yaml
# Hotkey — any key name recognized by evdev, with optional modifiers.
hotkey: "fn"

# Mode: how the hotkey triggers recording.
#   auto     — hold > 300ms = hold-to-talk, double-tap = toggle (default)
#   hold     — hold to record, release to stop
#   toggle   — press once to start, press again to stop
#   vad-auto — press to start, silence auto-stops
mode: "auto"

# Speech-to-text engine
stt:
  backend: "whisper-cpp"       # whisper-cpp | faster-whisper | moonshine
  model: "whisper-large-v3-turbo"
  device: "rocm"               # rocm (GPU) | cpu — auto-falls back to CPU
  threads: 0                   # CPU threads (0 = auto-detect)

# Polish pipeline — cleans up raw transcripts
polish:
  enabled: true                # false to get raw STT output
  disfluency: true             # Stage 4a: remove fillers and repetitions
  punctuation: true            # Stage 4b: add punctuation and capitalization
  formatting: true             # Stage 4d: format numbers, dates, times, currency
  llm: true                    # Stage 4c: resolve self-corrections via LLM
  llm_always: false            # true = run LLM on every utterance
  context_awareness: true      # detect focused app, adjust LLM tone
  llm_backend: "llama-cpp"
  llm_model: "Qwen3-4B-Q4_K_M"
  llm_device: "rocm"           # rocm (GPU) | cpu
  llm_threads: 0

# Audio capture
audio:
  sample_rate: 16000
  vad_threshold: 0.6
  silence_timeout: 0.5
  feedback_sounds: true
  buffer_size: 512
  auto_gain: true              # AGC for quiet/whispered speech

# Text injection
inject:
  method: "auto"               # auto | xdotool | ydotool | wtype | clipboard
  typing_delay: 0

# System tray
tray:
  enabled: true
  show_preview: false

# Voice snippets — trigger phrases that expand to saved text
snippets:
  # "my email": "nathan@example.com"
  # "meeting followup": |
  #   Hi team,
  #   Following up on our meeting...
```

## CLI Reference

```
linux-whisper [--version] [--config PATH] [-v|-vv] COMMAND
```

| Command | Description |
|---------|-------------|
| `run` | Start the dictation service (default if no command given) |
| `run --no-tray` | Start without system tray icon |
| `models list` | List available models with download status |
| `models download MODEL_ID` | Download a model |
| `models default MODEL_ID` | Set the default STT model |
| `config init` | Create default config file |
| `config show` | Print current resolved configuration |
| `config path` | Print config file path |
| `config validate` | Validate config and report errors |
| `listen-keys` | Show live key events from all input devices (diagnostic) |

Verbosity: `-v` for INFO, `-vv` for DEBUG.

## System Tray

The tray icon reflects application state:

| Icon | State | Meaning |
|------|-------|---------|
| Gray circle + mic | Idle | Ready, waiting for hotkey |
| Red circle + mic | Recording (silent) | Hotkey held, no speech detected yet |
| Green circle + mic + arcs | Recording (speech) | Hotkey held, speech detected |
| Amber circle + mic | Processing | Transcribing and polishing |
| Dark red circle + X | Error | Pipeline failure |

Right-click context menu:

- **Copy Last** -- copies the most recent transcription to the clipboard
- **Model** -- submenu to hot-swap STT model at runtime (persists to config)
- **Mode** -- submenu to switch between auto, hold, toggle, and VAD-auto modes
- **Snippets** -- shows configured voice snippet triggers
- **Latency** -- displays last and rolling-average end-to-end latency
- **Quit** -- clean shutdown

## Development

### Project structure

```
src/linux_whisper/
    __init__.py
    app.py              # Main orchestrator, wires all stages
    audio.py            # Ring buffer, Silero VAD, audio capture, AGC, feedback tones
    cli.py              # CLI entry point and subcommands
    config.py           # YAML config loading, validation, dataclasses
    focus.py            # Focused app detection (X11/Sway/Hyprland), context-aware prompts
    hotkey.py           # evdev global hotkey daemon
    snippets.py         # Voice snippet matching (fuzzy, case-insensitive)
    state.py            # Async state machine (IDLE/RECORDING/PROCESSING/ERROR)
    overlay.py          # GTK4 floating pill with audio level bars
    tray.py             # pystray system tray with icon generation
    stt/
        engine.py       # STTEngine protocol and factory
        whisper_gpu.py  # GPU STT engine — spawns isolated worker subprocess
        whisper_gpu_worker.py  # Worker process — loads pywhispercpp (ROCm)
        whisper_cpp.py  # whisper.cpp CPU backend (in-process)
        faster_whisper.py  # CTranslate2 INT8 backend (CPU-only)
        moonshine.py    # ONNX Runtime backend (streaming)
    polish/
        pipeline.py     # Four-stage orchestrator
        disfluency.py   # BERT ONNX / regex filler removal
        punctuation.py  # ELECTRA ONNX / rule-based punctuation
        formatting.py   # Rule-based number/date/time/currency/email formatting
        llm.py          # Qwen3 4B via llama-cpp-python (ROCm GPU offload)
    inject/
        injector.py     # Display server detection, xdotool/wtype/ydotool/clipboard
tests/
    conftest.py         # Shared fixtures, optional dependency stubs
    test_app.py         # Pipeline orchestration, state transitions, latency
    test_audio.py       # Ring buffer, tones, AGC
    test_cli.py
    test_config.py
    test_focus.py       # Focused app detection
    test_hotkey.py      # Key parsing, all 4 modes, callbacks
    test_inject.py
    test_polish.py      # All polish stages + pipeline integration
    test_snippets.py    # Snippet matching
    test_state.py
    test_stt.py         # STT engine protocol + device config
```

### Running tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=linux_whisper
```

Test markers:

```bash
pytest -m "not slow"          # Skip tests that load models
pytest -m "not integration"   # Skip full-pipeline integration tests
```

### Linting

```bash
ruff check src tests
ruff format src tests
```

## Technical Decisions

**Why GPU-first with CPU fallback.** The primary target is AMD systems with ROCm-capable GPUs. whisper.cpp and llama.cpp both support HIP via ggml. GPU STT brings latency from ~2.5s to ~300ms for a 5-second recording. On systems without ROCm, the same code paths fall back to CPU automatically.

**Why hybrid polish instead of a single LLM.** A single generative LLM would pay 300-500ms latency on every utterance, even trivial ones. Filler removal and punctuation are sequence-labeling tasks where encoder models are faster, deterministic, and cannot hallucinate. The LLM only activates for the ~20% of utterances containing self-corrections, saving latency and RAM for the common case.

**Why pre-roll buffer.** Users often start speaking the instant they press the hotkey, or even slightly before. The ring buffer continuously captures audio, and the 750ms pre-roll is fed into the STT engine at recording start. Without this, the first word is consistently clipped.

**Why evdev for hotkeys.** Alternatives like pynput are X11-only. The keyboard library requires root. D-Bus global shortcut portals are desktop-specific. evdev reads directly from `/dev/input/event*` and works universally on X11 and Wayland with only `input` group membership.

**Why no PyTorch.** All inference runs through ONNX Runtime, whisper.cpp, or llama.cpp. PyTorch would add ~1.2 GB of framework overhead and CUDA dependencies for no benefit.

## License

MIT License. See [LICENSE](LICENSE) for details.
