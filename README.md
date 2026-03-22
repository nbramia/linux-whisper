# Linux Whisper

Local voice dictation for Linux. Press a hotkey, speak naturally with filler words and self-corrections, release, and polished text appears at the cursor in under a second. Runs entirely on-device with zero network calls. Built for developers and power users on X11 and Wayland.

## How It Works

1. You hold a hotkey (default: `Fn`). A 750ms pre-roll buffer captures audio from *before* the keypress so the first syllable is never lost.
2. You speak naturally. Silero VAD v5 monitors speech activity in real time. The system tray icon and a floating pill overlay show whether speech is detected.
3. You release the hotkey. The audio is fed through a 6-stage pipeline: STT transcription (faster-whisper with built-in VAD filtering), then a three-stage polish pipeline that removes filler words (BERT), adds punctuation (ELECTRA / rules), and conditionally invokes a local LLM to resolve self-corrections.
4. Polished text is injected at the cursor position in whatever application is focused, via xdotool (X11), wtype (wlroots Wayland), ydotool (any Wayland), or clipboard fallback.

The entire pipeline targets under 800ms end-to-end on a modern multi-core CPU.

## Features

- Hold-to-talk, toggle, and VAD-auto-stop hotkey modes via kernel-level evdev (works on X11 and Wayland)
- Configurable hotkey with modifier support (`fn`, `ctrl+shift+e`, etc.)
- Pre-roll buffer (750ms) captures audio before the hotkey press to avoid clipping first words
- Silero VAD v5 voice activity detection with adaptive noise floor
- Audio feedback tones (rising chirp on start, falling on stop) generated programmatically
- Multiple STT backends: faster-whisper (CTranslate2, INT8), Moonshine v2 (ONNX), whisper.cpp
- Hot-swappable models from the system tray menu with automatic config persistence
- Three-stage polish pipeline: BERT disfluency removal, ELECTRA/rule-based punctuation, conditional LLM correction
- LLM stage (Qwen3 4B via llama-cpp) lazy-loaded only when self-corrections are detected, with hallucination guard and timeout
- Text injection auto-detects display server and compositor (xdotool, wtype, ydotool, clipboard)
- Clipboard injection saves and restores original clipboard contents
- System tray with state icons (idle/recording/speech/processing/error), model switcher, mode switcher, latency stats, copy-last-transcription
- Floating GTK4 pill overlay with animated audio level bars
- Full YAML configuration with validation
- CLI for model management, config inspection, and key diagnostics

## Architecture

```
Press hotkey         Capture audio        Transcribe           Polish text           Inject
   (evdev)          (sounddevice)      (faster-whisper)    (BERT+rules+LLM)     (xdotool/wtype)
     |                   |                   |                   |                    |
     v                   v                   v                   v                    v
 +--------+   +------------------+   +---------------+   +-----------------+   +------------+
 | Stage 1|-->| Stage 2          |-->| Stage 3       |-->| Stage 4a/4b/4c  |-->| Stage 5    |
 | Hotkey |   | Audio + VAD      |   | STT Engine    |   | Polish Pipeline |   | Text       |
 | <5ms   |   | Ring buf+preroll |   | <500ms batch  |   | 4a: <15ms BERT  |   | Injection  |
 |        |   | Silero VAD <1ms  |   | INT8 on CPU   |   | 4b: <15ms rules |   | <20ms      |
 +--------+   +------------------+   +---------------+   | 4c: <350ms LLM* |   +------------+
                                                          +-----------------+

 * Stage 4c only runs when self-corrections are detected.

 Simple utterance (no self-corrections):  <365ms total
 Complex utterance (LLM invoked):         <715ms total
```

The polish pipeline uses a hybrid approach:

- **Stage 4a** -- BERT token classifier (or regex fallback) removes filler words ("um", "uh", "like"), repetitions, and false starts. Sequence labeling cannot hallucinate.
- **Stage 4b** -- ELECTRA classifier (or rule-based fallback) adds punctuation and fixes capitalization. Also cannot hallucinate.
- **Stage 4c** -- Qwen3 4B LLM via llama-cpp resolves self-corrections ("at 2, actually 4" becomes "at 4") and fixes grammar. Only invoked when Stage 4a flags self-correction patterns. Lazy-loaded to save ~2.5GB RAM when not needed. Has a 3-second timeout and a 2x length hallucination guard.

This split means 80%+ of dictations never touch a generative model.

## STT Models

| Model | Backend | Params | WER | Speed | RAM | Use Case |
|-------|---------|--------|-----|-------|-----|----------|
| **large-v3-turbo** | faster-whisper | 809M | 7.25% | Batch, INT8 on CPU | ~4GB (Q8) | **Default.** Best quality on CPU. |
| distil-large-v3.5 | faster-whisper | 756M | 7.10% | Batch, INT8 on CPU | ~3.5GB (Q8) | English-only, slightly better WER. |
| medium.en | faster-whisper | 769M | ~8% | Batch, INT8 on CPU | ~2GB | Faster, English-only. |
| small.en | faster-whisper | 244M | ~10% | Batch, INT8 on CPU | ~1GB | Fastest Whisper variant. |
| moonshine-medium | moonshine | 244.9M | 6.65% | ONNX, streaming | ~500MB | Low latency, streaming capable. |
| moonshine-tiny | moonshine | 33.6M | 12.01% | ONNX, streaming | ~150MB | Instant inference, lower accuracy. |

The default is `faster-whisper` with `large-v3-turbo` (INT8 quantization via CTranslate2). Models are hot-swappable from the system tray menu at runtime without restarting the application. The selected model persists across restarts.

## Installation

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
git clone https://github.com/nathanramia/linux-whisper.git
cd linux-whisper
pip install -e ".[full]"
```

The `[full]` extra installs all optional dependencies: sounddevice, onnxruntime, moonshine, llama-cpp-python, evdev, pystray, and Pillow. Install individual extras as needed:

```bash
pip install -e ".[audio,vad,hotkey,tray]"       # Core without ML backends
pip install -e ".[audio,vad,hotkey,tray,llm]"    # Add LLM polish stage
```

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
# Examples: "fn", "ctrl+shift+e", "super+grave"
hotkey: "fn"

# Mode: how the hotkey triggers recording.
#   hold     — hold to record, release to stop (default)
#   toggle   — press once to start, press again to stop
#   vad-auto — press to start, silence auto-stops
mode: "hold"

# Speech-to-text engine
stt:
  backend: "faster-whisper"   # faster-whisper | moonshine | whisper-cpp
  model: "large-v3-turbo"     # see model table above
  threads: 0                  # CPU threads for inference (0 = auto-detect)

# Polish pipeline — cleans up raw transcripts
polish:
  enabled: true               # false to get raw STT output
  disfluency: true            # Stage 4a: remove fillers and repetitions
  punctuation: true           # Stage 4b: add punctuation and capitalization
  llm: true                   # Stage 4c: resolve self-corrections via LLM
  llm_always: false           # true = run LLM on every utterance, not just self-corrections
  llm_backend: "llama-cpp"    # LLM inference backend
  llm_model: "Qwen3-4B-Q4_K_M"  # GGUF model filename (in ~/.cache/linux-whisper/models/llm/)
  llm_threads: 0              # CPU threads for LLM inference (0 = auto-detect)

# Audio capture
audio:
  sample_rate: 16000           # 16kHz required by all STT models
  vad_threshold: 0.6           # Silero VAD speech probability threshold (0.0-1.0)
  silence_timeout: 0.5         # Seconds of silence before VAD auto-stop
  feedback_sounds: true        # Play start/stop audio cues
  buffer_size: 512             # Samples per audio callback (32ms at 16kHz)

# Text injection
inject:
  method: "auto"               # auto | xdotool | ydotool | wtype | clipboard
  typing_delay: 0              # Milliseconds between injected keystrokes (0 = instant)

# System tray
tray:
  enabled: true                # false to run headless
  show_preview: false          # Floating overlay with streaming transcript (future)
```

## CLI Reference

```
linux-whisper [--version] [--config PATH] [-v|-vv] COMMAND
```

| Command | Description |
|---------|-------------|
| `run` | Start the dictation service (default if no command given) |
| `run --no-tray` | Start without system tray icon |
| `models list` | List available models with download status, params, WER, RAM |
| `models download MODEL_ID` | Download a model (models also auto-download on first use) |
| `models default MODEL_ID` | Set the default STT model |
| `config init` | Create default config file at `~/.config/linux-whisper/config.yaml` |
| `config show` | Print current resolved configuration as YAML |
| `config path` | Print config file path |
| `config validate` | Validate current config and report errors |
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

- **Copy Last** -- copies the most recent transcription to the clipboard (via wl-copy or xclip)
- **Model** -- submenu to hot-swap STT model at runtime (persists to config)
- **Mode** -- submenu to switch between hold, toggle, and VAD-auto modes (persists to config)
- **Latency** -- displays last and rolling-average end-to-end latency
- **Quit** -- clean shutdown

## Memory Footprint

| Configuration | Idle RAM | Notes |
|---------------|----------|-------|
| faster-whisper large-v3-turbo + polish (no LLM loaded) | ~1.5 GB | LLM lazy-loads only when self-corrections detected |
| faster-whisper large-v3-turbo + polish (LLM warm) | ~4 GB | After first self-correction triggers LLM load |
| Moonshine medium + polish (no LLM) | ~870 MB | Lighter STT model |
| Moonshine tiny + no polish | ~350 MB | Minimal configuration |

The LLM (Qwen3 4B Q4_K_M, ~2.5 GB) is intentionally lazy-loaded. It stays unloaded until the disfluency detector first flags a self-correction, keeping idle memory at ~1.5 GB for the default configuration. Once loaded, it remains warm in RAM for subsequent use.

## Technical Decisions

**Why CPU-first.** The primary target is AMD CPUs with AVX-512 (specifically Ryzen AI MAX+ 395). CTranslate2 and llama.cpp both have explicit AVX-512 code paths. No NVIDIA GPU is assumed. ROCm for RDNA 3.5 iGPUs is experimental and deferred to a future release.

**Why hybrid polish instead of a single LLM.** A single generative LLM would pay 300-500ms latency on every utterance, even trivial ones. Filler removal and punctuation are sequence-labeling tasks where encoder models are faster, deterministic, and cannot hallucinate. The LLM only activates for the ~20% of utterances containing self-corrections, saving latency and RAM for the common case.

**Why pre-roll buffer.** Users often start speaking the instant they press the hotkey, or even slightly before. The ring buffer continuously captures audio, and the 750ms pre-roll is fed into the STT engine at recording start. Without this, the first word is consistently clipped.

**Why faster-whisper over Moonshine as default.** Despite Moonshine's streaming capability and smaller footprint, faster-whisper with large-v3-turbo produces noticeably better transcription quality in practice. The built-in Silero VAD filter in faster-whisper also handles silence trimming well. Moonshine remains available as an alternative for users who prefer lower latency or lower memory usage.

**Why evdev for hotkeys.** Alternatives like pynput are X11-only. The keyboard library requires root. D-Bus global shortcut portals are desktop-specific. evdev reads directly from `/dev/input/event*` and works universally on X11 and Wayland with only `input` group membership.

**Why no PyTorch.** All inference runs through ONNX Runtime or llama.cpp. PyTorch would add ~1.2 GB of framework overhead and CUDA dependencies for no benefit on a CPU-first architecture.

## Development

### Project structure

```
src/linux_whisper/
    __init__.py
    app.py              # Main orchestrator, wires all stages
    audio.py            # Ring buffer, Silero VAD, audio capture, feedback tones
    cli.py              # CLI entry point and subcommands
    config.py           # YAML config loading, validation, dataclasses
    hotkey.py           # evdev global hotkey daemon
    state.py            # Async state machine (IDLE/RECORDING/PROCESSING/ERROR)
    overlay.py          # GTK4 floating pill with audio level bars
    tray.py             # pystray system tray with icon generation
    stt/
        engine.py       # STTEngine protocol and factory
        faster_whisper.py  # CTranslate2 INT8 backend
        moonshine.py    # ONNX Runtime backend
    polish/
        pipeline.py     # Three-stage orchestrator
        disfluency.py   # BERT ONNX / regex filler removal
        punctuation.py  # ELECTRA ONNX / rule-based punctuation
        llm.py          # Qwen3 4B via llama-cpp-python
    inject/
        injector.py     # Display server detection, xdotool/wtype/ydotool/clipboard
tests/
    conftest.py
    test_audio.py
    test_cli.py
    test_config.py
    test_inject.py
    test_polish.py
    test_state.py
    test_stt.py
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

## License

MIT
