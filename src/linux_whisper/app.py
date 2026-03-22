"""Main application orchestration — connects all pipeline stages via asyncio."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import TYPE_CHECKING

import numpy as np

from linux_whisper.config import Config
from linux_whisper.state import AppState, StateMachine

if TYPE_CHECKING:
    from linux_whisper.audio import AudioPipeline
    from linux_whisper.hotkey import HotkeyDaemon
    from linux_whisper.inject import TextInjector
    from linux_whisper.overlay import Overlay
    from linux_whisper.polish.pipeline import PolishPipeline
    from linux_whisper.snippets import SnippetMatcher
    from linux_whisper.stt.engine import STTEngine
    from linux_whisper.tray import SystemTray

logger = logging.getLogger(__name__)


class App:
    """Main application coordinating all pipeline stages."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = StateMachine()
        self._shutdown_event = asyncio.Event()

        # Latency tracking
        self._latencies: list[float] = []
        self._max_latency_history = 100

        # Event loop reference for thread-safe callbacks
        self._loop: asyncio.AbstractEventLoop | None = None

        # Components initialized in setup()
        self._audio: AudioPipeline | None = None
        self._hotkey: HotkeyDaemon | None = None
        self._stt: STTEngine | None = None
        self._polish: PolishPipeline | None = None
        self._snippets: SnippetMatcher | None = None
        self._injector: TextInjector | None = None
        self._tray: SystemTray | None = None
        self._overlay: Overlay | None = None

    async def setup(self) -> None:
        """Initialize all components. Call before run()."""
        logger.info("Initializing Linux Whisper v0.1.0")
        errors = self.config.validate()
        if errors:
            for e in errors:
                logger.error("Config error: %s", e)
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        await self._setup_audio()
        await self._setup_stt()
        await self._setup_polish()
        await self._setup_snippets()
        await self._setup_injector()
        await self._setup_hotkey()
        await self._setup_tray()
        await self._setup_overlay()

        # Wire state change listener to tray
        if self._tray:
            self.state.on_state_change(lambda _old, new: self._tray.update_state(new))

        logger.info("All components initialized")

    async def _setup_audio(self) -> None:
        from linux_whisper.audio import AudioPipeline

        self._audio = AudioPipeline(self.config.audio)
        logger.info("Audio pipeline ready")

    async def _setup_stt(self) -> None:
        from linux_whisper.stt.engine import create_engine

        self._stt = create_engine(self.config)
        logger.info("STT engine ready: %s", self.config.stt.backend)

    async def _setup_polish(self) -> None:
        if not self.config.polish.enabled:
            logger.info("Polish pipeline disabled")
            return
        from linux_whisper.polish.pipeline import PolishPipeline

        self._polish = PolishPipeline(self.config.polish)
        logger.info("Polish pipeline ready")

    async def _setup_snippets(self) -> None:
        if not self.config.snippets:
            logger.info("No snippets configured")
            return
        from linux_whisper.snippets import SnippetMatcher

        self._snippets = SnippetMatcher(self.config.snippets)
        logger.info("Snippet matcher ready: %d triggers", len(self.config.snippets))

    async def _setup_injector(self) -> None:
        from linux_whisper.inject.injector import detect_injector

        self._injector = detect_injector(self.config.inject)
        logger.info("Text injector ready: %s", type(self._injector).__name__)

    async def _setup_hotkey(self) -> None:
        from linux_whisper.hotkey import HotkeyDaemon

        self._hotkey = HotkeyDaemon(
            hotkey_str=self.config.hotkey,
            mode=self.config.mode,
            on_start_recording=self._on_recording_start,
            on_stop_recording=self._on_recording_stop,
        )
        logger.info("Hotkey daemon ready: %s (%s mode)", self.config.hotkey, self.config.mode)

    async def _setup_tray(self) -> None:
        if not self.config.tray.enabled:
            logger.info("System tray disabled")
            return
        try:
            from linux_whisper.tray import SystemTray

            self._tray = SystemTray(
                self.config,
                on_quit=self._request_shutdown,
                on_mode_change=self._on_mode_change,
                on_model_change=self._on_model_change,
                on_open_settings=None,
            )
            logger.info("System tray ready")
        except ImportError:
            logger.warning("pystray not available, running without system tray")

    async def _setup_overlay(self) -> None:
        try:
            from linux_whisper.overlay import Overlay

            self._overlay = Overlay()
            if self._overlay.available:
                logger.info("Overlay ready")
            else:
                self._overlay = None
        except ImportError:
            logger.debug("Overlay not available (GTK4 missing)")

    async def run(self) -> None:
        """Run the application until shutdown is requested."""
        logger.info("Starting Linux Whisper")
        self._loop = asyncio.get_running_loop()

        # Start background components
        if self._hotkey:
            self._hotkey.start()
        if self._audio:
            await self._audio.start()
        if self._tray:
            self._tray.start()
        if self._overlay:
            self._overlay.start()

        logger.info("Ready — press %s to dictate", self.config.hotkey)

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        logger.info("Shutting down...")
        await self.cleanup()

    async def cleanup(self) -> None:
        """Stop all components cleanly."""
        if self._hotkey:
            self._hotkey.stop()
        if self._audio:
            await self._audio.stop()
        if self._tray:
            self._tray.stop()
        if self._overlay:
            self._overlay.stop()
        logger.info("Shutdown complete")

    def _request_shutdown(self) -> None:
        """Signal the main loop to exit."""
        self._shutdown_event.set()

    def _on_mode_change(self, mode: str) -> None:
        """Called from tray thread when user selects a different mode."""
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future, self._handle_mode_change(mode)
        )

    async def _handle_mode_change(self, mode: str) -> None:
        """Switch the hotkey mode and persist to config."""
        from linux_whisper.config import Config, CONFIG_PATH, _dataclass_to_dict
        import yaml

        logger.info("Switching mode to %s", mode)

        # Update in-memory config
        self.config = Config(
            hotkey=self.config.hotkey,
            mode=mode,
            stt=self.config.stt,
            polish=self.config.polish,
            audio=self.config.audio,
            inject=self.config.inject,
            tray=self.config.tray,
            snippets=self.config.snippets,
        )

        # Restart hotkey daemon with new mode
        if self._hotkey:
            self._hotkey.stop()
        from linux_whisper.hotkey import HotkeyDaemon
        self._hotkey = HotkeyDaemon(
            hotkey_str=self.config.hotkey,
            mode=mode,
            on_start_recording=self._on_recording_start,
            on_stop_recording=self._on_recording_stop,
        )
        self._hotkey.start()

        # Save config
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = _dataclass_to_dict(self.config)
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            logger.info("Mode changed to %s, config saved", mode)
        except Exception:
            logger.warning("Failed to save config", exc_info=True)

    def _on_model_change(self, backend: str, model: str) -> None:
        """Called from tray thread when user selects a different model."""
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future, self._handle_model_change(backend, model)
        )

    async def _handle_model_change(self, backend: str, model: str) -> None:
        """Hot-swap the STT engine to a different model."""
        from linux_whisper.config import STTConfig, Config, CONFIG_PATH, _dataclass_to_dict
        import yaml

        logger.info("Switching STT engine to %s/%s...", backend, model)

        # Update in-memory config
        new_stt = STTConfig(backend=backend, model=model, threads=self.config.stt.threads)
        self.config = Config(
            hotkey=self.config.hotkey,
            mode=self.config.mode,
            stt=new_stt,
            polish=self.config.polish,
            audio=self.config.audio,
            inject=self.config.inject,
            tray=self.config.tray,
            snippets=self.config.snippets,
        )

        # Save to config file so it persists across restarts
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = _dataclass_to_dict(self.config)
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            logger.info("Config saved to %s", CONFIG_PATH)
        except Exception:
            logger.warning("Failed to save config", exc_info=True)

        # Reload STT engine
        try:
            from linux_whisper.stt.engine import create_engine
            self._stt = create_engine(self.config)
            logger.info("STT engine switched to %s/%s", backend, model)
        except Exception:
            logger.exception("Failed to load new STT engine")
            if self._tray:
                self._tray.update_state(AppState.ERROR)

    def _on_recording_start(self) -> None:
        """Called by hotkey daemon (from its thread) when recording should begin.

        Starts audio capture immediately from the hotkey thread to minimize
        latency, then schedules the async state transition.
        """
        # Grab pre-roll audio BEFORE starting recording (which clears the ring buffer).
        # This captures ~0.75s of audio from before fn was pressed, ensuring
        # we don't lose the first words if the user starts speaking immediately.
        pre_roll = None
        if self._audio:
            pre_roll = self._audio.get_pre_roll(0.75)
            self._audio.start_recording()
        if self._stt:
            self._stt.start_stream()
            # Feed pre-roll into the STT engine immediately
            if pre_roll is not None and len(pre_roll) > 0:
                pre_roll_int16 = (pre_roll * 32767).astype(np.int16)
                self._stt.feed_audio(pre_roll_int16.tobytes())

        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(asyncio.ensure_future, self._handle_recording_start())

    def _on_recording_stop(self) -> None:
        """Called by hotkey daemon (from its thread) when recording should end."""
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(asyncio.ensure_future, self._handle_recording_stop())

    async def _handle_recording_start(self) -> None:
        """Async follow-up after recording already started in _on_recording_start."""
        # Audio capture and STT stream already started synchronously in the
        # hotkey thread for minimum latency. Just update async state here.
        await self.state.transition(AppState.RECORDING)

        if self._overlay:
            self._overlay.show()

        asyncio.ensure_future(self._feed_audio_levels())
        logger.debug("Recording started")

    async def _handle_recording_stop(self) -> None:
        """Stop recording and process the audio through the pipeline."""
        if not self.state.is_recording:
            return

        start_time = time.monotonic()

        if not await self.state.transition(AppState.PROCESSING):
            return

        try:
            text = await self._process_pipeline()
            if text:
                await self._inject_text(text)
                latency = time.monotonic() - start_time
                self._record_latency(latency)
                logger.info("Injected %d chars in %.0fms", len(text), latency * 1000)
                if self._tray:
                    self._tray.set_last_transcription(text)
            else:
                logger.debug("Empty transcription, discarding")
        except Exception:
            logger.exception("Pipeline error")
            await self.state.transition(AppState.ERROR)
            await asyncio.sleep(0.5)
        finally:
            if self._overlay:
                self._overlay.hide()
            await self.state.transition(AppState.IDLE)

    async def _feed_audio_levels(self) -> None:
        """Monitor audio levels and update tray/overlay accordingly.

        Uses RMS energy for speech detection rather than Silero VAD,
        since VAD threshold calibration varies by mic gain. A simple
        RMS threshold is more robust for visual indication.
        """
        if not self._audio:
            return
        last_speech = False
        # Track a rolling baseline of ambient noise
        noise_floor = 0.003  # initial estimate
        while self.state.is_recording:
            try:
                recent = self._audio.get_pre_roll(0.1)  # last 100ms
                if len(recent) > 0:
                    rms = float(np.sqrt(np.mean(recent ** 2)))
                    peak = float(np.max(np.abs(recent)))
                    # Slowly adapt noise floor
                    noise_floor = noise_floor * 0.995 + rms * 0.005
                    # Speech = RMS is significantly above noise floor
                    speech = rms > noise_floor * 3.0 and rms > 0.008
                    logger.debug(
                        "audio: rms=%.5f peak=%.5f floor=%.5f speech=%s",
                        rms, peak, noise_floor, speech,
                    )

                    if speech != last_speech:
                        last_speech = speech
                        if self._tray:
                            self._tray.set_speech_active(speech)
                        if self._overlay:
                            self._overlay.set_speech_active(speech)
            except Exception:
                pass
            await asyncio.sleep(0.1)

    async def _process_pipeline(self) -> str | None:
        """Run the full pipeline: collect audio → STT → polish → text."""
        if not self._audio or not self._stt:
            return None

        # Stop recording — this emits final audio chunks into the queue
        self._audio.stop_recording()

        # Collect all audio chunks from this recording session
        audio_segments: list[np.ndarray] = []
        async for chunk in self._audio.audio_chunks():
            if chunk.samples is not None and len(chunk.samples) > 0:
                audio_segments.append(chunk.samples)
            if chunk.is_final:
                break

        if not audio_segments:
            return None

        # Concatenate all audio
        audio_float = np.concatenate(audio_segments)
        if len(audio_float) == 0:
            return None

        duration = len(audio_float) / 16000
        logger.info("Recording: %.1fs audio (%d samples)", duration, len(audio_float))

        # Apply automatic gain control for quiet/whispered speech
        if self.config.audio.auto_gain:
            from linux_whisper.audio import apply_agc

            audio_float = apply_agc(audio_float)

        # Convert float32 [-1.0, 1.0] to int16 PCM bytes
        audio_int16 = (audio_float * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Feed audio to STT
        self._stt.feed_audio(audio_bytes)
        result = self._stt.finalize()
        self._stt.reset()

        if not result or not result.full_text.strip():
            return None

        text = result.full_text.strip()
        logger.debug("STT result: %s", text[:100])

        # Check snippets before polish — if a trigger matches, return the
        # snippet text directly, bypassing the entire polish pipeline.
        if self._snippets:
            snippet_text = self._snippets.match(text)
            if snippet_text is not None:
                logger.info(
                    "Snippet matched: '%s' -> %d chars", text[:50], len(snippet_text)
                )
                return snippet_text

        # Detect focused app for context-aware LLM prompts
        app_context: str | None = None
        if self.config.polish.context_awareness:
            from linux_whisper.focus import build_context_string, detect_focused_app

            focused = detect_focused_app()
            if focused is not None:
                app_context = build_context_string(focused)
                logger.debug("Focused app: %s (%s)", focused.app_name, focused.category.value)

        # Polish
        if self._polish:
            text = await asyncio.to_thread(
                self._polish.process, text, app_context
            )
            logger.debug("Polished: %s", text[:100])

        return text

    @staticmethod
    def _trim_silence(audio: np.ndarray, frame_ms: int = 30, threshold_factor: float = 3.0, pad_frames: int = 5) -> np.ndarray:
        """Remove leading/trailing/internal silence from audio.

        Splits audio into frames, computes RMS per frame, identifies speech
        frames (those above threshold_factor * median RMS), and keeps only
        speech regions with padding. Multiple speech regions separated by
        silence are concatenated with a short silence gap to preserve
        natural pacing for the STT model.
        """
        sample_rate = 16000
        frame_size = int(sample_rate * frame_ms / 1000)

        if len(audio) < frame_size:
            return audio

        # Compute per-frame RMS
        n_frames = len(audio) // frame_size
        frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Adaptive threshold: median RMS * factor, with a minimum floor
        median_rms = float(np.median(rms))
        threshold = max(median_rms * threshold_factor, 0.005)

        # Find speech frames
        is_speech = rms > threshold

        if not np.any(is_speech):
            # No speech detected at all — return the original (let STT decide)
            return audio

        # Expand speech regions by pad_frames on each side
        padded = np.copy(is_speech)
        for i in range(n_frames):
            if is_speech[i]:
                start = max(0, i - pad_frames)
                end = min(n_frames, i + pad_frames + 1)
                padded[start:end] = True

        # Extract speech regions and join with a small silence gap
        gap = np.zeros(int(sample_rate * 0.15), dtype=np.float32)  # 150ms gap
        regions: list[np.ndarray] = []
        in_region = False

        for i in range(n_frames):
            if padded[i]:
                if not in_region and regions:
                    regions.append(gap)
                in_region = True
                regions.append(frames[i])
            else:
                in_region = False

        if not regions:
            return audio

        return np.concatenate(regions)

    async def _inject_text(self, text: str) -> None:
        """Inject text at the cursor position."""
        if not self._injector:
            logger.warning("No text injector available")
            return
        success = await self._injector.inject(text)
        if not success:
            logger.error("Text injection failed")

    def _record_latency(self, latency: float) -> None:
        """Track latency for stats."""
        self._latencies.append(latency)
        if len(self._latencies) > self._max_latency_history:
            self._latencies.pop(0)
        if self._tray:
            avg = sum(self._latencies) / len(self._latencies)
            self._tray.update_stats(last_latency=latency, avg_latency=avg)

    @property
    def latency_stats(self) -> dict[str, float]:
        """Return current latency statistics."""
        if not self._latencies:
            return {"last": 0, "avg": 0, "p95": 0}
        sorted_lat = sorted(self._latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        return {
            "last": self._latencies[-1],
            "avg": sum(self._latencies) / len(self._latencies),
            "p95": sorted_lat[min(p95_idx, len(sorted_lat) - 1)],
        }


def create_app(config: Config | None = None) -> App:
    """Create an App instance with the given or default config."""
    if config is None:
        config = Config.load()
    return App(config)


async def run_app(config: Config | None = None) -> None:
    """Create, setup, and run the application."""
    app = create_app(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app._request_shutdown)

    await app.setup()
    await app.run()
