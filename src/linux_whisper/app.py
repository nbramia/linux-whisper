"""Main application orchestration — connects all pipeline stages via asyncio."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import TYPE_CHECKING

from linux_whisper.config import Config
from linux_whisper.state import AppState, StateMachine

if TYPE_CHECKING:
    from linux_whisper.audio import AudioPipeline
    from linux_whisper.hotkey import HotkeyDaemon
    from linux_whisper.inject import TextInjector
    from linux_whisper.polish import PolishPipeline
    from linux_whisper.stt import STTEngine
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

        # Components initialized in setup()
        self._audio: AudioPipeline | None = None
        self._hotkey: HotkeyDaemon | None = None
        self._stt: STTEngine | None = None
        self._polish: PolishPipeline | None = None
        self._injector: TextInjector | None = None
        self._tray: SystemTray | None = None

    async def setup(self) -> None:
        """Initialize all components. Call before run()."""
        logger.info("Initializing Linux Whisper v0.1.0")
        errors = self.config.validate()
        if errors:
            for e in errors:
                logger.error("Config error: %s", e)
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Import and initialize components
        # Each import is guarded so missing optional deps give clear errors
        await self._setup_audio()
        await self._setup_stt()
        await self._setup_polish()
        await self._setup_injector()
        await self._setup_hotkey()
        await self._setup_tray()

        # Wire state change listener to tray
        if self._tray:
            self.state.on_state_change(lambda _old, new: self._tray.update_state(new))

        logger.info("All components initialized")

    async def _setup_audio(self) -> None:
        from linux_whisper.audio import AudioPipeline

        self._audio = AudioPipeline(self.config.audio)
        logger.info("Audio pipeline ready")

    async def _setup_stt(self) -> None:
        from linux_whisper.stt import create_engine

        self._stt = create_engine(self.config)
        logger.info("STT engine ready: %s", self.config.stt.backend)

    async def _setup_polish(self) -> None:
        if not self.config.polish.enabled:
            logger.info("Polish pipeline disabled")
            return
        from linux_whisper.polish.pipeline import PolishPipeline

        self._polish = PolishPipeline(self.config.polish)
        await asyncio.to_thread(self._polish.load)
        logger.info("Polish pipeline ready")

    async def _setup_injector(self) -> None:
        from linux_whisper.inject import detect_injector

        self._injector = detect_injector(self.config.inject)
        logger.info("Text injector ready: %s", type(self._injector).__name__)

    async def _setup_hotkey(self) -> None:
        from linux_whisper.hotkey import HotkeyDaemon

        self._hotkey = HotkeyDaemon(
            hotkey=self.config.hotkey,
            mode=self.config.mode,
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
        )
        logger.info("Hotkey daemon ready: %s (%s mode)", self.config.hotkey, self.config.mode)

    async def _setup_tray(self) -> None:
        if not self.config.tray.enabled:
            logger.info("System tray disabled")
            return
        try:
            from linux_whisper.tray import SystemTray

            self._tray = SystemTray(
                on_quit=self._request_shutdown,
                on_mode_change=None,  # TODO: implement mode switching
                on_open_settings=None,
            )
            logger.info("System tray ready")
        except ImportError:
            logger.warning("pystray not available, running without system tray")

    async def run(self) -> None:
        """Run the application until shutdown is requested."""
        logger.info("Starting Linux Whisper")

        # Start background components
        if self._hotkey:
            self._hotkey.start()
        if self._audio:
            self._audio.start()
        if self._tray:
            self._tray.start()

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
            self._audio.stop()
        if self._tray:
            self._tray.stop()
        logger.info("Shutdown complete")

    def _request_shutdown(self) -> None:
        """Signal the main loop to exit."""
        self._shutdown_event.set()

    def _on_recording_start(self) -> None:
        """Called by hotkey daemon when recording should begin."""
        asyncio.get_event_loop().call_soon_threadsafe(
            asyncio.ensure_future, self._handle_recording_start()
        )

    def _on_recording_stop(self) -> None:
        """Called by hotkey daemon when recording should end."""
        asyncio.get_event_loop().call_soon_threadsafe(
            asyncio.ensure_future, self._handle_recording_stop()
        )

    async def _handle_recording_start(self) -> None:
        """Transition to RECORDING and start capturing audio."""
        if not await self.state.transition(AppState.RECORDING):
            return

        if self._audio:
            # Play start feedback
            if self.config.audio.feedback_sounds:
                await asyncio.to_thread(self._audio.play_start_tone)
            self._audio.start_recording()

        if self._stt:
            self._stt.start_stream()

        logger.debug("Recording started")

    async def _handle_recording_stop(self) -> None:
        """Stop recording and process the audio through the pipeline."""
        if not self.state.is_recording:
            return

        start_time = time.monotonic()

        # Play stop feedback
        if self._audio and self.config.audio.feedback_sounds:
            await asyncio.to_thread(self._audio.play_stop_tone)

        if not await self.state.transition(AppState.PROCESSING):
            return

        try:
            text = await self._process_pipeline()
            if text:
                await self._inject_text(text)
                latency = time.monotonic() - start_time
                self._record_latency(latency)
                logger.info("Injected %d chars in %.0fms", len(text), latency * 1000)
            else:
                logger.debug("Empty transcription, discarding")
        except Exception:
            logger.exception("Pipeline error")
            await self.state.transition(AppState.ERROR)
            await asyncio.sleep(0.5)
        finally:
            await self.state.transition(AppState.IDLE)

    async def _process_pipeline(self) -> str | None:
        """Run the full pipeline: audio → STT → polish → text."""
        if not self._audio or not self._stt:
            return None

        # Get recorded audio
        audio_data = self._audio.stop_recording()
        if audio_data is None or len(audio_data) == 0:
            return None

        # Feed audio to STT
        self._stt.feed_audio(audio_data)
        result = self._stt.finalize()
        self._stt.reset()

        if not result or not result.full_text.strip():
            return None

        text = result.full_text.strip()
        logger.debug("STT result: %s", text[:100])

        # Polish
        if self._polish:
            text = await asyncio.to_thread(self._polish.process, text)
            logger.debug("Polished: %s", text[:100])

        return text

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
