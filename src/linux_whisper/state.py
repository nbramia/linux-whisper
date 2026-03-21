"""Application state machine: IDLE → RECORDING → PROCESSING → IDLE."""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class AppState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


# Valid transitions: (from_state, to_state)
_TRANSITIONS: set[tuple[AppState, AppState]] = {
    (AppState.IDLE, AppState.RECORDING),
    (AppState.RECORDING, AppState.PROCESSING),
    (AppState.RECORDING, AppState.IDLE),  # cancelled recording (empty/noise)
    (AppState.PROCESSING, AppState.IDLE),
    (AppState.PROCESSING, AppState.ERROR),
    (AppState.ERROR, AppState.IDLE),
    # Allow any state to transition to ERROR for unexpected failures
    (AppState.IDLE, AppState.ERROR),
    (AppState.RECORDING, AppState.ERROR),
}

StateCallback = Callable[[AppState, AppState], None]


class StateMachine:
    """Thread-safe state machine with observer callbacks."""

    def __init__(self) -> None:
        self._state = AppState.IDLE
        self._lock = asyncio.Lock()
        self._listeners: list[StateCallback] = []
        self._state_event = asyncio.Event()

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == AppState.IDLE

    @property
    def is_recording(self) -> bool:
        return self._state == AppState.RECORDING

    @property
    def is_processing(self) -> bool:
        return self._state == AppState.PROCESSING

    def on_state_change(self, callback: StateCallback) -> None:
        """Register a callback for state transitions."""
        self._listeners.append(callback)

    async def transition(self, new_state: AppState) -> bool:
        """Attempt a state transition. Returns True if successful."""
        async with self._lock:
            old = self._state
            if old == new_state:
                return True
            if (old, new_state) not in _TRANSITIONS:
                logger.warning(
                    "Invalid state transition: %s → %s",
                    old.value,
                    new_state.value,
                )
                return False
            self._state = new_state
            logger.debug("State: %s → %s", old.value, new_state.value)

        # Notify outside the lock to avoid deadlocks
        for cb in self._listeners:
            try:
                cb(old, new_state)
            except Exception:
                logger.exception("State change callback failed")

        self._state_event.set()
        self._state_event.clear()
        return True

    async def wait_for(self, target: AppState, timeout: float | None = None) -> bool:
        """Wait until the state machine reaches the target state."""
        if self._state == target:
            return True
        try:
            while self._state != target:
                await asyncio.wait_for(self._state_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def reset(self) -> None:
        """Force-reset to IDLE. Use only for error recovery."""
        async with self._lock:
            old = self._state
            self._state = AppState.IDLE
            if old != AppState.IDLE:
                logger.info("Force reset from %s to IDLE", old.value)
                for cb in self._listeners:
                    try:
                        cb(old, AppState.IDLE)
                    except Exception:
                        logger.exception("State change callback failed during reset")
        self._state_event.set()
        self._state_event.clear()
