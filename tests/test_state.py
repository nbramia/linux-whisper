"""Tests for linux_whisper.state — state machine transitions, callbacks, wait_for, reset."""

from __future__ import annotations

import asyncio

import pytest

from linux_whisper.state import AppState, StateMachine, _TRANSITIONS


# ── Initial state ───────────────────────────────────────────────────────────


class TestInitialState:
    """Verify the state machine starts in IDLE."""

    def test_initial_state_is_idle(self):
        sm = StateMachine()
        assert sm.state == AppState.IDLE
        assert sm.is_idle is True
        assert sm.is_recording is False
        assert sm.is_processing is False


# ── Valid transitions ───────────────────────────────────────────────────────


class TestValidTransitions:

    async def test_idle_to_recording(self):
        sm = StateMachine()
        result = await sm.transition(AppState.RECORDING)
        assert result is True
        assert sm.state == AppState.RECORDING
        assert sm.is_recording is True

    async def test_recording_to_processing(self):
        sm = StateMachine()
        await sm.transition(AppState.RECORDING)
        result = await sm.transition(AppState.PROCESSING)
        assert result is True
        assert sm.is_processing is True

    async def test_processing_to_idle(self):
        sm = StateMachine()
        await sm.transition(AppState.RECORDING)
        await sm.transition(AppState.PROCESSING)
        result = await sm.transition(AppState.IDLE)
        assert result is True
        assert sm.is_idle is True

    async def test_recording_to_idle_cancel(self):
        sm = StateMachine()
        await sm.transition(AppState.RECORDING)
        result = await sm.transition(AppState.IDLE)
        assert result is True
        assert sm.is_idle is True

    async def test_processing_to_error(self):
        sm = StateMachine()
        await sm.transition(AppState.RECORDING)
        await sm.transition(AppState.PROCESSING)
        result = await sm.transition(AppState.ERROR)
        assert result is True
        assert sm.state == AppState.ERROR

    async def test_error_to_idle(self):
        sm = StateMachine()
        await sm.transition(AppState.ERROR)
        result = await sm.transition(AppState.IDLE)
        assert result is True
        assert sm.is_idle is True

    async def test_any_state_to_error(self):
        """All states defined in _TRANSITIONS can reach ERROR."""
        for from_state, to_state in _TRANSITIONS:
            if to_state == AppState.ERROR:
                sm = StateMachine()
                # Force state directly for setup
                sm._state = from_state
                result = await sm.transition(AppState.ERROR)
                assert result is True, f"Failed: {from_state} -> ERROR"

    async def test_same_state_transition_returns_true(self):
        sm = StateMachine()
        result = await sm.transition(AppState.IDLE)
        assert result is True  # no-op, same state


# ── Invalid transitions ─────────────────────────────────────────────────────


class TestInvalidTransitions:

    async def test_idle_to_processing_rejected(self):
        sm = StateMachine()
        result = await sm.transition(AppState.PROCESSING)
        assert result is False
        assert sm.state == AppState.IDLE  # unchanged

    async def test_processing_to_recording_rejected(self):
        sm = StateMachine()
        await sm.transition(AppState.RECORDING)
        await sm.transition(AppState.PROCESSING)
        result = await sm.transition(AppState.RECORDING)
        assert result is False
        assert sm.state == AppState.PROCESSING

    async def test_error_to_recording_rejected(self):
        sm = StateMachine()
        await sm.transition(AppState.ERROR)
        result = await sm.transition(AppState.RECORDING)
        assert result is False

    async def test_error_to_processing_rejected(self):
        sm = StateMachine()
        await sm.transition(AppState.ERROR)
        result = await sm.transition(AppState.PROCESSING)
        assert result is False


# ── Callbacks ───────────────────────────────────────────────────────────────


class TestCallbacks:

    async def test_callback_called_on_transition(self):
        sm = StateMachine()
        calls = []
        sm.on_state_change(lambda old, new: calls.append((old, new)))

        await sm.transition(AppState.RECORDING)
        assert calls == [(AppState.IDLE, AppState.RECORDING)]

    async def test_callback_not_called_on_same_state(self):
        sm = StateMachine()
        calls = []
        sm.on_state_change(lambda old, new: calls.append((old, new)))

        await sm.transition(AppState.IDLE)  # same state
        assert calls == []

    async def test_callback_not_called_on_invalid_transition(self):
        sm = StateMachine()
        calls = []
        sm.on_state_change(lambda old, new: calls.append((old, new)))

        await sm.transition(AppState.PROCESSING)  # invalid from IDLE
        assert calls == []

    async def test_multiple_callbacks(self):
        sm = StateMachine()
        calls_a = []
        calls_b = []
        sm.on_state_change(lambda old, new: calls_a.append((old, new)))
        sm.on_state_change(lambda old, new: calls_b.append((old, new)))

        await sm.transition(AppState.RECORDING)
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    async def test_callback_exception_does_not_prevent_transition(self):
        sm = StateMachine()

        def bad_callback(old, new):
            raise RuntimeError("boom")

        sm.on_state_change(bad_callback)
        result = await sm.transition(AppState.RECORDING)
        assert result is True
        assert sm.state == AppState.RECORDING

    async def test_callback_receives_correct_states(self):
        sm = StateMachine()
        calls = []
        sm.on_state_change(lambda old, new: calls.append((old, new)))

        await sm.transition(AppState.RECORDING)
        await sm.transition(AppState.PROCESSING)
        await sm.transition(AppState.IDLE)

        assert calls == [
            (AppState.IDLE, AppState.RECORDING),
            (AppState.RECORDING, AppState.PROCESSING),
            (AppState.PROCESSING, AppState.IDLE),
        ]


# ── wait_for ────────────────────────────────────────────────────────────────


class TestWaitFor:

    async def test_wait_for_already_in_target_state(self):
        sm = StateMachine()
        result = await sm.wait_for(AppState.IDLE)
        assert result is True

    async def test_wait_for_with_transition(self):
        sm = StateMachine()

        async def transition_after_delay():
            await asyncio.sleep(0.01)
            await sm.transition(AppState.RECORDING)

        task = asyncio.create_task(transition_after_delay())
        result = await sm.wait_for(AppState.RECORDING, timeout=1.0)
        assert result is True
        assert sm.state == AppState.RECORDING
        await task

    async def test_wait_for_timeout(self):
        sm = StateMachine()
        result = await sm.wait_for(AppState.RECORDING, timeout=0.05)
        assert result is False
        assert sm.state == AppState.IDLE  # no transition happened


# ── reset ───────────────────────────────────────────────────────────────────


class TestReset:

    async def test_reset_from_recording(self):
        sm = StateMachine()
        await sm.transition(AppState.RECORDING)
        await sm.reset()
        assert sm.state == AppState.IDLE

    async def test_reset_from_processing(self):
        sm = StateMachine()
        await sm.transition(AppState.RECORDING)
        await sm.transition(AppState.PROCESSING)
        await sm.reset()
        assert sm.state == AppState.IDLE

    async def test_reset_from_error(self):
        sm = StateMachine()
        await sm.transition(AppState.ERROR)
        await sm.reset()
        assert sm.state == AppState.IDLE

    async def test_reset_from_idle_is_noop(self):
        sm = StateMachine()
        calls = []
        sm.on_state_change(lambda old, new: calls.append((old, new)))

        await sm.reset()
        assert sm.state == AppState.IDLE
        # No callback fired because old state was already IDLE
        assert calls == []

    async def test_reset_fires_callback(self):
        sm = StateMachine()
        calls = []
        sm.on_state_change(lambda old, new: calls.append((old, new)))

        await sm.transition(AppState.RECORDING)
        calls.clear()

        await sm.reset()
        assert calls == [(AppState.RECORDING, AppState.IDLE)]

    async def test_reset_callback_exception_handled(self):
        sm = StateMachine()

        def bad_callback(old, new):
            raise RuntimeError("reset boom")

        sm.on_state_change(bad_callback)
        await sm.transition(AppState.RECORDING)

        # Should not raise
        await sm.reset()
        assert sm.state == AppState.IDLE


# ── Full cycle ──────────────────────────────────────────────────────────────


class TestFullCycle:

    async def test_complete_recording_cycle(self):
        sm = StateMachine()
        assert await sm.transition(AppState.RECORDING)
        assert await sm.transition(AppState.PROCESSING)
        assert await sm.transition(AppState.IDLE)
        assert sm.is_idle

    async def test_error_recovery_cycle(self):
        sm = StateMachine()
        assert await sm.transition(AppState.RECORDING)
        assert await sm.transition(AppState.PROCESSING)
        assert await sm.transition(AppState.ERROR)
        assert await sm.transition(AppState.IDLE)
        # Can start a new recording after error recovery
        assert await sm.transition(AppState.RECORDING)
