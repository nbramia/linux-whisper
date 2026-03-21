"""Global hotkey daemon using evdev for kernel-level input capture.

Reads from /dev/input/event* devices to detect key combinations regardless of
display server (works on both X11 and Wayland).

Requirements:
    The running user must be in the ``input`` group to read from /dev/input/::

        sudo usermod -aG input $USER

    Then log out and back in for the group change to take effect.

Supported modes (set in ``Config.mode``):
    * ``hold``     -- hold-to-talk: recording starts on key-down, stops on key-up
    * ``toggle``   -- press combo once to start, press again to stop
    * ``vad-auto`` -- press combo to start; stop is handled externally by VAD
"""

from __future__ import annotations

import logging
import select
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

from evdev import InputDevice, InputEvent, ecodes

logger = logging.getLogger(__name__)

# Re-scan /dev/input/ this often to pick up newly attached keyboards.
_DEVICE_POLL_INTERVAL: Final[float] = 3.0

# How long select() blocks before checking the stop flag.
_SELECT_TIMEOUT: Final[float] = 0.25


# ---------------------------------------------------------------------------
# Key-name → evdev code mapping
# ---------------------------------------------------------------------------

# Modifier keys that can appear on the left side of a combo string.
_MODIFIER_MAP: Final[dict[str, int]] = {
    "ctrl": ecodes.KEY_LEFTCTRL,
    "leftctrl": ecodes.KEY_LEFTCTRL,
    "rightctrl": ecodes.KEY_RIGHTCTRL,
    "shift": ecodes.KEY_LEFTSHIFT,
    "leftshift": ecodes.KEY_LEFTSHIFT,
    "rightshift": ecodes.KEY_RIGHTSHIFT,
    "alt": ecodes.KEY_LEFTALT,
    "leftalt": ecodes.KEY_LEFTALT,
    "rightalt": ecodes.KEY_RIGHTALT,
    "meta": ecodes.KEY_LEFTMETA,
    "super": ecodes.KEY_LEFTMETA,
    "leftmeta": ecodes.KEY_LEFTMETA,
    "rightmeta": ecodes.KEY_RIGHTMETA,
}

# Left/right equivalences — we treat either side as the same modifier.
_MODIFIER_PAIRS: Final[dict[int, int]] = {
    ecodes.KEY_RIGHTCTRL: ecodes.KEY_LEFTCTRL,
    ecodes.KEY_RIGHTSHIFT: ecodes.KEY_LEFTSHIFT,
    ecodes.KEY_RIGHTALT: ecodes.KEY_LEFTALT,
    ecodes.KEY_RIGHTMETA: ecodes.KEY_LEFTMETA,
}


def _normalize_modifier(code: int) -> int:
    """Map right-hand modifiers to their left-hand canonical form."""
    return _MODIFIER_PAIRS.get(code, code)


def _key_name_to_code(name: str) -> int:
    """Resolve a human-friendly key name to an evdev key code.

    Raises ``ValueError`` if the name cannot be resolved.
    """
    lower = name.lower().strip()

    # Check modifiers first.
    if lower in _MODIFIER_MAP:
        return _MODIFIER_MAP[lower]

    # Try KEY_<NAME> in ecodes (e.g. "e" → KEY_E, "f1" → KEY_F1).
    attr = f"KEY_{lower.upper()}"
    code = getattr(ecodes, attr, None)
    if code is not None:
        return code

    raise ValueError(
        f"Unknown key name '{name}'. Expected a modifier (ctrl, shift, alt, meta/super) "
        f"or a key name recognised by evdev (e.g. 'e', 'f1', 'space')."
    )


# ---------------------------------------------------------------------------
# Hotkey combo parsing
# ---------------------------------------------------------------------------

class HotkeyCombination:
    """Parsed representation of a hotkey string like ``ctrl+shift+e``."""

    __slots__ = ("modifiers", "key", "raw")

    def __init__(self, modifiers: frozenset[int], key: int, raw: str) -> None:
        self.modifiers: Final = modifiers
        self.key: Final = key
        self.raw: Final = raw

    @classmethod
    def parse(cls, combo_str: str) -> HotkeyCombination:
        """Parse a ``+``-delimited hotkey string.

        The last token is treated as the principal key; all preceding tokens
        are modifiers.  Example: ``ctrl+shift+e``.

        Raises ``ValueError`` on invalid input.
        """
        parts = [p.strip() for p in combo_str.lower().split("+") if p.strip()]
        if not parts:
            raise ValueError("Empty hotkey string")

        if len(parts) == 1:
            code = _key_name_to_code(parts[0])
            return cls(modifiers=frozenset(), key=code, raw=combo_str)

        *mod_names, key_name = parts
        mod_codes = frozenset(_normalize_modifier(_key_name_to_code(n)) for n in mod_names)
        key_code = _key_name_to_code(key_name)
        return cls(modifiers=mod_codes, key=key_code, raw=combo_str)

    def __repr__(self) -> str:
        return f"HotkeyCombination('{self.raw}')"


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def _find_keyboard_devices() -> list[InputDevice]:
    """Return a list of evdev ``InputDevice`` objects that have keyboard capabilities.

    Silently skips devices that cannot be opened (permission errors, etc.).
    """
    devices: list[InputDevice] = []
    input_dir = Path("/dev/input")
    for path in sorted(input_dir.glob("event*")):
        try:
            dev = InputDevice(str(path))
        except (PermissionError, OSError) as exc:
            logger.debug("Cannot open %s: %s", path, exc)
            continue

        caps = dev.capabilities(verbose=False)
        # EV_KEY = 1.  A device that reports key events is a candidate.
        if ecodes.EV_KEY in caps:
            key_caps: list[int] = caps[ecodes.EV_KEY]
            # Only include devices that actually report common keyboard keys,
            # not e.g. a power button that only reports KEY_POWER.
            if ecodes.KEY_A in key_caps or ecodes.KEY_ENTER in key_caps:
                devices.append(dev)
            else:
                dev.close()
        else:
            dev.close()

    return devices


# ---------------------------------------------------------------------------
# Hotkey daemon
# ---------------------------------------------------------------------------

class HotkeyDaemon:
    """Background thread that listens for a global hotkey combo via evdev.

    Parameters
    ----------
    hotkey_str:
        Human-readable combo, e.g. ``"ctrl+shift+e"``.
    mode:
        One of ``"hold"``, ``"toggle"``, or ``"vad-auto"``.
    on_start_recording:
        Called (from the daemon thread) when recording should begin.
    on_stop_recording:
        Called (from the daemon thread) when recording should end.
        Not used in ``vad-auto`` mode.

    Notes
    -----
    * The user must be in the ``input`` group to open ``/dev/input/event*``.
    * Devices are periodically re-scanned so hot-plugged keyboards are picked
      up automatically.
    * Call :meth:`start` to launch the daemon thread and :meth:`stop` to shut
      it down cleanly.
    """

    VALID_MODES: Final = ("hold", "toggle", "vad-auto")

    def __init__(
        self,
        hotkey_str: str,
        mode: str,
        on_start_recording: Callable[[], None],
        on_stop_recording: Callable[[], None],
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}', expected one of {self.VALID_MODES}")

        self._combo: Final = HotkeyCombination.parse(hotkey_str)
        self._mode: Final = mode
        self._on_start: Final = on_start_recording
        self._on_stop: Final = on_stop_recording

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Tracked state for modifier keys currently held down (normalized).
        self._held_modifiers: set[int] = set()
        # Whether the principal key is physically down right now.
        self._principal_held: bool = False
        # Toggle bookkeeping.
        self._toggle_active: bool = False
        # Track whether we are in a recording started by this daemon.
        self._recording: bool = False

        logger.info(
            "HotkeyDaemon configured: combo=%s  mode=%s",
            self._combo.raw,
            self._mode,
        )

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Start the daemon thread.  Safe to call multiple times."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("HotkeyDaemon already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="hotkey-daemon",
            daemon=True,
        )
        self._thread.start()
        logger.info("HotkeyDaemon started")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the daemon to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("HotkeyDaemon thread did not exit within %.1fs", timeout)
            self._thread = None
        logger.info("HotkeyDaemon stopped")

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- internal -----------------------------------------------------------

    def _run(self) -> None:
        """Main loop executed in the daemon thread."""
        devices: dict[str, InputDevice] = {}
        last_scan: float = 0.0

        try:
            while not self._stop_event.is_set():
                # Periodic device rescan.
                now = time.monotonic()
                if now - last_scan >= _DEVICE_POLL_INTERVAL:
                    self._rescan_devices(devices)
                    last_scan = now

                if not devices:
                    # No keyboards found — wait and retry.
                    self._stop_event.wait(timeout=_DEVICE_POLL_INTERVAL)
                    continue

                # select() on all open file descriptors.
                fds = {dev.fd: dev for dev in devices.values()}
                try:
                    readable, _, _ = select.select(
                        list(fds.keys()), [], [], _SELECT_TIMEOUT
                    )
                except (ValueError, OSError):
                    # A device fd was closed underneath us.  Force a rescan.
                    last_scan = 0.0
                    continue

                for fd in readable:
                    dev = fds.get(fd)
                    if dev is None:
                        continue
                    try:
                        for event in dev.read():
                            self._handle_event(event)
                    except OSError:
                        # Device disconnected.
                        logger.info("Device disconnected: %s", dev.path)
                        self._close_device(devices, dev.path)
                        last_scan = 0.0  # trigger immediate rescan
        except Exception:
            logger.exception("HotkeyDaemon thread crashed")
        finally:
            self._close_all_devices(devices)

    # -- device management --------------------------------------------------

    @staticmethod
    def _rescan_devices(current: dict[str, InputDevice]) -> None:
        """Add newly appeared devices and prune stale ones."""
        found = _find_keyboard_devices()
        found_paths = {dev.path for dev in found}

        # Add new devices.
        for dev in found:
            if dev.path not in current:
                logger.debug("Tracking new keyboard: %s (%s)", dev.path, dev.name)
                current[dev.path] = dev
            else:
                # We already track this path — close the duplicate handle.
                dev.close()

        # Remove devices that vanished.
        stale = [p for p in current if p not in found_paths]
        for path in stale:
            logger.debug("Removing stale device: %s", path)
            try:
                current[path].close()
            except OSError:
                pass
            del current[path]

    @staticmethod
    def _close_device(devices: dict[str, InputDevice], path: str) -> None:
        dev = devices.pop(path, None)
        if dev is not None:
            try:
                dev.close()
            except OSError:
                pass

    @staticmethod
    def _close_all_devices(devices: dict[str, InputDevice]) -> None:
        for dev in devices.values():
            try:
                dev.close()
            except OSError:
                pass
        devices.clear()

    # -- event handling -----------------------------------------------------

    def _handle_event(self, event: InputEvent) -> None:
        """Process a single evdev input event."""
        if event.type != ecodes.EV_KEY:
            return

        code: int = event.code
        # value: 0 = up, 1 = down, 2 = repeat (autorepeat)
        value: int = event.value

        normalized = _normalize_modifier(code)

        # Track modifier state.
        is_modifier = normalized in (
            ecodes.KEY_LEFTCTRL,
            ecodes.KEY_LEFTSHIFT,
            ecodes.KEY_LEFTALT,
            ecodes.KEY_LEFTMETA,
        )

        if is_modifier:
            if value in (1, 2):  # down or repeat
                self._held_modifiers.add(normalized)
            elif value == 0:  # up
                self._held_modifiers.discard(normalized)
                # In hold mode, releasing a modifier while principal key is
                # held should also stop recording.
                if (
                    self._mode == "hold"
                    and self._recording
                    and normalized in self._combo.modifiers
                ):
                    self._fire_stop()
            return

        # Non-modifier key.
        if code == self._combo.key:
            if value == 1:  # key down (ignore repeats for combo triggering)
                self._principal_held = True
                if self._modifiers_satisfied():
                    self._on_combo_press()
            elif value == 0:  # key up
                self._principal_held = False
                if self._mode == "hold" and self._recording:
                    self._fire_stop()

    def _modifiers_satisfied(self) -> bool:
        """Return True when all required modifier keys are currently held."""
        return self._combo.modifiers.issubset(self._held_modifiers)

    def _on_combo_press(self) -> None:
        """Dispatch based on the active mode."""
        match self._mode:
            case "hold":
                self._fire_start()
            case "toggle":
                if self._toggle_active:
                    self._fire_stop()
                    self._toggle_active = False
                else:
                    self._fire_start()
                    self._toggle_active = True
            case "vad-auto":
                # Only fires start; stop is handled externally by VAD.
                if not self._recording:
                    self._fire_start()

    def _fire_start(self) -> None:
        if self._recording:
            return
        self._recording = True
        logger.debug("Hotkey combo pressed — firing on_start_recording")
        try:
            self._on_start()
        except Exception:
            logger.exception("on_start_recording callback raised")
            self._recording = False

    def _fire_stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        logger.debug("Hotkey released/toggled — firing on_stop_recording")
        try:
            self._on_stop()
        except Exception:
            logger.exception("on_stop_recording callback raised")

    def notify_external_stop(self) -> None:
        """Call this when an external system (e.g. VAD) has stopped recording.

        Resets internal bookkeeping so the daemon can accept new triggers.
        """
        self._recording = False
        self._toggle_active = False
