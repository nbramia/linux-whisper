"""Text injection backends for X11 and Wayland."""

from linux_whisper.inject.injector import (
    ClipboardInjector,
    DisplayServer,
    TextInjector,
    WtypeInjector,
    XdotoolInjector,
    YdotoolInjector,
    detect_injector,
)

__all__ = [
    "ClipboardInjector",
    "DisplayServer",
    "TextInjector",
    "WtypeInjector",
    "XdotoolInjector",
    "YdotoolInjector",
    "detect_injector",
]
