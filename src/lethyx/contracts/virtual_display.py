"""Auto-provision a virtual display for GUI commands on headless hosts.

Native desktop GUI toolkits (tkinter, PyQt/PySide, wxPython, kivy, pygame, ...)
need an X server to construct widgets, even for "headless" unit-style proofs
that never call ``mainloop``. On a headless Linux host (no ``$DISPLAY``) those
commands fail with errors like ``no display name and no $DISPLAY environment
variable`` or ``tkinter.TclError: couldn't connect to display``.

Rather than make the agent fight the display every run, the platform wraps GUI
commands with ``xvfb-run`` when a real display is missing. This is a runtime
convenience, not a policy gate: if ``xvfb-run`` is not installed we leave the
command untouched and surface a clear note telling the agent how to recover
(install Xvfb, or fall back to a headless / localhost-web proof lane).

macOS and Windows expose a usable window server to the process, so we never
wrap there.
"""

import os
import shutil
import sys
from typing import Any

# Capability tags / harness profiles that imply a native desktop GUI runtime.
_GUI_CAPABILITY_TAGS = frozenset(
    {
        "desktop_gui_runtime",
        "desktop_gui_headless",
        "desktop_gui",
        "native_gui",
        "display",
        "x11",
    }
)
_GUI_HARNESS_PROFILES = frozenset(
    {
        "desktop_gui_runtime",
        "desktop_gui_headless",
    }
)
# Substrings that, when present in an argv, imply the command needs a display.
_GUI_COMMAND_MARKERS = (
    "tkinter",
    "customtkinter",
    "import tk",
    "tk.tk(",
    "tkinter as tk",
    "pyqt5",
    "pyqt6",
    "pyside2",
    "pyside6",
    "qapplication",
    "wxpython",
    "import wx",
    "kivy",
    "pygame",
    "pyautogui",
    "gi.repository",
    "gtk",
)

_DEFAULT_SERVER_ARGS = "-screen 0 1280x1024x24"

_INSTALL_HINT = (
    "no $DISPLAY and `xvfb-run` is not installed on this host, so a real GUI "
    "window cannot be created. Recover in one of three ways: (1) install a "
    "virtual display with provision_workspace_environment "
    '{"kind":"apt_install","packages":["xvfb"]} (or the host package '
    "manager) and retry, (2) use the desktop_gui_headless harness and prove "
    "behavior through the controller/model adapter, or (3) build the UI as a "
    "localhost web app and prove it with an http_boot/behavioral_http browser "
    "proof."
)


def display_available() -> bool:
    """Return True when the process already has a usable window server.

    Windows and macOS always provide a native display session. On POSIX/X11
    hosts we require an explicit ``DISPLAY`` or ``WAYLAND_DISPLAY``.
    """

    if os.name != "posix":
        return True
    if sys.platform == "darwin":
        return True
    return bool(
        str(os.environ.get("DISPLAY") or "").strip()
        or str(os.environ.get("WAYLAND_DISPLAY") or "").strip()
    )


def xvfb_run_path() -> str | None:
    """Path to the ``xvfb-run`` helper if it is installed, else None."""

    return shutil.which("xvfb-run")


def _is_gui_capability(capability_tag: str) -> bool:
    return str(capability_tag or "").strip().lower() in _GUI_CAPABILITY_TAGS


def _is_gui_profile(harness_profile: str) -> bool:
    return str(harness_profile or "").strip().lower() in _GUI_HARNESS_PROFILES


def _argv_needs_display(argv: list[str] | tuple[str, ...]) -> bool:
    text = " ".join(str(part) for part in argv).lower()
    return any(marker in text for marker in _GUI_COMMAND_MARKERS)


def command_needs_display(
    argv: list[str] | tuple[str, ...],
    *,
    capability_tag: str = "",
    harness_profile: str = "",
    gui_intent: bool = False,
) -> bool:
    """Heuristic: does this command/proof need a display server to run?"""

    if gui_intent:
        return True
    if _is_gui_capability(capability_tag):
        return True
    if _is_gui_profile(harness_profile):
        return True
    return _argv_needs_display(argv)


def wrap_with_virtual_display(
    argv: list[str] | tuple[str, ...],
    *,
    server_args: str = _DEFAULT_SERVER_ARGS,
) -> list[str]:
    """Prefix ``argv`` with ``xvfb-run`` so it runs under a fresh virtual X server."""

    xvfb = xvfb_run_path()
    if not xvfb:
        return [str(part) for part in argv]
    return [xvfb, "-a", f"--server-args={server_args}", *[str(part) for part in argv]]


def apply_virtual_display(
    argv: list[str] | tuple[str, ...],
    *,
    capability_tag: str = "",
    harness_profile: str = "",
    gui_intent: bool = False,
    server_args: str = _DEFAULT_SERVER_ARGS,
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(argv, info)`` wrapped under Xvfb when a GUI command lacks a display.

    ``info`` always carries the keys ``virtual_display`` (bool, whether the argv
    was wrapped), ``display_unavailable`` (bool), ``xvfb_missing`` (bool), and
    ``note`` (str, recovery guidance when we could not provision a display).
    """

    resolved = [str(part) for part in (argv or [])]
    info: dict[str, Any] = {
        "virtual_display": False,
        "display_unavailable": False,
        "xvfb_missing": False,
        "note": "",
    }
    if not resolved:
        return resolved, info
    if not command_needs_display(
        resolved,
        capability_tag=capability_tag,
        harness_profile=harness_profile,
        gui_intent=gui_intent,
    ):
        return resolved, info
    if display_available():
        return resolved, info
    info["display_unavailable"] = True
    if not xvfb_run_path():
        info["xvfb_missing"] = True
        info["note"] = _INSTALL_HINT
        return resolved, info
    info["virtual_display"] = True
    info["note"] = "ran under an auto-provisioned Xvfb virtual display"
    return wrap_with_virtual_display(resolved, server_args=server_args), info
