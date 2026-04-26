"""Find and capture the Minecraft Java client window on Windows.

Why we need this: ``mss`` will happily capture the entire monitor, but
the Minecraft client almost never fills the monitor edge-to-edge —
there's a Windows title bar, borders, taskbar, etc. If we compute the
hotbar ROI from monitor dimensions but capture the whole monitor, the
ROI lands on the desktop / taskbar, not on the in-game hotbar. That's
why ``slot_log_px`` was stuck at 0 in the first smoke run.

This module enumerates top-level windows, finds the one whose title
matches a substring (default ``"minecraft"``, case-insensitive,
excluding ``"launcher"``), and returns the **client area** of that
window in screen-pixel coordinates. The client area is the part MC
actually renders into — it excludes title bar and borders.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

if sys.platform != "win32":  # pragma: no cover
    raise RuntimeError("mine_diamonds.capture is Windows-only")

user32 = ctypes.windll.user32

# Make sure ClientToScreen returns physical pixels even when the user has
# display scaling > 100% in Windows.
try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
except (AttributeError, OSError):
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


@dataclass(frozen=True)
class WindowRegion:
    """Screen-pixel rectangle for a captured window's client area.

    Use ``as_mss_monitor()`` to plug into ``mss.mss().grab(...)``.
    """

    title: str
    left: int
    top: int
    width: int
    height: int

    def as_mss_monitor(self) -> dict:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }


_DEFAULT_EXCLUDES: tuple[str, ...] = (
    "launcher",
    # Editors / IDEs whose titles often contain a project or PR named
    # "minecraft" (this exact bug showed up in the smoke run when Cursor
    # was titled 'Review: ... Minecraft cr... - Cursor').
    "cursor",
    "visual studio",
    "vscode",
    " - code",
    "intellij",
    "pycharm",
    # Chat / docs / browsers that may have a "minecraft" tab/title.
    "discord",
    "slack",
    "obsidian",
    "notion",
    "chrome",
    "firefox",
    "edge",
    "brave",
    "opera",
    # Terminals.
    "powershell",
    "command prompt",
    "windows terminal",
)


def find_minecraft_window(
    *,
    title_substr: str = "minecraft",
    exclude_substrs: tuple[str, ...] = _DEFAULT_EXCLUDES,
    require_prefix: bool = True,
) -> WindowRegion | None:
    """Return the topmost visible window whose title matches ``title_substr``.

    Args:
        title_substr: Case-insensitive substring to look for in window
            titles. Default ``"minecraft"`` matches typical client titles
            like ``"Minecraft 1.20.1"`` or ``"Minecraft* 1.21.1"``.
        exclude_substrs: Skip any window whose title contains any of these
            substrings (case-insensitive). Default excludes the launcher
            plus common IDEs, browsers, and chat apps that could have a
            tab/file/PR named "minecraft".
        require_prefix: If True (default), the title must START with
            ``title_substr`` (case-insensitive). The MC client titles its
            window ``"Minecraft <version> - ..."`` so this is the right
            disambiguator: it rules out e.g. Cursor windows whose title
            contains "Minecraft" mid-string from a file or PR name.
    """
    title_substr = title_substr.lower()
    exclude_substrs = tuple(s.lower() for s in exclude_substrs)
    found: list[WindowRegion] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        title_low = title.lower()
        if require_prefix:
            if not title_low.startswith(title_substr):
                return True
        else:
            if title_substr not in title_low:
                return True
        if any(s in title_low for s in exclude_substrs):
            return True
        client = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(client)):
            return True
        cw = int(client.right - client.left)
        ch = int(client.bottom - client.top)
        if cw <= 0 or ch <= 0:
            return True
        pt = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return True
        found.append(
            WindowRegion(title=title, left=int(pt.x), top=int(pt.y), width=cw, height=ch)
        )
        return True

    user32.EnumWindows(callback, 0)
    return found[0] if found else None


def list_candidate_windows(
    *,
    title_substr: str = "minecraft",
) -> list[str]:
    """Return all visible top-level window titles that contain ``title_substr``.
    Diagnostic helper for when detection picks the wrong window.
    """
    title_substr = title_substr.lower()
    titles: list[str] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        if title_substr in title.lower():
            titles.append(title)
        return True

    user32.EnumWindows(callback, 0)
    return titles


def focus_window_by_title(title_substr: str = "minecraft") -> bool:
    """Best-effort: bring the named window to the foreground.

    Useful for self-healing in long training runs if focus is lost. Returns
    True on success, False if no matching window exists or Windows refused
    the foreground request (which it can do for security reasons).
    """
    region = find_minecraft_window(title_substr=title_substr)
    if region is None:
        return False
    title_substr_low = title_substr.lower()
    found_hwnd: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if title_substr_low in buf.value.lower():
            found_hwnd.append(hwnd)
            return False
        return True

    user32.EnumWindows(callback, 0)
    if not found_hwnd:
        return False
    return bool(user32.SetForegroundWindow(found_hwnd[0]))
