"""Windows SendInput: **relative** mouse motion (what Java Minecraft expects).

PyAutoGUI's Windows ``moveRel`` ultimately calls ``SetCursorPos``, which moves the
desktop cursor to absolute coordinates. Fullscreen games that use raw / relative
mouse deltas often ignore that, so the camera never turns. This module uses
``SendInput`` with ``MOUSEEVENTF_MOVE`` (no ``MOUSEEVENTF_ABSOLUTE``) so dx/dy are
true mickeys-style relative deltas.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys

if sys.platform != "win32":
    raise RuntimeError("win_sendinput is Windows-only")

user32 = ctypes.windll.user32

try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
except (AttributeError, OSError):
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MAPVK_VK_TO_VSC = 0


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.wintypes.DWORD),
        ("wParamL", ctypes.wintypes.WORD),
        ("wParamH", ctypes.wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.wintypes.DWORD), ("u", _INPUT_UNION)]


def _extra() -> ctypes.POINTER(ctypes.wintypes.ULONG):
    return ctypes.pointer(ctypes.c_ulong(0))


def _send_mouse(dx: int, dy: int, flags: int) -> None:
    dx = int(max(-32767, min(32767, dx)))
    dy = int(max(-32767, min(32767, dy)))
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(dx, dy, 0, flags, 0, _extra())
    sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput(mouse) returned {sent}; last error may be set")


def move_rel(dx: int, dy: int) -> None:
    if dx == 0 and dy == 0:
        return
    _send_mouse(dx, dy, MOUSEEVENTF_MOVE)


def mouse_left(down: bool) -> None:
    flags = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
    _send_mouse(0, 0, flags)


_VK = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "space": 0x20,
}


def key_vk(name: str, down: bool) -> None:
    key = name.lower()
    vk = _VK.get(key)
    if vk is None:
        raise ValueError(f"win_sendinput: unsupported key {name!r} (use w,a,s,d,space)")
    scan = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) & 0xFF
    flags = KEYEVENTF_SCANCODE
    if not down:
        flags |= KEYEVENTF_KEYUP
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki = KEYBDINPUT(0, scan, flags, 0, _extra())
    sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput(keyboard) returned {sent}")
