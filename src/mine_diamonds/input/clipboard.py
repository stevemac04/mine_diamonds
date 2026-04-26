"""Tiny Win32 clipboard helper. Pure ctypes, zero new deps.

Used for sending Minecraft chat commands like ``/clear @s`` without having
to type them character by character (which would require shift handling
and per-keyboard-layout VK tables for ``/`` and ``@``).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

if sys.platform != "win32":  # pragma: no cover
    raise RuntimeError("mine_diamonds.input.clipboard is Windows-only")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

_GMEM_MOVEABLE = 0x0002
_CF_UNICODETEXT = 13

kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
kernel32.GlobalAlloc.restype = wintypes.HANDLE
kernel32.GlobalLock.argtypes = (wintypes.HANDLE,)
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = (wintypes.HANDLE,)
kernel32.GlobalUnlock.restype = wintypes.BOOL
user32.OpenClipboard.argtypes = (wintypes.HWND,)
user32.OpenClipboard.restype = wintypes.BOOL
user32.EmptyClipboard.restype = wintypes.BOOL
user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
user32.SetClipboardData.restype = wintypes.HANDLE
user32.CloseClipboard.restype = wintypes.BOOL


def set_clipboard_text(text: str) -> bool:
    """Replace the system clipboard with ``text``. Returns True on success.

    Caller is responsible for ensuring the foreground window has focus when
    Ctrl+V is sent afterwards.
    """
    payload = text.encode("utf-16-le") + b"\x00\x00"
    if not user32.OpenClipboard(0):
        return False
    try:
        user32.EmptyClipboard()
        h_mem = kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(payload))
        if not h_mem:
            return False
        p_mem = kernel32.GlobalLock(h_mem)
        if not p_mem:
            return False
        try:
            ctypes.memmove(p_mem, payload, len(payload))
        finally:
            kernel32.GlobalUnlock(h_mem)
        if not user32.SetClipboardData(_CF_UNICODETEXT, h_mem):
            return False
        return True
    finally:
        user32.CloseClipboard()
