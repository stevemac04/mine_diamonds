"""Abstract movement/click keys for demos: Windows SendInput (default) or PyAutoGUI."""

from __future__ import annotations

import platform
import sys
from typing import Literal

BackendName = Literal["auto", "sendinput", "pyautogui"]

_backend: BackendName = "auto"
_impl: str = "unset"  # "sendinput" | "pyautogui"


def configure(backend: BackendName = "auto") -> str:
    """Select input implementation. Call once before move_rel / keys. Returns impl name."""
    global _backend, _impl
    _backend = backend
    want = backend
    if want == "auto":
        want = "sendinput" if platform.system() == "Windows" else "pyautogui"

    if want == "sendinput":
        if platform.system() != "Windows":
            want = "pyautogui"
        else:
            try:
                import mine_diamonds.input.win_sendinput as _wsi  # noqa: F401

                _ = _wsi.move_rel  # force module load
                _impl = "sendinput"
                return _impl
            except (OSError, RuntimeError, ImportError, AttributeError):
                want = "pyautogui"

    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    _impl = "pyautogui"
    return _impl


def current_impl() -> str:
    if _impl == "unset":
        configure("auto")
    return _impl


def move_rel(dx: int, dy: int) -> None:
    if _impl == "unset":
        configure("auto")
    if _impl == "sendinput":
        from mine_diamonds.input.win_sendinput import move_rel as _mr

        _mr(int(dx), int(dy))
    else:
        import pyautogui

        pyautogui.moveRel(int(dx), int(dy), duration=0)


def mouse_left(down: bool) -> None:
    if _impl == "unset":
        configure("auto")
    if _impl == "sendinput":
        from mine_diamonds.input.win_sendinput import mouse_left as _ml

        _ml(down)
    else:
        import pyautogui

        if down:
            pyautogui.mouseDown(button="left")
        else:
            pyautogui.mouseUp(button="left")


def release_mouse_left() -> None:
    mouse_left(False)


def mouse_right(down: bool) -> None:
    if _impl == "unset":
        configure("auto")
    if _impl == "sendinput":
        from mine_diamonds.input.win_sendinput import mouse_right as _mr

        _mr(down)
    else:
        import pyautogui

        if down:
            pyautogui.mouseDown(button="right")
        else:
            pyautogui.mouseUp(button="right")


def click_at(x: int, y: int, *, button: str = "left", hold_ms: int = 60) -> None:
    """Move cursor to absolute (x, y) and press/release the given button.

    Used for inventory / recipe-book clicks. Not for in-game camera control.
    """
    if _impl == "unset":
        configure("auto")
    import time

    if _impl == "sendinput":
        from mine_diamonds.input.win_sendinput import set_cursor_pos

        set_cursor_pos(int(x), int(y))
    else:
        import pyautogui

        pyautogui.moveTo(int(x), int(y), duration=0)
    time.sleep(0.02)
    if button == "left":
        mouse_left(True)
        time.sleep(hold_ms / 1000.0)
        mouse_left(False)
    elif button == "right":
        mouse_right(True)
        time.sleep(hold_ms / 1000.0)
        mouse_right(False)
    else:
        raise ValueError(f"unsupported button {button!r}")


def type_text(text: str, *, per_char_ms: int = 35) -> None:
    """Type ASCII text into whatever has focus (lowercase only, no shift).

    Used for the recipe-book search bar. Letters and digits supported.
    """
    import time

    for ch in text.lower():
        if ch == " ":
            key_down("space")
            time.sleep(per_char_ms / 2000.0)
            key_up("space")
        elif ch.isalnum():
            key_down(ch)
            time.sleep(per_char_ms / 2000.0)
            key_up(ch)
        else:
            continue
        time.sleep(per_char_ms / 1000.0)


def key_down(name: str) -> None:
    if _impl == "unset":
        configure("auto")
    if _impl == "sendinput":
        from mine_diamonds.input.win_sendinput import key_vk

        key_vk(name, True)
    else:
        import pyautogui

        pyautogui.keyDown(name)


def key_up(name: str) -> None:
    if _impl == "unset":
        configure("auto")
    if _impl == "sendinput":
        from mine_diamonds.input.win_sendinput import key_vk

        key_vk(name, False)
    else:
        import pyautogui

        pyautogui.keyUp(name)


def release_move_keys() -> None:
    for k in ("w", "a", "s", "d"):
        try:
            key_up(k)
        except Exception:
            pass


def release_all() -> None:
    """Release WASD, space, and left mouse (safe teardown)."""
    release_move_keys()
    try:
        key_up("space")
    except Exception:
        pass
    try:
        mouse_left(False)
    except Exception:
        pass


def chat_command(
    cmd: str,
    *,
    open_chat_key: str = "t",
    open_delay_s: float = 0.20,
    paste_delay_s: float = 0.10,
    submit_delay_s: float = 0.20,
) -> bool:
    """Open Minecraft chat, paste ``cmd``, press Enter. Returns True if the
    clipboard write succeeded.

    NOTE: Minecraft must be the foreground window. Caller is responsible for
    ensuring that. We open chat with ``T`` (no prefilled slash), so callers
    should include the leading ``/`` in ``cmd`` (e.g. ``/clear @s``).

    Uses the Windows clipboard + Ctrl+V rather than typing each character so
    that ``/``, ``@``, and other shifted/punctuation chars work regardless of
    keyboard layout.
    """
    import time

    if _impl == "unset":
        configure("auto")

    try:
        from mine_diamonds.input.clipboard import set_clipboard_text
    except RuntimeError:  # pragma: no cover - non-windows
        return False

    if not set_clipboard_text(cmd):
        return False

    release_all()
    time.sleep(0.05)

    key_down(open_chat_key)
    time.sleep(0.04)
    key_up(open_chat_key)
    time.sleep(open_delay_s)

    key_down("ctrl")
    time.sleep(0.02)
    key_down("v")
    time.sleep(0.02)
    key_up("v")
    time.sleep(0.02)
    key_up("ctrl")
    time.sleep(paste_delay_s)

    key_down("enter")
    time.sleep(0.04)
    key_up("enter")
    time.sleep(submit_delay_s)
    return True
