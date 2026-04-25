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
