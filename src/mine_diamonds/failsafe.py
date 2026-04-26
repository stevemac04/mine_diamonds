"""Emergency-stop watcher for live-MC training runs.

Why this exists: ``MinecraftRealEnv`` uses Win32 SendInput to hold W and the
left mouse button. If anything goes wrong mid-training, the user needs a
RELIABLE way to (a) release those keys instantly and (b) stop the training
loop without nuking their machine.

This module spins up a single daemon thread that polls
``GetAsyncKeyState(VK_F12)`` and an optional stop-file. When either
condition fires, it:

  1. Calls ``mine_diamonds.input.game_input.release_all()`` immediately so
     the user regains control of WASD and the left mouse button in <50ms.
  2. Sets a process-wide stop event. ``is_stopping()`` returns True forever
     after that.

A small SB3 callback (``FailsafeCallback`` in the trainer) returns False on
its next ``_on_step`` so PPO exits its rollout collection loop. The trainer
then saves a final checkpoint and returns.

USAGE:
  from mine_diamonds.failsafe import install, is_stopping
  install(stop_file=Path('runs/mc_real_v1/STOP'))
  ...
  if is_stopping(): break

The watcher is idempotent — calling ``install`` twice is a no-op the second
time. Daemon thread, so it dies with the process.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from pathlib import Path

if sys.platform != "win32":  # pragma: no cover
    raise RuntimeError("mine_diamonds.failsafe is Windows-only")

_user32 = ctypes.windll.user32

# Virtual-key codes we watch. F12 is the primary hotkey because it's
# unbound in vanilla MC and unlikely to collide with anything the user is
# doing accidentally.
_VK_F12 = 0x7B
_VK_PAUSE = 0x13  # second hotkey: 'Pause' / 'Break' on most keyboards.

_stop_event = threading.Event()
_stop_reason: str = ""
_stop_file: Path | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def is_stopping() -> bool:
    """Return True once a stop has been requested. Never resets."""
    return _stop_event.is_set()


def stop_reason() -> str:
    return _stop_reason


def _emergency_release() -> None:
    """Best-effort: release every key the env could be holding right now.

    Called from the watcher thread, so it must not raise. We try the high-
    level ``release_all`` first; if anything fails we fall back to the
    raw win_sendinput primitives.
    """
    try:
        from mine_diamonds.input import game_input

        game_input.release_all()
        return
    except Exception:
        pass

    try:
        from mine_diamonds.input import win_sendinput as wsi

        for k in ("w", "a", "s", "d", "space", "shift", "ctrl"):
            try:
                wsi.key_vk(k, False)
            except Exception:
                pass
        try:
            wsi.mouse_left(False)
        except Exception:
            pass
        try:
            wsi.mouse_right(False)
        except Exception:
            pass
    except Exception:
        pass


def request_stop(reason: str = "manual") -> None:
    """Programmatic stop. Same effect as F12."""
    global _stop_reason
    if _stop_event.is_set():
        return
    _stop_reason = reason
    _emergency_release()
    _stop_event.set()
    print(f"\n[FAILSAFE] stop requested: {reason}", flush=True)
    print(
        "[FAILSAFE] all keys/mouse released. Training will exit at the next\n"
        "[FAILSAFE] env.step boundary (<50ms typical).",
        flush=True,
    )


def _watcher() -> None:
    while not _stop_event.is_set():
        if _user32.GetAsyncKeyState(_VK_F12) & 0x8000:
            request_stop("F12 hotkey")
            return
        if _user32.GetAsyncKeyState(_VK_PAUSE) & 0x8000:
            request_stop("Pause/Break hotkey")
            return
        if _stop_file is not None and _stop_file.exists():
            request_stop(f"stop-file {_stop_file} appeared")
            return
        time.sleep(0.04)


def install(stop_file: Path | None = None) -> None:
    """Start the watcher thread (idempotent). ``stop_file`` is optional."""
    global _stop_file, _thread
    with _lock:
        if _thread is not None:
            return
        _stop_file = stop_file
        _thread = threading.Thread(
            target=_watcher, daemon=True, name="mine-diamonds-failsafe"
        )
        _thread.start()


def banner(stop_file: Path | None = None) -> str:
    """Return a one-screen description of the failsafe for the user."""
    lines = [
        "=== FAILSAFE — read this BEFORE clicking into Minecraft ===",
        "  At any time, hit F12 (or Pause/Break). The watcher runs",
        "  outside Minecraft's focus, so it works even while MC is",
        "  capturing input. On press it will:",
        "     1. release W, A, S, D, Space, Ctrl, Shift, LMB, RMB",
        "     2. set a stop flag; PPO exits at the next env step",
        "     3. save a final checkpoint to <run_dir>/final.zip",
    ]
    if stop_file is not None:
        lines.append(
            f"  You can also stop from another terminal with:"
        )
        lines.append(
            f'     New-Item -ItemType File -Force -Path "{stop_file}"'
        )
    lines.append(
        "  If the keyboard is somehow unresponsive: Win+L to lock the"
    )
    lines.append(
        "  screen, then Ctrl+Shift+Esc -> kill 'python' from Task Manager."
    )
    lines.append("===========================================================")
    return "\n".join(lines)
