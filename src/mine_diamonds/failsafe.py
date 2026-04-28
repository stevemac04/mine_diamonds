"""Emergency-stop watcher for live-MC training runs.

Why this exists: ``MinecraftRealEnv`` uses Win32 SendInput to hold W and the
left mouse button. If anything goes wrong mid-training, the user needs a
RELIABLE way to (a) release those keys instantly and (b) stop the training
loop without nuking their machine.

This module spins up a single daemon thread that polls ``GetAsyncKeyState``
for **F12** (debounced — several consecutive polls must see it down) and,
after a short grace period, an optional **stop-file**. When a stop fires:

  1. Calls ``mine_diamonds.input.game_input.release_all()`` immediately.
  2. Sets a process-wide stop event. ``is_stopping()`` returns True forever
     after that.

**Pause/Break is not watched** — on many laptops ``GetAsyncKeyState`` for
VK_PAUSE glitches high and was stopping runs with no user input.

**Stop-file grace:** for a few seconds after ``install()``, the watcher
ignores the stop-file so cloud sync / tooling cannot instantly kill a new
run right after ``/clear`` (the user symptom was "stops after clear").

**Windows Sticky Keys / Filter Keys** can make ``GetAsyncKeyState(F12)``
report \"down\" when you did not press it — disable those accessibility
keyboard features while training.

USAGE:
  from mine_diamonds.failsafe import install, is_stopping
  install(stop_file=Path('runs/mc_real_v1/STOP'))
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

_VK_F12 = 0x7B

_stop_event = threading.Event()
_stop_reason: str = ""
_stop_file: Path | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()
# Monotonic time when install() finished starting the watcher; stop-file
# ignored until this + STOP_FILE_GRACE_S (seconds).
_watch_start_perf: float = 0.0
STOP_FILE_GRACE_S: float = 4.0
# Sticky Keys / accessibility can latch F12; extra polls reduce false stops.
_HOTKEY_CONSECUTIVE_POLLS = 5
_POLL_INTERVAL_S = 0.03


def is_stopping() -> bool:
    """Return True once a stop has been requested. Never resets."""
    return _stop_event.is_set()


def stop_reason() -> str:
    return _stop_reason


def _emergency_release() -> None:
    """Best-effort: release every key the env could be holding right now."""
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
    global _watch_start_perf
    f12_down_ticks = 0
    while not _stop_event.is_set():
        f12_now = bool(_user32.GetAsyncKeyState(_VK_F12) & 0x8000)
        if f12_now:
            f12_down_ticks += 1
            if f12_down_ticks >= _HOTKEY_CONSECUTIVE_POLLS:
                request_stop("F12 hotkey")
                return
        else:
            f12_down_ticks = 0

        if _stop_file is not None:
            elapsed = time.perf_counter() - _watch_start_perf
            if elapsed >= STOP_FILE_GRACE_S and _stop_file.exists():
                request_stop(f"stop-file {_stop_file} appeared")
                return

        time.sleep(_POLL_INTERVAL_S)


def install(stop_file: Path | None = None) -> None:
    """Start the watcher thread (idempotent).

    Clears any prior stop state, removes a leftover stop-file, then starts
    the watcher. Stop-file polling is disabled for STOP_FILE_GRACE_S so a
    stray file cannot end the run before the first env steps.
    """
    global _stop_file, _thread, _stop_reason, _watch_start_perf
    with _lock:
        if _thread is not None:
            return
        _stop_event.clear()
        _stop_reason = ""
        if stop_file is not None:
            p = Path(stop_file)
            if p.exists():
                try:
                    p.unlink()
                    print(
                        f"[FAILSAFE] removed leftover stop file: {p}",
                        flush=True,
                    )
                except OSError as e:
                    print(f"[FAILSAFE] WARN: could not remove {p}: {e}", flush=True)
        _stop_file = Path(stop_file) if stop_file is not None else None
        _watch_start_perf = time.perf_counter()
        _thread = threading.Thread(
            target=_watcher, daemon=True, name="mine-diamonds-failsafe"
        )
        _thread.start()


def banner(stop_file: Path | None = None) -> str:
    """Return a one-screen description of the failsafe for the user."""
    lines = [
        "=== FAILSAFE — read this BEFORE clicking into Minecraft ===",
        "  Hold F12 ~0.15s to stop (debounced; brief taps are ignored).",
        "  Turn OFF Windows Sticky Keys / Filter Keys / Toggle Keys",
        "  (Settings → Accessibility → Keyboard) — they fake F12 to the watcher.",
        "  Stop-file is ignored for the first few seconds after start.",
        "  On stop:",
        "     1. release W, A, S, D, Space, Ctrl, Shift, LMB, RMB",
        "     2. set a stop flag; PPO exits at the next env step",
        "     3. save a final checkpoint to <run_dir>/final.zip",
    ]
    if stop_file is not None:
        lines.append("  You can also stop from another terminal with:")
        lines.append(f'     New-Item -ItemType File -Force -Path "{stop_file}"')
    lines.append(
        "  If the keyboard is somehow unresponsive: Win+L to lock the"
    )
    lines.append(
        "  screen, then Ctrl+Shift+Esc -> kill 'python' from Task Manager."
    )
    lines.append("===========================================================")
    return "\n".join(lines)
