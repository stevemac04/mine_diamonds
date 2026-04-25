"""Minecraft Java: automated tree-chopping demo (keyboard/mouse only, not RL).

This script sends real inputs to whichever window has focus. It does **not**
load your PPO checkpoint; it is a deterministic macro for a midpoint demo.

Before you run:
  - Single-player world (Peaceful is fine).
  - Stand **close to a tree trunk** (oak/spruce/etc.) with clear line of sight.
  - Hold nothing in off-hand that blocks clicks.
  - **Click inside the Minecraft window** so it captures mouse, then alt-tab to
    this terminal to start the script; when the countdown ends, click back into
    Minecraft immediately (or start from a second monitor with MC focused).

Controls:
  - Hold **Esc** during the run to stop (releases keys/mouse in a finally block).

Install (main project venv):
  pip install -e ".[minecraft-demo]"

Run:
  python scripts\\demo_treechop_macro.py
  python scripts\\demo_treechop_macro.py --cycles 12 --seconds-per-cycle 12
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import keyboard
import pyautogui


def _stop() -> bool:
    return keyboard.is_pressed("esc")


def _release_all() -> None:
    for k in ("w", "a", "s", "d", "space"):
        try:
            pyautogui.keyUp(k)
        except Exception:
            pass
    try:
        pyautogui.mouseUp(button="left")
    except Exception:
        pass


def run_treechop_demo(
    *,
    countdown: int,
    cycles: int,
    seconds_per_cycle: float,
    yaw_step: int,
    pitch_step: int,
    jump_every_s: float,
) -> None:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    print(
        "\n=== Minecraft tree-chop macro ===\n"
        "1) In-game: stand touching a tree trunk, first-person, survival punch.\n"
        "2) Click the Minecraft window so it has focus.\n"
        f"3) You have {countdown}s to focus Minecraft after the timer starts here.\n"
        "4) Hold Esc in-game to abort.\n"
        "Tip: raise mouse sensitivity slightly if the camera barely moves.\n"
    )

    for i in range(countdown, 0, -1):
        print(f"Starting in {i}… (focus Minecraft now)")
        time.sleep(1)
        if _stop():
            print("Aborted before start.")
            return

    print("Mining — hold Esc to stop.\n")

    try:
        for c in range(cycles):
            if _stop():
                print("Stopped (Esc).")
                return

            deadline = time.perf_counter() + seconds_per_cycle
            pyautogui.mouseDown(button="left")
            pyautogui.keyDown("w")
            t = 0.0
            jump_clock = 0.0

            while time.perf_counter() < deadline:
                if _stop():
                    return

                dt = 0.05
                t += dt
                jump_clock += dt

                # Gentle figure-eight on the crosshair: finds trunk/leaves voxels better than yaw-only.
                yaw = int(yaw_step * math.sin(t * 2.2))
                pitch = int(pitch_step * math.sin(t * 1.7))
                pyautogui.moveRel(yaw, pitch, duration=0)

                # Occasional jump helps when the trunk is one block above feet or you're slightly stuck.
                if jump_clock >= jump_every_s:
                    jump_clock = 0.0
                    pyautogui.keyDown("space")
                    time.sleep(0.06)
                    pyautogui.keyUp("space")

                time.sleep(dt)

            pyautogui.mouseUp(button="left")
            pyautogui.keyUp("w")
            # Tiny strafe nudge between cycles so you don't stare at the same miss forever.
            if c % 2 == 0:
                pyautogui.keyDown("d")
                time.sleep(0.08)
                pyautogui.keyUp("d")
            else:
                pyautogui.keyDown("a")
                time.sleep(0.08)
                pyautogui.keyUp("a")
            time.sleep(0.15)

            print(f"  cycle {c + 1}/{cycles} done")
    finally:
        _release_all()
        print("\nMacro finished (keys released).")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Minecraft tree-chop keyboard/mouse macro.")
    p.add_argument("--countdown", type=int, default=5, help="Seconds before inputs start.")
    p.add_argument("--cycles", type=int, default=10, help="How many long mine bursts.")
    p.add_argument(
        "--seconds-per-cycle",
        type=float,
        default=10.0,
        help="Seconds holding attack+forward per cycle (longer = more blocks broken).",
    )
    p.add_argument(
        "--yaw-step",
        type=int,
        default=6,
        help="Crosshair yaw amplitude (pixels per tick). Increase if camera barely moves.",
    )
    p.add_argument(
        "--pitch-step",
        type=int,
        default=4,
        help="Crosshair pitch amplitude (pixels per tick).",
    )
    p.add_argument(
        "--jump-every",
        type=float,
        default=2.5,
        help="Seconds between small jumps while mining.",
    )
    args = p.parse_args(argv)

    if args.countdown < 0 or args.cycles < 1 or args.seconds_per_cycle <= 0:
        print("Invalid arguments.", file=sys.stderr)
        return 2

    try:
        run_treechop_demo(
            countdown=args.countdown,
            cycles=args.cycles,
            seconds_per_cycle=args.seconds_per_cycle,
            yaw_step=args.yaw_step,
            pitch_step=args.pitch_step,
            jump_every_s=args.jump_every,
        )
    except KeyboardInterrupt:
        _release_all()
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
