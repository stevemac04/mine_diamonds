"""Chop ONE log with a hand-aimed scripted attack.

This is the "is the plumbing alive?" sanity demo. You manually face a tree;
we send a continuous FORWARD_ATTACK for ``--attack-seconds`` seconds (default
6, comfortably above the ~3s fist-mining time for an oak log) and report
whether the env saw ``log_acquired``.

If this demo succeeds, you have proof that:
  1. Window detection grabbed the right MC window.
  2. Hotbar slot 1 ROI is correctly placed.
  3. Synthetic input reaches MC and is interpreted as held-LMB (not a series
     of click-release-clicks, which would never break a log).
  4. The diff detector fires when a log appears in slot 1.

If this demo FAILS even though you were aimed at a tree, do NOT start the
4-hour training run — something is broken in the input or capture path.

USAGE (PowerShell):
  .\\.venv\\Scripts\\Activate.ps1
  $env:PYTHONUTF8=1
  python scripts\\demo_chop_one_log.py --countdown 5

You'll be prompted to focus Minecraft, face a tree (crosshair on a log
block), and stay still. Then the script holds the attack for ~6s.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mine_diamonds.envs.minecraft_real import (  # noqa: E402
    FORWARD_ATTACK,
    NOOP,
    MinecraftRealConfig,
    MinecraftRealEnv,
)


def countdown(seconds: int, msg: str) -> None:
    for i in range(seconds, 0, -1):
        print(f"  {msg} ({i}s)", flush=True)
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--attack-seconds", type=float, default=6.0)
    parser.add_argument("--gui-scale", type=int, default=2)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument(
        "--no-clear-on-reset",
        action="store_true",
        help="Skip the /clear @s before the demo. Use if you're already empty.",
    )
    args = parser.parse_args()

    cfg = MinecraftRealConfig(
        monitor_index=args.monitor,
        max_seconds=args.attack_seconds + 5.0,
        gui_scale=int(args.gui_scale),
        # If user passed --no-clear-on-reset, drop the chat command so we
        # don't accidentally trash inventory during a focused demo.
        auto_reset_chat_commands=()
        if args.no_clear_on_reset
        else ("/clear @s",),
    )
    env = MinecraftRealEnv(cfg)
    env._ensure_capture()  # type: ignore[attr-defined]
    print(f"\n=== capture source: {env._capture_source} ===")  # type: ignore[attr-defined]
    print(f"  hotbar ROI = {env._hotbar}")  # type: ignore[attr-defined]
    print(f"  fovea ROI  = {env._fovea}\n")  # type: ignore[attr-defined]

    print("DEMO: chop one log.")
    print("  1. Click into Minecraft so it has KEYBOARD FOCUS.")
    print("  2. Stand still, face a tree, crosshair on a LOG block.")
    print(f"  3. We will hold FORWARD+ATTACK for {args.attack_seconds}s.")
    print("  4. Stay still. Don't press anything until the demo finishes.\n")
    countdown(args.countdown, "starting in")

    obs, reset_info = env.reset()
    diff_thresh = float(reset_info.get("slot_diff_threshold", 0.0))
    print(
        f"  reset done. baseline_px={reset_info['slot_empty_baseline_px']}, "
        f"diff_threshold={diff_thresh:.1f}"
    )

    n_steps = max(1, int(args.attack_seconds / cfg.frame_time_s))
    print(f"  attacking for {n_steps} steps ({args.attack_seconds:.1f}s)...\n")

    saw_log = False
    max_diff = 0.0
    last_print = 0.0
    for i in range(n_steps):
        obs, reward, term, trunc, info = env.step(FORWARD_ATTACK)
        max_diff = max(max_diff, float(info["slot_diff"]))
        now = time.time()
        if now - last_print > 0.5:
            print(
                f"    t={i*cfg.frame_time_s:4.1f}s  "
                f"slot_diff={info['slot_diff']:6.2f}  "
                f"slot_log_px={info['slot_log_px']:>5}  "
                f"have_log_now={info['have_log_now']!s:<5}  "
                f"fovea={info['fovea_log_frac']:.3f}",
                flush=True,
            )
            last_print = now
        if info["log_acquired"]:
            print(
                f"  ** LOG ACQUIRED at t={i*cfg.frame_time_s:.2f}s "
                f"(slot_diff={info['slot_diff']:.2f}) **",
                flush=True,
            )
            saw_log = True
            break
        if term or trunc:
            break
    env.step(NOOP)

    print(
        f"\n  result: saw_log_acquired={saw_log}, "
        f"max_slot_diff={max_diff:.2f} (threshold={diff_thresh:.1f})"
    )
    if saw_log:
        print(
            "\n  PASS. Mining works end-to-end. You're cleared to start the\n"
            "  4-hour training run."
        )
        return 0

    print(
        "\n  FAIL. The agent didn't break a log in the attack window.\n"
        "  Common causes (check in order):\n"
        "    - You weren't aimed at a log block. Re-run; make sure the\n"
        "      crosshair is on a LOG (not leaves, not air).\n"
        "    - MC window lost focus. Click into MC during the countdown\n"
        "      and don't alt-tab. The MC title bar should be highlighted.\n"
        "    - Survival without a fist? You were holding a tool that\n"
        "      doesn't break wood (e.g. a pickaxe). Empty the hand.\n"
        "    - The slot ROI is misaligned and a real log went in but\n"
        "      we didn't see it. Open eval/smoke_minecraft/hotbar_slot.png\n"
        "      after the next smoke run and verify what's being read.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
