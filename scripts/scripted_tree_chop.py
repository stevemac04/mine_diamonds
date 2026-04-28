"""Hand-coded tree chopper — find a tree, walk to it, mine it. No RL.

State machine that runs at ~10 Hz:

  SCAN     : no log pixels visible. Yaw right in steady ticks until we
             see one. Re-pitches toward the horizon every full sweep so
             we don't get stuck staring at the sky/feet.
  CENTER   : log visible but its centroid is off-center horizontally.
             Yaw toward it in proportion to the offset.
  APPROACH : log centered but small (small fovea fraction). Hold W to
             walk toward it. If the path looks blocked, hop with Space.
  MINE     : log fills enough of the fovea. Hold W + LMB until slot 1
             changes (slot_diff_from_empty exceeds threshold).
  DONE     : log acquired. Release everything and exit.

This is the "the building blocks work, just glue them together" demo.
If THIS reliably chops a tree, RL is purely an exploration / generalization
upgrade on top of a working pipeline. If this doesn't work, RL never
will either.

USAGE (PowerShell):
  .\\.venv\\Scripts\\Activate.ps1
  python scripts\\scripted_tree_chop.py --countdown 8

F12 / Pause stops it instantly. Holds nothing on exit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mine_diamonds import failsafe as _failsafe  # noqa: E402
from mine_diamonds.envs.minecraft_real import (  # noqa: E402
    MinecraftRealConfig,
    MinecraftRealEnv,
)
from mine_diamonds.input import game_input as ginput  # noqa: E402


def find_log_centroid(
    mask: np.ndarray, *, min_pixels: int = 80, top_crop_frac: float = 0.0,
    bottom_crop_frac: float = 0.12,
) -> tuple[float, float, float] | None:
    """Return ``(cx, cy, frac)`` of the log mask, or None if too few log pixels.

    ``cx, cy`` are normalized to [0, 1] (0 = left/top of the cropped region).
    ``frac`` is the fraction of cropped pixels that are log.

    The hotbar is excluded by cropping the bottom of the frame; ``top_crop_frac``
    can be used to exclude the sky. We use ``cv2.moments`` so the centroid is
    biased toward big blobs (i.e. tree trunks) rather than scattered specks
    of black UI text.
    """
    h, w = mask.shape[:2]
    top = int(h * top_crop_frac)
    bot = int(h * (1.0 - bottom_crop_frac))
    if bot <= top:
        return None
    sub = mask[top:bot, :]
    n = int(np.count_nonzero(sub))
    if n < min_pixels:
        return None
    M = cv2.moments(sub, binaryImage=True)
    if M["m00"] < 1:
        return None
    cx = M["m10"] / M["m00"] / sub.shape[1]
    cy = M["m01"] / M["m00"] / sub.shape[0]
    return float(cx), float(cy), n / sub.size


class HeldState:
    """Cheap idempotent wrapper around ginput so we only fire transitions."""

    def __init__(self) -> None:
        self.w = False
        self.lmb = False
        self.space = False

    def set_w(self, on: bool) -> None:
        if on != self.w:
            (ginput.key_down if on else ginput.key_up)("w")
            self.w = on

    def set_lmb(self, on: bool) -> None:
        if on != self.lmb:
            ginput.mouse_left(on)
            self.lmb = on

    def set_space(self, on: bool) -> None:
        if on != self.space:
            (ginput.key_down if on else ginput.key_up)("space")
            self.space = on

    def release_all(self) -> None:
        self.set_w(False)
        self.set_lmb(False)
        self.set_space(False)


IMMORTAL_CMDS = (
    "/gamemode survival",
    "/effect give @s minecraft:resistance 999999 4 true",
    "/effect give @s minecraft:regeneration 999999 4 true",
    "/effect give @s minecraft:saturation 999999 4 true",
    "/effect give @s minecraft:fire_resistance 999999 0 true",
    "/effect give @s minecraft:water_breathing 999999 0 true",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--countdown", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=180.0,
                   help="Hard cap on the search-and-chop phase; releases "
                        "inputs and exits if exceeded.")
    p.add_argument("--target-logs", type=int, default=1)
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--gui-scale", type=int, default=2)
    p.add_argument("--tick-hz", type=float, default=10.0,
                   help="Decision rate. Higher is more responsive, but each "
                        "yaw step is smaller; the camera looks twitchy.")
    p.add_argument("--scan-yaw-px", type=int, default=80,
                   help="Mickeys per yaw tick during SCAN sweep.")
    p.add_argument("--mine-fovea-frac", type=float, default=0.05,
                   help="Fovea log fraction at which we switch to MINE.")
    p.add_argument("--center-tol", type=float, default=0.04,
                   help="Allowed horizontal centroid offset from 0.5 before "
                        "we stop yaw-correcting and start walking.")
    p.add_argument("--center-yaw-gain", type=float, default=400.0,
                   help="Mickeys per unit centroid error. 0.1 off-center "
                        "with gain 400 -> 40 mickeys per tick.")
    p.add_argument("--approach-jump-frac", type=float, default=0.02,
                   help="If we've been APPROACHing for a while with no "
                        "growth in fovea fraction, hop. Helps over fences "
                        "and 1-block rises.")
    p.add_argument("--acquire-confirm-frames", type=int, default=3,
                   help="Require this many consecutive ticks with slot_diff "
                        "above threshold before counting a mined log.")
    p.add_argument("--min-mine-seconds", type=float, default=1.0,
                   help="Require spending at least this many seconds in MINE "
                        "before slot-diff can count as acquired.")
    p.add_argument("--acquire-margin", type=float, default=2.0,
                   help="Extra margin added to slot_diff_threshold to avoid "
                        "single-frame false positives from UI/lighting.")
    p.add_argument("--mine-commit-seconds", type=float, default=1.4,
                   help="Once MINE starts, keep holding W+LMB for at least "
                        "this long before allowing center-corrections. "
                        "Prevents jittery MINE<->CENTER flapping.")
    p.add_argument("--acquire-log-px-min", type=int, default=80,
                   help="Also require at least this many log-colored pixels "
                        "inside slot 1 before counting acquisition. Helps "
                        "ignore apple/seed pickups that change slot diff.")
    p.add_argument("--mine-exit-fovea-frac", type=float, default=0.015,
                   help="While in MINE, stay committed as long as fovea log "
                        "fraction is above this value.")
    p.add_argument("--mine-exit-blind-frames", type=int, default=8,
                   help="Require this many consecutive near-blind frames "
                        "before leaving MINE.")
    p.add_argument("--no-immortal", action="store_true",
                   help="Skip the resistance/regen/etc. setup commands. "
                        "Use only if you're already invincible or want a "
                        "raw run.")
    p.add_argument("--craft", choices=("none", "table", "pickaxe"),
                   default="pickaxe",
                   help="After acquiring the log(s), run the scripted "
                        "crafting sequence. 'pickaxe' implies 'table' as a "
                        "prerequisite. Default: pickaxe.")
    p.add_argument("--logs-for-pickaxe", type=int, default=3,
                   help="If --craft=pickaxe, mine this many logs before "
                        "switching to crafting. Pickaxe needs 3 (1 for "
                        "the table, 2 for sticks+head).")
    args = p.parse_args()

    _failsafe.install()

    if args.craft == "pickaxe":
        target_logs = max(args.target_logs, args.logs_for_pickaxe)
    else:
        target_logs = args.target_logs

    cfg = MinecraftRealConfig(
        monitor_index=args.monitor,
        gui_scale=int(args.gui_scale),
        max_seconds=args.max_seconds + 5.0,
        action_repeat=1,
        init_chat_commands=() if args.no_immortal else IMMORTAL_CMDS,
        auto_reset_chat_commands=("/clear @s",),
        timeout_extra_chat_commands=(),
    )
    env = MinecraftRealEnv(cfg)
    env._ensure_capture()  # type: ignore[attr-defined]
    print(f"capture source: {env._capture_source}")  # type: ignore[attr-defined]
    print(f"hotbar ROI    : {env._hotbar}")  # type: ignore[attr-defined]
    print(f"fovea ROI     : {env._fovea}\n")  # type: ignore[attr-defined]

    print("SCRIPTED TREE CHOP" + ("  +  CRAFT " + args.craft.upper() if args.craft != "none" else ""))
    print("  - Open MC, stand in a forested biome where trees are visible.")
    print("  - World must have CHEATS ENABLED (we /gamemode + /effect on init).")
    print("  - Make sure slot 1 is empty BEFORE the countdown ends.")
    print(f"  - We'll find + chop {target_logs} log(s)"
          + (f", then craft '{args.craft}'." if args.craft != "none" else ", then exit."))
    print("  - F12 (or Pause) stops everything immediately.\n")

    for i in range(args.countdown, 0, -1):
        if _failsafe.is_stopping():
            print("F12 pressed before start; exiting.")
            return 130
        print(f"  starting in {i}s")
        time.sleep(1.0)

    obs, reset_info = env.reset()
    diff_thresh = float(reset_info.get("slot_diff_threshold", 8.0))
    slot_log_px_thresh = int(reset_info.get("effective_log_threshold", args.acquire_log_px_min))
    slot_log_px_thresh = max(int(args.acquire_log_px_min), min(slot_log_px_thresh, 220))
    acquire_thresh = diff_thresh + float(args.acquire_margin)
    print(
        f"reset done. slot_diff_threshold={diff_thresh:.1f}, "
        f"acquire_thresh={acquire_thresh:.1f}, "
        f"slot_log_px_thresh={slot_log_px_thresh}, "
        f"confirm_frames={int(args.acquire_confirm_frames)}, "
        f"min_mine_s={float(args.min_mine_seconds):.1f}\n"
    )

    held = HeldState()
    tick_dt = 1.0 / max(1.0, args.tick_hz)

    logs_collected = 0
    chop_target = target_logs
    state = "SCAN"
    state_started_at = time.perf_counter()
    mine_started_at: float | None = None
    mine_commit_until: float = 0.0
    mine_blind_consec: int = 0
    last_fovea_at_state_start = 0.0
    sweep_start = time.perf_counter()
    deadline = time.perf_counter() + args.max_seconds
    acquire_consec = 0

    def transition(new_state: str, fovea: float, full: float) -> None:
        nonlocal state, state_started_at, last_fovea_at_state_start, mine_started_at, mine_commit_until, mine_blind_consec, acquire_consec
        if new_state != state:
            print(
                f"  [t={time.perf_counter() - sweep_start:5.1f}s] "
                f"{state:<8} -> {new_state:<8} "
                f"fovea={fovea:.3f} full={full:.3f}"
            )
            state = new_state
            state_started_at = time.perf_counter()
            last_fovea_at_state_start = fovea
            if new_state == "MINE":
                mine_started_at = state_started_at
                mine_commit_until = state_started_at + float(args.mine_commit_seconds)
                mine_blind_consec = 0
                acquire_consec = 0
            else:
                mine_started_at = None

    try:
        while logs_collected < chop_target:
            if _failsafe.is_stopping():
                print("F12 pressed; stopping.")
                break
            if time.perf_counter() > deadline:
                print(f"hit max-seconds={args.max_seconds:.0f}; stopping.")
                break

            tick_start = time.perf_counter()

            frame = env._grab_full_bgr()  # type: ignore[attr-defined]
            if frame.size == 0:
                time.sleep(tick_dt)
                continue

            mask = env._log_mask(frame)  # type: ignore[attr-defined]
            ctr = find_log_centroid(mask)
            full_frac = env._log_pixel_frac_full(frame)  # type: ignore[attr-defined]
            fovea_frac = env._log_pixel_frac_fovea(frame)  # type: ignore[attr-defined]
            slot_diff = env._slot_diff_from_empty(frame)  # type: ignore[attr-defined]
            slot_log_px = env._log_pixels_in_slot(frame)  # type: ignore[attr-defined]

            # Robust acquisition gate: only count when we've truly been mining
            # for a beat and slot diff is stable above threshold for multiple ticks.
            mine_for_s = (
                (time.perf_counter() - mine_started_at)
                if (mine_started_at is not None and state == "MINE")
                else 0.0
            )
            if slot_diff >= acquire_thresh:
                acquire_consec += 1
            else:
                acquire_consec = 0

            if (
                state == "MINE"
                and mine_for_s >= float(args.min_mine_seconds)
                and acquire_consec >= max(1, int(args.acquire_confirm_frames))
                and slot_log_px >= int(slot_log_px_thresh)
            ):
                logs_collected += 1
                print(
                    f"  [t={time.perf_counter() - sweep_start:5.1f}s] "
                    f"LOG ACQUIRED #{logs_collected} "
                    f"(slot_diff={slot_diff:.1f} >= {acquire_thresh:.1f}, "
                    f"slot_log_px={slot_log_px} >= {slot_log_px_thresh}, "
                    f"consec={acquire_consec}, mine_for={mine_for_s:.1f}s)"
                )
                held.release_all()
                if logs_collected >= chop_target:
                    transition("DONE", fovea_frac, full_frac)
                    break
                # Multiple-log path: continue mining the SAME tree (oak
                # has 4-7 stacked log blocks). Don't reset the baseline —
                # we want to keep detecting NEW logs against the slot's
                # progressively-fuller state, but the simple slot_diff
                # detector compares to the EMPTY baseline so it won't
                # re-fire until the next reset. For multi-log mining we
                # just sleep a beat and keep mining; the next log entering
                # the same stack also raises slot_diff above threshold,
                # but staying above threshold doesn't re-fire (we track
                # the rising edge only). To force re-detection we briefly
                # re-snapshot the now-occupied slot so the NEXT log is
                # detected as a fresh diff.
                time.sleep(0.4)
                fresh_frame = env._grab_full_bgr()  # type: ignore[attr-defined]
                fresh_slot = env._slot_crop(fresh_frame)  # type: ignore[attr-defined]
                if fresh_slot.size > 0:
                    env._slot_empty_frame = fresh_slot.copy()  # type: ignore[attr-defined]
                state = "MINE"  # keep chopping the same tree
                state_started_at = time.perf_counter()
                mine_started_at = state_started_at
                mine_commit_until = state_started_at + float(args.mine_commit_seconds)
                acquire_consec = 0
                # Resume holding W + LMB right away.
                held.set_w(True)
                held.set_lmb(True)
                continue

            if ctr is None:
                # No log pixels at all -> sweep.
                transition("SCAN", fovea_frac, full_frac)
                held.set_w(False)
                held.set_lmb(False)
                held.set_space(False)
                ginput.move_rel(args.scan_yaw_px, 0)
                # Every ~3.6s of sweeping, nudge pitch toward horizon in case
                # we're staring at the sky or our feet. Cheap and bounded.
                if time.perf_counter() - state_started_at > 3.6:
                    ginput.move_rel(0, -8)
                    state_started_at = time.perf_counter()
                _sleep_remaining(tick_start, tick_dt)
                continue

            cx, cy, frac = ctr
            err_x = cx - 0.5
            now = time.perf_counter()
            in_mine_commit = (state == "MINE" and now < mine_commit_until)
            if state == "MINE":
                if fovea_frac <= float(args.mine_exit_fovea_frac):
                    mine_blind_consec += 1
                else:
                    mine_blind_consec = 0
            else:
                mine_blind_consec = 0

            # While in MINE, keep holding attack unless we've been truly blind
            # for several consecutive frames. This prevents CENTER<->MINE
            # flapping that interrupts block breaking.
            if state == "MINE" and (
                in_mine_commit
                or mine_blind_consec < max(1, int(args.mine_exit_blind_frames))
            ):
                # Important: stand still while mining. Walking around once
                # the log is centered causes needless jitter and can break
                # line-of-sight to the same block. Hold only LMB.
                held.set_w(False)
                held.set_lmb(True)
                held.set_space(False)
                _sleep_remaining(tick_start, tick_dt)
                continue

            # MINE: trunk is huge in the middle of the screen.
            if fovea_frac >= args.mine_fovea_frac and abs(err_x) < args.center_tol * 1.5:
                transition("MINE", fovea_frac, full_frac)
                held.set_w(False)
                held.set_lmb(True)
                held.set_space(False)
                _sleep_remaining(tick_start, tick_dt)
                continue

            # CENTER: log visible but off-center horizontally.
            if abs(err_x) > args.center_tol:
                transition("CENTER", fovea_frac, full_frac)
                held.set_w(False)
                held.set_lmb(False)
                held.set_space(False)
                step = int(np.clip(err_x * args.center_yaw_gain, -100, 100))
                if step == 0:
                    step = 1 if err_x > 0 else -1
                ginput.move_rel(step, 0)
                # Mild pitch correction toward the centroid. Trees are tall;
                # if cy < 0.35 we're aiming above the trunk top -> pitch down.
                if cy < 0.30:
                    ginput.move_rel(0, 8)
                elif cy > 0.75:
                    ginput.move_rel(0, -8)
                _sleep_remaining(tick_start, tick_dt)
                continue

            # APPROACH: log centered but still small. Walk forward.
            transition("APPROACH", fovea_frac, full_frac)
            held.set_w(True)
            held.set_lmb(False)

            # Auto-jump if we've been approaching but the log isn't getting
            # bigger -> probably stuck on a fence / sapling / 1-block step.
            stuck = (
                time.perf_counter() - state_started_at > 1.5
                and (fovea_frac - last_fovea_at_state_start) < args.approach_jump_frac
            )
            held.set_space(stuck)

            _sleep_remaining(tick_start, tick_dt)
    finally:
        held.release_all()

    if logs_collected < chop_target:
        env.close()
        print(f"\nFAIL: only chopped {logs_collected}/{chop_target} log(s).")
        return 1

    print(f"\nCHOPPED {logs_collected} log(s) successfully.")

    if args.craft == "none":
        env.close()
        return 0

    # Hand off to the scripted crafting pipeline. We deliberately re-import
    # at this point so users without the scripted/ submodule installed can
    # still run the chop-only path without an import error at module load.
    from mine_diamonds.scripted.craft_table import (
        craft_planks_then_table,
        get_wooden_pickaxe,
    )
    from mine_diamonds.scripted.inventory_layout import (
        default_layout_for_minecraft,
        default_layout_for_monitor,
    )

    print(f"\nCRAFTING phase: target = {args.craft}")
    layout = default_layout_for_minecraft(gui_scale=int(args.gui_scale))
    if layout is None:
        import mss as _mss
        with _mss.mss() as sct:
            mon = sct.monitors[args.monitor]
        layout = default_layout_for_monitor(
            int(mon["width"]), int(mon["height"]),
            gui_scale=int(args.gui_scale),
        )

    time.sleep(0.6)  # give MC a beat to settle after the last LMB swing
    try:
        if args.craft == "table":
            craft_planks_then_table(
                layout, open_inv_first=True, place_after=True,
                hotbar_slot=1,
            )
        else:
            get_wooden_pickaxe(
                layout, open_inv_first=True, table_hotbar_slot=1,
            )
    finally:
        env.close()

    print(f"\nPASS: chopped {logs_collected} log(s) and crafted '{args.craft}'.")
    return 0


def _sleep_remaining(tick_start: float, tick_dt: float) -> None:
    remaining = tick_dt - (time.perf_counter() - tick_start)
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
