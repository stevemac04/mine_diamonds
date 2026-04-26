"""Interactive smoke + calibration for ``MinecraftRealEnv``.

Run this BEFORE the first training session. It verifies, end-to-end, that:

  1. The MC window is detected (so we capture only the MC client area,
     not the whole monitor — required for windowed mode).
  2. The hotbar slot 1 ROI lands on the actual slot (visualized as a red
     box on the saved capture).
  3. Synthetic input is reaching MC (you'll see the player walk + look
     during a small input burst).
  4. The reward signal fires: when you manually put a log in slot 1, the
     env reports ``log_acquired=True``.

Outputs:
  * eval/smoke_minecraft/capture_initial.png      — MC client area grab
    with hotbar ROI and fovea overlaid.
  * eval/smoke_minecraft/capture_after_input.png  — same, after sending a
    test input burst.
  * eval/smoke_minecraft/hotbar_slot.png          — close-up of just the
    slot region (debug aid: this is what the env actually reads).
  * eval/smoke_minecraft/calibration.json         — capture source + hotbar
    + fovea + smoke summary.

USAGE (PowerShell):
  .\\.venv\\Scripts\\Activate.ps1
  $env:PYTHONUTF8=1
  python scripts\\smoke_minecraft_real.py --countdown 5

Mid-test you'll be prompted (in this terminal!) to focus Minecraft. Click
the MC window so it has keyboard focus, then come back to read the prompt.

If the auto-detected hotbar ROI is wrong (red box not on slot 1), pass
``--manual-hotbar X Y W H`` (in MC-client-area pixels — open the saved
capture and read coordinates from your image viewer) to override.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mine_diamonds import failsafe as _failsafe  # noqa: E402
from mine_diamonds.capture import list_candidate_windows  # noqa: E402
from mine_diamonds.envs.minecraft_real import (  # noqa: E402
    FORWARD,
    FORWARD_ATTACK,
    FORWARD_JUMP,
    NOOP,
    PITCH_DOWN,
    PITCH_UP,
    YAW_LEFT,
    YAW_RIGHT,
    HotbarSpec,
    MinecraftRealConfig,
    MinecraftRealEnv,
)


def annotate(frame_bgr: np.ndarray, env: MinecraftRealEnv) -> np.ndarray:
    out = frame_bgr.copy()
    hb = env._hotbar  # type: ignore[attr-defined]
    fov = env._fovea  # type: ignore[attr-defined]
    if hb is not None:
        cv2.rectangle(out, (hb.x, hb.y), (hb.x + hb.w, hb.y + hb.h), (0, 0, 255), 2)
        cv2.putText(
            out,
            "hotbar slot 1",
            (hb.x, hb.y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if fov is not None:
        x, y, w, h = fov
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            out,
            "fovea (aim shaping)",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def countdown(seconds: int, msg: str) -> None:
    for i in range(seconds, 0, -1):
        print(f"  {msg} ({i}s)", flush=True)
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Seconds to wait after prompts so you can focus Minecraft.",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Fallback mss monitor index when MC window detection fails.",
    )
    parser.add_argument(
        "--gui-scale",
        type=int,
        default=2,
        help="Minecraft GUI Scale setting (Video Settings -> GUI Scale).",
    )
    parser.add_argument(
        "--no-window-detect",
        action="store_true",
        help="Disable MC window auto-detection; use --monitor instead.",
    )
    parser.add_argument(
        "--manual-hotbar",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help=(
            "Override the hotbar slot 1 ROI in CAPTURED-region pixels "
            "(top-left origin). Use when auto-detect is wrong."
        ),
    )
    parser.add_argument(
        "--reward-window",
        type=int,
        default=30,
        help="Seconds to watch for a manual log-acquired reward signal.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval/smoke_minecraft",
        help="Where to dump the annotated screenshots and calibration json.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manual = args.manual_hotbar
    cfg = MinecraftRealConfig(
        monitor_index=args.monitor,
        max_seconds=999.0,
        gui_scale=int(args.gui_scale),
        use_window_detection=not args.no_window_detect,
        hotbar=HotbarSpec(*manual) if manual else None,
    )

    _failsafe.install()
    print(_failsafe.banner())

    env = MinecraftRealEnv(cfg)
    env._ensure_capture()  # type: ignore[attr-defined]
    if env._monitor is None:
        print("ERROR: failed to read monitor info from mss.", flush=True)
        return 2
    mon_w = int(env._monitor["width"])  # type: ignore[index]
    mon_h = int(env._monitor["height"])  # type: ignore[index]
    print(
        f"\n=== capture source: {env._capture_source} | "  # type: ignore[attr-defined]
        f"size: {mon_w}x{mon_h} | "
        f"GUI scale: {cfg.gui_scale} ===",
        flush=True,
    )

    # Sanity check: the actual MC client titles its window
    # 'Minecraft <version> - <world>'. If our captured title doesn't start
    # with 'Minecraft ', we caught the wrong window (e.g. an IDE/browser
    # tab named 'minecraft'). List candidates and bail.
    cap_src = env._capture_source  # type: ignore[attr-defined]
    candidates = list_candidate_windows(title_substr="minecraft")
    if cap_src.startswith("window:"):
        captured_title = cap_src.split("window:", 1)[1].strip("'\"").split("'")[0]
        if not captured_title.lower().startswith("minecraft "):
            print(
                "  ERROR: detected window does NOT look like the Minecraft\n"
                f"    Java client. Captured title: {captured_title!r}\n"
                "    The MC client titles its window 'Minecraft <version> - ...'\n"
                "    All visible windows whose title contains 'minecraft':\n"
                + "\n".join(f"      - {t!r}" for t in candidates)
                + "\n    Open Minecraft Java Edition (the actual game), make\n"
                "    sure it's visible (not minimized), then re-run.\n"
            )
            return 3
    else:
        print(
            "  WARNING: didn't detect a Minecraft window. Falling back to\n"
            "    full-monitor capture, which is almost certainly wrong if MC\n"
            "    is windowed. Visible windows containing 'minecraft':\n"
            + ("\n".join(f"      - {t!r}" for t in candidates) or "      (none)")
            + "\n    Open Minecraft and re-run, or pass --no-window-detect\n"
            "    intentionally if you're testing without MC running.\n"
        )

    print(f"  hotbar ROI = {env._hotbar}")
    print(f"  fovea ROI  = {env._fovea}\n")

    # ---- 1) initial capture ------------------------------------------------
    print("STEP 1/4: Capture the screen as-is.")
    print("  Make sure Minecraft is open and visible. Don't focus it yet.")
    countdown(args.countdown, "capturing in")
    frame0 = env._grab_full_bgr()  # type: ignore[attr-defined]
    annotated0 = annotate(frame0, env)
    init_path = out_dir / "capture_initial.png"
    cv2.imwrite(str(init_path), annotated0)
    print(f"  wrote {init_path}")
    print(
        "  -> Open that file. The RED box should be over hotbar slot 1.\n"
        "     The YELLOW box is the fovea used for dense aim reward.\n"
        "     If RED is misaligned, rerun with --gui-scale set to whatever\n"
        "     your Minecraft Video Settings -> GUI Scale shows.\n"
    )

    # ---- 2) input smoke ----------------------------------------------------
    agent_step_s = cfg.frame_time_s * cfg.action_repeat
    print("STEP 2/4: Send a small input burst to Minecraft.")
    print("  Click into the MC window so it has KEYBOARD FOCUS, then wait.")
    print(
        f"  action_repeat={cfg.action_repeat}, so each step below holds the\n"
        f"  action for ~{agent_step_s*1000:.0f} ms before deciding again.\n"
        f"  Yaw substeps = {cfg.yaw_step_px} mickeys; one decision = "
        f"{cfg.yaw_step_px * cfg.action_repeat} mickeys.\n"
        "  We'll send: forward -> yaw_left ~180deg -> yaw_right ~360deg ->\n"
        "  pitch_up/down -> jump x4 -> attack(0.5s).\n"
        "  PRO TIP: hit F12 mid-burst to test the failsafe — keys should\n"
        "  release in <50ms and the script will exit early."
    )
    countdown(args.countdown, "sending in")
    env.reset()  # baseline capture
    print("  ... forward (8 decisions ~ 1.6s)")
    for _ in range(8):
        env.step(FORWARD)
    print("  ... yaw left (6 decisions, expect ~half-circle)")
    for _ in range(6):
        env.step(YAW_LEFT)
    print("  ... yaw right (12 decisions, expect a full circle)")
    for _ in range(12):
        env.step(YAW_RIGHT)
    print("  ... pitch up + down (4+4 decisions)")
    for _ in range(4):
        env.step(PITCH_UP)
    for _ in range(4):
        env.step(PITCH_DOWN)
    print("  ... forward+jump (4 decisions, expect 2-4 jumps)")
    for _ in range(4):
        env.step(FORWARD_JUMP)
    print("  ... attack burst (5 decisions ~ 1.0s)")
    for _ in range(5):
        env.step(FORWARD_ATTACK)
    env.step(NOOP)
    frame1 = env._grab_full_bgr()  # type: ignore[attr-defined]
    annotated1 = annotate(frame1, env)
    after_path = out_dir / "capture_after_input.png"
    cv2.imwrite(str(after_path), annotated1)
    print(f"  wrote {after_path}")
    print(
        "  -> Check what you SAW in MC during the burst. You should have seen:\n"
        "       1. player walks forward for ~1.6 s\n"
        "       2. camera spins ~half a circle to the LEFT\n"
        "       3. camera spins a FULL circle to the RIGHT (overshoots)\n"
        "       4. camera tilts up to the sky, then back down\n"
        "       5. player walks + jumps a few times (Space held)\n"
        "       6. player swings for ~1 s\n"
        "     If the camera barely moved, something killed action_repeat —\n"
        "     re-check MC's mouse sensitivity (Options -> Controls). If it\n"
        "     was below ~50%, raise it; the env's default mickeys assume\n"
        "     ~100%. If MC didn't get focus, nothing moves.\n"
    )

    # ---- 3) reward signal observation -------------------------------------
    print("STEP 3/4: Reward signal check.")
    print(
        f"  For the next ~{args.reward_window}s I'll watch the hotbar.\n"
        "  Manually go mine a log so it appears in slot 1 of your hotbar.\n"
        "  When the agent will see 'log acquired', this script will print it.\n"
        "  (You can use creative mode and /give yourself a log if quicker.)"
    )
    print("  Click into MC now, then start mining...")
    countdown(args.countdown, "watching in")

    obs, reset_info = env.reset()
    baseline_px = int(reset_info.get("slot_empty_baseline_px", 0))
    eff_thresh = int(reset_info.get("effective_log_threshold", 0))
    diff_thresh = float(reset_info.get("slot_diff_threshold", 0.0))
    use_diff = bool(reset_info.get("use_slot_diff_detector", True))
    detector = "slot_diff (primary)" if use_diff else "slot_log_px (legacy)"
    print(
        f"  detector            = {detector}\n"
        f"  slot_diff_threshold = {diff_thresh:.1f}  (mean BGR delta vs empty)\n"
        f"  empty-slot baseline = {baseline_px} px  (legacy signal)\n"
        f"  effective threshold = {eff_thresh} px   (legacy signal)"
    )
    deadline = time.time() + args.reward_window
    saw_log = False
    max_slot_px = 0
    max_diff = 0.0
    max_fovea = 0.0
    last_print = 0.0
    while time.time() < deadline:
        obs, reward, term, trunc, info = env.step(NOOP)
        max_slot_px = max(max_slot_px, int(info["slot_log_px"]))
        max_diff = max(max_diff, float(info["slot_diff"]))
        max_fovea = max(max_fovea, float(info["fovea_log_frac"]))
        now = time.time()
        if now - last_print > 1.0:
            print(
                f"    slot_log_px={info['slot_log_px']:>5}  "
                f"slot_diff={info['slot_diff']:6.2f}  "
                f"have_log_now={info['have_log_now']!s:<5}  "
                f"fovea_log_frac={info['fovea_log_frac']:.3f}",
                flush=True,
            )
            last_print = now
        if info["log_acquired"]:
            print("  ** LOG ACQUIRED detected (reward fires) **", flush=True)
            saw_log = True
            break
        if term or trunc:
            break

    # Save a close-up of the slot ROI right now, so the user can see
    # what the env is reading (often the fastest way to spot a misalignment).
    hb = env._hotbar  # type: ignore[attr-defined]
    if hb is not None:
        last_frame = env._grab_full_bgr()  # type: ignore[attr-defined]
        slot_crop = last_frame[hb.y : hb.y + hb.h, hb.x : hb.x + hb.w]
        if slot_crop.size > 0:
            slot_path = out_dir / "hotbar_slot.png"
            scaled = cv2.resize(slot_crop, (slot_crop.shape[1] * 6, slot_crop.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(slot_path), scaled)
            print(f"  wrote {slot_path}  (close-up of the hotbar ROI)")

    print(
        f"\n  summary: saw_log_acquired={saw_log}, "
        f"max_slot_diff={max_diff:.2f} (threshold={diff_thresh:.1f}), "
        f"max_slot_log_px={max_slot_px}, "
        f"max_fovea_log_frac={max_fovea:.3f}"
    )
    if not saw_log:
        print(
            "  WARNING: no log_acquired event fired in the watch window.\n"
            f"    - max_slot_diff reached {max_diff:.2f} (need >= {diff_thresh:.1f})\n"
            "    - If max_slot_diff stayed near 0, slot ROI never changed —\n"
            "      either you didn't put a log in slot 1, or the ROI is\n"
            "      misaligned. Open eval/smoke_minecraft/hotbar_slot.png\n"
            "      and verify it's showing slot 1.\n"
            "    - If max_slot_diff is high (~10+) but log_acquired never\n"
            "      fired, the slot_diff_threshold may need tuning. Lower\n"
            "      cfg.slot_diff_threshold (currently 8.0).\n"
        )

    # ---- 4) calibration json ----------------------------------------------
    print("STEP 4/4: Save calibration.")
    calib = {
        "capture_source": env._capture_source,  # type: ignore[attr-defined]
        "capture_region": {"width": mon_w, "height": mon_h},
        "monitor_index": args.monitor,
        "gui_scale": args.gui_scale,
        "hotbar_slot1": {
            "x": env._hotbar.x,  # type: ignore[union-attr]
            "y": env._hotbar.y,  # type: ignore[union-attr]
            "w": env._hotbar.w,  # type: ignore[union-attr]
            "h": env._hotbar.h,  # type: ignore[union-attr]
        },
        "fovea": list(env._fovea) if env._fovea else None,  # type: ignore[arg-type]
        "smoke": {
            "saw_log_acquired": bool(saw_log),
            "max_slot_log_px": int(max_slot_px),
            "max_slot_diff": float(max_diff),
            "max_fovea_log_frac": float(max_fovea),
            "slot_empty_baseline_px": int(baseline_px),
            "effective_log_threshold": int(eff_thresh),
            "slot_diff_threshold": float(diff_thresh),
            "detector": detector,
        },
    }
    calib_path = out_dir / "calibration.json"
    calib_path.write_text(json.dumps(calib, indent=2))
    print(f"  wrote {calib_path}")
    print("\nSMOKE COMPLETE.")
    print(
        "If all three checks looked right (hotbar ROI on the slot, MC moved on\n"
        "the input burst, log_acquired fired during the reward window), you're\n"
        "ready to train. If not, fix one thing at a time and re-run."
    )

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
