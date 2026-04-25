"""Vision-guided Minecraft interaction (screen capture + BGR mask).

Uses the project's **texture pack BGR table** in ``mine_diamonds.vision.pack_colors``
(black logs, brown planks, grey stone, white iron ore, gold ingots, cyan sticks,
purple diamond ore). Still **not** RL: color blob → aim / mine.

Requirements:
  pip install -e ".[minecraft-demo]"

Run (smart tree chop — align, mine one block, collect, repeat):
  python scripts\\demo_treechop_vision.py --material logs --verbose

Timed evaluation episodes (manual world reset between runs; append JSONL):
  python scripts\\demo_treechop_vision.py --episodes 3 --duration 60 --target-logs 1 --episode-jsonl runs.jsonl --pause-between

Confirm each break was really a log (cocoa/leaves fool the mask):
  python scripts\\demo_treechop_vision.py --manual-log-confirm --target-logs 1

More exploration if SEARCH sticks on empty patch (wide yaw burst; tune --explore-stuck-frames):
  python scripts\\demo_treechop_vision.py --search-yaw 80 --explore-stuck-frames 50

Range uses **fovea blob area** (tune --min-fovea-area-to-mine / --mine-fovea-soft-frac / --max-approach-w-sec):
  python scripts\\demo_treechop_vision.py --min-fovea-area-to-mine 4500 --mine-fovea-soft-frac 0.88 --max-approach-w-sec 0.09

Verify Minecraft sees synthetic mouse (camera should yaw ~3s):
  python scripts\\demo_treechop_vision.py --test-input

Debug vision (white = matched pixels in PNG):
  python scripts\\demo_treechop_vision.py --material logs --save-mask debug_mask.png

Old “hold forward” behavior (not recommended for demo):
  python scripts\\demo_treechop_vision.py --material logs --mode legacy

List built-in BGR bands:
  python scripts\\demo_treechop_vision.py --list-materials

Calibrate (point crosshair at target block; prints suggested ``--bgr-low``/``high``):
  python scripts\\demo_treechop_vision.py --calibrate --material logs

Stop: hold **Esc** (or Ctrl+C in terminal).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
import json
from enum import Enum, auto
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
import keyboard
import mss
import numpy as np

import mine_diamonds.input.game_input as ginput
from mine_diamonds.vision.pack_colors import MATERIAL_KEYS, describe_pack_palette, get_bgr_range


def _stop() -> bool:
    return keyboard.is_pressed("esc")


def _release_all() -> None:
    try:
        ginput.release_all()
    except Exception:
        pass


def _aim_delta(
    ax: int,
    ay: int,
    cx: int,
    cy: int,
    *,
    aim_gain: float,
    max_yaw: int,
    max_pitch: int,
) -> tuple[int, int]:
    """Integer mouse steps; never return (0,0) while aim is meaningfully off-center."""
    rdx = (ax - cx) * aim_gain
    rdy = (ay - cy) * aim_gain
    dx = int(np.clip(round(rdx), -max_yaw, max_yaw))
    dy = int(np.clip(round(rdy), -max_pitch, max_pitch))
    if dx == 0 and abs(ax - cx) > 2:
        dx = int(np.sign(ax - cx)) * min(max_yaw, max(1, int(round(abs(rdx))) or 1))
    if dy == 0 and abs(ay - cy) > 2:
        dy = int(np.sign(ay - cy)) * min(max_pitch, max(1, int(round(abs(rdy))) or 1))
    return dx, dy


@dataclass
class _ApproachClock:
    """Rate-limits short W pulses so we inch toward a trunk without sprinting through it."""

    t: float = 0.0


def forward_approach_pulse(
    *,
    now: float,
    clock: _ApproachClock,
    fovea_area: int,
    min_close_area: int,
    max_w_sec: float,
    ratio_cc: float,
    inch_until_cc_ratio: float,
) -> float | None:
    """Return W hold duration to inch toward the trunk.

    Stops only when **both** the lower-fovea blob is large enough **and** the crosshair CC
    pin is strong (``ratio_cc`` ≥ ``inch_until_cc_ratio``). That way a huge grass/leaves blob
    no longer blocks forward motion while the crosshair is still not on solid log.
    """
    if min_close_area <= 0:
        return None
    close_f = fovea_area >= min_close_area
    inch = max(1e-6, float(inch_until_cc_ratio))
    close_c = ratio_cc >= inch
    if close_f and close_c:
        return None
    p_f = max(0.0, min(1.0, 1.0 - fovea_area / float(max(1, min_close_area))))
    p_c = max(0.0, min(1.0, 1.0 - ratio_cc / inch))
    p = max(p_f, p_c)
    gap = 0.08 + (1.0 - p) * 0.20
    if now - clock.t < gap:
        return None
    clock.t = now
    cap = min(0.16, max(0.06, max_w_sec))
    dur = 0.05 + p * max(0.0, cap - 0.05)
    return float(dur)


def grab_bgr(sct: mss.mss, monitor_index: int) -> np.ndarray:
    mon = sct.monitors[monitor_index]
    raw = np.array(sct.grab(mon))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


@dataclass
class MaskConfig:
    """BGR or HSV bounds for ``cv2.inRange``."""

    use_hsv: bool
    # BGR inRange (if use_hsv False)
    bgr_low: tuple[int, int, int]
    bgr_high: tuple[int, int, int]
    # HSV inRange (if use_hsv True) — OpenCV H 0-179, S/V 0-255
    hsv_low: tuple[int, int, int]
    hsv_high: tuple[int, int, int]


@dataclass
class TreeChopEpisodeResult:
    """Summary after a smart FSM run (for JSONL logging / simple evaluation loops)."""

    logs_mined: int
    target_logs: int
    goal_met: bool
    wall_seconds: float
    stopped_esc: bool
    aborted_before_start: bool
    mine_timeouts: int
    fake_wood_guard_triggers: int


def material_mask(frame_bgr: np.ndarray, cfg: MaskConfig) -> np.ndarray:
    if cfg.use_hsv:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        low = np.array(cfg.hsv_low, dtype=np.uint8)
        high = np.array(cfg.hsv_high, dtype=np.uint8)
        return cv2.inRange(hsv, low, high)
    low = np.array(cfg.bgr_low, dtype=np.uint8)
    high = np.array(cfg.bgr_high, dtype=np.uint8)
    return cv2.inRange(frame_bgr, low, high)


def refine_mask(mask: np.ndarray, morph: int) -> np.ndarray:
    if morph <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)


def largest_blob_centroid(mask: np.ndarray, min_pixels: int) -> tuple[int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        a = int(cv2.contourArea(c))
        if a < min_pixels:
            continue
        if a > best_area:
            best_area = a
            m = cv2.moments(c)
            if m["m00"] <= 1e-6:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            best = (cx, cy)
    return best


def largest_contour_area_any(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max((int(cv2.contourArea(c)) for c in contours), default=0)


def aim_with_fallback(mask_world: np.ndarray, min_blob_px: int) -> tuple[int, int] | None:
    """Prefer strict min area; fall back so small distant trunks still steer SEARCH."""
    a = largest_blob_centroid(mask_world, min_blob_px)
    if a is not None:
        return a
    lo = max(20, min(80, min_blob_px // 2))
    return largest_blob_centroid(mask_world, lo)


def component_at_crosshair(
    mask: np.ndarray,
    cx: int,
    cy: int,
    *,
    min_area: int,
    max_area: int,
) -> np.ndarray | None:
    """Keep only the 8-connected region under the crosshair (one block face), not the whole world."""
    if cy < 0 or cx < 0 or cy >= mask.shape[0] or cx >= mask.shape[1]:
        return None
    if int(mask[cy, cx]) == 0:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lid = int(labels[cy, cx])
    if lid <= 0:
        return None
    area = int(stats[lid, cv2.CC_STAT_AREA])
    if area < min_area or area > max_area:
        return None
    return ((labels == lid).astype(np.uint8) * 255)


def aim_fovea_lower_center(mask_world: np.ndarray, min_blob_px: int) -> tuple[int, int] | None:
    """Largest log blob in lower-center view (trunk), not centroid of the whole mask."""
    h, w = mask_world.shape[:2]
    x0, x1 = int(w * 0.26), int(w * 0.74)
    y0 = int(h * 0.30)
    roi = mask_world[y0:h, x0:x1]
    a = aim_with_fallback(roi, min_blob_px)
    if a is None:
        return None
    return (a[0] + x0, a[1] + y0)


def largest_blob_area_in_rect(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> int:
    if y1 <= y0 or x1 <= x0:
        return 0
    roi = mask[y0:y1, x0:x1]
    return largest_contour_area_any(roi)


def aim_fovea_steer(
    mask_world: np.ndarray,
    min_blob_px: int,
    *,
    material: str,
    steer_max_fovea_blob: int,
) -> tuple[int, int] | None:
    """Where to look while **searching** — fovea on full mask.

    If ``steer_max_fovea_blob`` > 0 and the largest blob in that fovea exceeds it, return
    None (optional guard against one giant ground plane). **0 disables** this check.
    """
    if material != "logs":
        return aim_fovea_lower_center(mask_world, min_blob_px)
    if steer_max_fovea_blob <= 0:
        return aim_fovea_lower_center(mask_world, min_blob_px)
    h, w = mask_world.shape[:2]
    x0, x1 = int(w * 0.26), int(w * 0.74)
    y0 = int(h * 0.30)
    if largest_blob_area_in_rect(mask_world, x0, x1, y0, h) > steer_max_fovea_blob:
        return None
    return aim_fovea_lower_center(mask_world, min_blob_px)


def mask_binary_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] < 12.0:
        return None
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


def center_patch_ratio(mask: np.ndarray, frac: float) -> float:
    h, w = mask.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    patch = mask[y0 : y0 + ch, x0 : x0 + cw]
    return float(np.count_nonzero(patch)) / float(patch.size + 1e-9)


def run_calibrate(monitor: int, seconds: float, material: str) -> None:
    print(f"Point crosshair at **{material}** (texture color). Sampling center of screen…")
    t_end = time.perf_counter() + seconds
    samples: list[np.ndarray] = []
    with mss.mss() as sct:
        while time.perf_counter() < t_end:
            frame = grab_bgr(sct, monitor)
            h, w = frame.shape[:2]
            cy, cx = h // 2, w // 2
            patch = frame[cy - 5 : cy + 5, cx - 5 : cx + 5]
            samples.append(patch.reshape(-1, 3))
            time.sleep(0.1)
    if not samples:
        return
    all_px = np.vstack(samples).astype(np.float32)
    mean = all_px.mean(axis=0)
    std = all_px.std(axis=0)
    print("\nBGR under crosshair (mean ± std):")
    print(f"  B={mean[0]:.1f}±{std[0]:.1f}  G={mean[1]:.1f}±{std[1]:.1f}  R={mean[2]:.1f}±{std[2]:.1f}")
    print("\nSuggested BGR inRange (loose) - paste/adjust:")
    lo = np.clip(mean - 2.5 * std - 10, 0, 255).astype(int)
    hi = np.clip(mean + 2.5 * std + 10, 0, 255).astype(int)
    print(f"  --material {material} --bgr-low {int(lo[0])},{int(lo[1])},{int(lo[2])} --bgr-high {int(hi[0])},{int(hi[1])},{int(hi[2])}")


class _Phase(Enum):
    SEARCH = auto()
    ALIGN = auto()
    MINE = auto()
    COLLECT = auto()


def _perceive(
    sct: mss.mss,
    monitor: int,
    cfg: MaskConfig,
    morph: int,
    min_blob_px: int,
    center_frac: float,
    sky_frac: float,
    crosshair_frac: float,
    material: str,
    cc_max_area: int,
    cc_min_area: int,
    steer_max_fovea_blob: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int] | None,
    tuple[int, int] | None,
    float,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    frame = grab_bgr(sct, monitor)
    mask = refine_mask(material_mask(frame, cfg), morph)
    h, w = mask.shape[:2]
    y_cut = int(h * sky_frac)
    mask_world = mask.copy()
    mask_world[:y_cut, :] = 0
    cx, cy = w // 2, h // 2
    n_px_full = int(np.count_nonzero(mask_world))
    fx0, fx1 = int(w * 0.26), int(w * 0.74)
    fy0 = int(h * 0.30)
    fovea_max_area = largest_blob_area_in_rect(mask_world, fx0, fx1, fy0, h)

    # Crosshair mask: logs = 8-connected component under cursor (mine / pin signal).
    mask_cc = mask_world
    if material == "logs":
        comp = component_at_crosshair(
            mask_world, cx, cy, min_area=cc_min_area, max_area=cc_max_area
        )
        if comp is not None:
            mask_cc = comp
        else:
            mask_cc = np.zeros_like(mask_world)

    n_px_cc = int(np.count_nonzero(mask_cc))
    aim_cc = (
        mask_binary_centroid(mask_cc)
        if material == "logs" and n_px_cc >= 48
        else None
    )

    aim_steer = aim_fovea_steer(
        mask_world,
        min_blob_px,
        material=material,
        steer_max_fovea_blob=steer_max_fovea_blob,
    )

    ratio_pin_cc = center_patch_ratio(mask_cc, crosshair_frac)
    # Same crosshair patch on **full** mask — true "is center of screen black / log-colored?"
    ratio_pin_world = center_patch_ratio(mask_world, crosshair_frac)
    ratio_wide = center_patch_ratio(mask_world, center_frac)
    max_a_cc = largest_contour_area_any(mask_cc)
    return (
        frame,
        mask_world,
        aim_steer,
        aim_cc,
        ratio_pin_cc,
        ratio_pin_world,
        ratio_wide,
        cx,
        cy,
        n_px_cc,
        max_a_cc,
        n_px_full,
        fovea_max_area,
    )


def run_fsm_tree_chopper(
    *,
    monitor: int,
    countdown: int,
    duration_sec: float,
    cfg: MaskConfig,
    material_label: str,
    morph: int,
    min_blob_px: int,
    center_frac: float,
    mine_threshold: float,
    aim_gain: float,
    max_yaw_per_tick: int,
    max_pitch_per_tick: int,
    target_logs: int,
    align_px: int,
    collect_sec: float,
    lost_frames: int,
    break_ratio: float,
    search_yaw_step: int,
    align_timeout_sec: float,
    mine_timeout_sec: float,
    sky_frac: float,
    crosshair_frac: float,
    cc_max_area: int,
    cc_min_area: int,
    steer_max_fovea_blob: int,
    search_flip_sec: float,
    enter_align_px: float,
    verbose: bool,
    save_mask_path: str | None,
    input_backend: str,
    mine_pulse_forward: bool,
    manual_log_confirm: bool,
    align_pulse_forward: bool,
    explore_override_sec: float,
    explore_stuck_frames: int,
    min_fovea_area_to_mine: int,
    mine_fovea_soft_frac: float,
    max_approach_w_sec: float,
    inch_until_cc_ratio: float,
    search_pitch_up: int,
) -> TreeChopEpisodeResult:
    """Search -> align crosshair on pack-colored log -> mine until mask drops -> collect -> repeat."""
    impl = ginput.configure(input_backend)
    print(
        f"\n=== Smart tree chop (material={material_label}) ===\n"
        f"Input backend: **{impl}** — on Windows, **sendinput** uses relative mouse (what Java MC reads).\n"
        "Stand near trees (1-3 blocks). **Fullscreen Minecraft** on the monitor you capture.\n"
        "Phases: SEARCH -> ALIGN -> MINE (attack; distance-scaled W when trunk blob still small) -> COLLECT -> next.\n"
        f"Goal: {target_logs} log(s). Esc to stop.\n"
        "\nIf it feels like 'nothing happens':\n"
        "  - Click Minecraft during/after countdown so **mouse + keys go to the game**.\n"
        "  - Run **--test-input** once: the camera should yaw; if not, try **Run as administrator**.\n"
        "  - Run with **--verbose**; if mask_px stays ~0, your BGR band does not match the pack — use **--calibrate**.\n"
        "  - Try **--monitor 2** if capture is the wrong screen (desktop / black).\n"
    )
    if manual_log_confirm:
        print(
            "**Manual log confirm** is ON: after each assumed break you must answer y/n "
            "(cocoa / leaves can fool the mask).\n"
        )
    if not mine_pulse_forward:
        print(
            "**Mine:** no extra **--mine-pulse-forward** rhythm taps; short **approach** W still applies "
            "when the fovea trunk blob shrinks (e.g. after partial breaks).\n"
        )
    if not align_pulse_forward:
        print(
            "**Align: no extra W from legacy align-pulse** (mouse + distance-aware approach below). "
            "Use **--align-pulse-forward** for optional nudges.\n"
        )
    print(
        "**Approach:** W while **either** fovea trunk blob is small **or** crosshair CC pin "
        f"is below **{inch_until_cc_ratio:.3f}** (--inch-until-cc-ratio). "
        f"MINE when blob ≥ {int(min_fovea_area_to_mine * mine_fovea_soft_frac)} px **and** "
        f"CC ≥ that ratio (or blob huge as fallback).\n"
    )
    for i in range(countdown, 0, -1):
        print(f"Starting in {i}…")
        time.sleep(1)
        if _stop():
            print("Aborted.")
            return TreeChopEpisodeResult(
                logs_mined=0,
                target_logs=target_logs,
                goal_met=False,
                wall_seconds=0.0,
                stopped_esc=True,
                aborted_before_start=True,
                mine_timeouts=0,
                fake_wood_guard_triggers=0,
            )

    t_wall0 = time.perf_counter()
    t_end = t_wall0 + duration_sec
    phase = _Phase.SEARCH
    logs_done = 0
    search_dir = 1
    align_t0 = 0.0
    mine_t0 = 0.0
    lost = 0
    saw_wood_while_mining = False
    post_search_turn = 0
    last_hb = 0.0
    search_flip_t0 = time.perf_counter()
    last_align_w = 0.0
    mine_timeouts = 0
    fake_wood_aborts = 0
    no_cc_streak = 0
    episode_out: TreeChopEpisodeResult | None = None
    search_low_wood_frames = 0
    explore_override_until = 0.0
    approach_clock = _ApproachClock()

    with mss.mss() as sct:
        try:
            frame0, mw0, steer0, cc0, r0_cc, r0_world, r0_wide, _, _, ncc0, maxa0, n0full, fov0 = _perceive(
                sct,
                monitor,
                cfg,
                morph,
                min_blob_px,
                center_frac,
                sky_frac,
                crosshair_frac,
                material_label,
                cc_max_area,
                cc_min_area,
                steer_max_fovea_blob,
            )
            hh, ww = frame0.shape[:2]
            mb = frame0.mean(axis=(0, 1))
            wood0 = max(r0_cc, r0_world)
            print(
                f"Capture monitor={monitor} size={ww}x{hh}  mean_BGR=({mb[0]:.0f},{mb[1]:.0f},{mb[2]:.0f})\n"
                f"  full_mask_px={n0full:,}  cc_mask_px={ncc0:,}  max_cc_area={maxa0}\n"
                f"  pin_cc={r0_cc:.3f}  pin_world={r0_world:.3f}  wood=max={wood0:.3f}  wide={r0_wide:.3f}\n"
                f"  aim_steer={steer0!r}  aim_cc={cc0!r}  fovea_max_blob_area={fov0}\n"
                f"  (inch until pin_cc ≥ {inch_until_cc_ratio:.3f} **and** fovea blob big; "
                f"mine blob ≥ {int(min_fovea_area_to_mine * mine_fovea_soft_frac)} px + that CC)\n"
            )
            if n0full > hh * ww * 0.14:
                print(
                    "  WARNING: >14% of screen matches — BGR band still too wide or wrong monitor.\n"
                    "    Tighten with: python scripts\\demo_treechop_vision.py --calibrate --material logs\n"
                )
            if wood0 < mine_threshold * 0.25 and steer0 is None:
                print(
                    "  ^ Low **wood** at crosshair and no fovea aim — tighten BGR / calibrate, or stand closer to a tree.\n"
                    f'    python scripts\\demo_treechop_vision.py --calibrate --material {material_label}\n'
                )
            if save_mask_path:
                cv2.imwrite(save_mask_path, mw0)
                print(f"  Wrote debug mask to {save_mask_path}")

            while time.perf_counter() < t_end and logs_done < target_logs:
                if _stop():
                    print("Stopped (Esc).")
                    episode_out = TreeChopEpisodeResult(
                        logs_mined=logs_done,
                        target_logs=target_logs,
                        goal_met=logs_done >= target_logs,
                        wall_seconds=time.perf_counter() - t_wall0,
                        stopped_esc=True,
                        aborted_before_start=False,
                        mine_timeouts=mine_timeouts,
                        fake_wood_guard_triggers=fake_wood_aborts,
                    )
                    break

                (
                    _frame,
                    mask_world,
                    aim_steer,
                    aim_cc,
                    ratio_cc,
                    ratio_world,
                    ratio_wide,
                    cx,
                    cy,
                    n_px_cc,
                    max_a_cc,
                    n_full,
                    fovea_max_area,
                ) = _perceive(
                    sct,
                    monitor,
                    cfg,
                    morph,
                    min_blob_px,
                    center_frac,
                    sky_frac,
                    crosshair_frac,
                    material_label,
                    cc_max_area,
                    cc_min_area,
                    steer_max_fovea_blob,
                )
                now = time.perf_counter()
                wood = max(ratio_cc, ratio_world)
                aim_target = aim_cc if aim_cc is not None else aim_steer
                soft_fovea_need = int(min_fovea_area_to_mine * mine_fovea_soft_frac)

                if verbose and now - last_hb > 1.2:
                    print(
                        f"[hb] {phase.name} wood={wood:.3f} cc={ratio_cc:.3f} world={ratio_world:.3f} "
                        f"wide={ratio_wide:.3f} cc_px={n_px_cc} full_px={n_full} fovea_a={fovea_max_area} "
                        f"steer={aim_steer!r} step={search_dir * search_yaw_step}"
                    )
                    last_hb = now
                aligned = False
                if aim_target is not None:
                    ax, ay = aim_target
                    aligned = (
                        abs(ax - cx) <= align_px
                        and abs(ay - cy) <= align_px
                        and wood >= mine_threshold * 0.48
                    )

                prev = phase

                if phase is _Phase.SEARCH:
                    ginput.mouse_left(False)
                    ginput.release_move_keys()
                    if post_search_turn != 0:
                        ginput.move_rel(int(np.sign(post_search_turn) * min(abs(post_search_turn), 400)), 0)
                        post_search_turn = 0
                    near_steer = (
                        aim_steer is not None
                        and math.hypot(aim_steer[0] - cx, aim_steer[1] - cy) < enter_align_px
                    )
                    enter_align = (
                        wood >= mine_threshold * 0.28
                        or (aim_cc is not None and ratio_cc >= 0.022)
                        or (near_steer and ratio_wide >= mine_threshold * 0.32)
                    )
                    if enter_align:
                        phase = _Phase.ALIGN
                        align_t0 = time.perf_counter()
                        search_low_wood_frames = 0
                        explore_override_until = 0.0
                        approach_clock = _ApproachClock()
                    elif now < explore_override_until:
                        step_big = int(max(abs(search_yaw_step), 16) * search_dir * 2.35)
                        ginput.move_rel(
                            step_big,
                            int(7 * math.sin(now * 2.85)) + search_pitch_up,
                        )
                        if wood >= mine_threshold * 0.14:
                            w_d = forward_approach_pulse(
                                now=now,
                                clock=approach_clock,
                                fovea_area=fovea_max_area,
                                min_close_area=min_fovea_area_to_mine,
                                max_w_sec=max_approach_w_sec,
                                ratio_cc=ratio_cc,
                                inch_until_cc_ratio=inch_until_cc_ratio,
                            )
                            if w_d is not None:
                                ginput.key_down("w")
                                time.sleep(w_d)
                                ginput.key_up("w")
                        if now - search_flip_t0 >= search_flip_sec * 0.55:
                            search_dir *= -1
                            search_flip_t0 = now
                    elif aim_steer is not None:
                        sx, sy = aim_steer
                        dx, dy = _aim_delta(
                            sx,
                            sy,
                            cx,
                            cy,
                            aim_gain=min(aim_gain * 0.62, 0.44),
                            max_yaw=max_yaw_per_tick,
                            max_pitch=max_pitch_per_tick,
                        )
                        if sy > cy + 95:
                            dy = int(np.clip(dy - 11, -max_pitch_per_tick, max_pitch_per_tick))
                        ginput.move_rel(dx, dy + search_pitch_up)
                        facing_ok = (
                            wood >= mine_threshold * 0.14
                            or math.hypot(sx - cx, sy - cy) < 440
                        )
                        if facing_ok:
                            w_d = forward_approach_pulse(
                                now=now,
                                clock=approach_clock,
                                fovea_area=fovea_max_area,
                                min_close_area=min_fovea_area_to_mine,
                                max_w_sec=max_approach_w_sec,
                                ratio_cc=ratio_cc,
                                inch_until_cc_ratio=inch_until_cc_ratio,
                            )
                            if w_d is not None:
                                ginput.key_down("w")
                                time.sleep(w_d)
                                ginput.key_up("w")
                    else:
                        if now - search_flip_t0 >= search_flip_sec:
                            search_dir *= -1
                            search_flip_t0 = now
                        step = max(abs(search_yaw_step), 14) * search_dir
                        ginput.move_rel(step, search_pitch_up)
                        if wood >= mine_threshold * 0.16 and aim_steer is None:
                            w_d = forward_approach_pulse(
                                now=now,
                                clock=approach_clock,
                                fovea_area=fovea_max_area,
                                min_close_area=min_fovea_area_to_mine,
                                max_w_sec=max_approach_w_sec,
                                ratio_cc=ratio_cc,
                                inch_until_cc_ratio=inch_until_cc_ratio,
                            )
                            if w_d is not None:
                                ginput.key_down("w")
                                time.sleep(w_d)
                                ginput.key_up("w")

                    if phase is _Phase.SEARCH:
                        if now < explore_override_until:
                            search_low_wood_frames = 0
                        elif wood < mine_threshold * 0.22:
                            search_low_wood_frames += 1
                        else:
                            search_low_wood_frames = 0
                        if search_low_wood_frames >= explore_stuck_frames:
                            explore_override_until = now + explore_override_sec
                            search_low_wood_frames = 0
                            search_dir *= -1
                            search_flip_t0 = now
                            print(
                                "  [explore] low wood here for a while — wide yaw + pitch for ~"
                                f"{explore_override_sec:.1f}s"
                            )

                elif phase is _Phase.ALIGN:
                    ginput.mouse_left(False)
                    ginput.release_move_keys()
                    if (aim_cc is None and aim_steer is None) or (
                        wood < mine_threshold * 0.06 and aim_steer is None
                    ):
                        phase = _Phase.SEARCH
                        search_flip_t0 = now
                        search_low_wood_frames = 0
                    elif aligned and wood >= mine_threshold * 0.62:
                        hh, ww = mask_world.shape[:2]
                        giant = n_full > int(hh * ww * 0.36)
                        if ratio_cc < 0.02 and ratio_world > 0.55 and giant:
                            print(
                                "  [guard] skip mine: center matches mask but no CC + huge fill (likely dirt/stone)."
                            )
                            fake_wood_aborts += 1
                            phase = _Phase.SEARCH
                            search_flip_t0 = now
                            search_low_wood_frames = 0
                        else:
                            blob_ready = fovea_max_area >= soft_fovea_need
                            blob_force = fovea_max_area >= int(min_fovea_area_to_mine * 1.06)
                            cc_super = ratio_cc >= inch_until_cc_ratio
                            if blob_force or (blob_ready and cc_super):
                                phase = _Phase.MINE
                                mine_t0 = time.perf_counter()
                                lost = 0
                                saw_wood_while_mining = False
                                no_cc_streak = 0
                                approach_clock = _ApproachClock()
                            else:
                                w_d = forward_approach_pulse(
                                    now=now,
                                    clock=approach_clock,
                                    fovea_area=fovea_max_area,
                                    min_close_area=min_fovea_area_to_mine,
                                    max_w_sec=max_approach_w_sec,
                                    ratio_cc=ratio_cc,
                                    inch_until_cc_ratio=inch_until_cc_ratio,
                                )
                                if w_d is not None:
                                    ginput.key_down("w")
                                    time.sleep(w_d)
                                    ginput.key_up("w")
                                    if verbose:
                                        print(
                                            f"  [range] inch cc={ratio_cc:.3f} (want ≥{inch_until_cc_ratio:.3f}) "
                                            f"fovea={fovea_max_area} (mine blob≥{soft_fovea_need})"
                                        )
                    elif time.perf_counter() - align_t0 > align_timeout_sec:
                        phase = _Phase.SEARCH
                        search_dir *= -1
                        search_flip_t0 = now
                        search_low_wood_frames = 0
                    else:
                        if aim_target is not None:
                            ax, ay = aim_target
                            dx, dy = _aim_delta(
                                ax,
                                ay,
                                cx,
                                cy,
                                aim_gain=aim_gain,
                                max_yaw=max_yaw_per_tick,
                                max_pitch=max_pitch_per_tick,
                            )
                            if ay > cy + 85:
                                dy = int(np.clip(dy - 9, -max_pitch_per_tick, max_pitch_per_tick))
                            if dx != 0 or dy != 0:
                                ginput.move_rel(dx, dy)
                            if wood >= mine_threshold * 0.14:
                                w_d = forward_approach_pulse(
                                    now=now,
                                    clock=approach_clock,
                                    fovea_area=fovea_max_area,
                                    min_close_area=min_fovea_area_to_mine,
                                    max_w_sec=max_approach_w_sec,
                                    ratio_cc=ratio_cc,
                                    inch_until_cc_ratio=inch_until_cc_ratio,
                                )
                                if w_d is not None:
                                    ginput.key_down("w")
                                    time.sleep(w_d)
                                    ginput.key_up("w")
                            if align_pulse_forward:
                                if (
                                    wood < mine_threshold * 0.75
                                    and abs(ax - cx) < 115
                                    and abs(ay - cy) < align_px + 55
                                ):
                                    if now - last_align_w > 0.22:
                                        ginput.key_down("w")
                                        time.sleep(0.055)
                                        ginput.key_up("w")
                                        last_align_w = now

                elif phase is _Phase.MINE:
                    hh, ww = mask_world.shape[:2]
                    suspicious = (
                        ratio_cc < 0.015
                        and ratio_world > 0.48
                        and n_full > int(hh * ww * 0.32)
                    )
                    if suspicious:
                        no_cc_streak += 1
                    else:
                        no_cc_streak = 0

                    if no_cc_streak >= 28:
                        print("  [guard] bail mine: no crosshair CC while world pin high + huge mask (dirt dig).")
                        fake_wood_aborts += 1
                        no_cc_streak = 0
                        ginput.mouse_left(False)
                        ginput.release_move_keys()
                        phase = _Phase.SEARCH
                        search_dir *= -1
                        search_flip_t0 = now
                    else:
                        w_approach = forward_approach_pulse(
                            now=now,
                            clock=approach_clock,
                            fovea_area=fovea_max_area,
                            min_close_area=soft_fovea_need,
                            max_w_sec=max_approach_w_sec,
                            ratio_cc=ratio_cc,
                            inch_until_cc_ratio=inch_until_cc_ratio,
                        )
                        if wood >= mine_threshold * 0.42:
                            saw_wood_while_mining = True
                        if wood < break_ratio:
                            lost += 1
                        else:
                            lost = 0

                        # Hold attack continuously while mining. Do **not** release LMB for W
                        # pulses — that reads as repeated punches instead of breaking the block.
                        ginput.mouse_left(True)
                        if w_approach is not None:
                            ginput.key_down("w")
                            time.sleep(w_approach)
                            ginput.key_up("w")
                        elif mine_pulse_forward and wood >= mine_threshold * 0.32:
                            ginput.key_down("w")
                            time.sleep(0.09)
                            ginput.key_up("w")
                        else:
                            ginput.release_move_keys()

                        if saw_wood_while_mining and lost >= lost_frames:
                            ginput.mouse_left(False)
                            ginput.release_move_keys()
                            if manual_log_confirm:
                                print(
                                    "\n  [confirm] Vision says the block broke (cocoa, vines, leaves, "
                                    "or dirt can look the same).\n"
                                    "  Did you actually break a **log** block here?  [y]es count / "
                                    "[n]o skip re-search\n"
                                )
                                try:
                                    ans = input("  > ").strip().lower()
                                except EOFError:
                                    ans = "n"
                                if ans.startswith("y"):
                                    logs_done += 1
                                    print(f"  [mined] counted log {logs_done}/{target_logs}")
                                    phase = _Phase.COLLECT
                                else:
                                    print("  [mined] not counted — continuing search")
                                    phase = _Phase.SEARCH
                                    search_flip_t0 = time.perf_counter()
                            else:
                                logs_done += 1
                                print(
                                    f"  [mined] log {logs_done}/{target_logs} "
                                    "(mask dropped; use --manual-log-confirm if this is often wrong)"
                                )
                                phase = _Phase.COLLECT
                        elif time.perf_counter() - mine_t0 > mine_timeout_sec:
                            ginput.mouse_left(False)
                            ginput.release_move_keys()
                            print("  [mine] timeout; re-searching")
                            mine_timeouts += 1
                            phase = _Phase.SEARCH
                            search_dir *= -1
                            search_flip_t0 = now

                elif phase is _Phase.COLLECT:
                    ginput.mouse_left(False)
                    ginput.release_move_keys()
                    t0 = time.perf_counter()
                    collect_stop = False
                    while time.perf_counter() - t0 < collect_sec:
                        if _stop():
                            collect_stop = True
                            break
                        ginput.move_rel(0, 6)
                        ginput.key_down("a")
                        time.sleep(0.07)
                        ginput.key_up("a")
                        ginput.key_down("d")
                        time.sleep(0.07)
                        ginput.key_up("d")
                        time.sleep(0.04)
                    if collect_stop:
                        episode_out = TreeChopEpisodeResult(
                            logs_mined=logs_done,
                            target_logs=target_logs,
                            goal_met=logs_done >= target_logs,
                            wall_seconds=time.perf_counter() - t_wall0,
                            stopped_esc=True,
                            aborted_before_start=False,
                            mine_timeouts=mine_timeouts,
                            fake_wood_guard_triggers=fake_wood_aborts,
                        )
                        break
                    post_search_turn = 350 * search_dir
                    search_dir *= -1
                    search_flip_t0 = time.perf_counter()
                    phase = _Phase.SEARCH
                    print("  [collect] done; searching for next tree")

                if phase != prev:
                    print(f"  -> {phase.name}")

                time.sleep(0.025)

            if episode_out is None:
                wall = time.perf_counter() - t_wall0
                episode_out = TreeChopEpisodeResult(
                    logs_mined=logs_done,
                    target_logs=target_logs,
                    goal_met=logs_done >= target_logs,
                    wall_seconds=wall,
                    stopped_esc=False,
                    aborted_before_start=False,
                    mine_timeouts=mine_timeouts,
                    fake_wood_guard_triggers=fake_wood_aborts,
                )
                if logs_done >= target_logs:
                    print(f"\nFinished goal: {logs_done} log(s). Nice.")
                else:
                    print(f"\nTime limit. Mined {logs_done}/{target_logs} log(s).")
            print(
                f"\n[episode] goal_met={episode_out.goal_met} logs={episode_out.logs_mined}/"
                f"{episode_out.target_logs} wall_s={episode_out.wall_seconds:.1f} "
                f"mine_timeouts={episode_out.mine_timeouts} fake_guard={episode_out.fake_wood_guard_triggers}"
            )
        finally:
            _release_all()

    return episode_out


def run_legacy_vision_demo(
    *,
    monitor: int,
    countdown: int,
    duration_sec: float,
    cfg: MaskConfig,
    material_label: str,
    morph: int,
    min_blob_px: int,
    center_frac: float,
    mine_threshold: float,
    aim_gain: float,
    max_yaw_per_tick: int,
    max_pitch_per_tick: int,
    sky_frac: float,
    crosshair_frac: float,
    cc_max_area: int,
    cc_min_area: int,
    steer_max_fovea_blob: int,
    input_backend: str,
) -> None:
    ginput.configure(input_backend)
    print(
        f"\n=== LEGACY vision demo (material={material_label}) — hold W when mining ===\n"
        f"Input backend: **{ginput.current_impl()}**\n"
        "Prefer:  --mode smart\n"
    )
    for i in range(countdown, 0, -1):
        print(f"Starting in {i}…")
        time.sleep(1)
        if _stop():
            print("Aborted.")
            return

    t_end = time.perf_counter() + duration_sec
    mining = False
    with mss.mss() as sct:
        try:
            while time.perf_counter() < t_end:
                if _stop():
                    print("Stopped (Esc).")
                    return

                _, mask_world, aim_steer, aim_cc, ratio_cc, ratio_world, _, cx, cy, _, _, _, _ = _perceive(
                    sct,
                    monitor,
                    cfg,
                    morph,
                    min_blob_px,
                    center_frac,
                    sky_frac,
                    crosshair_frac,
                    material_label,
                    cc_max_area,
                    cc_min_area,
                    steer_max_fovea_blob,
                )
                aim = aim_cc if aim_cc is not None else aim_steer
                on_target = max(ratio_cc, ratio_world) >= mine_threshold

                if on_target:
                    if not mining:
                        ginput.mouse_left(True)
                        ginput.key_down("w")
                        mining = True
                    time.sleep(0.09)
                else:
                    if mining:
                        ginput.mouse_left(False)
                        ginput.key_up("w")
                        mining = False
                    if aim is not None:
                        ax, ay = aim
                        dx, dy = _aim_delta(
                            ax,
                            ay,
                            cx,
                            cy,
                            aim_gain=aim_gain,
                            max_yaw=max_yaw_per_tick,
                            max_pitch=max_pitch_per_tick,
                        )
                        if dx != 0 or dy != 0:
                            ginput.move_rel(dx, dy)
                    else:
                        ginput.move_rel(max_yaw_per_tick, 0)

                time.sleep(0.02)
        finally:
            _release_all()

    print("Done.")


def run_test_input(*, countdown: int) -> None:
    """Relative yaw only — proves Java MC is receiving synthetic mouse."""
    print(
        "\n=== --test-input ===\n"
        "Click **fullscreen Minecraft** before the timer ends. The view should yaw for ~3s.\n"
        "If nothing moves: try **PowerShell Run as administrator**, or ``--input-backend pyautogui``.\n"
    )
    for i in range(countdown, 0, -1):
        print(f"Starting in {i}…")
        time.sleep(1)
        if _stop():
            print("Aborted.")
            return
    print(f"Backend: **{ginput.current_impl()}**\n")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.2:
        ginput.move_rel(44, 0)
        time.sleep(0.04)
        if _stop():
            break
    ginput.release_all()
    print("Test finished.")


def parse_tri(s: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma ints, got: {s!r}")
    return parts[0], parts[1], parts[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Vision-guided Minecraft material detection + chop.")
    p.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary).")
    p.add_argument("--countdown", type=int, default=5)
    p.add_argument("--duration", type=float, default=120.0, help="Total seconds to run.")
    p.add_argument("--calibrate", action="store_true", help="Sample BGR under crosshair then exit.")
    p.add_argument("--calibrate-seconds", type=float, default=4.0)
    p.add_argument(
        "--list-materials",
        action="store_true",
        help="Print built-in BGR bands from pack_colors.py and exit.",
    )
    p.add_argument("--use-hsv", action="store_true", help="Use HSV inRange instead of BGR.")
    p.add_argument(
        "--material",
        choices=MATERIAL_KEYS,
        default="logs",
        help="Which pack color to track (defaults to logs for tree chop).",
    )

    p.add_argument(
        "--bgr-low",
        type=str,
        default=None,
        help="Override BGR lower (B,G,R). If set, --bgr-high must also be set.",
    )
    p.add_argument(
        "--bgr-high",
        type=str,
        default=None,
        help="Override BGR upper. If omitted, defaults from --material are used.",
    )
    p.add_argument("--hsv-low", type=str, default="0,0,0")
    p.add_argument("--hsv-high", type=str, default="179,80,90")

    p.add_argument(
        "--morph",
        type=int,
        default=2,
        help="Morphology kernel size (0 disables). Smaller = thinner flat textures survive.",
    )
    p.add_argument(
        "--min-blob",
        type=int,
        default=120,
        help="Min contour area for strict blob; loose fallback is used if none qualify.",
    )
    p.add_argument("--center-frac", type=float, default=0.12, help="Center patch size as fraction of min(H,W).")
    p.add_argument("--mine-threshold", type=float, default=0.12, help="Fraction of center patch on wood to mine.")
    p.add_argument("--aim-gain", type=float, default=0.55, help="Aim correction gain toward blob centroid.")
    p.add_argument("--max-yaw", type=int, default=18)
    p.add_argument("--max-pitch", type=int, default=10)

    p.add_argument(
        "--mode",
        choices=("smart", "legacy"),
        default="smart",
        help="smart = align/mine/collect FSM; legacy = old hold-W when mining.",
    )
    p.add_argument("--target-logs", type=int, default=5, help="(smart) Stop after this many successful breaks.")
    p.add_argument(
        "--align-px",
        type=int,
        default=56,
        help="(smart) Max |blob-center - crosshair| pixels to count aligned.",
    )
    p.add_argument("--collect-sec", type=float, default=1.35, help="(smart) Wiggle duration after a break.")
    p.add_argument("--lost-frames", type=int, default=18, help="(smart) Frames below break-ratio to count as block gone.")
    p.add_argument("--break-ratio", type=float, default=0.035, help="(smart) Center patch ratio below this counts as 'no wood'.")
    p.add_argument("--search-yaw", type=int, default=64, help="(smart) Relative yaw per tick while searching.")
    p.add_argument("--align-timeout", type=float, default=5.5)
    p.add_argument("--mine-timeout", type=float, default=16.0)
    p.add_argument("--sky-frac", type=float, default=0.22, help="Top fraction of frame masked out (sky).")
    p.add_argument(
        "--crosshair-frac",
        type=float,
        default=0.065,
        help="(smart) Tiny center patch for pin_ratio (crosshair on log).",
    )
    p.add_argument(
        "--cc-max-area",
        type=int,
        default=22000,
        help="(smart, logs) Reject connected region under crosshair if larger (ground leak).",
    )
    p.add_argument(
        "--cc-min-area",
        type=int,
        default=40,
        help="(smart, logs) Ignore tiny noise blobs under crosshair.",
    )
    p.add_argument(
        "--steer-max-fovea-blob",
        type=int,
        default=0,
        help="(smart, logs) 0=off. If >0, skip aim_steer when lower-fovea largest blob exceeds this (mega-ground).",
    )
    p.add_argument(
        "--search-flip-sec",
        type=float,
        default=1.2,
        help="(smart) When not using aim_steer, reverse sweep direction this often (avoids endless one-way pan).",
    )
    p.add_argument(
        "--enter-align-px",
        type=float,
        default=148.0,
        help="(smart) If crosshair is this close to aim_steer and wide ratio is OK, enter ALIGN.",
    )
    p.add_argument("--verbose", action="store_true", help="(smart) Print heartbeat stats every ~1.2s.")
    p.add_argument(
        "--save-mask",
        type=str,
        default=None,
        metavar="PATH",
        help="(smart) After countdown, save one B&W mask PNG (debug vision).",
    )
    p.add_argument(
        "--input-backend",
        choices=("auto", "sendinput", "pyautogui"),
        default="auto",
        help="Mouse/keys: auto uses SendInput relative motion on Windows (recommended for Java MC).",
    )
    p.add_argument(
        "--test-input",
        action="store_true",
        help="After countdown, send relative yaw only (~3s) to verify the game receives synthetic mouse.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="(smart) Run this many timed runs in a row (manual world reset between runs if --pause-between).",
    )
    p.add_argument(
        "--episode-jsonl",
        type=str,
        default=None,
        metavar="PATH",
        help="(smart) Append one JSON line per episode with counts, timeouts, goal_met, wall_seconds.",
    )
    p.add_argument(
        "--pause-between",
        action="store_true",
        help="(smart) When --episodes>1, wait for Enter between episodes so you can reset the world.",
    )
    p.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit 1 if the last smart episode did not reach --target-logs (for simple batch eval).",
    )
    p.add_argument(
        "--mine-pulse-forward",
        action="store_true",
        help="(smart) In MINE, tap W while attacking. Default is **stand still** (LMB only) so you do not walk past the trunk.",
    )
    p.add_argument(
        "--manual-log-confirm",
        action="store_true",
        help="(smart) After each assumed block break, prompt y/n before counting toward --target-logs (avoids cocoa/leaves false positives).",
    )
    p.add_argument(
        "--align-pulse-forward",
        action="store_true",
        help="(smart) Allow short W taps while ALIGNing (default: off — was walking through trunks).",
    )
    p.add_argument(
        "--explore-override-sec",
        type=float,
        default=1.45,
        help="(smart) After low-wood stuck in SEARCH, this many seconds of wide yaw + pitch wiggle.",
    )
    p.add_argument(
        "--explore-stuck-frames",
        type=int,
        default=65,
        help="(smart) SEARCH frames (wood < threshold) before triggering explore override (~frames*25ms).",
    )
    p.add_argument(
        "--min-fovea-area-to-mine",
        type=int,
        default=3200,
        help="(smart) Reference trunk-blob area (lower fovea, px); mining starts at --mine-fovea-soft-frac of this.",
    )
    p.add_argument(
        "--mine-fovea-soft-frac",
        type=float,
        default=0.90,
        metavar="FRAC",
        help="(smart) Enter MINE / stop inching when fovea blob area ≥ FRAC × --min-fovea-area-to-mine (default 0.90).",
    )
    p.add_argument(
        "--max-approach-w-sec",
        type=float,
        default=0.12,
        metavar="SEC",
        help="(smart) Upper bound on each forward W pulse when inching (smaller = gentler; typical 0.08–0.16).",
    )
    p.add_argument(
        "--inch-until-cc-ratio",
        type=float,
        default=0.088,
        metavar="R",
        help="(smart) Keep inching forward until crosshair CC pin ratio_cc ≥ R (higher = walk closer before mining).",
    )
    p.add_argument(
        "--search-pitch-up",
        type=int,
        default=-4,
        help="(smart) Added to vertical mouse each SEARCH tick (negative = look up, away from grass).",
    )

    args = p.parse_args(argv)

    if args.episodes < 1:
        print("--episodes must be >= 1", file=sys.stderr)
        return 2
    if args.episodes > 1 and args.mode != "smart":
        print("--episodes > 1 requires --mode smart", file=sys.stderr)
        return 2
    if args.episode_jsonl and args.mode != "smart":
        print("--episode-jsonl requires --mode smart", file=sys.stderr)
        return 2
    if not (0.5 <= args.mine_fovea_soft_frac <= 1.0):
        print("--mine-fovea-soft-frac must be between 0.5 and 1.0", file=sys.stderr)
        return 2
    if not (0.02 <= args.max_approach_w_sec <= 0.35):
        print("--max-approach-w-sec must be between 0.02 and 0.35", file=sys.stderr)
        return 2
    if not (0.035 <= args.inch_until_cc_ratio <= 0.22):
        print("--inch-until-cc-ratio must be between 0.035 and 0.22", file=sys.stderr)
        return 2

    if args.list_materials:
        print(describe_pack_palette())
        return 0

    if args.test_input:
        ginput.configure(args.input_backend)
        run_test_input(countdown=args.countdown)
        return 0

    if args.calibrate:
        run_calibrate(args.monitor, args.calibrate_seconds, args.material)
        return 0

    if args.use_hsv:
        blo = parse_tri(args.hsv_low)
        bhi = parse_tri(args.hsv_high)
        cfg = MaskConfig(
            use_hsv=True,
            bgr_low=(0, 0, 0),
            bgr_high=(0, 0, 0),
            hsv_low=blo,
            hsv_high=bhi,
        )
    else:
        if (args.bgr_low is None) ^ (args.bgr_high is None):
            print("Provide both --bgr-low and --bgr-high, or neither to use --material defaults.", file=sys.stderr)
            return 2
        if args.bgr_low is not None:
            blo, bhi = parse_tri(args.bgr_low), parse_tri(args.bgr_high)
        else:
            blo, bhi = get_bgr_range(args.material)
        cfg = MaskConfig(
            use_hsv=False,
            bgr_low=blo,
            bgr_high=bhi,
            hsv_low=parse_tri(args.hsv_low),
            hsv_high=parse_tri(args.hsv_high),
        )

    last_res: TreeChopEpisodeResult | None = None
    try:
        if args.mode == "smart":
            for ep in range(args.episodes):
                if ep > 0 and args.pause_between:
                    print(
                        "\n--- Next episode: reset Minecraft (new world / respawn / move to trees), "
                        "then press Enter in this terminal ---\n"
                    )
                    try:
                        input()
                    except EOFError:
                        print("EOF: stopping episode loop.")
                        break
                last_res = run_fsm_tree_chopper(
                    monitor=args.monitor,
                    countdown=args.countdown,
                    duration_sec=args.duration,
                    cfg=cfg,
                    material_label=args.material,
                    morph=args.morph,
                    min_blob_px=args.min_blob,
                    center_frac=args.center_frac,
                    mine_threshold=args.mine_threshold,
                    aim_gain=args.aim_gain,
                    max_yaw_per_tick=args.max_yaw,
                    max_pitch_per_tick=args.max_pitch,
                    target_logs=args.target_logs,
                    align_px=args.align_px,
                    collect_sec=args.collect_sec,
                    lost_frames=args.lost_frames,
                    break_ratio=args.break_ratio,
                    search_yaw_step=args.search_yaw,
                    align_timeout_sec=args.align_timeout,
                    mine_timeout_sec=args.mine_timeout,
                    sky_frac=args.sky_frac,
                    crosshair_frac=args.crosshair_frac,
                    cc_max_area=args.cc_max_area,
                    cc_min_area=args.cc_min_area,
                    steer_max_fovea_blob=args.steer_max_fovea_blob,
                    search_flip_sec=args.search_flip_sec,
                    enter_align_px=args.enter_align_px,
                    verbose=args.verbose,
                    save_mask_path=args.save_mask if ep == 0 else None,
                    input_backend=args.input_backend,
                    mine_pulse_forward=args.mine_pulse_forward,
                    manual_log_confirm=args.manual_log_confirm,
                    align_pulse_forward=args.align_pulse_forward,
                    explore_override_sec=args.explore_override_sec,
                    explore_stuck_frames=args.explore_stuck_frames,
                    min_fovea_area_to_mine=args.min_fovea_area_to_mine,
                    mine_fovea_soft_frac=args.mine_fovea_soft_frac,
                    max_approach_w_sec=args.max_approach_w_sec,
                    inch_until_cc_ratio=args.inch_until_cc_ratio,
                    search_pitch_up=args.search_pitch_up,
                )
                if args.episode_jsonl and last_res is not None:
                    row = asdict(last_res)
                    row["episode_index"] = ep
                    row["material"] = args.material
                    row["duration_requested_sec"] = args.duration
                    row["manual_log_confirm"] = args.manual_log_confirm
                    row["mine_pulse_forward"] = args.mine_pulse_forward
                    row["align_pulse_forward"] = args.align_pulse_forward
                    with open(args.episode_jsonl, "a", encoding="utf-8") as fj:
                        fj.write(json.dumps(row) + "\n")
        else:
            run_legacy_vision_demo(
                monitor=args.monitor,
                countdown=args.countdown,
                duration_sec=args.duration,
                cfg=cfg,
                material_label=args.material,
                morph=args.morph,
                min_blob_px=args.min_blob,
                center_frac=args.center_frac,
                mine_threshold=args.mine_threshold,
                aim_gain=args.aim_gain,
                max_yaw_per_tick=args.max_yaw,
                max_pitch_per_tick=args.max_pitch,
                sky_frac=args.sky_frac,
                crosshair_frac=args.crosshair_frac,
                cc_max_area=args.cc_max_area,
                cc_min_area=args.cc_min_area,
                steer_max_fovea_blob=args.steer_max_fovea_blob,
                input_backend=args.input_backend,
            )
    except KeyboardInterrupt:
        _release_all()
        return 130
    if args.strict_exit and last_res is not None and not last_res.goal_met:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
