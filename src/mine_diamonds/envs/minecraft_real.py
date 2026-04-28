"""Gymnasium env that wraps the running Minecraft Java client.

This is real-game RL: observations are downscaled screenshots of the MC
window captured with ``mss``; actions are synthetic keyboard/mouse events
delivered with the project's ``mine_diamonds.input.game_input`` layer
(Windows ``SendInput`` for raw relative mouse deltas, which Minecraft
needs); reward is computed from the same screenshot by checking whether
log-colored pixels have appeared in the player's hotbar slot 1 (vision
detection via the project's "simple-colors" texture pack BGR ranges).

Task: **mine one log**. Episode ends when the agent picks up at least
``logs_to_succeed`` log-colored pixel cluster in the hotbar (success), or
``max_seconds`` of wall-clock time elapses (truncation).

Caveats:
  * Single env only. Real Minecraft can't be parallelized.
  * Throughput is bounded by ``frame_time_s`` (default 0.05 s = 20 Hz),
    so ~20 transitions per second per env.
  * The MC window must keep keyboard focus for the entire run; alt-tabbing
    away will break inputs.
  * Configure Minecraft to a known window size before running. The
    hotbar position and dimensions are computed from the captured monitor
    region; you can override them with explicit pixel rects via
    ``HotbarSpec``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import gymnasium as gym
import mss
import numpy as np
from gymnasium import spaces

from mine_diamonds.capture import WindowRegion, find_minecraft_window
from mine_diamonds.input import game_input as ginput
from mine_diamonds.vision.pack_colors import get_bgr_range
from mine_diamonds import failsafe as _failsafe


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

# Discrete(8); kept small for sample efficiency. The agent can only run
# WASD-forward + LMB simultaneously by selecting `FORWARD_ATTACK`; this is
# the dominant action for tree-mining once aim is correct. ``FORWARD_JUMP``
# holds W + Space so the agent can hop up onto blocks while exploring (MC
# auto-jumps when Space is held and the player tries to walk into a 1-block
# rise).
NOOP = 0
FORWARD = 1
YAW_LEFT = 2
YAW_RIGHT = 3
PITCH_UP = 4
PITCH_DOWN = 5
FORWARD_ATTACK = 6
FORWARD_JUMP = 7

ACTION_NAMES: tuple[str, ...] = (
    "noop",
    "forward",
    "yaw_left",
    "yaw_right",
    "pitch_up",
    "pitch_down",
    "forward_attack",
    "forward_jump",
)
NUM_ACTIONS = len(ACTION_NAMES)


@dataclass
class HotbarSpec:
    """Pixel rect inside the captured monitor for the hotbar slot to watch.

    Coordinates are relative to the captured monitor's top-left in pixels.
    Use ``HotbarSpec.from_monitor`` for sane defaults at common resolutions
    with default GUI scale.
    """

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_monitor(
        cls, mon_w: int, mon_h: int, *, slot: int = 1, gui_scale: int = 2
    ) -> "HotbarSpec":
        """Compute slot ROI from the captured region's pixel dimensions.

        Vanilla MC hotbar is 182 GUI px wide x 22 px tall, centered at
        the bottom of the rendered area. At GUI scale 2 (the default on
        1080p) the rendered hotbar is 364x44 actual pixels. Slot 1 is
        the leftmost. Slots are 20 GUI px wide. The icon inside each
        slot is inset ~3 GUI px on each side. We further pad in by
        ``pad`` actual pixels to be tolerant of texture-pack icon
        positioning.

        ``mon_w`` and ``mon_h`` should be the dimensions of whatever
        we're actually capturing. If you've narrowed capture to the MC
        window's client area (recommended), pass those — not the full
        monitor.
        """
        gs = int(gui_scale)
        slot_gui_w = 20
        slot_w = slot_gui_w * gs
        slot_h = 20 * gs
        hotbar_gui_w = 182
        hotbar_w = hotbar_gui_w * gs
        cx = mon_w // 2
        hot_left = cx - hotbar_w // 2
        hotbar_h = 22 * gs
        hot_bottom = mon_h - 3 * gs
        hot_top = hot_bottom - hotbar_h
        slot_left = hot_left + (1 + slot_gui_w * (slot - 1)) * gs
        slot_top = hot_top + 3 * gs
        pad = 4
        return cls(
            x=slot_left + pad,
            y=slot_top + pad,
            w=slot_w - 2 * pad,
            h=slot_h - 2 * pad,
        )


@dataclass
class FoveaSpec:
    """Center-of-screen patch for dense reward shaping.

    Defaults to a centered square covering the middle 1/3 of width/height
    of the captured monitor.
    """

    fraction: float = 1.0 / 3.0

    def rect(self, mon_w: int, mon_h: int) -> tuple[int, int, int, int]:
        fw = int(mon_w * self.fraction)
        fh = int(mon_h * self.fraction)
        x = (mon_w - fw) // 2
        y = (mon_h - fh) // 2
        return x, y, fw, fh


@dataclass
class MinecraftRealConfig:
    """Top-level env configuration.

    The defaults assume:
      * Minecraft Java client running, visible, with the simple-colors
        texture pack loaded. The env auto-detects the MC window and
        captures only its client area (so windowed mode works fine).
      * GUI scale 2.
      * Survival, peaceful, on roughly flat ground near a tree.
    """

    # Capture: prefer to detect the MC window automatically. If the env
    # can't find a window matching ``window_title_substr``, it falls back
    # to monitor capture at ``monitor_index`` and warns.
    use_window_detection: bool = True
    window_title_substr: str = "minecraft"
    monitor_index: int = 1            # mss monitor index used for fallback only
    obs_shape: tuple[int, int] = (84, 84)  # downscaled (H, W) for CnnPolicy
    frame_time_s: float = 0.05        # 20 Hz target *game* tick rate
    # Sticky-actions / frame skip: each call to env.step() applies the chosen
    # action ``action_repeat`` times, sleeping ``frame_time_s`` between each
    # repeat, before reading the next observation. This is the standard
    # Atari-style trick. Without it, a single yaw step is only ~50 ms long
    # and PPO oscillates between yaw and other actions before any visible
    # rotation accumulates. With ``action_repeat=4`` each yaw decision plays
    # out for ~200 ms, giving the camera time to actually swing.
    action_repeat: int = 4
    max_seconds: float = 60.0         # episode timeout (wall-clock)
    yaw_step_px: int = 60             # mouse dx per yaw_left/right *substep*
    pitch_step_px: int = 35            # mouse dy per pitch_up/down *substep*
    # Superflat: ignore pitch actions (no vertical look).
    disable_pitch_actions: bool = True
    # Hold W+LMB this many wall seconds per env.step() when choosing FORWARD_ATTACK.
    mine_hold_duration_s: float = 3.6
    logs_to_succeed: int = 1          # episode ends after this many logs in hotbar slot 1
    gui_scale: int = 2                # MC GUI scale; controls hotbar pixel math
    log_pixel_threshold: int = 80     # absolute floor for "log present" in slot
    log_pixel_baseline_margin: int = 200  # auto-baseline: adding a log must produce
    # at least this many extra log-colored pixels over the empty-slot baseline.
    auto_baseline_log_threshold: bool = True
    # Primary detection signal: snapshot the empty slot at reset, then trigger
    # when the slot ROI looks "different enough" from baseline. This is the
    # robust signal — pixel-count thresholds get fooled by simple-colors
    # rendering the empty-slot background dark enough to look like a log.
    slot_diff_threshold: float = 8.0  # mean BGR delta vs. empty-slot baseline
    use_slot_diff_detector: bool = True
    # Acquisition confirmation gate for RL training. Prevents one-frame
    # slot-diff spikes from counting as "got wood".
    log_confirm_steps: int = 3
    slot_diff_margin: float = 2.0
    # ---- exploration shaping -------------------------------------------------
    # Rewards applied per step ONLY when the fovea sees no logs (i.e., the
    # agent is "blind" and needs to look around). Kept small — they're just
    # tiebreakers above the step penalty so spinning beats standing still.
    # The "wood-seeking" gradient comes from ``reward_log_visible_max`` and
    # ``reward_log_approach_max`` below, which are MUCH larger than these.
    # When blind, prefer *moving* to new terrain over spinning in one spot.
    reward_explore_forward_when_blind: float = 0.07
    reward_explore_yaw_when_blind: float = 0.012
    # Small bonus for sprinting blind exploration (W + look around elsewhere).
    reward_sprint_blind: float = 0.04
    # Penalty for many consecutive yaw-only steps while still blind+treeless
    # (reduces in-place circles in empty areas).
    blind_yaw_streak_threshold: int = 8
    penalty_blind_yaw_streak: float = 0.028
    # Below this fovea_log_frac the agent is considered "blind" for shaping.
    explore_blind_threshold: float = 0.02
    # "Treeless" for streak penalty: no meaningful wood in world frame.
    explore_treeless_log_frac: float = 0.012
    # Heuristic: bright upper band and almost no log pixels => staring at sky.
    penalty_sky_stare: float = 0.06
    sky_bright_min_gray: float = 105.0
    sky_bright_saturate: float = 100.0  # (gray_mean - min) / saturate -> 0..1
    # EMA of pitch-up (+1) vs pitch-down (-1) per env step. When |EMA| is
    # large the camera is drifting into sky/ground; we apply a small blind
    # penalty so PPO learns to balance look-up with look-down.
    pitch_imbalance_ema_alpha: float = 0.2
    penalty_pitch_imbalance: float = 0.045
    # Fovea reward scales linearly until this fraction of the fovea is log
    # pixels, then caps (stronger signal when actually staring at wood).
    fovea_aim_reference_frac: float = 0.10
    # Frame-difference (curiosity) shaping. Reward visual change vs. the
    # frame ``frame_diff_lookback_steps`` ago, scaled by ``reward_frame_diff_max``.
    # Also kept small — it's only here to slightly favor moving the camera
    # over staring at a wall.
    reward_frame_diff_max: float = 0.02
    # Lookback is in agent steps (post-action-repeat). With action_repeat=4
    # and frame_time_s=0.05, 5 agent steps = ~1 second of game time.
    frame_diff_lookback_steps: int = 5
    frame_diff_saturation: float = 25.0  # mean BGR delta that maxes out the reward
    # ---- wood-seeking shaping (the dominant gradient) -----------------------
    # Per-step reward proportional to how many log-colored pixels are visible
    # ANYWHERE on screen (not just the fovea). This is the "I can see wood"
    # signal — it fires the moment a tree edges into peripheral view, so the
    # agent gets a clear gradient as it rotates toward visible wood.
    # Saturates at ``log_visible_saturation_frac`` of the screen being log.
    reward_log_visible_max: float = 0.40
    log_visible_saturation_frac: float = 0.10  # ~10% of screen filled = max
    # Per-step reward proportional to the INCREASE in visible-log fraction
    # vs. the previous step. Encodes "you are approaching / centering on
    # wood" — walking toward a tree (or yawing to bring it closer to center)
    # makes it bigger on screen, which fires this. Negative deltas (wood
    # leaving view) get zero, not a penalty, to avoid a "stand still" trap.
    # Saturation tuned so a typical "walking step" toward a tree (~0.005
    # delta per agent step) earns ~half of the cap, giving a smooth pull.
    reward_log_approach_max: float = 1.35
    log_approach_saturation_delta: float = 0.01  # +1% of screen per step = max
    # ---- left / right peripheral (yaw toward the tree) --------------------
    peripheral_log_saturation_frac: float = 0.055
    reward_peripheral_wood_max: float = 0.42
    peripheral_yaw_imb_threshold: float = 0.12
    reward_yaw_toward_peripheral_max: float = 0.62
    penalty_yaw_away_peripheral_max: float = 0.52
    # ---- walk toward visible wood ------------------------------------------
    forward_close_min_full_log_frac: float = 0.005
    forward_close_full_ref_frac: float = 0.065
    reward_forward_close_max: float = 1.15
    # ---- teacher-assisted bootstrap -----------------------------------------
    # Hard-coded tree-seeking controller that can override PPO actions early
    # in training (teacher forcing), then gradually hand control back.
    # This is a practical curriculum for sparse-reward real-MC training:
    # pure random PPO often never reaches a tree often enough to learn.
    teacher_force_start: float = 0.0   # 0 disables teacher forcing entirely
    teacher_force_end: float = 0.0
    teacher_force_decay_steps: int = 10_000
    teacher_center_tol: float = 0.08
    teacher_blind_pitch_every: int = 8
    teacher_pitch_recover_px: int = 16
    teacher_scan_flip_every: int = 14
    # Blind exploration: 1 of every N steps yaws; the rest are FORWARD
    # so the agent leaves treeless ground instead of idling in circles.
    teacher_blind_run_yaw_cycle: int = 5
    # When |pitch EMA| exceeds this in teacher blind mode, nudge pitch down
    # (sky) or up (ground) instead of only recovering from ground.
    teacher_pitch_balance_ema: float = 0.6
    # Slightly more aggressive mine threshold so teacher commits to W+LMB sooner.
    teacher_mine_fovea_frac: float = 0.05
    # ---- auto-reset (in-game) ------------------------------------------------
    # After every episode the env can run an in-game chat command via
    # T -> Ctrl+V -> Enter. Defaults match the "find a tree, mine it, repeat"
    # loop the user wants for unattended training.
    # Chat commands that run ONCE at the very first reset of a training
    # session. Use this for one-shot world setup like enabling immediate
    # respawn or setting the agent's spawn point.
    init_chat_commands: tuple[str, ...] = ()

    # Chat commands that run AFTER the death screen is dismissed on a
    # timeout reset. Use for things /kill wipes that need re-applying once
    # the player is back in the world (effect buffs, gamemode, etc.).
    # Only fires on timeouts — keeps normal "got a log" terminate-resets
    # cheap.
    post_respawn_chat_commands: tuple[str, ...] = ()

    auto_reset_chat_commands: tuple[str, ...] = ("/clear @s",)
    # Optional /tp @s ~dx ~ ~dz after reset (0 = off; large values risk chunk/sync).
    reset_random_spread_blocks: int = 0
    # When the agent times out without acquiring a log, run *these* commands
    # in addition to ``auto_reset_chat_commands``. Use this for a "harder"
    # reset (e.g. teleport back to spawn) when the agent has wandered into
    # an uninteresting part of the world.
    timeout_extra_chat_commands: tuple[str, ...] = ()
    # Wall-clock pause AFTER chat commands run, so the player can finish
    # respawning / re-entering control before the new episode starts.
    post_reset_pause_s: float = 0.5
    # Watchdog: at every reset(), re-detect the MC window. If it moved, the
    # capture region is updated. If MC is gone (alt-tabbed away, minimized,
    # crashed) the env busy-waits with a loud warning until either MC
    # reappears or the failsafe trips. Without this, a 4h run can silently
    # train on a frozen desktop frame.
    require_window_each_reset: bool = True
    window_watchdog_poll_s: float = 1.0
    fovea_log_pixel_norm: int = 4000  # divisor for fovea-pixel shaping (capped at 1.0)
    reward_step_penalty: float = -0.005
    reward_aim_dense: float = 1.35
    fovea_hold_ema_alpha: float = 0.22
    reward_fovea_hold_dense_max: float = 0.65
    reward_attack_when_aimed: float = 2.0
    reward_forward_when_aimed: float = 0.5
    penalty_yaw_when_aimed: float = -0.55
    fovea_soft_yaw_penalty_frac: float = 0.022
    penalty_yaw_when_fovea_soft: float = -0.14
    penalty_aimed_forward_jump: float = -0.2
    penalty_move_when_aimed_not_attacking: float = -0.18  # pitch only if enabled
    close_mine_full_log_frac: float = 0.10
    penalty_move_while_close_mine: float = -2.0
    reward_mine_commit_bonus: float = 0.35
    reward_log_acquired: float = 25.0
    aimed_threshold_frac: float = 0.05
    hotbar: HotbarSpec | None = None  # if set, overrides auto computation
    fovea: FoveaSpec = field(default_factory=FoveaSpec)


class MinecraftRealEnv(gym.Env):
    """Gymnasium env over the real Minecraft Java client (Windows-only).

    Observations: ``uint8`` ``(H, W, 3)`` RGB at ``cfg.obs_shape``.
    Actions: ``Discrete(8)`` — see ``ACTION_NAMES``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: MinecraftRealConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or MinecraftRealConfig()
        self.render_mode = render_mode

        h, w = self.cfg.obs_shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        log_low, log_high = get_bgr_range("logs")
        self._log_low = np.array(log_low, dtype=np.uint8)
        self._log_high = np.array(log_high, dtype=np.uint8)

        # Lazily configure the input backend (Windows SendInput).
        self._input_impl = ginput.configure("auto")

        # Held state: True means "currently held down" so we only emit
        # transitions when the desired held-state changes.
        self._held: dict[str, bool] = {
            "w": False,
            "lmb": False,
            "space": False,
        }

        self._sct: mss.mss | None = None
        self._monitor: dict | None = None
        self._mc_window: WindowRegion | None = None
        self._capture_source: str = "unknown"
        self._hotbar: HotbarSpec | None = None
        self._fovea: tuple[int, int, int, int] | None = None

        self._episode_start: float | None = None
        self._steps: int = 0
        self._logs_collected: int = 0
        self._had_log_last_step: bool = False
        self._slot_empty_baseline: int = 0
        self._effective_log_threshold: int = int(self.cfg.log_pixel_threshold)
        self._slot_empty_frame: np.ndarray | None = None
        self._have_log_streak: int = 0
        self._reset_due_to_timeout: bool = False
        # First-reset flag: triggers the one-shot ``init_chat_commands``.
        self._init_done: bool = False
        self._last_obs: np.ndarray | None = None
        # Full-resolution BGR frame from the most recent step(). Exposed
        # so callbacks can dump high-quality screenshots on log_acquired
        # without paying to grab a fresh frame.
        self._last_full_frame: np.ndarray | None = None
        # Ring buffer of small downsampled frames for the frame-diff
        # curiosity reward. Stored at obs resolution so the diff is cheap.
        self._obs_history: list[np.ndarray] = []
        # Tracks fraction of full-frame pixels matching the log BGR range,
        # used by the "approaching wood" (delta) reward. Reset to the
        # initial-frame fraction in reset() so the very first step's delta
        # isn't a spurious +reward.
        self._prev_full_frame_log_frac: float = 0.0
        self._global_steps: int = 0
        self._teacher_prev_fovea: float = 0.0
        self._teacher_stuck_steps: int = 0
        self._teacher_blind_steps: int = 0
        self._teacher_scan_dir: int = 1  # +1 = right, -1 = left
        # Running imbalance of pitch-up vs pitch-down (rough proxy for sky/ground drift).
        self._pitch_imbalance_ema: float = 0.0
        self._blind_yaw_streak: int = 0
        self._fovea_hold_ema: float = 0.0

    # ---- core helpers -----------------------------------------------------

    def _ensure_capture(self) -> None:
        cfg = self.cfg
        if self._sct is None:
            self._sct = mss.mss()
        if self._monitor is None:
            if cfg.use_window_detection:
                region = find_minecraft_window(
                    title_substr=cfg.window_title_substr
                )
                if region is not None:
                    self._mc_window = region
                    self._monitor = region.as_mss_monitor()
                    self._capture_source = (
                        f"window:{region.title!r}@{region.left},{region.top}"
                    )
            if self._monitor is None:
                self._monitor = self._sct.monitors[cfg.monitor_index]
                self._capture_source = f"monitor:{cfg.monitor_index}"
            mw = int(self._monitor["width"])
            mh = int(self._monitor["height"])
            if cfg.hotbar is not None:
                self._hotbar = cfg.hotbar
            else:
                self._hotbar = HotbarSpec.from_monitor(
                    mw, mh, slot=1, gui_scale=cfg.gui_scale
                )
            self._fovea = cfg.fovea.rect(mw, mh)

    def _refresh_window_or_wait(self) -> None:
        """Re-detect the MC window each reset and update the capture region.

        If MC's window can't be found at all, busy-wait (printing a warning
        every poll) until it comes back, the failsafe trips, or the user
        kills the process. Without this watchdog, alt-tabbing away or
        minimizing MC silently causes the env to capture a stale/garbage
        region — which is exactly how v11 spent 25 minutes "training" on a
        frozen desktop frame.
        """
        cfg = self.cfg
        if not cfg.require_window_each_reset or not cfg.use_window_detection:
            return
        warned = False
        while True:
            try:
                region = find_minecraft_window(
                    title_substr=cfg.window_title_substr
                )
            except (OSError, RuntimeError):
                region = None
            if region is not None:
                if (
                    self._mc_window is None
                    or region.left != self._mc_window.left
                    or region.top != self._mc_window.top
                    or region.width != self._mc_window.width
                    or region.height != self._mc_window.height
                ):
                    self._mc_window = region
                    self._monitor = region.as_mss_monitor()
                    mw = int(self._monitor["width"])
                    mh = int(self._monitor["height"])
                    if cfg.hotbar is not None:
                        self._hotbar = cfg.hotbar
                    else:
                        self._hotbar = HotbarSpec.from_monitor(
                            mw, mh, slot=1, gui_scale=cfg.gui_scale
                        )
                    self._fovea = cfg.fovea.rect(mw, mh)
                    if warned:
                        print(
                            "  [watchdog] Minecraft window is back: "
                            f"{region.left},{region.top} "
                            f"{region.width}x{region.height} -- resuming."
                        )
                return
            if not warned:
                print(
                    "  [watchdog] WARN: no Minecraft window found. "
                    "Bring MC to the foreground (alt-tab to it). "
                    "Training is paused; press F12 to abort."
                )
                warned = True
            if _failsafe.is_stopping():
                return
            time.sleep(cfg.window_watchdog_poll_s)

    def _grab_full_bgr(self) -> np.ndarray:
        assert self._sct is not None and self._monitor is not None
        raw = np.array(self._sct.grab(self._monitor))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def _make_obs(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = self.cfg.obs_shape
        small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def _log_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.inRange(frame_bgr, self._log_low, self._log_high)

    def _slot_crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        assert self._hotbar is not None
        hb = self._hotbar
        return frame_bgr[hb.y : hb.y + hb.h, hb.x : hb.x + hb.w]

    def _log_pixels_in_slot(self, frame_bgr: np.ndarray) -> int:
        slot = self._slot_crop(frame_bgr)
        if slot.size == 0:
            return 0
        return int(self._log_mask(slot).sum() // 255)

    def _slot_diff_from_empty(self, frame_bgr: np.ndarray) -> float:
        """Mean BGR distance between the current slot ROI and the empty-slot
        snapshot taken at reset. Returns 0.0 if no baseline has been captured.
        """
        if self._slot_empty_frame is None:
            return 0.0
        slot = self._slot_crop(frame_bgr)
        if slot.size == 0 or slot.shape != self._slot_empty_frame.shape:
            return 0.0
        return float(cv2.absdiff(slot, self._slot_empty_frame).mean())

    def _log_pixel_frac_fovea(self, frame_bgr: np.ndarray) -> float:
        assert self._fovea is not None
        x, y, w, h = self._fovea
        patch = frame_bgr[y : y + h, x : x + w]
        if patch.size == 0:
            return 0.0
        m = self._log_mask(patch)
        return float(m.mean()) / 255.0

    def _world_view_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """World pixels only: exclude bottom ~10% (hotbar / UI)."""
        if frame_bgr.size == 0:
            return frame_bgr
        h = frame_bgr.shape[0]
        cutoff = int(h * 0.9)
        return frame_bgr[:cutoff]

    def _log_pixel_frac_full(self, frame_bgr: np.ndarray) -> float:
        """Fraction of *world-view* pixels matching the log BGR range.

        Excludes the bottom 10% of the frame so the hotbar (which always
        contains log-colored UI once the agent picks up wood) doesn't
        permanently inflate the visible-log reward and trap the policy
        into "do nothing once you have wood".
        """
        world = self._world_view_bgr(frame_bgr)
        if world.size == 0:
            return 0.0
        return float(self._log_mask(world).mean()) / 255.0

    def _log_pixel_frac_left_right_world(
        self, frame_bgr: np.ndarray
    ) -> tuple[float, float]:
        """Log-colored fraction in left vs right half of world view (0..1 each)."""
        world = self._world_view_bgr(frame_bgr)
        if world.size == 0:
            return 0.0, 0.0
        _, ww = world.shape[:2]
        mid = max(1, ww // 2)
        left = world[:, :mid]
        right = world[:, mid:]
        fl = float(self._log_mask(left).mean()) / 255.0 if left.size else 0.0
        fr = float(self._log_mask(right).mean()) / 255.0 if right.size else 0.0
        return fl, fr

    def _sky_hint_score(self, frame_bgr: np.ndarray) -> float:
        """Rough 0..1: bright, log-empty upper band (typical sky view)."""
        if frame_bgr.size == 0:
            return 0.0
        h = frame_bgr.shape[0]
        top_end = int(h * 0.3)
        if top_end < 4:
            return 0.0
        band = frame_bgr[:top_end, :]
        m = self._log_mask(band)
        if float(m.mean()) / 255.0 > 0.02:
            return 0.0
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        mu = float(gray.mean())
        sat = max(1e-3, float(self.cfg.sky_bright_saturate))
        t = (mu - float(self.cfg.sky_bright_min_gray)) / sat
        return float(np.clip(t, 0.0, 1.0))

    def _log_centroid(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Centroid of log-colored pixels in cropped world view, normalized [0,1]."""
        if frame_bgr.size == 0:
            return None
        h, w = frame_bgr.shape[:2]
        top = int(h * 0.05)
        bot = int(h * 0.88)  # ignore hotbar + most of inventory overlay region
        if bot <= top:
            return None
        sub = frame_bgr[top:bot, :]
        m = self._log_mask(sub)
        if int(np.count_nonzero(m)) < 80:
            return None
        moms = cv2.moments(m, binaryImage=True)
        if moms["m00"] <= 0:
            return None
        cx = float(moms["m10"] / moms["m00"] / m.shape[1])
        cy = float(moms["m01"] / moms["m00"] / m.shape[0])
        return cx, cy

    def _teacher_force_prob(self) -> float:
        cfg = self.cfg
        start = float(np.clip(cfg.teacher_force_start, 0.0, 1.0))
        end = float(np.clip(cfg.teacher_force_end, 0.0, 1.0))
        if start <= 0.0 and end <= 0.0:
            return 0.0
        decay = max(1, int(cfg.teacher_force_decay_steps))
        t = min(1.0, float(self._global_steps) / float(decay))
        return start + (end - start) * t

    def _teacher_action(self, frame_bgr: np.ndarray) -> int:
        """Simple scripted tree-seeking controller for bootstrap curriculum."""
        fovea = self._log_pixel_frac_fovea(frame_bgr)
        ctr = self._log_centroid(frame_bgr)
        # If we are "blind", keep sweeping yaw but periodically pitch up/down
        # to recover from staring at ground/sky.
        if ctr is None:
            self._teacher_blind_steps += 1
            if (
                self._teacher_blind_steps
                % max(1, int(self.cfg.teacher_scan_flip_every))
                == 0
            ):
                self._teacher_scan_dir *= -1
            if not bool(self.cfg.disable_pitch_actions):
                if (
                    self._teacher_blind_steps
                    % max(1, int(self.cfg.teacher_blind_pitch_every))
                    == 0
                ):
                    px = int(self.cfg.teacher_pitch_recover_px)
                    ema = float(self._pitch_imbalance_ema)
                    bal = float(self.cfg.teacher_pitch_balance_ema)
                    if ema > bal:
                        ginput.move_rel(0, px)
                    elif ema < -bal:
                        ginput.move_rel(0, -px)
                    else:
                        ginput.move_rel(0, -px)
            # Run forward most steps; occasionally yaw to scan. Escapes
            # empty patches without in-place circle spam.
            cyc = max(2, int(self.cfg.teacher_blind_run_yaw_cycle))
            if (self._teacher_blind_steps - 1) % cyc < cyc - 1:
                return FORWARD
            return YAW_RIGHT if self._teacher_scan_dir > 0 else YAW_LEFT
        self._teacher_blind_steps = 0
        cx, _cy = ctr
        err_x = cx - 0.5
        if fovea >= float(self.cfg.teacher_mine_fovea_frac) and abs(err_x) <= float(self.cfg.teacher_center_tol) * 1.4:
            # Close + centered: fully commit to walking into the trunk and
            # continuously punching. No jumping in this regime.
            self._teacher_stuck_steps = 0
            return FORWARD_ATTACK
        if abs(err_x) > float(self.cfg.teacher_center_tol):
            return YAW_RIGHT if err_x > 0 else YAW_LEFT
        # Centered but still small: approach.
        if fovea <= self._teacher_prev_fovea + 0.003:
            self._teacher_stuck_steps += 1
        else:
            self._teacher_stuck_steps = 0
        self._teacher_prev_fovea = fovea
        # Jump only when clearly far from the trunk and truly stuck.
        if (
            self._teacher_stuck_steps >= 6
            and fovea < float(self.cfg.teacher_mine_fovea_frac) * 0.55
        ):
            return FORWARD_JUMP
        return FORWARD

    # ---- input application ------------------------------------------------

    def _set_held(self, key: str, want: bool) -> None:
        if self._held.get(key) == want:
            return
        self._held[key] = want
        if key == "lmb":
            ginput.mouse_left(want)
        elif key in ("w", "space"):
            (ginput.key_down if want else ginput.key_up)(key)
        else:
            raise KeyError(key)

    def _release_movement_and_attack(self) -> None:
        self._set_held("w", False)
        self._set_held("lmb", False)
        self._set_held("space", False)

    def _run_reset_chat_commands(self, *, timed_out: bool) -> None:
        # NOTE: ``timeout_extra_chat_commands`` are NOT run here; they run at
        # the end of ``step()`` when truncation fires, so the death-screen
        # transition has already begun by the time reset() is called. If we
        # ran /kill @s here, the very next ``_grab_full_bgr()`` (the
        # baseline-frame grab below) would capture the death screen as the
        # "empty slot" reference, poisoning the diff detector.
        del timed_out  # kept in signature for callers / future use
        cfg = self.cfg
        cmds: list[str] = list(cfg.auto_reset_chat_commands)
        if not cmds:
            return
        for cmd in cmds:
            try:
                ginput.chat_command(cmd)
            except (OSError, RuntimeError):
                # Don't let a transient SendInput hiccup take down training.
                pass

    def _random_reset_spread_tp_if_configured(self) -> None:
        r = int(self.cfg.reset_random_spread_blocks)
        if r <= 0:
            return
        dx = int(self.np_random.integers(-r, r + 1))
        dz = int(self.np_random.integers(-r, r + 1))
        xs = "~" if dx == 0 else f"~{dx}"
        zs = "~" if dz == 0 else f"~{dz}"
        try:
            ginput.chat_command(f"/tp @s {xs} ~ {zs}")
        except (OSError, RuntimeError):
            pass

    def _run_truncate_chat_commands(self) -> None:
        """Run the timeout chat commands at episode truncation.

        Runs INSIDE ``step()`` right before returning the truncated
        transition, so by the time the next ``reset()`` fires the player
        is already on the death screen / mid-respawn. This keeps the
        baseline frame for the diff detector clean (it's grabbed AFTER
        we dismiss the death screen, with the player back in the world).
        """
        cfg = self.cfg
        for cmd in cfg.timeout_extra_chat_commands:
            try:
                ginput.chat_command(cmd)
            except (OSError, RuntimeError):
                pass

    def _dismiss_death_screen(self) -> None:
        """Respawn after death. Re-detects MC window EVERY call.

        Strategy, in order:
          1. Re-find the MC window fresh (don't trust ``_mc_window`` —
             MC may have been moved, resized, or briefly lost focus
             since capture started). If detection fails, abort: a click
             at a stale coordinate is exactly the failure mode that
             clicked Cursor's tab bar in v6.
          2. Force MC to the foreground via ``SetForegroundWindow``.
             The Enter key and the click below only mean anything if
             MC is actually receiving them.
          3. Press Enter. On every modern MC version the Respawn button
             is the highlighted button on the death screen, and Enter
             activates the highlighted button without any coordinate
             math. This is the cheapest, most reliable path.
          4. As a backup: move the OS cursor to the computed button
             center inside the MC window and click. The math is
             ``window_h/2 + 10 * gui_scale`` from the top of the
             client area (DeathScreen.java places the button at GUI
             ``(W/2 - 100, H/2, 200, 20)``). Print the coords so the
             user can sanity-check them in the run log.
          5. Press Enter once more — covers the case where the click
             landed on the Respawn button but Windows ate the click
             due to focus stealing during the foreground transition.
        """
        from mine_diamonds.capture import find_minecraft_window  # local import; avoids hard win32 dep at module load

        region = find_minecraft_window()
        if region is None:
            print(
                "  [dismiss] WARN: no Minecraft window found; falling "
                "back to Enter key only.",
                flush=True,
            )
            try:
                ginput.key_down("enter")
                time.sleep(0.04)
                ginput.key_up("enter")
            except (OSError, RuntimeError):
                pass
            return

        # Cache the freshly-detected region so subsequent capture also
        # uses it. (If the window moved, this self-heals capture.)
        self._mc_window = region
        self._monitor = region.as_mss_monitor()

        # Bring MC to the foreground before sending input.
        try:
            from mine_diamonds.capture import focus_window_by_title

            focus_window_by_title()
        except (OSError, RuntimeError, ImportError):
            pass
        time.sleep(0.08)

        # 3) Enter (the cheap, coord-free path).
        try:
            ginput.key_down("enter")
            time.sleep(0.04)
            ginput.key_up("enter")
        except (OSError, RuntimeError):
            pass
        time.sleep(0.18)

        # 4) Backup click on the literal button center.
        cx = int(region.left + region.width // 2)
        cy = int(region.top + region.height // 2 + 10 * int(self.cfg.gui_scale))
        # Sanity: the click must fall INSIDE the detected MC window. If
        # somehow it doesn't, refuse to click — better to skip the dismiss
        # than to click on whatever's behind MC (Cursor tab, browser, etc).
        in_x = region.left <= cx <= region.left + region.width
        in_y = region.top <= cy <= region.top + region.height
        print(
            f"  [dismiss] mc_window=({region.left},{region.top},"
            f"{region.width}x{region.height}) "
            f"click=({cx},{cy}) inside={in_x and in_y}",
            flush=True,
        )
        if in_x and in_y:
            try:
                ginput.click_at(cx, cy, button="left", hold_ms=80)
            except (OSError, RuntimeError):
                pass
        time.sleep(0.1)

        # 5) One more Enter for good measure.
        try:
            ginput.key_down("enter")
            time.sleep(0.04)
            ginput.key_up("enter")
        except (OSError, RuntimeError):
            pass

    def _apply_action(self, action: int) -> None:
        cfg = self.cfg
        if action == NOOP:
            self._release_movement_and_attack()
        elif action == FORWARD:
            self._set_held("w", True)
            self._set_held("lmb", False)
            self._set_held("space", False)
        elif action == YAW_LEFT:
            self._release_movement_and_attack()
            ginput.move_rel(-cfg.yaw_step_px, 0)
        elif action == YAW_RIGHT:
            self._release_movement_and_attack()
            ginput.move_rel(cfg.yaw_step_px, 0)
        elif action == PITCH_UP:
            self._release_movement_and_attack()
            if not bool(cfg.disable_pitch_actions):
                ginput.move_rel(0, -cfg.pitch_step_px)
        elif action == PITCH_DOWN:
            self._release_movement_and_attack()
            if not bool(cfg.disable_pitch_actions):
                ginput.move_rel(0, cfg.pitch_step_px)
        elif action == FORWARD_ATTACK:
            self._set_held("w", True)
            self._set_held("lmb", True)
            self._set_held("space", False)
        elif action == FORWARD_JUMP:
            # Hold W + Space. In MC, holding Space while walking auto-jumps
            # whenever the player tries to walk into a 1-block rise, so this
            # is exactly what the agent needs to scale terrain while looking
            # for trees. LMB stays released so we don't break leaves on
            # sapling tops while bunny-hopping.
            self._set_held("w", True)
            self._set_held("space", True)
            self._set_held("lmb", False)
        else:
            raise ValueError(f"invalid action {action}")

    # ---- gymnasium API ----------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_capture()
        self._refresh_window_or_wait()
        self._release_movement_and_attack()
        # Make sure MC is the foreground window before we start sending
        # chat commands or Enter presses. If the user's focus drifted to
        # Cursor / Discord / a browser between the countdown and the first
        # reset, our keystrokes would land on the wrong app and the agent
        # would never receive its init buffs.
        try:
            from mine_diamonds.capture import focus_window_by_title

            focus_window_by_title()
        except (OSError, RuntimeError, ImportError):
            pass
        time.sleep(0.08)
        # ALWAYS try to dismiss any blocking UI screen at reset, not just on
        # timeouts. Why: even outside the truncate-on-timeout path, the agent
        # can land on the death screen via two routes that bit us in v8:
        #   (a) gameplay death (fall, lava, mob) the env doesn't know about
        #   (b) the death-screen overlay covers slot 1, the slot-diff
        #       detector reads "different from empty" and terminates the
        #       episode as if a log was acquired.
        # In either case, the next reset() inherits a death-screen frame.
        # The Enter key is a no-op in normal world view (it's not bound by
        # default in MC Java) but activates the highlighted Respawn button
        # on the death screen, so it's safe to send unconditionally.
        timed_out = bool(self._reset_due_to_timeout)
        # Run one-shot init commands on the very first reset of training.
        # (e.g. "/gamerule doImmediateRespawn true", "/effect give @s
        # minecraft:resistance ...") — these set up world state we want to
        # hold for the entire run.
        if not self._init_done:
            for cmd in self.cfg.init_chat_commands:
                try:
                    ginput.chat_command(cmd)
                except (OSError, RuntimeError):
                    pass
            self._init_done = True
        if timed_out:
            time.sleep(0.7)  # let the death screen render fully
            self._dismiss_death_screen()
            time.sleep(0.5)
            self._dismiss_death_screen()
            time.sleep(0.7)  # let the respawn animation finish
            # Now that the player is alive again, re-apply anything /kill
            # cleared (effect buffs, gamemode, etc.). This is the ONLY
            # place these commands run on subsequent episodes; keeping
            # them out of ``auto_reset_chat_commands`` is what stopped
            # the chat-spam death loop in v10.
            for cmd in self.cfg.post_respawn_chat_commands:
                try:
                    ginput.chat_command(cmd)
                except (OSError, RuntimeError):
                    pass
        else:
            # Cheap unconditional Enter — clears a stuck UI screen that
            # snuck through (false-positive log_acquired on the death-
            # overlay, gameplay death, etc.).
            try:
                ginput.key_down("enter")
                time.sleep(0.04)
                ginput.key_up("enter")
            except (OSError, RuntimeError):
                pass
            time.sleep(0.2)
        # In-game soft reset: clear inventory (and optionally teleport) via
        # MC chat. Keeps the reward signal alive across episodes — without
        # this, slot 1 stays full of wood after the first success and the
        # diff detector stops firing.
        self._run_reset_chat_commands(timed_out=timed_out)
        self._random_reset_spread_tp_if_configured()
        self._reset_due_to_timeout = False
        if self.cfg.post_reset_pause_s > 0:
            time.sleep(self.cfg.post_reset_pause_s)
        frame = self._grab_full_bgr()
        obs = self._make_obs(frame)
        self._last_obs = obs

        # Snapshot the empty-slot ROI as the reference for the diff detector,
        # and snapshot the empty-slot pixel count for the legacy detector.
        # CALLER MUST: open MC, focus the world (NOT inventory), make sure
        # slot 1 is empty before reset(). If slot 1 is already full, every
        # episode starts with logs_collected=1 and terminates immediately.
        cfg = self.cfg
        slot = self._slot_crop(frame)
        if slot.size > 0:
            self._slot_empty_frame = slot.copy()
        else:
            self._slot_empty_frame = None
        slot_px = self._log_pixels_in_slot(frame)
        if cfg.auto_baseline_log_threshold:
            self._slot_empty_baseline = int(slot_px)
            self._effective_log_threshold = int(
                max(
                    cfg.log_pixel_threshold,
                    slot_px + cfg.log_pixel_baseline_margin,
                )
            )
        else:
            self._slot_empty_baseline = 0
            self._effective_log_threshold = int(cfg.log_pixel_threshold)

        # We just snapshotted the empty slot, so by definition diff = 0 here.
        # Slot is empty by assumption; if it isn't, nothing the env can do.
        had_log_now = False
        self._had_log_last_step = had_log_now
        self._have_log_streak = 0
        self._logs_collected = 0
        self._episode_start = time.perf_counter()
        self._steps = 0
        self._teacher_prev_fovea = 0.0
        self._teacher_stuck_steps = 0
        self._teacher_blind_steps = 0
        self._teacher_scan_dir = 1 if np.random.random() < 0.5 else -1
        self._pitch_imbalance_ema = 0.0
        self._blind_yaw_streak = 0
        self._obs_history = [obs.copy()]
        # Initialize the approach-reward baseline to whatever's visible at
        # episode start so the first step's delta is ~0. Without this, an
        # agent that respawns next to a tree would get a huge bogus +reward
        # for "approaching" on step 1.
        self._prev_full_frame_log_frac = self._log_pixel_frac_full(frame)
        self._fovea_hold_ema = float(self._log_pixel_frac_fovea(frame))
        info = {
            "had_log_at_reset": had_log_now,
            "slot_empty_baseline_px": int(self._slot_empty_baseline),
            "effective_log_threshold": int(self._effective_log_threshold),
            "slot_diff_threshold": float(cfg.slot_diff_threshold),
            "slot_diff_margin": float(cfg.slot_diff_margin),
            "log_confirm_steps": int(cfg.log_confirm_steps),
            "use_slot_diff_detector": bool(cfg.use_slot_diff_detector),
            "hotbar_rect": (
                self._hotbar.x, self._hotbar.y, self._hotbar.w, self._hotbar.h
            ) if self._hotbar else None,
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_start is None:
            raise RuntimeError("step() called before reset()")
        cfg = self.cfg
        step_start = time.perf_counter()

        # Failsafe: F12 (debounced) or stop-file after grace. The watcher already
        # released keys, but skip applying any new action and truncate
        # this episode so the SB3 callback can stop training cleanly.
        if _failsafe.is_stopping():
            self._release_movement_and_attack()
            frame = self._grab_full_bgr() if self._sct is not None else np.zeros(
                (self.cfg.obs_shape[0], self.cfg.obs_shape[1], 3), dtype=np.uint8
            )
            obs = self._make_obs(frame) if frame.size else self._last_obs
            if obs is None:
                obs = np.zeros(
                    (self.cfg.obs_shape[0], self.cfg.obs_shape[1], 3),
                    dtype=np.uint8,
                )
            info = {
                "elapsed_s": float(time.perf_counter() - self._episode_start),
                "fovea_log_frac": 0.0,
                "slot_log_px": 0,
                "slot_diff": 0.0,
                "have_log_now": False,
                "log_acquired": False,
                "logs_collected": int(self._logs_collected),
                "aimed": False,
                "action_name": ACTION_NAMES[int(action)],
                "failsafe_stop": True,
            }
            return obs, 0.0, False, True, info

        # Optional teacher-forced action override (curriculum bootstrap).
        selected_action = int(action)
        teacher_prob = self._teacher_force_prob()
        teacher_used = False
        if teacher_prob > 0.0 and self._sct is not None and np.random.random() < teacher_prob:
            try:
                pre_frame = self._grab_full_bgr()
                selected_action = self._teacher_action(pre_frame)
                teacher_used = True
            except Exception:
                selected_action = int(action)
                teacher_used = False

        # Sticky actions / frame skip. Each call to env.step() applies the
        # chosen action ``action_repeat`` times, sleeping ``frame_time_s``
        # between each application. For mouse-delta actions (yaw/pitch)
        # the deltas accumulate visibly: a single YAW_LEFT decision with
        # action_repeat=4 and yaw_step_px=60 sweeps the camera by ~240
        # mickeys before the agent gets to choose again. For held keys
        # (W, Space, LMB) the second-onwards _set_held calls are no-ops, so
        # the keys simply stay pressed across the substeps — which is also
        # what we want for sustained movement / attack.
        n_repeats = max(1, int(cfg.action_repeat))
        if int(selected_action) == FORWARD_ATTACK:
            per = max(1e-4, float(cfg.frame_time_s))
            need = int(np.ceil(float(cfg.mine_hold_duration_s) / per))
            n_repeats = max(n_repeats, need)
        # Bail out of repeats early if the failsafe fires mid-step. We've
        # already passed the gate above but the user could press F12 during
        # a 200ms substep window; checking each repeat keeps response time
        # under ~50 ms.
        for r in range(n_repeats):
            if _failsafe.is_stopping():
                break
            self._apply_action(int(selected_action))
            target_elapsed = (r + 1) * cfg.frame_time_s
            elapsed_so_far = time.perf_counter() - step_start
            sleep_for = target_elapsed - elapsed_so_far
            if sleep_for > 0:
                time.sleep(sleep_for)

        sig = 0.0
        if not bool(cfg.disable_pitch_actions):
            if int(selected_action) == PITCH_UP:
                sig = 1.0
            elif int(selected_action) == PITCH_DOWN:
                sig = -1.0
        pa = float(np.clip(cfg.pitch_imbalance_ema_alpha, 1e-3, 1.0))
        self._pitch_imbalance_ema = (1.0 - pa) * self._pitch_imbalance_ema + pa * sig

        frame = self._grab_full_bgr()
        obs = self._make_obs(frame)
        self._last_obs = obs
        self._last_full_frame = frame

        slot_log_px = self._log_pixels_in_slot(frame)
        slot_diff = self._slot_diff_from_empty(frame)
        diff_gate = slot_diff >= (cfg.slot_diff_threshold + cfg.slot_diff_margin)
        px_gate = slot_log_px >= self._effective_log_threshold
        if cfg.use_slot_diff_detector and self._slot_empty_frame is not None:
            candidate_have_log = bool(diff_gate and px_gate)
        else:
            candidate_have_log = bool(px_gate)
        if candidate_have_log:
            self._have_log_streak += 1
        else:
            self._have_log_streak = 0
        have_log_now = self._have_log_streak >= max(1, int(cfg.log_confirm_steps))
        fovea_log_frac = self._log_pixel_frac_fovea(frame)
        full_log_frac = self._log_pixel_frac_full(frame)
        aimed = fovea_log_frac >= cfg.aimed_threshold_frac
        close_mine = bool(aimed) and full_log_frac >= float(
            cfg.close_mine_full_log_frac
        )
        frac_L, frac_R = self._log_pixel_frac_left_right_world(frame)
        max_lr = max(frac_L, frac_R)
        t_lr = max(1e-8, frac_L + frac_R)
        imb_lr = (frac_L - frac_R) / t_lr

        # ---- reward ------------------------------------------------------
        reward = float(cfg.reward_step_penalty)
        forward_close_reward = 0.0
        yaw_periph_reward = 0.0

        # 1) "I can see wood" — per-step reward for log pixels visible
        #    ANYWHERE in the world view. Fires the moment a tree edges
        #    into peripheral view, so the policy gets a clean gradient
        #    while rotating toward visible wood. Hotbar region is
        #    excluded inside `_log_pixel_frac_full` so this doesn't pin
        #    high once the agent has wood in inventory.
        sat_v = max(1e-6, cfg.log_visible_saturation_frac)
        log_visible_norm = min(1.0, full_log_frac / sat_v)
        vis_scale = 0.0 if close_mine else (0.35 if aimed else 1.0)
        reward += cfg.reward_log_visible_max * log_visible_norm * vis_scale

        # 2) "I'm approaching the wood" — delta full-frame log fraction.
        delta_full = full_log_frac - self._prev_full_frame_log_frac
        approach_reward = 0.0
        if (not close_mine) and delta_full > 0:
            sat_a = max(1e-6, cfg.log_approach_saturation_delta)
            approach_reward = (
                cfg.reward_log_approach_max * min(1.0, delta_full / sat_a)
            )
            reward += approach_reward
        self._prev_full_frame_log_frac = full_log_frac

        # 2b) Peripheral wood + walk toward any visible wood.
        sat_p = max(1e-6, float(cfg.peripheral_log_saturation_frac))
        peripheral_wood_norm = min(1.0, max_lr / sat_p)
        reward += float(cfg.reward_peripheral_wood_max) * peripheral_wood_norm
        if (
            (not close_mine)
            and full_log_frac >= float(cfg.forward_close_min_full_log_frac)
            and selected_action in (FORWARD, FORWARD_ATTACK)
        ):
            refc = max(1e-6, float(cfg.forward_close_full_ref_frac))
            closeness = min(1.0, full_log_frac / refc)
            forward_close_reward = float(cfg.reward_forward_close_max) * closeness
            reward += forward_close_reward

        # 3) Fovea aim + EMA for sustained wood in the crosshair.
        ref = max(1e-6, float(cfg.fovea_aim_reference_frac))
        aim_linear = min(1.0, fovea_log_frac / ref)
        a_hold = float(np.clip(cfg.fovea_hold_ema_alpha, 1e-4, 1.0))
        self._fovea_hold_ema = (1.0 - a_hold) * float(self._fovea_hold_ema) + a_hold * float(
            fovea_log_frac
        )
        hold_linear = min(1.0, float(self._fovea_hold_ema) / ref)
        if not close_mine:
            reward += cfg.reward_aim_dense * aim_linear
            reward += float(cfg.reward_fovea_hold_dense_max) * hold_linear

        # 4) Mine / approach / distractions while crosshair on wood.
        if selected_action == FORWARD_ATTACK and aimed:
            reward += cfg.reward_attack_when_aimed
            reward += float(cfg.reward_mine_commit_bonus)
        elif close_mine and selected_action != FORWARD_ATTACK:
            reward += float(cfg.penalty_move_while_close_mine)
        elif aimed:
            if selected_action == FORWARD:
                reward += float(cfg.reward_forward_when_aimed)
            elif selected_action in (YAW_LEFT, YAW_RIGHT):
                reward += float(cfg.penalty_yaw_when_aimed)
            elif selected_action == FORWARD_JUMP:
                reward += float(cfg.penalty_aimed_forward_jump)
            elif (not bool(cfg.disable_pitch_actions)) and selected_action in (
                PITCH_UP,
                PITCH_DOWN,
            ):
                reward += cfg.penalty_move_when_aimed_not_attacking
        elif (
            (not close_mine)
            and fovea_log_frac >= float(cfg.fovea_soft_yaw_penalty_frac)
            and selected_action in (YAW_LEFT, YAW_RIGHT)
        ):
            reward += float(cfg.penalty_yaw_when_fovea_soft)

        # 5) Exploratory tie-breakers when still searching (blind in fovea).
        treeless = full_log_frac < float(cfg.explore_treeless_log_frac)
        sky_hint = self._sky_hint_score(frame)
        if fovea_log_frac < cfg.explore_blind_threshold:
            if selected_action in (YAW_LEFT, YAW_RIGHT):
                reward += cfg.reward_explore_yaw_when_blind
            if selected_action in (FORWARD, FORWARD_ATTACK, FORWARD_JUMP):
                reward += cfg.reward_explore_forward_when_blind
            if (
                treeless
                and selected_action == FORWARD_JUMP
            ):
                reward += float(cfg.reward_sprint_blind)
            if treeless and sky_hint > 0.15:
                reward -= float(cfg.penalty_sky_stare) * float(sky_hint)
            if treeless:
                if selected_action in (YAW_LEFT, YAW_RIGHT):
                    self._blind_yaw_streak = min(400, int(self._blind_yaw_streak) + 1)
                else:
                    self._blind_yaw_streak = max(0, int(self._blind_yaw_streak) - 1)
                thr = int(cfg.blind_yaw_streak_threshold)
                if int(self._blind_yaw_streak) > thr:
                    over = int(self._blind_yaw_streak) - thr
                    reward -= float(cfg.penalty_blind_yaw_streak) * over
            else:
                self._blind_yaw_streak = 0
            # Discourage runaway pitch (sky/ground) while blind.
            reward -= cfg.penalty_pitch_imbalance * abs(
                float(self._pitch_imbalance_ema)
            )
        else:
            self._blind_yaw_streak = 0

        # 5b) Yaw toward the side with more log pixels (tree left vs right).
        if (
            (not close_mine)
            and (not aimed)
            and max_lr >= float(cfg.explore_treeless_log_frac)
        ):
            thr = float(cfg.peripheral_yaw_imb_threshold)
            if abs(imb_lr) > thr:
                strength = min(1.0, (abs(imb_lr) - thr) / (1.0 - thr + 1e-6)) * min(
                    1.0, max_lr / sat_p
                )
                if imb_lr > 0.0:
                    if selected_action == YAW_LEFT:
                        yaw_periph_reward += (
                            float(cfg.reward_yaw_toward_peripheral_max) * strength
                        )
                    elif selected_action == YAW_RIGHT:
                        yaw_periph_reward -= (
                            float(cfg.penalty_yaw_away_peripheral_max) * strength
                        )
                else:
                    if selected_action == YAW_RIGHT:
                        yaw_periph_reward += (
                            float(cfg.reward_yaw_toward_peripheral_max) * strength
                        )
                    elif selected_action == YAW_LEFT:
                        yaw_periph_reward -= (
                            float(cfg.penalty_yaw_away_peripheral_max) * strength
                        )
        reward += yaw_periph_reward

        # 6) Curiosity / frame-diff shaping: tiny bonus for visual change
        #    vs. the observation ``frame_diff_lookback_steps`` ago. Just
        #    enough to favor moving the camera over staring at a wall.
        frame_diff = 0.0
        if self._obs_history:
            ref = self._obs_history[0]
            if ref.shape == obs.shape:
                frame_diff = float(cv2.absdiff(obs, ref).mean())
        diff_norm = min(1.0, frame_diff / max(1e-6, cfg.frame_diff_saturation))
        reward += cfg.reward_frame_diff_max * diff_norm
        self._obs_history.append(obs)
        if len(self._obs_history) > cfg.frame_diff_lookback_steps:
            self._obs_history.pop(0)

        # Sparse: log just appeared in the hotbar slot.
        log_acquired = bool(have_log_now and not self._had_log_last_step)
        if log_acquired:
            reward += cfg.reward_log_acquired
            self._logs_collected += 1
        self._had_log_last_step = have_log_now

        self._steps += 1
        self._global_steps += 1
        elapsed = time.perf_counter() - self._episode_start
        terminated = self._logs_collected >= cfg.logs_to_succeed
        truncated = (not terminated) and (elapsed >= cfg.max_seconds)
        if terminated or truncated:
            self._release_movement_and_attack()
        if truncated:
            self._reset_due_to_timeout = True
            # Fire the timeout chat commands NOW (e.g. /clear @s,
            # /kill @e[type=item], /kill @s). Running them here — not in
            # reset() — means by the time reset() takes its baseline frame
            # the player is already past the death screen, so the empty-slot
            # snapshot is clean. Order matters: clear inventory first so
            # /kill @s drops nothing, then sweep stray drops, then kill.
            self._run_truncate_chat_commands()

        info = {
            "elapsed_s": float(elapsed),
            "fovea_log_frac": float(fovea_log_frac),
            "full_log_frac": float(full_log_frac),
            "log_approach_reward": float(approach_reward),
            "slot_log_px": int(slot_log_px),
            "slot_diff": float(slot_diff),
            "slot_empty_baseline_px": int(self._slot_empty_baseline),
            "effective_log_threshold": int(self._effective_log_threshold),
            "slot_diff_threshold": float(cfg.slot_diff_threshold),
            "slot_diff_margin": float(cfg.slot_diff_margin),
            "log_confirm_steps": int(cfg.log_confirm_steps),
            "have_log_now": bool(have_log_now),
            "log_acquired": log_acquired,
            "logs_collected": int(self._logs_collected),
            "aimed": bool(aimed),
            "close_mine": bool(close_mine),
            "fovea_hold_ema": float(self._fovea_hold_ema),
            "log_frac_left": float(frac_L),
            "log_frac_right": float(frac_R),
            "peripheral_imb": float(imb_lr),
            "forward_close_reward": float(forward_close_reward),
            "yaw_periph_reward": float(yaw_periph_reward),
            "frame_diff": float(frame_diff),
            "env_substeps": int(n_repeats),
            "action_name": ACTION_NAMES[int(selected_action)],
            "policy_action_name": ACTION_NAMES[int(action)],
            "teacher_used": bool(teacher_used),
            "teacher_force_prob": float(teacher_prob),
            "pitch_imbalance_ema": float(self._pitch_imbalance_ema),
            "sky_hint": float(sky_hint),
            "blind_yaw_streak": int(self._blind_yaw_streak),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        if self._last_obs is None:
            return None
        return self._last_obs

    def close(self) -> None:
        try:
            self._release_movement_and_attack()
        except Exception:
            pass
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
            self._monitor = None
