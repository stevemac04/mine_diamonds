"""PPO + CnnPolicy training on the real Minecraft Java client.

Single env (real MC can't be parallelized), so this is intrinsically slow:
~20 transitions/sec real-time. The rollout/update budget is therefore
much smaller per iteration than in parallel simulators — we use short ``n_steps``,
small minibatches, and frequent checkpoints so that interrupting the run
costs at most a couple of minutes of progress.

USAGE (PowerShell):
  .\\.venv\\Scripts\\Activate.ps1
  $env:PYTHONUTF8=1

  # Default: 2-hour run. With action_repeat=4 each agent decision is 200 ms,
  # so 36k decisions ~= 2 hours wall-clock. Reasonable target for a
  # tree-chopping policy with this reward shaping.
  python scripts\\train_ppo_minecraft_real.py --run-name mc_real_v1

  # Longer run if the 2hr policy is still flailing:
  python scripts\\train_ppo_minecraft_real.py `
      --total-steps 108000 `
      --run-name mc_real_v2_long  # ~6 hours

  # Tweak exploration knobs from the CLI:
  python scripts\\train_ppo_minecraft_real.py `
      --action-repeat 6 --yaw-step-px 80

PREREQUISITES (do NOT skip):
  1. Run scripts\\smoke_minecraft_real.py first and confirm:
       - capture_source starts with 'window:'  (MC window detected)
       - the red hotbar ROI is on slot 1 in capture_initial.png
       - log_acquired fires when you mine manually
     Don't train until smoke passes — silent reward-signal bugs will
     burn the entire run.
  2. Spawn near trees on a peaceful survival world with the simple-colors
     resource pack loaded. Empty hotbar (slot 1 visibly empty).
  3. Click the MC window so it has keyboard focus. Don't touch the
     keyboard or mouse for the duration of the run. Alt-tabbing breaks
     input delivery; the run keeps going but the agent will stop seeing
     responses.

Logs go to ``runs/<run_name>/`` (TensorBoard) and checkpoints go to
``runs/<run_name>/checkpoints/``. Open TensorBoard in a separate shell:

  .\\.venv\\Scripts\\Activate.ps1
  tensorboard --logdir runs --port 6006
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from mine_diamonds import failsafe as _failsafe  # noqa: E402
from mine_diamonds.envs.minecraft_real import (  # noqa: E402
    MinecraftRealConfig,
    MinecraftRealEnv,
)


class EpisodeJSONLCallback(BaseCallback):
    """Per-episode ground-truth log: one JSON object per finished episode.

    This is the *paper-grade* data source. TensorBoard shows rolling means;
    this writes one row per episode with raw counts, lengths, returns, and
    (most importantly) ``logs_acquired`` — the unambiguous "did the agent
    mine a log under its own policy" boolean.

    Output: ``runs/<run_name>/episodes.jsonl``. One JSON object per line:

        {"episode": 7, "wall_t": 1737055000.123, "elapsed_s": 312.4,
         "global_step": 3584, "ep_len": 256, "ep_return": 41.7,
         "logs_acquired": 1, "ep_aim_rate": 0.31, "ep_yaw_rate": 0.18,
         "ep_jump_rate": 0.07, "ep_full_log_max": 0.41,
         "ep_full_log_mean": 0.18, "ep_approach_total": 5.7,
         "ep_attack_when_aimed": 12, "truncated": true}
    """

    def __init__(self, jsonl_path: Path, t0: float, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._path = Path(jsonl_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = float(t0)
        self._fp = self._path.open("a", encoding="utf-8")
        self._episode_idx: int = 0
        self._reset_running()

    def _reset_running(self) -> None:
        self._ep_steps: int = 0
        self._ep_return: float = 0.0
        self._ep_logs: int = 0
        self._ep_aimed: int = 0
        self._ep_attack_when_aimed: int = 0
        self._ep_yaw: int = 0
        self._ep_jump: int = 0
        self._ep_full_log_sum: float = 0.0
        self._ep_full_log_max: float = 0.0
        self._ep_approach_sum: float = 0.0
        self._ep_was_truncated: bool = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            if info.get("log_acquired"):
                self._ep_logs += 1
            action_name = info.get("action_name")
            if info.get("aimed"):
                self._ep_aimed += 1
                if action_name == "forward_attack":
                    self._ep_attack_when_aimed += 1
            if action_name in ("yaw_left", "yaw_right"):
                self._ep_yaw += 1
            if action_name == "forward_jump":
                self._ep_jump += 1
            full_lf = float(info.get("full_log_frac", 0.0) or 0.0)
            self._ep_full_log_sum += full_lf
            if full_lf > self._ep_full_log_max:
                self._ep_full_log_max = full_lf
            self._ep_approach_sum += float(
                info.get("log_approach_reward", 0.0) or 0.0
            )
            self._ep_steps += 1
            self._ep_return += float(rewards[i]) if i < len(rewards) else 0.0

            if i < len(dones) and bool(dones[i]):
                steps = max(1, self._ep_steps)
                # SB3 reports `dones=True` for both terminated and
                # truncated; the env tells us which via the prior info
                # state (truncated => reset_due_to_timeout was just set).
                # We look at elapsed_s vs. logs to disambiguate without
                # peeking inside the env.
                truncated = self._ep_logs == 0
                self._episode_idx += 1
                rec = {
                    "episode": self._episode_idx,
                    "wall_t": time.time(),
                    "elapsed_s": time.time() - self._t0,
                    "global_step": int(self.num_timesteps),
                    "ep_len": int(steps),
                    "ep_return": float(self._ep_return),
                    "logs_acquired": int(self._ep_logs),
                    "ep_aim_rate": float(self._ep_aimed / steps),
                    "ep_yaw_rate": float(self._ep_yaw / steps),
                    "ep_jump_rate": float(self._ep_jump / steps),
                    "ep_full_log_max": float(self._ep_full_log_max),
                    "ep_full_log_mean": float(self._ep_full_log_sum / steps),
                    "ep_approach_total": float(self._ep_approach_sum),
                    "ep_attack_when_aimed": int(self._ep_attack_when_aimed),
                    "truncated": bool(truncated),
                }
                self._fp.write(json.dumps(rec) + "\n")
                self._fp.flush()
                self._reset_running()
        return True

    def _on_training_end(self) -> None:
        try:
            self._fp.close()
        except (OSError, ValueError):
            pass


class LogAcquiredSnapshotCallback(BaseCallback):
    """Save a full-resolution screenshot every time ``log_acquired`` fires.

    These are the visual evidence frames for the paper: literally a picture
    of what the policy saw at the moment it got a log. Stored as
    ``runs/<run_name>/snaps/log_<seq>_step<step>.png`` so they're easy to
    sort by capture order. The first 12 are usually enough for a figure.
    """

    def __init__(self, out_dir: Path, max_snaps: int = 64, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._max_snaps = int(max_snaps)
        self._n_saved: int = 0

    def _on_step(self) -> bool:
        if self._n_saved >= self._max_snaps:
            return True
        infos = self.locals.get("infos", []) or []
        if not any(i.get("log_acquired") for i in infos):
            return True
        # The env exposes the most recent BGR frame on _last_full_frame.
        try:
            frames = self.training_env.get_attr("_last_full_frame")
        except (AttributeError, RuntimeError):
            return True
        for i, info in enumerate(infos):
            if not info.get("log_acquired"):
                continue
            if i >= len(frames) or frames[i] is None:
                continue
            self._n_saved += 1
            png = self._out_dir / (
                f"log_{self._n_saved:03d}_step{int(self.num_timesteps):08d}.png"
            )
            try:
                cv2.imwrite(str(png), frames[i])
            except (OSError, cv2.error):
                pass
            if self._n_saved >= self._max_snaps:
                break
        return True


class FailsafeCallback(BaseCallback):
    """Stops PPO at the first env step after the failsafe trips.

    The failsafe daemon thread already released all held keys before this
    callback fires; this class just tells SB3 to exit ``model.learn``."""

    def _on_step(self) -> bool:
        return not _failsafe.is_stopping()


class MCRealStatsCallback(BaseCallback):
    """Tracks per-episode log_acquired rate and dense-shaping diagnostics."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._ep_log_count: int = 0
        self._ep_aimed_steps: int = 0
        self._ep_attack_when_aimed: int = 0
        self._ep_steps: int = 0
        self._ep_yaw_steps: int = 0
        self._ep_jump_steps: int = 0
        self._ep_frame_diff_sum: float = 0.0
        self._ep_full_log_frac_sum: float = 0.0
        self._ep_full_log_frac_max: float = 0.0
        self._ep_approach_reward_sum: float = 0.0
        self._ep_returns: list[float] = []
        self._ep_logs: list[int] = []
        self._ep_aim_rate: list[float] = []
        self._ep_yaw_rate: list[float] = []
        self._ep_jump_rate: list[float] = []
        self._ep_frame_diff: list[float] = []
        self._ep_full_log_mean: list[float] = []
        self._ep_full_log_max: list[float] = []
        self._ep_approach_total: list[float] = []
        self._ep_running_return: float = 0.0
        self._t_last_log: float = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if info.get("log_acquired"):
                self._ep_log_count += 1
            action_name = info.get("action_name")
            if info.get("aimed"):
                self._ep_aimed_steps += 1
                if action_name == "forward_attack":
                    self._ep_attack_when_aimed += 1
            if action_name in ("yaw_left", "yaw_right"):
                self._ep_yaw_steps += 1
            if action_name == "forward_jump":
                self._ep_jump_steps += 1
            self._ep_frame_diff_sum += float(info.get("frame_diff", 0.0) or 0.0)
            full_lf = float(info.get("full_log_frac", 0.0) or 0.0)
            self._ep_full_log_frac_sum += full_lf
            if full_lf > self._ep_full_log_frac_max:
                self._ep_full_log_frac_max = full_lf
            self._ep_approach_reward_sum += float(
                info.get("log_approach_reward", 0.0) or 0.0
            )
            self._ep_steps += 1

            r = float(rewards[i]) if i < len(rewards) else 0.0
            self._ep_running_return += r

            if i < len(dones) and bool(dones[i]):
                self._ep_returns.append(self._ep_running_return)
                self._ep_logs.append(self._ep_log_count)
                steps = max(1, self._ep_steps)
                self._ep_aim_rate.append(self._ep_aimed_steps / steps)
                self._ep_yaw_rate.append(self._ep_yaw_steps / steps)
                self._ep_jump_rate.append(self._ep_jump_steps / steps)
                self._ep_frame_diff.append(self._ep_frame_diff_sum / steps)
                self._ep_full_log_mean.append(
                    self._ep_full_log_frac_sum / steps
                )
                self._ep_full_log_max.append(self._ep_full_log_frac_max)
                self._ep_approach_total.append(self._ep_approach_reward_sum)
                self._ep_running_return = 0.0
                self._ep_log_count = 0
                self._ep_aimed_steps = 0
                self._ep_attack_when_aimed = 0
                self._ep_yaw_steps = 0
                self._ep_jump_steps = 0
                self._ep_frame_diff_sum = 0.0
                self._ep_full_log_frac_sum = 0.0
                self._ep_full_log_frac_max = 0.0
                self._ep_approach_reward_sum = 0.0
                self._ep_steps = 0

        # Log a rolling window every ~10s wall.
        now = time.time()
        if now - self._t_last_log > 10.0 and self._ep_returns:
            window = 10
            recent_ret = np.mean(self._ep_returns[-window:])
            recent_logs = np.mean(self._ep_logs[-window:])
            recent_aim = np.mean(self._ep_aim_rate[-window:])
            recent_yaw = np.mean(self._ep_yaw_rate[-window:])
            recent_jump = np.mean(self._ep_jump_rate[-window:])
            recent_diff = np.mean(self._ep_frame_diff[-window:])
            recent_full_mean = np.mean(self._ep_full_log_mean[-window:])
            recent_full_max = np.mean(self._ep_full_log_max[-window:])
            recent_approach = np.mean(self._ep_approach_total[-window:])
            self.logger.record("mc/ep_return_mean", float(recent_ret))
            self.logger.record("mc/ep_logs_mean", float(recent_logs))
            self.logger.record("mc/ep_aim_rate", float(recent_aim))
            self.logger.record("mc/ep_yaw_rate", float(recent_yaw))
            self.logger.record("mc/ep_jump_rate", float(recent_jump))
            self.logger.record("mc/ep_frame_diff_mean", float(recent_diff))
            # The two metrics that tell you wood-seeking is working:
            #   ep_full_log_mean:  avg fraction of screen showing wood per
            #     step, averaged across an episode. >0 means the agent is
            #     spending time looking at trees. Climbs as policy improves.
            #   ep_full_log_max:  peak per-episode "wood on screen". Climbs
            #     fast — first sign the policy has learned to face trees.
            #   ep_approach_total: summed approach reward — climbs as the
            #     agent learns to walk toward visible wood.
            self.logger.record("mc/ep_full_log_mean", float(recent_full_mean))
            self.logger.record("mc/ep_full_log_max", float(recent_full_max))
            self.logger.record("mc/ep_approach_total", float(recent_approach))
            self.logger.record("mc/episodes_seen", len(self._ep_returns))
            self._t_last_log = now

        return True


def make_env(cfg: MinecraftRealConfig):
    def _f():
        env = MinecraftRealEnv(cfg)
        return env

    return _f


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default="mc_real_v1")
    # With action_repeat=4 + frame_time_s=0.05, 1 agent step = 200 ms wall.
    # 36k agent steps ~= 2 hr. (7200 s / 0.2 s = 36000.)
    parser.add_argument("--total-steps", type=int, default=36_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--frame-time-s", type=float, default=0.05)
    parser.add_argument("--log-confirm-steps", type=int, default=3,
                        help="Consecutive confirmed frames required before counting log acquisition.")
    parser.add_argument("--slot-diff-margin", type=float, default=2.0,
                        help="Extra margin above slot_diff_threshold for acquisition confirmation.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=4,
        help=(
            "Sticky-actions / frame skip. Each env.step() applies the chosen "
            "action this many times before reading the next observation. "
            "Larger = bigger camera sweeps per decision and faster learning, "
            "at the cost of less reactive control. Default 4 = 200 ms per "
            "agent step at 20 Hz."
        ),
    )
    parser.add_argument(
        "--yaw-step-px",
        type=int,
        default=60,
        help="Mouse dx in raw mickeys per yaw substep. Total per agent "
             "decision = action_repeat * yaw_step_px.",
    )
    parser.add_argument(
        "--pitch-step-px",
        type=int,
        default=35,
        help="Mouse dy in raw mickeys per pitch substep.",
    )
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--gui-scale", type=int, default=2)
    parser.add_argument(
        "--no-window-detect",
        action="store_true",
        help="Disable MC window auto-detection (use full monitor instead).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--countdown", type=int, default=10)
    parser.add_argument(
        "--reset-cmd",
        action="append",
        default=None,
        help=(
            "MC chat command to run after EVERY episode (can pass multiple "
            "times). Default is '/clear @s' to keep the reward signal alive "
            "(slot 1 stays clearable). Pass '' to disable."
        ),
    )
    parser.add_argument(
        "--timeout-cmd",
        action="append",
        default=None,
        help=(
            "MC chat command to run ONLY when an episode times out without "
            "a log acquired. Useful for hard-resets like teleport-back-to-spawn, "
            "e.g. '/tp @s 0 64 0' or '/kill @s'. Can pass multiple times."
        ),
    )
    parser.add_argument(
        "--allow-kill-timeout",
        action="store_true",
        help=(
            "Allow '/kill ...' timeout commands. Disabled by default because "
            "death-screen timing is the most failure-prone reset path. "
            "Without this flag, any timeout '/kill' commands are dropped."
        ),
    )
    parser.add_argument(
        "--post-reset-pause-s",
        type=float,
        default=1.5,
        help=(
            "Wall-clock seconds to wait after running reset chat commands "
            "before starting the next episode. Bump to 1.5+ if you use "
            "'/kill @s' as a timeout command — respawning takes time even "
            "with doImmediateRespawn=true."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to a saved PPO checkpoint .zip to resume from "
            "(e.g. runs/mc_real_v4_seek/final.zip). Loads the policy "
            "weights into a fresh model so reward-shaping changes take "
            "effect, but you keep the learned exploration/aim behaviors."
        ),
    )
    parser.add_argument(
        "--immortal",
        action="store_true",
        help=(
            "Make the agent un-killable except by our own /kill on timeout. "
            "Adds resistance + regeneration + saturation effects at level 4 "
            "(numeric 255 = full immunity in resistance) plus immediate "
            "respawn. Eliminates the most common cause of death-screen "
            "deadlocks: the random policy walking into lava / off cliffs / "
            "into mobs and the env not noticing."
        ),
    )
    parser.add_argument(
        "--init-cmd",
        action="append",
        default=None,
        help=(
            "MC chat command to run ONCE at training start (can pass "
            "multiple). Use for world setup like '/gamerule "
            "doImmediateRespawn true' or '/spawnpoint @s ~ ~ ~'. Anything "
            "added by --immortal is appended after these."
        ),
    )
    parser.add_argument(
        "--teacher-force-start",
        type=float,
        default=0.0,
        help=(
            "Hard-coded teacher override probability at step 0. "
            "Set >0 (e.g. 0.7) to bootstrap tree-seeking behavior."
        ),
    )
    parser.add_argument(
        "--teacher-force-end",
        type=float,
        default=0.0,
        help=(
            "Teacher override probability after decay. Typical curriculum: "
            "--teacher-force-start 0.7 --teacher-force-end 0.05"
        ),
    )
    parser.add_argument(
        "--teacher-force-decay-steps",
        type=int,
        default=10_000,
        help="Number of env steps over which teacher forcing decays.",
    )
    args = parser.parse_args()

    if args.reset_cmd is None:
        reset_cmds: tuple[str, ...] = ("/clear @s",)
    else:
        reset_cmds = tuple(c for c in args.reset_cmd if c)
    timeout_cmds: tuple[str, ...] = tuple(args.timeout_cmd or ())
    if timeout_cmds and not args.allow_kill_timeout:
        filtered = []
        dropped = []
        for cmd in timeout_cmds:
            if cmd.strip().lower().startswith("/kill"):
                dropped.append(cmd)
            else:
                filtered.append(cmd)
        if dropped:
            print("WARNING: dropping timeout kill commands (use --allow-kill-timeout to keep):")
            for cmd in dropped:
                print(f"  - {cmd}")
        timeout_cmds = tuple(filtered)

    init_cmds: list[str] = list(args.init_cmd or ())
    post_respawn_cmds: list[str] = []
    if args.immortal:
        # /kill clears all effects on respawn, so we need to re-apply the
        # buffs after EVERY timeout. But not after every regular reset
        # (when the agent successfully terminated by acquiring a log) —
        # buffs are still active there. Putting them in
        # ``post_respawn_chat_commands`` instead of
        # ``auto_reset_chat_commands`` is the difference between "5 min
        # of training" and "5 min of chat spam". The init list also runs
        # them once at the very first reset of the run.
        immortal_cmds = (
            "/gamerule doImmediateRespawn true",
            "/gamemode survival",
            "/effect give @s minecraft:resistance 999999 4 true",
            "/effect give @s minecraft:regeneration 999999 4 true",
            "/effect give @s minecraft:saturation 999999 4 true",
            "/effect give @s minecraft:fire_resistance 999999 0 true",
            "/effect give @s minecraft:water_breathing 999999 0 true",
        )
        init_cmds.extend(immortal_cmds)
        post_respawn_cmds.extend(immortal_cmds)

    cfg = MinecraftRealConfig(
        monitor_index=args.monitor_index,
        max_seconds=float(args.max_seconds),
        frame_time_s=float(args.frame_time_s),
        action_repeat=int(args.action_repeat),
        yaw_step_px=int(args.yaw_step_px),
        pitch_step_px=int(args.pitch_step_px),
        gui_scale=int(args.gui_scale),
        use_window_detection=not args.no_window_detect,
        init_chat_commands=tuple(init_cmds),
        post_respawn_chat_commands=tuple(post_respawn_cmds),
        auto_reset_chat_commands=reset_cmds,
        timeout_extra_chat_commands=timeout_cmds,
        post_reset_pause_s=float(args.post_reset_pause_s),
        log_confirm_steps=int(args.log_confirm_steps),
        slot_diff_margin=float(args.slot_diff_margin),
        teacher_force_start=float(args.teacher_force_start),
        teacher_force_end=float(args.teacher_force_end),
        teacher_force_decay_steps=int(args.teacher_force_decay_steps),
    )

    run_dir = ROOT / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    agent_step_s = cfg.frame_time_s * cfg.action_repeat
    wall_hours = args.total_steps * agent_step_s / 3600.0
    yaw_per_decision = cfg.yaw_step_px * cfg.action_repeat

    print(f"=== mc_real PPO ===")
    print(f"  run_dir         {run_dir}")
    print(f"  total_steps     {args.total_steps}  (~{wall_hours:.2f} h wall)")
    print(f"  n_steps         {args.n_steps}  (PPO rollout per iter)")
    print(f"  max_seconds     {cfg.max_seconds:.1f}  (per-episode timeout)")
    print(f"  frame_time_s    {cfg.frame_time_s:.3f}  (~{1/cfg.frame_time_s:.1f} Hz game tick)")
    print(f"  action_repeat   {cfg.action_repeat}  (1 decision = {agent_step_s*1000:.0f} ms)")
    print(f"  yaw_step_px     {cfg.yaw_step_px}  ({yaw_per_decision} mickeys/decision)")
    print(f"  pitch_step_px   {cfg.pitch_step_px}")
    print(f"  device          {args.device}")
    print(f"  monitor_index   {cfg.monitor_index}")
    print(f"  init cmds       {list(cfg.init_chat_commands) or '<none>'}")
    print(f"  reset cmds      {list(cfg.auto_reset_chat_commands) or '<none>'}")
    print(f"  post-respawn    {list(cfg.post_respawn_chat_commands) or '<none>'}")
    print(f"  timeout cmds    {list(cfg.timeout_extra_chat_commands) or '<none>'}")
    print(
        f"  acquire gate   diff>={cfg.slot_diff_threshold + cfg.slot_diff_margin:.1f}, "
        f"confirm_steps={cfg.log_confirm_steps}, "
        f"slot_log_px>={cfg.log_pixel_threshold}"
    )
    print(
        "  teacher force  "
        f"{cfg.teacher_force_start:.2f} -> {cfg.teacher_force_end:.2f} "
        f"over {cfg.teacher_force_decay_steps} steps"
    )
    print(f"  immortal mode   {bool(args.immortal)}")
    print()

    stop_file = run_dir / "STOP"
    _failsafe.install(stop_file=stop_file)
    print(_failsafe.banner(stop_file=stop_file))
    print()

    print(f"You have {args.countdown}s to focus the Minecraft window.")
    print("Click into MC. The countdown starts now. DO NOT touch input again until")
    print(f"training is done (~{wall_hours:.2f} hours).")
    print("If something goes wrong: tap F12. Keys release in <50ms.\n")
    for i in range(args.countdown, 0, -1):
        print(f"  starting in {i}s", flush=True)
        time.sleep(1.0)

    vec = DummyVecEnv([make_env(cfg)])

    model = PPO(
        "CnnPolicy",
        vec,
        verbose=1,
        tensorboard_log=str(run_dir),
        device=args.device,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
    )
    if args.resume_from:
        ckpt = Path(args.resume_from)
        if not ckpt.exists():
            print(f"WARNING: --resume-from {ckpt} not found; starting fresh.")
        else:
            print(f"loading policy weights from {ckpt}")
            # set_parameters keeps the new env + new value-function head's
            # output but restores the learned policy + feature extractor.
            model.set_parameters(str(ckpt), exact_match=False, device=args.device)

    snaps_dir = run_dir / "snaps"
    callbacks = [
        FailsafeCallback(),
        MCRealStatsCallback(),
        EpisodeJSONLCallback(
            jsonl_path=run_dir / "episodes.jsonl",
            t0=time.time(),
        ),
        LogAcquiredSnapshotCallback(out_dir=snaps_dir, max_snaps=64),
        CheckpointCallback(
            save_freq=int(args.checkpoint_every),
            save_path=str(ckpt_dir),
            name_prefix="ppo_mc",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]

    try:
        model.learn(
            total_timesteps=int(args.total_steps),
            callback=callbacks,
            tb_log_name="ppo",
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Saving final checkpoint...")
        _failsafe.request_stop("KeyboardInterrupt")
    finally:
        # Belt and braces: make sure no key is left held even if a callback
        # bug prevented the env from cleaning up.
        try:
            from mine_diamonds.input import game_input as _gi
            _gi.release_all()
        except Exception:
            pass
        final = run_dir / "final.zip"
        model.save(str(final))
        print(f"saved {final}")
        if _failsafe.is_stopping():
            print(f"stopped via failsafe: {_failsafe.stop_reason()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
