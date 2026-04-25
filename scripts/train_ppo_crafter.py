"""PPO on Crafter — actual reinforcement learning with a crafting tech tree.

Logs TensorBoard scalars for:
  * rollout/ep_rew_mean (standard)
  * crafter/ep_ach_count_mean  -- avg # achievements unlocked per episode
  * crafter/unlock_rate/<name> -- fraction of recent episodes that unlocked
    each specific achievement (e.g. make_wood_pickaxe, collect_diamond).

Run (short sanity check):
    python scripts/train_ppo_crafter.py --total-steps 200000 --n-envs 8

Run (overnight training, recommended):
    python scripts/train_ppo_crafter.py --total-steps 3000000 --n-envs 16 \\
        --run-name crafter_ppo_long

View TensorBoard:
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force UTF-8 for any subprocess / logging on Windows so crafter asset paths
# and TB metadata don't hit cp1252 encoding errors.
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import mine_diamonds.envs  # noqa: F401  (registers MineDiamonds/Crafter-v0)
from mine_diamonds.envs import ACHIEVEMENTS
from mine_diamonds.envs.crafter_env import CrafterGymnasiumEnv


def make_env(seed: int):
    # VecMonitor wraps the whole VecEnv, so no inner Monitor here.
    def _init():
        return CrafterGymnasiumEnv(seed=seed)

    return _init


class AchievementTBCallback(BaseCallback):
    """Track per-achievement unlock rate + episode achievement count.

    Uses a sliding window of the last `window` finished episodes across all
    vectorized envs, which matches SB3's default `rollout/ep_rew_mean` style.
    """

    def __init__(self, window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self._ep_ach_counts: deque[int] = deque(maxlen=window)
        self._ep_unlocked_masks: deque[np.ndarray] = deque(maxlen=window)
        self._index = {name: i for i, name in enumerate(ACHIEVEMENTS)}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            if not done:
                continue
            # SB3 stashes the real final info here when auto-reset happens.
            final = info.get("final_info") if "final_info" in info else info
            if final is None:
                continue
            unlocked = final.get("achievements_unlocked", frozenset())
            mask = np.zeros(len(ACHIEVEMENTS), dtype=np.float32)
            for name in unlocked:
                idx = self._index.get(name)
                if idx is not None:
                    mask[idx] = 1.0
            self._ep_ach_counts.append(int(mask.sum()))
            self._ep_unlocked_masks.append(mask)
        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_ach_counts:
            return
        self.logger.record(
            "crafter/ep_ach_count_mean",
            float(np.mean(self._ep_ach_counts)),
        )
        self.logger.record(
            "crafter/ep_ach_count_max",
            float(np.max(self._ep_ach_counts)),
        )
        stacked = np.stack(self._ep_unlocked_masks, axis=0)  # (N, 22)
        rates = stacked.mean(axis=0)
        for name, rate in zip(ACHIEVEMENTS, rates):
            self.logger.record(f"crafter/unlock_rate/{name}", float(rate))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO on Crafter.")
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default="crafter_ppo")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cuda', or 'cpu'")
    p.add_argument("--checkpoint-every", type=int, default=100_000,
                   help="Save checkpoint every N env steps (0 disables).")
    p.add_argument("--no-subproc", action="store_true",
                   help="Use DummyVecEnv (single-process) instead of SubprocVecEnv.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    ckpt_dir = ROOT / "checkpoints" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(seed=args.seed + i) for i in range(args.n_envs)]
    if args.no_subproc or args.n_envs == 1:
        venv = DummyVecEnv(env_fns)
    else:
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    venv = VecMonitor(venv)

    model = PPO(
        "CnnPolicy",
        venv,
        verbose=1,
        tensorboard_log=str(runs_dir),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        seed=args.seed,
        device=args.device,
    )

    callbacks: list[BaseCallback] = [AchievementTBCallback(window=100)]
    if args.checkpoint_every > 0:
        save_freq = max(1, args.checkpoint_every // args.n_envs)
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(ckpt_dir),
                name_prefix="ppo_crafter",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

    model.learn(
        total_timesteps=args.total_steps,
        tb_log_name=args.run_name,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    final_path = ckpt_dir / "ppo_crafter_final.zip"
    model.save(str(final_path))
    print(f"saved final checkpoint: {final_path}")
    venv.close()


if __name__ == "__main__":
    main()
