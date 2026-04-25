"""Sanity-check the Crafter Gymnasium adapter (spaces + short random rollout)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym
import numpy as np

import mine_diamonds.envs  # noqa: F401  (registers MineDiamonds/Crafter-v0)
from mine_diamonds.envs import ACHIEVEMENTS


def main() -> None:
    env = gym.make("MineDiamonds/Crafter-v0")
    obs, info = env.reset(seed=0)

    assert env.observation_space.shape == (64, 64, 3)
    assert env.observation_space.dtype == np.uint8
    assert env.action_space.n == 17
    assert env.observation_space.contains(obs), "reset obs outside obs_space"
    print(f"spaces ok: obs {env.observation_space}, act {env.action_space}")

    rng = np.random.default_rng(0)
    total_reward = 0.0
    unlocked: set[str] = set()
    action_counter: Counter[int] = Counter()

    for t in range(1500):
        a = int(rng.integers(0, env.action_space.n))
        obs, r, term, trunc, info = env.step(a)
        total_reward += float(r)
        action_counter[a] += 1
        unlocked |= set(info.get("achievements_unlocked", set()))
        if term or trunc:
            obs, info = env.reset()
    env.close()

    print(f"random rollout: 1500 steps, return {total_reward:+.2f}")
    print(f"random policy unlocked {len(unlocked)}/{len(ACHIEVEMENTS)} achievements:")
    for name in sorted(unlocked):
        print(f"  - {name}")
    print("smoke ok")


if __name__ == "__main__":
    main()
