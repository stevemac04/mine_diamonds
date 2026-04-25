"""Sanity-check the grid env (spaces + short random rollout)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym

import mine_diamonds.envs  # noqa: F401 — registers MineDiamonds/SimpleGather-v0
from mine_diamonds.envs.simple_gather import SimpleGatherEnv


def main() -> None:
    env = gym.make("MineDiamonds/SimpleGather-v0", grid_size=7, max_episode_steps=50, render_mode="ansi")
    assert env.observation_space.contains(env.reset(seed=0)[0])

    obs, _ = env.reset(seed=0)
    total = 0.0
    for t in range(30):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            obs, _ = env.reset()
        if t == 0:
            txt = env.render()
            if txt:
                print(txt)
    env.close()
    print("smoke ok, sample return ~", round(total, 3))


if __name__ == "__main__":
    main()
