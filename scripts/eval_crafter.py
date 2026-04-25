"""Evaluate a trained Crafter PPO checkpoint.

Loads a checkpoint, rolls out N episodes, writes:
  * eval/<run_name>/rollout.mp4  (upscaled 8x for legibility; one episode)
  * eval/<run_name>/summary.json (per-achievement unlock rate across episodes)
  * eval/<run_name>/episodes.jsonl (one row per episode: return, length,
    achievements unlocked)

Usage:
    python scripts/eval_crafter.py \\
        --checkpoint checkpoints/crafter_ppo_long/ppo_crafter_final.zip \\
        --episodes 10 --run-name crafter_ppo_long

Video requires imageio + ffmpeg (installed by the project's extras).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("PYTHONUTF8", "1")

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

import mine_diamonds.envs  # noqa: F401
from mine_diamonds.envs import ACHIEVEMENTS
from mine_diamonds.envs.crafter_env import CrafterGymnasiumEnv


def upscale_nn(frame: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor upscale so 64x64 Crafter frames are legible in a video."""
    if factor == 1:
        return frame
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Crafter PPO checkpoint.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to SB3 .zip checkpoint")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--run-name", type=str, default="eval")
    p.add_argument("--deterministic", action="store_true",
                   help="Greedy action selection (default: sample from policy).")
    p.add_argument("--record-episode-index", type=int, default=0,
                   help="Which episode to record to MP4 (0-indexed).")
    p.add_argument("--upscale", type=int, default=8,
                   help="Nearest-neighbor upscale factor for video frames.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / "eval" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.checkpoint, device="auto")
    print(f"loaded {args.checkpoint} on device {model.device}")

    episodes_log_path = out_dir / "episodes.jsonl"
    summary_path = out_dir / "summary.json"
    video_path = out_dir / "rollout.mp4"

    unlock_counts = {name: 0 for name in ACHIEVEMENTS}
    episode_records: list[dict] = []

    # We re-create the env per episode to vary seeds cleanly.
    for ep in range(args.episodes):
        env = CrafterGymnasiumEnv(seed=args.seed + ep)
        obs, info = env.reset()
        done = False
        ret = 0.0
        steps = 0
        frames: list[np.ndarray] = [] if ep == args.record_episode_index else []

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(int(action))
            ret += float(r)
            steps += 1
            if ep == args.record_episode_index:
                frames.append(upscale_nn(obs, args.upscale))
            done = bool(term or trunc)

        ep_unlocked = sorted(info.get("achievements_unlocked", frozenset()))
        for name in ep_unlocked:
            unlock_counts[name] = unlock_counts.get(name, 0) + 1

        record = {
            "episode": ep,
            "return": ret,
            "length": steps,
            "achievements_unlocked": ep_unlocked,
        }
        episode_records.append(record)
        print(
            f"  ep {ep:02d}  return={ret:+7.2f}  length={steps:4d}"
            f"  achievements={len(ep_unlocked):2d} -> {ep_unlocked}"
        )

        if ep == args.record_episode_index and frames:
            imageio.mimwrite(
                video_path, frames, fps=args.fps, quality=8, macro_block_size=1
            )
            print(f"  wrote video: {video_path}")

        env.close()

    summary = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "mean_return": float(np.mean([r["return"] for r in episode_records])),
        "mean_length": float(np.mean([r["length"] for r in episode_records])),
        "mean_achievements_per_episode": float(
            np.mean([len(r["achievements_unlocked"]) for r in episode_records])
        ),
        "unlock_rate": {
            name: unlock_counts[name] / args.episodes for name in ACHIEVEMENTS
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with episodes_log_path.open("w", encoding="utf-8") as f:
        for rec in episode_records:
            f.write(json.dumps(rec) + "\n")

    print("\n=== Summary ===")
    print(f"mean_return                    {summary['mean_return']:+.3f}")
    print(f"mean_length                    {summary['mean_length']:.1f}")
    print(
        f"mean_achievements_per_episode  "
        f"{summary['mean_achievements_per_episode']:.2f}"
    )
    print("unlock rates (sorted):")
    for name, rate in sorted(
        summary["unlock_rate"].items(), key=lambda kv: -kv[1]
    ):
        bar = "#" * int(round(rate * 20))
        print(f"  {name:22s} {rate*100:5.1f}%  {bar}")
    print(f"\nwrote {summary_path}")
    print(f"wrote {episodes_log_path}")


if __name__ == "__main__":
    main()
