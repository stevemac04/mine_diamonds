"""Build a presentation-ready storyboard figure for one training run.

This creates ONE visual that is easy to show in class:

  docs/figs/<run_name>/storyboard.png

Layout:
  - Top: cumulative logs vs PPO timesteps (from episodes.jsonl)
  - Bottom: sampled log-acquisition snapshots from runs/<run>/snaps/
            (with step labels), showing how behavior looked through time.

Usage:
  python scripts/build_storyboard.py --run-name mc_real_v14_no_kill_reset
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SNAP_RE = re.compile(r"log_\d+_step(\d+)\.png$", re.IGNORECASE)


def load_episodes(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def snap_step(path: Path) -> int:
    m = SNAP_RE.search(path.name)
    return int(m.group(1)) if m else -1


def sample_snapshots(snaps: list[Path], k: int) -> list[Path]:
    if not snaps:
        return []
    if len(snaps) <= k:
        return snaps
    idx = np.linspace(0, len(snaps) - 1, num=k).round().astype(int)
    return [snaps[i] for i in idx]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--out-dir", default=None,
                   help="default: docs/figs/<run-name>/")
    p.add_argument("--n-snaps", type=int, default=6,
                   help="Number of snapshots to place in storyboard.")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / args.run_name
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "docs" / "figs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_episodes(run_dir / "episodes.jsonl")
    if not rows:
        print(f"No episodes.jsonl found for {args.run_name}")
        return 1

    steps = np.array([int(r.get("global_step", 0)) for r in rows], dtype=np.int64)
    logs = np.array([int(r.get("logs_acquired", 0)) for r in rows], dtype=np.int64)
    cum_logs = np.cumsum(logs)

    snaps_all = sorted((run_dir / "snaps").glob("log_*.png"), key=snap_step)
    snaps = sample_snapshots(snaps_all, max(1, int(args.n_snaps)))

    cols = max(3, len(snaps))
    fig = plt.figure(figsize=(3.8 * cols, 9.0), dpi=140)
    gs = fig.add_gridspec(2, cols, height_ratios=[1.1, 1.0], hspace=0.28, wspace=0.08)

    # Top chart spans all columns.
    ax = fig.add_subplot(gs[0, :])
    ax.plot(steps, cum_logs, color="#1f77b4", linewidth=2.6)
    ax.scatter(steps, cum_logs, s=8, alpha=0.45, color="#1f77b4")
    ax.set_title(f"{args.run_name}: cumulative logs mined under policy")
    ax.set_xlabel("PPO timesteps")
    ax.set_ylabel("cumulative logs")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.01,
        0.98,
        f"Episodes: {len(rows)}   Total logs: {int(cum_logs[-1])}   "
        f"Success rate: {float((logs > 0).mean() * 100):.1f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.7"),
    )

    # Bottom row: snapshots
    if snaps:
        for i in range(cols):
            ax_i = fig.add_subplot(gs[1, i])
            if i < len(snaps):
                img = plt.imread(str(snaps[i]))
                ax_i.imshow(img)
                st = snap_step(snaps[i])
                ax_i.set_title(f"acq @ step {st:,}", fontsize=9)
            else:
                ax_i.text(0.5, 0.5, "no snapshot", ha="center", va="center", fontsize=9)
            ax_i.axis("off")
    else:
        ax_none = fig.add_subplot(gs[1, :])
        ax_none.text(
            0.5,
            0.5,
            "No snapshots found in runs/<run>/snaps/.\n"
            "Enable/keep log-acquisition snapshot callback during training.",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax_none.axis("off")

    fig.suptitle("Minecraft RL Training Storyboard", fontsize=16, y=0.98)
    out = out_dir / "storyboard.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

