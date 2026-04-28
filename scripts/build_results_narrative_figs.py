"""Build Results-section figures directly from committed ``episodes.csv`` exports.

Each PNG is aligned to the experiment narrative: early no-kill run vs long
superflat run, with titles/subtitles using numbers computed from the same
rows (so the figure always matches the table in the CSV).

Usage (from repo root):

  .\\.venv\\Scripts\\python.exe scripts\\build_results_narrative_figs.py

Optional:

  --v14-csv  path   (default: docs/figs/mc_real_v14_no_kill_reset/episodes.csv)
  --long-csv path   (default: docs/figs/mc_real_superflat_long_y0/episodes.csv)
  --out-dir  path   (default: docs/figs/narrative)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_episodes_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if v is None:
                    continue
                try:
                    if "." in v or "e" in v.lower():
                        row[k] = float(v)
                    else:
                        row[k] = int(v)
                except (ValueError, TypeError):
                    if k == "truncated":
                        row[k] = v.lower() == "true"
            rows.append(row)
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"name": name, "n_ep": 0}
    last = rows[-1]
    logs = np.array([int(r.get("logs_acquired", 0)) for r in rows])
    ep_lens = np.array([float(r.get("ep_len", 0)) for r in rows])
    aim = np.array([float(r.get("ep_aim_rate", 0)) for r in rows])
    fmax = np.array([float(r.get("ep_full_log_max", 0)) for r in rows])
    steps = [int(r.get("global_step", 0)) for r in rows]
    elapsed_s = float(last.get("elapsed_s", 0.0))
    return {
        "name": name,
        "n_ep": n,
        "total_timesteps": int(steps[-1]) if steps else 0,
        "wall_min": elapsed_s / 60.0,
        "mean_ep_len": float(ep_lens.mean()),
        "sum_logs": int(logs.sum()),
        "episodes_with_log": int((logs > 0).sum()),
        "mean_aim": float(aim.mean()),
        "mean_full_log_max": float(fmax.mean()),
    }


def fig_early_run(out: Path, rows: list[dict], s: dict) -> None:
    """Paragraph: early 40-ep run, false positives, tightened criteria."""
    ep = np.array([int(r["episode"]) for r in rows], dtype=float)
    aim = np.array([float(r.get("ep_aim_rate", 0)) for r in rows])
    fmax = np.array([float(r.get("ep_full_log_max", 0)) for r in rows])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    fig.suptitle(
        "Early iteration (no-kill reset): behavior signals from captured logs\n"
        f"{s['n_ep']} episodes, {s['total_timesteps']:,} timesteps, "
        f"~{s['wall_min']:.1f} min wall, mean episode length {s['mean_ep_len']:.1f} steps",
        fontsize=11,
    )
    ax0.plot(ep, aim, "o-", ms=3, lw=1, color="#1f77b4")
    ax0.set_ylabel("Aim rate\n(frac. steps on log in fovea)", fontsize=9)
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, alpha=0.3)

    ax1.plot(ep, fmax, "o-", ms=3, lw=1, color="#d62728")
    ax1.set_ylabel("Peak log pixels\n(frac. of screen, full frame)", fontsize=9)
    ax1.set_xlabel("Episode", fontsize=9)
    ax1.grid(True, alpha=0.3)

    fig.text(
        0.5,
        0.01,
        "Use with text: high aim / visibility do not always equal verified hotbar wood "
        "after stricter slot-change checks.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_long_superflat(out: Path, rows: list[dict], s: dict) -> None:
    """Paragraph: long superflat, stable behavior, timesteps, cumulative wood."""
    steps = np.array([int(r.get("global_step", 0)) for r in rows], dtype=float)
    logs = np.array([int(r.get("logs_acquired", 0)) for r in rows])
    cum = np.cumsum(logs)
    ep = np.array([int(r["episode"]) for r in rows], dtype=float)
    fmax = np.array([float(r.get("ep_full_log_max", 0)) for r in rows])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=False)
    fig.suptitle(
        "Long superflat run (y=0 + respread resets): learning curve + visibility\n"
        f"{s['n_ep']} episodes, {s['total_timesteps']:,} timesteps, "
        f"~{s['wall_min']:.1f} min wall, mean episode length {s['mean_ep_len']:.1f} steps",
        fontsize=11,
    )
    ax0.plot(steps, cum, color="#2ca02c", lw=2)
    ax0.set_ylabel("Cumulative logs acquired\n(summed from episodes.csv)", fontsize=9)
    ax0.set_xlabel("PPO / agent timestep", fontsize=9)
    ax0.grid(True, alpha=0.3)

    ax1.plot(ep, fmax, "o", ms=2, alpha=0.5, color="#9467bd")
    ax1.set_ylabel("Peak wood on screen\n(per episode)", fontsize=9)
    ax1.set_xlabel("Episode", fontsize=9)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_overview(out: Path, s14: dict, s_long: dict) -> None:
    """First paragraph: what was measured, single-client constraint (text + comparison)."""
    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_axes((0.08, 0.1, 0.84, 0.8))
    ax.axis("off")
    lines = [
        "Metrics below are computed from the same per-episode CSVs exported during training",
        "(ground truth: episodes.jsonl → episodes.csv in each run folder).",
        "",
        f"Run A — early (no-kill):  {s14['n_ep']} ep  |  {s14['total_timesteps']:,} timesteps  |  "
        f"~{s14['wall_min']:.1f} min wall  |  mean len {s14['mean_ep_len']:.1f}  |  "
        f"sum logs = {s14['sum_logs']}  |  mean aim {s14['mean_aim']:.3f}  |  "
        f"mean peak wood frac {s14['mean_full_log_max']:.3f}",
        f"Run B — long (superflat):  {s_long['n_ep']} ep  |  {s_long['total_timesteps']:,} timesteps  |  "
        f"~{s_long['wall_min']:.1f} min wall  |  mean len {s_long['mean_ep_len']:.1f}  |  "
        f"sum logs = {s_long['sum_logs']}  |  mean aim {s_long['mean_aim']:.3f}  |  "
        f"mean peak wood frac {s_long['mean_full_log_max']:.3f}",
        "",
        "We logged wood acquisition, aim, log visibility, and episode return each step; "
        "PPO training curves (entropy / value loss / KL) come from TensorBoard, not the CSV.",
        "",
        "Constraint: one Minecraft window → one policy rollout at a time, so wall time grows fast.",
    ]
    y = 0.98
    for line in lines:
        ax.text(0.0, y, line, transform=ax.transAxes, va="top", fontsize=9, family="monospace")
        y -= 0.065
    fig.suptitle("Evaluation: measured quantities (from captured episode logs)", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--v14-csv",
        type=Path,
        default=ROOT / "docs" / "figs" / "mc_real_v14_no_kill_reset" / "episodes.csv",
    )
    ap.add_argument(
        "--long-csv",
        type=Path,
        default=ROOT / "docs" / "figs" / "mc_real_superflat_long_y0" / "episodes.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figs" / "narrative")
    args = ap.parse_args()

    v14 = load_episodes_csv(args.v14_csv)
    long_r = load_episodes_csv(args.long_csv)
    if not v14 or not long_r:
        print("ERROR: need both CSVs with at least one row", flush=True)
        return 1

    s14 = summarize("v14", v14)
    s_long = summarize("long", long_r)
    odir = args.out_dir

    fig_overview(odir / "results_01_overview_from_csv.png", s14, s_long)
    fig_early_run(odir / "results_02_v14_aim_and_visibility.png", v14, s14)
    fig_long_superflat(odir / "results_03_superflat_cumulative_and_visibility.png", long_r, s_long)

    print(f"Wrote figures under {odir.resolve()}")
    print("  - results_01_overview_from_csv.png")
    print("  - results_02_v14_aim_and_visibility.png")
    print("  - results_03_superflat_cumulative_and_visibility.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
