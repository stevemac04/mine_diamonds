"""Turn ``runs/<name>/episodes.jsonl`` (and TB events) into paper-grade
figures and a short results summary.

Why a custom plotter when TensorBoard exists? Because TB shows rolling
means; the paper needs RAW per-episode counts on a clean axis you can
drop into LaTeX. This script:

  * Reads ``runs/<run_name>/episodes.jsonl`` (one JSON object per finished
    episode — the ground truth).
  * Optionally reads the run's TensorBoard event file too, to overlay
    PPO training-side metrics (entropy, value loss, explained variance
    — the "the policy IS being updated" sanity proof).
  * Writes everything under ``docs/figs/<run_name>/``:

        episodes_logs.png            per-episode logs_acquired count
        episodes_aim_rate.png        per-episode fraction of steps aimed
        episodes_full_log_max.png    per-episode peak wood-on-screen
        episodes_return.png          per-episode return
        cumulative_logs.png          cumulative logs vs. global timesteps
                                     (the "RL learning curve" plot)
        ppo_train.png                4-panel: entropy, value loss,
                                     explained_var, approx_kl over time
        results_summary.md           paragraph + table for the paper
        episodes.csv                 the JSONL re-flattened to CSV

Usage:

    python scripts/build_paper_figs.py --run-name mc_real_v12_paper

    # multi-run overlay (e.g. baseline vs. with-immortal):
    python scripts/build_paper_figs.py --run-name mc_real_v12_paper \\
        --overlay mc_real_v8_robust_dismiss
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:
    EventAccumulator = None  # type: ignore[assignment]


def load_episodes(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.is_file():
        return []
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_tb_scalars(run_dir: Path) -> dict[str, dict[str, list]]:
    """Return {tag: {"step": [...], "value": [...]}} for one run dir."""
    if EventAccumulator is None:
        return {}
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for ev in run_dir.rglob("events.out.tfevents.*"):
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except (RuntimeError, OSError):
            continue
        for tag in acc.Tags().get("scalars", []):
            for s in acc.Scalars(tag):
                out[tag][int(s.step)] = float(s.value)
    final: dict[str, dict[str, list]] = {}
    for tag, m in out.items():
        steps = sorted(m.keys())
        final[tag] = {"step": steps, "value": [m[s] for s in steps]}
    return final


def write_episodes_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def lineplot(
    xs: list[float],
    ys: list[float],
    *,
    out: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    overlay: list[tuple[str, list[float], list[float]]] | None = None,
    rolling: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=140)
    ax.plot(xs, ys, label="raw", linewidth=1.0, alpha=0.45, color="C0")
    if rolling > 1 and len(ys) >= rolling:
        # Right-aligned rolling mean: each point is the mean of the
        # previous `rolling` episodes, so the curve doesn't anticipate.
        kernel = np.ones(rolling) / rolling
        smoothed = np.convolve(ys, kernel, mode="valid")
        ax.plot(
            xs[rolling - 1 :],
            smoothed,
            label=f"rolling-{rolling} mean",
            linewidth=2.4,
            color="C0",
        )
    if overlay:
        for name, ox, oy in overlay:
            ax.plot(ox, oy, label=name, linewidth=1.4, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def fourpanel_train(
    tb: dict[str, dict[str, list]], out: Path, run_name: str
) -> bool:
    tags = [
        ("train/entropy_loss", "entropy loss"),
        ("train/value_loss", "value loss"),
        ("train/explained_variance", "explained variance"),
        ("train/approx_kl", "approx KL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=140)
    plotted_any = False
    for ax, (tag, label) in zip(axes.flat, tags):
        s = tb.get(tag)
        if s is None or not s["value"]:
            ax.set_axis_off()
            continue
        ax.plot(s["step"], s["value"], linewidth=1.6, color="C1")
        ax.set_title(label)
        ax.set_xlabel("PPO timesteps")
        ax.grid(True, alpha=0.3)
        plotted_any = True
    fig.suptitle(f"{run_name}  —  PPO training-side diagnostics")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return plotted_any


def write_summary_md(
    rows: list[dict], out: Path, run_name: str, tb: dict[str, dict[str, list]]
) -> None:
    n_ep = len(rows)
    if n_ep == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"# {run_name} — no episodes recorded\n\n"
            "The episodes.jsonl was empty. Either the run died before its "
            "first episode finished, or the JSONL callback wasn't installed.\n",
            encoding="utf-8",
        )
        return
    logs = np.array([r.get("logs_acquired", 0) for r in rows])
    aim = np.array([r.get("ep_aim_rate", 0.0) for r in rows])
    full_max = np.array([r.get("ep_full_log_max", 0.0) for r in rows])
    returns = np.array([r.get("ep_return", 0.0) for r in rows])
    lengths = np.array([r.get("ep_len", 0) for r in rows])
    success = (logs > 0).astype(np.float32)
    last_step = int(rows[-1].get("global_step", 0))
    elapsed = float(rows[-1].get("elapsed_s", 0.0))
    halves = max(1, n_ep // 2)
    early_succ = float(success[:halves].mean()) if halves > 0 else 0.0
    late_succ = float(success[-halves:].mean()) if halves > 0 else 0.0

    lines: list[str] = []
    lines.append(f"# {run_name} — results")
    lines.append("")
    lines.append(
        f"PPO + CnnPolicy on the real Minecraft Java client. "
        f"{n_ep} episodes over {last_step:,} agent timesteps "
        f"(~{elapsed/60:.1f} min wall-clock). "
        f"Each episode is capped at 60 s in-game; success = at least one "
        f"log mined under policy."
    )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| episodes | {n_ep} |")
    lines.append(f"| total timesteps | {last_step:,} |")
    lines.append(f"| wall-clock minutes | {elapsed/60:.1f} |")
    lines.append(f"| total logs mined under policy | **{int(logs.sum())}** |")
    lines.append(
        f"| success rate (logs ≥ 1) | **{success.mean()*100:.1f}%** "
        f"({int(success.sum())}/{n_ep}) |"
    )
    lines.append(
        f"| success rate, first half | {early_succ*100:.1f}% |"
    )
    lines.append(
        f"| success rate, last half  | {late_succ*100:.1f}%  "
        f"(Δ {(late_succ-early_succ)*100:+.1f} pp) |"
    )
    lines.append(f"| mean ep aim rate | {aim.mean()*100:.1f}% |")
    lines.append(f"| mean ep full-log-max | {full_max.mean():.3f} |")
    lines.append(f"| mean ep return | {returns.mean():.2f} |")
    lines.append(f"| mean ep length (agent steps) | {lengths.mean():.1f} |")
    lines.append("")

    # PPO sanity: did the policy update at all?
    train_tags = {
        "entropy_loss": tb.get("train/entropy_loss", {}),
        "value_loss":   tb.get("train/value_loss", {}),
        "explained_variance": tb.get("train/explained_variance", {}),
        "approx_kl":    tb.get("train/approx_kl", {}),
    }
    lines.append("## Policy update diagnostics (PPO internals)")
    lines.append("")
    lines.append(
        "These are the smoking-gun \"PPO is actually updating the policy\" "
        "metrics. We report first vs. last logged value:"
    )
    lines.append("")
    lines.append("| metric | first | last | Δ |")
    lines.append("|---|---|---|---|")
    for name, s in train_tags.items():
        vals = s.get("value", [])
        if not vals:
            lines.append(f"| {name} | — | — | — |")
            continue
        first = vals[0]
        last_v = vals[-1]
        lines.append(
            f"| {name} | {first:+.4f} | {last_v:+.4f} | "
            f"{last_v - first:+.4f} |"
        )
    lines.append("")
    lines.append("## How to read these")
    lines.append("")
    lines.append(
        "* `cumulative_logs.png` is the headline RL plot: total logs mined "
        "under policy as a function of agent timesteps. A monotonically "
        "rising line is RL working."
    )
    lines.append(
        "* `episodes_logs.png` and `episodes_aim_rate.png` are per-episode "
        "raw counts (no rolling mean) — paper-grade scatter."
    )
    lines.append(
        "* `episodes_full_log_max.png` is *peak fraction of screen showing "
        "wood per episode*. This is the cleanest signal of \"the agent "
        "learned to look at trees\" before it learned to mine them."
    )
    lines.append(
        "* `ppo_train.png` is the policy-update sanity panel: entropy "
        "decay = exploring less, value loss → 0 = value head fitting, "
        "explained_var rising = value head explaining returns, KL "
        "bounded by clip = updates aren't blowing up."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--out-dir", default=None,
                    help="default: docs/figs/<run-name>/")
    ap.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Other run names to overlay on the cumulative-logs plot.",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / args.run_name
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} not found", file=sys.stderr)
        return 1
    out_dir = (
        Path(args.out_dir) if args.out_dir else ROOT / "docs" / "figs" / args.run_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_episodes(run_dir / "episodes.jsonl")
    tb = load_tb_scalars(run_dir)
    write_episodes_csv(rows, out_dir / "episodes.csv")

    if not rows:
        print(
            f"WARN: no episodes.jsonl rows under {run_dir}. "
            f"Did the run finish at least one episode?"
        )
        write_summary_md(rows, out_dir / "results_summary.md", args.run_name, tb)
        return 0

    eps = [r["episode"] for r in rows]
    logs = [r["logs_acquired"] for r in rows]
    aim = [r["ep_aim_rate"] for r in rows]
    full_max = [r["ep_full_log_max"] for r in rows]
    returns = [r["ep_return"] for r in rows]
    steps = [r["global_step"] for r in rows]
    cum = np.cumsum(logs).tolist()

    overlay_cum: list[tuple[str, list[float], list[float]]] = []
    for ov in args.overlay:
        ov_rows = load_episodes(runs_dir / ov / "episodes.jsonl")
        if not ov_rows:
            continue
        ov_steps = [r["global_step"] for r in ov_rows]
        ov_logs = [r["logs_acquired"] for r in ov_rows]
        overlay_cum.append((ov, ov_steps, np.cumsum(ov_logs).tolist()))

    lineplot(
        eps, logs,
        out=out_dir / "episodes_logs.png",
        title=f"{args.run_name}: logs acquired per episode",
        xlabel="episode #", ylabel="logs acquired",
        rolling=10,
    )
    lineplot(
        eps, aim,
        out=out_dir / "episodes_aim_rate.png",
        title=f"{args.run_name}: aim rate per episode",
        xlabel="episode #", ylabel="fraction of steps aimed at log",
        rolling=10,
    )
    lineplot(
        eps, full_max,
        out=out_dir / "episodes_full_log_max.png",
        title=f"{args.run_name}: peak wood-on-screen per episode",
        xlabel="episode #", ylabel="peak fraction of screen = log pixels",
        rolling=10,
    )
    lineplot(
        eps, returns,
        out=out_dir / "episodes_return.png",
        title=f"{args.run_name}: episode return",
        xlabel="episode #", ylabel="return (sum of step rewards)",
        rolling=10,
    )
    lineplot(
        steps, cum,
        out=out_dir / "cumulative_logs.png",
        title=f"{args.run_name}: cumulative logs mined under policy",
        xlabel="PPO timesteps", ylabel="cumulative logs",
        overlay=overlay_cum or None,
    )
    fourpanel_train(tb, out_dir / "ppo_train.png", args.run_name)
    write_summary_md(rows, out_dir / "results_summary.md", args.run_name, tb)

    print(f"wrote artifacts to {out_dir.resolve()}")
    print(f"  - episodes:        {len(rows)}")
    print(f"  - total logs:      {sum(logs)}")
    print(f"  - success rate:    {sum(1 for x in logs if x > 0)/len(rows)*100:.1f}%")
    print(f"  - last timestep:   {steps[-1] if steps else 0:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
