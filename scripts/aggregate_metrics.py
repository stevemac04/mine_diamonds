"""Aggregate TensorBoard event files across all runs/ subdirs into a
single set of presentation-ready artifacts:

  docs/metrics/summary.md   markdown table, one row per run, with per-run
                            peaks (max ep_logs_mean, max ep_aim_rate, etc.)
                            and how many steps it ran for. Easy to drop
                            into the lightning-talk slides.
  docs/metrics/runs.csv     long-form CSV (run, step, tag, value) — the
                            same data, pivot however you want in pandas /
                            Excel.
  docs/metrics/<tag>.png    one chart per scalar tag, all runs overlaid,
                            for slide screenshots. Saved at 1280x720 so
                            they fit a 16:9 slide cleanly.

Why a custom aggregator instead of "just look at TensorBoard"? Because
the talk needs ONE PNG you can drop into a slide, not a live dashboard
behind window-focus issues mid-presentation. And the markdown table
gives you the quote-able numbers without scrubbing.

Usage (from repo root):

    .\\.venv\\Scripts\\python.exe scripts\\aggregate_metrics.py

Optional flags:

    --runs-dir runs           where to find runs/<name>/ppo_*/events.*
    --out-dir  docs/metrics   where to write artifacts
    --skip     mc_real_v3_jump,mc_real_v4_seek
                              comma-list of run names to ignore (e.g.
                              early scaffolding runs that aren't worth
                              talking about).

This script is read-only on the runs/ tree; it only WRITES to --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# tensorboard ships with stable-baselines3 already, so this import is
# free for anyone who has the project's runtime deps.
try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError as exc:  # pragma: no cover
    print(
        "ERROR: tensorboard isn't importable. Install it with "
        "`pip install tensorboard` (it's a transitive dep of "
        "stable-baselines3 anyway).",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

# matplotlib is the lone dep we need that isn't already pulled in by
# the rest of the project.
try:
    import matplotlib

    matplotlib.use("Agg")  # no display required
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]


# Scalars worth charting. Order matters — the markdown summary table
# uses the same order. Keep this short; a slide can absorb 4-5 charts,
# not 15.
HEADLINE_TAGS: tuple[str, ...] = (
    "mc/ep_logs_mean",
    "mc/ep_aim_rate",
    "mc/ep_full_log_max",
    "mc/ep_approach_total",
    "mc/ep_return_mean",
    "mc/episodes_seen",
)

# Pretty labels for the markdown table header.
TAG_LABELS: dict[str, str] = {
    "mc/ep_logs_mean":      "max logs/ep",
    "mc/ep_aim_rate":       "max aim rate",
    "mc/ep_full_log_max":   "max wood-on-screen",
    "mc/ep_approach_total": "max approach",
    "mc/ep_return_mean":    "max return",
    "mc/episodes_seen":     "episodes",
}


@dataclass
class RunSeries:
    """One scalar tag from one run, as parallel step + value lists."""

    run: str
    tag: str
    steps: list[int]
    values: list[float]

    @property
    def max_value(self) -> float | None:
        return max(self.values) if self.values else None

    @property
    def last_step(self) -> int | None:
        return self.steps[-1] if self.steps else None


def discover_runs(runs_dir: Path) -> list[Path]:
    """Return run directories that contain at least one TB event file.

    A "run" here is the immediate child of ``runs_dir``. We don't recurse
    further, because PPO writes events under ``runs/<name>/ppo_*/`` and we
    treat all of those as belonging to ``<name>``.
    """
    if not runs_dir.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(child.rglob("events.out.tfevents.*")):
            out.append(child)
    return out


def load_run(run_dir: Path) -> dict[str, RunSeries]:
    """Read every event file under ``run_dir`` and merge the scalar
    streams across them by tag. Multiple ``ppo_*`` subdirs (which happen
    when training resumes) get concatenated by step.
    """
    by_tag: dict[str, dict[int, float]] = defaultdict(dict)
    for ev_path in sorted(run_dir.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(
            str(ev_path),
            size_guidance={"scalars": 0},  # 0 = load everything
        )
        try:
            acc.Reload()
        except (RuntimeError, OSError):
            # Corrupt event file — skip rather than aborting the whole
            # aggregation. We'd rather show partial metrics than none.
            continue
        for tag in acc.Tags().get("scalars", []):
            for ev in acc.Scalars(tag):
                # Last-write-wins on duplicate steps (ppo_2 overwriting
                # ppo_1 mid-step): the most recent write reflects the
                # most recent state of the policy.
                by_tag[tag][int(ev.step)] = float(ev.value)
    out: dict[str, RunSeries] = {}
    for tag, step_to_val in by_tag.items():
        steps = sorted(step_to_val.keys())
        values = [step_to_val[s] for s in steps]
        out[tag] = RunSeries(
            run=run_dir.name, tag=tag, steps=steps, values=values
        )
    return out


def write_csv(run_data: dict[str, dict[str, RunSeries]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["run", "tag", "step", "value"])
        for run, by_tag in run_data.items():
            for tag, series in sorted(by_tag.items()):
                for step, value in zip(series.steps, series.values):
                    w.writerow([run, tag, step, f"{value:.6g}"])


def write_summary_md(
    run_data: dict[str, dict[str, RunSeries]], path: Path
) -> None:
    """Markdown table summarizing each run.

    One row per run; columns are the per-tag peak (max) values plus
    final-step counters. Designed to drop straight into a slide.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Training run summary")
    lines.append("")
    lines.append(
        "Auto-generated by `scripts/aggregate_metrics.py`. One row per "
        "run; values are the peak observed for each scalar across the "
        "full TB stream (so a brief flash of success counts)."
    )
    lines.append("")
    headers = ["run", "steps"] + [TAG_LABELS[t] for t in HEADLINE_TAGS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for run in sorted(run_data.keys()):
        by_tag = run_data[run]
        # Highest step seen across any logged scalar = how far this run
        # actually trained, regardless of which scalar we look at.
        last_step = 0
        for series in by_tag.values():
            if series.last_step is not None and series.last_step > last_step:
                last_step = series.last_step
        cells = [f"`{run}`", f"{last_step:,}"]
        for tag in HEADLINE_TAGS:
            series = by_tag.get(tag)
            if series is None or series.max_value is None:
                cells.append("—")
            else:
                v = series.max_value
                if tag == "mc/episodes_seen":
                    cells.append(f"{int(v)}")
                else:
                    cells.append(f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## What to read")
    lines.append("")
    lines.append(
        "* **max logs/ep** is the headline. ~1.0 means the agent mined a "
        "log in at least one of the last-10 episodes the callback "
        "averaged. Several runs hit this briefly; only v11 (with `--immortal`) "
        "sustained it across ~25 episodes before MC lost focus."
    )
    lines.append(
        "* **max aim rate** is the share of steps the agent was looking "
        "directly at log pixels. This is the prerequisite skill for "
        "mining; it climbs first."
    )
    lines.append(
        "* **max wood-on-screen** is the per-episode peak fraction of "
        "the screen filled by log-colored pixels. Stuck >0.4 with "
        "no other progress = the env is capturing a frozen frame "
        "(see v11 post-iter-3 — that's when MC lost focus)."
    )
    lines.append(
        "* **steps** is total PPO timesteps. Each step is `action_repeat=4` "
        "low-level actions at 20 Hz, so 1k steps ≈ 200 seconds wall-clock."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_charts(
    run_data: dict[str, dict[str, RunSeries]], out_dir: Path
) -> None:
    """One PNG per HEADLINE_TAG with all runs overlaid."""
    if plt is None:
        print(
            "  [warn] matplotlib not installed; skipping charts. "
            "`pip install matplotlib` to enable."
        )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in HEADLINE_TAGS:
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        any_data = False
        for run in sorted(run_data.keys()):
            series = run_data[run].get(tag)
            if series is None or not series.values:
                continue
            ax.plot(
                series.steps, series.values, label=run, linewidth=1.6
            )
            any_data = True
        if not any_data:
            plt.close(fig)
            continue
        ax.set_title(f"{tag} across runs")
        ax.set_xlabel("PPO timesteps")
        ax.set_ylabel(TAG_LABELS.get(tag, tag))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        safe = tag.replace("/", "_").replace(" ", "_")
        png = out_dir / f"{safe}.png"
        fig.savefig(png)
        plt.close(fig)
        print(f"  wrote {png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/metrics"))
    ap.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated run names to skip (e.g. early scaffolding "
        "runs with no useful data).",
    )
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    run_dirs = [
        d for d in discover_runs(args.runs_dir) if d.name not in skip
    ]
    if not run_dirs:
        print(
            f"No runs with TB events found under {args.runs_dir}. "
            f"Nothing to aggregate."
        )
        return 1

    print(f"aggregating {len(run_dirs)} run(s) from {args.runs_dir}/")
    run_data: dict[str, dict[str, RunSeries]] = {}
    for d in run_dirs:
        by_tag = load_run(d)
        run_data[d.name] = by_tag
        n_scalars = sum(len(s.values) for s in by_tag.values())
        print(f"  {d.name:32s}  {len(by_tag):2d} tags, {n_scalars:5d} scalar points")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_md(run_data, args.out_dir / "summary.md")
    write_csv(run_data, args.out_dir / "runs.csv")
    write_charts(run_data, args.out_dir)
    print(f"\nartifacts in {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
