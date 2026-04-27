# Data for papers and slides

This repo is built so you can **regenerate** figures from training runs
without hand-copying TensorBoard.

## What gets produced during training

For each `scripts/train_ppo_minecraft_real.py` run, under `runs/<name>/`:

| Artifact | Purpose |
|----------|---------|
| `ppo/` TensorBoard event files | Standard SB3 + custom `mc/*` scalars |
| `episodes.jsonl` | **Per-episode ground truth** (return, `logs_acquired`, aim rate, length, etc.) — best for tables and scatter plots |
| `snaps/log_*.png` | Full-resolution captures when a log is acquired under policy — figure panels |
| `checkpoints/ppo_mc_*.zip` | Interim policies |
| `final.zip` | Policy at end of run (or after F12 stop) |

`runs/` is typically **gitignored**; keep a zip of your best run for submission archives.

## Regenerate talk-ready charts

From repo root, after at least one run exists:

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_metrics.py
```

Writes `docs/metrics/summary.md`, `docs/metrics/runs.csv`, and
`docs/metrics/mc_*.png` (compare runs at a glance).

For **one run** with paper-style curves + `results_summary.md`:

```powershell
.\.venv\Scripts\python.exe scripts\build_paper_figs.py --run-name mc_real_v12_paper
```

Output: `docs/figs/<run-name>/` (cumulative logs, per-episode metrics,
PPO diagnostics, CSV).

For a single high-impact visual combining learning curve + example
acquisition frames:

```powershell
.\.venv\Scripts\python.exe scripts\build_storyboard.py --run-name <run-name>
```

Output: `docs/figs/<run-name>/storyboard.png`

## Slides and narrative

- `DEMO.md` — live demo checklist (scripted + optional RL monitoring)
- `docs/PAPER_RESULTS_ASSETS.md` — figure/table index for report and slides

## TensorBoard (live)

```powershell
tensorboard --logdir runs --port 6006
```

Use a **second display or phone** so the Minecraft window keeps focus
while training.
