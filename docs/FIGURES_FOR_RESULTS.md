# What to put in the **Results** section of the paper

Use **exactly these** assets so you are not choosing between 20 similar PNGs in the same folder.

## Table 1 (numbers for the text)

Build one comparison table from the three markdown summaries (same headline metrics in each file):

- `docs/figs/mc_real_v14_no_kill_reset/results_summary.md`
- `docs/figs/mc_real_superflat_long_y0/results_summary.md`
- `docs/figs/mc_real_superflat_retry_bidir/results_summary.md`

(Each folder also has `episodes.csv` if you need per-episode detail.)

## Figures (in order; matches `docs/PAPER_WORKING_DRAFT.md`)

| # | File | What it shows |
|---|------|----------------|
| **1** | `docs/figs/mc_real_v14_no_kill_reset/episodes_logs.png` | No-kill run: per-episode log counts (supports the false-positive / early-detector point). |
| **2** | `docs/figs/mc_real_superflat_long_y0/cumulative_logs.png` | Long superflat run: total logs over training timesteps (main “does learning add wood?” curve). |
| **3** | `docs/figs/mc_real_superflat_retry_bidir/episodes_full_log_max.png` | Retry run: peak “wood on screen” per episode vs strict validation still failing. |
| **4** | `docs/figs/mc_real_superflat_long_y0/ppo_train.png` | PPO is actually updating: entropy, KL, value loss, explained variance. |
| **5** (optional) | `docs/figs/mc_real_superflat_long_y0/storyboard.png` | One poster-style combo (curve + example frames) if the assignment wants a “highlight” figure. |

## What you can skip for Results

- Other files in the same `docs/figs/...` folders (e.g. `episodes_return.png`, `episodes_aim_rate.png`, `cumulative_logs` in the v14 or retry folders) are for debugging or alternate angles — **not** required for the default Results set above.
- **Methodology** (what the network sees): use a local smoke capture if you have it (`eval/smoke_minecraft/` after running the smoke script — that path is usually not committed; see `README.md`).

## Regenerating the plots

```powershell
python scripts\build_paper_figs.py --run-name <run_name>
```

Output goes to `docs/figs/<run_name>/`.
