# mine-diamonds (DS340 / Minecraft RL)

PPO in **real** Minecraft Java: screen capture in, SendInput out. The write-up
and the plots that go in the **Results** section live under `docs/`.

## Grader: start here

| What | Where |
|------|--------|
| **Paper draft** | `docs/PAPER_WORKING_DRAFT.md` |
| **Which figures go in Results** | `docs/FIGURES_FOR_RESULTS.md` |
| **RL environment + rewards** | `src/mine_diamonds/envs/minecraft_real.py` |
| **Train** | `scripts/train_ppo_minecraft_real.py` |
| **Smoke test (capture + ROI)** | `scripts/smoke_minecraft_real.py` (writes to `eval/` — not in git) |
| **Scripted tree chop (non-RL demo)** | `scripts/scripted_tree_chop.py` |
| **Texture pack note** | `assets/texture_pack/README.md` |

`docs/figs/<run>/` has PNGs + `results_summary.md` + `episodes.csv` per
experiment. You only need the files listed in `FIGURES_FOR_RESULTS.md` unless
you want the extras for context.

`runs/` and `docs/metrics/` are **gitignored** (training output and
multi-run TensorBoard scrapes). Regenerate local charts with
`scripts/aggregate_metrics.py` if you want them.

`DEMO.md` is a short on-stage checklist. `docs/PRESENTATION_DATA.md` explains
TensorBoard + `episodes.jsonl` if someone is reproducing from scratch.

## Install (Windows, GPU optional)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[minecraft-demo]"
# RTX 50-series: may need PyTorch cu128 build from pytorch.org
```

## Train (Minecraft must stay focused; F12 = stop)

```powershell
$env:PYTHONUTF8=1
python scripts\train_ppo_minecraft_real.py --run-name my_run --total-steps 18000 --immortal --countdown 25
```

## Regenerate paper figures for one run

```powershell
python scripts\build_paper_figs.py --run-name <run_name>
```

## Repo layout (short)

- `src/mine_diamonds/` — env, capture, input, vision, failsafe  
- `scripts/` — training, metrics, demos  
- `docs/` — paper text + committed figure exports  
- `assets/texture_pack/` — what the vision pipeline expects
