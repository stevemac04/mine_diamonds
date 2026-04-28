# mine-diamonds

PPO on real Minecraft Java: window capture in, `SendInput` out.  
Code: `src/mine_diamonds/`. Training: `scripts/train_ppo_minecraft_real.py`.

## Quick map

| | |
|---|---|
| Env + rewards | `src/mine_diamonds/envs/minecraft_real.py` |
| Train | `scripts/train_ppo_minecraft_real.py` |
| Figs tied to your Results text (from committed `episodes.csv`) | `python scripts/build_results_narrative_figs.py` → `docs/figs/narrative/` |
| Regenerate per-run charts from a local `runs/<name>/` | `python scripts/build_paper_figs.py --run-name <name>` |
| What’s in `docs/figs/` | `docs/figs/README.txt` |
| Texture pack | `assets/texture_pack/README.md` |
| On-stage demo checklist | `DEMO.md` |
| Check hotbar + fovea on your resolution (one PNG) | `python scripts\window_fit_check.py` |
| Full smoke (inputs, reward) | `python scripts\smoke_minecraft_real.py` |

`runs/` and `eval/` are gitignored (training output and local smoke captures). `docs/metrics/` is gitignored too (`aggregate_metrics.py` writes there if you use it).

## Install (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[minecraft-demo]"
```

## Train (keep Minecraft focused; F12 stops)

```powershell
$env:PYTHONUTF8=1
python scripts\train_ppo_minecraft_real.py --run-name my_run --total-steps 18000 --immortal --countdown 25
```
