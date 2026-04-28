# mine-diamonds

PPO on real Minecraft Java: window capture in, `SendInput` out.  
Code: `src/mine_diamonds/`. Training: `scripts/train_ppo_minecraft_real.py`.

## Quick map

| | |
|---|---|
| Env + rewards | `src/mine_diamonds/envs/minecraft_real.py` |
| Train | `scripts/train_ppo_minecraft_real.py` |
| Regenerate figs from a run | `python scripts/build_paper_figs.py --run-name <name>` (reads `runs/<name>/`, writes `docs/figs/<name>/`) |
| Example exported plots (from past runs) | `docs/figs/` — see `docs/figs/README.txt` for which PNGs match what |
| Texture pack | `assets/texture_pack/README.md` |
| On-stage demo checklist | `DEMO.md` |

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
