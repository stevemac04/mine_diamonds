# mine-diamonds

**PPO + screen capture on the real Minecraft Java client:** find wood, aim, mine a log; crafting the wooden pickaxe is a **scripted** recipe-book sequence (same vision + input stack).

| Doc | What it’s for |
|-----|----------------|
| [`DEMO.md`](DEMO.md) | On-stage checklist (smoke test + scripted demo) |
| [`docs/PRESENTATION_DATA.md`](docs/PRESENTATION_DATA.md) | **Figures & JSONL for papers/slides** — what to export, how to regenerate |
| [`docs/PAPER_RESULTS_ASSETS.md`](docs/PAPER_RESULTS_ASSETS.md) | Ready-to-use figure/table list for the final report |

```
RL (real MC)                    Scripted (after)
-----------------------------   ----------------------------------
spawn → seek tree → mine log    inv → planks → table → wooden pickaxe
```

## Repository layout

```
scripts/
  smoke_minecraft_real.py       Capture + input + reward smoke test
  train_ppo_minecraft_real.py   PPO on real MC (TB + episodes.jsonl + snaps)
  scripted_tree_chop.py         Deterministic tree find + mine (+ optional craft)
  craft_after_rl.py             Recipe-book craft only
  calibrate_inventory.py        GUI click calibration
  aggregate_metrics.py          runs/ → docs/metrics/ (compare runs)
  build_paper_figs.py           One run → docs/figs/<name>/ paper plots
src/mine_diamonds/
  envs/minecraft_real.py        Gymnasium env (capture, reward, reset)
  capture.py                    Find + focus MC window
  input/                        Windows SendInput + chat paste
  vision/pack_colors.py         BGR ranges for log detection
  scripted/                     Recipe-book choreography
  failsafe.py                   F12 / STOP file — emergency stop
assets/texture_pack/README.md   Resource pack expectations (flat-color logs)
docs/metrics/                   Regenerated TB summary + charts (commit or copy for submission)
runs/                           Per-run logs (usually gitignored — back up your best run)
eval/                           Smoke/calibration screenshots (gitignored)
archive/                        Old experiments (not part of the default workflow)
```

## Install (Windows + GPU)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[minecraft-demo]"

# RTX 50-series may need the cu128 PyTorch wheel:
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
```

## Before you train (once)

1. **Resource pack:** flat-color / simple pack so logs read as dark BGR (see [`assets/texture_pack/README.md`](assets/texture_pack/README.md)).
2. **Minecraft:** cheats on, **GUI scale 2**, **windowed**, peaceful or safe world, **slot 1 empty**, stand near trees.
3. **Smoke test:** `python scripts\smoke_minecraft_real.py --countdown 5` — confirm `window:` capture and hotbar ROI on `eval/smoke_minecraft/capture_initial.png`.

## Training (real Minecraft — keep MC focused)

```powershell
$env:PYTHONUTF8=1
python scripts\train_ppo_minecraft_real.py --run-name my_run --total-steps 18000 --immortal --countdown 25
```

- **F12** stops safely; a `STOP` file in the run dir does the same.
- Use **TensorBoard on another device** (`tensorboard --logdir runs --port 6006`) so you do not steal focus from Minecraft.

Per-run outputs: TensorBoard, `episodes.jsonl`, `snaps/` on log acquisition — see [`docs/PRESENTATION_DATA.md`](docs/PRESENTATION_DATA.md).

## Regenerate figures for a paper or deck

```powershell
python scripts\aggregate_metrics.py
python scripts\build_paper_figs.py --run-name my_run
```

## Limits

- One env, ~3–5 agent steps per second of wall time — long runs are normal.
- Reward signal is **texture-pack and ROI dependent**; re-tune `pack_colors.py` if your pack differs.
- **Do not** alt-tab during training; the env can otherwise capture the wrong window (a watchdog helps, but focus is still required).
