# mine-diamonds

Real-Minecraft RL (PPO) that learns to get wood from screen pixels + synthetic input.

## Start Here (Quick Path)

If you only review 10 files/artifacts, review these:

1. **Main environment + reward shaping (core method):** `src/mine_diamonds/envs/minecraft_real.py`
2. **Training entrypoint (PPO setup, callbacks, run config):** `scripts/train_ppo_minecraft_real.py`
3. **Failsafe + input safety for live game control:** `src/mine_diamonds/failsafe.py`
4. **Color detector assumptions (resource pack range):** `src/mine_diamonds/vision/pack_colors.py`
5. **Capture + MC window detection:** `src/mine_diamonds/capture.py`
6. **Input backend (`SendInput`) wrapper:** `src/mine_diamonds/input/game_input.py`
7. **Smoke test script (end-to-end sanity):** `scripts/smoke_minecraft_real.py`
8. **Results figure 1:** `docs/figs/narrative/results_01_overview_from_csv.png`
9. **Results figure 2:** `docs/figs/narrative/results_02_v14_aim_and_visibility.png`
10. **Results figure 3:** `docs/figs/narrative/results_03_superflat_cumulative_and_visibility.png`

## Successful Run Video

- **Wood acquisition run (recorded):** [WOOD!.mp4 (Google Drive)](https://drive.google.com/file/d/1X8ne4xbKFtzojJPiqBfAn4OejCizwwOO/view?usp=sharing)

## Key RL Method Notes

- **Algorithm:** PPO with `CnnPolicy` on a single real-game environment (no simulator shortcuts).
- **Observation:** live Minecraft window capture (`mss`) downscaled to policy input.
- **Action space:** discrete control over forward/yaw/pitch/attack/jump combinations.
- **Reward design:** dense shaping for tree visibility/approach/aim/commitment + sparse `log_acquired`.
- **Real-game constraints handled:** death-screen dismissal, reset commands, watchdog for lost window focus, failsafe emergency stop.

## Run It (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[minecraft-demo]"
```

```powershell
$env:PYTHONUTF8=1
python scripts\train_ppo_minecraft_real.py --run-name my_run --total-steps 18000 --immortal --countdown 12
```

## Rebuild Narrative Figures

```powershell
python scripts\build_results_narrative_figs.py
```

Outputs:

- `docs/figs/narrative/results_01_overview_from_csv.png`
- `docs/figs/narrative/results_02_v14_aim_and_visibility.png`
- `docs/figs/narrative/results_03_superflat_cumulative_and_visibility.png`
