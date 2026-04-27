# Paper Results Assets

Use this page to quickly insert figures/tables into the final report.

## Figure candidates

1. **Agent visual input + fovea region**
   - File: `eval/smoke_minecraft/capture_after_input.png`
   - Purpose: shows exactly what the model sees from the game window.
   - Suggested caption: "Captured Minecraft frame used as RL observation, with center fovea region used for aim-related reward shaping."

2. **Cumulative logs mined under policy**
   - File: `docs/figs/mc_real_v14_no_kill_reset/cumulative_logs.png`
   - Purpose: headline learning/performance curve.
   - Suggested caption: "Cumulative logs collected versus PPO timesteps for the no-kill-reset run."

3. **Episode aim rate over time**
   - File: `docs/figs/mc_real_v14_no_kill_reset/episodes_aim_rate.png`
   - Purpose: demonstrates visual-targeting skill progression.
   - Suggested caption: "Per-episode aim rate (fraction of steps centered on log-colored targets) with rolling mean."

4. **Episode peak wood-on-screen**
   - File: `docs/figs/mc_real_v14_no_kill_reset/episodes_full_log_max.png`
   - Purpose: shows tree-seeking behavior before/while mining.
   - Suggested caption: "Peak per-episode log visibility in the world frame."

5. **PPO diagnostics panel**
   - File: `docs/figs/mc_real_v14_no_kill_reset/ppo_train.png`
   - Purpose: confirms policy updates are happening (KL, entropy, value loss, explained variance).
   - Suggested caption: "Training-side PPO diagnostics for the same run."

## Table 1: Multi-run comparison (from `docs/metrics/summary.md`)

| run | steps | max logs/ep | max aim rate | max wood-on-screen | max approach | max return | episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mc_real_v11_no_chat_spam` | 6,656 | 1.000 | 0.525 | 0.452 | 12.140 | 98.379 | 54 |
| `mc_real_v4_seek` | 1,024 | 0.900 | 0.597 | 0.409 | 5.802 | 69.054 | 31 |
| `mc_real_v6_clickrespawn` | 512 | 1.000 | 0.249 | 0.038 | 1.663 | 28.238 | 29 |
| `mc_real_v7_fresh_clean` | 1,024 | 1.000 | 0.005 | 0.047 | 3.160 | 30.151 | 104 |
| `mc_real_v8_robust_dismiss` | 512 | 1.000 | 0.078 | 0.116 | 6.260 | 37.812 | 18 |
| `mc_sanity` | 1,536 | 1.000 | 0.800 | — | — | 38.434 | 21 |

## Table 2: Main run headline metrics (`mc_real_v14_no_kill_reset`)

| metric | value |
|---|---|
| episodes | 40 |
| total timesteps | 2,018 |
| wall-clock minutes | 14.5 |
| total logs mined under policy | 34 |
| success rate (logs >= 1) | 85.0% (34/40) |
| success rate, first half | 100.0% |
| success rate, last half | 70.0% (Delta -30.0 pp) |
| mean ep aim rate | 82.2% |
| mean ep full-log-max | 0.669 |
| mean ep return | 90.80 |
| mean ep length (agent steps) | 50.5 |

## Table 3: PPO update diagnostics (`mc_real_v14_no_kill_reset`)

| metric | first | last | delta |
|---|---:|---:|---:|
| entropy_loss | -2.0592 | -2.0544 | +0.0049 |
| value_loss | +355.6167 | +154.6847 | -200.9320 |
| explained_variance | -0.5388 | -1.2323 | -0.6936 |
| approx_kl | +0.0074 | +0.0106 | +0.0032 |

## Notes for writing Results

- Use **Table 1** as parameter/setting search context across runs.
- Use **Table 2 + cumulative_logs.png** as main objective evidence.
- Use **Table 3 + ppo_train.png** as evidence that PPO updates occurred.
- Pair one "what the model sees" image (`capture_after_input.png`) with one outcome image from `runs/<run>/snaps/` to connect method to behavior.
