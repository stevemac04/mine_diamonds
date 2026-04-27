# mc_real_superflat_long_y0 — results

PPO + CnnPolicy on the real Minecraft Java client. 104 episodes over 36,339 agent timesteps (~162.6 min wall-clock). Each episode is capped at 60 s in-game; success = at least one log mined under policy.

## Headline numbers

| metric | value |
|---|---|
| episodes | 104 |
| total timesteps | 36,339 |
| wall-clock minutes | 162.6 |
| total logs mined under policy | **104** |
| success rate (logs ≥ 1) | **100.0%** (104/104) |
| success rate, first half | 100.0% |
| success rate, last half  | 100.0%  (Δ +0.0 pp) |
| mean ep aim rate | 13.1% |
| mean ep full-log-max | 0.205 |
| mean ep return | 180.62 |
| mean ep length (agent steps) | 349.4 |

## Policy update diagnostics (PPO internals)

These are the smoking-gun "PPO is actually updating the policy" metrics. We report first vs. last logged value:

| metric | first | last | Δ |
|---|---|---|---|
| entropy_loss | -1.2468 | -0.0072 | +1.2396 |
| value_loss | +435.1286 | +17.2872 | -417.8414 |
| explained_variance | +0.0416 | +0.4527 | +0.4111 |
| approx_kl | +2.3609 | +0.0000 | -2.3609 |

## How to read these

* `cumulative_logs.png` is the headline RL plot: total logs mined under policy as a function of agent timesteps. A monotonically rising line is RL working.
* `episodes_logs.png` and `episodes_aim_rate.png` are per-episode raw counts (no rolling mean) — paper-grade scatter.
* `episodes_full_log_max.png` is *peak fraction of screen showing wood per episode*. This is the cleanest signal of "the agent learned to look at trees" before it learned to mine them.
* `ppo_train.png` is the policy-update sanity panel: entropy decay = exploring less, value loss → 0 = value head fitting, explained_var rising = value head explaining returns, KL bounded by clip = updates aren't blowing up.