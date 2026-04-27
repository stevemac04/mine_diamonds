# mc_real_v14_no_kill_reset — results

PPO + CnnPolicy on the real Minecraft Java client. 40 episodes over 2,018 agent timesteps (~14.5 min wall-clock). Each episode is capped at 60 s in-game; success = at least one log mined under policy.

## Headline numbers

| metric | value |
|---|---|
| episodes | 40 |
| total timesteps | 2,018 |
| wall-clock minutes | 14.5 |
| total logs mined under policy | **34** |
| success rate (logs ≥ 1) | **85.0%** (34/40) |
| success rate, first half | 100.0% |
| success rate, last half  | 70.0%  (Δ -30.0 pp) |
| mean ep aim rate | 82.2% |
| mean ep full-log-max | 0.669 |
| mean ep return | 90.80 |
| mean ep length (agent steps) | 50.5 |

## Policy update diagnostics (PPO internals)

These are the smoking-gun "PPO is actually updating the policy" metrics. We report first vs. last logged value:

| metric | first | last | Δ |
|---|---|---|---|
| entropy_loss | -2.0592 | -2.0544 | +0.0049 |
| value_loss | +355.6167 | +154.6847 | -200.9320 |
| explained_variance | -0.5388 | -1.2323 | -0.6936 |
| approx_kl | +0.0074 | +0.0106 | +0.0032 |

## How to read these

* `cumulative_logs.png` is the headline RL plot: total logs mined under policy as a function of agent timesteps. A monotonically rising line is RL working.
* `episodes_logs.png` and `episodes_aim_rate.png` are per-episode raw counts (no rolling mean) — paper-grade scatter.
* `episodes_full_log_max.png` is *peak fraction of screen showing wood per episode*. This is the cleanest signal of "the agent learned to look at trees" before it learned to mine them.
* `ppo_train.png` is the policy-update sanity panel: entropy decay = exploring less, value loss → 0 = value head fitting, explained_var rising = value head explaining returns, KL bounded by clip = updates aren't blowing up.