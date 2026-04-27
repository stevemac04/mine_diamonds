# mc_real_superflat_retry_bidir — results

PPO + CnnPolicy on the real Minecraft Java client. 4 episodes over 9,454 agent timesteps (~41.1 min wall-clock). Each episode is capped at 60 s in-game; success = at least one log mined under policy.

## Headline numbers

| metric | value |
|---|---|
| episodes | 4 |
| total timesteps | 9,454 |
| wall-clock minutes | 41.1 |
| total logs mined under policy | **0** |
| success rate (logs ≥ 1) | **0.0%** (0/4) |
| success rate, first half | 0.0% |
| success rate, last half  | 0.0%  (Δ +0.0 pp) |
| mean ep aim rate | 3.2% |
| mean ep full-log-max | 0.376 |
| mean ep return | 415.72 |
| mean ep length (agent steps) | 2363.5 |

## Policy update diagnostics (PPO internals)

These are the smoking-gun "PPO is actually updating the policy" metrics. We report first vs. last logged value:

| metric | first | last | Δ |
|---|---|---|---|
| entropy_loss | -0.0230 | -0.0935 | -0.0705 |
| value_loss | +4.1581 | +2.0530 | -2.1051 |
| explained_variance | -0.4159 | +0.9201 | +1.3359 |
| approx_kl | +0.0003 | +0.0007 | +0.0004 |

## How to read these

* `cumulative_logs.png` is the headline RL plot: total logs mined under policy as a function of agent timesteps. A monotonically rising line is RL working.
* `episodes_logs.png` and `episodes_aim_rate.png` are per-episode raw counts (no rolling mean) — paper-grade scatter.
* `episodes_full_log_max.png` is *peak fraction of screen showing wood per episode*. This is the cleanest signal of "the agent learned to look at trees" before it learned to mine them.
* `ppo_train.png` is the policy-update sanity panel: entropy decay = exploring less, value loss → 0 = value head fitting, explained_var rising = value head explaining returns, KL bounded by clip = updates aren't blowing up.