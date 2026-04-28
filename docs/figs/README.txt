PNG files to pair with a typical Results writeup (not auto-generated; pick what fits your text).

General / training (single-window = slow wall time, policy still updating):
  mc_real_superflat_long_y0/ppo_train.png

Early 40-episode run — strong looking behavior (aim, wood on screen) before stricter slot checks; use to support the false-positive story (do NOT use as "final success" proof by itself):
  mc_real_v14_no_kill_reset/episodes_aim_rate.png
  mc_real_v14_no_kill_reset/episodes_full_log_max.png

Long superflat run — cumulative logs vs timesteps (main learning-curve style plot):
  mc_real_superflat_long_y0/cumulative_logs.png

Optional (per-episode return / extra context):
  mc_real_v14_no_kill_reset/episodes_return.png
  mc_real_superflat_long_y0/episodes_return.png

Regenerate plots from a run folder: python scripts/build_paper_figs.py --run-name <name>
(JSON source: runs/<name>/episodes.jsonl; runs/ is gitignored.)
