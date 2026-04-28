Figures built to match the Results text (numbers in the titles come from the
same data as the CSV — regenerate after updating episodes.csv):

  narrative/results_01_overview_from_csv.png
  narrative/results_02_v14_aim_and_visibility.png
  narrative/results_03_superflat_cumulative_and_visibility.png

Generate them:

  python scripts/build_results_narrative_figs.py

Source data (committed): mc_real_v14_no_kill_reset/episodes.csv and
mc_real_superflat_long_y0/episodes.csv. Raw logs also live in runs/<name>/
as episodes.jsonl (gitignored) when you train locally.

Optional TensorBoard / older pipeline plots (same runs, not from this script):
  mc_real_*/cumulative_logs.png, ppo_train.png, etc. from build_paper_figs.py
