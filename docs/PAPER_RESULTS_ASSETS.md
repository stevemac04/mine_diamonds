# Paper: figures and tables (short)

The canonical list of **which PNGs to embed in Results** and which markdown files hold Table 1 numbers is:

**`docs/FIGURES_FOR_RESULTS.md`**

`docs/PAPER_WORKING_DRAFT.md` is the current working essay (same repo).

To regenerate figures after a new training run:

```powershell
python scripts\build_paper_figs.py --run-name <run_name>
```

Optional: compare many TensorBoard runs into charts (outputs `docs/metrics/` locally — that folder is gitignored so it does not clutter the hand-in repo):

```powershell
python scripts\aggregate_metrics.py
```
