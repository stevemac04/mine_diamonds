# DEMO runbook

**Single-page, on-stage cheat sheet.** Don't improvise; follow this in
order. The RL agent is fragile in front of an audience. The scripted
chopper is not. The demo is the scripted chopper.

---

## Pre-demo (do this BEFORE the talk starts)

1. Plug in laptop. Disable sleep / display-off.
2. Close Chrome, Slack, Discord, any window with "Minecraft" in its
   title (the launcher counts).
3. Start Minecraft Java, single-player, **superflat** or **forest**
   biome with cheats enabled (`/give` and `/effect` need cheats).
4. Settings:
   * **GUI Scale = 2** (Video Settings → GUI Scale).
   * **Windowed mode**, not fullscreen. Fullscreen breaks alt-tab.
   * **Enable resource pack `simple-colors`** (Options → Resource
     Packs → drop `archive/legacy_assets/simple-colors.zip` if you
     deleted the top-level zip). Logs render as flat black with this
     pack — that's the reward detector's whole life.
5. Stand the player **facing a tree from ~10 blocks away**. The
   easiest setup: superflat world, `/setblock ~ ~ ~5 oak_log` to
   spawn one tree, then turn around to face it. Or just walk to one.
6. Empty the hotbar. Slot 1 (the leftmost) **must be empty** — the
   reward detector treats it as the empty-slot baseline.
7. Activate the venv:

   ```powershell
   cd C:\Users\steve\Desktop\mine-diamonds
   .\.venv\Scripts\Activate.ps1
   $env:PYTHONUTF8=1
   ```

8. Quick smoke (off-stage, you alone):

   ```powershell
   python scripts\smoke_minecraft_real.py --countdown 5
   ```

   Confirms `capture_source: window:'Minecraft ...'`. Check the red
   box in `eval/smoke_minecraft/capture_initial.png` is on slot 1.

---

## Demo path A: scripted (the safe one — RUN THIS)

Two-minute timer; this finishes in 30-90s on a tree you're already
near.

```powershell
python scripts\scripted_tree_chop.py --craft pickaxe --countdown 8
```

What happens, and what to say while it runs:

1. **Countdown 8s.** "Click into Minecraft now." ALT-TAB to MC.
2. **Init chat commands** fire (gamerules + immortality buffs). MC
   chat opens and closes 7 times. Mention: "this is the part that
   used to spam 2 million times — fixed by splitting init from
   per-episode commands."
3. **`SCAN`** → camera sweeps right, looking for log pixels.
   "It can't see a tree yet, so it rotates."
4. **`CENTER`** → it found one. Camera locks onto the trunk.
   "Vision detector found log-colored pixels; centroid math points
   the camera."
5. **`APPROACH`** → walks forward.
6. **`MINE`** → holds left-click. Block breaks (~3s for oak with no
   tool). `LOG ACQUIRED #1` prints.
7. Repeats for logs 2 and 3 (pickaxe needs 3 logs total).
8. **`CRAFTING phase`** — opens inventory, recipe-book search,
   crafting table appears in slot 1.
9. Places the table, opens its 3×3 grid, crafts pickaxe.

**If anything goes wrong:** F12. Releases all keys + mouse.

---

## Demo path B: RL metrics (the spicy one — only if A finished early)

Don't run live training in front of the audience — first iteration is
~3 minutes of nothing visible because PPO needs `n_steps=512` before
its first update. Instead, show the metrics aggregate and TensorBoard:

```powershell
python scripts\aggregate_metrics.py
# opens docs/metrics/summary.md and the .png charts
tensorboard --logdir runs --port 6006
# open http://localhost:6006 in browser
```

```powershell
tensorboard --logdir runs --port 6006
# open http://localhost:6006 in browser
```

Filter to `mc/ep_logs_mean`, `mc/ep_aim_rate`, `mc/ep_full_log_max`.
The collapse-to-zero on v11 is the slide.

---

## After the demo

```powershell
# Show git status — empty repo (no junk files appearing means
# the watchdog is doing its job).
git status
```

For follow-up: show `docs/figs/README.txt` (which PNGs exist) and TensorBoard
from `runs/<name>/` if they want the raw scalars.

---

## Failure-mode quick reference

| Symptom | Cause | Fix on stage |
|---|---|---|
| Camera doesn't move at all | MC isn't focused | Click MC. F12, restart. |
| `LOG ACQUIRED` immediately on start | Slot 1 wasn't empty | F12. `/clear @s`. Restart. |
| Bot stares at sky / feet | SCAN pitch out of range | Wait — pitch re-centers every 3.6s. |
| Bot walks past the tree | Lighting / shadow making log pixels disappear | Stand closer to a tree before starting. |
| Crafting clicks miss | GUI Scale ≠ 2 | F12. Set GUI Scale = 2. Restart. |
| Junk files appear in repo | MC lost focus, chat commands hit Cursor | F12. The watchdog should prevent this; if it triggers, that's actually a fun on-stage moment. |
