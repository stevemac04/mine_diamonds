# Simple-colors texture pack (reward detector)

The RL reward and vision code assumes **logs render as a narrow band of
near-black BGR** (see `src/mine_diamonds/vision/pack_colors.py`). A
flat-color or “simple” resource pack that replaces vanilla log textures
with a solid dark color is required for reliable detection from screen
capture.

## What to do

1. Obtain a resource pack you trust that makes oak logs a flat dark
   color, or build a minimal one (only the log block textures you need).
2. Place the pack **ZIP** in this folder (e.g. `assets/texture_pack/simple-colors.zip`) **or** anywhere you prefer — Minecraft only needs the file path when you add the pack in-game.
3. In Minecraft: **Options → Resource Packs** → add the pack and move
   it to the top of the list.
4. Re-run `scripts/smoke_minecraft_real.py` and confirm log-colored
   pixels show up in the hotbar when you hold a log.

If your pack’s log color is not near-black, adjust the `logs` BGR
range in `pack_colors.py` to match a sampled pixel from
`eval/smoke_minecraft/hotbar_slot.png` after a successful smoke test.

## GUI

Use **GUI Scale 2** and **windowed** mode so `capture.py` can find the
client window and hotbar ROIs line up with defaults.

## Successful RL evidence

- Wood acquisition run video: [WOOD!.mp4 (Google Drive)](https://drive.google.com/file/d/1X8ne4xbKFtzojJPiqBfAn4OejCizwwOO/view?usp=sharing)
