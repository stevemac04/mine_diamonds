"""Calibrate the inventory + recipe-book click positions for your setup.

The defaults in :func:`mine_diamonds.scripted.inventory_layout.default_layout_for_monitor`
assume vanilla 1.20.x at GUI scale 2 on a centered window. If your MC
client looks different, this tool helps:

  1. Captures the screen with the inventory open.
  2. Overlays the predicted click anchors (search bar, first recipe slot,
     2x2 output slot) on the screenshot.
  3. Saves the annotated image to ``eval/calibrate_inventory/preview.png``.
  4. You eyeball it, edit the anchor offsets if needed, save the JSON.

USAGE (PowerShell):

  # Open Minecraft, open your inventory (E), make sure the recipe book
  # panel is OPEN (click the recipe-book book icon if not). You should
  # see the search bar and a grid of recipes.
  # Run this script:

  .\\.venv\\Scripts\\Activate.ps1
  $env:PYTHONUTF8=1
  python scripts\\calibrate_inventory.py --countdown 5

  # Open eval\\calibrate_inventory\\preview.png. Cyan dots should be:
  #   * search bar (top of recipe panel)
  #   * first recipe tile (top-left of recipe grid)
  #   * 2x2 craft output slot (right of the 2x2 grid in the inventory)
  # If they're aligned, you're good. If not, adjust --gui-scale and rerun,
  # or hand-edit the saved layout JSON.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mine_diamonds.capture import find_minecraft_window  # noqa: E402
from mine_diamonds.scripted.inventory_layout import (  # noqa: E402
    InventoryLayout,
    default_layout_for_monitor,
)


def _dot(img: np.ndarray, x: int, y: int, label: str, color=(255, 255, 0)) -> None:
    cv2.circle(img, (int(x), int(y)), 8, color, -1)
    cv2.circle(img, (int(x), int(y)), 12, (0, 0, 0), 2)
    cv2.putText(
        img,
        label,
        (int(x) + 14, int(y) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--gui-scale", type=int, default=2)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--out", type=str, default="eval/calibrate_inventory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Open Minecraft, press E to open the inventory, and confirm the\n"
        "RECIPE BOOK PANEL is visible (click the book icon next to the\n"
        "crafting grid if not). You should see a search bar and recipe grid."
    )
    for i in range(args.countdown, 0, -1):
        print(f"  capturing in {i}s", flush=True)
        time.sleep(1.0)

    region = find_minecraft_window()
    with mss.mss() as sct:
        if region is not None:
            mon = region.as_mss_monitor()
            print(
                f"  detected MC window: {region.title!r} at "
                f"({region.left},{region.top}) {region.width}x{region.height}"
            )
        else:
            mon = sct.monitors[args.monitor]
            print(
                f"  no MC window detected; falling back to monitor {args.monitor}: "
                f"{mon['width']}x{mon['height']}"
            )
        mw = int(mon["width"])
        mh = int(mon["height"])
        offx = int(mon.get("left", 0))
        offy = int(mon.get("top", 0))
        raw = np.array(sct.grab(mon))
    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    layout = default_layout_for_monitor(
        mw, mh, gui_scale=args.gui_scale, offset_x=offx, offset_y=offy
    )

    print(f"  predicted layout (GUI scale {args.gui_scale}):")
    print("    " + layout.to_json().replace("\n", "\n    "))

    # Convert screen-space anchors to image-space by subtracting capture offset.
    annotated = bgr.copy()
    _dot(annotated, layout.recipe_search_x - offx, layout.recipe_search_y - offy, "search", (255, 255, 0))
    _dot(annotated, layout.recipe_first_result_x - offx, layout.recipe_first_result_y - offy, "first recipe", (0, 255, 255))
    _dot(annotated, layout.craft_output_x - offx, layout.craft_output_y - offy, "craft output", (0, 200, 255))
    _dot(annotated, layout.screen_center_x - offx, layout.screen_center_y - offy, "screen center", (255, 0, 255))

    preview_path = out_dir / "preview.png"
    cv2.imwrite(str(preview_path), annotated)
    layout_path = out_dir / "layout.json"
    layout.save(layout_path)

    print(f"  wrote {preview_path}")
    print(f"  wrote {layout_path}")
    print(
        "\nOpen the preview PNG. The cyan/yellow dots should be ON or VERY CLOSE\n"
        "to the actual UI elements. If they're systematically shifted (e.g. all\n"
        "of them are 100 px too far right), edit layout.json by the same offset\n"
        "and re-run; or rerun with a different --gui-scale.\n"
        "\nWhen you run craft_table_after_rl.py, pass --layout-json to point at\n"
        "this file."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
