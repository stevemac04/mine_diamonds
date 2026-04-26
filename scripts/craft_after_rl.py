"""End-to-end runner for the scripted post-RL crafting phase.

After the RL agent has mined some wood logs, run this to script the
inventory UI and turn those logs into either a placed crafting table
(``--target table``) or a wooden pickaxe (``--target pickaxe``).

Pre-conditions:
  * Minecraft is running, focused, with the simple-colors texture pack
    and known GUI scale (default 2). The MC window will be auto-detected.
  * Player has wood logs in their inventory:
      ``--target table``    needs >= 1 log.
      ``--target pickaxe``  needs >= 3 logs.
  * Hotbar slot 1 is empty (or the chosen ``--table-hotbar-slot`` is)
    so the freshly-crafted table lands there.
  * Player is standing on solid ground looking forward at empty space.

USAGE (PowerShell):

  .\\.venv\\Scripts\\Activate.ps1
  $env:PYTHONUTF8=1

  # Just place a crafting table.
  python scripts\\craft_after_rl.py --target table --countdown 5

  # Full wooden-pickaxe sequence.
  python scripts\\craft_after_rl.py --target pickaxe --countdown 5
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
from mine_diamonds.scripted.craft_table import (  # noqa: E402
    craft_planks_then_table,
    get_wooden_pickaxe,
)
from mine_diamonds.scripted.inventory_layout import (  # noqa: E402
    InventoryLayout,
    default_layout_for_minecraft,
    default_layout_for_monitor,
)


def grab_capture_region() -> tuple[np.ndarray, dict]:
    """Capture either the MC window (preferred) or the primary monitor."""
    region = find_minecraft_window()
    with mss.mss() as sct:
        if region is not None:
            mon = region.as_mss_monitor()
        else:
            mon = sct.monitors[1]
        raw = np.array(sct.grab(mon))
    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    return bgr, mon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["table", "pickaxe"],
        default="table",
        help="What to craft. 'table' needs >=1 log; 'pickaxe' needs >=3.",
    )
    parser.add_argument("--gui-scale", type=int, default=2)
    parser.add_argument(
        "--layout-json",
        type=str,
        default=None,
        help="Optional InventoryLayout JSON to override auto-detected anchors.",
    )
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--run-name", type=str, default="craft_demo_v1")
    parser.add_argument("--table-hotbar-slot", type=int, default=1)
    args = parser.parse_args()

    out_dir = ROOT / "eval" / "scripted_craft" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.layout_json:
        layout = InventoryLayout.load(args.layout_json)
        print(f"loaded layout from {args.layout_json}")
    else:
        layout = default_layout_for_minecraft(gui_scale=args.gui_scale)
        if layout is None:
            print(
                "WARNING: no Minecraft window detected. Falling back to\n"
                "  default layout for primary monitor; crafting clicks may miss."
            )
            with mss.mss() as sct:
                mon = sct.monitors[1]
            layout = default_layout_for_monitor(
                int(mon["width"]),
                int(mon["height"]),
                gui_scale=args.gui_scale,
            )
    print(layout.to_json())

    print(
        f"\nFocus Minecraft now. Game must be running and your character\n"
        f"in the world (NOT in inventory). Player has wood in inventory.\n"
        f"Target: {args.target}"
    )
    for i in range(args.countdown, 0, -1):
        print(f"  starting in {i}s", flush=True)
        time.sleep(1.0)

    before, _ = grab_capture_region()
    cv2.imwrite(str(out_dir / "before.png"), before)

    if args.target == "table":
        craft_planks_then_table(
            layout,
            open_inv_first=True,
            place_after=True,
            hotbar_slot=int(args.table_hotbar_slot),
        )
    else:
        get_wooden_pickaxe(
            layout,
            open_inv_first=True,
            table_hotbar_slot=int(args.table_hotbar_slot),
        )

    time.sleep(0.6)
    after, _ = grab_capture_region()
    cv2.imwrite(str(out_dir / "after.png"), after)

    print(f"\nwrote {out_dir / 'before.png'}")
    print(f"wrote {out_dir / 'after.png'}")
    print(
        "\nOpen those screenshots and verify:\n"
        f"  - target was '{args.target}': for 'pickaxe' you should see a\n"
        "    crafting table placed AND a wooden pickaxe in your hotbar.\n"
        "    For 'table' you should see a crafting table placed in the world.\n"
        "If clicks landed on empty space, run scripts\\calibrate_inventory.py\n"
        "to tune the layout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
