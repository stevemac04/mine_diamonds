"""Pixel-coordinate layout for the Minecraft survival inventory + recipe book.

Reference: vanilla 1.20+ inventory texture is 176x166 GUI px, recipe book
panel is 152x166 GUI px, both at the user's GUI scale. We compute
absolute screen-pixel anchors from the monitor size assuming the
inventory window is centered on the monitor.

If your MC layout looks different (different resource pack widget sizes,
different GUI scale, monitor with task bar reserving pixels, etc.) run
``scripts/calibrate_inventory.py`` to fine-tune the offsets and save a
JSON override.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class InventoryLayout:
    """Absolute screen-pixel anchors for inventory UI elements."""

    monitor_w: int
    monitor_h: int
    gui_scale: int

    # Recipe-book panel anchors.
    recipe_search_x: int
    recipe_search_y: int
    recipe_first_result_x: int
    recipe_first_result_y: int

    # 2x2 crafting grid output slot.
    craft_output_x: int
    craft_output_y: int

    # Center of the screen — used for placing blocks in the world.
    screen_center_x: int
    screen_center_y: int

    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "InventoryLayout":
        return cls(**json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "InventoryLayout":
        return cls.from_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


def default_layout_for_monitor(
    mon_w: int,
    mon_h: int,
    *,
    gui_scale: int = 2,
    offset_x: int = 0,
    offset_y: int = 0,
) -> InventoryLayout:
    """Compute reasonable defaults for a centered inventory at ``gui_scale``.

    The inventory texture (176x166) sits centered on the rendered area.
    The recipe book panel (152x166) sits to the LEFT of the inventory
    when open, with a 2 GUI-px gap.

    All resulting coordinates are absolute screen pixels: pass
    ``(offset_x, offset_y)`` if you computed the layout for the MC
    window's client area (use
    :func:`mine_diamonds.capture.find_minecraft_window` to get that
    region's screen-space top-left).
    """
    gs = int(gui_scale)
    inv_w = 176 * gs
    inv_w  # quiet linter; kept for clarity
    inv_h = 166 * gs
    inv_h  # quiet linter; kept for clarity
    rec_w = 152 * gs

    inner_cx = mon_w // 2
    inner_cy = mon_h // 2

    inv_left_local = inner_cx - 176 * gs // 2
    inv_top_local = inner_cy - 166 * gs // 2

    rec_right_local = inv_left_local - 2 * gs
    rec_left_local = rec_right_local - rec_w
    rec_top_local = inv_top_local

    rec_search_x = rec_left_local + (9 + 134 // 2) * gs
    rec_search_y = rec_top_local + (6 + 12 // 2) * gs
    rec_first_x = rec_left_local + (11 + 25 // 2) * gs
    rec_first_y = rec_top_local + (31 + 25 // 2) * gs
    out_x = inv_left_local + 124 * gs
    out_y = inv_top_local + 28 * gs
    cx = inner_cx
    cy = inner_cy

    return InventoryLayout(
        monitor_w=int(mon_w),
        monitor_h=int(mon_h),
        gui_scale=gs,
        recipe_search_x=int(rec_search_x + offset_x),
        recipe_search_y=int(rec_search_y + offset_y),
        recipe_first_result_x=int(rec_first_x + offset_x),
        recipe_first_result_y=int(rec_first_y + offset_y),
        craft_output_x=int(out_x + offset_x),
        craft_output_y=int(out_y + offset_y),
        screen_center_x=int(cx + offset_x),
        screen_center_y=int(cy + offset_y),
    )


def default_layout_for_minecraft(*, gui_scale: int = 2) -> InventoryLayout | None:
    """Find the running MC window and compute a layout in screen pixels.

    Returns ``None`` if no Minecraft window is running. Otherwise builds
    the layout assuming MC's inventory is centered on its client area.
    """
    from mine_diamonds.capture import find_minecraft_window

    region = find_minecraft_window()
    if region is None:
        return None
    return default_layout_for_monitor(
        region.width,
        region.height,
        gui_scale=gui_scale,
        offset_x=region.left,
        offset_y=region.top,
    )
