"""Scripted recipe-book sequences: logs -> planks -> sticks -> crafting
table -> wooden pickaxe.

Modern Minecraft (1.14+) recipe book lets us shift-click any unlocked
recipe to auto-craft as many copies as possible from materials in the
inventory. We rely on that throughout — every "make X" function is
just: focus search bar, type item name, shift-click the first result.

Sequence to get wooden tools end-to-end (assuming the RL agent has
deposited >= 3 wood logs in the player inventory):

  1. Open inventory (E).
  2. Recipe book: shift-click "planks"   — converts ALL logs to planks.
  3. Recipe book: shift-click "stick"     — converts a couple planks to
     sticks (we'll have plenty).
  4. Recipe book: shift-click "crafting table" — uses 4 planks.
  5. Close inventory.
  6. Select hotbar slot containing the table.
  7. Right-click ground to place the table.
  8. Right-click the placed table to open it. Player should still be
     aimed at it.
  9. Inside the table's 3x3 GUI, recipe book: shift-click "wooden
     pickaxe" — uses 3 planks + 2 sticks.
  10. Close (E).

This module only does the click-and-type choreography. Verification
(did the wooden pickaxe actually appear in the hotbar?) is the caller's
job. See ``scripts/craft_after_rl.py`` for an end-to-end runner with
before/after screenshots.

Coordinates come from an :class:`InventoryLayout`. Defaults work for
1.20+ at GUI scale 2 with the inventory centered on the MC client area;
calibrate via ``scripts/calibrate_inventory.py`` if your setup differs.

Mouse moves are absolute (``set_cursor_pos``). Do NOT call this while
the RL agent is acting on the env: the env relies on raw relative mouse
deltas, and absolute moves would corrupt that.
"""

from __future__ import annotations

import time

from mine_diamonds.input import game_input as ginput
from mine_diamonds.scripted.inventory_layout import InventoryLayout


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


def _wait(ms: int) -> None:
    time.sleep(ms / 1000.0)


def open_inventory(*, settle_ms: int = 350) -> None:
    """Press E to open the inventory and wait for it to render."""
    ginput.key_down("e")
    _wait(40)
    ginput.key_up("e")
    _wait(settle_ms)


def close_inventory(*, settle_ms: int = 200) -> None:
    """Press E to close the inventory."""
    ginput.key_down("e")
    _wait(40)
    ginput.key_up("e")
    _wait(settle_ms)


def select_hotbar_slot(slot: int, *, settle_ms: int = 180) -> None:
    """Press 1-9 to switch to that hotbar slot."""
    s = str(int(slot))
    if s not in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
        raise ValueError(f"hotbar slot must be 1..9, got {slot}")
    ginput.key_down(s)
    _wait(40)
    ginput.key_up(s)
    _wait(settle_ms)


def right_click_world(*, settle_ms: int = 280) -> None:
    """Right-click the world without moving the mouse.

    Used both for placing a held block and for interacting with a placed
    block (e.g. opening a crafting table). The player should be looking
    at whatever they want to interact with.
    """
    ginput.mouse_right(True)
    _wait(80)
    ginput.mouse_right(False)
    _wait(settle_ms)


def _click_search_and_clear(layout: InventoryLayout) -> None:
    """Focus the recipe book search box and clear any previous text."""
    ginput.click_at(layout.recipe_search_x, layout.recipe_search_y, button="left")
    _wait(140)
    for _ in range(20):
        ginput.key_down("backspace")
        _wait(15)
        ginput.key_up("backspace")
        _wait(15)
    _wait(80)


def _shift_click(x: int, y: int, *, hold_ms: int = 60) -> None:
    """Shift+left-click at (x,y) — auto-moves stack to inventory/hotbar."""
    ginput.key_down("lshift")
    _wait(40)
    ginput.click_at(x, y, button="left", hold_ms=hold_ms)
    _wait(40)
    ginput.key_up("lshift")


# ---------------------------------------------------------------------------
# Recipe-book "shift-click to craft max" primitive
# ---------------------------------------------------------------------------


def craft_recipe_shift(
    layout: InventoryLayout,
    search_term: str,
    *,
    after_ms: int = 320,
) -> None:
    """Search the recipe book and shift-click the first result.

    In MC 1.14+, shift-clicking a recipe in the recipe book auto-crafts
    as many copies as the available materials allow and sends them to
    the inventory. Single-click is one craft at a time.

    Falls back gracefully: if shift-click doesn't actually multi-craft
    on the user's MC version, the recipe still gets a single craft and
    the caller can call again.
    """
    _click_search_and_clear(layout)
    ginput.type_text(search_term)
    _wait(220)
    _shift_click(
        layout.recipe_first_result_x,
        layout.recipe_first_result_y,
        hold_ms=80,
    )
    _wait(after_ms)
    # Backup: also shift-click the output slot in case the recipe-book
    # shift-click only placed items in the grid without auto-collecting.
    _shift_click(layout.craft_output_x, layout.craft_output_y, hold_ms=60)
    _wait(220)


# ---------------------------------------------------------------------------
# High-level recipes
# ---------------------------------------------------------------------------


def make_planks(layout: InventoryLayout) -> None:
    """Convert all logs in the inventory to planks."""
    craft_recipe_shift(layout, "planks")


def make_sticks(layout: InventoryLayout) -> None:
    """Convert some planks to sticks (4 sticks per 2 planks)."""
    craft_recipe_shift(layout, "stick")


def make_crafting_table(layout: InventoryLayout) -> None:
    """Convert 4 planks to 1 crafting table."""
    craft_recipe_shift(layout, "crafting table")


def make_wooden_pickaxe(layout: InventoryLayout) -> None:
    """Craft 1 wooden pickaxe (3 planks + 2 sticks).

    Requires the 3x3 crafting interface — call this AFTER opening a
    crafting table (right-click on a placed table). Inside the
    inventory's 2x2 grid this recipe is unavailable.
    """
    craft_recipe_shift(layout, "wooden pickaxe")


# ---------------------------------------------------------------------------
# End-to-end orchestrators
# ---------------------------------------------------------------------------


def craft_planks_then_table(
    layout: InventoryLayout,
    *,
    open_inv_first: bool = True,
    place_after: bool = True,
    hotbar_slot: int = 1,
) -> None:
    """Quick demo path: logs -> planks -> crafting table (placed).

    Minimum requirement: >= 1 wood log in inventory.
    """
    if open_inv_first:
        open_inventory()
    make_planks(layout)
    make_crafting_table(layout)
    if place_after:
        close_inventory()
        select_hotbar_slot(hotbar_slot)
        right_click_world()


def get_wooden_pickaxe(
    layout: InventoryLayout,
    *,
    open_inv_first: bool = True,
    table_hotbar_slot: int = 1,
    keep_inventory_open_after: bool = False,
) -> None:
    """End-to-end: mined logs -> wooden pickaxe in inventory.

    Sequence:
      1. Open inventory.
      2. Recipe book: planks (uses all logs).
      3. Recipe book: sticks.
      4. Recipe book: crafting table.
      5. Close inventory; switch to hotbar slot ``table_hotbar_slot``;
         right-click to PLACE the crafting table.
      6. Right-click again to OPEN the placed table (player should still
         be aimed at it).
      7. Recipe book: wooden pickaxe.
      8. Close inventory (unless ``keep_inventory_open_after``).

    Minimum requirement: >= 3 wood logs in inventory at start (12 planks
    of headroom; we use 4 for table + 3 for pickaxe + 2 for sticks).
    """
    if open_inv_first:
        open_inventory()
    make_planks(layout)
    make_sticks(layout)
    make_crafting_table(layout)
    close_inventory()
    select_hotbar_slot(table_hotbar_slot)
    right_click_world()
    _wait(450)
    right_click_world()
    _wait(450)
    make_wooden_pickaxe(layout)
    if not keep_inventory_open_after:
        close_inventory()
