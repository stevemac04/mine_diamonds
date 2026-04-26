"""Scripted (non-RL) action sequences for Minecraft.

The RL agent in ``MinecraftRealEnv`` learns the perception-heavy task
(navigate to a tree, aim at it, hold attack until logs are mined). The
deterministic UI-manipulation parts — opening the inventory, using the
recipe-book search, clicking specific slots — are handled here. They're
fragile to MC version / window resolution / GUI scale, so each runtime
piece is parameterized over an :class:`InventoryLayout` that you can
calibrate once for your setup.
"""

from mine_diamonds.scripted.inventory_layout import (
    InventoryLayout,
    default_layout_for_minecraft,
    default_layout_for_monitor,
)
from mine_diamonds.scripted.craft_table import (
    close_inventory,
    craft_planks_then_table,
    craft_recipe_shift,
    get_wooden_pickaxe,
    make_crafting_table,
    make_planks,
    make_sticks,
    make_wooden_pickaxe,
    open_inventory,
    right_click_world,
    select_hotbar_slot,
)

__all__ = [
    "InventoryLayout",
    "default_layout_for_minecraft",
    "default_layout_for_monitor",
    "craft_planks_then_table",
    "craft_recipe_shift",
    "get_wooden_pickaxe",
    "make_crafting_table",
    "make_planks",
    "make_sticks",
    "make_wooden_pickaxe",
    "open_inventory",
    "close_inventory",
    "right_click_world",
    "select_hotbar_slot",
]
