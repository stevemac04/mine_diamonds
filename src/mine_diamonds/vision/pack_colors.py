"""BGR detection ranges for the project's simplified Minecraft texture pack.

All values are **OpenCV BGR** order: (Blue, Green, Red), each 0-255.

Intended pack mapping (flat colors; everything else can stay vanilla):

| Material      | Appearance | Notes |
|---------------|------------|-------|
| logs          | Black        | All log variants |
| planks        | Brown        | All plank variants (incl. furnace fuel) |
| stone         | Grey         | Stone, cobble, deepslate, etc. |
| iron_ore      | White        | Iron ore / white iron-in-world blocks |
| iron_ingot    | Gold / yellow | Ingots only (distinct from white ore) |
| sticks        | Cyan         | Stick item / icon |
| diamond_ore   | Purple       | Diamond ore variants |

Ranges are tolerant for slight capture noise; use ``demo_treechop_vision.py --calibrate``
under real in-game lighting and tighten with ``--bgr-low`` / ``--bgr-high`` if needed.

Nominal reference sRGB to BGR centers used to derive margins:

- Black logs: #000000
- Brown planks: ~#8B4513 (saddle brown)
- Grey stone: ~#9E9E9E
- White iron ore: #FFFFFF
- Gold ingots: ~#FFD700
- Cyan sticks: #00FFFF
- Purple diamond ore: ~#A020F0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BgrRange:
    """Inclusive OpenCV ``cv2.inRange`` bounds in BGR order."""

    low: tuple[int, int, int]
    high: tuple[int, int, int]


# Keys used by CLI --material
MATERIAL_BGR: dict[str, BgrRange] = {
    # True flat black logs only. (0-50 matched dirt/grass/UI — whole screen went "wood".)
    # If your pack reads slightly grey, run --calibrate or pass --bgr-high 28,28,28 etc.
    "logs": BgrRange((0, 0, 0), (22, 22, 22)),
    # Brown planks - keep separated from logs (higher channels).
    "planks": BgrRange((8, 40, 90), (55, 110, 200)),
    # Mid grey stone family
    "stone": BgrRange((105, 105, 105), (200, 200, 200)),
    # Bright neutral white (ore / highlights). May overlap UI whites - mine in-world.
    "iron_ore": BgrRange((200, 200, 200), (255, 255, 255)),
    # Gold ingots: high R+G, low B - distinct from grey and from cyan sticks.
    "iron_ingot": BgrRange((0, 150, 200), (80, 255, 255)),
    # Cyan: high B+G, low R
    "sticks": BgrRange((200, 200, 0), (255, 255, 70)),
    # Purple: high B and R, mid-low G
    "diamond_ore": BgrRange((130, 0, 130), (255, 100, 255)),
}

MATERIAL_KEYS: tuple[str, ...] = tuple(sorted(MATERIAL_BGR.keys()))


def get_bgr_range(material: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if material not in MATERIAL_BGR:
        raise KeyError(f"Unknown material {material!r}. Choose one of: {', '.join(MATERIAL_KEYS)}")
    r = MATERIAL_BGR[material]
    return r.low, r.high


def describe_pack_palette() -> str:
    lines = ["Texture pack to BGR mask keys:", ""]
    for k in MATERIAL_KEYS:
        spec = MATERIAL_BGR[k]
        lines.append(f"  {k:12}  low={spec.low}  high={spec.high}")
    return "\n".join(lines)
