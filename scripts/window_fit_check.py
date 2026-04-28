"""Quick check: Minecraft client capture + hotbar + fovea alignment.

Run with MC in windowed mode, same as training. Saves one annotated PNG
under eval/window_fit/ (gitignored) and prints dimensions so you can
confirm the ROI is on slot 1 and the fovea is centered on the crosshair.

  .\\.venv\\Scripts\\python.exe scripts\\window_fit_check.py --countdown 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mine_diamonds.capture import find_minecraft_window  # noqa: E402
from mine_diamonds.envs.minecraft_real import FoveaSpec, HotbarSpec  # noqa: E402

import mss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--gui-scale", type=int, default=2)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval" / "window_fit" / "capture_annotated.png",
    )
    args = ap.parse_args()

    for i in range(args.countdown, 0, -1):
        print(f"  focus Minecraft... ({i}s)", flush=True)
        time.sleep(1.0)

    region = find_minecraft_window()
    if region is None:
        print("ERROR: no Minecraft window found.", flush=True)
        return 1

    mon = region.as_mss_monitor()
    with mss.mss() as sct:
        raw = np.array(sct.grab(mon))
    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]

    hb = HotbarSpec.from_monitor(w, h, slot=1, gui_scale=int(args.gui_scale))
    fx, fy, fw, fh = FoveaSpec().rect(w, h)

    out = bgr.copy()
    cv2.rectangle(
        out, (hb.x, hb.y), (hb.x + hb.w, hb.y + hb.h), (0, 0, 255), 2
    )
    cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
    cv2.putText(
        out,
        "slot1",
        (hb.x, max(12, hb.y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "fovea",
        (fx, max(12, fy - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out)

    # Heuristic: hotbar should sit in the bottom part of the client view.
    hb_center_y = hb.y + hb.h // 2
    frac_from_bottom = (h - hb_center_y) / max(1, h)

    print("Window / capture:", flush=True)
    print(f"  client area: {w} x {h} px  ({region.title!r})", flush=True)
    print(f"  hotbar slot1 ROI: x={hb.x} y={hb.y} w={hb.w} h={hb.h}", flush=True)
    print(f"  fovea ROI: x={fx} y={fy} w={fw} h={fh}", flush=True)
    print(
        f"  hotbar vertical: center at {100 * hb_center_y / h:.0f}% from top "
        f"({frac_from_bottom:.2f} of height from bottom — expect small if ROI is near bottom bar)",
        flush=True,
    )
    print(f"  wrote {args.out.resolve()}", flush=True)
    if frac_from_bottom < 0.08 or hb.y < h * 0.5:
        print(
            "  WARN: hotbar box looks too high; check GUI scale 2, windowed MC, or --gui-scale",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
