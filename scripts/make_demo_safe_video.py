"""Generate the demo-safe fixture: a BUS-ONLY crop with declared synthetic motion.

Why this exists
---------------
The historical M1 smoke clip (``scripts/make_demo_video.py`` ->
``data/demo/demo_bus.mp4``) translates the *whole* Ultralytics sample photo,
which contains recognizable foreground pedestrians. Their public availability
does not establish consent, so that clip is not used for the demo-safe journey.
The historical M1 smoke/generator is intentionally left untouched.

This script instead crops the photo to the bus region only, excluding the
foreground pedestrians, and translates that crop across a plain canvas. YOLO
still runs genuine inference and detects a real ``bus``; only the motion is
synthetic. Nothing here fabricates a model prediction.

Source / crop / motion (also recorded in docs/LICENSES.md)
----------------------------------------------------------
* Source asset : ``ultralytics/assets/bus.jpg`` (bundled with the AGPL-3.0
  ultralytics pip package; pinned 810x1080 — WxH, i.e. shape (1080, 810)).
* Crop (xyxy)  : [350, 220, 755, 665] -> the bus, excluding foreground people.
* Output       : 960x540, 30 fps, 180 frames (~6 s), the crop resized to
  364x400 and swept left -> right across a flat gray canvas (synthetic motion).

Generation command
-------------------
    services/api/.venv/bin/python scripts/make_demo_safe_video.py \
        data/demo/demo_bus_private_crop.mp4

No video/photo/weights are committed; regenerate locally before a demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import ASSETS

CROP_XYXY = (350, 220, 755, 665)  # x1, y1, x2, y2 — bus only, no pedestrians
CANVAS = (960, 540)  # width, height
FRAMES = 180
FPS = 30.0


def generate(destination: Path) -> Path:
    photo = cv2.imread(str(ASSETS / "bus.jpg"))
    if photo is None or photo.shape[:2] != (1080, 810):
        raise RuntimeError(
            "Expected the pinned Ultralytics 1080x810 bus asset; inspect the "
            "changed asset before use (the crop coordinates assume it)."
        )
    x1, y1, x2, y2 = CROP_XYXY
    bus = cv2.resize(photo[y1:y2, x1:x2], (364, 400))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), FPS, CANVAS
    )
    if not writer.isOpened():
        raise RuntimeError("MP4 writer could not open")
    w, h = CANVAS
    try:
        for i in range(FRAMES):
            frame = np.full((h, w, 3), 60, dtype=np.uint8)
            x = int(i / (FRAMES - 1) * (w - 364))
            frame[120:520, x:x + 364] = bus
            writer.write(frame)
    finally:
        writer.release()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination", nargs="?", default="data/demo/demo_bus_private_crop.mp4"
    )
    print(generate(Path(parser.parse_args().destination)))
