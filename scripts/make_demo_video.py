"""Build a demo MP4 from REAL photographic content (Ultralytics sample assets).

The photo (bus + pedestrians, bundled with the ultralytics package under the
AGPL-licensed distribution) is translated across a canvas so objects move
through zones. Detection in the pipeline is genuine YOLO inference on real
image content; only the motion is synthetic. This provenance is disclosed in
docs/LICENSES.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


def make_demo_video(out_path: str, seconds: float = 6.0, fps: float = 30.0) -> str:
    from ultralytics.utils import ASSETS  # bundled sample images

    photo = cv2.imread(str(ASSETS / "bus.jpg"))
    if photo is None:
        raise RuntimeError("ultralytics bus.jpg asset not found")

    canvas_w, canvas_h = 960, 540
    scale = 400 / photo.shape[0]
    photo = cv2.resize(photo, (int(photo.shape[1] * scale), 400))
    ph, pw = photo.shape[:2]
    y0 = canvas_h - ph - 20

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h)
    )
    import numpy as np

    total = int(seconds * fps)
    travel = canvas_w - pw
    for i in range(total):
        frame = np.full((canvas_h, canvas_w, 3), 60, dtype="uint8")
        x0 = int(i / max(1, total - 1) * travel)
        frame[y0 : y0 + ph, x0 : x0 + pw] = photo
        writer.write(frame)
    writer.release()
    return str(out)


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "data/demo/demo_bus.mp4"
    print(make_demo_video(dest))
