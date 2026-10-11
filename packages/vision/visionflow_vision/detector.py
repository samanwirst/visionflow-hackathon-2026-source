"""Object detection + tracking wrapper.

Default implementation uses Ultralytics YOLO with the built-in ByteTrack
tracker (`model.track(persist=True)`), which performs REAL inference on every
frame it is given. The abstract interface exists so integration tests can
inject a deterministic stub without touching production code paths — any stub
usage is confined to tests and never ships as a demo fallback.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from visionflow_vision.events import TrackedObject


class BaseDetectorTracker(ABC):
    """Produces tracked objects for one frame at a time, in order."""

    @abstractmethod
    def track_frame(self, frame, timestamp: float) -> list[TrackedObject]:
        """frame: BGR numpy array. Returns tracked objects with stable ids."""

    def reset(self) -> None:  # new video session
        pass


class YoloByteTrackDetector(BaseDetectorTracker):
    """Real YOLO detection with ByteTrack association (Ultralytics)."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.35,
        classes: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        import os

        # Keep CPU inference sequential on the limited-RAM host.
        threads = int(os.environ.get("VISIONFLOW_THREADS", "2"))
        try:
            import torch

            torch.set_num_threads(threads)
        except Exception:  # noqa: BLE001 — torch optional at import time
            pass

        from ultralytics import YOLO  # deferred: heavy import

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device
        self.class_filter = set(classes) if classes else None
        self.last_inference_ms: float = 0.0

    def reset(self) -> None:
        # Drop tracker state between videos so track ids restart cleanly.
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor.trackers = []
            self.model.predictor = None

    def track_frame(self, frame, timestamp: float) -> list[TrackedObject]:
        t0 = time.perf_counter()
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        out: list[TrackedObject] = []
        res = results[0]
        if res.boxes is None or res.boxes.id is None:
            return out
        names = res.names
        boxes = res.boxes
        for i in range(len(boxes)):
            cls_name = names[int(boxes.cls[i])]
            if self.class_filter and cls_name not in self.class_filter:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            out.append(
                TrackedObject(
                    track_id=int(boxes.id[i]),
                    class_name=cls_name,
                    confidence=float(boxes.conf[i]),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )
        return out
