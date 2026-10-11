"""Polygon zones and point-inclusion tests.

Zones store polygons in normalized coordinates (0..1 relative to frame size)
so they stay valid across resolutions. Object anchor point is the bottom-center
of the bounding box — the point where a vehicle or person touches the ground,
which is the most stable anchor for floor-plane zones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test. Polygon is a list of (x, y) vertices."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def bbox_anchor(x1: float, y1: float, x2: float, y2: float) -> Point:
    """Bottom-center anchor of a bounding box in absolute pixel coordinates."""
    return ((x1 + x2) / 2.0, y2)


@dataclass(frozen=True)
class Zone:
    """A named polygonal region with normalized (0..1) vertex coordinates."""

    id: int
    name: str
    polygon: tuple[Point, ...]  # normalized

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"Zone '{self.name}' needs >= 3 vertices, got {len(self.polygon)}")
        for x, y in self.polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"Zone '{self.name}' vertex ({x}, {y}) outside normalized range")

    def contains(self, anchor_px: Point, frame_w: int, frame_h: int) -> bool:
        """Test an absolute-pixel anchor point against this normalized polygon."""
        nx = anchor_px[0] / frame_w
        ny = anchor_px[1] / frame_h
        return point_in_polygon((nx, ny), self.polygon)
