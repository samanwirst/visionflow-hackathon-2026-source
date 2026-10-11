import pytest

from visionflow_vision.zones import Zone, bbox_anchor, point_in_polygon

SQUARE = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))


def test_point_inside_square():
    assert point_in_polygon((0.5, 0.5), SQUARE)


def test_point_outside_square():
    assert not point_in_polygon((0.1, 0.5), SQUARE)
    assert not point_in_polygon((0.5, 0.9), SQUARE)


def test_degenerate_polygon_is_never_inside():
    assert not point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 1.0)])


def test_concave_polygon():
    # L-shape: notch at top-right
    poly = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon((0.25, 0.25), poly)  # in the vertical arm
    assert point_in_polygon((0.75, 0.75), poly)  # in the horizontal arm
    assert not point_in_polygon((0.75, 0.25), poly)  # in the notch


def test_bbox_anchor_is_bottom_center():
    assert bbox_anchor(10, 20, 30, 60) == (20, 60)


def test_zone_requires_three_vertices():
    with pytest.raises(ValueError):
        Zone(id=1, name="bad", polygon=((0.1, 0.1), (0.9, 0.9)))


def test_zone_rejects_unnormalized_coordinates():
    with pytest.raises(ValueError):
        Zone(id=1, name="bad", polygon=((0.0, 0.0), (2.0, 0.0), (1.0, 1.0)))


def test_zone_contains_uses_pixel_anchor():
    zone = Zone(id=1, name="bay", polygon=SQUARE)
    # 100x100 frame: pixel (50, 50) -> normalized (0.5, 0.5) inside
    assert zone.contains((50, 50), 100, 100)
    assert not zone.contains((5, 50), 100, 100)
