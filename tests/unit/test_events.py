"""Temporal event engine correctness tests (PRD section 9)."""

from visionflow_vision.events import EventType, TrackedObject, ZoneEventEngine
from visionflow_vision.zones import Zone

# 100x100 frame; zone covers the right half.
ZONE = Zone(id=1, name="bay", polygon=((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)))
W, H = 100, 100


def obj(track_id: int, cx: float, cls: str = "car") -> TrackedObject:
    """A 10x10 box whose bottom-center anchor is at pixel x=cx, y=50."""
    return TrackedObject(
        track_id=track_id, class_name=cls, confidence=0.9,
        x1=cx - 5, y1=40, x2=cx + 5, y2=50,
    )


def make_engine(event_type: EventType, **kw) -> ZoneEventEngine:
    defaults = dict(
        frame_w=W, frame_h=H, debounce_frames=2, cooldown_seconds=5.0,
        lost_track_grace_seconds=1.0, rule_key="test",
    )
    defaults.update(kw)
    return ZoneEventEngine(ZONE, "car", event_type, **defaults)


def test_enter_requires_debounce():
    eng = make_engine(EventType.ENTER)
    # 1st frame inside: no event yet (debounce = 2 frames)
    assert eng.update([obj(1, 80)], 0.0) == []
    events = eng.update([obj(1, 80)], 0.2)
    assert len(events) == 1
    assert events[0].type == EventType.ENTER
    assert events[0].track_id == 1


def test_single_flicker_outside_does_not_exit():
    eng = make_engine(EventType.EXIT)
    eng.update([obj(1, 80)], 0.0)
    eng.update([obj(1, 80)], 0.2)  # inside confirmed
    assert eng.update([obj(1, 20)], 0.4) == []  # one frame out: debounced
    eng.update([obj(1, 80)], 0.6)  # back inside
    assert eng.update([obj(1, 80)], 0.8) == []


def test_exit_after_debounced_departure():
    eng = make_engine(EventType.EXIT)
    eng.update([obj(1, 80)], 0.0)
    eng.update([obj(1, 80)], 0.2)
    eng.update([obj(1, 20)], 0.4)
    events = eng.update([obj(1, 20)], 0.6)
    assert [e.type for e in events] == [EventType.EXIT]


def test_enter_fires_once_per_visit_no_repeat_while_staying():
    eng = make_engine(EventType.ENTER)
    eng.update([obj(1, 80)], 0.0)
    total = eng.update([obj(1, 80)], 0.2)
    for i in range(3, 30):
        total += eng.update([obj(1, 80)], i * 0.2)
    assert len(total) == 1


def test_dwell_fires_after_threshold_once():
    eng = make_engine(EventType.DWELL, dwell_seconds=2.0)
    events = []
    for i in range(40):  # 0.2s per frame -> 8 seconds inside
        events += eng.update([obj(1, 80)], i * 0.2)
    assert len(events) == 1
    assert events[0].type == EventType.DWELL
    assert events[0].timestamp >= 2.0


def test_dwell_resets_after_true_exit():
    eng = make_engine(EventType.DWELL, dwell_seconds=1.0, cooldown_seconds=0.0)
    events = []
    for i in range(10):  # inside 0..1.8s -> dwell at ~1.2s
        events += eng.update([obj(1, 80)], i * 0.2)
    for i in range(10, 13):  # leave
        events += eng.update([obj(1, 20)], i * 0.2)
    for i in range(13, 25):  # re-enter and dwell again
        events += eng.update([obj(1, 80)], i * 0.2)
    assert [e.type for e in events] == [EventType.DWELL, EventType.DWELL]
    # dedupe keys must differ between visits
    assert events[0].dedupe_key != events[1].dedupe_key


def test_cooldown_suppresses_rapid_reentry():
    eng = make_engine(EventType.ENTER, cooldown_seconds=100.0)
    events = []
    # enter, leave, re-enter quickly
    for i, cx in enumerate([80, 80, 20, 20, 80, 80]):
        events += eng.update([obj(1, cx)], i * 0.2)
    assert len(events) == 1  # second entry inside cooldown window


def test_lost_track_treated_as_exit_after_grace():
    eng = make_engine(EventType.EXIT, lost_track_grace_seconds=0.5)
    eng.update([obj(1, 80)], 0.0)
    eng.update([obj(1, 80)], 0.2)  # inside
    assert eng.update([], 0.4) == []  # within grace: no exit
    events = eng.update([], 1.0)  # grace exceeded
    assert [e.type for e in events] == [EventType.EXIT]


def test_other_classes_are_ignored():
    eng = make_engine(EventType.ENTER)
    events = []
    for i in range(5):
        events += eng.update([obj(1, 80, cls="person")], i * 0.2)
    assert events == []


def test_count_exceeded_fires_on_rising_edge_only():
    eng = make_engine(EventType.COUNT_EXCEEDED, count_threshold=1, cooldown_seconds=0.0)
    events = []
    # two cars enter the zone
    for i in range(4):
        events += eng.update([obj(1, 80), obj(2, 90)], i * 0.2)
    assert len(events) == 1
    assert events[0].type == EventType.COUNT_EXCEEDED
    # still above threshold: no repeat
    events += eng.update([obj(1, 80), obj(2, 90)], 1.0)
    assert len(events) == 1


def test_count_rearms_with_distinct_dedupe_keys():
    """Each rising threshold crossing must produce a distinct dedupe key so
    persistence does not silently discard later crossings (M1-002)."""
    eng = make_engine(EventType.COUNT_EXCEEDED, count_threshold=1, cooldown_seconds=0.0)
    keys: list[str] = []
    # crossing 1: two cars enter
    for i in range(3):
        keys += [e.dedupe_key for e in eng.update([obj(1, 80), obj(2, 90)], i * 0.2)]
    # drop below threshold: both leave (debounced true exit)
    for i in range(3, 6):
        keys += [e.dedupe_key for e in eng.update([obj(1, 20), obj(2, 20)], i * 0.2)]
    # crossing 2: both re-enter
    for i in range(6, 10):
        keys += [e.dedupe_key for e in eng.update([obj(1, 80), obj(2, 90)], i * 0.2)]
    assert len(keys) == 2
    assert keys[0] != keys[1]


def test_independent_tracks_each_get_enter():
    eng = make_engine(EventType.ENTER, cooldown_seconds=5.0)
    events = []
    for i in range(3):
        events += eng.update([obj(1, 80), obj(2, 85)], i * 0.2)
    assert sorted(e.track_id for e in events) == [1, 2]
