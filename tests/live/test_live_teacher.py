"""Teacher observation adapter: non-actioning, uncertain, bounded, cached."""

import pytest

from app.live import config, teacher
from app.live.teacher import Observation


@pytest.fixture(autouse=True)
def _clean():
    teacher._reset_for_tests()
    teacher.transport = None
    yield
    teacher.transport = None


def test_unavailable_when_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "TEACHER_BASE_URL", "")
    obs = teacher.observe("room1", "is it open?", ["open", "closed"], [b"jpeg"])
    assert obs.status == "unavailable"
    assert obs.uncertain is True


def test_observation_is_non_actioning_and_uncertain():
    teacher.transport = lambda q, labels, imgs: Observation(
        status="ok", label="open", confidence=0.91, uncertain=True, model="smolvlm")
    obs = teacher.observe("room1", "is it open?", ["open", "closed"], [b"abc"])
    assert obs.status == "ok"
    # Even a high reported confidence stays flagged uncertain (uncalibrated model).
    assert obs.uncertain is True
    assert obs.image_hash is not None


def test_result_cached_by_image_hash():
    calls = {"n": 0}

    def _t(q, labels, imgs):
        calls["n"] += 1
        return Observation(status="ok", label="closed", uncertain=True)

    teacher.transport = _t
    teacher.observe("room1", "q", ["a", "b"], [b"same"])
    teacher.observe("room1", "q", ["a", "b"], [b"same"])
    assert calls["n"] == 1  # second call served from cache


def test_budget_exhausted(monkeypatch):
    monkeypatch.setattr(config, "TEACHER_ROOM_BUDGET", 1)
    monkeypatch.setattr(config, "TEACHER_COOLDOWN_SECONDS", 0.0)
    teacher.transport = lambda q, labels, imgs: Observation(status="ok", label="a", uncertain=True)
    assert teacher.observe("room1", "q", ["a", "b"], [b"one"]).status == "ok"
    assert teacher.observe("room1", "q", ["a", "b"], [b"two"]).status == "budget"
