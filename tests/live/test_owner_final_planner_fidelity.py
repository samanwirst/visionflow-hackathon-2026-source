"""QA_ONLY proposals, exercising strict canonical conversion (no model calls)."""
import pytest
from app.live import config, planner


def proposal(**extra):
    return {"supported": True, "name": "QA_ONLY", "object_class": "car",
            "zone": {"type": "full_frame", "polygon": []},
            "trigger": "dwell_in_zone", "duration_seconds": 1.25,
            "action": "notify", "focus": False, **extra}


def finalize(value):
    return planner._finalize(planner.PlannerResult(status="ok", proposal=value))


@pytest.mark.parametrize("cooldown", [0, 1.25, 2.5, None])
def test_requested_cooldown_and_decimal_duration_survive_conversion(cooldown):
    result = finalize(proposal(cooldown_seconds=cooldown))
    assert result.status == "ok"
    assert result.task["duration_seconds"] == 1.25
    assert result.task["cooldown_seconds"] == cooldown


@pytest.mark.parametrize("extra", [
    {"cooldown_seconds": True}, {"cooldown_seconds": "2.5"},
    {"cooldown_seconds": float("nan")}, {"focus": "false"},
    {"zone": {"type": "full_frame", "polygon": [], "ignored_constraint": "red"}},
    {"visual_observation": {"condition": "state transition"}},
    {"visual_observation": {"question": "", "labels": [], "temporal": True}},
])
def test_invalid_or_unrepresented_conditions_cannot_be_dropped(monkeypatch, extra):
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", False)
    result = finalize(proposal(**extra))
    assert result.status == "unsupported" and result.task is None


def test_model_schema_requires_nullable_cooldown():
    assert "cooldown_seconds" in planner.PROPOSAL_SCHEMA["required"]
    assert planner.PROPOSAL_SCHEMA["properties"]["cooldown_seconds"]["type"] == ["number", "null"]
