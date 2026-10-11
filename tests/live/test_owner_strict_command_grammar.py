"""Actual parser/validation regressions. Injected proposals are QA_ONLY, not VLM evidence."""
import copy
import pytest
from pydantic import ValidationError
from app.live import dsl, planner


@pytest.fixture(autouse=True)
def clear(monkeypatch):
    monkeypatch.setattr(planner, "transport", None)
    monkeypatch.setattr(planner, "_room_attempts", {})
    monkeypatch.setattr(planner, "_room_last_attempt", {})
    monkeypatch.setattr(planner.config, "PLANNER_COOLDOWN_SECONDS", 0)


def proposal(intent):
    return {"supported": True, "name": "QA_ONLY", "explanation": "QA_ONLY",
            **copy.deepcopy(intent), "duration_seconds": intent["duration_seconds"] or 0,
            "threshold": 0,
            "visual_observation": {"question": "", "labels": [], "temporal": False}}


@pytest.mark.parametrize("text,trigger,duration,cooldown", [
    (planner.SUPPORTED_COMMAND_EXAMPLES[0], "presence_in_zone", None, None),
    (planner.SUPPORTED_COMMAND_EXAMPLES[1], "dwell_in_zone", 1.25, 2.5),
    (planner.SUPPORTED_COMMAND_EXAMPLES[2], "exited_zone", None, 3.5),
    ("Send a local notification whenever a truck is present in the whole image, with an exact cooldown of 0 seconds.", "presence_in_zone", None, 0),
    ("For this recording, notify when a person stays in the entire image for precisely 1.25 seconds. Keep exactly 2.5 seconds between notifications.", "dwell_in_zone", 1.25, 2.5),
    ("Notify whenever a car departs from the whole image.", "exited_zone", None, None),
])
def test_equivalent_supported_wording_requires_provider_and_preserves_exact_intent(monkeypatch, text, trigger, duration, cooldown):
    intent = planner.parse_supported_command(text)
    assert intent and intent["trigger"] == trigger
    assert intent["duration_seconds"] == duration and intent["cooldown_seconds"] == cooldown
    calls = []
    def transport(original, room):
        calls.append((original, room))
        return planner.PlannerResult(status="ok", proposal=proposal(intent), model="QA_ONLY")
    monkeypatch.setattr(planner, "transport", transport)
    result = planner.plan(text, "strict-room")
    assert result.status == "ok" and len(calls) == 1
    assert all(result.task[key] == value for key, value in intent.items())
    assert planner.room_attempts_used("strict-room") == 1


@pytest.mark.parametrize("text", [
    "Notify when a car enters the full frame.",
    "Notify when a car is visible in the service bay.",
    "Notify when a red car is visible in the full frame.",
    "Notify when a car is visible in the full frame except on Sundays.",
    "Notify when a car is visible in the full frame and a person leaves.",
    "Notify when a car is not visible in the full frame.",
    "Notify when two people each remain in the full frame for five seconds.",
    "Email me when a car is visible in the full frame.",
    "Notify when a car stays in the full frame for more than 1.25 seconds.",
    "Notify when a car stays in the full frame for less than 1.25 seconds.",
    "Notify when a car is visible in the full frame. Also open the gate.",
    "Focus on the bottle if anybody opens it.",
    "Notify when a car is visible in the full frame for 1.25 seconds.",
    "Notify when a car stays in the full frame.",
    "Notify when a car stays in the full frame for 0.25 seconds.",
    "Notify when a car is visible in the full frame; set the cooldown to 3601 seconds.",
    "Notify when a car stays in the full frame for 1.25000000000000001 seconds.",
])
def test_whole_unsupported_command_rejects_before_provider_and_budget(monkeypatch, text):
    monkeypatch.setattr(planner, "transport", lambda *_: pytest.fail("provider admitted unsupported text"))
    monkeypatch.setattr(planner, "_run_openai", lambda *_: pytest.fail("paid call admitted"))
    result = planner.plan(text, "strict-room")
    assert result.status == "unsupported" and result.task is None
    assert result.attempt == 0 and planner.room_attempts_used("strict-room") == 0


@pytest.mark.parametrize("delta", [
    {"trigger": "presence_in_zone"}, {"object_class": "person"},
    {"duration_seconds": 1}, {"cooldown_seconds": 10},
    {"action": "focus_notify", "focus": True}, {"threshold": 2},
    {"zone": {"type": "polygon", "polygon": [[0,0],[1,0],[0.5,0.5]]}},
])
def test_provider_proposal_cannot_drop_or_add_any_semantic_field(monkeypatch, delta):
    text = planner.SUPPORTED_COMMAND_EXAMPLES[1]
    value = proposal(planner.parse_supported_command(text))
    value.update(delta)
    monkeypatch.setattr(planner, "transport", lambda *_: planner.PlannerResult(status="ok", proposal=value))
    result = planner.plan(text, "strict-room")
    assert result.status == "unsupported" and result.task is None
    assert all(result.proposal[key] == val for key, val in delta.items())  # No silent repair.


@pytest.mark.parametrize("coordinate", [True, False, "0.5", None, float("nan"), float("inf"), -0.1, 1.1])
def test_coordinates_reject_before_pydantic_coercion(coordinate):
    with pytest.raises(ValidationError):
        dsl.ZoneSpec(type="polygon", polygon=[[0,0], [1,0], [coordinate,1]])


def test_numeric_coordinates_and_canonical_round_trip_are_valid():
    zone = dsl.ZoneSpec(type="polygon", polygon=[[0,0], [1,0], [0.5,1]])
    assert zone.polygon[2] == (0.5,1.0)
    assert dsl.ZoneSpec.model_validate(dsl.ZoneSpec().model_dump()) == dsl.ZoneSpec()


@pytest.mark.parametrize("polygon", [None, "[]", [[0,0],[1,0],[1]],
                                      [[0,0],[1,0],[1,1,0]]])
def test_missing_or_malformed_coordinate_pairs_reject(polygon):
    with pytest.raises(ValidationError):
        dsl.ZoneSpec(type="polygon", polygon=polygon)


def test_accepted_command_still_requires_genuine_provider(monkeypatch):
    monkeypatch.setattr(planner.config, "PLANNER_ENABLED", True)
    monkeypatch.setenv("VISIONFLOW_LIVE_PLANNER_PROVIDER", "openai")
    calls = []
    def unavailable(text):
        calls.append(text)
        return planner.PlannerResult(status="unavailable", error="QA_ONLY provider timeout")
    monkeypatch.setattr(planner, "_run_openai", unavailable)
    result = planner.plan(planner.SUPPORTED_COMMAND_EXAMPLES[0], "accepted-provider-required")
    assert result.status == "unavailable" and result.task is None
    assert calls == [planner.SUPPORTED_COMMAND_EXAMPLES[0]]


def test_accepted_command_without_credential_has_no_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("VISIONFLOW_LIVE_PLANNER_PROVIDER", "openai")
    monkeypatch.setattr(planner.config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(planner, "_run_cli", lambda *_: pytest.fail("CLI fallback"))
    result = planner.plan(planner.SUPPORTED_COMMAND_EXAMPLES[0], "accepted-no-credential")
    assert result.status == "unavailable" and "credential missing" in result.error
    assert result.task is None


def test_accepted_command_never_selects_subscription_cli_on_secure_surface(monkeypatch):
    monkeypatch.setenv("VISIONFLOW_LIVE_PLANNER_PROVIDER", "claude_cli")
    monkeypatch.setattr(planner.config, "COOKIE_SECURE", True)
    monkeypatch.setattr(planner.config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(planner, "_run_cli", lambda *_: pytest.fail("CLI selected"))
    result = planner.plan(planner.SUPPORTED_COMMAND_EXAMPLES[0], "accepted-secure-provider")
    assert result.status == "unavailable" and result.task is None
