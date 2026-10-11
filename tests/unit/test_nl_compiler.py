"""Deterministic NL -> DSL compiler tests (M2-AC1).

Covers: successful compilation of the supported trigger families, strict
rejection of unsupported/ambiguous requests, visual-predicate detection, and
the guarantee that the draft is always validatable by WorkflowDSL (no free-form
or executable output)."""

from app.nl_compiler import compile_text
from app.schemas import WorkflowDSL

ZONES = [{"id": 7, "name": "Service Bay A"}]


def test_enter_compiles_to_valid_dsl():
    r = compile_text("Alert me when a car enters Service Bay A", 1, ZONES)
    assert r.ok and r.draft is not None
    assert r.draft["object_class"] == "car"
    assert r.draft["trigger"]["type"] == "object_entered_zone"
    assert r.draft["trigger"]["zone_id"] == 7
    assert r.draft["requires_approval"] is True
    WorkflowDSL.model_validate(r.draft)  # must satisfy the strict DSL


def test_dwell_parses_duration():
    r = compile_text("notify when a truck stays in Service Bay A for 10 minutes", 1, ZONES)
    assert r.ok, r.errors
    assert r.draft["trigger"]["type"] == "dwell_in_zone"
    assert r.draft["trigger"]["duration_seconds"] == 600.0


def test_count_parses_threshold():
    r = compile_text("tell me when more than 3 cars are in Service Bay A", 1, ZONES)
    assert r.ok, r.errors
    assert r.draft["trigger"]["type"] == "object_count_exceeded"
    assert r.draft["trigger"]["threshold"] == 3


def test_dwell_without_duration_rejected():
    r = compile_text("notify when a car stays in Service Bay A", 1, ZONES)
    assert not r.ok
    assert any("duration" in e for e in r.errors)


def test_unsupported_class_rejected():
    r = compile_text("alert when a spaceship enters Service Bay A", 1, ZONES)
    assert not r.ok
    assert any("object class" in e for e in r.errors)


def test_ambiguous_vehicle_rejected():
    r = compile_text("alert when a vehicle enters Service Bay A", 1, ZONES)
    assert not r.ok
    assert any("ambiguous" in e.lower() or "vehicle" in e.lower() for e in r.errors)


def test_unknown_trigger_rejected():
    r = compile_text("do something interesting with a car", 1, ZONES)
    assert not r.ok
    assert any("trigger" in e for e in r.errors)


def test_ambiguous_zone_rejected_when_multiple():
    zones = [{"id": 1, "name": "Bay A"}, {"id": 2, "name": "Bay B"}]
    r = compile_text("alert when a car enters", 1, zones)
    assert not r.ok
    assert any("zone" in e.lower() for e in r.errors)


def test_single_zone_assumed_with_warning():
    r = compile_text("alert when a car enters", 1, ZONES)
    assert r.ok, r.errors
    assert any("assuming zone" in w for w in r.warnings)


def test_external_delivery_rejected():
    r = compile_text("when a car enters Service Bay A send a webhook", 1, ZONES)
    assert not r.ok
    assert any("external delivery" in e for e in r.errors)


def test_visual_predicate_detected():
    r = compile_text(
        "when a car enters Service Bay A, verify the hood is open", 1, ZONES
    )
    assert r.ok, r.errors
    assert "visual_predicate" in r.draft
    vp = r.draft["visual_predicate"]
    assert vp["expected_label"] in vp["labels"]
    WorkflowDSL.model_validate(r.draft)


def test_empty_rejected():
    assert not compile_text("", 1, ZONES).ok
    assert not compile_text("  ", 1, ZONES).ok


# -- M2-001: strict grammar — preserve supported conditions, reject the rest --

def test_with_condition_preserves_visual_predicate():
    # "with its hood open" is a supported visual condition and must be preserved,
    # not silently dropped into a plain geometric notification.
    r = compile_text(
        "When a car enters Service Bay A with its hood open, create an alert",
        1, ZONES,
    )
    assert r.ok, r.errors
    assert "visual_predicate" in r.draft
    assert "hood" in r.draft["visual_predicate"]["description"]
    WorkflowDSL.model_validate(r.draft)


def test_unsupported_visual_condition_rejected():
    r = compile_text(
        "When a car enters Service Bay A with its windows tinted, alert me", 1, ZONES
    )
    assert not r.ok
    assert any("condition" in e.lower() for e in r.errors)


def test_email_action_rejected():
    r = compile_text("When a car enters Service Bay A, email the advisor", 1, ZONES)
    assert not r.ok
    assert any("external delivery" in e for e in r.errors)


def test_code_execution_rejected():
    r = compile_text("When a car enters Service Bay A, run Python code", 1, ZONES)
    assert not r.ok
    assert any("executing code" in e for e in r.errors)


def test_multiple_classes_rejected():
    r = compile_text(
        "When a car or truck enters Service Bay A, create an alert", 1, ZONES
    )
    assert not r.ok
    assert any("more than one object class" in e for e in r.errors)


# -- M2-001 (re-opened): negation, unsupported actions, alternatives, counts ---

def test_negated_visual_condition_rejected():
    # "hood not open" must NOT become a positive hood-open predicate.
    r = compile_text(
        "When a car enters Service Bay A with its hood not open, create an alert",
        1, ZONES,
    )
    assert not r.ok
    assert any("negat" in e.lower() for e in r.errors)
    assert r.draft is None


def test_closed_state_condition_rejected():
    r = compile_text(
        "When a car enters Service Bay A with its hood closed, create an alert",
        1, ZONES,
    )
    assert not r.ok
    assert any("negat" in e.lower() for e in r.errors)


def test_unsupported_physical_action_rejected():
    # "unlock the gate" must be rejected, not silently downgraded to a notification.
    r = compile_text("When a car enters Service Bay A, unlock the gate", 1, ZONES)
    assert not r.ok
    assert any("action is not available" in e for e in r.errors)
    assert r.draft is None


def test_open_the_gate_action_rejected():
    r = compile_text("When a car enters Service Bay A, open the gate", 1, ZONES)
    assert not r.ok
    assert any("action is not available" in e for e in r.errors)


def test_alternative_triggers_rejected():
    r = compile_text(
        "When a car enters or exits Service Bay A, create an alert", 1, ZONES
    )
    assert not r.ok
    assert any("more than one trigger" in e for e in r.errors)


def test_inclusive_count_converted_to_exclusive_threshold():
    # "at least 3" is inclusive (>=3); the engine fires on strictly-greater-than,
    # so the threshold must be 2 (count > 2 == count >= 3), with a disclosed note.
    r = compile_text("alert when at least 3 cars are in Service Bay A", 1, ZONES)
    assert r.ok, r.errors
    assert r.draft["trigger"]["type"] == "object_count_exceeded"
    assert r.draft["trigger"]["threshold"] == 2
    assert any("inclusive" in w.lower() or "at least" in w.lower() for w in r.warnings)
    WorkflowDSL.model_validate(r.draft)


def test_upper_bound_count_rejected():
    r = compile_text("alert when fewer than 3 cars are in Service Bay A", 1, ZONES)
    assert not r.ok
    assert any("lower-bound" in e for e in r.errors)


def test_inclusive_count_too_low_rejected():
    r = compile_text("alert when at least 1 car is in Service Bay A", 1, ZONES)
    assert not r.ok
    assert any("too low" in e for e in r.errors)


def test_unsupported_colour_attribute_rejected():
    r = compile_text("alert me when a red car enters Service Bay A", 1, ZONES)
    assert not r.ok
    assert any("attribute" in e.lower() or "colour" in e.lower() for e in r.errors)


def test_plain_more_than_count_still_exclusive():
    # Regression guard: the common exclusive phrasing is unchanged.
    r = compile_text("tell me when more than 3 cars are in Service Bay A", 1, ZONES)
    assert r.ok, r.errors
    assert r.draft["trigger"]["threshold"] == 3


# -- M2-001 (exceptional recovery): anchored grammar rejects unconsumed semantics
# Each of these reproduced as an HTTP 200 ok=true draft at the frozen snapshot;
# the desired behaviour is a rejection with NO draft.

TWO_ZONES = [{"id": 1, "name": "Service Bay A"}, {"id": 2, "name": "Service Bay B"}]


def test_delete_video_action_rejected():
    # "delete the video" must NOT become a create_notification draft.
    r = compile_text("When a car enters Service Bay A, delete the video", 1, ZONES)
    assert not r.ok
    assert r.draft is None
    assert any("delet" in e.lower() or "unsupported" in e.lower() for e in r.errors)


def test_negated_trigger_rejected():
    # "does not enter" must NOT silently become an entry rule.
    r = compile_text(
        "When a car does not enter Service Bay A, create an alert", 1, ZONES
    )
    assert not r.ok
    assert r.draft is None
    assert any("negat" in e.lower() for e in r.errors)


def test_upper_bound_dwell_duration_rejected():
    # "less than 10 minutes" must NOT become a dwell >= 600s rule.
    r = compile_text(
        "When a car stays in Service Bay A for less than 10 minutes, create an alert",
        1, ZONES,
    )
    assert not r.ok
    assert r.draft is None
    assert any("minimum dwell" in e.lower() or "upper bound" in e.lower() for e in r.errors)


def test_alternative_zones_rejected():
    # Two named zones must be rejected, not reduced to the first match.
    r = compile_text(
        "When a car enters Service Bay A or Service Bay B, create an alert", 1, TWO_ZONES
    )
    assert not r.ok
    assert r.draft is None
    assert any("more than one zone" in e.lower() for e in r.errors)


def test_extra_visual_condition_rejected():
    # A supported condition ("hood open") plus an unsupported extra ("leaking oil")
    # must be rejected, not silently reduced to only the hood-open predicate.
    r = compile_text(
        "When a car enters Service Bay A with its hood open and leaking oil, create an alert",
        1, ZONES,
    )
    assert not r.ok
    assert r.draft is None
    assert any("leaking" in e.lower() or "unsupported" in e.lower() for e in r.errors)


def test_unknown_if_condition_rejected():
    # "if its engine is running" is not a supported visual condition.
    r = compile_text(
        "When a car enters Service Bay A if its engine is running, create an alert",
        1, ZONES,
    )
    assert not r.ok
    assert r.draft is None
    assert any("condition" in e.lower() or "unsupported" in e.lower() for e in r.errors)


def test_single_named_zone_among_many_still_ok():
    # Regression guard: naming exactly one of several zones still compiles.
    r = compile_text("alert me when a car enters Service Bay B", 1, TWO_ZONES)
    assert r.ok, r.errors
    assert r.draft["trigger"]["zone_id"] == 2


def test_longer_than_dwell_still_ok():
    # Regression guard: a lower-bound dwell phrasing is preserved.
    r = compile_text(
        "notify when a car stays in Service Bay A longer than 5 minutes", 1, ZONES
    )
    assert r.ok, r.errors
    assert r.draft["trigger"]["type"] == "dwell_in_zone"
    assert r.draft["trigger"]["duration_seconds"] == 300.0
