"""Workflow DSL validation tests."""

import pytest
from pydantic import ValidationError

from app.schemas import WorkflowDSL


def base(**overrides) -> dict:
    d = {
        "name": "Bay alert",
        "source_id": 1,
        "object_class": "car",
        "trigger": {"type": "object_entered_zone", "zone_id": 1},
        "action": {"type": "create_notification", "config": {}},
        "cooldown_seconds": 10,
    }
    d.update(overrides)
    return d


def test_valid_enter_workflow():
    wf = WorkflowDSL.model_validate(base())
    assert wf.trigger.type == "object_entered_zone"
    assert wf.requires_approval is True


def test_dwell_requires_duration():
    with pytest.raises(ValidationError, match="duration_seconds"):
        WorkflowDSL.model_validate(
            base(trigger={"type": "dwell_in_zone", "zone_id": 1})
        )


def test_count_requires_threshold():
    with pytest.raises(ValidationError, match="threshold"):
        WorkflowDSL.model_validate(
            base(trigger={"type": "object_count_exceeded", "zone_id": 1})
        )


def test_unsupported_trigger_rejected():
    with pytest.raises(ValidationError):
        WorkflowDSL.model_validate(
            base(trigger={"type": "object_levitates", "zone_id": 1})
        )


def test_unsupported_class_rejected():
    with pytest.raises(ValidationError, match="unsupported object_class"):
        WorkflowDSL.model_validate(base(object_class="unicorn"))


def test_webhook_action_now_rejected():
    # External webhook delivery was removed (M1-011/M1-012); send_webhook is no
    # longer a valid ActionType, so creating such a workflow fails validation.
    with pytest.raises(ValidationError):
        WorkflowDSL.model_validate(
            base(action={"type": "send_webhook", "config": {"url": "http://x/y"}})
        )


def test_arbitrary_action_rejected():
    with pytest.raises(ValidationError):
        WorkflowDSL.model_validate(base(action={"type": "run_shell", "config": {"cmd": "rm -rf /"}}))
