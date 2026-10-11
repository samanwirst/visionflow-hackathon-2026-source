"""Fail-closed regression for the VISIONFLOW_DEMO_SAFE parser (DEMO task).

The demo-safe guard must default ON and may be turned OFF only by an explicitly
recognized false value. A typo or any other unrecognized value must keep the
guard ON rather than silently fail-open (the original bug: "treu" -> False).
"""

import pytest

from app import config


@pytest.mark.parametrize(
    "raw",
    ["0", "false", "FALSE", "False", "no", "NO", "off", "Off", "  false  "],
)
def test_recognized_false_disables(monkeypatch, raw):
    monkeypatch.setenv("VISIONFLOW_DEMO_SAFE", raw)
    assert config._demo_safe_from_env() is False


@pytest.mark.parametrize(
    "raw",
    ["1", "true", "TRUE", "True", "yes", "on", "  true  "],
)
def test_recognized_true_enables(monkeypatch, raw):
    monkeypatch.setenv("VISIONFLOW_DEMO_SAFE", raw)
    assert config._demo_safe_from_env() is True


@pytest.mark.parametrize(
    "raw",
    ["treu", "flase", "disable", "2", "", "  ", "nope", "y", "n", "enable", "safe"],
)
def test_unrecognized_value_fails_closed(monkeypatch, raw):
    """Unknown spellings (including the 'treu' typo) must keep safe mode ON."""
    monkeypatch.setenv("VISIONFLOW_DEMO_SAFE", raw)
    assert config._demo_safe_from_env() is True


def test_unset_defaults_to_safe(monkeypatch):
    monkeypatch.delenv("VISIONFLOW_DEMO_SAFE", raising=False)
    assert config._demo_safe_from_env() is True


def test_health_reports_demo_safe(monkeypatch):
    """/api/health must advertise the demo_safe capability for the frontend."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(config, "DEMO_SAFE", True)
    with TestClient(app) as c:
        body = c.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["demo_safe"] is True
