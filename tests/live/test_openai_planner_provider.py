"""Dummy-key transport regressions; no real provider or paid call is used."""
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.live import planner, planner_api_worker as worker

MODEL = "gpt-4.1-mini-2025-04-14"


def proposal():
    return {"supported": True, "name": "Exact dwell", "explanation": "Geometric dwell",
            "object_class": "car", "zone": {"type": "full_frame", "polygon": []},
            "trigger": "dwell_in_zone", "duration_seconds": 1.25, "threshold": 0,
            "cooldown_seconds": 2.5, "action": "notify", "focus": False,
            "visual_observation": {"question": "", "labels": [], "temporal": False},
            "clarification_needed": "", "unsupported_reason": ""}


def response():
    return {"model": MODEL, "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(proposal()), "refusal": None}}],
        "usage": {"prompt_tokens": 321, "completion_tokens": 89, "total_tokens": 410}}


def request():
    return {"text": "Watch cars in the whole image for precisely 1.25 seconds; cooldown2.5.",
            "system": planner._system_prompt(), "schema": planner.PROPOSAL_SCHEMA}


def fake_http(monkeypatch, content):
    captured = {}

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            captured["closed"] = True
        def read(self, amount):
            captured["read_limit"] = amount
            return content[:amount]

    class Opener:
        def open(self, req, timeout):
            captured.update(url=req.full_url, headers=dict(req.header_items()),
                            body=json.loads(req.data), timeout=timeout)
            return Response()

    def build(*handlers):
        captured["handlers"] = handlers
        return Opener()
    monkeypatch.setattr(worker.urllib.request, "build_opener", build)
    return captured


def test_real_contract_strict_output_and_exact_decimals(monkeypatch):
    captured = fake_http(monkeypatch, json.dumps(response()).encode())
    out = worker.infer(request(), "dummy-private-token", MODEL, 5)
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    body = captured["body"]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["max_completion_tokens"] == 1200
    assert body["store"] is False and body["stream"] is False and body["n"] == 1
    assert "tools" not in body
    assert "dummy-private-token" not in json.dumps(body)
    assert captured["read_limit"] == 65537 and captured["closed"] is True
    assert any(isinstance(h, worker.NoRedirect) for h in captured["handlers"])
    assert any(isinstance(h, worker.urllib.request.ProxyHandler) and h.proxies == {}
               for h in captured["handlers"])
    out = planner._finalize(planner.PlannerResult(status="ok", proposal=out["structured_output"]))
    assert out.task["duration_seconds"] == 1.25
    assert out.task["cooldown_seconds"] == 2.5


@pytest.mark.parametrize("fault", ["oversize", "refusal", "length", "wrong_model", "bool_usage", "nan"])
def test_provider_failures_never_become_plans(monkeypatch, fault):
    data = response()
    if fault == "refusal":
        data["choices"][0]["message"]["refusal"] = "cannot comply"
    elif fault == "length":
        data["choices"][0]["finish_reason"] = "length"
    elif fault == "wrong_model":
        data["model"] = "unidentified-model"
    elif fault == "bool_usage":
        data["usage"]["prompt_tokens"] = True
    elif fault == "nan":
        data["choices"][0]["message"]["content"] = '{"supported":true,"duration_seconds":NaN}'
    payload = b"x" * 70000 if fault == "oversize" else json.dumps(data).encode()
    fake_http(monkeypatch, payload)
    with pytest.raises((ValueError, KeyError)):
        worker.infer(request(), "dummy-key", MODEL, 5)


def test_redirects_are_refused_before_auth_can_follow():
    req = worker.urllib.request.Request(worker.ENDPOINT)
    with pytest.raises(worker.urllib.error.HTTPError):
        worker.NoRedirect().redirect_request(req, None, 307, "redirect", {}, "https://evil.invalid")


@pytest.mark.parametrize("key,model,timeout", [("", MODEL, 5), ("dummy", "bad", 5),
                                              ("dummy", MODEL, float("nan")),
                                              ("dummy", MODEL, 0)])
def test_invalid_configuration_makes_no_network_call(monkeypatch, key, model, timeout):
    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *_: pytest.fail("network admitted"))
    with pytest.raises(ValueError):
        worker.infer(request(), key, model, timeout)


def test_body_limit_precedes_network(monkeypatch):
    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *_: pytest.fail("network admitted"))
    data = request()
    data["system"] = "x" * 20000
    with pytest.raises(ValueError):
        worker.infer(data, "dummy-key", MODEL, 5)


def test_global_attempt_cap_survives_memory_reset(monkeypatch, tmp_path):
    monkeypatch.setattr(planner.app_config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("VISIONFLOW_LIVE_OPENAI_MAX_CALLS", "2")
    assert planner._reserve_api_attempt() is True
    planner._room_attempts.clear()
    assert planner._reserve_api_attempt() is True
    assert planner._reserve_api_attempt() is False
    assert json.loads((tmp_path / "openai-planner-budget.json").read_text())["reserved_calls"] == 2


@pytest.mark.parametrize("bad", ["", '{"version":true,"reserved_calls":0}', '{"version":1,"reserved_calls":true}',
                               '{"version":1,"reserved_calls":-1}', "x" * 513])
def test_corrupt_persisted_counter_fails_closed(monkeypatch, tmp_path, bad):
    monkeypatch.setattr(planner.app_config, "DATA_DIR", tmp_path)
    (tmp_path / "openai-planner-budget.json").write_text(bad)
    with pytest.raises((ValueError, TypeError)):
        planner._reserve_api_attempt()


def test_subprocess_credential_is_env_only_and_no_cli_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(planner.app_config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-sensitive-token")
    monkeypatch.setenv("VAST_API_KEY", "never-forward-this-dummy")
    monkeypatch.setenv("VISIONFLOW_LIVE_PLANNER_PROVIDER", "openai")
    monkeypatch.setattr(planner.config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(planner.config, "PLANNER_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(planner, "_run_cli", lambda *_: pytest.fail("CLI fallback"))
    captured = {}
    def spawn(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(planner.subprocess, "Popen", spawn)
    monkeypatch.setattr(planner, "_communicate_bounded", lambda *_: (1, "", "dummy-sensitive-token", False))
    out = planner.plan("A fresh supported request", "api-private-test-room")
    assert out.status == "unavailable"
    assert "dummy-sensitive-token" not in str(out)
    assert "dummy-sensitive-token" not in repr(captured["argv"])
    assert captured["env"]["OPENAI_API_KEY"] == "dummy-sensitive-token"
    assert "VAST_API_KEY" not in captured["env"] and "HOME" not in captured["env"]
    assert captured["shell"] is False and captured["start_new_session"] is True
    assert json.loads((tmp_path / "openai-planner-budget.json").read_text())["reserved_calls"] == 1


def test_provider_timeout_kills_descendant_and_releases_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(planner.app_config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setattr(planner.config, "PLANNER_TIMEOUT_SECONDS", 0.4)
    fake_worker = tmp_path / "fake_worker.py"
    marker = tmp_path / "descendant.pid"
    fake_worker.write_text("import subprocess,time,pathlib,sys\n"
                          "sys.stdin.read()\n"
                          "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])\n"
                          f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
                          "time.sleep(10)\n")
    monkeypatch.setattr(planner, "Path", lambda *_: SimpleNamespace(with_name=lambda *_: fake_worker))
    start = time.monotonic()
    out = planner._run_openai("A bounded request")
    assert out.status == "unavailable" and out.error == "planner timeout"
    assert time.monotonic() - start < 4
    assert marker.exists()
    pid = int(marker.read_text())
    proc = Path(f"/proc/{pid}/stat")
    # A reparented zombie has no running operation and cannot issue requests.
    assert not proc.exists() or proc.read_text().split()[2] == "Z"
    assert planner._planner_lock.acquire(blocking=False)
    planner._planner_lock.release()


def test_worker_failure_stdout_and_stderr_do_not_disclose_key(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-sensitive-token")
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"invalid")))
    assert worker.main() == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_shared_counter_admission_is_synchronized_and_private(monkeypatch, tmp_path):
    monkeypatch.setattr(planner.app_config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("VISIONFLOW_LIVE_OPENAI_MAX_CALLS", "7")
    assert planner._reserve_api_attempt()
    ledger = tmp_path / "openai-planner-budget.json"
    assert ledger.stat().st_mode & 0o777 == 0o600
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: planner._reserve_api_attempt(), range(20)))
    assert sum(results) == 6
    assert json.loads(ledger.read_text())["reserved_calls"] == 7


def test_counter_symlink_and_exhaustion_prevent_transport(monkeypatch, tmp_path):
    monkeypatch.setattr(planner.app_config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    outside = tmp_path / "unrelated.json"
    outside.write_text('{"version":1,"reserved_calls":0}')
    ledger = tmp_path / "openai-planner-budget.json"
    ledger.symlink_to(outside)
    monkeypatch.setattr(planner.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("transport admitted"))
    out = planner._run_openai("Do not start a request")
    assert out.status == "unavailable"
    assert json.loads(outside.read_text())["reserved_calls"] == 0
    ledger.unlink()
    ledger.write_text('{"version":1,"reserved_calls":50}')
    out = planner._run_openai("Do not start a request")
    assert out.status == "budget_exhausted"
    assert planner._planner_lock.acquire(blocking=False)
    planner._planner_lock.release()


def test_secure_surface_never_selects_subscription_cli(monkeypatch):
    monkeypatch.setenv("VISIONFLOW_LIVE_PLANNER_PROVIDER", "claude_cli")
    monkeypatch.setattr(planner.config, "COOKIE_SECURE", True)
    monkeypatch.setattr(planner.config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(planner, "transport", None)
    monkeypatch.setattr(planner, "_run_cli", lambda *_: pytest.fail("subscription CLI selected"))
    result = planner.plan("Observe a new request", "secure-provider-test")
    assert result.status == "unavailable"


def test_missing_api_credential_has_no_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VISIONFLOW_LIVE_PLANNER_PROVIDER", raising=False)
    monkeypatch.setattr(planner.config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(planner, "transport", None)
    monkeypatch.setattr(planner, "_run_cli", lambda *_: pytest.fail("subscription CLI fallback"))
    result = planner.plan("A new supported task", "no-credential-test")
    assert result.status == "unavailable" and "credential missing" in result.error
