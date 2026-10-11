"""Teacher adapter strict-validation tests (M2-AC3).

These exercise the adapter's response handling with a mocked HTTP transport.
They explicitly do NOT verify real visual inference — a real provider smoke is a
separate, credential-dependent step and is documented as a blocker when no key
is configured. The point here is that malformed / out-of-vocabulary / error
responses can never become a confirmed positive visual condition."""

import time

import httpx
import pytest

from app import teacher as teacher_mod
from app.teacher import OpenAICompatibleTeacher, UnavailableTeacher, get_teacher, reset_teacher


def make_teacher(**kw):
    return OpenAICompatibleTeacher(
        api_base="https://provider.example/v1",
        api_key="test-key",
        model="test-vlm",
        prompt_version="v1",
        **kw,
    )


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _envelope(content, usage=None):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


def test_unavailable_teacher_never_confirms():
    t = UnavailableTeacher()
    r = t.classify(b"img", "is the hood open?", ["yes", "no"])
    assert r.status == "missing_credentials"
    assert r.confirmed is False
    assert t.available is False


def test_valid_response_confirmed(monkeypatch):
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": 0.9}')),
    )
    r = t.classify(b"img", "is the hood open?", ["yes", "no"])
    assert r.status == "ok"
    assert r.label == "yes"
    assert r.confidence == 0.9
    assert r.confirmed is True
    assert r.usage.get("total_tokens") == 13


def test_unknown_label_not_confirmed(monkeypatch):
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _Resp(200, _envelope('{"label": "unknown"}'))
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "unknown"
    assert r.confirmed is False


def test_out_of_vocabulary_label_rejected(monkeypatch):
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _Resp(200, _envelope('{"label": "maybe"}'))
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"
    assert r.confirmed is False


def test_non_json_content_rejected(monkeypatch):
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _Resp(200, _envelope("I think the hood is open!"))
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"


def test_embedded_json_extracted(monkeypatch):
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('Sure: {"label": "no", "confidence": 0.7} done')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "ok"
    assert r.label == "no"


def test_http_error_becomes_error(monkeypatch):
    t = make_teacher()
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(500, None, "upstream boom"))
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "error"
    assert r.confirmed is False


def test_timeout_becomes_timeout(monkeypatch):
    t = make_teacher()

    def boom(*a, **k):
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr(httpx, "post", boom)
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "timeout"
    assert r.confirmed is False


def test_extra_fields_rejected(monkeypatch):
    # A response with unexpected fields is rejected (extra=forbid), so a model
    # cannot smuggle extra data into a confirmed action.
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": 0.9, "x": 1}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"


def test_auth_failure_not_retried(monkeypatch):
    calls = {"n": 0}

    def post(*a, **k):
        calls["n"] += 1
        return _Resp(401, None, "unauthorized")

    t = make_teacher(max_retries=3)
    monkeypatch.setattr(httpx, "post", post)
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "error"
    assert "auth" in (r.error or "").lower()
    assert calls["n"] == 1  # not retried


def test_transient_5xx_is_retried(monkeypatch):
    calls = {"n": 0}

    def post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(503, None, "busy")
        return _Resp(200, _envelope('{"label": "yes", "confidence": 0.9}'))

    t = make_teacher(max_retries=1)
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(teacher_mod.time, "sleep", lambda *_: None)
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "ok"
    assert calls["n"] == 2


# -- M2-002: confidence required + minimum threshold --------------------------

def test_missing_confidence_on_positive_rejected(monkeypatch):
    # A classified (non-unknown) label with no confidence cannot confirm.
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _Resp(200, _envelope('{"label": "yes"}'))
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"
    assert r.confirmed is False


def test_zero_confidence_not_confirmed(monkeypatch):
    t = make_teacher(min_confidence=0.5)
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": 0.0}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "low_confidence"
    assert r.label == "yes"
    assert r.confirmed is False


def test_below_minimum_confidence_not_confirmed(monkeypatch):
    t = make_teacher(min_confidence=0.6)
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": 0.4}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "low_confidence"
    assert r.confirmed is False


def test_at_minimum_confidence_confirms(monkeypatch):
    t = make_teacher(min_confidence=0.5)
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": 0.5}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "ok"
    assert r.confirmed is True


# -- M2-008: confidence must be an actual JSON number (no coercion) ------------

def test_boolean_confidence_rejected(monkeypatch):
    # `true` must NOT be coerced to 1.0 and confirm an action (the reproduction).
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": true}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"
    assert r.confirmed is False
    assert r.confidence is None


def test_false_confidence_rejected(monkeypatch):
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": false}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"
    assert r.confirmed is False


def test_numeric_string_confidence_rejected(monkeypatch):
    # "0.9" is a string, not a JSON number — it must not confirm.
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": "0.9"}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"
    assert r.confirmed is False


def test_nan_confidence_rejected(monkeypatch):
    # json.loads parses NaN by default; a non-finite confidence must be rejected.
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": NaN}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"
    assert r.confirmed is False


def test_infinity_confidence_rejected(monkeypatch):
    t = make_teacher()
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": Infinity}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "invalid_response"
    assert r.confirmed is False


def test_integer_confidence_still_valid(monkeypatch):
    # A genuine JSON integer (1) is a number and remains valid.
    t = make_teacher(min_confidence=0.5)
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, _envelope('{"label": "yes", "confidence": 1}')),
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "ok"
    assert r.confidence == 1.0
    assert r.confirmed is True


# -- M2-003: one overall deadline bounds queue, attempts and backoff ----------

def test_backoff_bounded_by_deadline(monkeypatch):
    # Two read timeouts with a tiny overall deadline must not sleep the full
    # fixed backoff: the whole call stays near the deadline, not 0.5s+.
    t = make_teacher(overall_timeout=0.1, max_retries=1)

    def boom(*a, **k):
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr(httpx, "post", boom)  # real time.sleep left in place
    start = time.perf_counter()
    r = t.classify(b"img", "q", ["yes", "no"])
    elapsed = time.perf_counter() - start
    assert r.status == "timeout"
    assert elapsed < 0.35, elapsed  # would be ~0.5s+ without a deadline-bounded backoff


def test_semaphore_acquire_bounded_by_deadline():
    # If the single concurrency slot is held, acquisition is bounded by the
    # overall deadline and returns a timeout instead of waiting indefinitely.
    t = make_teacher(max_concurrency=1, overall_timeout=0.1)
    assert t._semaphore.acquire()
    try:
        start = time.perf_counter()
        r = t.classify(b"img", "q", ["yes", "no"])
        elapsed = time.perf_counter() - start
        assert r.status == "timeout"
        assert elapsed < 0.5, elapsed
    finally:
        t._semaphore.release()


def test_oversize_image_rejected():
    t = make_teacher(max_image_bytes=10)
    r = t.classify(b"x" * 100, "q", ["yes", "no"])
    assert r.status == "error"
    assert "max_image_bytes" in (r.error or "")


def test_factory_defaults_to_unavailable(monkeypatch):
    # With VLM_BASE_URL unset (repo default) the factory yields an unavailable
    # adapter — an honest blocker, not a silent stub.
    reset_teacher()
    monkeypatch.setattr(teacher_mod.config, "VLM_BASE_URL", "")
    t = get_teacher()
    assert t.available is False
    reset_teacher()


def test_factory_enables_with_base_url(monkeypatch):
    reset_teacher()
    monkeypatch.setattr(teacher_mod.config, "VLM_BASE_URL", "http://127.0.0.1:18080/v1")
    monkeypatch.setattr(teacher_mod.config, "VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    t = get_teacher()
    assert t.available is True
    assert t.provider_name == "openai_compatible"
    reset_teacher()
