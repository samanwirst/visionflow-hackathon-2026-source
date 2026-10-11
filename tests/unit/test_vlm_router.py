"""VLM router tests (M2-AC4): caching, sampling, cooldown, budget, in-flight
dedup, versioned invalidation and counters. No real provider is used; a fake
teacher counts invocations so we can prove unnecessary calls are suppressed."""

import threading

from app.teacher import TeacherAdapter, TeacherResult
from app.vlm_router import VersionedTTLCache, VlmRouter


class FakeTeacher(TeacherAdapter):
    provider_name = "fake"
    model = "fake-vlm"

    def __init__(self, status="ok", label="yes"):
        self._status = status
        self._label = label
        self.calls = 0
        self._lock = threading.Lock()

    def classify(self, image_bytes, prompt, labels, on_attempt=None):
        with self._lock:
            self.calls += 1
        return TeacherResult(
            status=self._status, label=self._label, confidence=0.8,
            model=self.model, latency_ms=1.0, usage={"total_tokens": 5},
        )


class AttemptTeacher(TeacherAdapter):
    """Reports outbound attempts via the router's guard (M2-AC4). It retries once
    on a simulated transient 5xx, so a budget of 1 must block the retry."""

    provider_name = "attempt"
    model = "attempt-vlm"

    def __init__(self, max_retries=1):
        self.max_retries = max_retries
        self.attempts = 0

    def classify(self, image_bytes, prompt, labels, on_attempt=None):
        last = TeacherResult(status="error", model=self.model, error="no attempt")
        for i in range(self.max_retries + 1):
            if on_attempt is not None and not on_attempt():
                last = TeacherResult(
                    status="error", model=self.model,
                    error="budget exhausted before attempt",
                )
                break
            self.attempts += 1
            if i == 0:  # first attempt is a transient failure
                last = TeacherResult(status="error", model=self.model, error="503")
                continue
            return TeacherResult(
                status="ok", label="yes", confidence=0.9, model=self.model,
                latency_ms=1.0, usage={"total_tokens": 5},
            )
        return last


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_router(teacher, clock=None, **kw):
    clk = clock or Clock()
    return VlmRouter(teacher_provider=lambda: teacher, clock=clk, **kw)


def route(router, roi=b"roi-a", scope="scope", task="task"):
    return router.route(
        task_key=task, scope=scope, prompt="is the hood open?",
        labels=["yes", "no"], image_bytes=roi,
    )


def test_cache_hit_suppresses_second_call():
    t = FakeTeacher()
    r = make_router(t)
    d1 = route(r)
    d2 = route(r)  # identical ROI
    assert d1.status == "called"
    assert d2.status == "cache_hit"
    assert t.calls == 1
    assert r.counters.cache_hits == 1


def test_changed_roi_invalidates_reuse():
    t = FakeTeacher()
    r = make_router(t, budget_per_task=10, sample_interval_s=0, cooldown_s=0)
    route(r, roi=b"roi-a")
    route(r, roi=b"roi-b")  # different ROI -> different key -> new call
    assert t.calls == 2


def test_budget_exhaustion_suppresses_without_call():
    t = FakeTeacher()
    r = make_router(t, budget_per_task=1, sample_interval_s=0, cooldown_s=0)
    d1 = route(r, roi=b"roi-a")
    d2 = route(r, roi=b"roi-b")
    assert d1.status == "called"
    assert d2.status == "suppressed_budget"
    assert t.calls == 1
    assert r.counters.suppressed_budget == 1


def test_sampling_interval_suppresses():
    t = FakeTeacher()
    clk = Clock()
    r = make_router(t, clock=clk, budget_per_task=10, sample_interval_s=100, cooldown_s=0)
    route(r, roi=b"roi-a")  # call at t=1000
    clk.t += 1  # still within interval
    d2 = route(r, roi=b"roi-b")
    assert d2.status == "suppressed_sampling"
    assert t.calls == 1


def test_sampling_allows_after_interval():
    t = FakeTeacher()
    clk = Clock()
    r = make_router(t, clock=clk, budget_per_task=10, sample_interval_s=10, cooldown_s=0)
    route(r, roi=b"roi-a")
    clk.t += 20
    d2 = route(r, roi=b"roi-b")
    assert d2.status == "called"
    assert t.calls == 2


def test_cooldown_suppresses_same_scope():
    t = FakeTeacher()
    clk = Clock()
    r = make_router(t, clock=clk, budget_per_task=10, sample_interval_s=0, cooldown_s=100)
    route(r, roi=b"roi-a", scope="s1")
    clk.t += 1
    d2 = route(r, roi=b"roi-b", scope="s1")
    assert d2.status == "suppressed_cooldown"
    assert t.calls == 1


def test_counters_track_outcomes():
    t = FakeTeacher()
    r = make_router(t, budget_per_task=10, sample_interval_s=0, cooldown_s=0)
    route(r, roi=b"roi-a")
    route(r, roi=b"roi-a")  # cache hit
    snap = r.counters.snapshot()
    assert snap["calls"] == 1
    assert snap["cache_hits"] == 1
    assert snap["confirmed"] == 1


def test_failure_status_not_cached_and_not_confirmed():
    t = FakeTeacher(status="error", label=None)
    r = make_router(t, budget_per_task=10, sample_interval_s=0, cooldown_s=0)
    d1 = route(r, roi=b"roi-a")
    assert d1.confirmed is False
    # transient error not cached: a second identical ROI calls again
    d2 = route(r, roi=b"roi-a")
    assert d2.status == "called"
    assert t.calls == 2
    assert r.counters.failed == 2


def test_ttl_cache_expires():
    clk = Clock()
    cache = VersionedTTLCache(max_entries=10, ttl_seconds=5)
    res = TeacherResult(status="ok", label="yes")
    cache.put("k", res, clk())
    assert cache.get("k", clk()) is not None
    clk.t += 6
    assert cache.get("k", clk()) is None


def test_cache_lru_eviction():
    cache = VersionedTTLCache(max_entries=2, ttl_seconds=100)
    res = TeacherResult(status="ok", label="yes")
    cache.put("a", res, 0)
    cache.put("b", res, 0)
    cache.put("c", res, 0)  # evicts "a"
    assert cache.get("a", 0) is None
    assert cache.get("c", 0) is not None
    assert cache.evictions == 1


def test_unavailable_teacher_route_not_confirmed():
    from app.teacher import UnavailableTeacher

    r = make_router(UnavailableTeacher(), budget_per_task=10, sample_interval_s=0, cooldown_s=0)
    d = route(r)
    assert d.status == "called"
    assert d.result.status == "missing_credentials"
    assert d.confirmed is False


# -- M2-004: budget accounting at the transport boundary ----------------------

def test_budget_consumes_every_outbound_attempt(monkeypatch):
    # budget_per_task=1 with a teacher that would retry: the retry must be blocked
    # because the first outbound attempt already consumed the only budget unit.
    t = AttemptTeacher(max_retries=1)
    r = make_router(t, budget_per_task=1, sample_interval_s=0, cooldown_s=0)
    d = route(r, roi=b"roi-a")
    assert t.attempts == 1  # retry blocked by budget
    assert r.counters.outbound_attempts == 1
    assert r.counters.adapter_invocations == 1
    assert d.result.status == "error"  # transient failure, never confirmed
    assert d.confirmed is False


def test_budget_allows_retry_within_budget():
    t = AttemptTeacher(max_retries=1)
    r = make_router(t, budget_per_task=5, sample_interval_s=0, cooldown_s=0)
    d = route(r, roi=b"roi-a")
    assert t.attempts == 2  # 503 then valid
    assert r.counters.outbound_attempts == 2
    assert r.counters.completed == 1
    assert d.confirmed is True


def test_unavailable_contributes_zero_requests():
    from app.teacher import UnavailableTeacher

    r = make_router(UnavailableTeacher(), budget_per_task=10, sample_interval_s=0, cooldown_s=0)
    route(r)
    snap = r.counters.snapshot()
    assert snap["unavailable"] == 1
    assert snap["outbound_attempts"] == 0  # no actual request made
    assert snap["calls"] == 0              # no completed classification
    assert snap["adapter_invocations"] == 1


# -- M2-005: in-flight dedup precedes the sampling/budget/cooldown gates -------

def test_inflight_duplicate_shares_before_sampling_gate():
    import time as _time

    release = threading.Event()

    class BlockingTeacher(TeacherAdapter):
        provider_name = "block"
        model = "block-vlm"

        def __init__(self):
            self.calls = 0

        def classify(self, image_bytes, prompt, labels, on_attempt=None):
            self.calls += 1
            release.wait(2.0)
            return TeacherResult(status="ok", label="yes", confidence=0.9,
                                 model=self.model, latency_ms=1.0)

    t = BlockingTeacher()
    # sampling=2, cooldown=5: a naive design would reject the concurrent duplicate
    # as suppressed_sampling; dedup-before-gates must instead share the call.
    r = make_router(t, clock=_time.monotonic, budget_per_task=10,
                    sample_interval_s=2, cooldown_s=5)
    results = {}

    def worker(name):
        results[name] = route(r, roi=b"same-roi")

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    _time.sleep(0.3)  # let the first own the in-flight slot and block in classify
    t2.start()
    _time.sleep(0.3)  # let the duplicate attach to the in-flight slot (dedup wait)
    release.set()     # now unblock the owner's classify; it publishes the result
    t1.join(3)
    t2.join(3)
    statuses = sorted(d.status for d in results.values())
    assert statuses == ["called", "deduped"]
    assert t.calls == 1  # the duplicate shared the single call
    assert all(d.confirmed for d in results.values())


# -- M2-004 (re-opened): real-adapter preflight consumes no attempt/budget -----

def test_real_adapter_preflight_consumes_no_budget(monkeypatch):
    import json as _json

    import httpx

    from app.teacher import OpenAICompatibleTeacher

    http = {"n": 0}

    def forbidden_post(*a, **k):
        http["n"] += 1
        raise AssertionError("oversize preflight must make no outbound request")

    monkeypatch.setattr(httpx, "post", forbidden_post)
    teacher = OpenAICompatibleTeacher(
        api_base="https://p/v1", api_key="k", model="vlm", prompt_version="v1",
        max_image_bytes=2,
    )
    r = make_router(teacher, budget_per_task=1, sample_interval_s=0, cooldown_s=0)

    d1 = route(r, roi=b"oversized")  # 9 bytes > max_image_bytes=2 -> local reject
    assert http["n"] == 0
    assert d1.result.status == "error"
    assert r.counters.outbound_attempts == 0
    assert r.counters.completed == 0
    assert r.counters.calls == 0
    assert r.counters.failed == 1

    # Budget was NOT consumed by the preflight rejection: an admissible image now
    # routes and actually calls out, consuming exactly one unit.
    class _R:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [
                    {"message": {"content": _json.dumps({"label": "yes", "confidence": 0.9})}}
                ],
                "usage": {"total_tokens": 7},
            }

    def ok_post(*a, **k):
        http["n"] += 1
        return _R()

    monkeypatch.setattr(httpx, "post", ok_post)
    d2 = route(r, roi=b"x")  # 1 byte <= max_image_bytes=2
    assert http["n"] == 1
    assert d2.result.status == "ok"
    assert r.counters.outbound_attempts == 1
    assert r.counters.completed == 1
    assert r.counters.calls == 1


# -- M2-007 (re-opened): admission revoked at the transport boundary -----------

def test_admit_false_cancels_without_outbound(monkeypatch):
    import httpx

    from app.teacher import OpenAICompatibleTeacher

    http = {"n": 0}

    def forbidden_post(*a, **k):
        http["n"] += 1
        raise AssertionError("no outbound request when admission is revoked")

    monkeypatch.setattr(httpx, "post", forbidden_post)
    teacher = OpenAICompatibleTeacher(
        api_base="https://p/v1", api_key="k", model="vlm", prompt_version="v1",
    )
    r = make_router(teacher, budget_per_task=5, sample_interval_s=0, cooldown_s=0)

    d = r.route(
        task_key="t", scope="s", prompt="q", labels=["yes", "no"],
        image_bytes=b"roi", admit=lambda: False,
    )
    assert d.status == "cancelled"
    assert d.confirmed is False
    assert http["n"] == 0
    assert r.counters.outbound_attempts == 0
    assert r.counters.completed == 0
    assert r.counters.calls == 0
    assert r.counters.by_status.get("cancelled") == 1
