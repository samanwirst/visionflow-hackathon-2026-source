"""Teacher transport wall-clock deadline tests (M2-003).

These use genuine httpx + httpcore execution over a ``MockTransport`` whose
response body is delivered in small chunks (each within the per-read timeout) but
whose *total* transfer exceeds the overall deadline. Without a wall-clock
deadline across the active transport and body consumption, httpx's per-read
timeouts reset on every chunk and the request runs far past the configured
``overall_timeout``. The adapter must stop at the deadline and return a
non-confirming ``timeout`` instead of consuming a late response.

This does NOT verify real visual inference — the provider is mocked — but it does
exercise the real streaming/body-read path rather than a fake response object."""

import json
import threading
import time

import httpx

from app.teacher import OpenAICompatibleTeacher


def _envelope_bytes() -> bytes:
    env = {
        "choices": [
            {"message": {"content": json.dumps({"label": "yes", "confidence": 0.9})}}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    return json.dumps(env).encode()


class _TrickleStream(httpx.SyncByteStream):
    """Delivers the body in ``n_chunks`` pieces, sleeping ``delay`` between each.

    Each gap is within the per-read timeout, so no single read times out; only the
    cumulative wall-clock time exceeds the overall deadline."""

    def __init__(self, body: bytes, n_chunks: int, delay: float) -> None:
        size = max(1, (len(body) + n_chunks - 1) // n_chunks)
        self._chunks = [body[i : i + size] for i in range(0, len(body), size)]
        self._delay = delay

    def __iter__(self):
        for c in self._chunks:
            time.sleep(self._delay)
            yield c

    def close(self) -> None:
        pass


def _make_fake_post(done_box, *, delay: float, n_chunks: int):
    body = _envelope_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_TrickleStream(body, n_chunks, delay),
        )

    def fake_post(url, json=None, headers=None, timeout=None):
        # Real httpx client over a mock transport: client.post reads the full body
        # synchronously, incurring the trickle delay — exactly what the adapter's
        # deadline guard must bound.
        client = httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout)
        try:
            resp = client.post(url, json=json, headers=headers)
            done_box["n"] += 1
            return resp
        finally:
            client.close()

    return fake_post


def test_phase_compliant_trickle_exceeds_overall_deadline(monkeypatch):
    done_box = {"n": 0}
    # 18 chunks * 25 ms ~= 0.45 s total transfer; each gap < the 50 ms read timeout.
    monkeypatch.setattr(httpx, "post", _make_fake_post(done_box, delay=0.025, n_chunks=18))
    t = OpenAICompatibleTeacher(
        api_base="https://provider.example/v1", api_key="k", model="vlm",
        prompt_version="v1", overall_timeout=0.1, read_timeout=0.05,
        connect_timeout=0.05, max_retries=0,
    )
    start = time.perf_counter()
    r = t.classify(b"img", "is the hood open?", ["yes", "no"])
    elapsed = time.perf_counter() - start
    assert r.status == "timeout", r
    assert r.confirmed is False
    # Stopped at the ~0.1 s deadline, not at the ~0.45 s full transfer.
    assert elapsed < 0.35, elapsed


def test_fast_streamed_response_still_confirms(monkeypatch):
    # Control: when the streamed body fits inside the deadline, the same genuine
    # path reads the full response and confirms.
    done_box = {"n": 0}
    monkeypatch.setattr(httpx, "post", _make_fake_post(done_box, delay=0.001, n_chunks=4))
    t = OpenAICompatibleTeacher(
        api_base="https://provider.example/v1", api_key="k", model="vlm",
        prompt_version="v1", overall_timeout=2.0, max_retries=0,
    )
    r = t.classify(b"img", "q", ["yes", "no"])
    assert r.status == "ok"
    assert r.confirmed is True
    assert done_box["n"] == 1


# -- M2-003: a returned timeout must not free capacity while the request is
# still in flight. At concurrency 1, four sequential trickling requests must
# never produce more than ONE active transport worker/body-read. --------------

def test_timeout_does_not_free_capacity_for_concurrent_workers(monkeypatch):
    lock = threading.Lock()
    release = threading.Event()
    state = {"active": 0, "peak": 0, "started": 0}
    body = json.dumps(
        {"choices": [{"message": {"content": '{"label":"yes","confidence":0.9}'}}]}
    ).encode()

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            with lock:
                state["active"] += 1
                state["started"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                for byte in body:
                    if release.wait(0.02):
                        yield body
                        break
                    yield bytes([byte])
            finally:
                with lock:
                    state["active"] -= 1

        def close(self):
            pass

    def post(url, **kwargs):
        with httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=Stream())),
            timeout=kwargs.pop("timeout"),
        ) as client:
            return client.post(url, **kwargs)

    monkeypatch.setattr(httpx, "post", post)
    teacher = OpenAICompatibleTeacher(
        "https://qa.invalid/v1", "", "qa-vlm", "v1",
        overall_timeout=0.06, read_timeout=0.05, max_retries=0, max_concurrency=1,
    )
    try:
        statuses = [
            teacher.classify(b"img", "hood open?", ["yes", "no"]).status
            for _ in range(4)
        ]
        # All four deadline out (non-confirming), but capacity was never freed
        # while a worker remained in flight: at most one active/started worker.
        assert statuses == ["timeout"] * 4
        assert state["peak"] <= 1, state
        assert state["started"] <= 1, state
    finally:
        release.set()
        deadline = time.monotonic() + 2
        while state["active"] and time.monotonic() < deadline:
            time.sleep(0.005)
