"""End-to-end smoke test with REAL YOLO inference.

Flow: build demo video from real photographic content -> start the API ->
upload -> draw zone -> create + enable workflow -> process -> assert that a
genuine detection produced a zone event, evidence JPEG, and an executed action.

Usage:
    .venv/bin/python scripts/smoke_e2e.py [--api-url http://host:port]

Without --api-url the script starts its own uvicorn server on port 8010
against a temporary data directory, so it never touches real data.

Exit code 0 = pass. Run from repo root or services/api; paths are resolved.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_demo_video import make_demo_video  # noqa: E402


def wait_for(predicate, timeout: float, interval: float = 0.5, desc: str = "condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting for {desc}")


def run_smoke(api_url: str) -> None:
    c = httpx.Client(base_url=api_url, timeout=30.0)

    print("[1/7] health check")
    assert c.get("/api/health").json()["status"] == "ok"

    print("[2/7] build demo video from real photographic content")
    video = make_demo_video(str(REPO_ROOT / "data" / "demo" / "demo_bus.mp4"))

    print("[3/7] upload video")
    with open(video, "rb") as f:
        r = c.post("/api/sources/upload", files={"file": ("demo_bus.mp4", f, "video/mp4")})
    assert r.status_code == 201, r.text
    source = r.json()
    print(f"      source id={source['id']} {source['width']}x{source['height']}")

    print("[4/7] create zone (right 40% of frame = 'Service Bay A')")
    r = c.post(
        "/api/zones",
        json={
            "source_id": source["id"],
            "name": "Service Bay A",
            "polygon": [[0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]],
        },
    )
    assert r.status_code == 201, r.text
    zone = r.json()

    print("[5/7] create + enable workflow: bus enters zone -> notification")
    r = c.post(
        "/api/workflows",
        json={
            "name": "Bus entered Service Bay A",
            "source_id": source["id"],
            "object_class": "bus",
            "trigger": {"type": "object_entered_zone", "zone_id": zone["id"]},
            "action": {"type": "create_notification", "config": {"title": "Bay occupied"}},
            "cooldown_seconds": 5,
        },
    )
    assert r.status_code == 201, r.text
    wf = r.json()
    assert c.post(f"/api/workflows/{wf['id']}/enable").json()["enabled"] is True

    print("[6/7] process video with real YOLO (first run downloads weights)")
    r = c.post(f"/api/sources/{source['id']}/process")
    assert r.status_code == 202, r.text

    def done():
        s = c.get(f"/api/sources/{source['id']}").json()
        return s if s["status"] in ("processed", "failed") else None

    final = wait_for(done, timeout=600, desc="processing")
    assert final["status"] == "processed", f"processing failed: {final['error']}"
    print(
        f"      processed {final['frames_processed']} frames, "
        f"avg inference {final['avg_inference_ms']:.1f} ms"
    )

    print("[7/7] verify event, evidence, and executed action")
    events = c.get("/api/events", params={"source_id": source["id"]}).json()
    enter = [e for e in events if e["type"] == "object_entered_zone"]
    assert enter, f"no enter events; events={events}"
    ev = enter[0]
    assert ev["class_name"] == "bus"
    assert ev["action_status"] == "done", ev

    img = c.get(ev["evidence_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    actions = c.get(f"/api/events/{ev['id']}/actions").json()
    assert actions and actions[0]["status"] == "success", actions
    notifications = c.get("/api/notifications").json()
    assert any(n["event_id"] == ev["id"] for n in notifications)

    print(
        f"\nSMOKE PASS: event id={ev['id']} '{ev['detail']}' "
        f"at t={ev['video_timestamp']:.1f}s conf={ev['confidence']:.2f}, "
        f"evidence + notification persisted."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=None)
    args = parser.parse_args()

    if args.api_url:
        run_smoke(args.api_url)
        return 0

    port = 8010
    data_dir = tempfile.mkdtemp(prefix="visionflow-smoke-")
    env = {**os.environ, "VISIONFLOW_DATA_DIR": data_dir}
    server = subprocess.Popen(
        [
            str(REPO_ROOT / "services" / "api" / ".venv" / "bin" / "uvicorn"),
            "app.main:app", "--port", str(port), "--host", "127.0.0.1",
        ],
        cwd=REPO_ROOT / "services" / "api",
        env=env,
    )
    try:
        wait_for(
            lambda: _health_ok(f"http://127.0.0.1:{port}"), timeout=30, desc="API startup"
        )
        run_smoke(f"http://127.0.0.1:{port}")
        return 0
    finally:
        server.terminate()
        server.wait(timeout=10)


def _health_ok(url: str) -> bool:
    try:
        return httpx.get(f"{url}/api/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


if __name__ == "__main__":
    sys.exit(main())
