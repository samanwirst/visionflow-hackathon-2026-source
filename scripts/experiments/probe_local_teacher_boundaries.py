#!/usr/bin/env python3
"""Probe real private endpoint admission/limits, including abandoned callers."""
import argparse
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import time

import httpx

MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


async def run(image, output):
    raw = image.read_bytes()
    part = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}}
    request = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": [part, {"type": "text", "text": "Describe the image in detail."}]}]}
    token = os.environ.get("LOCAL_TEACHER_TOKEN", "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    results = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:18081", headers=headers, timeout=75, trust_env=False) as client:
        cases = []
        for value in [True, 0, 129, "64"]:
            body = copy.deepcopy(request); body["max_tokens"] = value
            cases.append(("invalid max_tokens " + repr(value), body, 400))
        body = copy.deepcopy(request); body["messages"][0]["content"][0]["image_url"]["url"] = "http://127.0.0.1:9001/private"
        cases.append(("external image URL forbidden", body, 400))
        body = copy.deepcopy(request); body["messages"][0]["content"] = [part] * 3
        cases.append(("three images rejected", body, 400))
        body = copy.deepcopy(request); body["model"] = "unconfigured-model"
        cases.append(("unknown model rejected", body, 400))
        body = copy.deepcopy(request); body["stream"] = True
        cases.append(("streaming rejected", body, 400))
        body = copy.deepcopy(request); body["messages"][0]["content"][0]["image_url"]["url"] = "data:image/jpeg;base64," + base64.b64encode(b"not-an-image").decode()
        cases.append(("invalid encoded image rejected", body, 400))
        for name, body, expected in cases:
            response = await client.post("/v1/chat/completions", json=body)
            results.append({"probe": name, "expected": expected, "actual": response.status_code, "passed": response.status_code == expected})
        response = await client.post("/v1/chat/completions", content=b" " * 2_000_001)
        results.append({"probe": "body byte limit", "expected": 413, "actual": response.status_code, "passed": response.status_code == 413})
        # This is genuine model work, not a mocked sleep. Cancel the caller only
        # after the server has admitted this exact active request.
        task = asyncio.create_task(client.post("/v1/chat/completions", json=request))
        active = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = (await client.get("/v1/models")).json()["experimental"]
            active = state["active_request"]
            if active:
                break
            await asyncio.sleep(0.01)
        if not active:
            raise RuntimeError("No actual in-flight request observed; cannot claim cancellation test")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        response = await client.post("/v1/chat/completions", json=request)
        after = (await client.get("/v1/models")).json()["experimental"]["active_request"]
        results.append({"probe": "cancelled HTTP caller retains actual GPU capacity", "active_request": active,
                        "active_request_after_cancel": after, "expected": 429, "actual": response.status_code,
                        "passed": response.status_code == 429 and after == active})
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            state = (await client.get("/v1/models")).json()["experimental"]
            if state["active_request"] is None:
                break
            await asyncio.sleep(0.2)
        else:
            raise RuntimeError("Actual model generation did not finish in bounded probe")
        results.append({"probe": "capacity restored only after actual generation completion", "passed": True})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"probes": results, "all_passed": all(x["passed"] for x in results)}, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if not all(x["passed"] for x in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".agents/runtime/local-teacher/boundaries.json"))
    args = parser.parse_args()
    asyncio.run(run(args.image, args.output))
