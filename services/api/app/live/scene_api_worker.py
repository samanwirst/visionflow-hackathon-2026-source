"""Fixed-destination vision transport; parent kills/reaps on hard deadline."""
from __future__ import annotations

import base64
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

MODEL = "gpt-4.1-mini-2025-04-14"
ENDPOINT = "https://api.openai.com/v1/chat/completions"
MAX_INPUT_BYTES = 1100 * 1024
MAX_RESPONSE_BYTES = 64 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _reject(value):
    raise ValueError("nonfinite JSON")


def infer(request, key, timeout):
    if not key or len(key) > 4096 or not math.isfinite(timeout) or not 0 < timeout <= 35:
        raise ValueError("provider configuration")
    if (not isinstance(request, dict) or set(request) != {"question", "system", "schema", "frames"}
            or not isinstance(request["question"], str) or not 1 <= len(request["question"]) <= 1000
            or not isinstance(request["system"], str) or len(request["system"]) > 4000
            or not isinstance(request["schema"], dict)
            or not isinstance(request["frames"], list) or not 1 <= len(request["frames"]) <= 3):
        raise ValueError("request shape")
    content = [{"type": "text", "text": request["question"]}]
    total_bytes = 0
    for frame in request["frames"]:
        if (not isinstance(frame, dict)
                or set(frame) != {"frame_id", "video_timestamp_seconds", "jpeg_base64"}
                or type(frame["frame_id"]) is not int or not 0 <= frame["frame_id"] < 5000
                or type(frame["video_timestamp_seconds"]) not in (int, float)
                or not math.isfinite(frame["video_timestamp_seconds"])
                or not 0 <= frame["video_timestamp_seconds"] <= 120
                or not isinstance(frame["jpeg_base64"], str)):
            raise ValueError("frame shape")
        image = base64.b64decode(frame["jpeg_base64"], validate=True)
        if not 0 < len(image) <= 256 * 1024 or not image.startswith(b"\xff\xd8"):
            raise ValueError("image bytes")
        total_bytes += len(image)
        content.extend([
            {"type": "text", "text": f"frame_id={frame['frame_id']}; source_seconds={frame['video_timestamp_seconds']:.6f}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame["jpeg_base64"], "detail": "high"}},
        ])
    if total_bytes > 750 * 1024:
        raise ValueError("image budget")
    body = {"model": MODEL, "messages": [{"role": "system", "content": request["system"]},
                                             {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "visionflow_scene", "strict": True, "schema": request["schema"]}},
            "max_completion_tokens": 700, "temperature": 0, "stream": False,
            "n": 1, "store": False}
    encoded = json.dumps(body, allow_nan=False).encode("utf-8")
    # Separately bound the token-bearing text/schema envelope; image base64 is
    # metered as visual input, not as the equivalent number of text tokens.
    text_bytes = len(json.dumps({k: request[k] for k in ("question", "system", "schema")}).encode())
    if text_bytes > 16 * 1024 or len(encoded) > MAX_INPUT_BYTES:
        raise ValueError("request budget")
    req = urllib.request.Request(ENDPOINT, data=encoded, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    start = time.monotonic()
    with opener.open(req, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("provider status")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("response bound")
    value = json.loads(raw, parse_constant=_reject)
    choice = value["choices"][0]
    if (choice.get("finish_reason") != "stop" or choice["message"].get("refusal")
            or value.get("model") != MODEL):
        raise ValueError("incomplete or refused output")
    result = json.loads(choice["message"]["content"], parse_constant=_reject)
    usage = value["usage"]
    if not isinstance(result, dict) or not isinstance(usage, dict):
        raise ValueError("result shape")
    return {"subtype": "success", "structured_output": result, "model": value["model"],
            "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "duration_ms": (time.monotonic() - start) * 1000}


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("request bound")
        result = infer(json.loads(raw, parse_constant=_reject),
            os.environ.get("OPENAI_API_KEY", "").strip(),
            float(os.environ.get("VISIONFLOW_LIVE_API_TIMEOUT", "35")))
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
