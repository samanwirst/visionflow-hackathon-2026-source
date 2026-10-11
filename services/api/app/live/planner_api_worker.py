"""Tool-free, fixed-destination OpenAI planner child; killed by parent deadline.

Only structured proposal/model/usage/timing are printed. Provider error bodies,
credentials, request headers and exception details never reach stdout/stderr.
"""
from __future__ import annotations

import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.openai.com/v1/chat/completions"
MAX_INPUT_BYTES = 32 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 16 * 1024
MODELS = {"gpt-4.1-mini", "gpt-4.1-mini-2025-04-14",
          "gpt-4.1-nano", "gpt-4.1-nano-2025-04-14"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _reject_constant(value):
    raise ValueError("nonfinite JSON")


def infer(request: dict, key: str, model: str, timeout: float) -> dict:
    if not key or len(key) > 4096 or model not in MODELS:
        raise ValueError("invalid provider configuration")
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 300:
        raise ValueError("invalid timeout")
    if (not isinstance(request, dict) or set(request) != {"text", "system", "schema"}
            or not isinstance(request["text"], str) or len(request["text"]) > 1000
            or not isinstance(request["system"], str)
            or not isinstance(request["schema"], dict)):
        raise ValueError("invalid input")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": request["system"]},
                     {"role": "user", "content": request["text"]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "visionflow_task", "strict": True, "schema": request["schema"]}},
        "max_completion_tokens": 1200, "temperature": 0, "stream": False,
        "n": 1, "store": False,
    }, allow_nan=False).encode("utf-8")
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError("provider request too large")
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    start = time.monotonic()
    with opener.open(req, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("provider status")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("provider response too large")
    decoded = json.loads(raw, parse_constant=_reject_constant)
    choice = decoded["choices"][0]
    if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
        raise ValueError("incomplete or refused output")
    proposal = json.loads(choice["message"]["content"], parse_constant=_reject_constant)
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be an object")
    actual_model = decoded["model"]
    if not isinstance(actual_model, str) or actual_model not in MODELS:
        raise ValueError("unrecognized provider model identity")
    usage = decoded["usage"]
    if (not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"))):
        raise ValueError("invalid measured token usage")
    return {"subtype": "success", "structured_output": proposal, "model": actual_model,
            "usage": {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "duration_ms": (time.monotonic() - start) * 1000}


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("input too large")
        request = json.loads(raw, parse_constant=_reject_constant)
        result = infer(request, os.environ.get("OPENAI_API_KEY", "").strip(),
                       os.environ.get("VISIONFLOW_LIVE_PLANNER_MODEL", "gpt-4.1-mini-2025-04-14"),
                       float(os.environ.get("VISIONFLOW_LIVE_API_TIMEOUT", "60")))
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception:
        # A stable failure code, with no server body/header/exception disclosure.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
