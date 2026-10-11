#!/usr/bin/env python3
"""Real loopback model probe. Retains actual content, never invents labels."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--question", default="Is the main visible vehicle a bus?")
    parser.add_argument("--labels", default="bus,not_bus,unknown")
    parser.add_argument("--output", type=Path, default=Path(".agents/runtime/local-teacher/smoke.json"))
    args = parser.parse_args()
    labels = args.labels.split(",")
    prompt = (args.question + ' Reply only with a JSON object with exactly label and confidence. '
              + 'Allowed labels: ' + json.dumps(labels) + '. '
              + 'Confidence must be a numeric value from0 to1, your own estimate. '
              + 'Example format: {"label":"unknown","confidence":0.0}. Do not explain.')
    endpoint = urlsplit(args.url)
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1":
        raise SystemExit("Only a private loopback Teacher is supported by this smoke")
    key = os.environ.get("LOCAL_TEACHER_TOKEN", "")
    headers = {"Authorization": "Bearer " + key} if key else {}
    records = []
    with httpx.Client(timeout=httpx.Timeout(75, connect=3), headers=headers, trust_env=False) as client:
        model = client.get(args.url + "/models")
        model.raise_for_status()
        for path in args.images:
            raw = path.read_bytes()
            request = {"model": MODEL, "max_tokens": 64, "temperature": 0,
                       "messages": [{"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": "data:image/" + ("png" if path.suffix.lower() == ".png" else "jpeg") + ";base64," + base64.b64encode(raw).decode()}},
                           {"type": "text", "text": prompt},
                       ]}]}
            t0 = time.monotonic()
            response = client.post(args.url + "/chat/completions", json=request)
            record = {"image": path.name, "image_sha256": hashlib.sha256(raw).hexdigest(),
                      "http_status": response.status_code, "wall_seconds": time.monotonic() - t0}
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                record.update(actual_content=content, usage=result.get("usage"),
                              metrics=result.get("experimental_metrics"))
                try:
                    label = json.loads(content)
                    valid = (isinstance(label, dict) and set(label) == {"label", "confidence"}
                             and label["label"] in labels
                             and type(label["confidence"]) in {int, float}
                             and math.isfinite(label["confidence"]) and 0 <= label["confidence"] <= 1)
                    record["strict_schema_valid"] = valid
                    record["parsed_teacher_label"] = label if valid else None
                except (ValueError, TypeError):
                    record["strict_schema_valid"] = False
                    record["parsed_teacher_label"] = None
            else:
                record["error"] = response.text[:500]
            record["human_verified"] = False
            records.append(record)
    result = {"model": model.json(), "prompt": prompt, "allowed_labels": labels, "records": records,
              "accuracy": "UNVERIFIED: no human-verified held-out labels",
              "rental_cost": "no rented GPU; no external paid inference",
              "local_electricity_cost": "UNVERIFIED; not measured",
              "teacher_workflow_actions": "disabled"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
