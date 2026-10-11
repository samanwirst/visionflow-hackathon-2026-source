#!/usr/bin/env python3
"""Bounded pinned Qwen download inside the one approved instance; never rents.

No credentials required: the pinned Apache-2.0 model is publicly downloadable.
This experimental infrastructure script does not create Teacher labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MAX_MODEL_BYTES = 22 * 1024**3
MAX_NETWORK_BYTES = 25 * 1024**3
SMALL_FILES = {
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.json", "chat_template.jinja", "model.safetensors.index.json",
}


def fetch(url, max_bytes):
    # Public HTTPS requests only. No account or inference authorization header.
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError("Public metadata exceeds its byte limit")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(fetch(
        f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",
        2 * 1024**2,
    ))
    if metadata.get("sha") != REVISION:
        raise RuntimeError("Model revision mismatch; refusing download")
    entries = []
    for item in metadata.get("siblings", []):
        name = item.get("rfilename", "")
        if name not in SMALL_FILES and not (name.startswith("model-") and name.endswith(".safetensors")):
            continue
        if Path(name).name != name or ".." in name:
            raise RuntimeError("Unsafe model asset path")
        size = item.get("size")
        if type(size) is not int or size <= 0:
            raise RuntimeError("Missing model asset size; cannot enforce budget")
        entries.append((name, size, item.get("lfs", {}).get("sha256")))
    expected = sum(size for _, size, _ in entries)
    if not entries or expected > MAX_MODEL_BYTES or not any(name.endswith(".safetensors") for name, _, _ in entries):
        raise RuntimeError("Missing weights or model exceeds its download budget")
    started = time.monotonic()
    receipt = {"repo": REPO, "revision": REVISION, "expected_bytes": expected,
               "maximum_network_bytes": MAX_NETWORK_BYTES, "downloaded_bytes": 0,
               "files": [], "automatic_retries": 0}
    for name, size, expected_sha in entries:
        target = args.output / name
        part = target.with_suffix(target.suffix + ".part")
        if target.exists():
            # Never redownload a completed unverified asset implicitly.
            raise RuntimeError("Output already contains assets; inspect receipt before retry")
        digest = hashlib.sha256()
        actual = 0
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name)}"
        with urllib.request.urlopen(url, timeout=45) as response, part.open("wb") as stream:
            while chunk := response.read(1024**2):
                actual += len(chunk)
                receipt["downloaded_bytes"] += len(chunk)
                if actual > size or receipt["downloaded_bytes"] > MAX_NETWORK_BYTES:
                    raise RuntimeError("Asset or aggregate network byte limit exceeded")
                stream.write(chunk)
                digest.update(chunk)
        if actual != size or (expected_sha and digest.hexdigest() != expected_sha):
            raise RuntimeError("Asset size or SHA256 mismatch; refusing model startup")
        os.replace(part, target)
        receipt["files"].append({"path": name, "bytes": actual,
                                 "sha256": digest.hexdigest(),
                                 "expected_sha256": expected_sha})
        receipt["download_wall_seconds"] = time.monotonic() - started
        (args.output / "download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"completed_asset": name, "bytes": actual,
                          "downloaded_total_bytes": receipt["downloaded_bytes"]}), flush=True)
    print(json.dumps({"pinned_download_complete": True, "repo": REPO,
                      "revision": REVISION, "downloaded_bytes": receipt["downloaded_bytes"],
                      "download_wall_seconds": receipt["download_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
