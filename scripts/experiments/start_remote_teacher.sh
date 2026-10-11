#!/usr/bin/env bash
# Run inside the explicitly approved rented instance. This script never rents.
set -euo pipefail

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo 'Set VLLM_API_KEY securely in the remote ignored environment first.' >&2
  exit 1
fi
if [[ -z "${LOCAL_TEACHER_MODEL_DIR:-}" ]]; then
  echo 'Set LOCAL_TEACHER_MODEL_DIR to the completed bounded pinned download.' >&2
  exit 1
fi
python - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
version = importlib.metadata.version('vllm')
if version != '0.31.0':
    raise SystemExit('Expected pinned vLLM 0.31.0, found ' + version)
directory = Path(os.environ['LOCAL_TEACHER_MODEL_DIR'])
receipt = json.loads((directory / 'download-receipt.json').read_text())
if receipt.get('repo') != 'Qwen/Qwen2.5-VL-7B-Instruct' or receipt.get('revision') != 'cc594898137f460bfe9f0759e9844b3ce807cfb5':
    raise SystemExit('Pinned local model identity mismatch')
if receipt.get('downloaded_bytes') != receipt.get('expected_bytes'):
    raise SystemExit('Incomplete pinned model download')
if any(not (directory / row['path']).is_file() or (directory / row['path']).stat().st_size != row['bytes'] for row in receipt['files']):
    raise SystemExit('Missing or changed local model asset')
PY
# Environment key is read by vLLM; it is never placed in command-line arguments.
# Only SSH is published. Never add an inference -p port mapping to the container.
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec vllm serve "$LOCAL_TEACHER_MODEL_DIR" \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":262144}'
