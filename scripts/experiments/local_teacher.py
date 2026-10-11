#!/usr/bin/env python3
"""Private experimental image Teacher harness; never imports VisionFlow actions.

Serves a small, pinned, genuine Transformers image model on loopback only.
It returns the actual generated content; no parser repairs or synthetic labels.
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForVision2Seq, AutoProcessor, StoppingCriteria, StoppingCriteriaList
import uvicorn

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"
MAX_BODY = 2_000_000
MAX_IMAGE_BYTES = 650_000
MAX_IMAGES = 2
MAX_GENERATION_SECONDS = 60
Image.MAX_IMAGE_PIXELS = 4_000_000
token = os.environ.get("LOCAL_TEACHER_TOKEN", "")
scoring_labels = [x.strip() for x in os.environ.get("LOCAL_TEACHER_SCORING_LABELS", "").split(",") if x.strip()]
if scoring_labels and (not 2 <= len(scoring_labels) <= 8 or len(set(scoring_labels)) != len(scoring_labels)
                       or any(not re.fullmatch(r"[A-Za-z0-9 _-]{1,40}", x) for x in scoring_labels)):
    raise SystemExit("Invalid explicitly configured candidate scoring labels")
evidence_dir = Path(os.environ.get("LOCAL_TEACHER_EVIDENCE", ".agents/runtime/local-teacher/evidence"))
evidence_dir.mkdir(parents=True, exist_ok=True)
capacity = threading.BoundedSemaphore(1)
active_lock = threading.Lock()
active_request = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="real-teacher")
processor = None
model = None
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class Deadline(StoppingCriteria):
    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, input_ids, scores, **kwargs):
        return time.monotonic() >= self.deadline


def load_model():
    global processor, model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: this GPU experiment must not silently switch to CPU")
    torch.set_num_threads(2)
    t0 = time.monotonic()
    source = os.environ.get("LOCAL_TEACHER_MODEL_DIR", MODEL_ID)
    if source != MODEL_ID:
        local = Path(source)
        manifest = json.loads((local / "source.json").read_text())
        if manifest["model"] != MODEL_ID or manifest["revision"] != REVISION:
            raise RuntimeError("Local model manifest does not match pinned identity")
        weights = local / "model.safetensors"
        if hashlib.file_digest(weights.open("rb"), "sha256").hexdigest() != manifest["weights_sha256"]:
            raise RuntimeError("Local model weights do not match pinned download")
    processor = AutoProcessor.from_pretrained(source, revision=REVISION)
    # Bound each image to one 512px visual region instead of a 2048px tiling budget.
    processor.image_processor.size = {"longest_edge": 512}
    model = AutoModelForVision2Seq.from_pretrained(
        source, revision=REVISION, torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa", trust_remote_code=False,
        device_map="cuda", low_cpu_mem_usage=True,
    ).eval()
    torch.cuda.synchronize()
    metadata = {
        "model": MODEL_ID, "revision": REVISION, "license": "Apache-2.0",
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "load_seconds": time.monotonic() - t0,
        "gpu_allocated_bytes": torch.cuda.memory_allocated(),
        "gpu_reserved_bytes": torch.cuda.memory_reserved(),
        "max_images": MAX_IMAGES, "max_concurrency": 1,
        "input_longest_edge": 512, "experimental": True,
        "teacher_actions_enabled": False,
        "output_mode": "candidate_sequence_scoring" if scoring_labels else "raw_generation",
        "scoring_labels": scoring_labels,
    }
    (evidence_dir / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


@app.on_event("startup")
def startup():
    load_model()


def check_private(request):
    # Extra defense; the executable also refuses any non-loopback listener.
    if request.client and request.client.host not in {"127.0.0.1", "::1"}:
        raise HTTPException(403, "loopback only")
    if token and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
        raise HTTPException(401, "authentication required")


@app.get("/v1/models")
def models(request: Request):
    check_private(request)
    with active_lock:
        current = active_request
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local-experiment", "revision": REVISION}],
            "experimental": {"active_request": current, "max_concurrency": 1,
                             "output_mode": "candidate_sequence_scoring" if scoring_labels else "raw_generation",
                             "scoring_labels": scoring_labels}}


def parse_payload(payload):
    if payload.get("model") != MODEL_ID:
        raise HTTPException(400, "wrong pinned model id")
    if payload.get("stream", False):
        raise HTTPException(400, "streaming unsupported")
    max_tokens = payload.get("max_tokens", 96)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise HTTPException(400, "max_tokens must be an integer in [1,128]")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 3:
        raise HTTPException(400, "one to three messages required")
    chat, images = [], []
    characters = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise HTTPException(400, "invalid message role")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise HTTPException(400, "invalid message content")
        prepared = []
        for part in content:
            if not isinstance(part, dict):
                raise HTTPException(400, "invalid content part")
            if part.get("type") == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise HTTPException(400, "text required")
                characters += len(text)
                prepared.append({"type": "text", "text": text})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if not isinstance(url, str) or not url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
                    raise HTTPException(400, "only inline JPEG/PNG data URLs allowed; no external fetching")
                try:
                    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                    if len(raw) > MAX_IMAGE_BYTES:
                        raise HTTPException(413, "encoded image exceeds limit")
                    image = Image.open(io.BytesIO(raw))
                    if image.format not in {"JPEG", "PNG"}:
                        raise HTTPException(400, "decoded image must be JPEG or PNG")
                    if image.width * image.height > 4_000_000:
                        raise HTTPException(413, "image pixel limit exceeded")
                    image.load()
                    image = image.convert("RGB")
                    image.thumbnail((512, 512))
                except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
                    raise HTTPException(400, "invalid image") from None
                images.append(image)
                prepared.append({"type": "image"})
            else:
                raise HTTPException(400, "unsupported content part")
        chat.append({"role": message["role"], "content": prepared})
    if not 1 <= len(images) <= MAX_IMAGES or characters > 4000:
        raise HTTPException(400, "require one or two images and at most 4000 text characters")
    return chat, images, max_tokens


def infer(chat, images, max_tokens, request_id):
    global active_request
    # This slot is released by the real underlying worker, never by an HTTP timeout.
    t0 = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
        if scoring_labels:
            prompt += '{"label":"'
        inputs = processor(text=prompt, images=images, return_tensors="pt")
        inputs = inputs.to("cuda")
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(torch.bfloat16)
        input_tokens = int(inputs["input_ids"].shape[-1])
        deadline = time.monotonic() + MAX_GENERATION_SECONDS
        scoring = None
        with torch.inference_mode():
            if scoring_labels:
                log_likelihoods, actual_tokens, candidate_token_ids = [], 0, []
                for label in scoring_labels:
                    if time.monotonic() >= deadline:
                        raise HTTPException(504, "actual scoring deadline reached")
                    # Include the closing quote so candidates represent complete
                    # labels, rather than merely matching a candidate prefix.
                    ids = processor.tokenizer.encode(label + '"', add_special_tokens=False)
                    suffix = torch.tensor([ids], device="cuda", dtype=torch.long)
                    candidate = dict(inputs)
                    candidate["input_ids"] = torch.cat((inputs["input_ids"], suffix), dim=1)
                    candidate["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(suffix)), dim=1)
                    logits = model(**candidate, use_cache=False).logits[:, input_tokens - 1:input_tokens + len(ids) - 1, :]
                    token_log_probabilities = torch.log_softmax(logits.float(), dim=-1).gather(-1, suffix.unsqueeze(-1)).squeeze(-1)
                    log_likelihoods.append(float(token_log_probabilities.sum().item()))
                    actual_tokens += input_tokens + len(ids)
                    candidate_token_ids.append(ids)
                probabilities = torch.softmax(torch.tensor(log_likelihoods, dtype=torch.float64), dim=0).tolist()
                winner = max(range(len(probabilities)), key=probabilities.__getitem__)
                text = json.dumps({"label": scoring_labels[winner], "confidence": probabilities[winner]})
                generation_tokens = 0
                scoring = {"method": "candidate_sequence_softmax", "uncalibrated": True,
                           "formula": "s(label)=sum_t log P(candidate_token_t | image(s), prompt, prior_candidate_tokens); p(label)=exp(s)/sum_candidates exp(s)",
                           "candidate_labels": scoring_labels, "candidate_token_ids": candidate_token_ids,
                           "candidate_log_likelihoods": log_likelihoods, "normalized_probabilities": probabilities,
                           "model_forward_calls": len(scoring_labels), "actual_model_processed_tokens": actual_tokens,
                           "output_is_generated_json": False}
            else:
                generated = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    stopping_criteria=StoppingCriteriaList([Deadline(deadline)]),
                )
                output = generated[:, input_tokens:]
                text = processor.batch_decode(output, skip_special_tokens=True)[0]
                generation_tokens = int(output.shape[-1])
        torch.cuda.synchronize()
        metrics = {
            "request_id": request_id, "model": MODEL_ID, "revision": REVISION,
            "image_count": len(images), "input_tokens": input_tokens,
            "output_tokens": generation_tokens,
            "inference_seconds": time.monotonic() - t0,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "deadline_reached": time.monotonic() >= deadline,
            "scoring": scoring,
        }
        with (evidence_dir / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if metrics["deadline_reached"]:
            raise HTTPException(504, "actual generation deadline reached; no label fabricated")
        return {
            "id": request_id, "object": "chat.completion", "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "length" if generation_tokens == max_tokens else "stop"}],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": generation_tokens,
                      "total_tokens": input_tokens + generation_tokens},
            "experimental_metrics": metrics,
        }
    finally:
        with active_lock:
            active_request = None
        capacity.release()


@app.post("/v1/chat/completions")
async def completion(request: Request):
    global active_request
    check_private(request)
    body = bytearray()
    try:
        async with asyncio.timeout(5):
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_BODY:
                    raise HTTPException(413, "request body exceeds limit")
                body.extend(chunk)
    except TimeoutError:
        raise HTTPException(408, "request body read deadline reached") from None
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("object required")
    except (ValueError, UnicodeError):
        raise HTTPException(400, "JSON object required") from None
    chat, images, max_tokens = parse_payload(payload)
    if not capacity.acquire(blocking=False):
        raise HTTPException(429, "one genuine inference already running; no queue")
    request_id = "local-" + uuid.uuid4().hex
    with active_lock:
        active_request = request_id
    future = asyncio.get_running_loop().run_in_executor(executor, infer, chat, images, max_tokens, request_id)
    # Cancelled callers do not cancel/abandon/release the underlying inference slot.
    return await asyncio.shield(future)


if __name__ == "__main__":
    host = os.environ.get("LOCAL_TEACHER_HOST", "127.0.0.1")
    if host != "127.0.0.1":
        raise SystemExit("Refusing non-loopback listener")
    uvicorn.run(app, host=host, port=int(os.environ.get("LOCAL_TEACHER_PORT", "18081")), workers=1, access_log=False)
