"""Runtime configuration from environment variables."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

# Repo root = services/api/app/config.py -> three levels up
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(os.environ.get("VISIONFLOW_DATA_DIR", REPO_ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
EVIDENCE_DIR = DATA_DIR / "evidence"
DB_PATH = DATA_DIR / "visionflow.db"

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
PROCESS_FPS = float(os.environ.get("PROCESS_FPS", "5"))
DETECTOR_MODEL = os.environ.get("DETECTOR_MODEL", "yolo11n.pt")
DETECTOR_CONFIDENCE = float(os.environ.get("DETECTOR_CONFIDENCE", "0.35"))
DETECTOR_DEVICE = os.environ.get("DETECTOR_DEVICE", "cpu")

# Minimum fraction of the container-reported frame count that must actually be
# decoded for a run to count as complete. A larger shortfall indicates a
# truncated/corrupt upload and is surfaced as a failed source rather than
# silently reported "processed" (M1-013). Tolerance absorbs the small
# frame-count inaccuracies some codecs report.
DECODE_MIN_COMPLETENESS = float(os.environ.get("DECODE_MIN_COMPLETENESS", "0.9"))

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]

# --- Teacher VLM (M2, docs/VLM_ARCHITECTURE.md) ----------------------------
# Event-routed visual reasoning. The Teacher is called ONLY on eligible
# candidate events for a supported visual predicate, never per frame.
#
# Server-only configuration (separate from user workflow input). The adapter
# targets an operator-set OpenAI-compatible /v1/chat/completions endpoint. When
# VLM_BASE_URL is unset the Teacher is UNAVAILABLE: it returns an explicit
# missing_credentials result and can never confirm a visual action (honest
# blocker, M2-AC3). VLM_API_KEY is optional (only for a private authenticated
# tunnel). Never commit secrets.
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "").strip()
VLM_MODEL = os.environ.get("VLM_MODEL", "").strip()
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
# Bounded connect/read/overall limits, retries and concurrency (AC3a).
VLM_CONNECT_TIMEOUT = float(os.environ.get("VLM_CONNECT_TIMEOUT", "5"))
VLM_READ_TIMEOUT = float(os.environ.get("VLM_READ_TIMEOUT", "20"))
VLM_TIMEOUT_SECONDS = float(os.environ.get("VLM_TIMEOUT_SECONDS", "25"))
VLM_MAX_RETRIES = int(os.environ.get("VLM_MAX_RETRIES", "1"))  # extra attempts on transient errors
VLM_MAX_CONCURRENCY = int(os.environ.get("VLM_MAX_CONCURRENCY", "1"))
VLM_MAX_IMAGE_BYTES = int(os.environ.get("VLM_MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))
VLM_MAX_TOKENS = int(os.environ.get("VLM_MAX_TOKENS", "200"))
# Minimum confidence a classified Teacher response must report before it may
# confirm a visual action (M2-AC3). A classified label missing confidence is an
# invalid response; a classified label below this threshold is non-confirming
# (status ``low_confidence``), kept distinct from an explicit ``unknown``.
VLM_MIN_CONFIDENCE = float(os.environ.get("VLM_MIN_CONFIDENCE", "0.5"))
# Bumped whenever the prompt template or image preprocessing changes so cached
# results from an older prompt/preprocessing version are never reused (M2-AC4).
TEACHER_PROMPT_VERSION = os.environ.get("TEACHER_PROMPT_VERSION", "v1")

# --- VLM router defaults (M2-AC4) ------------------------------------------
# Per-(task,source) minimum seconds between Teacher calls (sampling throttle).
VLM_SAMPLE_INTERVAL_SECONDS = float(os.environ.get("VLM_SAMPLE_INTERVAL_SECONDS", "2"))
# Per-(task,source) cooldown after a call for the same scope.
VLM_COOLDOWN_SECONDS = float(os.environ.get("VLM_COOLDOWN_SECONDS", "5"))
# Maximum real Teacher calls per (task,source) within one process lifetime.
VLM_BUDGET_PER_TASK = int(os.environ.get("VLM_BUDGET_PER_TASK", "50"))
# Versioned result cache bounds.
VLM_CACHE_TTL_SECONDS = float(os.environ.get("VLM_CACHE_TTL_SECONDS", "600"))
VLM_CACHE_MAX_ENTRIES = int(os.environ.get("VLM_CACHE_MAX_ENTRIES", "512"))

# CPU thread caps (limited-RAM host): keep inference/training sequential.
THREAD_COUNT = int(os.environ.get("VISIONFLOW_THREADS", "2"))


# --- Demo-safe mode (DEMO task) --------------------------------------------
# Default TRUE. Demo-safe mode enforces the demo's safety boundary in the
# BACKEND (not just the UI):
#   * the natural-language compiler is never invoked (M2-001 is unresolved);
#   * Teacher-routed visual-predicate workflows cannot be created, enabled, or
#     processed — zero Teacher admission even if a VLM endpoint is configured or
#     a legacy visual workflow row is already enabled. Such a workflow is
#     refused, never silently coerced into a geometric rule;
#   * VLM stats report an explicit disabled state WITHOUT instantiating a real
#     Teacher adapter.
# Set VISIONFLOW_DEMO_SAFE=false to exercise the historical M2 experimental
# paths (the original M2 tests opt out with an explicit fixture).
#
# The guard is FAIL-CLOSED: it is on by default, and ONLY an explicitly
# recognized false value may turn it off. A typo or any other unrecognized
# value (e.g. "treu") keeps the guard ON — an unknown value must never silently
# disable the demo's safety boundary. Recognized true/false spellings are kept
# small and explicit so the behavior is obvious.
_DEMO_SAFE_TRUE = {"1", "true", "yes", "on"}
_DEMO_SAFE_FALSE = {"0", "false", "no", "off"}


def _demo_safe_from_env(name: str = "VISIONFLOW_DEMO_SAFE") -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return True  # default: safe
    value = raw.strip().lower()
    if value in _DEMO_SAFE_FALSE:
        return False
    if value in _DEMO_SAFE_TRUE:
        return True
    # Unrecognized value: keep the guard ON (fail-closed) and make the reason
    # visible rather than silently fail-open.
    _log.warning(
        "Unrecognized %s=%r; keeping demo-safe mode ON. Use one of %s to disable.",
        name,
        raw,
        sorted(_DEMO_SAFE_FALSE),
    )
    return True


DEMO_SAFE = _demo_safe_from_env()

DEMO_SAFE_COMPILE_MESSAGE = (
    "Natural-language compilation is unavailable in demo-safe mode. Build the "
    "rule with the guided controls, then review the server-validated JSON before "
    "activation."
)
DEMO_SAFE_VISUAL_MESSAGE = (
    "Teacher-routed visual-predicate workflows are disabled in demo-safe mode. "
    "Only the real geometric detector / tracker / zone rules run in this demo."
)


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, SNAPSHOT_DIR, EVIDENCE_DIR):
        d.mkdir(parents=True, exist_ok=True)
