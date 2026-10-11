"""Make the API app importable and isolate test data in a temp directory."""

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "api"))

# Must be set before app.config is imported anywhere.
os.environ.setdefault("VISIONFLOW_DATA_DIR", tempfile.mkdtemp(prefix="visionflow-test-"))
