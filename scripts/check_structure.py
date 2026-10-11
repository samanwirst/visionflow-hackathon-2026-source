"""Check the repository scaffold without third-party dependencies."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = (
    ".github/workflows/ci.yml",
    ".agents/PROTOCOL.md",
    ".agents/STATE.md",
    ".agents/HANDOFF.md",
    ".agents/reviews/M1.md",
    ".agents/reviews/M2.md",
    ".agents/reviews/M3.md",
    ".agents/reviews/M4.md",
    "docs/PRD.md",
    "docs/TRACEABILITY.md",
    "docs/ARCHITECTURE.md",
    "docs/METRICS.md",
    "docs/DEMO.md",
    "docs/LICENSES.md",
    ".env.example",
    "docker-compose.yml",
    "Makefile",
    "README.md",
    "CLAUDE.md",
    "AGENTS.md",
)

REQUIRED_DIRECTORIES = (
    "docs/source",
    "apps/web",
    "services/api",
    "packages/vision",
    "packages/workflows",
    "packages/autolearn",
    "packages/integrations",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "scripts",
)


def main() -> int:
    errors = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing file: {relative}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"Empty file: {relative}")

    for relative in REQUIRED_DIRECTORIES:
        path = ROOT / relative
        if not path.is_dir():
            errors.append(f"Missing directory: {relative}")
        elif not any(path.iterdir()):
            errors.append(f"Directory has no Git-trackable placeholder: {relative}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("Repository scaffold: OK")
    if not (ROOT / "docs/source/FINAL_TASK_WIUT_HACKATHON.pdf").is_file():
        print("NOTE: original FINAL_TASK_WIUT_HACKATHON.pdf has not been provided.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
