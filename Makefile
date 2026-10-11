PYTHON ?= python3
API_PYTHON ?= services/api/.venv/bin/python

.PHONY: help check structure test smoke autopilot

help:
	@printf '%s\n' 'make check - structure, tests, frontend build' 'make smoke - real YOLO end-to-end check' 'make autopilot - run sequential Claude/Codex milestones' 'python3 scripts/orchestrate.py --resume | --dry-run | --stop'

structure:
	$(PYTHON) scripts/check_structure.py

test:
	$(API_PYTHON) -m pytest tests/ -q

check: structure test
	cd apps/web && npm run build

smoke:
	$(API_PYTHON) scripts/smoke_e2e.py

autopilot:
	$(PYTHON) scripts/orchestrate.py --from M1 --to M4
