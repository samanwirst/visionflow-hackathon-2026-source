#!/usr/bin/env python3
"""Sequential, restartable VisionFlow milestone controller (stdlib only).

LLMs implement/review; this process alone owns scheduling, checkpoints,
test gates, review commits and pushes. Never merges into main.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

MILESTONES = ("M1", "M2", "M3", "M4")
TASKS = {
    "M1": "Real MP4 -> YOLO tracking -> polygon temporal rules -> persisted events, evidence and real notification. Fix correctness/security defects; do not fake inference.",
    "M2": "Satisfy PRD M2-AC1 through M2-AC5: usable strict NL workflow drafts with rejection, editing and manual approval; preserve cheap real detector/tracker/temporal rules. Add candidate-event-triggered VLM reasoning on selected frames/object ROIs, a real configurable Teacher adapter, strict response validation and explicit unknown/error states. Geometric rules make zero VLM calls. Implement configurable sampling/cooldowns/budgets, in-flight dedupe, bounded versioned caching/invalidation and actual routing/usage/timing counters. Test API/UI and authorization/SSRF boundaries. A disclosed deterministic NL compiler is acceptable without text-model credentials; mocked Teacher tests do not verify real visual inference. No generated code execution or per-frame VLM loop. Missing provider access is a documented blocker, never fabricated success.",
    "M3": "Satisfy PRD M3-AC1 through M3-AC6: real Teacher labels -> human-verified traceable diverse dataset -> frozen pretrained embeddings -> genuinely fitted lightweight Student with saved/reloaded artifacts. Session-disjoint train/validation/held-out splits must prevent adjacent-frame/near-duplicate leakage. Compute held-out metrics/counts/coverage/latency, not constants. Shadow compares real Teacher/Student without actions; activation needs a predefined quality gate and explicit approval. Test budgeted Teacher fallback, previous-version rollback and restart persistence. Manual-only labeling is a disclosed development fallback, not verification of the Teacher stage. Provide a small reproducible training demo; sequential memory-bounded ML. Never train on unverified Student labels.",
    "M4": "Satisfy PRD M4-AC1 through M4-AC4: reproducible paired baseline VLM-per-sampled-frame, event-routed/cached Teacher, and Student/hybrid experiments on the same held-out task/inputs. Preserve raw measured calls/usage/timings/predictions/config/hashes; compare quality, p50/p95 latency, cache/fallback coverage, cold/warm conditions and sample counts. Cost must derive from actual usage with dated cited rates/bill; keep local compute/training distinct. NEVER claim savings percentages without comparable positive measured baselines; missing real-provider runs remain NOT TESTED. Reliable startup/Docker/check/smoke, unseen-video real inference/actions/evidence, consent-safe licensed data, privacy/auth, honest README/metrics/deck, 3-month pilot and 7-minute pitch. Public deployment REQUIRES human approval; prepare artifacts only. No robot hardware operation or fabricated robustness.",
}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "CHANGES_REQUIRED", "BLOCKED"]},
        "target_sha": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {k: {"type": "string"} for k in
                           ("id", "severity", "file_line", "reproduction", "expected", "actual", "fix")},
            "required": ["id", "severity", "file_line", "reproduction", "expected", "actual", "fix"],
        }},
        "report_markdown": {"type": "string"},
    },
    "required": ["verdict", "target_sha", "findings", "report_markdown"],
}


class Blocked(RuntimeError):
    pass


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def validate_review(result: dict, sha: str) -> dict:
    if not isinstance(result, dict) or set(result) != set(REVIEW_SCHEMA["required"]):
        raise Blocked("Missing/malformed reviewer result")
    if result["target_sha"] != sha:
        raise Blocked("Reviewer TARGET_SHA differs from frozen implementation SHA")
    if result["verdict"] not in ("PASS", "CHANGES_REQUIRED", "BLOCKED"):
        raise Blocked("Unknown reviewer verdict")
    if not isinstance(result["findings"], list) or not isinstance(result["report_markdown"], str):
        raise Blocked("Malformed review findings/report")
    for finding in result["findings"]:
        if (not isinstance(finding, dict) or set(finding) != set(REVIEW_SCHEMA["properties"]["findings"]["items"]["required"])
                or any(not isinstance(v, str) for v in finding.values())
                or finding["severity"] not in ("P0", "P1", "P2", "P3")):
            raise Blocked("Malformed finding")
    if f"TARGET_SHA: {sha}" not in result["report_markdown"] or f"VERDICT: {result['verdict']}" not in result["report_markdown"]:
        raise Blocked("Markdown lacks matching machine-readable verdict/SHA")
    if result["verdict"] == "PASS" and any(f["severity"] in ("P0", "P1") for f in result["findings"]):
        raise Blocked("PASS contradicts unresolved P0/P1 findings")
    return result


def classify_failure(output: str) -> str:
    lower = output.lower()
    for kind, terms in (
        ("authentication", ("not logged in", "authentication failed", "invalid api key", "unauthorized", "please login")),
        ("rate_limit", ("rate limit", "usage limit", "quota exceeded", "too many requests")),
        ("compute", ("out of memory", "cannot allocate memory", "no space left")),
        ("dependency", ("modulenotfounderror", "command not found", "no such file or directory")),
        ("transient", ("connection reset", "temporarily unavailable", "502 bad gateway", "503 service", "overloaded")),
    ):
        if any(term in lower for term in terms):
            return kind
    return "execution_failure"


class Controller:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.repo).resolve()
        self.runtime = self.root / ".agents/runtime"
        self.state_path = self.runtime / "state.json"
        self.child = None
        self.state = {}
        self.started = time.monotonic()

    def git(self, *args, cwd=None, check=True):
        result = subprocess.run(["git", *args], cwd=cwd or self.root, capture_output=True,
                                text=True, timeout=120)
        if check and result.returncode:
            raise Blocked(f"git {args[0]} failed: {result.stderr[-1500:]}")
        return result.stdout.strip()

    def clean(self, cwd=None):
        if self.git("status", "--porcelain", cwd=cwd):
            raise Blocked("Working tree is dirty; preserve changes and inspect before resume")

    def fix_limit(self):
        # An explicit operator exception grants one extra M1 pass only.
        # It never resets the persisted counter or changes M2-M4 limits.
        return self.args.max_fix_cycles + int(
            self.args.exceptional_m1_recovery and self.state.get("milestone") == "M1"
        )

    def checkpoint(self, **values):
        self.state.update(values)
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(self.state_path, self.state)
        print(f"[{self.state.get('milestone')}] {self.state.get('phase')}: {values.get('reason', '')}", flush=True)

    def terminate_child(self):
        if self.child and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()

    def run(self, argv, label, cwd=None, prompt=None, timeout=None, agent=False, retry=0):
        if time.monotonic() - self.started + self.state.get("elapsed_seconds", 0) > self.args.total_timeout:
            raise Blocked("Configured total time limit reached")
        if agent:
            if (self.args.exceptional_m1_recovery and self.state.get("milestone") == "M1"
                    and self.state.get("fix_cycles") == 3):
                role = "implementation" if label == "M1-fix" else "review"
                attempted = dict(self.state.get("exceptional_m1_calls", {}))
                if attempted.get(role):
                    raise Blocked(f"Exceptional M1 {role} was already launched; no additional agent call authorized")
                attempted[role] = True
                self.checkpoint(exceptional_m1_calls=attempted)
            if self.state.get("agent_calls", 0) >= self.args.max_agent_calls:
                raise Blocked("Configured agent-call limit reached")
            self.checkpoint(agent_calls=self.state.get("agent_calls", 0) + 1)
        initial_head = self.git("rev-parse", "HEAD", cwd=cwd)
        log_dir = self.runtime / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        prefix = log_dir / f"{label}-{time.time_ns()}"
        if prompt:
            prefix.with_suffix(".prompt.txt").write_text(prompt)
        begin = time.monotonic()
        env = dict(os.environ)
        env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                    "PYTHONPATH": os.pathsep.join([str((cwd or self.root) / "services/api"),
                                                  str((cwd or self.root) / "packages/vision")])})
        # Logs are local and ignored. Do not pass credentials in argv/prompts.
        with prefix.with_suffix(".stdout").open("w") as out, prefix.with_suffix(".stderr").open("w") as err:
            self.child = subprocess.Popen(argv, cwd=cwd or self.root, env=env,
                                          stdin=subprocess.PIPE if prompt else subprocess.DEVNULL,
                                          stdout=out, stderr=err, text=True, start_new_session=True)
            atomic_json(self.runtime / "child.json", {"pid": self.child.pid, "label": label})
            try:
                self.child.communicate(input=prompt, timeout=timeout or self.args.agent_timeout)
            except subprocess.TimeoutExpired:
                self.terminate_child()
                atomic_json(prefix.with_suffix(".meta.json"), {"exit_code": self.child.returncode, "timeout": True,
                                                              "seconds": time.monotonic() - begin})
                raise Blocked(f"{label}: subprocess timeout; evidence at {prefix}")
            finally:
                (self.runtime / "child.json").unlink(missing_ok=True)
        stdout = prefix.with_suffix(".stdout").read_text(errors="replace")
        stderr = prefix.with_suffix(".stderr").read_text(errors="replace")
        usage = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
                if isinstance(event, dict) and any(k in event for k in ("usage", "total_cost_usd")):
                    usage.append({k: event[k] for k in ("usage", "total_cost_usd", "duration_ms", "num_turns") if k in event})
            except (ValueError, TypeError):
                pass
        atomic_json(prefix.with_suffix(".meta.json"), {"exit_code": self.child.returncode,
                                                      "seconds": time.monotonic() - begin, "usage": usage})
        if self.child.returncode:
            kind = classify_failure(stdout + stderr)
            if (kind == "transient" and agent and retry < self.args.transient_retries
                    and not (self.args.exceptional_m1_recovery and self.state.get("milestone") == "M1"
                             and self.state.get("fix_cycles") == 3)):
                self.clean(cwd)
                if self.git("rev-parse", "HEAD", cwd=cwd) == initial_head:
                    time.sleep(min(2 ** retry, 10))
                    return self.run(argv, label + "-retry", cwd, prompt, timeout, agent, retry + 1)
            raise Blocked(f"{label}: {kind} (exit {self.child.returncode}); see {prefix}\n{(stdout + stderr)[-3000:]}")
        return stdout

    def initialize(self):
        # ACCEPT is a recoverable transaction: only our exact reviewed report
        # may be dirty or committed after a crash. All other changes block.
        saved = json.loads(self.state_path.read_text()) if self.args.resume and self.state_path.exists() else {}
        saved_phase = saved.get("resume_phase") if saved.get("phase") in ("BLOCKED", "INTERRUPTED") else saved.get("phase")
        if saved_phase == "ACCEPT":
            relative = f".agents/reviews/{saved['milestone']}.md"
            dirty = self.git("status", "--porcelain")
            if dirty and (len(dirty.splitlines()) != 1 or dirty[3:] != relative or
                          (self.root / relative).read_text().strip() != saved["review"]["report_markdown"].strip()):
                raise Blocked("Unexpected changes during report acceptance; preserve and inspect")
        else:
            self.clean()
        if self.git("branch", "--show-current") != self.args.branch or self.args.branch == "main":
            raise Blocked(f"Run from implementation branch {self.args.branch}, never main")
        if self.args.resume:
            if not self.state_path.exists():
                raise Blocked("No saved state to resume")
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("version") != 1:
                raise Blocked("Unsupported checkpoint version")
            if self.state["phase"] in ("BLOCKED", "INTERRUPTED"):
                self.checkpoint(phase=self.state["resume_phase"], reason="Explicit resume")
            if self.state["phase"] in ("IMPLEMENT", "FIX") and self.git("rev-parse", "HEAD") != self.state["before_sha"]:
                self.verify_handoff()
                self.checkpoint(phase="TEST" if self.state["phase"] == "IMPLEMENT" else "RETEST", target_sha=self.git("rev-parse", "HEAD"))
            elif self.state["phase"] not in ("IMPLEMENT", "FIX", "DONE", "ACCEPT") and self.git("rev-parse", "HEAD") != self.state.get("target_sha"):
                raise Blocked("HEAD changed since checkpoint; refusing stale review/test results")
            return
        if self.state_path.exists():
            raise Blocked("Saved execution exists; use --resume (never overwrite progress)")
        start_sha = self.git("rev-parse", "HEAD")
        self.state = {"version": 1, "milestone": self.args.start, "end": self.args.end,
                      "phase": "IMPLEMENT", "base_sha": self.git("rev-parse", "main"),
                      "before_sha": start_sha, "target_sha": start_sha,
                      "fix_cycles": 0, "review_attempt": 0, "agent_calls": 0, "accepted": {}, "elapsed_seconds": 0}
        # Existing M1 handoff means review it; do not redo successful implementation.
        handoff = self.root / ".agents/HANDOFF.md"
        if self.args.start == "M1" and handoff.exists() and "Milestone:** M1" in handoff.read_text():
            self.verify_handoff()
            self.state["phase"] = "TEST"
        self.checkpoint()

    def verify_handoff(self):
        path = self.root / ".agents/HANDOFF.md"
        if not path.exists():
            raise Blocked("Implementation did not produce a handoff")
        content = path.read_text()
        shas = re.findall(r"\b[0-9a-f]{40}\b", content)
        if self.state["milestone"] not in content or not shas:
            raise Blocked("Handoff missing milestone/exact SHA")
        valid = subprocess.run(["git", "merge-base", "--is-ancestor", shas[0], "HEAD"], cwd=self.root).returncode == 0
        if not valid:
            raise Blocked("Handoff SHA is not an implementation ancestor")

    def implement(self, fix=False):
        milestone = self.state["milestone"]
        if shutil_which(self.args.claude) is None:
            raise Blocked("Claude executable is missing")
        status = subprocess.run([self.args.claude, "auth", "status"], capture_output=True, text=True, timeout=30)
        try:
            if not json.loads(status.stdout).get("loggedIn"):
                raise Blocked("Claude authentication required: run claude auth login")
        except ValueError:
            raise Blocked("Could not verify Claude authentication")
        findings = ""
        if fix:
            findings = json.dumps(self.state.get("review", {}).get("findings", []), indent=2)
            if milestone == "M1" and self.args.exceptional_m1_recovery and self.state["fix_cycles"] == 3:
                findings = json.dumps([f for f in self.state.get("review", {}).get("findings", [])
                                       if f["severity"] in ("P0", "P1")], indent=2)
                findings += """\nThis is the ONE exceptional narrowly scoped M1 recovery pass authorized by the operator. Address only M1-010, M1-011, M1-012, M1-013 and their regressions. Read .agents/reviews/M1.md for exact reproductions. Original executable defect reproductions are /tmp/visionflow-m1-cc59523-QS9GQO/test_focused.py (assertions demonstrate bugs; convert to desired regressions).
M1-010: pause dispatch after event commit; API disable returns enabled=false; then release. Serialize disable and authoritative dispatch admission; cancellation must be explicit. Test the actual race.
M1-011: receiver returns 200, crash before DB action commit, then recovery/reprocess delivers twice. Prefer simplifying nonessential external integrations. Persist dispatch intent before any external side effect; possibly-delivered interrupted attempts must not blindly replay. Use stable idempotency keys only where receiver enforces them; do not claim generic exactly-once delivery.
M1-012: allowlisted hostname resolves public during preflight, loopback for HTTPX. Prefer disabling arbitrary outbound webhooks cleanly rather than introducing a risky custom TLS/DNS transport. Retain the working local notification/mock CRM action; reject unavailable external actions visibly and ensure persisted legacy workflows cannot bypass the restriction. Update API/UI/docs/tests honestly if webhook support is removed. No network lookup or delivery may occur for a disabled feature.
M1-013: 60-frame MP4 -> ffmpeg copy +faststart -> truncate bytes to 65% -> upload/process silently reports processed after only 7/10 samples. Reject input or expose visible failed/partial decode using actual decoded progress versus reliable metadata with tolerance. Test valid videos still complete and real smoke passes.
Add meaningful regressions for all four reported defects, run full checks/frontend build/real YOLO smoke, and commit exact handoff. No M2/M3/M4 implementation, no P2-only changes in this pass. If safe external action support is infeasible, remove that capability explicitly rather than pretending it is secure.\n"""
            if self.state.get("test_failure"):
                findings += "\nTest failure: " + self.state["test_failure"]
        prompt = f"""You are the sole VisionFlow implementation agent. Human explicitly authorized the autonomous goal; complete ONLY {milestone} and stop after committed handoff. Do not spawn agents or invoke Claude/Codex recursively. Do not edit scripts/orchestrate.py or tests/orchestrator, review reports, or official source brief. Preserve unrelated files and API contracts. No public deployment, hardware, purchases, secret reads, force push, reset, or main merges. Controller handles pushes and review reports.
Read CLAUDE.md, docs/source/README.md, relevant docs/PRD.md requirements including the assigned milestone's section 16 acceptance criteria, docs/VISUAL_ANALYTICS_AUDIT.md, .agents/STATE.md and handoff; use focused reads thereafter. The goal's orchestration overrides manual milestone waiting: finish your assigned milestone then return.
Task: {TASKS[milestone]}
{'FIX only these P0/P1 or failing-test defects first:' if fix else 'Implement the missing requirements; preserve useful existing implementation.'}
{findings}
Use services/api/.venv and installed apps/web/node_modules. CPU-only, limited RAM; sequential training/inference/build, set thread counts to 2. Browse official primary docs only if needed; ordinary package installs are authorized.
Run meaningful tests, record actual results/limitations, update docs/TRACEABILITY.md and .agents/STATE.md, commit production work on {self.args.branch}. Then write .agents/HANDOFF.md with milestone {milestone}, EXACT implementation SHA (40 hex), base {self.state['base_sha']}, features, changed files, actual tests, limitations and review items; commit handoff. Do not claim public deployment complete. Return JSON output with your summary; nonzero or error results block execution.
"""
        allowed = ["Read", "Write", "Edit", "Glob", "Grep"] + [f"Bash({x} *)" for x in
                    ("git status", "git diff", "git log", "git show", "git rev-parse", "git add", "git commit",
                     "python3", "services/api/.venv/bin/python", "services/api/.venv/bin/pip", "npm", "make",
                     "rg", "ls", "cat", "mkdir", "curl", "ffmpeg", "cp", "cd", "test")]
        argv = [self.args.claude, "--print", "--output-format", "json", "--permission-mode", "dontAsk",
                "--allowedTools", ",".join(allowed), "--tools", "Read,Write,Edit,Bash,Glob,Grep",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "project",
                "--no-session-persistence"]
        if self.args.claude_budget is not None:
            argv.extend(["--max-budget-usd", str(self.args.claude_budget)])
        raw = self.run(argv, f"{milestone}-{'fix' if fix else 'implement'}", prompt=prompt, agent=True)
        try:
            response = json.loads(raw)
            if not isinstance(response, dict) or response.get("is_error") or response.get("subtype", "success") != "success":
                raise Blocked("Claude returned an error/incomplete result")
        except ValueError:
            raise Blocked("Claude returned malformed JSON")
        self.clean()
        sha = self.git("rev-parse", "HEAD")
        if sha == self.state["before_sha"]:
            raise Blocked("Implementation returned without a new commit")
        if self.git("branch", "--show-current") != self.args.branch:
            raise Blocked("Implementation changed the assigned branch")
        if subprocess.run(["git", "merge-base", "--is-ancestor", self.state["before_sha"], sha], cwd=self.root).returncode:
            raise Blocked("Implementation rewrote history")
        changed = self.git("diff", "--name-only", self.state["before_sha"], sha).splitlines()
        if any(p == "scripts/orchestrate.py" or p.startswith("tests/orchestrator/") or
               p.startswith(".agents/reviews/") or p.startswith("docs/source/") for p in changed):
            raise Blocked("Implementation changed controller, independent review, or official source files")
        self.verify_handoff()
        self.checkpoint(phase="RETEST" if fix else "TEST", target_sha=sha, test_failure=None)

    def tests(self):
        self.clean()
        if self.git("rev-parse", "HEAD") != self.state["target_sha"]:
            raise Blocked("Test target no longer matches HEAD")
        try:
            result = self.run(["make", "check"], f"{self.state['milestone']}-check", timeout=self.args.test_timeout)
            result += self.run(["make", "smoke"], f"{self.state['milestone']}-smoke", timeout=self.args.test_timeout)
        except Blocked as exc:
            if self.state["fix_cycles"] >= self.fix_limit():
                raise
            self.checkpoint(phase="FIX", before_sha=self.state["target_sha"],
                            fix_cycles=self.state["fix_cycles"] + 1, test_failure=str(exc))
            return
        self.clean()
        self.checkpoint(phase="REVIEW", test_evidence=result[-12000:], review=None)

    def review(self):
        milestone, sha = self.state["milestone"], self.state["target_sha"]
        attempt = self.state["review_attempt"]
        directory = self.runtime / "worktrees" / f"{milestone}-{attempt}"
        branch = f"review/{milestone}" + (f"-r{attempt}" if attempt else "")
        if not directory.exists():
            self.git("worktree", "add", "-b", branch, str(directory), sha)
        if self.git("rev-parse", "HEAD", cwd=directory) != sha:
            raise Blocked("Review worktree is at unexpected SHA")
        schema_path = self.runtime / "review-schema.json"
        atomic_json(schema_path, REVIEW_SCHEMA)
        result_path = self.runtime / f"{milestone}-{attempt}-review.json"
        if not result_path.exists():
            # Share installed dependencies, never production source or data.
            for rel in ("services/api/.venv", "apps/web/node_modules"):
                dest, source = directory / rel, self.root / rel
                if source.exists() and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.symlink_to(source, target_is_directory=True)
            prompt = f"""Independent QA review of {milestone}, frozen TARGET_SHA {sha}, base {self.state['base_sha']}.
Human explicitly requested this milestone review through the deterministic orchestrator. Follow AGENTS.md review criteria, but controller alone commits/pushes report; do not modify tracked files or Git refs, fetch, merge, implement, spawn agents, or poll. You are in an isolated worktree; never access credentials or unrelated personal files.
Read brief docs/source/README.md, relevant PRD/TRACEABILITY/protocol and handoff once. Inspect changed files first. Run existing automated tests plus focused integration/security reproductions. Installed dependencies linked into worktree; editable vision package must resolve to this worktree using PYTHONPATH=packages/vision:services/api. Do not rerun expensive smoke just to repeat this controller evidence unless needed:
{self.state.get('test_evidence', '')[-7000:]}
Required scope: {TASKS[milestone]}
For the exceptional M1 recovery, verify all four prior P1 reproductions and intentional feature simplification. Explicitly unavailable outbound webhooks may be rejected; verify zero egress and safe local actions, rather than requiring arbitrary webhook support. No M2-M4 feature implementation is required for M1 acceptance. In this restricted review sandbox, TestClient startup previously stalled in an asyncio selector wait. If reproduced, a disclosed temporary test-runner-only 10 ms asyncio heartbeat can unblock the unchanged suite; never alter production code or disguise the workaround.
Verify real functionality and approval boundaries, unsupported input rejection, state transitions/idempotency, SSRF, shadow/activation/rollback where relevant. At most five high-value findings unless more P0/P1 defects. PASS may contain P2/P3 risks; unresolved P0/P1 => CHANGES_REQUIRED. Genuine untestable required behavior => BLOCKED; explicitly document optional/cloud/hardware NOT TESTED without fabricating outcomes.
Final JSON conforms to supplied schema. Markdown uses AGENTS.md consolidated review structure and MUST contain literal lines VERDICT: <verdict> and TARGET_SHA: {sha}. Include exact base, commands actually executed, tests/results, verified/unverified behaviors, numbered findings severity/file:line/repro/expected/actual/minimal fix, acceptance criteria and recommendation. Explain limitations. Report reviewed SHA, never a different commit.
"""
            argv = [self.args.codex, "-a", "never", "exec", "--sandbox", "workspace-write",
                    "-c", "sandbox_workspace_write.network_access=false", "-c", "mcp_servers={}",
                    "--ephemeral", "--json", "--output-schema", str(schema_path),
                    "--output-last-message", str(result_path), "--cd", str(directory), "-"]
            self.run(argv, f"{milestone}-review-{attempt}", cwd=directory, prompt=prompt,
                     agent=True, timeout=self.args.review_timeout)
        try:
            review = validate_review(json.loads(result_path.read_text()), sha)
        except (ValueError, OSError) as exc:
            raise Blocked(f"Missing/malformed review result: {exc}")
        self.clean(directory)
        if self.git("rev-parse", "HEAD", cwd=directory) != sha:
            raise Blocked("Reviewer changed frozen HEAD")
        self.checkpoint(review=review, review_branch=branch, review_directory=str(directory), phase="REPORT")

    def record_review(self):
        milestone, sha = self.state["milestone"], self.state["target_sha"]
        directory = Path(self.state["review_directory"])
        review, branch = self.state["review"], self.state["review_branch"]
        head = self.git("rev-parse", "HEAD", cwd=directory)
        relative = f".agents/reviews/{milestone}.md"
        report = directory / f".agents/reviews/{milestone}.md"
        if head == sha:
            dirty = self.git("status", "--porcelain", cwd=directory)
            if dirty and (len(dirty.splitlines()) != 1 or dirty[3:] != relative or
                          report.read_text().strip() != review["report_markdown"].strip()):
                raise Blocked("Unexpected review worktree changes; refusing report commit")
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(review["report_markdown"] + "\n")
            self.git("add", relative, cwd=directory)
            self.git("commit", "-m", f"review({milestone}): QA findings for {sha[:7]}", cwd=directory)
        elif (self.git("rev-parse", "HEAD^", cwd=directory) != sha or
              self.git("diff", "--name-only", sha, "HEAD", cwd=directory) != relative or
              report.read_text().strip() != review["report_markdown"].strip()):
            raise Blocked("Report transaction changed unexpected commit/files")
        if not self.args.no_push:
            self.git("push", "origin", branch, cwd=directory)
        self.checkpoint(phase="DECIDE")

    def decide(self):
        review = self.state["review"]
        if review["verdict"] == "BLOCKED":
            raise Blocked("Independent reviewer blocked acceptance; inspect report")
        if review["verdict"] == "CHANGES_REQUIRED":
            if self.state["fix_cycles"] >= self.fix_limit():
                raise Blocked(f"{self.fix_limit()} fix cycles exhausted; unresolved reviewer defects")
            self.checkpoint(phase="FIX", fix_cycles=self.state["fix_cycles"] + 1,
                            review_attempt=self.state["review_attempt"] + 1, before_sha=self.state["target_sha"])
            return
        self.checkpoint(phase="ACCEPT")

    def accept(self):
        milestone = self.state["milestone"]
        report = self.root / f".agents/reviews/{milestone}.md"
        head = self.git("rev-parse", "HEAD")
        if head == self.state["target_sha"]:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(self.state["review"]["report_markdown"] + "\n")
            self.git("add", str(report.relative_to(self.root)))
            if self.git("diff", "--cached", "--name-only"):
                self.git("commit", "-m", f"review({milestone}): retain accepted QA evidence for {self.state['target_sha'][:7]}")
        elif (self.git("rev-parse", "HEAD^") != self.state["target_sha"] or
              self.git("diff", "--name-only", self.state["target_sha"], "HEAD") != str(report.relative_to(self.root)) or
              report.read_text().strip() != self.state["review"]["report_markdown"].strip()):
            raise Blocked("Unexpected HEAD during report acceptance")
        accepted = dict(self.state["accepted"])
        accepted[milestone] = {"target_sha": self.state["target_sha"], "review_branch": self.state["review_branch"], "verdict": "PASS"}
        self.checkpoint(phase="PUSH", target_sha=self.git("rev-parse", "HEAD"), accepted=accepted)

    def push_next(self):
        if not self.args.no_push:
            self.git("push", "origin", self.args.branch)
        if self.state["milestone"] == self.state["end"]:
            self.checkpoint(phase="DONE", reason="All requested milestone gates passed; public deployment remains approval-gated")
            return
        next_milestone = MILESTONES[MILESTONES.index(self.state["milestone"]) + 1]
        self.checkpoint(phase="IMPLEMENT", milestone=next_milestone, base_sha=self.state["target_sha"],
                        before_sha=self.state["target_sha"], fix_cycles=0, review_attempt=0, review=None, test_failure=None)

    def execute(self):
        self.runtime.mkdir(parents=True, exist_ok=True)
        with (self.runtime / "lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another orchestrator holds the repository lock", file=sys.stderr)
                return 2
            atomic_json(self.runtime / "controller.json", {"pid": os.getpid()})
            try:
                self.initialize()
                while self.state["phase"] != "DONE":
                    phase = self.state["phase"]
                    if phase in ("IMPLEMENT", "FIX"):
                        self.implement(fix=phase == "FIX")
                    elif phase in ("TEST", "RETEST"):
                        self.tests()
                    elif phase == "REVIEW":
                        self.review()
                    elif phase == "REPORT":
                        self.record_review()
                    elif phase == "DECIDE":
                        self.decide()
                    elif phase == "ACCEPT":
                        self.accept()
                    elif phase == "PUSH":
                        self.push_next()
                    else:
                        raise Blocked(f"Unknown state {phase}")
                return 0
            except (Blocked, KeyboardInterrupt, OSError) as exc:
                self.terminate_child()
                phase = self.state.get("phase", "IMPLEMENT")
                if self.state and phase not in ("BLOCKED", "INTERRUPTED"):
                    self.checkpoint(phase="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "BLOCKED",
                                    resume_phase=phase, reason=str(exc) or "Stopped by signal")
                print(f"Stopped: {exc}. Recovery: inspect .agents/runtime/state.json and logs; python3 scripts/orchestrate.py --resume", file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 2
            finally:
                if self.state:
                    self.state["elapsed_seconds"] = self.state.get("elapsed_seconds", 0) + time.monotonic() - self.started
                    atomic_json(self.state_path, self.state)
                (self.runtime / "controller.json").unlink(missing_ok=True)


def shutil_which(cmd):
    import shutil
    return shutil.which(cmd)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--branch", default="feature/visionflow")
    p.add_argument("--from", dest="start", choices=MILESTONES, default="M1")
    p.add_argument("--to", dest="end", choices=MILESTONES, default="M4")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--detach", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--claude", default="claude")
    p.add_argument("--codex", default="codex")
    p.add_argument("--agent-timeout", type=int, default=1800)
    p.add_argument("--review-timeout", type=int, default=1200)
    p.add_argument("--test-timeout", type=int, default=900)
    p.add_argument("--total-timeout", type=int, default=21600)
    p.add_argument("--max-agent-calls", type=int, default=20)
    p.add_argument("--transient-retries", type=int, choices=(0, 1, 2), default=1)
    p.add_argument("--max-fix-cycles", type=int, choices=(0, 1, 2), default=2)
    p.add_argument("--exceptional-m1-recovery", action="store_true",
                   help="Operator-authorized ONE extra M1 fix/review cycle; preserves counters and all other milestone limits")
    p.add_argument("--claude-budget", type=float, default=None, help="Optional per-call CLI USD cap; never enables new paid services")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if MILESTONES.index(args.start) > MILESTONES.index(args.end):
        raise SystemExit("--from must precede --to")
    if any(v <= 0 for v in (args.agent_timeout, args.review_timeout, args.test_timeout, args.total_timeout, args.max_agent_calls)):
        raise SystemExit("Limits must be positive")
    if args.dry_run:
        print(json.dumps({"milestones": MILESTONES[MILESTONES.index(args.start):MILESTONES.index(args.end)+1],
                          "phases": "IMPLEMENT -> TEST -> REVIEW -> FIX -> RETEST -> REVIEW -> ACCEPT -> NEXT",
                          "limits": {k: getattr(args, k) for k in ("agent_timeout", "review_timeout", "max_fix_cycles", "max_agent_calls", "total_timeout")},
                          "exceptional_m1_recovery": args.exceptional_m1_recovery,
                          "push": not args.no_push}, indent=2))
        return 0
    runtime = Path(args.repo).resolve() / ".agents/runtime"
    if args.stop:
        try:
            pid = json.loads((runtime / "controller.json").read_text())["pid"]
            # Verify process identity before signaling a potentially recycled PID.
            if b"orchestrate.py" not in Path(f"/proc/{pid}/cmdline").read_bytes():
                raise SystemExit("Stale PID; refusing signal")
            os.kill(pid, signal.SIGTERM)
            print(f"Stop requested for controller {pid}")
        except FileNotFoundError:
            print("No running controller")
        return 0
    if args.detach:
        runtime.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), *[a for a in (argv if argv is not None else sys.argv[1:]) if a != "--detach"]]
        with (runtime / "controller.log").open("a") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True, cwd=args.repo)
        print(f"Detached PID {process.pid}; logs: {runtime / 'controller.log'}; stop: python3 scripts/orchestrate.py --stop")
        return 0
    def stop_handler(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop_handler)
    return Controller(args).execute()


if __name__ == "__main__":
    sys.exit(main())
