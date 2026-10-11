"""Behavioral orchestration tests using real temp Git repos and fake CLIs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/orchestrate.py"
spec = importlib.util.spec_from_file_location("orchestrate", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

FAKE_AGENT = r'''#!/usr/bin/env python3
import json, os, pathlib, re, subprocess, sys, time
root = pathlib.Path.cwd()
if sys.argv[1:3] == ['auth', 'status']:
    print(json.dumps({'loggedIn': os.environ.get('FAKE_AUTH', '1') == '1'}))
    sys.exit(0)
prompt = sys.stdin.read()
if '--print' in sys.argv:
    m = re.search(r'complete ONLY (M\d)', prompt).group(1)
    (root / ('implemented-' + m)).write_text('fixture change ' + str(time.time_ns()) + '\n')
    subprocess.run(['git', 'add', 'implemented-' + m], check=True, stdout=sys.stderr)
    subprocess.run(['git', 'commit', '-m', 'feat: ' + m], check=True, stdout=sys.stderr)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (root / '.agents/HANDOFF.md').write_text('Milestone: ' + m + '\nSHA: ' + sha + '\n')
    subprocess.run(['git', 'add', '.agents/HANDOFF.md'], check=True, stdout=sys.stderr)
    subprocess.run(['git', 'commit', '-m', 'handoff: ' + m], check=True, stdout=sys.stderr)
    print(json.dumps({'subtype': 'success', 'is_error': False, 'result': 'done', 'usage': {'input_tokens': 7}}))
else:
    sha = re.search(r'frozen TARGET_SHA ([0-9a-f]{40})', prompt).group(1)
    target = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])
    count_file = target.parent / 'fake-count'
    count = int(count_file.read_text()) if count_file.exists() else 0
    count_file.write_text(str(count + 1))
    if os.environ.get('FAKE_TIMEOUT'):
        time.sleep(10)
    verdict = 'CHANGES_REQUIRED' if count < int(os.environ.get('FAKE_REVIEW_FAILS', '0')) else 'PASS'
    finding = {'id': 'M1-001', 'severity': 'P1', 'file_line': 'fixture:1', 'reproduction': 'repeat', 'expected': 'ok', 'actual': 'bad', 'fix': 'minimal'}
    if os.environ.get('FAKE_WRONG_SHA'):
        sha = '0' * 40
    result = {'verdict': verdict, 'target_sha': sha, 'findings': [finding] if verdict != 'PASS' else [],
              'report_markdown': '# Review\nVERDICT: ' + verdict + '\nTARGET_SHA: ' + sha + '\nTests: fixture check ran; real inference NOT TESTED.\n'}
    target.write_text(json.dumps(result))
    print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 9}}))
'''


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="visionflow-controller-test-")
        self.root = Path(self.temp.name)
        git(self.root, "init", "-b", "main")
        git(self.root, "config", "user.email", "qa@example.test")
        git(self.root, "config", "user.name", "QA Fixture")
        (self.root / ".agents").mkdir()
        (self.root / ".agents/HANDOFF.md").write_text("Initial fixture\n")
        (self.root / ".gitignore").write_text(".agents/runtime/\n.venv\nnode_modules\n")
        (self.root / "Makefile").write_text("check:\n\t@echo fixture-check\nsmoke:\n\t@echo fixture-smoke\n")
        git(self.root, "add", ".")
        git(self.root, "commit", "-m", "fixture")
        self.base = git(self.root, "rev-parse", "HEAD")
        git(self.root, "switch", "-c", "feature/visionflow")
        self.fake = self.root / ".agents/runtime/fake-agent"
        self.fake.parent.mkdir()
        self.fake.write_text(FAKE_AGENT)
        self.fake.chmod(0o700)

    def tearDown(self):
        self.temp.cleanup()

    def run_controller(self, *args, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(self.root),
                               "--claude", str(self.fake), "--codex", str(self.fake), "--no-push",
                               "--from", "M1", "--to", "M1", *args],
                              env={**os.environ, **(env or {})}, text=True, capture_output=True, timeout=30)

    def state(self):
        return json.loads((self.fake.parent / "state.json").read_text())

    def test_full_cycle_and_completed_resume_do_not_repeat(self):
        result = self.run_controller()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.state()["phase"], "DONE")
        self.assertEqual(self.state()["agent_calls"], 2)
        review = self.state()["accepted"]["M1"]
        self.assertEqual(git(self.root, "rev-parse", "review/M1^"), review["target_sha"])
        self.assertEqual(git(self.root, "rev-parse", "main"), self.base)
        result = self.run_controller("--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.state()["agent_calls"], 2)

    def test_dependency_symlinks_are_ignored(self):
        for relative in ("services/api/.venv", "apps/web/node_modules"):
            (self.root / relative).mkdir(parents=True)
        result = self.run_controller()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        review_dir = Path(self.state()["review_directory"])
        self.assertTrue((review_dir / "services/api/.venv").is_symlink())
        self.assertTrue((review_dir / "apps/web/node_modules").is_symlink())
        self.assertEqual(git(review_dir, "status", "--porcelain"), "")

    def test_two_fix_cycles_then_block(self):
        result = self.run_controller(env={"FAKE_REVIEW_FAILS": "99"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.state()["phase"], "BLOCKED")
        # First fake fix changes only handoff (sufficient to exercise scheduling).
        self.assertEqual(self.state()["fix_cycles"], 2)
        self.assertEqual(self.state()["agent_calls"], 6)
        self.assertEqual(int((self.fake.parent / "fake-count").read_text()), 3)

    def test_one_fix_then_pass(self):
        result = self.run_controller(env={"FAKE_REVIEW_FAILS": "1"})
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.state()["fix_cycles"], 1)
        self.assertEqual(self.state()["agent_calls"], 4)

    def test_exceptional_m1_recovery_resumes_once_and_accepts(self):
        self.assertEqual(self.run_controller(env={"FAKE_REVIEW_FAILS": "3"}).returncode, 2)
        self.assertEqual(self.state()["fix_cycles"], 2)
        result = self.run_controller("--resume", "--exceptional-m1-recovery", env={"FAKE_REVIEW_FAILS": "3"})
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.state()["fix_cycles"], 3)
        self.assertEqual(self.state()["agent_calls"], 8)
        self.assertEqual(int((self.fake.parent / "fake-count").read_text()), 4)
        self.assertEqual(self.run_controller("--resume", "--exceptional-m1-recovery").returncode, 0)
        self.assertEqual(self.state()["agent_calls"], 8)

    def test_exceptional_m1_recovery_failure_cannot_repeat(self):
        self.assertEqual(self.run_controller(env={"FAKE_REVIEW_FAILS": "99"}).returncode, 2)
        result = self.run_controller("--resume", "--exceptional-m1-recovery", env={"FAKE_REVIEW_FAILS": "99"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.state()["fix_cycles"], 3)
        self.assertEqual(self.state()["agent_calls"], 8)
        self.assertIn("3 fix cycles exhausted", self.state()["reason"])
        result = self.run_controller("--resume", "--exceptional-m1-recovery", env={"FAKE_REVIEW_FAILS": "99"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.state()["agent_calls"], 8)
        self.assertEqual(int((self.fake.parent / "fake-count").read_text()), 4)

    def test_exception_is_scoped_to_m1(self):
        result = self.run_controller("--from", "M2", "--to", "M2", "--exceptional-m1-recovery",
                                     env={"FAKE_REVIEW_FAILS": "99"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.state()["milestone"], "M2")
        self.assertEqual(self.state()["fix_cycles"], 2)
        self.assertEqual(self.state()["agent_calls"], 6)

    def test_exceptional_review_timeout_cannot_launch_a_second_review(self):
        self.assertEqual(self.run_controller(env={"FAKE_REVIEW_FAILS": "99"}).returncode, 2)
        result = self.run_controller("--resume", "--exceptional-m1-recovery", "--review-timeout", "1",
                                     env={"FAKE_TIMEOUT": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.state()["agent_calls"], 8)
        result = self.run_controller("--resume", "--exceptional-m1-recovery")
        self.assertEqual(result.returncode, 2)
        self.assertIn("review was already launched", self.state()["reason"])
        self.assertEqual(self.state()["agent_calls"], 8)

    def test_wrong_review_sha_never_accepted(self):
        result = self.run_controller(env={"FAKE_WRONG_SHA": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("M1", self.state()["accepted"])
        self.assertIn("TARGET_SHA", self.state()["reason"])

    def test_auth_failure_is_not_retried(self):
        result = self.run_controller(env={"FAKE_AUTH": "0"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.state()["agent_calls"], 0)
        self.assertIn("authentication", self.state()["reason"])

    def test_timeout_kills_child_and_saves_checkpoint(self):
        result = self.run_controller("--review-timeout", "1", env={"FAKE_TIMEOUT": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("timeout", self.state()["reason"])
        self.assertFalse((self.fake.parent / "child.json").exists())
        self.assertFalse((self.fake.parent / "controller.json").exists())

    def test_dirty_tree_preserved(self):
        (self.root / "personal-notes.txt").write_text("keep me")
        result = self.run_controller()
        self.assertEqual(result.returncode, 2)
        self.assertEqual((self.root / "personal-notes.txt").read_text(), "keep me")
        self.assertEqual(git(self.root, "rev-parse", "HEAD"), self.base)

    def test_dry_run_never_launches_agents_or_saves_state(self):
        result = self.run_controller("--dry-run")
        self.assertEqual(result.returncode, 0)
        self.assertFalse((self.fake.parent / "state.json").exists())
        self.assertEqual(git(self.root, "rev-parse", "HEAD"), self.base)

    def test_agent_limit_blocks(self):
        result = self.run_controller("--max-agent-calls", "1")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.state()["agent_calls"], 1)
        self.assertIn("agent-call limit", self.state()["reason"])

    def test_restart_without_resume_does_not_overwrite(self):
        self.assertEqual(self.run_controller().returncode, 0)
        before = (self.fake.parent / "state.json").read_text()
        result = self.run_controller()
        self.assertEqual(result.returncode, 2)
        self.assertEqual((self.fake.parent / "state.json").read_text(), before)

    def test_report_commit_interruption_recovers_without_new_review(self):
        self.assertEqual(self.run_controller().returncode, 0)
        state = self.state()
        state.update(phase="REPORT", target_sha=state["accepted"]["M1"]["target_sha"])
        # Root is restored to implementation only in this disposable fixture.
        git(self.root, "reset", "--hard", state["target_sha"])
        module.atomic_json(self.fake.parent / "state.json", state)
        result = self.run_controller("--resume")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.state()["agent_calls"], 2)

    def test_accept_commit_interruption_recovers_without_new_review(self):
        self.assertEqual(self.run_controller().returncode, 0)
        state = self.state()
        state.update(phase="ACCEPT", target_sha=state["accepted"]["M1"]["target_sha"])
        module.atomic_json(self.fake.parent / "state.json", state)
        result = self.run_controller("--resume")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.state()["agent_calls"], 2)

    def test_changed_head_rejects_stale_review(self):
        self.assertEqual(self.run_controller().returncode, 0)
        state = self.state()
        state["phase"] = "REVIEW"
        module.atomic_json(self.fake.parent / "state.json", state)
        (self.root / "extra").write_text("unreviewed")
        git(self.root, "add", "extra")
        git(self.root, "commit", "-m", "unreviewed")
        result = self.run_controller("--resume")
        self.assertEqual(result.returncode, 2)
        self.assertIn("HEAD changed", self.state()["reason"])

    def test_contradictory_pass_rejected(self):
        sha = "a" * 40
        with self.assertRaises(module.Blocked):
            module.validate_review({"verdict": "PASS", "target_sha": sha,
                                    "report_markdown": f"VERDICT: PASS\nTARGET_SHA: {sha}",
                                    "findings": [{k: "P1" if k == "severity" else "x" for k in
                                                  ("id", "severity", "file_line", "reproduction", "expected", "actual", "fix")}]}, sha)

    def test_failure_classification(self):
        self.assertEqual(module.classify_failure("not logged in"), "authentication")
        self.assertEqual(module.classify_failure("503 service unavailable"), "transient")
        self.assertEqual(module.classify_failure("AssertionError"), "execution_failure")


if __name__ == "__main__":
    unittest.main()
