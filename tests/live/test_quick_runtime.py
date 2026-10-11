"""Small real-boundary regressions for the owner-scoped recorded demo."""
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from app.live import planner, video


def test_nonreading_child_input_is_bounded_and_reaped():
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    started = time.monotonic()
    rc, _, _, expired = planner._communicate_bounded(proc, 'x' * 500_000, .06, 1024)
    assert expired and rc is not None and proc.poll() is not None
    assert time.monotonic() - started < .5


def test_actual_child_output_cap_kills_and_reaps():
    proc = subprocess.Popen([sys.executable, '-c',
                             'import sys,time;sys.stdout.write("x"*100000);sys.stdout.flush();time.sleep(5)'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    rc, out, _, expired = planner._communicate_bounded(proc, '', .5, 128)
    assert expired and rc is not None and len(out) <= 128


def test_revoked_worker_has_truthful_terminal_state(monkeypatch):
    room = SimpleNamespace(room_id='owned-test', lock=threading.RLock(),
                           job_version=7, active_generation=0,
                           job_status='running', job_error=None)
    job = object.__new__(video.VideoJob)
    job.room, job.version = room, 7
    monkeypatch.setattr(video, '_jobs', {room.room_id: job})
    monkeypatch.setattr(video, '_active_permits', 1)
    job._finish('cancelled', None)
    assert room.job_status == 'cancelled'
    assert video._active_permits == 0 and not video._jobs


def test_weights_are_part_of_detector_identity():
    a = SimpleNamespace(model_name='yolo11n.pt', weights_sha256='a' * 64)
    b = SimpleNamespace(model_name='yolo11n.pt', weights_sha256='b' * 64)
    assert video._model_identity(a) != video._model_identity(b)
    assert 'sha256:' in video._model_identity(a)
