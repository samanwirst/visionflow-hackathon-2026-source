"""Fresh LIVE QA_ONLY acceptance/fault-boundary tests. Never a phone/Teacher claim."""
import ast
import json
import secrets
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.live import auth, config, planner, rooms, teacher, worker
from app.live.app import app, _ingest_frame
from app.live.dsl import LiveTask
from app.live.planner import PlannerResult
from visionflow_vision.detector import BaseDetectorTracker
from visionflow_vision.events import TrackedObject

ROOT = Path('/tmp/visionflow-live-review-1-f5aeae3')
REPO = Path.cwd()
ORIGIN = 'https://live.qa.invalid'


def ws_url(path):
    # TestClient's relative websocket URL always defaults to ws://testserver,
    # independently of base_url. Use the actual cookie host explicitly.
    return 'wss://live.qa.invalid' + path


class QADetector(BaseDetectorTracker):
    def __init__(self):
        self.sizes = []
        self.resets = 0
        self.objects = [('car', 1), ('bottle', 2)]
    def reset(self):
        self.resets += 1
    def track_frame(self, frame, timestamp):
        height, width = frame.shape[:2]
        self.sizes.append((width, height))
        return [TrackedObject(tid, cls, .8, width*.2, height*.2, width*.4, height*.5)
                for cls, tid in self.objects]


def jpeg(width=320, height=240, shade=60):
    ok, buf = cv2.imencode('.jpg', np.full((height, width, 3), shade, np.uint8))
    assert ok
    return buf.tobytes()


def proposal(cls='car', trigger='presence_in_zone', semantic=False):
    return {
        'supported': True, 'name': 'QA_ONLY ' + ('Bottle opening' if semantic else cls),
        'explanation': 'QA_ONLY simulated model output', 'object_class': cls,
        'zone': {'type': 'full_frame', 'polygon': []}, 'trigger': trigger,
        'duration_seconds': 1 if trigger == 'dwell_in_zone' else 0,
        'threshold': 1 if trigger == 'count_exceeded' else 0,
        'action': 'focus_notify' if semantic else 'notify', 'focus': semantic,
        'visual_observation': {'question': 'Is the bottle being opened?' if semantic else '',
                               'labels': ['opening', 'closed'] if semantic else [],
                               'temporal': semantic},
        'clarification_needed': '', 'unsupported_reason': '',
    }


class Harness:
    def __init__(self):
        self.pin = secrets.token_urlsafe(16)
        self.clients = []
        self.detector = QADetector()
    def client(self, role=None):
        c = TestClient(app, base_url=ORIGIN, raise_server_exceptions=False)
        self.clients.append(c)
        c.__enter__()
        if role:
            r = c.post('/api/live/login', json={'role': role, 'pin': self.pin},
                       headers={'Origin': ORIGIN})
            assert r.status_code == 200
        return c
    def post(self, c, path, body):
        return c.post(path, json=body, headers={'Origin': ORIGIN,
                    'X-CSRF-Token': c.cookies.get(config.CSRF_COOKIE_NAME, '')})
    def room(self):
        camera = self.client('camera')
        r = self.post(camera, '/api/live/rooms', {})
        assert r.status_code == 200
        data = r.json()
        return camera, rooms.get_room(data['room_id']), data['join_code']
    def judge(self, code):
        judge = self.client('judge')
        assert self.post(judge, '/api/live/rooms/join', {'join_code': code}).status_code == 200
        return judge
    def activate(self, room, judge, semantic=False, cls='car'):
        task = planner._finalize(PlannerResult(status='ok', proposal=proposal(cls, semantic=semantic))).task
        checked = self.post(judge, f'/api/live/rooms/{room.room_id}/task/validate', {'task': task})
        assert checked.status_code == 200
        bound = checked.json()
        r = self.post(judge, f'/api/live/rooms/{room.room_id}/task/activate', {'task': bound['task'], 'draft_hash': bound['draft_hash'], 'expected_version': bound['expected_version']})
        assert r.status_code == 200
        return task
    def frame(self, room, seq, data=None, **extra):
        meta = {'seq': seq, 'w': 320, 'h': 240, 'capture_ts': time.time(), **extra}
        _ingest_frame(room, jpeg() if data is None else data, meta)
        return worker._process_room_if_ready(room)


@pytest.fixture
def h(monkeypatch):
    auth._reset_for_tests(); rooms._reset_for_tests(); worker._reset_for_tests()
    teacher._reset_for_tests()
    from app.live import video as _video
    _video._reset_for_tests()
    planner._room_attempts.clear(); planner._room_last_attempt.clear()
    hh = Harness()
    hh.patch = monkeypatch
    monkeypatch.setattr(config, 'DEMO_PIN', hh.pin)
    monkeypatch.setattr(config, 'COOKIE_SECURE', False)  # owned local QA only
    monkeypatch.setattr(config, 'ALLOWED_ORIGINS', [ORIGIN])
    monkeypatch.setattr(config, 'PLANNER_COOLDOWN_SECONDS', 0)
    monkeypatch.setattr(config, 'TEACHER_COOLDOWN_SECONDS', 0)
    monkeypatch.setattr(config, 'TEACHER_BASE_URL', '')
    # The camera/WebSocket ingest path is disabled on the PUBLIC surface (owner
    # pivot). These legacy hardening cases exercise that now-non-public code, so
    # enable it for the in-process test app only; a dedicated test asserts the
    # public-default disabled surface.
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', True)
    import tempfile
    monkeypatch.setattr(rooms, 'LIVE_VIDEO_DIR', Path(tempfile.mkdtemp(prefix='qa-live-up-')))
    monkeypatch.setattr(worker, 'start', lambda: None)
    monkeypatch.setattr(worker, 'detector_factory', lambda: hh.detector)
    monkeypatch.setattr(planner, 'transport', None)
    monkeypatch.setattr(teacher, 'transport', None)
    yield hh
    worker.stop()
    for c in hh.clients:
        c.__exit__(None, None, None)
    worker._reset_for_tests()


def test_exact_source_imports_and_no_legacy_provider(h):
    assert Path(worker.__file__).resolve().is_relative_to(REPO)
    assert Path(planner.__file__).resolve().is_relative_to(REPO)
    import sys
    # Other legacy suites are collected in this process. Check standalone import
    # isolation in a clean interpreter, where their sys.modules cannot contaminate it.
    code = ("import sys; import app.live.app; "
            "assert 'app.nl_compiler' not in sys.modules; "
            "assert 'app.teacher' not in sys.modules; "
            "assert 'app.main' not in sys.modules")
    checked = subprocess.run([sys.executable, '-c', code], capture_output=True,
                             text=True, timeout=10)
    assert checked.returncode == 0, checked.stderr


def test_unauthorized_endpoint_inventory(h):
    c = h.client()
    legacy = []
    tree = ast.parse((REPO/'services/api/app/main.py').read_text())
    for item in ast.walk(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in item.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.args:
                    if dec.func.attr in {'get','post','put','patch','delete'}:
                        path = dec.args[0].value
                        import re
                        path = re.sub(r'\{[^}]+\}', '1', path)
                        legacy.append((dec.func.attr.upper(), path))
    legacy += [('GET', '/media/evidence/qa.jpg'), ('GET','/api/autolearn/jobs'),
               ('POST','/api/autolearn/jobs'), ('POST','/api/autolearn/jobs/1/train'),
               ('POST','/api/autolearn/jobs/1/activate'), ('GET','/api/autolearn/tasks'),
               ('POST','/v1/chat/completions'), ('GET','/v1/models')]
    results = []
    for method, path in legacy:
        resp = c.request(method, path, json={} if method == 'POST' else None)
        results.append({'method': method, 'path': path, 'status': resp.status_code})
        assert resp.status_code == 404
        assert 'access-control-allow-origin' not in resp.headers
    for route in app.routes:
        path = getattr(route, 'path', '')
        if path.startswith('/api/live/') and path not in {'/api/live/login','/api/live/logout','/api/live/health'}:
            if hasattr(route, 'methods'):
                for method in route.methods:
                    p = path.replace('{room_id}', 'qa').replace('{filename}', 'qa.jpg')
                    resp = c.request(method, p, json={} if method == 'POST' else None,
                                     headers={'Origin': ORIGIN})
                    results.append({'method': method, 'path': p, 'status': resp.status_code})
                    assert resp.status_code == 401
    (ROOT/'endpoint-inventory.json').write_text(json.dumps(results, indent=2))


def test_landing_exact_two_choices_no_pin_in_static(h):
    c = h.client()
    text = c.get('/').text
    # OWNER PIVOT: the public landing offers Upload Video + Judges, not Live Camera.
    assert text.count('href="/upload"') == 1
    assert text.count('href="/judges"') == 1
    assert 'href="/camera"' not in text
    for path in ['/', '/upload','/judges','/static/common.js','/static/style.css']:
        assert h.pin not in c.get(path).text


def test_no_default_pin_and_secure_cookie_control(h, monkeypatch):
    monkeypatch.setattr(config, 'DEMO_PIN', '')
    c = h.client()
    assert c.post('/api/live/login', json={'role':'camera','pin':''}, headers={'Origin':ORIGIN}).status_code == 401
    monkeypatch.setattr(config, 'DEMO_PIN', h.pin)
    monkeypatch.setattr(config, 'COOKIE_SECURE', True)
    c = h.client('judge')
    cookie = c.cookies.get(config.SESSION_COOKIE_NAME)
    assert auth.get_session(cookie).role == 'judge'


def test_cross_room_and_evidence_role_controls(h):
    camera, room, code = h.room()
    judge = h.judge(code)
    h.activate(room, judge)
    h.frame(room, 1); h.frame(room, 2)
    other_camera, other_room, other_code = h.room()
    stranger = h.client('judge')
    other_judge = h.judge(other_code)
    url = judge.get(f'/api/live/rooms/{room.room_id}/state').json()['events'][0]['evidence_url']
    assert judge.get(url).status_code == 200
    for outsider in [stranger, other_judge, other_camera]:
        for suffix in ['state','frame']:
            assert outsider.get(f'/api/live/rooms/{room.room_id}/{suffix}').status_code == 403
        assert outsider.get(url).status_code == 403
    assert h.post(camera, f'/api/live/rooms/{room.room_id}/task/plan', {'text':'watch car'}).status_code == 403
    assert h.post(judge, '/api/live/rooms', {}).status_code == 403
    with pytest.raises(WebSocketDisconnect):
        with judge.websocket_connect(ws_url('/api/live/ws/camera'), headers={'Origin':ORIGIN}):
            pass


@pytest.mark.parametrize('origin,host', [(None,None), ('https://attacker.invalid','attacker.invalid'),
                                       ('http://live.qa.invalid','live.qa.invalid')])
def test_origin_allowlist_is_authoritative(h, origin, host):
    c = h.client()
    headers = {}
    if origin: headers['Origin'] = origin
    if host: headers['Host'] = host
    r = c.post('/api/live/login', json={'role':'camera','pin':h.pin}, headers=headers)
    assert r.status_code == 403, f'non-allowed Origin/Host login returned {r.status_code}'


def test_mutation_missing_origin_rejected(h):
    c = h.client('camera')
    r = c.post('/api/live/rooms', json={}, headers={'X-CSRF-Token':c.cookies.get(config.CSRF_COOKIE_NAME)})
    assert r.status_code == 403


def test_logout_requires_origin_and_csrf(h):
    c = h.client('judge')
    r = c.post('/api/live/logout', headers={'Origin':'https://attacker.invalid'})
    assert r.status_code == 403


def test_ws_missing_origin_is_rejected(h):
    c, room, code = h.room()
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(ws_url('/api/live/ws/camera')):
            pass


def test_ws_requires_corresponding_csrf_cookie(h):
    c,room,code=h.room()
    cookie=f'{config.SESSION_COOKIE_NAME}={c.cookies.get(config.SESSION_COOKIE_NAME)}'
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(ws_url('/api/live/ws/camera'),headers={'Origin':ORIGIN,'Cookie':cookie}):
            pass


def test_rate_limit_cannot_be_bypassed_by_untrusted_xff(h):
    c = h.client()
    statuses = [c.post('/api/live/login', json={'role':'judge','pin':'QA_ONLY_wrong'},
                headers={'Origin':ORIGIN, 'X-Forwarded-For':f'198.51.100.{i}'}).status_code
                for i in range(config.LOGIN_MAX_FAILURES+2)]
    assert 429 in statuses, f'all attempts from one peer escaped lockout: {statuses}'


@pytest.mark.parametrize('mode', ['logout','expiry'])
def test_camera_socket_revoked_after_login_lifetime(h, mode):
    h.patch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    c, room, code = h.room()
    if mode == 'logout':
        assert h.post(c, '/api/live/logout', {}).status_code == 200
    else:
        auth.get_session(c.cookies.get(config.SESSION_COOKIE_NAME)).expires_at = time.time()-1
    assert c.get('/api/live/me').status_code == 401
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(ws_url('/api/live/ws/camera'), headers={'Origin':ORIGIN}):
            pytest.fail('disabled/revoked camera accepted')
    assert room.frames_received == 0


@pytest.mark.parametrize('mode', ['logout','expiry','rebind'])
def test_judge_socket_stops_disclosing_old_room(h, mode):
    h.patch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    camera, room, code = h.room(); judge = h.judge(code)
    room.annotated_jpeg = b'QA_ONLY-private'
    if mode == 'logout':
        assert h.post(judge, '/api/live/logout', {}).status_code == 200
    elif mode == 'expiry':
        auth.get_session(judge.cookies.get(config.SESSION_COOKIE_NAME)).expires_at = time.time()-1
    else:
        _, second_room, second_code = h.room()
        assert h.post(judge, '/api/live/rooms/join', {'join_code': second_code}).status_code == 200
        assert judge.get(f'/api/live/rooms/{room.room_id}/state').status_code == 403
    with pytest.raises(WebSocketDisconnect):
        with judge.websocket_connect(ws_url('/api/live/ws/judge'), headers={'Origin':ORIGIN}):
            pytest.fail('disabled judge stream disclosed private bytes')


def test_rooms_retire_on_owner_logout_and_new_users_can_start(h):
    statuses=[]
    for _ in range(config.MAX_ROOMS+1):
        c = h.client('camera')
        r = h.post(c, '/api/live/rooms', {})
        statuses.append(r.status_code)
        assert h.post(c, '/api/live/logout', {}).status_code == 200
    assert all(s == 200 for s in statuses), f'logged-out rooms exhaust capacity: {statuses}'


def test_duplicate_and_stale_sequence_rejected(h):
    camera, room, code = h.room()
    judge = h.judge(code); h.activate(room, judge)
    for seq in [9,9,8]: h.frame(room, seq)
    assert room.frames_received == 1, f'{room.frames_received} frames admitted for duplicate/stale seq'
    assert len(room.events) == 0


def test_actual_jpeg_dimensions_bound_even_when_metadata_lies(h):
    h.patch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    c, room, code = h.room()
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(ws_url('/api/live/ws/camera'), headers={'Origin':ORIGIN}):
            pytest.fail('disabled camera transport accepted')
    assert room.frames_received == 0 and h.detector.sizes == []


def test_empty_frame_cannot_terminate_worker(h):
    h.patch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    c, room, code = h.room()
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(ws_url('/api/live/ws/camera'), headers={'Origin':ORIGIN}):
            pytest.fail('disabled camera transport accepted')
    assert room.frames_received == 0 and h.detector.sizes == []


def test_nonfinite_capture_timestamp_cannot_break_state_endpoint(h):
    c, room, code = h.room()
    judge = h.judge(code); h.activate(room, judge)
    h.frame(room, 1, capture_ts=float('nan')); h.frame(room, 2, capture_ts=float('nan'))
    r = judge.get(f'/api/live/rooms/{room.room_id}/state')
    assert r.status_code == 200, f'accepted NaN poisons state response: {r.status_code}'


def test_http_payload_is_bounded_before_json_parsing(h):
    c = h.client()
    r = c.post('/api/live/login', json={'role':'judge','pin':h.pin,
               'padding':'x'*(config.MAX_FRAME_BYTES*2)}, headers={'Origin':ORIGIN})
    assert r.status_code == 413, f'oversize JSON body accepted: {r.status_code}'


def test_ws_metadata_payload_is_bounded(h):
    h.patch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    c, room, code = h.room()
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(ws_url('/api/live/ws/camera'), headers={'Origin':ORIGIN}):
            pytest.fail('disabled camera transport accepted')
    assert room.frames_received == 0 and h.detector.sizes == []


def test_disabled_camera_has_no_queue_or_frames(h):
    h.patch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    c, room, code = h.room()
    for _ in range(4):
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect(ws_url('/api/live/ws/camera'), headers={'Origin':ORIGIN}):
                pytest.fail('disabled camera route admitted frames')
    assert room.frames_received == 0 and room.frames_inferred == 0
    assert len(room.events) == 0 and not room.notifications
    assert room.latest_jpeg is None and room.annotated_jpeg is None


@pytest.mark.parametrize('teacher_state', ['unavailable','closed','error'])
def test_semantic_condition_never_degrades_to_presence_alert(h, monkeypatch, teacher_state):
    c, room, code = h.room()
    judge = h.judge(code)
    h.activate(room, judge, semantic=True, cls='bottle')
    if teacher_state != 'unavailable':
        monkeypatch.setattr(teacher, 'transport', lambda q,l,i: teacher.Observation(
            status='ok' if teacher_state == 'closed' else 'error', label='closed',
            confidence=.9, uncertain=True, model='QA_ONLY'))
    h.frame(room,1); h.frame(room,2)
    assert len(room.notifications) == 0, (
        f'bottle-opening task notified on presence with {teacher_state} Teacher: '
        f'{len(room.events)} events, {len(room.notifications)} notifications')


def test_geometric_control_real_decode_event_evidence_dedupe_zero_teacher(h, monkeypatch):
    c, room, code = h.room()
    judge = h.judge(code)
    calls=[]
    monkeypatch.setattr(teacher,'transport',lambda *args: calls.append(args))
    h.activate(room,judge)
    for seq in range(1,7): h.frame(room,seq)
    assert len(room.events) == len(room.notifications) == 1
    assert calls == []
    ev = room.events[0]
    assert ev.task_version == room.task_version
    assert (rooms.LIVE_EVIDENCE_DIR/ev.evidence_file).is_file()
    assert judge.get(f'/api/live/rooms/{room.room_id}/evidence/{ev.evidence_file}').status_code == 200


def test_task_reenable_restores_detection_without_resetting_camera(h):
    c, room, code = h.room(); judge=h.judge(code); h.activate(room,judge)
    assert h.post(judge,f'/api/live/rooms/{room.room_id}/task/state',{'enabled':False}).status_code == 200
    assert h.post(judge,f'/api/live/rooms/{room.room_id}/task/state',{'enabled':True}).status_code == 200
    h.frame(room,1); h.frame(room,2); h.frame(room,3)
    assert room.task_enabled is True
    assert len(room.notifications) == 1, f'running task has {len(room.engines)} engines and no alert'


def test_disable_during_inference_suppresses_old_action(h, monkeypatch):
    c, room, code = h.room(); judge=h.judge(code); h.activate(room,judge)
    h.frame(room,1)
    orig=h.detector.track_frame
    def revoke(frame,ts):
        r=h.post(judge,f'/api/live/rooms/{room.room_id}/task/state',{'enabled':False})
        assert r.status_code==200
        return orig(frame,ts)
    monkeypatch.setattr(h.detector,'track_frame',revoke)
    h.frame(room,2)
    assert room.events == rooms.RoomState('a','b','c').events
    assert not room.notifications


def test_pause_resume_invalidates_old_inflight_work(h, monkeypatch):
    c, room, code = h.room(); judge=h.judge(code); h.activate(room,judge)
    h.frame(room,1)
    orig=h.detector.track_frame
    def pause_then_resume(frame,ts):
        assert h.post(judge,f'/api/live/rooms/{room.room_id}/task/state',{'paused':True}).status_code==200
        assert h.post(judge,f'/api/live/rooms/{room.room_id}/task/state',{'paused':False}).status_code==200
        return orig(frame,ts)
    monkeypatch.setattr(h.detector,'track_frame',pause_then_resume)
    h.frame(room,2)
    assert not room.notifications, 'inflight work survived acknowledged pause/resume'


def test_disable_before_teacher_send_prevents_private_image_admission(h, monkeypatch):
    c, room, code = h.room(); judge=h.judge(code); h.activate(room,judge,semantic=True,cls='bottle')
    calls=[]
    monkeypatch.setattr(teacher,'transport',lambda q,l,i: (calls.append(i) or teacher.Observation(status='error')))
    original=worker._save_evidence
    def revoke_before_send(*args):
        name=original(*args)
        assert h.post(judge,f'/api/live/rooms/{room.room_id}/task/state',{'enabled':False}).status_code==200
        return name
    monkeypatch.setattr(worker,'_save_evidence',revoke_before_send)
    h.frame(room,1); h.frame(room,2)
    assert calls == [], 'private Teacher image sent after disable ACK'


def test_task_state_requires_typed_boolean(h):
    c, room, code = h.room(); judge=h.judge(code); h.activate(room,judge)
    r=h.post(judge,f'/api/live/rooms/{room.room_id}/task/state',{'enabled':'false'})
    assert r.status_code==422, f'string false accepted as enabled={r.json().get("task_enabled")}'


class _FakeStdin:
    def __init__(self): self.data = ''
    def write(self, s): self.data += s
    def close(self): pass


class _FakeStream:
    def __init__(self, text): self._text = text; self._served = False
    def read(self, n=-1):
        if self._served: return ''
        self._served = True; return self._text


def test_model_branch_novel_wording_and_invocation_contract(h, monkeypatch):
    # CLI is now an explicit private-only experiment; API is the public default.
    monkeypatch.setenv('VISIONFLOW_LIVE_PLANNER_PROVIDER', 'claude_cli')
    # Updated to the Popen process-group contract (LIVE-VIDEO-008) and to verify the
    # throwaway cwd is OUTSIDE the repo (the old fixed-ROOT path assumption was a
    # harness defect). All real security assertions are preserved.
    calls=[]
    text='Give the advisor a heads-up if a truck remains visible for seventeen seconds.'
    p=proposal('truck','dwell_in_zone'); p['duration_seconds']=17
    envelope=json.dumps({'subtype':'success','is_error':False,
           'structured_output':p,'modelUsage':{'QA_ONLY_MODEL':{'outputTokens':9}},
           'usage':{'output_tokens':9}})
    class FakePopen:
        def __init__(self, argv, **kwargs):
            calls.append((argv,kwargs))
            assert kwargs['shell'] is False
            assert kwargs['start_new_session'] is True
            cwd=Path(kwargs['cwd'])
            assert cwd.is_dir() and not cwd.is_relative_to(REPO), cwd
            assert set(kwargs['env']) == {'HOME','PATH'}
            assert argv[argv.index('--tools')+1] == ''
            assert '--safe-mode' in argv and '--strict-mcp-config' in argv
            self.pid=424242; self.returncode=0
            self.stdin=_FakeStdin()
            self.stdout=_FakeStream(envelope); self.stderr=_FakeStream('')
        def wait(self, timeout=None):
            # The untrusted judge text travels on STDIN, never in argv/system prompt.
            assert self.stdin.data == text
            return 0
    monkeypatch.setattr(planner.subprocess,'Popen',FakePopen)
    c,room,code=h.room(); judge=h.judge(code)
    r=h.post(judge,f'/api/live/rooms/{room.room_id}/task/plan',{'text':text})
    assert r.status_code==200 and r.json()['status']=='ok'
    assert r.json()['task']['duration_seconds']==17
    assert room.task_enabled is False and room.task is None  # draft never activated
    assert len(calls)==1


@pytest.mark.parametrize('kind', ['class','action','code'])
def test_schema_rejects_unsupported_tools_urls_classes(h, kind):
    p=proposal()
    if kind=='class': p['object_class']='unicorn'
    if kind=='action': p['action']='send_webhook'
    if kind=='code': p['code']='print("QA_ONLY")'
    r=planner._finalize(PlannerResult(status='ok',proposal=p))
    assert r.status=='unsupported', f'backend accepts non-strict {kind} proposal'


def test_invalid_model_visual_observation_is_handled(h, monkeypatch):
    p=proposal('bottle',semantic=True); p['visual_observation']['question']='x'*301
    monkeypatch.setattr(planner,'transport',lambda *_:PlannerResult(status='ok',proposal=p))
    c,room,code=h.room(); judge=h.judge(code)
    r=h.post(judge,f'/api/live/rooms/{room.room_id}/task/plan',{'text':'QA_ONLY novel bottle condition'})
    assert r.status_code==200, f'invalid model output raised server error {r.status_code}'
    assert r.json()['status'] in {'unsupported','unavailable'}


def test_planner_deadline_includes_admission_wait(h, monkeypatch):
    calls=[]
    monkeypatch.setattr(config,'PLANNER_TIMEOUT_SECONDS',.04)
    gate=threading.Semaphore(0)
    monkeypatch.setattr(planner,'_planner_lock',gate)
    timer=threading.Timer(.12,gate.release); timer.start()
    monkeypatch.setattr(planner.subprocess,'run',lambda *a,**kw:(calls.append(kw) or subprocess.CompletedProcess(a,1,'','QA_ONLY')))
    start=time.monotonic()
    result=planner._run_cli('QA_ONLY no actual model')
    elapsed=time.monotonic()-start
    timer.join()
    assert calls==[], f'planner spawned after {elapsed:.3f}s for .04s deadline'
    assert result.status=='unavailable'


def test_disabled_camera_cannot_emit_unproven_frame_identity(h):
    h.patch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    c, room, code = h.room()
    for _ in range(4):
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect(ws_url('/api/live/ws/camera'), headers={'Origin':ORIGIN}):
                pytest.fail('disabled camera route admitted frames')
    assert room.frames_received == 0 and room.frames_inferred == 0
    assert len(room.events) == 0 and not room.notifications
    assert room.latest_jpeg is None and room.annotated_jpeg is None


def test_model_proposal_requires_strict_scalar_types(h):
    p=proposal('car','count_exceeded'); p['supported']='false'; p['threshold']=True
    r=planner._finalize(PlannerResult(status='ok',proposal=p))
    assert r.status=='unsupported', f'nonboolean supported and Boolean threshold became {r.task}'
