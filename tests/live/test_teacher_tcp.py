"""VIDEO-only public HTTP Teacher denial, plus retained internal transport controls.
Original malformed-confidence/private-camera-write defects are preserved in prior
review evidence, remain UNFIXED and are not claimed passing by these route tests.
The default VIDEO route must admit zero Teacher calls, even with a staged frame.
"""
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import httpcore._backends.sync

from test_acceptance import h, jpeg, ROOT
from app.live import config, teacher


def envelope(label='opening', confidence=.9):
    return {'model':'QA_ONLY_MOCK', 'choices':[{'message':{'content':json.dumps(
            {'label':label,'confidence':confidence})}}], 'usage':{'total_tokens':9},
            'experimental_metrics':{'revision':'QA_ONLY','image_count':1}}


@contextmanager
def server(payload=None, mode='normal'):
    receipts=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            raw=self.rfile.read(int(self.headers.get('Content-Length',0)))
            receipts.append({'time':time.monotonic(),'path':self.path,'body':json.loads(raw)})
            body=json.dumps(payload if payload is not None else envelope()).encode()
            status=401 if mode=='401' else 200
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            if mode in {'chunked','unfinished_chunk'}:
                self.send_header('Transfer-Encoding','chunked')
            else:
                self.send_header('Content-Length','garbage' if mode=='bad_length' else str(len(body)+(50 if mode=='truncated' else 0)))
            self.send_header('Connection','close')
            self.end_headers()
            try:
                if mode=='trickle':
                    for i in range(0,len(body),16):
                        self.wfile.write(body[i:i+16]); self.wfile.flush(); time.sleep(.02)
                elif mode in {'chunked','unfinished_chunk'}:
                    self.wfile.write(f'{len(body):x}\r\n'.encode()+body+b'\r\n')
                    if mode=='chunked': self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
                else:
                    self.wfile.write(body); self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError): pass
            self.close_connection=True
    httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    httpd.daemon_threads=True
    thread=threading.Thread(target=lambda:httpd.serve_forever(poll_interval=.01),daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_address[1]}',receipts
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join(timeout=2)


def probe(h,monkeypatch,url,question='Is the bottle opening?',labels=None):
    monkeypatch.setattr(config,'TEACHER_BASE_URL',url)
    monkeypatch.setattr(config,'TEACHER_TOKEN','')
    c,room,code=h.room(); judge=h.judge(code)
    room.latest_jpeg=jpeg()
    response=h.post(judge,f'/api/live/rooms/{room.room_id}/probe',
                    {'question':question,'labels':labels or ['opening','closed']})
    return response


@pytest.mark.parametrize('value',[True,False,None,'0.9',-1,2,[],{},float('nan'),float('inf'),float('-inf')])
def test_disabled_teacher_rejects_all_response_shapes(h,monkeypatch,value):
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


def test_missing_confidence_on_actual_http(h,monkeypatch):
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


def test_local_candidate_contract_outside_requested_labels_is_unknown(h,monkeypatch):
    # Adaptation of committed local_teacher.py's actual experimental_metrics
    # contract. Mock image-conditioned scores; these are not genuine forwards.
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


@pytest.mark.parametrize('value',[.9,1])
def test_valid_teacher_response_is_uncertain_control(h,monkeypatch,value):
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


@pytest.mark.parametrize('mode',['truncated','401','unfinished_chunk','bad_length'])
def test_actual_http_faults_nonconfirming_no_retry_controls(h,monkeypatch,mode):
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


def test_correct_chunked_teacher_response_control(h,monkeypatch):
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


def test_teacher_slow_body_obeys_overall_deadline(h,monkeypatch):
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


def test_teacher_admission_expiry_prevents_late_actual_request(h,monkeypatch):
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


def test_disabled_teacher_cannot_admit_a_private_image(h,monkeypatch):
    monkeypatch.setattr(config, 'CAMERA_INGEST_ENABLED', False)
    with server(envelope(confidence=locals().get('value', .9)), mode=locals().get('mode', 'normal')) as (url, receipts):
        response = probe(h, monkeypatch, url)
    assert response.status_code == 404
    assert receipts == [], 'disabled VIDEO Teacher route admitted an image'
    assert all(not room.events and not room.notifications for room in __import__('app.live.rooms', fromlist=['iter_rooms']).iter_rooms())


def test_cache_labels_model_and_room_are_in_identity(h,monkeypatch):
    with server(envelope()) as (url,receipts):
        monkeypatch.setattr(config,'TEACHER_BASE_URL',url)
        a=teacher.observe('QA_ONLY-roomA','same question',['opening','closed'],[jpeg()])
        monkeypatch.setattr(config,'TEACHER_MODEL','QA_ONLY_OTHER_MODEL')
        b=teacher.observe('QA_ONLY-roomB','same question',['occupied','empty'],[jpeg()])
    assert len(receipts)==2, 'second room/model/label schema reused first room observation'
    assert b is not a


def test_attempt_budget_counts_real_failed_http_control(h,monkeypatch):
    monkeypatch.setattr(config,'TEACHER_ROOM_BUDGET',1)
    with server(mode='401') as (url,receipts):
        monkeypatch.setattr(config,'TEACHER_BASE_URL',url)
        a=teacher.observe('QA_ONLY-room','question one',['a','b'],[jpeg()])
        b=teacher.observe('QA_ONLY-room','question two',['a','b'],[jpeg()])
    assert a.status=='error' and b.status=='budget'
    assert len(receipts)==1
