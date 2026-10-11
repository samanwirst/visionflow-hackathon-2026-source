"""Bounded upload/real worker lifetime regressions; QA_ONLY generated MP4.

Direct async production-route calls use real local test sessions/room authority;
no genuine detector/provider, public HTTP, browser or model claim.
"""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.live import config, rooms, video
from app.live import app as live_app
from test_owner_final_security import vid, ORIGIN, HOST


def request(owner,data,delay=0):
    async def stream():
        yield data
        if delay: await asyncio.sleep(delay)
        yield b""
    return SimpleNamespace(headers={"origin":ORIGIN,"host":HOST,
        "x-csrf-token":owner.cookies.get(config.CSRF_COOKIE_NAME)},
        cookies={config.SESSION_COOKIE_NAME:owner.cookies.get(config.SESSION_COOKIE_NAME)},stream=stream)


def test_upload_receive_deadline_cleans_file_and_releases_admission(vid,qa_mp4,monkeypatch):
    owner,room,_=vid.room()
    monkeypatch.setattr(config,"REQUEST_RECEIVE_DEADLINE_SECONDS",.02)
    start=time.monotonic()
    with pytest.raises(HTTPException) as err:
        asyncio.run(live_app.upload_video(room.room_id,request(owner,qa_mp4,delay=.2)))
    assert err.value.status_code==408 and time.monotonic()-start<.15
    assert room.video_path is None
    assert owner.cookies.get(config.SESSION_COOKIE_NAME) not in live_app._upload_owners
    assert not list(rooms.LIVE_VIDEO_DIR.glob(f"room_{room.room_id}_*"))


def test_cancelled_probe_retains_real_global_and_owner_upload_capacity(vid,qa_mp4,monkeypatch):
    owner,room,_=vid.room(); other,second,_=vid.room()
    monkeypatch.setattr(config,"MAX_CONCURRENT_VIDEO_JOBS",1)
    entered=threading.Event();release=threading.Event(); original=video.probe
    def held(path):
        entered.set();assert release.wait(2)
        return original(path)
    monkeypatch.setattr(video,"probe",held)
    async def exercise():
        pending=asyncio.create_task(live_app.upload_video(room.room_id,request(owner,qa_mp4)))
        try:
            assert await asyncio.to_thread(entered.wait,.5), "probe blocks async receive/event loop"
            for applicant,target,status in ((owner,room,429),(other,second,503)):
                with pytest.raises(HTTPException) as err:
                    await live_app.upload_video(target.room_id,request(applicant,qa_mp4))
                assert err.value.status_code==status
            pending.cancel()
            with pytest.raises(asyncio.CancelledError): await pending
            with pytest.raises(HTTPException) as err:
                await live_app.upload_video(second.room_id,request(other,qa_mp4))
            assert err.value.status_code==503, "abandoned decoder released live capacity"
        finally:
            release.set()
        end=time.monotonic()+2
        token=owner.cookies.get(config.SESSION_COOKIE_NAME)
        while token in live_app._upload_owners and time.monotonic()<end: await asyncio.sleep(.005)
        assert token not in live_app._upload_owners
        assert room.video_path is None and not list(rooms.LIVE_VIDEO_DIR.glob(f"room_{room.room_id}_*"))
    asyncio.run(exercise())


def test_owner_logout_during_probe_cannot_attach_a_source(vid,qa_mp4,monkeypatch):
    owner,room,_=vid.room()
    entered=threading.Event();release=threading.Event();original=video.probe
    def held(path):
        entered.set();assert release.wait(2)
        return original(path)
    monkeypatch.setattr(video,"probe",held)
    async def exercise():
        pending=asyncio.create_task(live_app.upload_video(room.room_id,request(owner,qa_mp4)))
        try:
            assert await asyncio.to_thread(entered.wait,.5)
            assert vid.post(owner,"/api/live/logout",{}).status_code==200
        finally:
            release.set()
        with pytest.raises(HTTPException) as err: await pending
        assert err.value.status_code==401
        assert rooms.get_room(room.room_id) is None and room.video_path is None
        assert not list(rooms.LIVE_VIDEO_DIR.glob(f"room_{room.room_id}_*"))
    asyncio.run(exercise())
