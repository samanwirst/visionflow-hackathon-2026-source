"""Owner-scoped mandatory approval + private source playback regressions.

QA_ONLY local FastAPI requests and generated MP4 fixtures. No real model,
provider, public deployment or application video-inference claim.
"""
import hashlib

import pytest

from app.live import config, rooms
from test_owner_final_security import vid, _task, ORIGIN


def _draft(vid, judge, room):
    r=vid.post(judge,f"/api/live/rooms/{room.room_id}/task/validate",{"task":_task()})
    assert r.status_code==200,r.text
    return r.json()


def _approval(draft):
    return {"task":draft["task"],"draft_hash":draft["draft_hash"],
            "expected_version":draft["expected_version"]}


@pytest.mark.parametrize("included",[(),("draft_hash",),("expected_version",)])
def test_video_approval_requires_both_binding_fields_even_for_valid_stored_draft(vid,included):
    _,room,code=vid.room(); judge=vid.judge(code)
    draft=_draft(vid,judge,room)
    body={"task":draft["task"],**{key:draft[key] for key in included}}
    r=vid.post(judge,f"/api/live/rooms/{room.room_id}/task/activate",body)
    assert r.status_code==422,r.text
    assert room.task is None and room.task_version==0 and not room.notifications


def test_video_approval_rejects_changed_task_with_original_digest(vid):
    _,room,code=vid.room(); judge=vid.judge(code)
    body=_approval(_draft(vid,judge,room)); body["task"]={**body["task"],"name":"CHANGED"}
    r=vid.post(judge,f"/api/live/rooms/{room.room_id}/task/activate",body)
    assert r.status_code==422 and room.task is None


def test_video_approval_rejects_replaced_source(vid):
    _,room,code=vid.room(); judge=vid.judge(code)
    body=_approval(_draft(vid,judge,room))
    with room.lock: room.video_sha256="b"*64
    r=vid.post(judge,f"/api/live/rooms/{room.room_id}/task/activate",body)
    assert r.status_code in (409,422) and room.task is None


def test_video_approval_is_room_scoped(vid):
    _,first,code=vid.room(); judge=vid.judge(code)
    body=_approval(_draft(vid,judge,first))
    _,second,code2=vid.room(); other=vid.judge(code2)
    _draft(vid,other,second)
    r=vid.post(other,f"/api/live/rooms/{second.room_id}/task/activate",body)
    assert r.status_code==422 and second.task is None


def test_video_approval_rejects_unknown_request_fields(vid):
    _,room,code=vid.room(); judge=vid.judge(code)
    body={**_approval(_draft(vid,judge,room)),"ignored_constraint":True}
    r=vid.post(judge,f"/api/live/rooms/{room.room_id}/task/activate",body)
    assert r.status_code==422 and room.task is None


def test_owned_source_playback_is_authenticated_room_scoped_and_not_cached(vid,qa_mp4):
    owner,room,code=vid.room(); judge=vid.judge(code)
    up=owner.post(f"/api/live/rooms/{room.room_id}/video",content=qa_mp4,
        headers={"Origin":ORIGIN,"X-CSRF-Token":owner.cookies.get(config.CSRF_COOKIE_NAME),
                 "Content-Type":"video/mp4"})
    assert up.status_code==200,up.text
    sha=hashlib.sha256(qa_mp4).hexdigest(); path=f"/api/live/rooms/{room.room_id}/video/source?sha={sha}"
    for member in (owner,judge):
        r=member.get(path)
        assert r.status_code==200,r.text
        assert r.content==qa_mp4 and r.headers["content-type"]=="video/mp4"
        assert "no-store" in r.headers["cache-control"] and "private" in r.headers["cache-control"]
        assert r.headers.get("vary")=="Cookie"
        assert str(room.video_path) not in str(r.headers)
    assert vid.client().get(path).status_code==401
    _,other,code2=vid.room(); stranger=vid.judge(code2)
    assert stranger.get(path).status_code==403
    assert owner.get(f"/api/live/rooms/{other.room_id}/video/source").status_code==403
    assert judge.get(path.replace(sha,"b"*64)).status_code==409
    assert vid.post(owner,"/api/live/logout",{}).status_code==200
    assert judge.get(path).status_code in (403,404)


def test_owned_source_without_upload_returns_not_found(vid):
    _,room,code=vid.room(); judge=vid.judge(code)
    assert judge.get(f"/api/live/rooms/{room.room_id}/video/source").status_code==404
