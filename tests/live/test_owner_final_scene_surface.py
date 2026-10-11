"""QA_ONLY scene-route authorization and stale-session integration controls."""
import pytest
from app.live import auth, config, scene
from test_owner_final_security import vid, ORIGIN


def _request(vid, client, room, **patch):
    body = {"question": "What is visible?", "source_sha256": "a" * 64,
            "expected_version": room.task_version}
    body.update(patch)
    return vid.post(client, f"/api/live/rooms/{room.room_id}/scene", body)


def test_scene_requires_judge_csrf_and_room_membership(vid, monkeypatch):
    calls = []
    monkeypatch.setattr(scene, "inspect_scene", lambda *args, **kw: calls.append(1) or {})
    owner, room, code = vid.room()
    judge = vid.judge(code)
    url = f"/api/live/rooms/{room.room_id}/scene"
    assert vid.client().post(url, json={}, headers={"Origin": ORIGIN}).status_code == 401
    assert _request(vid, owner, room).status_code == 403
    assert judge.post(url, json={}, headers={"Origin": ORIGIN}).status_code == 403
    other, second, _ = vid.room()
    assert _request(vid, judge, second).status_code == 403
    assert calls == []


@pytest.mark.parametrize("patch", [{"extra": "ignored"}, {"expected_version": True},
    {"expected_version": -1}, {"source_sha256": "../file"}, {"question": ""},
    {"question": "x" * 1001}, {"question": 3}])
def test_scene_rejects_invalid_and_unknown_fields_before_provider(vid, monkeypatch, patch):
    monkeypatch.setattr(scene, "inspect_scene", lambda *a, **k: pytest.fail("provider invoked"))
    _, room, code = vid.room()
    assert _request(vid, vid.judge(code), room, **patch).status_code == 422


def test_scene_provides_current_session_guard_without_authorizing_task(vid, monkeypatch):
    _, room, code = vid.room()
    judge = vid.judge(code)
    session = auth.get_session(judge.cookies.get(config.SESSION_COOKIE_NAME))
    checks = []
    def inspect(actual_room, question, source, version, *, session_is_current):
        assert actual_room is room and source == "a" * 64 and version == room.task_version
        checks.append(session_is_current())
        auth.destroy_session(session.token)
        checks.append(session_is_current())
        return {"status": "stale", "nonactioning": True}
    monkeypatch.setattr(scene, "inspect_scene", inspect)
    response = _request(vid, judge, room)
    assert response.json() == {"status": "stale", "nonactioning": True}
    assert checks == [True, False]
    assert room.task is None and not room.events and not room.notifications
