"""QA_ONLY generated MP4 catalog fixtures; no model inference or public footage."""
import hashlib

import pytest

from app.live import auth, config, rooms
from test_owner_final_security import vid, ORIGIN, _task


@pytest.fixture
def catalog(tmp_path, monkeypatch, qa_mp4):
    directory = tmp_path / "prepared"
    directory.mkdir()
    for i in range(1, 6):
        (directory / f"Camera{i}-960x540-cfr24.mp4").write_bytes(qa_mp4)
    monkeypatch.setenv("VISIONFLOW_DEMO_VIDEO_DIR", str(directory))
    monkeypatch.setattr(rooms, "LIVE_VIDEO_DIR", tmp_path / "private")
    return directory


def _select(vid, owner, room, body):
    return vid.post(owner, f"/api/live/rooms/{room.room_id}/video/select", body)


def test_catalog_requires_authenticated_current_room_owner(vid, catalog):
    assert vid.client().get("/api/live/video-catalog").status_code == 401
    assert vid.client("camera").get("/api/live/video-catalog").status_code == 403
    owner, room, code = vid.room()
    judge = vid.judge(code)
    assert judge.get("/api/live/video-catalog").status_code == 403
    response = owner.get("/api/live/video-catalog")
    assert response.status_code == 200
    assert [x["id"] for x in response.json()["inputs"]] == [f"Camera{i}" for i in range(1, 6)]
    assert all(x["label"] == "owner-provided recorded video; visual interpretation requires validated inference"
               for x in response.json()["inputs"])
    assert "no-store" in response.headers["cache-control"]
    assert str(catalog) not in response.text


def test_unconfigured_catalog_is_honestly_unavailable(vid, monkeypatch):
    monkeypatch.delenv("VISIONFLOW_DEMO_VIDEO_DIR", raising=False)
    owner, room, _ = vid.room()
    assert owner.get("/api/live/video-catalog").json() == {"available": False, "inputs": []}
    assert _select(vid, owner, room, {"id": "Camera1"}).status_code == 404


@pytest.mark.parametrize("body", [{"id": "../Camera1"}, {"id": "/etc/passwd"},
    {"id": "Camera6"}, {"id": "Camera1", "path": "/etc/passwd"}, {"id": True}, {}])
def test_select_rejects_paths_unknown_ids_and_extra_fields(vid, catalog, body):
    owner, room, _ = vid.room()
    assert _select(vid, owner, room, body).status_code == 422
    assert room.video_path is None


def test_selection_is_csrf_protected_owner_only_and_room_scoped(vid, catalog):
    owner, room, code = vid.room()
    judge = vid.judge(code)
    endpoint = f"/api/live/rooms/{room.room_id}/video/select"
    assert vid.client().post(endpoint, json={"id": "Camera1"}, headers={"Origin": ORIGIN}).status_code == 401
    assert owner.post(endpoint, json={"id": "Camera1"}, headers={"Origin": ORIGIN}).status_code == 403
    assert _select(vid, judge, room, {"id": "Camera1"}).status_code == 403
    second_owner, second, _ = vid.room()
    assert _select(vid, second_owner, room, {"id": "Camera1"}).status_code == 403
    assert _select(vid, owner, second, {"id": "Camera1"}).status_code == 403
    assert room.video_path is None and second.video_path is None


def test_selection_copies_actual_source_and_preserves_private_source_identity(vid, catalog, qa_mp4):
    owner, room, code = vid.room()
    judge = vid.judge(code)
    draft = vid.post(judge, f"/api/live/rooms/{room.room_id}/task/validate", {"task": _task()}).json()
    response = _select(vid, owner, room, {"id": "Camera1"})
    assert response.status_code == 200, response.text
    source_sha = hashlib.sha256(qa_mp4).hexdigest()
    assert response.json()["sha256"] == source_sha
    assert response.json()["meta"]["frame_count"] == 15
    assert room.video_path.parent == rooms.LIVE_VIDEO_DIR
    assert room.video_path.name.startswith(f"room_{room.room_id}_")
    assert room.video_path.read_bytes() == qa_mp4
    assert room.video_path != catalog / "Camera1-960x540-cfr24.mp4"
    assert room.task is None and not room.notifications
    assert vid.post(judge, f"/api/live/rooms/{room.room_id}/task/activate", draft).status_code == 409
    assert judge.get(f"/api/live/rooms/{room.room_id}/video/source?sha={source_sha}").content == qa_mp4
    old = room.video_path
    again = _select(vid, owner, room, {"id": "Camera2"})
    assert again.status_code == 200 and room.video_path != old and not old.exists()
    assert (catalog / "Camera1-960x540-cfr24.mp4").read_bytes() == qa_mp4


def test_catalog_rejects_symlinks_and_invalid_video(vid, catalog, tmp_path):
    owner, room, _ = vid.room()
    path = catalog / "Camera1-960x540-cfr24.mp4"
    path.unlink()
    target = tmp_path / "outside.mp4"
    target.write_bytes(b"not a video")
    path.symlink_to(target)
    assert _select(vid, owner, room, {"id": "Camera1"}).status_code == 404
    assert "Camera1" not in [x["id"] for x in owner.get("/api/live/video-catalog").json()["inputs"]]
    path.unlink(); path.write_bytes(b"not a video")
    assert _select(vid, owner, room, {"id": "Camera1"}).status_code == 422
    assert room.video_path is None


def test_prepared_selection_reuses_upload_admission(vid, catalog, monkeypatch):
    from app.live import app as live_app
    owner, room, _ = vid.room()
    sess = next(s for s in auth._sessions.values()
                if s.room_id == room.room_id and s.owns_room)
    live_app._claim_upload(sess.token)
    try:
        assert _select(vid, owner, room, {"id": "Camera1"}).status_code == 429
    finally:
        live_app._release_upload(sess.token)
    assert room.video_path is None
