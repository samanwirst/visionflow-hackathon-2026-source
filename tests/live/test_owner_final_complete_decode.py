"""QA_ONLY metadata/transport fixtures; no model inference claims."""
import os
import time
from pathlib import Path
import pytest
from app.live import rooms, video


def info(decoded=60, advertised=60, avg="25/1"):
    return {"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "2.4"},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 320,
                         "height": 240, "r_frame_rate": "25/1", "avg_frame_rate": avg,
                         "nb_frames": str(advertised), "nb_read_frames": str(decoded)}]}


def test_even_one_missing_frame_cannot_be_marked_valid(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "_ffprobe", lambda path: info(decoded=59))
    with pytest.raises(video.VideoError, match="incomplete/truncated"):
        video.probe(tmp_path / "QA_ONLY.mp4")


def test_complete_decoded_count_is_the_admitted_count(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "_ffprobe", lambda path: info())
    meta = video.probe(tmp_path / "QA_ONLY.mp4")
    assert meta["frame_count"] == 60 and meta["frame_count_reliable"] is True


def test_variable_frame_clock_is_explicitly_unsupported(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "_ffprobe", lambda path: info(avg="20/1"))
    with pytest.raises(video.VideoError, match="variable frame rate"):
        video.probe(tmp_path / "QA_ONLY.mp4")


def test_no_unbounded_native_decoder_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "_ffprobe", lambda path: None)
    with pytest.raises(video.VideoError, match="ffprobe is required"):
        video.probe(tmp_path / "QA_ONLY.mp4")


def test_full_decode_subprocess_is_killed_and_reaped_on_timeout(monkeypatch, tmp_path):
    exe = tmp_path / "QA_ONLY_sleeping_probe"
    exe.write_text("#!/usr/bin/env python3\nimport os,time\nfrom pathlib import Path\nPath(" + repr(str(tmp_path / "pid")) + ").write_text(str(os.getpid()))\ntime.sleep(30)\n")
    exe.chmod(0o700)
    monkeypatch.setattr(video, "FFPROBE_BIN", str(exe))
    monkeypatch.setattr(video, "_FFPROBE_TIMEOUT_SECONDS", 0.15)
    start = time.monotonic()
    with pytest.raises(video.VideoError, match="timed out"):
        video._ffprobe(tmp_path / "QA_ONLY.mp4")
    assert time.monotonic() - start < 2
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_source_attachment_authority_rechecked_before_any_mutation(tmp_path):
    room = rooms.create_room("QA_ONLY_owner")
    old, new = tmp_path / "old.mp4", tmp_path / "new.mp4"
    old.write_bytes(b"QA_ONLY_old"); new.write_bytes(b"QA_ONLY_new")
    room.video_path = old; room.video_sha256 = "old"; room.active_generation = 123
    with pytest.raises(video.VideoError, match="authority revoked"):
        video.replace_source(room, new, "new", {"frame_count": 1}, authorize=lambda: False)
    assert room.video_path == old and room.video_sha256 == "old"
    assert room.active_generation == 123 and old.exists()
    rooms.close_room(room.room_id)


def test_retired_room_cannot_receive_a_late_uploaded_source(tmp_path):
    room = rooms.create_room("QA_ONLY_owner")
    rooms.close_room(room.room_id)
    new = tmp_path / "new.mp4"; new.write_bytes(b"QA_ONLY_new")
    with pytest.raises(video.VideoError, match="retired"):
        video.replace_source(room, new, "new", {"frame_count": 1}, authorize=lambda: True)
    assert room.video_path is None
