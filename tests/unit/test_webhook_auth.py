"""External webhook delivery removal (M1-011 / M1-012).

Arbitrary outbound webhooks were removed because the SSRF/allowlist guard could
not safely protect the actual delivery connection against DNS rebinding without
a risky custom DNS/TLS egress transport, and an interrupted delivery could not
be distinguished from a completed one on recovery.

These tests prove that a legacy ``send_webhook`` action is now rejected
*without any network lookup or delivery*, so neither the rebinding nor the
duplicate-delivery scenario can occur. ``socket.getaddrinfo`` is monkeypatched
to fail loudly if the action path ever attempts a DNS resolution.
"""

import socket

import pytest

from app import actions, db


@pytest.fixture
def session(tmp_path):
    db.init_db(str(tmp_path / "wh.db"))
    s = db.get_session()
    try:
        yield s
    finally:
        s.close()


def _make_event(session) -> db.TrackEvent:
    ev = db.TrackEvent(
        source_id=1,
        workflow_id=1,
        track_id=1,
        type="object_entered_zone",
        class_name="car",
        video_timestamp=1.0,
        confidence=0.9,
        detail="car entered",
        dedupe_key=f"k-{id(object())}",
    )
    session.add(ev)
    session.commit()
    return ev


def test_legacy_webhook_action_rejected_without_network(session, monkeypatch):
    """A persisted legacy send_webhook workflow must be rejected at execution
    with no DNS lookup and no delivery (M1-012: no network for a disabled
    feature)."""

    def exploding_lookup(*args, **kwargs):
        raise AssertionError("no DNS lookup may occur for a removed webhook action")

    monkeypatch.setattr(socket, "getaddrinfo", exploding_lookup)

    ev = _make_event(session)
    actions.execute_action(
        session, ev, "send_webhook", {"url": "http://approved.example:9876/hook"}
    )

    exec_row = (
        session.query(db.ActionExecution)
        .filter(db.ActionExecution.event_id == ev.id)
        .one()
    )
    assert exec_row.status == "rejected"
    assert "not supported" in (exec_row.error or "").lower()
    # No notification/side effect was produced.
    assert session.query(db.Notification).count() == 0
    assert ev.action_status == "failed"


def test_notification_action_still_works(session):
    """The retained local action remains fully functional."""
    ev = _make_event(session)
    actions.execute_action(session, ev, "create_notification", {"title": "bay alert"})

    exec_row = (
        session.query(db.ActionExecution)
        .filter(db.ActionExecution.event_id == ev.id)
        .one()
    )
    assert exec_row.status == "success"
    notif = session.query(db.Notification).filter(db.Notification.event_id == ev.id).one()
    assert notif.title == "bay alert"
    assert ev.action_status == "done"


def test_webhook_helpers_removed():
    """The SSRF/authorization helpers are gone; nothing can re-enable delivery
    by calling them."""
    assert not hasattr(actions, "_authorize_webhook")
    assert not hasattr(actions, "_post_webhook")
    assert not hasattr(actions, "webhook_transport")
