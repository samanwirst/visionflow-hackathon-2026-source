"""Authorized action execution.

M1 ships a single real, local, side-effect-free action:

- ``create_notification`` persists a Notification row (shown in the UI).

Arbitrary outbound webhooks (``send_webhook``) were intentionally removed
(M1-011 / M1-012). The previous SSRF/allowlist guard could authorize a hostname
that then re-resolved to an internal address for the actual delivery connection
(DNS rebinding), and an interrupted delivery could not be distinguished from a
completed one on recovery, risking duplicate external side effects. Rather than
ship a risky custom DNS/TLS egress transport, external delivery is disabled.
Any workflow (including a legacy row persisted before removal) that requests an
unsupported action is recorded as ``rejected`` with a clear reason and performs
**no** network lookup or delivery.

Execution is idempotent per event: the engine's dedupe_key makes each stored
event unique, and the retained local action runs exactly once per stored event
within the same transaction as its event-status update, so an interrupted run
either commits fully or rolls back fully (no partial external effect).
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.db import ActionExecution, Notification, TrackEvent, utcnow

logger = logging.getLogger("visionflow.actions")

# Actions that are safe to execute in M1. Anything outside this set is rejected
# without performing any external side effect.
SUPPORTED_ACTIONS = frozenset({"create_notification"})


def execute_action(session: Session, event: TrackEvent, action_type: str, action_config: dict) -> None:
    """Run one action for one stored event; records an ActionExecution row.

    The ActionExecution row, the Notification (on success) and the event's
    ``action_status`` are all committed together, so a crash before the commit
    leaves nothing behind and recovery can safely re-dispatch exactly once."""
    execution = ActionExecution(
        event_id=event.id, action_type=action_type, status="pending", attempt_count=1
    )
    session.add(execution)
    session.flush()

    try:
        if action_type == "create_notification":
            title = action_config.get("title") or f"VisionFlow: {event.type}"
            session.add(
                Notification(
                    event_id=event.id,
                    title=title,
                    message=event.detail or f"{event.type} at {event.video_timestamp:.1f}s",
                )
            )
            execution.status = "success"
        else:
            # Unsupported or disabled action (e.g. a legacy send_webhook row).
            # Reject visibly without any network lookup or delivery.
            execution.status = "rejected"
            execution.error = (
                f"action type '{action_type}' is not supported; external webhook "
                "delivery was removed (see actions.py). No request was sent."
            )
            logger.warning(
                "rejected unsupported action '%s' for event %s (no side effect)",
                action_type, event.id,
            )
    except Exception as exc:  # noqa: BLE001 — record, don't crash the pipeline
        execution.status = "failed"
        execution.error = str(exc)[:1000]
        logger.warning("action %s failed for event %s: %s", action_type, event.id, exc)

    execution.executed_at = utcnow()
    event.action_status = "done" if execution.status == "success" else "failed"
    session.commit()
