"""Invariants for the opt-in protected API-run approval contract."""

from __future__ import annotations

import contextvars
import math
import threading
import time
from contextlib import contextmanager

import pytest

from tools import protected_api_approval as protected


@pytest.fixture(autouse=True)
def discover_tools():
    # The policy validates the exact registry name at the protected boundary.
    import model_tools

    model_tools.discover_builtin_tools()


def _policy(*, run_id="run-a", approval_session="approval-a", expires_at=None, **overrides):
    return protected.ProtectedApiRunApprovalPolicy(
        allowed_tool_names=("write_file",),
        run_id=run_id,
        approval_session=approval_session,
        user_session_id=overrides.pop("user_session_id", "user-a"),
        hermes_session_id=overrides.pop("hermes_session_id", "hermes-a"),
        conversation_id=overrides.pop("conversation_id", "conversation-a"),
        expires_at=time.time() + 30 if expires_at is None else expires_at,
        **overrides,
    )


def _action(path="/calendar/event.json", content="approved"):
    return {"path": path, "content": content}


def _protected_event(events):
    assert events
    event = events[0]
    assert event["approval_type"] == "protected_api_run"
    assert event["choices"] == ["once", "deny"]
    assert "request_id" in event
    assert "action_digest" in event
    assert "args_digest" in event
    assert "args" not in event
    assert "arguments" not in event
    assert "command" not in event
    assert "/calendar/event.json" not in repr(event)
    return event


@contextmanager
def _bound(policy, *, store=None, pending_callback=None):
    with protected.bind_protected_api_run(
        policy,
        store=store or protected.ProtectedApiRunApprovalStore(),
        pending_callback=pending_callback,
    ) as binding:
        yield binding


def test_canonical_action_digest_is_stable_and_bounded():
    args_one = {"content": "approved", "path": "/calendar/event.json"}
    args_two = {"path": "/calendar/event.json", "content": "approved"}

    assert protected.canonical_action_json("write_file", args_one) == protected.canonical_action_json(
        "write_file", args_two
    )
    assert protected.canonical_action_digest("write_file", args_one) == protected.canonical_action_digest(
        "write_file", args_two
    )
    assert protected.canonical_args_digest(args_one) == protected.canonical_args_digest(args_two)

    with pytest.raises(protected.CanonicalActionError):
        protected.canonical_action_json("write_file", {"value": object()})
    with pytest.raises(protected.CanonicalActionError):
        protected.canonical_action_json("write_file", {"value": math.nan})
    with pytest.raises(protected.CanonicalActionError):
        protected.canonical_action_json("write_file", {"value": "x" * (protected.MAX_CANONICAL_ACTION_BYTES + 1)})


def test_policy_allowlist_requires_an_exact_registered_name():
    policy = _policy()
    assert policy.allows_tool("write_file")
    assert not policy.allows_tool("write_file ")

    store = protected.ProtectedApiRunApprovalStore()
    events = []
    with _bound(policy, store=store, pending_callback=events.append):
        result = protected.require_protected_api_run_approval("terminal", {"command": "calendar"})

    assert result["approved"] is False
    assert "allowlist" in result["message"].lower()
    assert events == []


def test_pending_status_payload_has_complete_safe_contract():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy(redacted_description="Create the calendar event")
    result = {}

    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: bool(events))
        event = events[0]
        assert event["request_id"]
        assert event["tool_name"] == "write_file"
        assert len(event["action_digest"]) == 64
        assert event["action_digest"] == event["action_digest"].lower()
        assert all(char in "0123456789abcdef" for char in event["action_digest"])
        assert event["expires_at"] == policy.expires_at
        assert event["choices"] == ["once", "deny"]
        assert event["redacted_description"] == "Create the calendar event"
        assert len(event["redacted_description"]) <= protected.MAX_REDACTED_DESCRIPTION_LENGTH
        assert "/calendar/event.json" not in repr(event)
        assert "approved" not in repr(event)
        store.retire_run(policy.run_id, policy.approval_session)
        worker.join(timeout=2)

    assert result["approved"] is False


def test_status_payload_is_rebuilt_from_server_policy():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy(
        run_id="run_status_policy",
        approval_session="approval_status_policy",
        redacted_description="Server-held calendar action",
    )
    result = {}

    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: bool(events))
        payload = protected.protected_api_run_status_payload(
            policy.run_id, policy.approval_session, events[0]["request_id"]
        )
        assert payload["request_id"] == events[0]["request_id"]
        assert payload["tool_name"] == "write_file"
        assert payload["action_digest"] == events[0]["action_digest"]
        assert payload["expires_at"] == policy.expires_at
        assert payload["choices"] == ["once", "deny"]
        assert payload["redacted_description"] == "Server-held calendar action"
        assert "/calendar/event.json" not in repr(payload)
        store.retire_run(policy.run_id, policy.approval_session)
        worker.join(timeout=2)

    assert result["approved"] is False


def test_retired_status_payload_is_not_available():
    store = protected._DEFAULT_STORE
    events = []
    policy = _policy(run_id="run_retired_status", approval_session="approval_retired_status")
    result = {}

    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: bool(events))
        store.retire_run(policy.run_id, policy.approval_session)
        worker.join(timeout=2)

        with pytest.raises(protected.ProtectedApprovalError) as exc:
            protected.protected_api_run_status_payload(
                policy.run_id, policy.approval_session, events[0]["request_id"]
            )
        assert exc.value.code == "approval_not_pending"

    assert result["approved"] is False


def test_attached_non_default_store_remains_visible_to_api_resolution():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy(run_id="run_custom_store", approval_session="approval_custom_store")
    result = {}

    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: bool(events))
        assert protected.is_protected_api_run(policy.run_id, policy.approval_session)

        response = protected.submit_protected_api_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body={
                "request_id": events[0]["request_id"],
                "action_digest": events[0]["action_digest"],
                "choice": "once",
            },
        )
        assert response["choice"] == "once"
        worker.join(timeout=2)

    assert result["approved"] is True


def test_registered_policy_is_not_silently_bypassed_by_core_gate():
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy(run_id="run_registered_policy", approval_session="approval_registered_policy")
    events = []

    def approve(event):
        events.append(event)
        store.submit_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body={
                "request_id": event["request_id"],
                "action_digest": event["action_digest"],
                "choice": "once",
            },
        )

    tokens = protected.set_current_api_run_context(
        run_id=policy.run_id,
        approval_session=policy.approval_session,
        pending_callback=approve,
    )
    try:
        store.register_policy(policy)
        result = protected.require_protected_api_run_approval("write_file", _action())
        verified = protected.verify_protected_dispatch("write_file", _action())
    finally:
        store.retire_run(policy.run_id, policy.approval_session)
        protected.reset_current_api_run_context(tokens)

    assert result["approved"] is True
    assert verified is True
    assert events


def test_equal_policy_re_registration_invalidates_old_dispatch_authorization():
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy(run_id="run_epoch", approval_session="approval_epoch")

    def approve(event):
        store.submit_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body={
                "request_id": event["request_id"],
                "action_digest": event["action_digest"],
                "choice": "once",
            },
        )

    tokens = protected.set_current_api_run_context(
        run_id=policy.run_id,
        approval_session=policy.approval_session,
        pending_callback=approve,
    )
    try:
        store.register_policy(policy, pending_callback=approve)
        result = protected.require_protected_api_run_approval("write_file", _action())
        assert result["approved"] is True
        assert protected.has_current_protected_dispatch_authorization()

        store.retire_run(policy.run_id, policy.approval_session)
        store.register_policy(policy, pending_callback=approve)

        assert protected.verify_protected_dispatch("write_file", _action()) is False
        assert not protected.has_current_protected_dispatch_authorization()
    finally:
        store.retire_run(policy.run_id, policy.approval_session)
        protected.reset_current_api_run_context(tokens)


def test_final_authorization_state_is_cleared_on_policy_early_return_and_resolution_error(monkeypatch):
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy(run_id="run_context_cleanup", approval_session="approval_context_cleanup")

    def approve(event):
        store.submit_approval(
            run_id=event["run_id"],
            approval_session=policy.approval_session,
            body={
                "request_id": event["request_id"],
                "action_digest": event["action_digest"],
                "choice": "once",
            },
        )

    with _bound(policy, store=store, pending_callback=approve):
        assert protected.require_protected_api_run_approval("write_file", _action())["approved"] is True
        assert protected.has_current_protected_dispatch_authorization()

        mismatched = _policy(run_id="different-run", approval_session="different-approval")
        blocked = protected.require_protected_api_run_approval(
            "write_file", _action(), policy=mismatched
        )
        assert blocked["approved"] is False
        assert not protected.has_current_protected_dispatch_authorization()

        monkeypatch.setattr(
            protected,
            "_resolve_current_protected_binding",
            lambda: (_ for _ in ()).throw(RuntimeError("resolution failed")),
        )
        failed = protected.require_protected_api_run_approval("write_file", _action())
        assert failed["approved"] is False
        assert not protected.has_current_protected_dispatch_authorization()


def test_consumed_records_are_removed_and_pending_capacity_is_bounded():
    store = protected.ProtectedApiRunApprovalStore(max_records=1)
    policy = _policy(run_id="run_record_bound", approval_session="approval_record_bound")
    events = []
    first_result = {}
    second_result = {}

    with _bound(policy, store=store, pending_callback=events.append):
        first = threading.Thread(
            target=_thread_target(
                lambda: first_result.update(
                    protected.require_protected_api_run_approval(
                        "write_file", _action("/calendar/first.json")
                    )
                )
            )
        )
        first.start()
        assert _wait_for(lambda: len(events) == 1)

        second = threading.Thread(
            target=_thread_target(
                lambda: second_result.update(
                    protected.require_protected_api_run_approval(
                        "write_file", _action("/calendar/second.json")
                    )
                )
            )
        )
        second.start()
        assert _wait_for(lambda: bool(second_result))
        assert second_result["approved"] is False
        assert "capacity" in second_result["message"].lower()

        store.submit_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body={
                "request_id": events[0]["request_id"],
                "action_digest": events[0]["action_digest"],
                "choice": "deny",
            },
        )
        first.join(timeout=2)
        assert not first.is_alive()
        assert first_result["approved"] is False
        assert store.pending_record_count == 0


def test_pending_metadata_and_body_require_exact_request_digest_and_choice():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    result = {}

    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: bool(events))
        event = _protected_event(events)

        with pytest.raises(protected.ProtectedApprovalError) as exc:
            store.submit_approval(
                run_id=policy.run_id,
                approval_session=policy.approval_session,
                body={
                    "request_id": event["request_id"],
                    "action_digest": "0" * 64,
                    "choice": "once",
                },
            )
        assert exc.value.code == "approval_action_mismatch"
        assert worker.is_alive()

        with pytest.raises(protected.ProtectedApprovalError) as exc:
            store.submit_approval(
                run_id=policy.run_id,
                approval_session=policy.approval_session,
                body={
                    "request_id": event["request_id"],
                    "action_digest": event["action_digest"],
                    "choice": "session",
                },
            )
        assert exc.value.code == "invalid_approval_choice"

        response = store.submit_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body={
                "request_id": event["request_id"],
                "action_digest": event["action_digest"],
                "choice": "once",
            },
        )
        assert response["choice"] == "once"
        worker.join(timeout=2)

    assert result["approved"] is True


def test_replay_is_single_use_and_fifo_or_resolve_all_cannot_be_used():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    with _bound(policy, store=store, pending_callback=events.append):
        result = {}
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: len(events) == 1)
        event = events[0]
        with pytest.raises(protected.ProtectedApprovalError) as exc:
            store.submit_approval(
                run_id=policy.run_id,
                approval_session=policy.approval_session,
                body={
                    "request_id": event["request_id"],
                    "action_digest": event["action_digest"],
                    "choice": "once",
                    "resolve_all": True,
                },
            )
        assert exc.value.code == "invalid_approval_body"
        store.submit_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body={
                "request_id": event["request_id"],
                "action_digest": event["action_digest"],
                "choice": "once",
            },
        )
        worker.join(timeout=2)

        with pytest.raises(protected.ProtectedApprovalError) as exc:
            store.submit_approval(
                run_id=policy.run_id,
                approval_session=policy.approval_session,
                body={
                    "request_id": event["request_id"],
                    "action_digest": event["action_digest"],
                    "choice": "once",
                },
            )
        assert exc.value.code == "approval_not_pending"


def test_expiry_blocks_approval_and_pending_waiter():
    now = [100.0]
    store = protected.ProtectedApiRunApprovalStore(clock=lambda: now[0])
    policy = _policy(expires_at=110.0)
    events = []
    result = {}
    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: len(events) == 1)
        now[0] = 111.0
        with pytest.raises(protected.ProtectedApprovalError) as exc:
            store.submit_approval(
                run_id=policy.run_id,
                approval_session=policy.approval_session,
                body={
                    "request_id": events[0]["request_id"],
                    "action_digest": events[0]["action_digest"],
                    "choice": "once",
                },
            )
        assert exc.value.code == "approval_expired"
        worker.join(timeout=2)
        assert store.pending_record_count == 0
    assert result["approved"] is False
    assert "expired" in result["message"].lower()


def test_cross_run_and_cross_session_resolution_cannot_touch_pending_request():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    result = {}
    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: len(events) == 1)
        body = {
            "request_id": events[0]["request_id"],
            "action_digest": events[0]["action_digest"],
            "choice": "once",
        }
        for run_id, approval_session in (("run-b", policy.approval_session), (policy.run_id, "approval-b")):
            with pytest.raises(protected.ProtectedApprovalError) as exc:
                store.submit_approval(run_id=run_id, approval_session=approval_session, body=body)
            assert exc.value.code == "approval_not_pending"
        assert worker.is_alive()
        store.submit_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body=body,
        )
        worker.join(timeout=2)
    assert result["approved"] is True


def test_concurrent_submissions_have_one_winner():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    result = {}
    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: len(events) == 1)
        event = events[0]
        outcomes = []
        barrier = threading.Barrier(3)

        def submit():
            barrier.wait()
            try:
                outcomes.append(
                    store.submit_approval(
                        run_id=policy.run_id,
                        approval_session=policy.approval_session,
                        body={
                            "request_id": event["request_id"],
                            "action_digest": event["action_digest"],
                            "choice": "once",
                        },
                    )
                )
            except protected.ProtectedApprovalError as exc:
                outcomes.append(exc.code)

        first = threading.Thread(target=submit)
        second = threading.Thread(target=submit)
        first.start()
        second.start()
        barrier.wait()
        first.join(timeout=2)
        second.join(timeout=2)
        worker.join(timeout=2)

    assert len([outcome for outcome in outcomes if isinstance(outcome, dict)]) == 1
    assert result["approved"] is True


def test_always_and_session_choices_are_not_protected_choices():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    result = {}
    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: len(events) == 1)
        for choice in ("session", "always"):
            with pytest.raises(protected.ProtectedApprovalError) as exc:
                store.submit_approval(
                    run_id=policy.run_id,
                    approval_session=policy.approval_session,
                    body={
                        "request_id": events[0]["request_id"],
                        "action_digest": events[0]["action_digest"],
                        "choice": choice,
                    },
                )
            assert exc.value.code == "invalid_approval_choice"
        store.submit_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body={
                "request_id": events[0]["request_id"],
                "action_digest": events[0]["action_digest"],
                "choice": "deny",
            },
        )
        worker.join(timeout=2)
    assert result["approved"] is False


def test_generic_always_or_approvals_off_cannot_authorize_protected_call(monkeypatch):
    # Protected policy never consults generic approval caches or settings.
    monkeypatch.setattr("tools.approval.is_approved", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("tools.approval.request_tool_approval", lambda *_args, **_kwargs: {"approved": True})
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    result = {}
    with _bound(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_thread_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", _action())
                )
            )
        )
        worker.start()
        assert _wait_for(lambda: len(events) == 1)
        assert worker.is_alive()
        store.submit_approval(
            run_id=policy.run_id,
            approval_session=policy.approval_session,
            body={
                "request_id": events[0]["request_id"],
                "action_digest": events[0]["action_digest"],
                "choice": "deny",
            },
        )
        worker.join(timeout=2)
    assert result["approved"] is False


def _thread_target(callback):
    context = contextvars.copy_context()
    return lambda: context.run(callback)


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())
