"""Core dispatch boundary tests for protected API-run approvals."""

from __future__ import annotations

import json

import model_tools
import pytest
from tools import protected_api_approval as protected


@pytest.fixture(autouse=True)
def discover_tools():
    model_tools.discover_builtin_tools()


def _policy():
    import time

    return protected.ProtectedApiRunApprovalPolicy(
        allowed_tool_names=("write_file",),
        run_id="dispatch-run",
        approval_session="dispatch-approval",
        user_session_id="dispatch-user",
        hermes_session_id="dispatch-hermes",
        conversation_id="dispatch-conversation",
        expires_at=time.time() + 30,
    )


def _args(content="approved"):
    return {"path": "/calendar/event.json", "content": content}


def _approve_callback(store, events, approval_session):
    def callback(event):
        events.append(event)
        store.submit_approval(
            run_id=event["run_id"],
            approval_session=approval_session,
            body={
                "request_id": event["request_id"],
                "action_digest": event["action_digest"],
                "choice": "once",
            },
        )

    return callback


def _run_handle(monkeypatch, args, *, middleware=False, skip_pre=True, hook=None):
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    dispatches = []
    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda name, dispatched_args, **kwargs: dispatches.append((name, dispatched_args)) or json.dumps({"ok": True}),
    )
    if middleware:
        monkeypatch.setattr(
            "hermes_cli.middleware.run_tool_execution_middleware",
            lambda name, middleware_args, next_call, **kwargs: next_call(_args("replacement")),
        )
    if hook is not None:
        monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", hook)
    with protected.bind_protected_api_run(
        policy,
        store=store,
        pending_callback=_approve_callback(store, events, policy.approval_session),
    ):
        result = model_tools.handle_function_call(
            "write_file",
            args,
            task_id="dispatch-task",
            session_id="dispatch-session",
            skip_pre_tool_call_hook=skip_pre,
            skip_tool_execution_middleware=not middleware,
        )
    return result, dispatches, events


def test_skip_pre_hook_cannot_bypass_protected_approval(monkeypatch):
    result, dispatches, events = _run_handle(monkeypatch, _args(), skip_pre=True)

    assert json.loads(result)["ok"] is True
    assert dispatches == [("write_file", _args())]
    assert len(events) == 1
    assert events[0]["choices"] == ["once", "deny"]


def test_hook_failure_still_reaches_core_protected_gate(monkeypatch):
    def broken_hook(*_args, **_kwargs):
        raise RuntimeError("calendar hook failed")

    result, dispatches, events = _run_handle(monkeypatch, _args(), skip_pre=False, hook=broken_hook)

    assert json.loads(result)["ok"] is True
    assert dispatches
    assert len(events) == 1


def test_execution_middleware_replacement_cannot_change_approved_action(monkeypatch):
    result, dispatches, events = _run_handle(monkeypatch, _args(), middleware=True, skip_pre=True)

    assert not dispatches
    assert "protected API approval" in result
    assert len(events) == 1


def test_protected_observer_does_not_receive_raw_arguments(monkeypatch):
    observed = []
    monkeypatch.setattr(
        model_tools,
        "_emit_post_tool_call_hook",
        lambda **kwargs: observed.append(kwargs),
    )
    result, dispatches, events = _run_handle(monkeypatch, _args(), skip_pre=True)

    assert json.loads(result)["ok"] is True
    assert dispatches
    assert observed
    assert all(entry["function_args"] == {} for entry in observed)
    assert len(events) == 1


def test_final_dispatch_authorization_is_single_use_and_exact():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    with protected.bind_protected_api_run(policy, store=store, pending_callback=_approve_callback(store, events, policy.approval_session)):
        decision = protected.require_protected_api_run_approval("write_file", _args())
        assert decision["approved"] is True
        assert protected.verify_protected_dispatch("write_file", _args()) is True
        assert protected.verify_protected_dispatch("write_file", _args()) is False


def test_changed_args_cannot_be_repaired_after_approval():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    with protected.bind_protected_api_run(policy, store=store, pending_callback=_approve_callback(store, events, policy.approval_session)):
        assert protected.require_protected_api_run_approval("write_file", _args())["approved"] is True
        assert protected.verify_protected_dispatch("write_file", _args("changed")) is False
        assert protected.verify_protected_dispatch("write_file", _args()) is False
