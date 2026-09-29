"""One-use dispatch attestation: handler can prove its exact approved call.

verify_protected_dispatch consumes _CURRENT_AUTHORIZATION before registry.dispatch
reaches the handler; the handler cannot re-examine that token.  These tests assert
the new contract: after a successful final dispatch, the handler sees a per-call
attestation scoped to its own invocation and cleared in every exit path.
"""

from __future__ import annotations

import json
import threading
import time

import model_tools
import pytest
from tools import protected_api_approval as protected


@pytest.fixture(autouse=True)
def _discover():
    model_tools.discover_builtin_tools()


def _policy():
    return protected.ProtectedApiRunApprovalPolicy(
        allowed_tool_names=("write_file",),
        run_id="attest-run",
        approval_session="attest-session",
        user_session_id="attest-user",
        hermes_session_id="attest-hermes",
        conversation_id="attest-convo",
        expires_at=time.time() + 30,
    )


def _args():
    return {"path": "/calendar/event.json", "content": "attest-ok"}


def _auto_approve(store, policy):
    def callback(event):
        store.submit_approval(
            run_id=event["run_id"],
            approval_session=policy.approval_session,
            body={
                "request_id": event["request_id"],
                "action_digest": event["action_digest"],
                "choice": "once",
            },
        )
    return callback


def test_handler_sees_attestation_for_approved_exact_action(monkeypatch):
    """Handler must be able to confirm its call passed the approved dispatch gate."""
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()
    seen = []

    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )

    def recording_dispatch(name, args, **kwargs):
        seen.append(protected.has_current_dispatch_attestation(name, args))
        return json.dumps({"ok": True})

    monkeypatch.setattr(model_tools.registry, "dispatch", recording_dispatch)

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=_auto_approve(store, policy)
    ):
        result = model_tools.handle_function_call(
            "write_file",
            _args(),
            task_id="t1",
            session_id="s1",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )

    assert json.loads(result)["ok"] is True
    assert seen == [True], f"handler attestation should be True, got {seen}"


def test_attestation_not_available_after_handler_returns(monkeypatch):
    """Attestation is cleared once registry.dispatch returns."""
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()

    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda name, args, **kw: json.dumps({"ok": True}),
    )

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=_auto_approve(store, policy)
    ):
        model_tools.handle_function_call(
            "write_file",
            _args(),
            task_id="t2",
            session_id="s2",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )
        # After dispatch the attestation must be gone even inside the run context.
        assert not protected.has_current_dispatch_attestation("write_file", _args())


def test_attestation_mismatch_on_wrong_args(monkeypatch):
    """Attestation check returns False when args differ from the approved call."""
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()
    mismatch_results = []

    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )

    def checking_dispatch(name, args, **kwargs):
        wrong_args = {"path": "/other/path.json", "content": "not-approved"}
        mismatch_results.append(protected.has_current_dispatch_attestation(name, wrong_args))
        return json.dumps({"ok": True})

    monkeypatch.setattr(model_tools.registry, "dispatch", checking_dispatch)

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=_auto_approve(store, policy)
    ):
        model_tools.handle_function_call(
            "write_file",
            _args(),
            task_id="t3",
            session_id="s3",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )

    assert mismatch_results == [False], f"mismatched args must return False, got {mismatch_results}"


def test_no_attestation_without_approved_protected_call(monkeypatch):
    """has_current_dispatch_attestation returns False without a protected approval."""
    # No bind_protected_api_run — plain unprotected call.
    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )
    unprotected_seen = []

    def checking_dispatch(name, args, **kwargs):
        unprotected_seen.append(protected.has_current_dispatch_attestation(name, args))
        return json.dumps({"ok": True})

    monkeypatch.setattr(model_tools.registry, "dispatch", checking_dispatch)

    model_tools.handle_function_call(
        "write_file",
        _args(),
        task_id="t4",
        session_id="s4",
        skip_pre_tool_call_hook=True,
        skip_tool_execution_middleware=True,
    )

    assert unprotected_seen == [False], f"no attestation without approval, got {unprotected_seen}"


def test_active_api_run_gets_one_use_attestation_for_core_dispatched_unprotected_call(monkeypatch):
    """A live API run may attest exact dispatch without creating an approval binding."""
    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )
    results = []

    def checking_dispatch(name, args, **kwargs):
        results.append(protected.consume_current_dispatch_attestation(name, args))
        results.append(protected.consume_current_dispatch_attestation(name, args))
        return json.dumps({"ok": True})

    monkeypatch.setattr(model_tools.registry, "dispatch", checking_dispatch)
    tokens = protected.set_current_api_run_context(
        run_id="unapproved-run", approval_session="unapproved-session"
    )
    try:
        model_tools.handle_function_call(
            "write_file",
            _args(),
            task_id="active-api-task",
            session_id="active-api-session",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )
    finally:
        protected.reset_current_api_run_context(tokens)

    assert results == [True, False]


def test_direct_dispatch_in_active_api_run_has_no_attestation(monkeypatch):
    """Only model_tools' final dispatch boundary can mint the one-use proof."""
    seen = []
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda name, args, **kwargs: seen.append(
            protected.consume_current_dispatch_attestation(name, args)
        ) or json.dumps({"ok": True}),
    )
    tokens = protected.set_current_api_run_context(
        run_id="direct-run", approval_session="direct-session"
    )
    try:
        model_tools.registry.dispatch("write_file", _args())
    finally:
        protected.reset_current_api_run_context(tokens)
    assert seen == [False]


def test_attestation_cleared_when_handler_raises(monkeypatch):
    """Attestation is cleared even if the registry handler raises."""
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()

    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda name, args, **kw: (_ for _ in ()).throw(RuntimeError("handler boom")),
    )

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=_auto_approve(store, policy)
    ):
        # handle_function_call catches handler exceptions and returns an error string.
        result = model_tools.handle_function_call(
            "write_file",
            _args(),
            task_id="t5",
            session_id="s5",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )
        assert not protected.has_current_dispatch_attestation("write_file", _args()), (
            "attestation must be cleared even after a handler exception"
        )


# ---------------------------------------------------------------------------
# consume_current_dispatch_attestation — atomic check-and-clear
# ---------------------------------------------------------------------------

def test_consume_returns_true_first_call_then_false(monkeypatch):
    """consume_current_dispatch_attestation is one-use: second call returns False."""
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()
    results = []

    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )

    def consuming_dispatch(name, args, **kwargs):
        # First consume must return True and clear the token.
        results.append(protected.consume_current_dispatch_attestation(name, args))
        # Second consume on the same invocation must return False.
        results.append(protected.consume_current_dispatch_attestation(name, args))
        return json.dumps({"ok": True})

    monkeypatch.setattr(model_tools.registry, "dispatch", consuming_dispatch)

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=_auto_approve(store, policy)
    ):
        model_tools.handle_function_call(
            "write_file",
            _args(),
            task_id="t6",
            session_id="s6",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )

    assert results == [True, False], (
        f"first consume must be True and second False, got {results}"
    )


def test_consume_mismatched_args_returns_false(monkeypatch):
    """consume_current_dispatch_attestation returns False for wrong args without consuming."""
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()
    results = []

    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )

    def checking_dispatch(name, args, **kwargs):
        wrong_args = {"path": "/other/path.json", "content": "not-approved"}
        # Mismatched args must return False but not consume the token.
        results.append(protected.consume_current_dispatch_attestation(name, wrong_args))
        # Correct args must still succeed (token was not consumed).
        results.append(protected.consume_current_dispatch_attestation(name, args))
        return json.dumps({"ok": True})

    monkeypatch.setattr(model_tools.registry, "dispatch", checking_dispatch)

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=_auto_approve(store, policy)
    ):
        model_tools.handle_function_call(
            "write_file",
            _args(),
            task_id="t7",
            session_id="s7",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
        )

    assert results == [False, True], (
        f"mismatched consume False then correct consume True, got {results}"
    )


def _connector_tool():
    return "connectors__test__write_action"


def _connector_policy():
    return protected.ProtectedApiRunApprovalPolicy(
        allowed_tool_names=(_connector_tool(),),
        run_id="conn-run",
        approval_session="conn-session",
        user_session_id="conn-user",
        hermes_session_id="conn-hermes",
        conversation_id="conn-convo",
        expires_at=time.time() + 30,
    )


def _conn_approve(store, policy):
    def callback(event):
        store.submit_approval(
            run_id=event["run_id"],
            approval_session=policy.approval_session,
            body={
                "request_id": event["request_id"],
                "action_digest": event["action_digest"],
                "choice": "once",
            },
        )
    return callback


def _setup_connector_mocks(monkeypatch):
    """Enable connector routing so handle_function_call reaches the connector branch."""
    from tools.tool_gateway import config as gateway_config
    from tools.registry import invalidate_check_fn_cache
    monkeypatch.setattr(gateway_config, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *a, **kw: None,
    )


def test_attestation_cleared_after_connector_dispatch(monkeypatch):
    """Attestation is cleared even when the connector branch handles the call."""
    _setup_connector_mocks(monkeypatch)
    store = protected.ProtectedApiRunApprovalStore()
    policy = _connector_policy()
    conn_args = {"path": "/calendar/event.json", "content": "conn-ok"}

    monkeypatch.setattr(
        "model_tools_connectors.dispatch_connector_call",
        lambda name, args, tool_call_id: json.dumps({"connector": True}),
    )

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=_conn_approve(store, policy)
    ):
        model_tools.handle_function_call(
            _connector_tool(),
            conn_args,
            task_id="t8",
            session_id="s8",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
            enabled_toolsets=["connections"],
        )
        assert not protected.has_current_dispatch_attestation(_connector_tool(), conn_args), (
            "attestation must be cleared after connector dispatch"
        )


def test_attestation_from_inline_dispatch_not_consumable_by_later_direct_handler(monkeypatch):
    """An inline executor's one-use attestation must not leak past its own dispatch.

    invoke_tool's inline-executor branch (agent_runtime_helpers.py) verifies dispatch
    and mints an attestation the same way model_tools._dispatch does, but has no
    matching finally-clear. A later direct handler call for the exact same tool+args
    must not be able to claim that leaked token.
    """
    from types import SimpleNamespace

    from agent import agent_runtime_helpers
    from tools.todo_tool import TodoStore

    agent = SimpleNamespace(session_id="inline-leak-session", _todo_store=TodoStore())
    todo_args = {"todos": [{"id": "1", "content": "x", "status": "pending"}]}

    tokens = protected.set_current_api_run_context(
        run_id="inline-leak-run", approval_session="inline-leak-approval-session"
    )
    try:
        result = agent_runtime_helpers.invoke_tool(
            agent,
            "todo_list",
            todo_args,
            "inline-leak-task",
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )
        json.loads(result)  # sanity: the inline executor actually ran
        assert protected.consume_current_dispatch_attestation("todo_list", todo_args) is False, (
            "a later direct handler must not be able to consume the inline dispatch's token"
        )
    finally:
        protected.reset_current_api_run_context(tokens)


def test_run_context_reset_clears_minted_attestation():
    """Resetting an active API run must invalidate any attestation minted under it.

    A minted attestation is only proof for the exact call that crossed dispatch; once
    the run context is reset (run retired / next run begins), a stale attestation for
    the same tool+args must not still be claimable.
    """
    tokens = protected.set_current_api_run_context(
        run_id="reset-run", approval_session="reset-session"
    )
    try:
        assert protected.verify_protected_dispatch("write_file", _args()) is True
        assert protected.has_current_dispatch_attestation("write_file", _args())
    finally:
        protected.reset_current_api_run_context(tokens)

    assert protected.has_current_dispatch_attestation("write_file", _args()) is False
    assert protected.consume_current_dispatch_attestation("write_file", _args()) is False


def test_attestation_cleared_after_connector_exception(monkeypatch):
    """Attestation is cleared even when the connector raises."""
    _setup_connector_mocks(monkeypatch)
    store = protected.ProtectedApiRunApprovalStore()
    policy = protected.ProtectedApiRunApprovalPolicy(
        allowed_tool_names=(_connector_tool(),),
        run_id="conn-exc-run",
        approval_session="conn-exc-session",
        user_session_id="conn-exc-user",
        hermes_session_id="conn-exc-hermes",
        conversation_id="conn-exc-convo",
        expires_at=time.time() + 30,
    )
    conn_args = {"path": "/calendar/event.json", "content": "conn-exc-ok"}

    monkeypatch.setattr(
        "model_tools_connectors.dispatch_connector_call",
        lambda name, args, tool_call_id: (_ for _ in ()).throw(RuntimeError("connector boom")),
    )

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=_conn_approve(store, policy)
    ):
        model_tools.handle_function_call(
            _connector_tool(),
            conn_args,
            task_id="t9",
            session_id="s9",
            skip_pre_tool_call_hook=True,
            skip_tool_execution_middleware=True,
            enabled_toolsets=["connections"],
        )
        assert not protected.has_current_dispatch_attestation(_connector_tool(), conn_args), (
            "attestation must be cleared even after connector exception"
        )
