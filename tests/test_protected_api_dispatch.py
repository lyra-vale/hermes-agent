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


def test_hook_failure_blocks_before_generic_dispatch(monkeypatch):
    def broken_hook(*_args, **_kwargs):
        raise RuntimeError("calendar hook failed")

    result, dispatches, events = _run_handle(monkeypatch, _args(), skip_pre=False, hook=broken_hook)

    assert "blocked" in result.lower() or "hook" in result.lower()
    assert dispatches == []
    assert events == []


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


def test_protected_observer_failure_still_hides_arguments(monkeypatch):
    observed = []
    monkeypatch.setattr(
        protected,
        "protected_observer_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer sanitizer failed")),
    )
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


def test_agent_runtime_observer_failure_still_hides_arguments(monkeypatch):
    from agent import tool_executor

    observed = []
    monkeypatch.setattr(
        protected,
        "protected_observer_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer sanitizer failed")),
    )
    monkeypatch.setattr(
        tool_executor,
        "_emit_terminal_post_tool_call",
        lambda *args, **kwargs: observed.append(kwargs),
    )
    policy = _policy()

    with protected.bind_protected_api_run(policy):
        ref = tool_executor._ToolCallRef(
            "write_file", _args(), "dispatch-task", "call-1", []
        )
        ref.emit_post(object(), "{}", status="blocked")

    assert observed
    assert observed[0]["function_args"] == {}


def test_protected_handler_exception_diagnostics_never_expose_action_args(monkeypatch, caplog):
    observed = []
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()
    raw = "/calendar/handler-secret.json raw-handler-secret"

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

    entry = model_tools.registry.get_entry("write_file")
    original_handler = entry.handler
    entry.handler = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(raw))
    monkeypatch.setattr(model_tools, "_emit_post_tool_call_hook", lambda **kwargs: observed.append(kwargs))
    try:
        with protected.bind_protected_api_run(policy, store=store, pending_callback=approve):
            with caplog.at_level("ERROR"):
                result = model_tools.handle_function_call(
                    "write_file", _args(), task_id="dispatch-task", skip_pre_tool_call_hook=True,
                    skip_tool_execution_middleware=True,
                )
    finally:
        entry.handler = original_handler

    assert raw not in result
    assert raw not in caplog.text
    assert observed
    assert all(raw not in repr(event) for event in observed)


def test_invoke_tool_observer_failure_still_hides_arguments(monkeypatch):
    from types import SimpleNamespace
    from agent import agent_runtime_helpers, inline_tool_executors

    observed = []
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()

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

    monkeypatch.setattr(
        protected,
        "protected_observer_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer sanitizer failed")),
    )
    monkeypatch.setattr(
        inline_tool_executors,
        "resolve_invoke_tool_executor",
        lambda *_args, **_kwargs: lambda *_executor_args, **_executor_kwargs: "{\"ok\": true}",
    )
    monkeypatch.setattr(
        inline_tool_executors,
        "emit_terminal_post_tool_call",
        lambda *args, **kwargs: observed.append(kwargs),
    )

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=approve
    ):
        result = agent_runtime_helpers.invoke_tool(
            SimpleNamespace(session_id="dispatch-session"),
            "write_file",
            _args(),
            "dispatch-task",
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    assert json.loads(result)["ok"] is True
    assert observed
    assert observed[0]["function_args"] == {}


def test_agent_runtime_diagnostic_callbacks_fail_closed_on_sanitizer_error(monkeypatch, capsys):
    from types import SimpleNamespace
    from agent import tool_executor

    starts = []
    completes = []
    monkeypatch.setattr(
        protected,
        "protected_observer_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer sanitizer failed")),
    )
    agent = SimpleNamespace(
        tool_progress_callback=None,
        tool_start_callback=lambda *args: starts.append(args),
        tool_complete_callback=lambda *args: completes.append(args),
        quiet_mode=False,
        tool_progress_mode="all",
        verbose_logging=False,
        log_prefix_chars=100,
        _current_tool=None,
        _touch_activity=lambda *_args: None,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
    )
    ref = tool_executor._ToolCallRef("write_file", _args(), "dispatch-task", "call-1", [])

    with protected.bind_protected_api_run(_policy()):
        tool_executor._begin_tool_execution(agent, ref, None)
        tool_executor._emit_tool_complete_and_risk(agent, ref, "{}", None, False)

    assert starts
    assert starts[0][2] == {}
    assert completes
    assert completes[0][2] == {}
    output = capsys.readouterr().out
    assert "/calendar/event.json" not in output
    assert "approved" not in output


def test_quiet_spinner_label_fails_closed_for_protected_args(monkeypatch):
    from types import SimpleNamespace
    from agent import tool_executor

    stopped = []

    class FakeSpinner:
        def __init__(self, label, **_kwargs):
            self.label = label

        @staticmethod
        def get_waiting_faces():
            return [":)"]

        def start(self):
            return None

        def stop(self, message):
            stopped.append(message)

    monkeypatch.setattr(tool_executor, "KawaiiSpinner", FakeSpinner)
    monkeypatch.setattr(
        protected,
        "protected_observer_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer sanitizer failed")),
    )
    agent = SimpleNamespace(
        _should_emit_quiet_tool_messages=lambda: True,
        _should_start_quiet_spinner=lambda: True,
        _print_fn=lambda *_args: None,
    )

    with protected.bind_protected_api_run(_policy()):
        spinner = tool_executor._start_quiet_tool_spinner(agent, "write_file", _args())
        tool_executor._finish_quiet_tool_spinner(agent, spinner, "write_file", _args(), 0.1, "{}")

    assert "/calendar/event.json" not in spinner.label
    assert "approved" not in spinner.label
    assert stopped
    assert "/calendar/event.json" not in stopped[0]
    assert "approved" not in stopped[0]


def test_delegate_spinner_label_fails_closed_for_protected_args():
    from agent import tool_executor

    with protected.bind_protected_api_run(_policy()):
        label = tool_executor._delegate_spinner_label({"goal": "/calendar/event.json approved"})

    assert "/calendar/event.json" not in label
    assert "approved" not in label


def test_concurrent_completion_diagnostic_fails_closed_for_protected_args(monkeypatch):
    from types import SimpleNamespace
    from agent import tool_executor

    printed = []
    monkeypatch.setattr(
        protected,
        "protected_observer_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer sanitizer failed")),
    )
    monkeypatch.setattr(
        tool_executor,
        "_commit_tool_result",
        lambda *_args, **_kwargs: (None, "{}", None),
    )
    monkeypatch.setattr(tool_executor, "_emit_tool_complete_and_risk", lambda *_args, **_kwargs: None)
    ref = tool_executor._ToolCallRef("write_file", _args(), "task", "call", [])
    batch = SimpleNamespace(
        parsed_calls=[SimpleNamespace(parse_error=None)],
        results=[SimpleNamespace(ref=ref, result="{}", duration=0.1, is_error=False, blocked=False)],
        timed_out_indices=set(),
    )
    agent = SimpleNamespace(
        _should_emit_quiet_tool_messages=lambda: True,
        _safe_print=lambda message: printed.append(message),
    )

    with protected.bind_protected_api_run(_policy()):
        assert tool_executor._append_batch_results(agent, [], "task", batch, object()) is True

    assert printed
    assert "/calendar/event.json" not in printed[0]
    assert "approved" not in printed[0]


def test_transform_observer_boundary_fails_closed_on_sanitizer_error(monkeypatch):
    observed = []
    monkeypatch.setattr(
        protected,
        "protected_observer_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer sanitizer failed")),
    )
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda _name, **kwargs: (observed.append(kwargs), [])[1],
    )

    with protected.bind_protected_api_run(_policy()):
        model_tools._apply_transform_tool_result_hook(
            "write_file",
            _args(),
            "{}",
            1,
            model_tools._CallIds("task", "session", "call", "turn", "request"),
        )

    assert observed
    assert observed[0]["args"] == {}


def test_post_tool_observer_boundary_fails_closed_on_sanitizer_error(monkeypatch):
    observed = []
    monkeypatch.setattr(
        protected,
        "protected_observer_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("observer sanitizer failed")),
    )
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda _name, **kwargs: observed.append(kwargs),
    )

    with protected.bind_protected_api_run(_policy()):
        model_tools._emit_post_tool_call_hook(
            function_name="write_file",
            function_args=_args(),
            result="{}",
        )

    assert observed
    assert observed[0]["args"] == {}


def test_protected_verifier_requires_authorization_for_every_action():
    policy = _policy()
    with protected.bind_protected_api_run(policy):
        assert protected.verify_protected_dispatch(
            "terminal", {"command": "cat /calendar/event.json"}
        ) is False


def test_retired_policy_invalidates_claimed_dispatch_authorization():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    with protected.bind_protected_api_run(
        policy,
        store=store,
        pending_callback=_approve_callback(store, events, policy.approval_session),
    ):
        assert protected.require_protected_api_run_approval("write_file", _args())["approved"] is True
        store.retire_run(policy.run_id, policy.approval_session)
        assert protected.verify_protected_dispatch("write_file", _args()) is False


def test_allowlist_mismatch_invalidates_pending_authorization():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    with protected.bind_protected_api_run(
        policy,
        store=store,
        pending_callback=_approve_callback(store, events, policy.approval_session),
    ):
        assert protected.require_protected_api_run_approval("write_file", _args())["approved"] is True
        blocked = protected.require_protected_api_run_approval("terminal", {"command": "calendar"})
        assert blocked["approved"] is False
        assert protected.verify_protected_dispatch("write_file", _args()) is False


def test_final_dispatch_authorization_is_single_use_and_exact(monkeypatch):
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    with protected.bind_protected_api_run(policy, store=store, pending_callback=_approve_callback(store, events, policy.approval_session)):
        decision = protected.require_protected_api_run_approval("write_file", _args())
        assert decision["approved"] is True
        assert protected.verify_protected_dispatch("write_file", _args()) is True
        assert protected.verify_protected_dispatch("write_file", _args()) is False


def test_final_dispatch_verifier_failure_cannot_fall_through_to_registry(monkeypatch):
    dispatches = []
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda name, dispatched_args, **kwargs: dispatches.append((name, dispatched_args)) or '{"dispatched": true}',
    )
    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        protected,
        "verify_protected_dispatch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("verifier unavailable")),
    )
    monkeypatch.setattr(
        protected,
        "current_protected_api_run_binding",
        lambda: (_ for _ in ()).throw(RuntimeError("protected detection unavailable")),
    )

    result = model_tools.handle_function_call(
        "write_file",
        _args(),
        task_id="dispatch-task",
        skip_pre_tool_call_hook=True,
        skip_tool_execution_middleware=True,
    )

    assert "protected API approval verification failed" in result
    assert dispatches == []


def test_agent_final_dispatch_verifier_failure_cannot_execute(monkeypatch):
    from types import SimpleNamespace
    from agent import tool_executor

    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy()
    events = []
    executions = []
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        protected,
        "verify_protected_dispatch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("verifier unavailable")),
    )
    monkeypatch.setattr(
        protected,
        "has_current_protected_dispatch_authorization",
        lambda: (_ for _ in ()).throw(RuntimeError("protected authorization unavailable")),
    )
    agent = SimpleNamespace(
        _tool_guardrails=SimpleNamespace(
            before_call=lambda *_args, **_kwargs: SimpleNamespace(allows_execution=True),
        ),
    )
    state = tool_executor._ManagedToolResult(
        result=None,
        args=_args(),
        middleware_trace=[],
        blocked=False,
        dispatched=False,
    )
    ref = tool_executor._ToolCallRef("write_file", _args(), "dispatch-task", "call-1", [])

    with protected.bind_protected_api_run(
        policy,
        store=store,
        pending_callback=_approve_callback(store, events, policy.approval_session),
    ):
        result = tool_executor._dispatch_authorized_once(
            agent,
            state,
            ref,
            execute=lambda args: executions.append(args) or '{"dispatched": true}',
            scope_block=None,
            display_index=None,
            begin_execution=lambda _callback=None: None,
            authorization_gate=None,
        )

    assert "protected API approval verification failed" in result
    assert state.blocked is True
    assert executions == []


def test_changed_args_cannot_be_repaired_after_approval():
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    policy = _policy()
    with protected.bind_protected_api_run(policy, store=store, pending_callback=_approve_callback(store, events, policy.approval_session)):
        assert protected.require_protected_api_run_approval("write_file", _args())["approved"] is True
        assert protected.verify_protected_dispatch("write_file", _args("changed")) is False
        assert protected.verify_protected_dispatch("write_file", _args()) is False
