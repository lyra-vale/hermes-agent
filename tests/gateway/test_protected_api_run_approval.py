"""API-server route contract for protected run approvals."""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_module
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms import api_server_runs
from tools import protected_api_approval as protected


@pytest.fixture(autouse=True)
def discover_tools():
    import model_tools

    model_tools.discover_builtin_tools()


def _policy(run_id):
    return protected.ProtectedApiRunApprovalPolicy(
        allowed_tool_names=("write_file",),
        run_id=run_id,
        approval_session=run_id,
        user_session_id="api-user",
        hermes_session_id="api-hermes",
        conversation_id="api-conversation",
        expires_at=time.time() + 30,
    )


def _request(run_id, body):
    request = MagicMock()
    request.headers = {}
    request.match_info = {"run_id": run_id}
    request.path = f"/v1/runs/{run_id}/approval"
    request.method = "POST"
    request.json = AsyncMock(return_value=body)
    return request


def _copy_context_target(callback):
    context = contextvars.copy_context()
    return lambda: context.run(callback)


@pytest.mark.asyncio
async def test_protected_run_detection_failure_does_not_fall_through_to_generic_approval(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_protected_detection_failure"
    request = _request(run_id, {"choice": "session"})
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(request)
    adapter._run_statuses[run_id] = {
        "object": "hermes.run", "run_id": run_id, "status": "waiting_for_approval"
    }
    adapter._run_approval_sessions[run_id] = run_id
    generic_calls = []

    def broken_detection(*_args, **_kwargs):
        raise RuntimeError("protected ledger unavailable")

    monkeypatch.setattr(protected, "is_protected_api_run", broken_detection)
    monkeypatch.setattr(
        "tools.approval.resolve_gateway_approval",
        lambda *args, **kwargs: generic_calls.append((args, kwargs)) or 1,
    )

    response = await api_server_runs._handle_run_approval(
        adapter, request, _api_server=api_server_module
    )

    assert response.status == 500
    assert json.loads(response.text)["error"]["code"] == "protected_approval_detection_failed"
    assert generic_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mapping", [None, "stale-approval-session"])
async def test_protected_approval_uses_authoritative_run_id_and_rejects_stale_mapping(monkeypatch, mapping):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = f"run_authoritative_lookup_{mapping or 'missing'}"
    request = _request(run_id, {})
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(request)
    adapter._run_statuses[run_id] = {
        "object": "hermes.run", "run_id": run_id, "status": "waiting_for_approval"
    }
    if mapping is not None:
        adapter._run_approval_sessions[run_id] = mapping
    generic_calls = []
    monkeypatch.setattr(
        "tools.approval.resolve_gateway_approval",
        lambda *args, **kwargs: generic_calls.append((args, kwargs)) or 1,
    )

    policy = _policy(run_id)
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    result = {}
    with protected.bind_protected_api_run(policy, store=store, pending_callback=events.append):
        worker = threading.Thread(
            target=_copy_context_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval(
                        "write_file", {"path": "/calendar/authoritative.json", "content": "secret"}
                    )
                )
            )
        )
        worker.start()
        assert await _wait_for(lambda: bool(events))
        response = await api_server_runs._handle_run_approval(
            adapter,
            _request(
                run_id,
                {
                    "request_id": events[0]["request_id"],
                    "action_digest": events[0]["action_digest"],
                    "choice": "once",
                },
            ),
            _api_server=api_server_module,
        )
        if mapping is None:
            assert response.status == 200
            worker.join(timeout=2)
            assert result["approved"] is True
        else:
            assert response.status == 409
            assert json.loads(response.text)["error"]["code"] == "approval_session_mismatch"
            assert worker.is_alive()
            store.retire_run(run_id)
            worker.join(timeout=2)
        assert generic_calls == []


@pytest.mark.asyncio
async def test_unregister_retires_protected_waiter_without_waiting_for_expiry():
    store = protected._DEFAULT_STORE
    policy = _policy("run_unregister_retires_waiter")
    events = []
    result = {}

    with protected.bind_protected_api_run(
        policy, store=store, pending_callback=events.append
    ):
        worker = threading.Thread(
            target=_copy_context_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval(
                        "write_file", {"path": "/calendar/cancel.json", "content": "secret"}
                    )
                )
            )
        )
        worker.start()
        assert await _wait_for(lambda: bool(events))

        api_server_runs._unregister_approval_notify(policy.run_id)
        worker.join(timeout=0.5)

        assert not worker.is_alive()
        assert result["approved"] is False
        assert "retired" in result["message"].lower()


@pytest.mark.asyncio
async def test_stop_retires_protected_waiter_before_executor_finishes(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_stop_retires_waiter"
    request = _request(run_id, {})
    request.path = f"/v1/runs/{run_id}/stop"
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(request)
    adapter._run_statuses[run_id] = {
        "object": "hermes.run", "run_id": run_id, "status": "running"
    }
    adapter._run_approval_sessions[run_id] = "corrupt-session-mapping"
    adapter._active_run_agents[run_id] = object()
    adapter._active_run_tasks[run_id] = object()
    monkeypatch.setattr(api_server_module, "request_hard_interrupt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_server_module, "_reap_disconnected_agent_processes", lambda *_args, **_kwargs: None)

    policy = _policy(run_id)
    result = {}
    with protected.bind_protected_api_run(policy, pending_callback=lambda _event: None):
        worker = threading.Thread(
            target=_copy_context_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval(
                        "write_file", {"path": "/calendar/stop.json", "content": "secret"}
                    )
                )
            )
        )
        worker.start()
        await asyncio.sleep(0.05)
        response = await api_server_runs._handle_stop_run(
            adapter, request, _api_server=api_server_module
        )
        assert response.status == 200
        worker.join(timeout=0.5)
        assert not worker.is_alive()
        assert result["approved"] is False
        assert "retired" in result["message"].lower()


@pytest.mark.asyncio
async def test_protected_tool_progress_preview_failure_fails_closed(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_protected_preview_failure"
    adapter._run_streams[run_id] = asyncio.Queue()
    adapter._run_statuses[run_id] = {"object": "hermes.run", "run_id": run_id, "status": "running"}
    callback = api_server_runs._make_run_event_callback(
        adapter,
        run_id,
        asyncio.get_running_loop(),
        _api_server=api_server_module,
    )

    def broken_binding_lookup():
        raise RuntimeError("protected context lookup failed")

    monkeypatch.setattr(protected, "current_protected_api_run_binding", broken_binding_lookup)
    with protected.bind_protected_api_run(_policy(run_id)):
        callback("tool.started", "write_file", "/calendar/event.json content=secret")
    await asyncio.sleep(0)
    event = adapter._run_streams[run_id].get_nowait()
    assert event["preview"] is None


@pytest.mark.asyncio
async def test_protected_tool_progress_hides_preview_without_context():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_protected_preview_without_context"
    adapter._run_streams[run_id] = asyncio.Queue()
    adapter._run_statuses[run_id] = {"object": "hermes.run", "run_id": run_id, "status": "running"}
    callback = api_server_runs._make_run_event_callback(
        adapter,
        run_id,
        asyncio.get_running_loop(),
        _api_server=api_server_module,
    )
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy(run_id)
    store.register_policy(policy)

    try:
        callback("tool.started", "write_file", "/calendar/event.json content=secret")
        await asyncio.sleep(0)
        event = adapter._run_streams[run_id].get_nowait()
        assert event["preview"] is None
    finally:
        store.retire_run(policy.run_id, policy.approval_session)


@pytest.mark.asyncio
async def test_protected_subagent_diagnostics_hide_preview_and_text():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_protected_subagent_diagnostics"
    adapter._run_streams[run_id] = asyncio.Queue()
    adapter._run_statuses[run_id] = {"object": "hermes.run", "run_id": run_id, "status": "running"}
    callback = api_server_runs._make_run_event_callback(
        adapter,
        run_id,
        asyncio.get_running_loop(),
        _api_server=api_server_module,
    )
    store = protected.ProtectedApiRunApprovalStore()
    policy = _policy(run_id)
    store.register_policy(policy)
    try:
        callback(
            "subagent.start",
            preview="/calendar/event.json approved",
            goal="write /calendar/event.json approved",
            summary="secret summary",
            task_count=1,
        )
        await asyncio.sleep(0)
        event = adapter._run_streams[run_id].get_nowait()
        assert "/calendar/event.json" not in repr(event)
        assert "approved" not in repr(event)
    finally:
        store.retire_run(policy.run_id, policy.approval_session)


@pytest.mark.asyncio
async def test_protected_status_ignores_callback_fields_and_uses_policy():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_protected_status_policy_source"
    queue = asyncio.Queue()
    adapter._run_streams[run_id] = queue
    adapter._run_statuses[run_id] = {"object": "hermes.run", "run_id": run_id, "status": "running"}
    run = SimpleNamespace(run_id=run_id, queue=queue, approval_session_key=run_id)
    notify = api_server_runs._make_approval_notify(
        adapter, run, _api_server=api_server_module
    )
    policy = _policy(run_id)
    policy = protected.ProtectedApiRunApprovalPolicy(
        allowed_tool_names=policy.allowed_tool_names,
        run_id=policy.run_id,
        approval_session=policy.approval_session,
        user_session_id=policy.user_session_id,
        hermes_session_id=policy.hermes_session_id,
        conversation_id=policy.conversation_id,
        expires_at=policy.expires_at,
        redacted_description="Policy-owned description",
    )
    store = protected.ProtectedApiRunApprovalStore()
    events = []
    result = {}

    with protected.bind_protected_api_run(policy, store=store, pending_callback=notify):
        worker = threading.Thread(
            target=_copy_context_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval(
                        "write_file", {"path": "/calendar/status.json", "content": "secret"}
                    )
                )
            )
        )
        worker.start()
        await _wait_for(lambda: adapter._run_statuses[run_id].get("approval") is not None)
        current = adapter._run_statuses[run_id]["approval"]
        events.append(dict(current))
        notify(
            {
                "approval_type": "protected_api_run",
                "request_id": current["request_id"],
                "tool_name": "write_file",
                "action_digest": current["action_digest"],
                "expires_at": current["expires_at"],
                "redacted_description": "/calendar/status.json raw callback text",
                "args": {"content": "secret"},
            }
        )
        await asyncio.sleep(0)
        approval = adapter._run_statuses[run_id]["approval"]
        assert approval["redacted_description"] == "Policy-owned description"
        assert "/calendar/status.json" not in repr(approval)
        assert "secret" not in repr(approval)
        store.retire_run(policy.run_id, policy.approval_session)
        worker.join(timeout=2)

    assert result["approved"] is False


@pytest.mark.asyncio
async def test_protected_api_route_accepts_only_exact_once_or_deny_body():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_protected_route"
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(_request(run_id, {}))
    adapter._run_statuses[run_id] = {
        "object": "hermes.run", "run_id": run_id, "status": "waiting_for_approval"
    }
    adapter._run_approval_sessions[run_id] = run_id
    events = []
    result = {}
    action = {"path": "/calendar/event.json", "content": "secret calendar payload"}

    policy = _policy(run_id)
    with protected.bind_protected_api_run(policy, pending_callback=events.append):
        worker = threading.Thread(
            target=_copy_context_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval("write_file", action)
                )
            )
        )
        worker.start()
        assert await _wait_for(lambda: bool(events))
        event = events[0]
        event_text = repr(event)
        assert "/calendar/event.json" not in event_text
        assert "secret calendar payload" not in event_text

        bad = await api_server_runs._handle_run_approval(
            adapter,
            _request(
                run_id,
                {
                    "request_id": event["request_id"],
                    "action_digest": event["action_digest"],
                    "choice": "approve",
                },
            ),
            _api_server=api_server_module,
        )
        assert bad.status == 400
        assert json.loads(bad.text)["error"]["code"] == "invalid_approval_choice"
        assert worker.is_alive()

        response = await api_server_runs._handle_run_approval(
            adapter,
            _request(
                run_id,
                {
                    "request_id": event["request_id"],
                    "action_digest": event["action_digest"],
                    "choice": "once",
                },
            ),
            _api_server=api_server_module,
        )
        assert response.status == 200
        response_body = json.loads(response.text)
        assert response_body["choice"] == "once"
        assert response_body["action_digest"] == event["action_digest"]
        worker.join(timeout=2)

    assert result["approved"] is True


@pytest.mark.asyncio
async def test_protected_api_route_rejects_fifo_resolve_all_and_cross_run():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_protected_route_2"
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(_request(run_id, {}))
    adapter._run_statuses[run_id] = {
        "object": "hermes.run", "run_id": run_id, "status": "waiting_for_approval"
    }
    adapter._run_approval_sessions[run_id] = run_id
    events = []
    result = {}
    policy = _policy(run_id)
    with protected.bind_protected_api_run(policy, pending_callback=events.append):
        worker = threading.Thread(
            target=_copy_context_target(
                lambda: result.update(
                    protected.require_protected_api_run_approval(
                        "write_file", {"path": "/calendar/two.json", "content": "x"}
                    )
                )
            )
        )
        worker.start()
        assert await _wait_for(lambda: bool(events))
        event = events[0]
        body = {
            "request_id": event["request_id"],
            "action_digest": event["action_digest"],
            "choice": "once",
            "resolve_all": True,
        }
        response = await api_server_runs._handle_run_approval(
            adapter, _request(run_id, body), _api_server=api_server_module
        )
        assert response.status == 400
        assert json.loads(response.text)["error"]["code"] == "invalid_approval_body"

        # A different run path cannot resolve the exact record even when the
        # caller has the same API-key scope.
        other_run = "run_other"
        adapter._run_owners[other_run] = adapter._run_owners[run_id]
        adapter._run_statuses[other_run] = {
            "object": "hermes.run", "run_id": other_run, "status": "waiting_for_approval"
        }
        adapter._run_approval_sessions[other_run] = other_run
        response = await api_server_runs._handle_run_approval(
            adapter,
            _request(
                other_run,
                {
                    "request_id": event["request_id"],
                    "action_digest": event["action_digest"],
                    "choice": "once",
                },
            ),
            _api_server=api_server_module,
        )
        assert response.status == 409
        assert json.loads(response.text)["error"]["code"] == "approval_not_pending"
        assert worker.is_alive()

        response = await api_server_runs._handle_run_approval(
            adapter,
            _request(
                run_id,
                {
                    "request_id": event["request_id"],
                    "action_digest": event["action_digest"],
                    "choice": "deny",
                },
            ),
            _api_server=api_server_module,
        )
        assert response.status == 200
        worker.join(timeout=2)

    assert result["approved"] is False


@pytest.mark.asyncio
async def test_protected_tool_progress_preview_is_not_published():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_protected_preview"
    adapter._run_streams[run_id] = asyncio.Queue()
    adapter._run_statuses[run_id] = {"object": "hermes.run", "run_id": run_id, "status": "running"}
    callback = api_server_runs._make_run_event_callback(
        adapter,
        run_id,
        asyncio.get_running_loop(),
        _api_server=api_server_module,
    )
    with protected.bind_protected_api_run(_policy(run_id)):
        callback("tool.started", "write_file", "/calendar/event.json content=secret")
    await asyncio.sleep(0)
    event = adapter._run_streams[run_id].get_nowait()
    assert event["preview"] is None


@pytest.mark.asyncio
async def test_real_api_run_lifecycle_admission_executor_status_and_stop(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_started = threading.Event()

    class ApprovalAgent:
        session_id = "lifecycle-session"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        _hermes_api_runtime = None

        def run_conversation(self, **_kwargs):
            policy = protected.ProtectedApiRunApprovalPolicy(
                allowed_tool_names=("write_file",),
                user_session_id="lifecycle-user",
                hermes_session_id="lifecycle-hermes",
                conversation_id="lifecycle-conversation",
                expires_at=time.time() + 30,
            )
            protected.attach_protected_api_run_policy(policy)
            run_started.set()
            decision = protected.require_protected_api_run_approval(
                "write_file", {"path": "/calendar/lifecycle.json", "content": "secret"}
            )
            return {"interrupted": not decision.get("approved"), "final_response": "done"}

    monkeypatch.setattr(api_server_module, "request_hard_interrupt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_server_module, "_reap_disconnected_agent_processes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: ApprovalAgent())
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"platform_toolsets": {"api_server": ["file"]}},
    )

    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)

    from aiohttp.test_utils import TestClient, TestServer

    async with TestClient(TestServer(app)) as client:
        admitted = await client.post("/v1/runs", json={"input": "create a calendar event"})
        assert admitted.status == 202
        run_id = (await admitted.json())["run_id"]
        assert run_started.wait(timeout=2)

        for _ in range(50):
            status_response = await client.get(f"/v1/runs/{run_id}")
            status_payload = await status_response.json()
            if status_payload.get("status") == "waiting_for_approval":
                break
            await asyncio.sleep(0.01)
        assert status_payload["status"] == "waiting_for_approval"
        assert status_payload["approval"]["approval_type"] == "protected_api_run"

        stopped = await client.post(f"/v1/runs/{run_id}/stop", json={})
        assert stopped.status == 200

        for _ in range(100):
            status_response = await client.get(f"/v1/runs/{run_id}")
            status_payload = await status_response.json()
            if status_payload.get("status") == "cancelled":
                break
            await asyncio.sleep(0.01)
        assert status_payload["status"] == "cancelled"
        assert run_id not in adapter._active_run_agents

        late_context = protected.set_current_api_run_context(
            run_id=run_id,
            approval_session=run_id,
        )
        try:
            with pytest.raises(protected.ProtectedApprovalError) as exc:
                protected.attach_protected_api_run_policy(_policy(run_id))
            assert exc.value.code == "approval_run_retired"
        finally:
            protected.reset_current_api_run_context(late_context)


async def _wait_for(predicate, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())
