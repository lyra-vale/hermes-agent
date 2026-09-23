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


async def _wait_for(predicate, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())
