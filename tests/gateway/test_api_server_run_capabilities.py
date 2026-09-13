"""Behavioral contract for per-run API toolset narrowing and session locks."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_runs import _validate_run_toolsets


_CONFIG = {"platform_toolsets": {"api_server": ["file", "web"]}}


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


class TestRunToolsetRequest:
    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ({"input": "hello"}, None),
            ({"input": "hello", "toolsets": ["web"]}, ["web"]),
            ({"input": "hello", "toolsets": []}, []),
        ],
    )
    def test_request_toolsets_have_explicit_subset_semantics(self, body, expected):
        assert _validate_run_toolsets(body, {"file", "web"}) == expected

    @pytest.mark.parametrize(
        "toolsets",
        [
            "web",
            ["web", "web"],
            ["terminal"],
            ["unknown"],
            [1],
            [""],
            [" web"],
        ],
    )
    def test_invalid_toolset_requests_fail_closed(self, toolsets):
        with pytest.raises(ValueError):
            _validate_run_toolsets({"toolsets": toolsets}, {"file", "web"})

    @pytest.mark.asyncio
    async def test_subset_is_passed_to_run_agent_and_omitted_preserves_call_shape(self):
        adapter = _adapter()
        with patch("gateway.run._load_gateway_config", return_value=_CONFIG):
            async with TestClient(TestServer(_app(adapter))) as cli:
                for body, expected in (
                    ({"input": "subset", "toolsets": ["web"]}, ["web"]),
                    ({"input": "omitted"}, None),
                    ({"input": "empty", "toolsets": []}, []),
                ):
                    agent = _agent()
                    with patch.object(adapter, "_create_agent", return_value=agent) as create:
                        response = await cli.post("/v1/runs", json=body)
                        assert response.status == 202
                        run_id = (await response.json())["run_id"]
                        for _ in range(40):
                            if adapter._run_statuses.get(run_id, {}).get("status") == "completed":
                                break
                            await asyncio.sleep(0.01)
                        kwargs = create.call_args.kwargs
                        if expected is None:
                            assert "requested_toolsets" not in kwargs
                        else:
                            assert kwargs["requested_toolsets"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "toolsets",
        ["web", ["web", "web"], ["terminal"], ["unknown"], [1], [""], [" web"]],
    )
    async def test_http_rejects_invalid_toolset_requests_without_admitting_run(self, toolsets):
        adapter = _adapter()
        with patch("gateway.run._load_gateway_config", return_value=_CONFIG):
            async with TestClient(TestServer(_app(adapter))) as cli:
                with patch.object(adapter, "_create_agent") as create:
                    response = await cli.post("/v1/runs", json={"input": "hello", "toolsets": toolsets})
                assert response.status == 400
                payload = await response.json()
                assert payload["error"]["code"] == "invalid_toolsets"
                create.assert_not_called()
                assert adapter._run_statuses == {}


class TestRunToolsetConstruction:
    def test_selected_toolsets_reach_the_constructed_agent(self):
        adapter = _adapter()
        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": [],
            }),
            patch("gateway.run._resolve_gateway_model", return_value="test/model"),
            patch("gateway.run._load_gateway_config", return_value=_CONFIG),
            patch("run_agent.AIAgent", return_value=MagicMock()) as agent_cls,
        ):
            adapter._create_agent(requested_toolsets=["web"])

        assert agent_cls.call_args.kwargs["enabled_toolsets"] == ["web"]

    def test_empty_selected_toolsets_construct_an_agent_with_no_tools(self):
        adapter = _adapter()
        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": [],
            }),
            patch("gateway.run._resolve_gateway_model", return_value="test/model"),
            patch("gateway.run._load_gateway_config", return_value=_CONFIG),
            patch("run_agent.AIAgent", return_value=MagicMock()) as agent_cls,
        ):
            adapter._create_agent(requested_toolsets=[])

        assert agent_cls.call_args.kwargs["enabled_toolsets"] == []

    def test_request_narrowing_cannot_bypass_hosted_room_policy(self):
        adapter = _adapter()
        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": [], "args": [],
            }),
            patch("gateway.run._resolve_gateway_model", return_value="test/model"),
            patch("gateway.run._load_gateway_config", return_value=_CONFIG),
            patch("gateway.hosted_room_execution_policy.RoomExecutionPolicy.from_mapping",
                  return_value=SimpleNamespace(enabled_toolsets=("file",), max_iterations=7)),
            patch("run_agent.AIAgent", return_value=MagicMock()) as agent_cls,
        ):
            adapter._create_agent(
                room_dispatch={"room_id": "room"}, room_execution_policy={}, requested_toolsets=["web"])

        assert agent_cls.call_args.kwargs["enabled_toolsets"] == []


def test_configured_unknown_passthrough_is_not_a_valid_requested_toolset(monkeypatch):
    config = {"platform_toolsets": {"api_server": ["file", "definitely_unknown"]}}
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    from hermes_cli.tools_config import _get_platform_tools

    available = _get_platform_tools(config, "api_server")

    assert "definitely_unknown" in available
    with pytest.raises(ValueError, match="unknown"):
        _validate_run_toolsets({"toolsets": ["definitely_unknown"]}, available)


def test_enabled_mcp_alias_remains_a_valid_requested_toolset(monkeypatch):
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        name="mcp__runserver__ping", toolset="mcp-runserver",
        schema={"name": "mcp__runserver__ping", "description": "Ping", "parameters": {}},
        handler=lambda _args, **_kwargs: "{}")
    registry.register_toolset_alias("runserver", "mcp-runserver")
    monkeypatch.setattr("tools.registry.registry", registry)

    assert _validate_run_toolsets({"toolsets": ["runserver"]}, {"runserver"}) == ["runserver"]


class _CompletedRunAgent:
    provider = "openrouter"
    model = "locked/model"
    session_prompt_tokens = 1
    session_completion_tokens = 2
    session_total_tokens = 3
    _hermes_api_runtime = {
        "provider": "openrouter", "model": "locked/model", "route_source": "session_model_lock"}

    def run_conversation(self, **_kwargs):
        return {"final_response": "done"}


def _session_app(adapter):
    app = _app(adapter)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post("/api/sessions/{session_id}/model", adapter._handle_session_model_lock)
    return app


@pytest.mark.asyncio
async def test_run_uses_persisted_browser_lock_and_reports_effective_runtime():
    adapter = _adapter()
    async with TestClient(TestServer(_session_app(adapter))) as cli:
        created = await cli.post("/api/sessions", json={"id": "locked-session"})
        assert created.status == 201
        locked = await cli.post(
            "/api/sessions/locked-session/model",
            json={"model": "locked/model", "provider": "openrouter"},
        )
        assert locked.status == 200
        with patch.object(adapter, "_create_agent", return_value=_CompletedRunAgent()) as create:
            response = await cli.post(
                "/v1/runs", json={"input": "hello", "session_id": "locked-session"})
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            for _ in range(40):
                if adapter._run_statuses.get(run_id, {}).get("status") == "completed":
                    break
                await asyncio.sleep(0.01)

        assert create.call_args.kwargs["requested_model"] == "locked/model"
        assert create.call_args.kwargs["requested_provider"] == "openrouter"
        assert create.call_args.kwargs["confirmed_runtime_lock"] is True
        assert adapter._run_statuses[run_id]["runtime"] == {
            "provider": "openrouter", "model": "locked/model", "route_source": "session_model_lock"}


@pytest.mark.asyncio
async def test_run_rejects_persisted_lock_when_its_model_route_disappears():
    adapter = _adapter()
    adapter._model_routes = {
        "locked-alias": {"model": "locked/model", "provider": "openrouter"}}
    async with TestClient(TestServer(_session_app(adapter))) as cli:
        assert (await cli.post("/api/sessions", json={"id": "route-session"})).status == 201
        assert (await cli.post(
            "/api/sessions/route-session/model", json={"model": "locked-alias"})).status == 200
        adapter._model_routes = {}
        with patch.object(adapter, "_create_agent") as create:
            response = await cli.post(
                "/v1/runs", json={"input": "hello", "session_id": "route-session"})
            payload = await response.json()

    assert response.status == 409
    assert payload["error"]["code"] == "model_lock_unavailable"
    create.assert_not_called()


@pytest.mark.asyncio
async def test_run_rejects_body_runtime_that_conflicts_with_persisted_lock():
    adapter = _adapter()
    async with TestClient(TestServer(_session_app(adapter))) as cli:
        assert (await cli.post("/api/sessions", json={"id": "conflict-session"})).status == 201
        assert (await cli.post(
            "/api/sessions/conflict-session/model",
            json={"model": "locked/model", "provider": "openrouter"},
        )).status == 200
        with patch.object(adapter, "_create_agent") as create:
            response = await cli.post(
                "/v1/runs",
                json={
                    "input": "hello", "session_id": "conflict-session",
                    "model": "other/model", "provider": "other-provider",
                },
            )
            payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == "model_lock_conflict"
    create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body_options", "accepted"),
    [
        (None, True),
        ({"service_tier": "priority"}, True),
        ({"service_tier": "PRIORITY"}, True),
        ({}, False),
        ({"service_tier": "default"}, False),
    ],
    ids=["absent", "same", "equivalent", "empty", "different"],
)
async def test_run_enforces_persisted_model_options_exactly(body_options, accepted):
    adapter = _adapter()
    async with TestClient(TestServer(_session_app(adapter))) as cli:
        assert (await cli.post("/api/sessions", json={"id": "options-session"})).status == 201
        assert (await cli.post(
            "/api/sessions/options-session/model",
            json={"model": "locked/model", "provider": "openrouter",
                  "model_options": {"service_tier": "priority"}},
        )).status == 200
        with patch.object(adapter, "_create_agent", return_value=_CompletedRunAgent()) as create:
            payload = {"input": "hello", "session_id": "options-session"}
            if body_options is not None:
                payload["model_options"] = body_options
            response = await cli.post("/v1/runs", json=payload)
            result = await response.json()

    if accepted:
        assert response.status == 202
        create.assert_called_once()
        assert create.call_args.kwargs["model_options"] == {"service_tier": "priority"}
    else:
        assert response.status == 400
        assert result["error"]["code"] == "model_lock_conflict"
        create.assert_not_called()


@pytest.mark.asyncio
async def test_run_requires_confirmed_existing_session_when_model_lock_is_required():
    cases = (
        ({"input": "hello", "require_model_lock": True}, "missing"),
        ({"input": "hello", "session_id": "does-not-exist", "require_model_lock": True}, "unknown"),
    )
    for body, _case in cases:
        adapter = _adapter()
        async with TestClient(TestServer(_session_app(adapter))) as cli:
            with patch.object(adapter, "_create_agent") as create:
                response = await cli.post("/v1/runs", json=body)
                payload = await response.json()
        assert response.status == 409
        assert payload["error"]["code"] == "model_lock_unavailable"
        create.assert_not_called()

    adapter = _adapter()
    async with TestClient(TestServer(_session_app(adapter))) as cli:
        assert (await cli.post("/api/sessions", json={"id": "unlocked-session"})).status == 201
        with patch.object(adapter, "_create_agent") as create:
            response = await cli.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "unlocked-session", "require_model_lock": True},
            )
            payload = await response.json()
    assert response.status == 409
    assert payload["error"]["code"] == "model_lock_unavailable"
    create.assert_not_called()


@pytest.mark.asyncio
async def test_run_without_marker_keeps_arbitrary_session_compatibility():
    adapter = _adapter()
    async with TestClient(TestServer(_session_app(adapter))) as cli:
        assert (await cli.post("/api/sessions", json={"id": "arbitrary-session"})).status == 201
        with patch.object(adapter, "_create_agent", return_value=_CompletedRunAgent()) as create:
            response = await cli.post(
                "/v1/runs", json={"input": "hello", "session_id": "arbitrary-session"})
    assert response.status == 202
    create.assert_called_once()


@pytest.mark.asyncio
async def test_run_persisted_route_lock_accepts_unchanged_alias_target_and_rejects_drift():
    adapter = _adapter()
    adapter._model_routes = {"locked-alias": {"model": "locked/model", "provider": "openrouter"}}
    async with TestClient(TestServer(_session_app(adapter))) as cli:
        assert (await cli.post("/api/sessions", json={"id": "alias-session"})).status == 201
        assert (await cli.post(
            "/api/sessions/alias-session/model", json={"model": "locked-alias"})).status == 200
        with patch.object(adapter, "_create_agent", return_value=_CompletedRunAgent()) as create:
            response = await cli.post(
                "/v1/runs", json={"input": "hello", "session_id": "alias-session"})
            assert response.status == 202
            create.assert_called_once()
            assert create.call_args.kwargs["route"] == {
                "model": "locked/model", "provider": "openrouter"}
        adapter._model_routes = {"locked-alias": {"model": "changed/model", "provider": "openrouter"}}
        with patch.object(adapter, "_create_agent") as drift_create:
            response = await cli.post(
                "/v1/runs", json={"input": "hello", "session_id": "alias-session"})
            payload = await response.json()

    assert response.status == 409
    assert payload["error"]["code"] == "model_lock_unavailable"
    drift_create.assert_not_called()


@pytest.mark.asyncio
async def test_run_idempotency_replays_locked_alias_after_route_drift():
    adapter = _adapter()
    adapter._model_routes = {
        "locked-alias": {"model": "locked/model", "provider": "openrouter"}}
    async with TestClient(TestServer(_session_app(adapter))) as cli:
        assert (await cli.post("/api/sessions", json={"id": "replay-alias-session"})).status == 201
        assert (await cli.post(
            "/api/sessions/replay-alias-session/model",
            json={"model": "locked-alias"},
        )).status == 200
        with patch.object(adapter, "_create_agent", return_value=_CompletedRunAgent()) as create:
            headers = {"Idempotency-Key": "locked-alias-replay"}
            first = await cli.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "replay-alias-session"},
                headers=headers,
            )
            first_body = await first.json()
            adapter._model_routes = {}
            replay = await cli.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "replay-alias-session"},
                headers=headers,
            )
            replay_body = await replay.json()
            fresh = await cli.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "replay-alias-session"},
                headers={"Idempotency-Key": "locked-alias-new"},
            )
            fresh_body = await fresh.json()

    assert first.status == 202
    assert replay.status == 202
    assert replay_body["run_id"] == first_body["run_id"]
    assert replay_body["replayed"] is True
    assert fresh.status == 409
    assert fresh_body["error"]["code"] == "model_lock_unavailable"
    create.assert_called_once()


def test_confirmed_lock_rejects_an_agent_that_resolves_to_a_different_runtime():
    adapter = _adapter()
    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
            "api_key": "test-key", "base_url": None, "provider": None,
            "api_mode": None, "command": None, "args": [],
        }),
        patch("gateway.run._resolve_gateway_model", return_value="global/model"),
        patch("gateway.run._load_gateway_config", return_value=_CONFIG),
        patch.object(adapter, "_resolve_provider_runtime", return_value={"provider": "openrouter"}),
        patch("run_agent.AIAgent", return_value=SimpleNamespace(provider="other", model="other/model")),
    ):
        with pytest.raises(RuntimeError, match="confirmed model lock runtime mismatch"):
            adapter._create_agent(
                requested_model="locked/model", requested_provider="openrouter",
                route={"model": "locked/model", "provider": "openrouter"},
                confirmed_runtime_lock=True)


def test_confirmed_lock_disables_fallback_and_reports_locked_runtime():
    adapter = _adapter()
    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
            "api_key": "test-key", "base_url": None, "provider": None,
            "api_mode": None, "command": None, "args": []}),
        patch("gateway.run._resolve_gateway_model", return_value="global/model"),
        patch("gateway.run._load_gateway_config", return_value=_CONFIG),
        patch.object(adapter, "_resolve_provider_runtime", return_value={"provider": "openrouter"}),
        patch("run_agent.AIAgent", return_value=SimpleNamespace(
            provider="openrouter", model="locked/model")) as agent_cls,
    ):
        agent = adapter._create_agent(
            requested_model="locked/model", requested_provider="openrouter",
            route={"model": "locked/model", "provider": "openrouter"},
            confirmed_runtime_lock=True)

    assert agent is not None
    assert agent_cls.call_args.kwargs["fallback_model"] is None
    assert agent._hermes_api_runtime == {
        "provider": "openrouter", "model": "locked/model",
        "route_source": "session_model_lock", "model_lock": "confirmed"}


def test_create_agent_real_import_resolves_selected_tool_definitions():
    adapter = _adapter()
    runtime = {
        "api_key": "test-key", "base_url": "http://127.0.0.1:1/v1", "provider": None,
        "api_mode": "chat_completions", "command": None, "args": [],
    }
    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime),
        patch("gateway.run._resolve_gateway_model", return_value="test/model"),
        patch("gateway.run._load_gateway_config", return_value=_CONFIG),
    ):
        web_agent = adapter._create_agent(requested_toolsets=["web"])
        empty_agent = adapter._create_agent(requested_toolsets=[])

    from toolsets import resolve_toolset

    web_names = {tool["function"]["name"] for tool in web_agent.tools}
    empty_names = {tool["function"]["name"] for tool in empty_agent.tools}
    assert web_names == set(resolve_toolset("web"))
    assert empty_names == set()


def test_create_agent_real_import_resolves_file_readonly_from_configured_file():
    adapter = _adapter()
    runtime = {
        "api_key": "test-key", "base_url": "http://127.0.0.1:1/v1", "provider": None,
        "api_mode": "chat_completions", "command": None, "args": [],
    }
    config = {"platform_toolsets": {"api_server": ["file"]}}
    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime),
        patch("gateway.run._resolve_gateway_model", return_value="test/model"),
        patch("gateway.run._load_gateway_config", return_value=config),
    ):
        agent = adapter._create_agent(requested_toolsets=["file_readonly"])

    assert {tool["function"]["name"] for tool in agent.tools} == {"read_file", "search_files"}


def test_request_rejects_derived_toolset_when_a_resolved_tool_is_unavailable(monkeypatch):
    from toolsets import TOOLSETS

    partial_name = "_test_file_readonly_partial"
    monkeypatch.setitem(
        TOOLSETS,
        partial_name,
        {"description": "Test partial file policy", "tools": ["read_file"], "includes": []},
    )

    with pytest.raises(ValueError, match="unavailable"):
        _validate_run_toolsets({"toolsets": ["file_readonly"]}, {partial_name})
