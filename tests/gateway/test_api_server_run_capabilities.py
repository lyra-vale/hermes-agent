"""Behavioral contract for per-run API toolset narrowing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


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
        from gateway.platforms.api_server import _validate_run_toolsets

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
        from gateway.platforms.api_server import _validate_run_toolsets

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
                "api_mode": None, "command": None, "args": [],
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
