"""Contract tests for API run file confinement and memory capability boundaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tools.file_operations_common import SearchResult


_GATEWAY_CONFIG = {"platform_toolsets": {"api_server": ["file", "memory"]}}


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


class _FileRunAgent:
    provider = "test"
    model = "test/model"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0

    def __init__(self, operation):
        self._operation = operation
        self.observed = None

    def run_conversation(self, *, task_id, **_kwargs):
        self.observed = self._operation(task_id)
        return {"final_response": "done"}


async def _wait_for_terminal_status(adapter: APIServerAdapter, run_id: str) -> dict:
    for _ in range(100):
        status = adapter._run_statuses.get(run_id, {})
        if status.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
            return status
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not settle: {adapter._run_statuses.get(run_id)}")


async def _post_file_run(adapter, body, agent):
    with patch("gateway.run._load_gateway_config", return_value=_GATEWAY_CONFIG), patch.object(
        adapter, "_create_agent", return_value=agent
    ):
        async with TestClient(TestServer(_app(adapter))) as cli:
            response = await cli.post("/v1/runs", json=body)
            payload = await response.json()
            if response.status == 202:
                await _wait_for_terminal_status(adapter, payload["run_id"])
            return response, payload


@pytest.mark.asyncio
async def test_file_readonly_run_reads_only_a_server_owned_root(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    allowed = root / "allowed.txt"
    allowed.write_text("allowed-content\n", encoding="utf-8")
    agent = _FileRunAgent(
        lambda task_id: __import__("tools.file_tools", fromlist=["read_file_tool"]).read_file_tool(
            str(allowed), task_id=task_id
        )
    )
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"file_readonly_roots": [str(root)]})
    )

    response, _payload = await _post_file_run(
        adapter, {"input": "read", "toolsets": ["file_readonly"]}, agent
    )

    assert response.status == 202
    result = json.loads(agent.observed)
    assert "allowed-content" in result["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path_kind", ["traversal", "symlink"])
async def test_file_readonly_run_denies_traversal_and_symlink_escape(tmp_path, path_kind):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    if path_kind == "traversal":
        requested = str(root / ".." / outside.name)
    else:
        link = root / "link.txt"
        link.symlink_to(outside)
        requested = str(link)

    agent = _FileRunAgent(
        lambda task_id: __import__("tools.file_tools", fromlist=["read_file_tool"]).read_file_tool(
            requested, task_id=task_id
        )
    )
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"file_readonly_roots": [str(root)]})
    )

    response, _payload = await _post_file_run(
        adapter, {"input": "read", "toolsets": ["file_readonly"]}, agent
    )

    assert response.status == 202
    result = json.loads(agent.observed)
    assert "error" in result
    assert "read-only root" in result["error"].lower()
    assert "outside-secret" not in agent.observed


@pytest.mark.asyncio
async def test_file_readonly_search_denies_external_target_and_external_result(tmp_path, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    link = root / "link.txt"
    link.symlink_to(outside)

    from tools import file_tools

    calls = []

    class _SearchOps:
        def search(self, **_kwargs):
            calls.append(True)
            return SearchResult(files=[str(link)], total_count=1)

    def _operation(task_id):
        external_target = file_tools.search_tool(
            "outside", target="content", path=str(outside.parent), task_id=task_id
        )
        monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: _SearchOps())
        symlink_result = file_tools.search_tool(
            "outside", target="files", path=str(root), task_id=task_id
        )
        return external_target, symlink_result

    agent = _FileRunAgent(_operation)
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"file_readonly_roots": [str(root)]})
    )

    response, _payload = await _post_file_run(
        adapter, {"input": "search", "toolsets": ["file_readonly"]}, agent
    )

    assert response.status == 202
    external_target, symlink_result = agent.observed
    target_payload = json.loads(external_target)
    result_payload = json.loads(symlink_result)
    assert "error" in target_payload
    assert "read-only root" in target_payload["error"].lower()
    assert str(link) not in symlink_result
    assert "files" not in result_payload
    assert calls == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [None, {"file_readonly_roots": []}, {"file_readonly_roots": ["relative/root"]}],
)
async def test_file_readonly_run_requires_valid_server_roots(tmp_path, extra):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra=extra or {}))
    agent = MagicMock()

    response, payload = await _post_file_run(
        adapter, {"input": "read", "toolsets": ["file_readonly"]}, agent
    )

    assert response.status == 400
    assert payload["error"]["code"] == "invalid_toolsets"
    agent.run_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_file_readonly_roots_are_not_supplied_by_the_request(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    agent = MagicMock()

    response, payload = await _post_file_run(
        adapter,
        {
            "input": "read",
            "toolsets": ["file_readonly"],
            "file_readonly_roots": [str(outside)],
        },
        agent,
    )

    assert response.status == 400
    assert payload["error"]["code"] == "invalid_toolsets"
    agent.run_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_file_readonly_scope_is_reset_after_a_run(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    allowed = root / "allowed.txt"
    allowed.write_text("allowed\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    from tools.file_tools import read_file_tool

    agent = _FileRunAgent(lambda task_id: read_file_tool(str(allowed), task_id=task_id))
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"file_readonly_roots": [str(root)]})
    )
    response, _payload = await _post_file_run(
        adapter, {"input": "read", "toolsets": ["file_readonly"]}, agent
    )
    assert response.status == 202

    # The run-local scope must not change ordinary non-API file behavior after reset.
    outside_result = json.loads(read_file_tool(str(outside), task_id="outside-run"))
    assert "outside" in outside_result["content"]


# The provider below is intentionally production-shaped: it enters through
# plugins.memory.load_memory_provider and exposes the Holographic schemas.
class _HolographicShapedProvider:
    name = "holographic"

    def __init__(self, schemas):
        self.schemas = schemas
        self.calls = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.session_id = session_id

    def get_tool_schemas(self):
        return self.schemas

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.calls.append((tool_name, args))
        return json.dumps({"handled": tool_name, "args": args})

    def shutdown(self):
        pass


def _memory_agent(toolsets, provider, *, config=None):
    from plugins.memory.holographic import FACT_FEEDBACK_SCHEMA, FACT_STORE_SCHEMA

    cfg = config or {"memory": {"provider": "holographic"}, "agent": {}}
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204800),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            enabled_toolsets=toolsets,
            session_id="memory-test",
        )


def _holographic_provider():
    from plugins.memory.holographic import FACT_FEEDBACK_SCHEMA, FACT_STORE_SCHEMA

    return _HolographicShapedProvider([FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA])


def _tool_map(agent):
    return {tool["function"]["name"]: tool["function"] for tool in agent.tools}


def test_explicit_empty_toolsets_strip_provider_tools_after_injection():
    provider = _holographic_provider()
    agent = _memory_agent([], provider)

    assert agent.valid_tool_names == set()
    assert _tool_map(agent) == {}
    assert provider.calls == []


def test_memory_append_exposes_only_recall_and_fact_add_with_narrow_schema():
    provider = _holographic_provider()
    agent = _memory_agent(["memory_append"], provider)
    tools = _tool_map(agent)

    assert set(tools) == {"fact_store"}
    assert agent.valid_tool_names == set(tools)
    fact_store = tools["fact_store"]
    actions = fact_store["parameters"]["properties"]["action"]["enum"]
    assert set(actions) == {"add", "search", "probe", "related", "reason", "contradict", "list"}
    assert "update" not in actions
    assert "remove" not in actions
    assert "fact_id" not in fact_store["parameters"]["properties"]
    assert "fact_feedback" not in tools
    assert "memory" not in tools


def test_memory_append_dispatch_rejects_mutation_and_feedback():
    provider = _holographic_provider()
    agent = _memory_agent(["memory_append"], provider)

    update = agent._invoke_tool(
        "fact_store", {"action": "update", "fact_id": 1, "content": "bad"}, "memory-task",
        pre_tool_block_checked=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )
    feedback = agent._invoke_tool(
        "fact_feedback", {"action": "helpful", "fact_id": 1}, "memory-task",
        pre_tool_block_checked=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert "error" in json.loads(update)
    assert "error" in json.loads(feedback)
    assert provider.calls == []


def test_full_memory_keeps_builtin_and_full_provider_surface():
    provider = _holographic_provider()
    agent = _memory_agent(["memory"], provider)
    tools = _tool_map(agent)

    assert {"memory", "fact_store", "fact_feedback"} <= set(tools)
    actions = tools["fact_store"]["parameters"]["properties"]["action"]["enum"]
    assert {"add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"} <= set(actions)

    result = agent._invoke_tool(
        "fact_store", {"action": "update", "fact_id": 1, "content": "allowed"}, "memory-task",
        pre_tool_block_checked=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )
    assert json.loads(result)["handled"] == "fact_store"
    assert provider.calls == [("fact_store", {"action": "update", "fact_id": 1, "content": "allowed"})]


def test_omitted_toolsets_keep_full_provider_surface():
    provider = _holographic_provider()
    agent = _memory_agent(None, provider)
    tools = _tool_map(agent)

    assert {"memory", "fact_store", "fact_feedback"} <= set(tools)
    assert agent.valid_tool_names == set(tools)


def test_memory_append_requires_backing_memory_toolset():
    from gateway.platforms.api_server_runs import _validate_run_toolsets

    with pytest.raises(ValueError, match="unavailable"):
        _validate_run_toolsets({"toolsets": ["memory_append"]}, {"bot_room", "file"})


def test_hosted_room_policy_cannot_name_memory_append_without_memory():
    from gateway.hosted_room_execution_policy import RoomExecutionPolicy, RoomExecutionPolicyError, _policy_digest

    unsigned = {
        "version": 1,
        "target_profile": "default",
        "enabled_toolsets": ["bot_room", "memory_append"],
        "approval_mode": "manual",
        "max_iterations": 5,
    }
    with pytest.raises(RoomExecutionPolicyError, match="memory"):
        RoomExecutionPolicy.from_mapping({**unsigned, "policy_digest": _policy_digest(unsigned)})
