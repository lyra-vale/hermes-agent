"""Behavior contracts for receipt-bound, append-only memory runs."""

from __future__ import annotations

import json

import pytest


def _request():
    from agent.memory_append_receipt import parse_memory_append_receipt

    return parse_memory_append_receipt({
        "reset_id": "voice-reset-abc123",
        "facts": [
            {"content": "Din prefers concise operational updates.", "category": "user_pref"},
            {"content": "Voice reset summaries retain only durable decisions and commitments.", "category": "project"},
        ],
    })


def test_memory_append_receipt_accepts_only_the_bound_facts_and_emits_verified_ids():
    policy = _request()

    assert policy.admit("fact_store", {
        "action": "add", "content": "Din prefers concise operational updates.", "category": "user_pref",
    }) is None
    policy.record("fact_store", {
        "action": "add", "content": "Din prefers concise operational updates.", "category": "user_pref",
    }, json.dumps({"fact_id": 71, "status": "added", "category": "user_pref"}))

    assert policy.admit("fact_store", {
        "action": "add", "content": "Voice reset summaries retain only durable decisions and commitments.", "category": "project",
    }) is None
    policy.record("fact_store", {
        "action": "add", "content": "Voice reset summaries retain only durable decisions and commitments.", "category": "project",
    }, json.dumps({"fact_id": 72, "status": "added", "category": "project"}))

    assert policy.receipt(run_id="run_test", session_id="session_test", provider="holographic") == {
        "object": "hermes.memory_append_receipt",
        "version": 1,
        "status": "committed",
        "run_id": "run_test",
        "session_id": "session_test",
        "reset_id": "voice-reset-abc123",
        "provider": "holographic",
        "fact_count": 2,
        "facts": [
            {"fact_id": 71, "category": "user_pref", "content_sha256": "d554852fd1f79127dda9028f851532f7f33a8f087f9a1f0361c7f2a96a024b75"},
            {"fact_id": 72, "category": "project", "content_sha256": "314a128cee5d030aa37ed2b5c1783294d6657442ae0864682228fa824bad360c"},
        ],
    }


def test_memory_append_receipt_rejects_unbound_or_duplicate_writes_before_provider_execution():
    policy = _request()

    assert policy.admit("fact_store", {
        "action": "add", "content": "Unrelated durable fact.", "category": "general",
    }) == "memory append receipt does not authorize this fact"
    assert policy.admit("fact_store", {
        "action": "add", "content": "Din prefers concise operational updates.", "category": "general",
    }) == "memory append receipt does not authorize this fact"
    assert policy.admit("fact_feedback", {"action": "helpful", "fact_id": 1}) == "memory append receipt permits only fact_store add"

    assert policy.admit("fact_store", {
        "action": "add", "content": "Din prefers concise operational updates.", "category": "user_pref",
    }) is None
    assert policy.admit("fact_store", {
        "action": "add", "content": "Din prefers concise operational updates.", "category": "user_pref",
    }) == "memory append receipt does not authorize this fact"


@pytest.mark.parametrize("payload", [
    {"reset_id": "bad reset id", "facts": [{"content": "valid", "category": "general"}]},
    {"reset_id": "voice-reset-abc123", "facts": []},
    {"reset_id": "voice-reset-abc123", "facts": [{"content": "contains\na transcript line", "category": "general"}]},
    {"reset_id": "voice-reset-abc123", "facts": [
        {"content": "same", "category": "general"}, {"content": "same", "category": "general"},
    ]},
])
def test_memory_append_receipt_rejects_invalid_candidate_shapes(payload):
    from agent.memory_append_receipt import parse_memory_append_receipt

    with pytest.raises(ValueError, match="memory append receipt"):
        parse_memory_append_receipt(payload)


def test_memory_append_receipt_rejects_a_provider_result_without_a_durable_fact_id():
    policy = _request()
    args = {"action": "add", "content": "Din prefers concise operational updates.", "category": "user_pref"}
    assert policy.admit("fact_store", args) is None

    with pytest.raises(ValueError, match="memory append receipt"):
        policy.record("fact_store", args, json.dumps({"status": "added"}))


def test_memory_manager_rejects_receipt_mode_for_a_provider_without_durable_append_contract():
    from agent.memory_manager import MemoryManager
    from agent.memory_append_receipt import bind_memory_append_receipt, reset_memory_append_receipt

    class Provider:
        name = "unverified-provider"

        def get_tool_schemas(self):
            return [{"name": "fact_store", "parameters": {"type": "object", "properties": {}}}]

        def handle_tool_call(self, _tool_name, _args, **_kwargs):
            raise AssertionError("provider must not be called")

    manager = MemoryManager()
    manager.add_provider(Provider())
    manager.configure_tool_surface("append")
    token = bind_memory_append_receipt(_request())
    try:
        result = manager.handle_tool_call("fact_store", {
            "action": "add", "content": "Din prefers concise operational updates.", "category": "user_pref",
        })
    finally:
        reset_memory_append_receipt(token)
    assert "requires a durable fact append provider" in result


def test_memory_manager_enforces_the_receipt_before_provider_write():
    from agent.memory_manager import MemoryManager
    from agent.memory_append_receipt import bind_memory_append_receipt, reset_memory_append_receipt

    class Provider:
        name = "receipt-provider"
        durable_fact_append_receipt_version = 1

        def __init__(self):
            self.calls = []

        def get_tool_schemas(self):
            return [{"name": "fact_store", "parameters": {"type": "object", "properties": {}}}]

        def handle_tool_call(self, tool_name, args, **_kwargs):
            self.calls.append((tool_name, args))
            return json.dumps({"fact_id": 91, "status": "added", "category": "user_pref"})

    manager, provider = MemoryManager(), Provider()
    manager.add_provider(provider)
    manager.configure_tool_surface("append")
    token = bind_memory_append_receipt(_request())
    try:
        denied = manager.handle_tool_call("fact_store", {
            "action": "add", "content": "unbound fact", "category": "general",
        })
        assert "does not authorize" in denied
        assert provider.calls == []

        allowed = manager.handle_tool_call("fact_store", {
            "action": "add", "content": "Din prefers concise operational updates.", "category": "user_pref",
        })
        assert json.loads(allowed) == {"fact_id": 91, "status": "added", "category": "user_pref"}
        assert provider.calls == [("fact_store", {
            "action": "add", "content": "Din prefers concise operational updates.", "category": "user_pref",
        })]
    finally:
        reset_memory_append_receipt(token)


@pytest.mark.asyncio
async def test_api_run_publishes_only_a_verified_memory_append_receipt():
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from agent.memory_append_receipt import current_memory_append_receipt

    class ReceiptAgent:
        provider = "test"
        model = "test/model"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        class _ReceiptProvider:
            name = "holographic"

            def append_receipt(self, reset_id, facts):
                assert reset_id == "voice-reset-abc123"
                assert facts == [{"content": "Din prefers concise operational updates.", "category": "user_pref"}]
                return {
                    "reset_id": reset_id,
                    "facts": [{
                        "fact_id": 313,
                        "category": "user_pref",
                        "content_sha256": "d554852fd1f79127dda9028f851532f7f33a8f087f9a1f0361c7f2a96a024b75",
                    }],
                }

        class _ReceiptManager:
            def receipt_append_provider(self):
                return ReceiptAgent._ReceiptProvider()

        _memory_manager = _ReceiptManager()

        def run_conversation(self, **_kwargs):
            raise AssertionError("receipt persistence must not invoke a model")

    from aiohttp import web
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    body = {
        "session_id": "locked-memory-session",
        "require_model_lock": True,
        "input": "Record approved durable facts.",
        "toolsets": ["memory_append"],
        "memory_append_receipt": {
            "reset_id": "voice-reset-abc123",
            "facts": [{"content": "Din prefers concise operational updates.", "category": "user_pref"}],
        },
    }
    from unittest.mock import AsyncMock, MagicMock, patch
    fake_db = MagicMock()
    fake_db.get_session.return_value = {"id": "locked-memory-session"}
    locked_runtime = {"persisted_lock": True, "require_model_lock": True, "route": None,
                      "requested": {}, "model_options": {}}
    with patch("gateway.run._load_gateway_config", return_value={"platform_toolsets": {"api_server": ["memory", "memory_append"]}}), patch.object(
        adapter, "_ensure_session_db_async", AsyncMock(return_value=fake_db)
    ), patch("gateway.platforms.api_server_runs._effective_run_runtime_request", return_value=locked_runtime), patch.object(
        adapter, "_runtime_lock_error", return_value=None
    ), patch.object(
        adapter, "_create_agent", return_value=ReceiptAgent(),
    ):
        async with TestClient(TestServer(app)) as client:
            started = await client.post("/v1/runs", json=body, headers={"Idempotency-Key": "voice-reset-abc123"})
            assert started.status == 202
            run_id = (await started.json())["run_id"]
            for _ in range(100):
                status = await client.get(f"/v1/runs/{run_id}")
                payload = await status.json()
                if payload["status"] in {"completed", "failed"}:
                    break
                await __import__("asyncio").sleep(0.01)
            retry = await client.post("/v1/runs", json={**body, "session_id": "fresh-retry-session"},
                                      headers={"Idempotency-Key": "voice-reset-abc123"})
            assert retry.status == 202
            assert (await retry.json())["run_id"] == run_id

    assert payload["status"] == "completed"
    receipt = payload["memory_append_receipt"]
    assert receipt == {
        "object": "hermes.memory_append_receipt",
        "version": 1,
        "status": "committed",
        "run_id": run_id,
        "session_id": "locked-memory-session",
        "reset_id": "voice-reset-abc123",
        "provider": "holographic",
        "fact_count": 1,
        "facts": [{
            "fact_id": 313,
            "category": "user_pref",
            "content_sha256": "d554852fd1f79127dda9028f851532f7f33a8f087f9a1f0361c7f2a96a024b75",
        }],
    }
    assert "concise operational" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_api_rejects_memory_append_without_a_receipt_envelope():
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.platforms.api_server import APIServerAdapter, PlatformConfig

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/runs", json={
            "input": "Record approved durable facts.",
            "toolsets": ["memory_append"],
        })
        assert response.status == 400
        payload = await response.json()
    assert payload["error"]["code"] == "invalid_memory_append_receipt"
