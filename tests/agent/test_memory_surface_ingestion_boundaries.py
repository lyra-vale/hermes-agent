"""External memory surface invariants for automatic transcript ingestion."""

from __future__ import annotations

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from run_agent import AIAgent


_USER = "RAW_USER_TRANSCRIPT_9f2c"
_ASSISTANT = "RAW_ASSISTANT_TRANSCRIPT_4a71"
_MESSAGES = [
    {"role": "user", "content": _USER},
    {"role": "assistant", "content": _ASSISTANT},
]


class _TranscriptSpyProvider(MemoryProvider):
    """Production-shaped provider that records every content-bearing callback."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    @property
    def name(self) -> str:
        return "transcript-spy"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        return None

    def system_prompt_block(self) -> str:
        self.events.append(("prompt",))
        return "provider read context"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self.events.append(("prefetch", query))
        return "provider recalled context"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self.events.append(("queue_prefetch", query))

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self.events.append(("turn_start", message))

    def sync_turn(
        self, user_content: str, assistant_content: str, *,
        session_id: str = "", messages=None, turn_author=None,
    ) -> None:
        self.events.append(("sync", user_content, assistant_content, messages))

    def get_tool_schemas(self) -> list[dict]:
        return []

    def on_session_end(self, messages) -> None:
        self.events.append(("session_end", messages))

    def on_pre_compress(self, messages) -> str:
        self.events.append(("pre_compress", messages))
        return "provider summary context"

    def on_memory_write(self, action, target, content, metadata=None) -> None:
        self.events.append(("memory_write", action, target, content))

    def on_delegation(self, task, result, *, child_session_id="", **kwargs) -> None:
        self.events.append(("delegation", task, result))

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        # Rebinding has no transcript payload and remains valid on restricted surfaces.
        self.events.append(("session_switch", new_session_id))


def _manager_for(surface: str) -> tuple[MemoryManager, _TranscriptSpyProvider]:
    manager = MemoryManager()
    provider = _TranscriptSpyProvider()
    manager.add_provider(provider)
    manager.configure_tool_surface(surface)
    return manager, provider


def _agent_for(manager: MemoryManager) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = manager
    agent.session_id = "session-under-test"
    agent._turn_author = None
    return agent


@pytest.mark.parametrize("surface", ["none", "append", "full"])
def test_automatic_transcript_callbacks_follow_memory_surface_policy(surface, tmp_path, monkeypatch):
    """Restricted surfaces never hand transcript data to automatic provider hooks."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    manager, provider = _manager_for(surface)

    manager.build_system_prompt()
    manager.prefetch_all(_USER, session_id="old-session")
    manager.queue_prefetch_all(_USER, session_id="old-session")
    manager.on_turn_start(1, _USER)
    manager.sync_all(_USER, _ASSISTANT, session_id="old-session", messages=_MESSAGES)
    manager.on_pre_compress(_MESSAGES)
    manager.on_memory_write("add", "memory", _ASSISTANT)
    manager.on_delegation(_USER, _ASSISTANT, child_session_id="child-session")
    manager.on_session_end(_MESSAGES)
    manager.commit_session_boundary_async(
        _MESSAGES, new_session_id="new-session", parent_session_id="old-session",
    )
    assert manager.flush_pending(timeout=5)

    automatic_transcript_events = {
        "turn_start", "sync", "pre_compress", "memory_write", "delegation", "session_end",
    }
    content_events = [event for event in provider.events if event[0] in automatic_transcript_events]

    if surface in {"none", "append"}:
        assert content_events == []
    else:
        assert ("turn_start", _USER) in provider.events
        assert ("sync", _USER, _ASSISTANT, _MESSAGES) in provider.events
        assert ("pre_compress", _MESSAGES) in provider.events
        assert ("memory_write", "add", "memory", _ASSISTANT) in provider.events
        assert ("delegation", _USER, _ASSISTANT) in provider.events
        assert ("session_end", _MESSAGES) in provider.events

    if surface == "none":
        assert [event[0] for event in provider.events] == ["session_switch"]
    elif surface == "append":
        assert ("prompt",) in provider.events
        assert ("prefetch", _USER) in provider.events
        assert ("queue_prefetch", _USER) in provider.events
        assert ("session_switch", "new-session") in provider.events
    else:
        assert ("prompt",) in provider.events
        assert ("prefetch", _USER) in provider.events
        assert ("queue_prefetch", _USER) in provider.events
        assert ("session_switch", "new-session") in provider.events


@pytest.mark.parametrize("surface", ["none", "append", "full"])
def test_agent_turn_sync_respects_restricted_memory_surface(surface, tmp_path, monkeypatch):
    """The agent-side turn bridge cannot queue sync or prefetch on restricted surfaces."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    manager, provider = _manager_for(surface)
    agent = _agent_for(manager)

    agent._sync_external_memory_for_turn(
        original_user_message=_USER,
        final_response=_ASSISTANT,
        interrupted=False,
        messages=_MESSAGES,
    )
    assert manager.flush_pending(timeout=5)

    sync_events = [event for event in provider.events if event[0] == "sync"]
    queued_prefetch_events = [event for event in provider.events if event[0] == "queue_prefetch"]
    if surface in {"none", "append"}:
        assert sync_events == []
        assert queued_prefetch_events == []
    else:
        assert sync_events == [("sync", _USER, _ASSISTANT, _MESSAGES)]
        assert queued_prefetch_events == [("queue_prefetch", _USER)]
