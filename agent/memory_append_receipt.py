"""Receipt-bound policy for one explicit append-only durable-memory run.

The policy lives in a ContextVar because /v1/runs execute concurrently. It is not
an HTTP endpoint and is inert unless the API run explicitly opts into the narrow
receipt protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Optional


MEMORY_APPEND_RECEIPT_VERSION = 1
_MAX_FACTS = 8
_MAX_FACT_CHARS = 280
_MAX_FACT_BYTES = 560
_RESET_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CATEGORIES = frozenset({"user_pref", "project", "tool", "general"})
_current_policy: ContextVar[Optional["MemoryAppendReceiptPolicy"]] = ContextVar(
    "memory_append_receipt_policy", default=None,
)


def _error() -> ValueError:
    return ValueError("memory append receipt is invalid")


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryAppendFact:
    content: str
    category: str


@dataclass
class MemoryAppendReceiptPolicy:
    reset_id: str
    facts: tuple[MemoryAppendFact, ...]
    _admitted: set[tuple[str, str]] = field(default_factory=set, init=False)
    _recorded: dict[tuple[str, str], int] = field(default_factory=dict, init=False)

    def admit(self, tool_name: str, args: Any) -> Optional[str]:
        """Reserve exactly one prevalidated fact; every other provider call is denied."""
        if tool_name != "fact_store":
            return "memory append receipt permits only fact_store add"
        if not isinstance(args, dict) or set(args) - {"action", "content", "category"}:
            return "memory append receipt does not authorize this fact"
        content, category = args.get("content"), args.get("category", "general")
        if args.get("action") != "add" or not isinstance(content, str) or not isinstance(category, str):
            return "memory append receipt does not authorize this fact"
        key = (content, category)
        if key not in {(fact.content, fact.category) for fact in self.facts} or key in self._admitted:
            return "memory append receipt does not authorize this fact"
        self._admitted.add(key)
        return None

    def record(self, tool_name: str, args: dict[str, Any], result: str) -> None:
        """Record only a provider-confirmed, durable Holographic fact identifier."""
        if tool_name != "fact_store":
            raise _error()
        key = (args.get("content"), args.get("category", "general"))
        if key not in self._admitted:
            raise _error()
        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError) as exc:
            raise _error() from exc
        if not isinstance(payload, dict):
            raise _error()
        fact_id = payload.get("fact_id")
        if (payload.get("status") != "added" or payload.get("category") != key[1]
                or not isinstance(fact_id, int) or fact_id <= 0):
            raise _error()
        if fact_id in self._recorded.values():
            raise _error()
        self._recorded[key] = fact_id

    def provider_facts(self) -> list[dict[str, str]]:
        """The only fact payload permitted to reach a receipt-capable provider."""
        return [{"content": fact.content, "category": fact.category} for fact in self.facts]

    def receipt_from_provider(self, payload: Any, *, run_id: str, session_id: str, provider: str) -> dict[str, Any]:
        """Validate the provider's committed ledger record before terminal publication."""
        if (not isinstance(payload, dict) or payload.get("reset_id") != self.reset_id
                or not isinstance(payload.get("facts"), list) or len(payload["facts"]) != len(self.facts)):
            raise _error()
        terminal_facts: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for expected, received in zip(self.facts, payload["facts"]):
            if not isinstance(received, dict):
                raise _error()
            fact_id = received.get("fact_id")
            if (not isinstance(fact_id, int) or fact_id <= 0 or fact_id in seen_ids
                    or received.get("category") != expected.category
                    or received.get("content_sha256") != _digest(expected.content)):
                raise _error()
            seen_ids.add(fact_id)
            terminal_facts.append({
                "fact_id": fact_id,
                "category": expected.category,
                "content_sha256": _digest(expected.content),
            })
        return {
            "object": "hermes.memory_append_receipt",
            "version": MEMORY_APPEND_RECEIPT_VERSION,
            "status": "committed",
            "run_id": run_id,
            "session_id": session_id,
            "reset_id": self.reset_id,
            "provider": provider,
            "fact_count": len(self.facts),
            "facts": terminal_facts,
        }

    def receipt(self, *, run_id: str = "", session_id: str = "", provider: str = "") -> dict[str, Any]:
        if len(self._recorded) != len(self.facts):
            raise _error()
        if run_id and not _RESET_ID.fullmatch(run_id):
            raise _error()
        if session_id and len(session_id) > 128:
            raise _error()
        return {
            "object": "hermes.memory_append_receipt",
            "version": MEMORY_APPEND_RECEIPT_VERSION,
            "status": "committed",
            **({"run_id": run_id} if run_id else {}),
            **({"session_id": session_id} if session_id else {}),
            "reset_id": self.reset_id,
            **({"provider": provider} if provider else {}),
            "fact_count": len(self.facts),
            "facts": [
                {
                    "fact_id": self._recorded[(fact.content, fact.category)],
                    "category": fact.category,
                    "content_sha256": _digest(fact.content),
                }
                for fact in self.facts
            ],
        }

    def model_instruction(self) -> str:
        """Bounded server-owned instruction; raw Voice transcript never crosses into Hermes."""
        facts = json.dumps(
            [{"content": fact.content, "category": fact.category} for fact in self.facts],
            ensure_ascii=False, separators=(",", ":"),
        )
        return (
            "Record every approved fact below using fact_store action=add with the exact content and category. "
            "Do not call any other tool or take any other action. Approved facts: "
            f"{facts}"
        )


def parse_memory_append_receipt(value: Any) -> MemoryAppendReceiptPolicy:
    """Validate a bounded, transcript-free receipt request before agent construction."""
    if not isinstance(value, dict) or set(value) != {"reset_id", "facts"}:
        raise _error()
    reset_id, raw_facts = value.get("reset_id"), value.get("facts")
    if not isinstance(reset_id, str) or not _RESET_ID.fullmatch(reset_id):
        raise _error()
    if not isinstance(raw_facts, list) or not 1 <= len(raw_facts) <= _MAX_FACTS:
        raise _error()
    facts: list[MemoryAppendFact] = []
    seen: set[str] = set()
    for raw in raw_facts:
        if not isinstance(raw, dict) or set(raw) - {"content", "category"}:
            raise _error()
        content, category = raw.get("content"), raw.get("category", "general")
        if (
            not isinstance(content, str)
            or content != content.strip()
            or not content
            or len(content) > _MAX_FACT_CHARS
            or len(content.encode("utf-8")) > _MAX_FACT_BYTES
            or any(ord(char) < 32 or ord(char) == 127 for char in content)
            or not isinstance(category, str)
            or category not in _CATEGORIES
            or content in seen
        ):
            raise _error()
        seen.add(content)
        facts.append(MemoryAppendFact(content=content, category=category))
    return MemoryAppendReceiptPolicy(reset_id=reset_id, facts=tuple(facts))


def bind_memory_append_receipt(policy: MemoryAppendReceiptPolicy) -> Token:
    return _current_policy.set(policy)


def reset_memory_append_receipt(token: Token) -> None:
    _current_policy.reset(token)


def current_memory_append_receipt() -> Optional[MemoryAppendReceiptPolicy]:
    return _current_policy.get()
