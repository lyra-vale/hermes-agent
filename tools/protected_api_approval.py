"""Fail-closed, opt-in approvals for actions made by an API run.

The normal approval system intentionally supports human-facing session and
permanent choices.  Calendar-style API integrations need a smaller contract:
one exact registered tool call, one exact API run, and one API response.  This
module is deliberately separate from :mod:`tools.approval`; enabling it is an
explicit adapter decision and none of the generic approval caches or choices
are consulted.

Only digests and opaque identity metadata leave this module.  Arguments are
canonicalized for comparison but are never retained in a pending record or
sent in a notification.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


MAX_CANONICAL_ACTION_BYTES = 64 * 1024
MAX_CANONICAL_ACTION_DEPTH = 32
MAX_CANONICAL_STRING_BYTES = 32 * 1024
MAX_IDENTITY_LENGTH = 256
MAX_REDACTED_DESCRIPTION_LENGTH = 256
MAX_PENDING_APPROVAL_RECORDS = 1024
MAX_RETIRED_RUN_TOMBSTONES = 4096
RETIRED_RUN_TOMBSTONE_TTL_SECONDS = 300.0

__all__ = [
    "CanonicalActionError", "ProtectedApprovalError", "ProtectedApiRunApprovalPolicy",
    "ProtectedApiRunApprovalStore", "ProtectedApiRunBinding", "MAX_CANONICAL_ACTION_BYTES", "MAX_CANONICAL_ACTION_DEPTH",
    "MAX_REDACTED_DESCRIPTION_LENGTH", "MAX_PENDING_APPROVAL_RECORDS",
    "MAX_RETIRED_RUN_TOMBSTONES", "RETIRED_RUN_TOMBSTONE_TTL_SECONDS",
    "bind_protected_api_run", "attach_protected_api_run_policy",
    "require_protected_api_run_approval", "verify_protected_dispatch", "canonical_action_json",
    "canonical_action_digest", "canonical_args_json", "canonical_args_digest",
    "normalize_registered_tool_name", "set_current_api_run_context", "reset_current_api_run_context",
    "submit_protected_api_approval", "is_protected_api_run", "retire_protected_api_run",
    "protected_api_run_status_payload",
    "protected_observer_args", "safe_protected_observer_args",
    "protected_api_run_context_active", "protected_api_run_redaction_active",
    "protected_exception_diagnostic",
]

_REQUIRED_APPROVAL_FIELDS = frozenset({"request_id", "action_digest", "choice"})
_ALLOWED_CHOICES = frozenset({"once", "deny"})


class CanonicalActionError(ValueError):
    """The action cannot be represented by the strict canonical JSON contract."""


class ProtectedApprovalError(RuntimeError):
    """A protected approval request was invalid, stale, or could not authorize."""

    def __init__(self, message: str, *, code: str, status: int = 409):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class ProtectedApiRunApprovalPolicy:
    """Allowlist and identity binding for one protected API-run surface.

    ``run_id`` and ``approval_session`` may be omitted when the policy is
    attached from an API-run context.  All three user-facing identity fields
    are required from the adapter; the core never invents or inherits them.
    """

    allowed_tool_names: frozenset[str]
    user_session_id: str
    hermes_session_id: str
    conversation_id: str
    expires_at: float
    run_id: Optional[str] = None
    approval_session: Optional[str] = None
    redacted_description: str = ""

    def __init__(
        self,
        *,
        allowed_tool_names: Iterable[str],
        user_session_id: str,
        hermes_session_id: str,
        conversation_id: str,
        expires_at: float,
        run_id: Optional[str] = None,
        approval_session: Optional[str] = None,
        redacted_description: Optional[str] = None,
    ) -> None:
        names = frozenset(_normalize_allowlist_name(name) for name in allowed_tool_names)
        if not names:
            raise ValueError("protected approval allowlist must not be empty")
        _validate_identity(user_session_id, "user_session_id")
        _validate_identity(hermes_session_id, "hermes_session_id")
        _validate_identity(conversation_id, "conversation_id")
        checked_expiry = _validate_expiry(expires_at)
        if run_id is not None:
            _validate_identity(run_id, "run_id")
        if approval_session is not None:
            _validate_identity(approval_session, "approval_session")
        description = _normalize_redacted_description(redacted_description)
        object.__setattr__(self, "allowed_tool_names", names)
        object.__setattr__(self, "user_session_id", user_session_id)
        object.__setattr__(self, "hermes_session_id", hermes_session_id)
        object.__setattr__(self, "conversation_id", conversation_id)
        object.__setattr__(self, "expires_at", checked_expiry)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "approval_session", approval_session)
        object.__setattr__(self, "redacted_description", description)

    def allows_tool(self, tool_name: str) -> bool:
        """Return whether *tool_name* is an exact, case-sensitive allowlist member."""
        return isinstance(tool_name, str) and tool_name in self.allowed_tool_names

    def for_run(self, *, run_id: str, approval_session: str) -> "ProtectedApiRunApprovalPolicy":
        """Bind the policy to an exact API run without changing its user identities."""
        _validate_identity(run_id, "run_id")
        _validate_identity(approval_session, "approval_session")
        if self.run_id is not None and self.run_id != run_id:
            raise ProtectedApprovalError(
                "Protected policy is bound to a different run", code="approval_run_mismatch"
            )
        if self.approval_session is not None and self.approval_session != approval_session:
            raise ProtectedApprovalError(
                "Protected policy is bound to a different approval session",
                code="approval_session_mismatch",
            )
        return ProtectedApiRunApprovalPolicy(
            allowed_tool_names=self.allowed_tool_names,
            user_session_id=self.user_session_id,
            hermes_session_id=self.hermes_session_id,
            conversation_id=self.conversation_id,
            expires_at=self.expires_at,
            run_id=run_id,
            approval_session=approval_session,
            redacted_description=self.redacted_description,
        )

    def attach(self, **kwargs: Any) -> "ProtectedApiRunBinding":
        """Attach this policy through the narrow adapter-facing API."""
        return attach_protected_api_run_policy(self, **kwargs)

    def require(self, tool_name: str, args: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Require this policy's one-use approval for an exact tool call."""
        return require_protected_api_run_approval(tool_name, args, policy=self, **kwargs)


@dataclass(frozen=True)
class ProtectedApiRunBinding:
    """The policy, store, and safe event callback bound to the current run."""

    policy: ProtectedApiRunApprovalPolicy
    store: "ProtectedApiRunApprovalStore"
    pending_callback: Optional[Callable[[Dict[str, Any]], None]]
    policy_epoch: int


@dataclass(frozen=True)
class _ApiRunContext:
    run_id: str
    approval_session: str
    pending_callback: Optional[Callable[[Dict[str, Any]], None]]


@dataclass
class _PendingApproval:
    run_id: str
    approval_session: str
    request_id: str
    tool_name: str
    args_digest: str
    action_digest: str
    expires_at: float
    policy: ProtectedApiRunApprovalPolicy
    policy_epoch: int
    callback: Optional[Callable[[Dict[str, Any]], None]]
    event: threading.Event
    choice: Optional[str] = None
    consumed: bool = False
    claimed: bool = False
    failure_message: Optional[str] = None


@dataclass(frozen=True)
class _DispatchAuthorization:
    binding: ProtectedApiRunBinding
    request_id: str
    tool_name: str
    args_digest: str
    action_digest: str
    policy_epoch: int


_CURRENT_API_RUN_CONTEXT: contextvars.ContextVar[Optional[_ApiRunContext]] = contextvars.ContextVar(
    "hermes_protected_api_run_context", default=None
)
_CURRENT_BINDING: contextvars.ContextVar[Optional[ProtectedApiRunBinding]] = contextvars.ContextVar(
    "hermes_protected_api_run_binding", default=None
)
_CURRENT_AUTHORIZATION: contextvars.ContextVar[Optional[_DispatchAuthorization]] = contextvars.ContextVar(
    "hermes_protected_api_run_authorization", default=None
)

_ACTIVE_STORE_LOCK = threading.RLock()
_ACTIVE_STORES: Dict[tuple[str, str], set[Any]] = {}
_RETIRED_RUN_TOMBSTONES: Dict[str, "_RetiredRunTombstone"] = {}
_RETIRED_POLICY_TOMBSTONES: Dict[tuple[str, Optional[str]], "_RetiredRunTombstone"] = {}


@dataclass(frozen=True)
class _RetiredRunTombstone:
    """Bounded process-local identity retained for late protected cleanup."""

    run_id: str
    approval_session: Optional[str]
    expires_at: float


def _prune_retired_tombstones_locked(now: Optional[float] = None) -> None:
    """Drop expired/oldest retirement markers; caller holds ``_ACTIVE_STORE_LOCK``."""
    current = time.monotonic() if now is None else now
    for tombstones in (_RETIRED_RUN_TOMBSTONES, _RETIRED_POLICY_TOMBSTONES):
        for key, tombstone in list(tombstones.items()):
            if current >= tombstone.expires_at:
                tombstones.pop(key, None)
        while len(tombstones) > MAX_RETIRED_RUN_TOMBSTONES:
            tombstones.pop(next(iter(tombstones)))


def _mark_run_retired(run_id: str, approval_session: Optional[str] = None) -> None:
    """Leave a bounded terminal tombstone so late attachment cannot revive a run."""
    with _ACTIVE_STORE_LOCK:
        now = time.monotonic()
        _prune_retired_tombstones_locked(now)
        _RETIRED_RUN_TOMBSTONES.pop(run_id, None)
        _RETIRED_RUN_TOMBSTONES[run_id] = _RetiredRunTombstone(
            run_id, approval_session, now + RETIRED_RUN_TOMBSTONE_TTL_SECONDS
        )
        _prune_retired_tombstones_locked(now)


def _mark_policy_retired(run_id: str, approval_session: Optional[str] = None) -> None:
    """Retain redaction identity after one policy generation is retired.

    Unlike a terminal run tombstone this marker does not prevent a deliberate
    equal-policy re-registration; it only keeps copied/unwinding contexts
    protected from falling back to ordinary observer output.
    """
    with _ACTIVE_STORE_LOCK:
        now = time.monotonic()
        _prune_retired_tombstones_locked(now)
        key = (run_id, approval_session)
        _RETIRED_POLICY_TOMBSTONES.pop(key, None)
        _RETIRED_POLICY_TOMBSTONES[key] = _RetiredRunTombstone(
            run_id, approval_session, now + RETIRED_RUN_TOMBSTONE_TTL_SECONDS
        )
        _prune_retired_tombstones_locked(now)


def _run_is_retired(run_id: str) -> bool:
    with _ACTIVE_STORE_LOCK:
        _prune_retired_tombstones_locked()
        return run_id in _RETIRED_RUN_TOMBSTONES


def _run_has_redaction_tombstone(run_id: str, approval_session: Optional[str]) -> bool:
    with _ACTIVE_STORE_LOCK:
        _prune_retired_tombstones_locked()
        return (
            run_id in _RETIRED_RUN_TOMBSTONES
            or (run_id, approval_session) in _RETIRED_POLICY_TOMBSTONES
            or (run_id, None) in _RETIRED_POLICY_TOMBSTONES
            or (
                approval_session is None
                and any(key[0] == run_id for key in _RETIRED_POLICY_TOMBSTONES)
            )
        )


def _clear_current_run_authorization(run_id: str, approval_session: Optional[str] = None) -> None:
    """Drop caller state when the run owning it is retired."""
    try:
        context = _CURRENT_API_RUN_CONTEXT.get()
        binding = _CURRENT_BINDING.get()
        context_matches = context is not None and context.run_id == run_id and (
            approval_session is None or context.approval_session == approval_session
        )
        binding_matches = binding is not None and binding.policy.run_id == run_id and (
            approval_session is None or binding.policy.approval_session == approval_session
        )
        if context_matches or binding_matches:
            _CURRENT_AUTHORIZATION.set(None)
    except BaseException:
        # Retirement is a fail-closed boundary even when ContextVar inspection
        # is unavailable.
        _CURRENT_AUTHORIZATION.set(None)


def _track_active_store(key: tuple[str, str], store: Any) -> None:
    with _ACTIVE_STORE_LOCK:
        _ACTIVE_STORES.setdefault(key, set()).add(store)


def _untrack_active_store(key: tuple[str, str], store: Any) -> None:
    with _ACTIVE_STORE_LOCK:
        stores = _ACTIVE_STORES.get(key)
        if stores is None:
            return
        stores.discard(store)
        if not stores:
            _ACTIVE_STORES.pop(key, None)


def _active_stores_for(run_id: str, approval_session: Optional[str] = None) -> tuple[Any, ...]:
    with _ACTIVE_STORE_LOCK:
        if approval_session is not None:
            return tuple(_ACTIVE_STORES.get((run_id, approval_session), ()))
        return tuple({
            store
            for (stored_run_id, _stored_session), stores in _ACTIVE_STORES.items()
            if stored_run_id == run_id
            for store in stores
        })


def _normalize_redacted_description(value: Optional[str]) -> str:
    """Normalize the server-owned display text without consulting action arguments."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("redacted_description must be a string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("redacted_description contains a control character")
    compact = " ".join(value.split())
    if len(compact) <= MAX_REDACTED_DESCRIPTION_LENGTH:
        return compact
    return compact[: MAX_REDACTED_DESCRIPTION_LENGTH - 1].rstrip() + "…"


def _validate_identity(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_IDENTITY_LENGTH or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string of at most {MAX_IDENTITY_LENGTH} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} contains a control character")
    return value


def _validate_expiry(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expires_at must be a finite number")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0:
        raise ValueError("expires_at must be a positive finite number")
    return checked


def _normalize_allowlist_name(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("protected allowlist entries must be exact trimmed tool names")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("protected allowlist entries cannot contain control characters")
    return value


def normalize_registered_tool_name(tool_name: str) -> str:
    """Validate a case-sensitive registered tool name without alias repair."""
    value = _normalize_allowlist_name(tool_name)
    try:
        from tools.registry import registry

        registered = registry.get_all_tool_names()
    except Exception as exc:  # pragma: no cover - registry import failure is defensive
        raise CanonicalActionError("tool registry is unavailable") from exc
    if value not in set(registered):
        raise CanonicalActionError(f"tool name is not an exact registered tool: {value}")
    return value


def _canonical_value(value: Any, *, depth: int, path: str) -> Any:
    if depth > MAX_CANONICAL_ACTION_DEPTH:
        raise CanonicalActionError("action arguments exceed the maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value.encode("utf-8")) > MAX_CANONICAL_STRING_BYTES:
            raise CanonicalActionError(f"action string at {path} is too large")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalActionError(f"non-finite number at {path} is not allowed")
        return value
    if isinstance(value, list):
        return [_canonical_value(item, depth=depth + 1, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        entries = list(value.items())
        for key, item in entries:
            if not isinstance(key, str):
                raise CanonicalActionError(f"action object key at {path} is not a string")
            if len(key.encode("utf-8")) > MAX_CANONICAL_STRING_BYTES:
                raise CanonicalActionError(f"action object key at {path} is too large")
            result[key] = item
        return {
            key: _canonical_value(result[key], depth=depth + 1, path=f"{path}.{key}")
            for key in sorted(result)
        }
    raise CanonicalActionError(f"unsupported action value at {path}: {type(value).__name__}")


def _dump_canonical(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        encoded.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CanonicalActionError("action cannot be encoded as canonical JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_CANONICAL_ACTION_BYTES:
        raise CanonicalActionError("canonical action exceeds the size limit")
    return encoded


def canonical_args_json(args: Mapping[str, Any]) -> str:
    """Return stable strict JSON for a tool argument object."""
    if not isinstance(args, dict):
        raise CanonicalActionError("tool arguments must be a JSON object")
    return _dump_canonical(_canonical_value(args, depth=0, path="$"))


def canonical_action_json(tool_name: str, args: Mapping[str, Any]) -> str:
    """Return stable strict JSON for the exact registered tool action."""
    normalized_name = normalize_registered_tool_name(tool_name)
    normalized_args = json.loads(canonical_args_json(args))
    return _dump_canonical({"args": normalized_args, "tool_name": normalized_name})


def canonical_args_digest(args: Mapping[str, Any]) -> str:
    """SHA-256 digest of canonical tool arguments."""
    return hashlib.sha256(canonical_args_json(args).encode("utf-8")).hexdigest()


def canonical_action_digest(tool_name: str, args: Mapping[str, Any]) -> str:
    """SHA-256 digest of the exact registered tool name and canonical arguments."""
    return hashlib.sha256(canonical_action_json(tool_name, args).encode("utf-8")).hexdigest()


def _safe_metadata(pending: _PendingApproval) -> Dict[str, Any]:
    """Build the only payload permitted to cross the pending-approval callback."""
    description = pending.policy.redacted_description or f"Approve protected {pending.tool_name} action"
    return {
        "approval_type": "protected_api_run",
        "run_id": pending.run_id,
        "request_id": pending.request_id,
        "tool_name": pending.tool_name,
        "args_digest": pending.args_digest,
        "action_digest": pending.action_digest,
        "expires_at": pending.expires_at,
        "choices": ["once", "deny"],
        "redacted_description": _normalize_redacted_description(description),
    }


class ProtectedApiRunApprovalStore:
    """Thread-safe in-memory ledger for protected API-run approvals."""

    def __init__(
        self,
        *,
        clock: Optional[Callable[[], float]] = None,
        max_records: int = MAX_PENDING_APPROVAL_RECORDS,
    ):
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        self._clock = clock or time.time
        self._max_records = max_records
        self._lock = threading.RLock()
        self._policy_epoch = 0
        self._policies: Dict[
            tuple[str, str], tuple[ProtectedApiRunApprovalPolicy, Optional[Callable], int]
        ] = {}
        self._records: Dict[tuple[str, str, str], _PendingApproval] = {}
        self._action_records: Dict[tuple[str, str, str, str], _PendingApproval] = {}

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    @property
    def pending_record_count(self) -> int:
        """Number of currently pending records held by this in-memory ledger."""
        with self._lock:
            return len(self._records)

    def register_policy(
        self,
        policy: ProtectedApiRunApprovalPolicy,
        *,
        pending_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ProtectedApiRunApprovalPolicy:
        if not policy.run_id or not policy.approval_session:
            raise ValueError("protected policy must be bound to run_id and approval_session")
        now = float(self._clock())
        if now >= policy.expires_at:
            raise ProtectedApprovalError("Protected policy has expired", code="approval_expired")
        if _run_is_retired(policy.run_id):
            raise ProtectedApprovalError(
                "Protected API run has retired", code="approval_run_retired"
            )
        key = (policy.run_id, policy.approval_session)
        # Serialize the tombstone check with active-store tracking. Otherwise a
        # retirement could mark the run and snapshot stores between this check
        # and a delayed registration, allowing the late attach to revive it.
        with _ACTIVE_STORE_LOCK:
            if _run_is_retired(policy.run_id):
                raise ProtectedApprovalError(
                    "Protected API run has retired", code="approval_run_retired"
                )
            with self._lock:
                existing = self._policies.get(key)
                if existing is not None:
                    if existing[0] != policy:
                        raise ProtectedApprovalError(
                            "A different protected policy is already bound to this run",
                            code="approval_policy_conflict",
                        )
                    if pending_callback is not None and existing[1] is not pending_callback:
                        self._policies[key] = (existing[0], pending_callback, existing[2])
                    _track_active_store(key, self)
                    return existing[0]
                self._policy_epoch += 1
                self._policies[key] = (policy, pending_callback, self._policy_epoch)
                _track_active_store(key, self)
        return policy

    def is_protected_run(self, run_id: str, approval_session: Optional[str] = None) -> bool:
        with self._lock:
            if approval_session is not None:
                return (run_id, approval_session) in self._policies
            return any(key[0] == run_id for key in self._policies)

    def binding_for(self, run_id: str, approval_session: str) -> Optional[ProtectedApiRunBinding]:
        with self._lock:
            registered = self._policies.get((run_id, approval_session))
            if registered is None:
                return None
            return ProtectedApiRunBinding(registered[0], self, registered[1], registered[2])

    def is_binding_current(self, binding: ProtectedApiRunBinding) -> bool:
        """Return whether a binding belongs to the active policy generation."""
        run_id = binding.policy.run_id
        approval_session = binding.policy.approval_session
        if not run_id or not approval_session:
            return False
        with self._lock:
            registered = self._policies.get((run_id, approval_session))
            return bool(
                registered is not None
                and registered[0] == binding.policy
                and registered[2] == binding.policy_epoch
            )

    def status_payload(
        self, *, run_id: str, approval_session: str, request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return a safe status envelope built from the stored policy/record."""
        with self._lock:
            record = self._records.get((run_id, approval_session, request_id))
            if record is None or record.consumed:
                return None
            return _safe_metadata(record)

    def _remove_record_locked(self, record: _PendingApproval) -> None:
        record_key = (record.run_id, record.approval_session, record.request_id)
        if self._records.get(record_key) is record:
            self._records.pop(record_key, None)
        action_key = (record.run_id, record.approval_session, record.tool_name, record.args_digest)
        if self._action_records.get(action_key) is record:
            self._action_records.pop(action_key, None)

    def _prune_records_locked(self, now: float) -> None:
        for record in list(self._records.values()):
            if record.consumed:
                self._remove_record_locked(record)
            elif now >= record.expires_at:
                self._expire_locked(record)

    def _expire_locked(self, record: _PendingApproval, *, message: str = "Protected approval expired") -> None:
        if record.consumed:
            self._remove_record_locked(record)
            return
        record.choice = "deny"
        record.consumed = True
        record.failure_message = message
        self._remove_record_locked(record)
        record.event.set()

    def request(
        self,
        policy: ProtectedApiRunApprovalPolicy,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        pending_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Wait for one exact API response and claim its one-use dispatch token."""
        # A fresh request cannot inherit a previous action's final token.
        _CURRENT_AUTHORIZATION.set(None)
        if not policy.run_id or not policy.approval_session:
            return {"approved": False, "message": "BLOCKED: protected policy is not bound to an API run"}
        try:
            normalized_tool = normalize_registered_tool_name(tool_name)
            action_digest = canonical_action_digest(normalized_tool, args)
            args_digest = canonical_args_digest(args)
        except CanonicalActionError as exc:
            return {"approved": False, "message": f"BLOCKED: invalid protected action: {exc}"}
        if not policy.allows_tool(normalized_tool):
            return {"approved": False, "message": f"BLOCKED: tool '{normalized_tool}' is not in the protected allowlist"}
        now = float(self._clock())
        if now >= policy.expires_at:
            return {"approved": False, "message": "BLOCKED: protected approval policy expired"}
        key = (policy.run_id, policy.approval_session)
        with self._lock:
            now = float(self._clock())
            self._prune_records_locked(now)
            registered = self._policies.get(key)
            current_binding = _CURRENT_BINDING.get()
            if (
                registered is None
                or registered[0] != policy
                or current_binding is None
                or current_binding.store is not self
                or current_binding.policy != policy
                or current_binding.policy_epoch != registered[2]
            ):
                return {"approved": False, "message": "BLOCKED: protected policy is not active for this run"}
            action_key = (policy.run_id, policy.approval_session, normalized_tool, args_digest)
            record = self._action_records.get(action_key)
            if record is not None:
                if record.consumed:
                    self._remove_record_locked(record)
                    return {"approved": False, "message": "BLOCKED: protected approval was already used"}
                created = False
            elif len(self._records) >= self._max_records:
                return {"approved": False, "message": "BLOCKED: protected approval capacity is exhausted"}
            else:
                record = _PendingApproval(
                    run_id=policy.run_id,
                    approval_session=policy.approval_session,
                    request_id=secrets.token_urlsafe(18),
                    tool_name=normalized_tool,
                    args_digest=args_digest,
                    action_digest=action_digest,
                    expires_at=min(policy.expires_at, now + max(policy.expires_at - now, 0.0)),
                    policy=policy,
                    policy_epoch=registered[2],
                    callback=pending_callback if pending_callback is not None else registered[1],
                    event=threading.Event(),
                )
                self._records[(record.run_id, record.approval_session, record.request_id)] = record
                self._action_records[action_key] = record
                created = True
        if created:
            callback = record.callback
            if callback is None:
                with self._lock:
                    self._expire_locked(record, message="Protected approval has no API response channel")
            else:
                try:
                    callback(_safe_metadata(record))
                except Exception:
                    with self._lock:
                        self._expire_locked(record, message="Protected approval notification failed")
        while True:
            remaining = float(record.expires_at) - float(self._clock())
            if remaining <= 0:
                with self._lock:
                    self._expire_locked(record)
                break
            if record.event.wait(min(remaining, 0.25)):
                break
        with self._lock:
            if record.choice != "once" or record.claimed:
                return {
                    "approved": False,
                    "message": f"BLOCKED: {record.failure_message or 'protected approval denied'}",
                    "request_id": record.request_id,
                }
            # The API response is consumed before a waiter receives a dispatch token.
            record.claimed = True
            binding = _CURRENT_BINDING.get()
            registered = self._policies.get(key)
            if (
                binding is None
                or binding.policy != policy
                or binding.store is not self
                or registered is None
                or binding.policy_epoch != registered[2]
                or record.policy_epoch != binding.policy_epoch
            ):
                _CURRENT_AUTHORIZATION.set(None)
                return {"approved": False, "message": "BLOCKED: protected approval context changed"}
            _CURRENT_AUTHORIZATION.set(
                _DispatchAuthorization(
                    binding=binding,
                    request_id=record.request_id,
                    tool_name=record.tool_name,
                    args_digest=record.args_digest,
                    action_digest=record.action_digest,
                    policy_epoch=record.policy_epoch,
                )
            )
            return {
                "approved": True,
                "request_id": record.request_id,
                "action_digest": record.action_digest,
            }

    def submit_approval(
        self,
        *,
        run_id: str,
        approval_session: str,
        body: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Atomically consume an exact API approval body."""
        if not isinstance(body, dict) or set(body) != _REQUIRED_APPROVAL_FIELDS:
            raise ProtectedApprovalError(
                "Protected approval body must contain exactly request_id, action_digest, and choice",
                code="invalid_approval_body",
                status=400,
            )
        request_id = body.get("request_id")
        action_digest = body.get("action_digest")
        choice = body.get("choice")
        if not isinstance(request_id, str) or not request_id or len(request_id) > MAX_IDENTITY_LENGTH:
            raise ProtectedApprovalError("request_id is invalid", code="invalid_approval_request", status=400)
        if not isinstance(action_digest, str) or len(action_digest) != 64 or any(
            char not in "0123456789abcdef" for char in action_digest
        ):
            raise ProtectedApprovalError("action_digest is invalid", code="invalid_action_digest", status=400)
        if choice not in _ALLOWED_CHOICES:
            raise ProtectedApprovalError(
                "Protected approval choice must be exactly once or deny",
                code="invalid_approval_choice",
                status=400,
            )
        key = (run_id, approval_session, request_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                raise ProtectedApprovalError(
                    "Protected approval request is not pending",
                    code="approval_not_pending",
                )
            if record.consumed:
                if record.failure_message == "Protected approval expired":
                    raise ProtectedApprovalError("Protected approval request expired", code="approval_expired")
                raise ProtectedApprovalError("Protected approval request was already used", code="approval_replayed")
            if float(self._clock()) >= record.expires_at:
                self._expire_locked(record)
                raise ProtectedApprovalError("Protected approval request expired", code="approval_expired")
            if not hmac.compare_digest(record.action_digest, action_digest):
                raise ProtectedApprovalError(
                    "Protected approval action digest does not match the pending action",
                    code="approval_action_mismatch",
                )
            record.choice = choice
            record.consumed = True
            self._remove_record_locked(record)
            record.event.set()
            return {
                "object": "hermes.protected_api_run_approval",
                "run_id": record.run_id,
                "approval_session": record.approval_session,
                "request_id": record.request_id,
                "action_digest": record.action_digest,
                "choice": choice,
                "resolved": 1,
            }

    def approve(
        self,
        *,
        run_id: str,
        approval_session: str,
        body: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Compatibility spelling for adapters that treat approval as a command."""
        return self.submit_approval(
            run_id=run_id, approval_session=approval_session, body=body
        )

    def retire_run(self, run_id: str, approval_session: Optional[str] = None) -> None:
        """Fail closed and forget policy bindings when an API run retires."""
        _mark_policy_retired(run_id, approval_session)
        _clear_current_run_authorization(run_id, approval_session)
        with _ACTIVE_STORE_LOCK:
            with self._lock:
                keys = [
                    key for key in self._policies
                    if key[0] == run_id and (approval_session is None or key[1] == approval_session)
                ]
                for key in keys:
                    self._policies.pop(key, None)
                    _untrack_active_store(key, self)
                for record in list(self._records.values()):
                    if record.run_id == run_id and (approval_session is None or record.approval_session == approval_session):
                        self._expire_locked(record, message="Protected API run retired")


_DEFAULT_STORE = ProtectedApiRunApprovalStore()


def set_current_api_run_context(
    *,
    run_id: str,
    approval_session: str,
    pending_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Bind the API-run identity used by a profile-local adapter."""
    _validate_identity(run_id, "run_id")
    _validate_identity(approval_session, "approval_session")
    context_token = _CURRENT_API_RUN_CONTEXT.set(
        _ApiRunContext(run_id, approval_session, pending_callback)
    )
    binding_token = _CURRENT_BINDING.set(None)
    authorization_token = _CURRENT_AUTHORIZATION.set(None)
    return context_token, binding_token, authorization_token


def reset_current_api_run_context(tokens) -> None:
    """Reset an API-run context and discard any unconsumed dispatch authorization."""
    try:
        context_token, binding_token, authorization_token = tokens
        _CURRENT_AUTHORIZATION.reset(authorization_token)
        _CURRENT_BINDING.reset(binding_token)
        _CURRENT_API_RUN_CONTEXT.reset(context_token)
    except Exception:
        _CURRENT_AUTHORIZATION.set(None)
        _CURRENT_BINDING.set(None)
        _CURRENT_API_RUN_CONTEXT.set(None)


def _clear_attachment_state_on_failure(function):
    """Make a failed policy attach unable to preserve caller authorization state."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException:
            _CURRENT_AUTHORIZATION.set(None)
            _CURRENT_BINDING.set(None)
            raise
    return wrapped


@_clear_attachment_state_on_failure
def attach_protected_api_run_policy(
    policy: ProtectedApiRunApprovalPolicy,
    *,
    store: Optional[ProtectedApiRunApprovalStore] = None,
    pending_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ProtectedApiRunBinding:
    """Attach a policy to the current API run; intended for a local adapter hook."""
    # Attachment starts a fresh authorization boundary. Clearing first also
    # prevents a failed policy bind from leaving an older token live.
    _CURRENT_AUTHORIZATION.set(None)
    target_store = store or _DEFAULT_STORE
    context = _CURRENT_API_RUN_CONTEXT.get()
    context_created = False
    if context is None:
        if not policy.run_id or not policy.approval_session:
            raise ProtectedApprovalError(
                "Protected policy must be attached inside an API-run context",
                code="approval_context_missing",
            )
        context = _ApiRunContext(policy.run_id, policy.approval_session, pending_callback)
        context_created = True
    if _run_is_retired(context.run_id):
        _CURRENT_AUTHORIZATION.set(None)
        raise ProtectedApprovalError(
            "Protected API run has retired", code="approval_run_retired"
        )
    bound = policy.for_run(run_id=context.run_id, approval_session=context.approval_session)
    callback = pending_callback if pending_callback is not None else context.pending_callback
    registered = target_store.register_policy(bound, pending_callback=callback)
    binding = target_store.binding_for(registered.run_id or "", registered.approval_session or "")
    if binding is None:
        _CURRENT_AUTHORIZATION.set(None)
        raise ProtectedApprovalError(
            "Protected policy could not be resolved after registration",
            code="approval_policy_unavailable",
            status=500,
        )
    current = _CURRENT_BINDING.get()
    if current is not None:
        try:
            current_active = current.store.is_binding_current(current)
        except Exception:
            current_active = False
        if not current_active:
            _CURRENT_AUTHORIZATION.set(None)
            _CURRENT_BINDING.set(None)
            current = None
    if current is not None and current != binding:
        _CURRENT_AUTHORIZATION.set(None)
        raise ProtectedApprovalError(
            "A different protected policy is already active in this API-run context",
            code="approval_policy_conflict",
        )
    if context_created:
        _CURRENT_API_RUN_CONTEXT.set(context)
    _CURRENT_BINDING.set(binding)
    return binding


@contextmanager
def bind_protected_api_run(
    policy: ProtectedApiRunApprovalPolicy,
    *,
    run_id: Optional[str] = None,
    approval_session: Optional[str] = None,
    store: Optional[ProtectedApiRunApprovalStore] = None,
    pending_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Convenience context for adapters and tests that own the API-run boundary."""
    target_run = run_id or policy.run_id
    target_session = approval_session or policy.approval_session
    if not target_run or not target_session:
        raise ProtectedApprovalError(
            "bind_protected_api_run requires run_id and approval_session",
            code="approval_context_missing",
        )
    target_store = store or _DEFAULT_STORE
    context_tokens = set_current_api_run_context(
        run_id=target_run,
        approval_session=target_session,
        pending_callback=pending_callback,
    )
    try:
        binding = attach_protected_api_run_policy(
            policy, store=target_store, pending_callback=pending_callback
        )
        yield binding
    finally:
        # This convenience context owns the local run lifecycle. The API
        # server instead calls retire_protected_api_run when its task retires.
        target_store.retire_run(target_run, target_session)
        reset_current_api_run_context(context_tokens)


def _resolve_current_protected_binding() -> Optional[ProtectedApiRunBinding]:
    """Resolve a policy registered for the current API run before generic fallback."""
    binding = _CURRENT_BINDING.get()
    if binding is not None:
        try:
            if binding.store.is_binding_current(binding):
                return binding
        except Exception:
            pass
        # A retired or replaced generation must never carry its token into a
        # later equal-policy registration in the same copied context.
        _CURRENT_AUTHORIZATION.set(None)
        _CURRENT_BINDING.set(None)
    context = _CURRENT_API_RUN_CONTEXT.get()
    if context is None:
        return None
    stores = _active_stores_for(context.run_id, context.approval_session)
    if not stores:
        return None
    if len(stores) != 1:
        raise ProtectedApprovalError(
            "Protected policy has conflicting active stores",
            code="approval_store_conflict",
            status=500,
        )
    binding = stores[0].binding_for(context.run_id, context.approval_session)
    if binding is None:
        return None
    if binding.pending_callback is None and context.pending_callback is not None:
        stores[0].register_policy(binding.policy, pending_callback=context.pending_callback)
        binding = ProtectedApiRunBinding(binding.policy, stores[0], context.pending_callback, binding.policy_epoch)
    _CURRENT_BINDING.set(binding)
    return binding


def current_protected_api_run_binding() -> Optional[ProtectedApiRunBinding]:
    """Return the current binding for core dispatch and narrow adapters."""
    return _resolve_current_protected_binding()


def _current_protected_redaction_active() -> bool:
    """Return whether this context must remain redacted after policy teardown."""
    binding = _CURRENT_BINDING.get()
    context = _CURRENT_API_RUN_CONTEXT.get()
    if binding is not None or context is not None:
        # A stale binding is still a protected cleanup context. Its store may
        # have removed the policy already, but late callbacks must not become
        # ordinary observers while they unwind.
        return True
    return False


def protected_observer_args(tool_name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return no raw arguments for any action observed inside a protected run."""
    if _current_protected_redaction_active() or _resolve_current_protected_binding() is not None:
        return {}
    return dict(args)


def safe_protected_observer_args(tool_name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed when observer sanitization itself cannot produce a payload."""
    try:
        return protected_observer_args(tool_name, args)
    except Exception:
        # A sanitizer/status lookup failure is ambiguous: never recover by
        # echoing the original action payload to a diagnostic observer.
        return {}


def protected_api_run_context_active() -> bool:
    """Return whether this execution has API-run authorization state attached."""
    return _CURRENT_API_RUN_CONTEXT.get() is not None or _CURRENT_BINDING.get() is not None


def protected_api_run_redaction_active(
    run_id: str, approval_session: Optional[str] = None
) -> bool:
    """Return whether events for *run_id* must use the protected redaction shape.

    This intentionally includes the bounded retirement tombstone.  Event
    callbacks can outlive both the policy ledger and the ContextVar context;
    treating that known terminal run as an ordinary run would expose a late
    tool preview.  It does not make the run active for approval or dispatch.
    """
    try:
        context = _CURRENT_API_RUN_CONTEXT.get()
        if context is not None and context.run_id == run_id and (
            approval_session is None or context.approval_session == approval_session
        ):
            return True
        binding = _CURRENT_BINDING.get()
        if binding is not None and binding.policy.run_id == run_id and (
            approval_session is None or binding.policy.approval_session == approval_session
        ):
            return True
        if _active_stores_for(run_id, approval_session):
            return True
        return _run_has_redaction_tombstone(run_id, approval_session)
    except BaseException:
        # An event redaction probe must never turn an uncertain security state
        # into a raw preview.
        return True


def protected_exception_diagnostic(tool_name: str, exc: BaseException) -> str:
    """Return a handler diagnostic that cannot interpolate protected action data."""
    try:
        active = protected_api_run_context_active()
    except BaseException:
        active = True
    if active:
        return f"Protected tool '{tool_name}' execution failed ({type(exc).__name__})"
    return f"Tool execution failed: {type(exc).__name__}: {exc}"


def has_current_protected_dispatch_authorization() -> bool:
    """Return whether a protected approval token is awaiting final dispatch."""
    return _CURRENT_AUTHORIZATION.get() is not None


def clear_current_protected_dispatch_authorization() -> None:
    """Clear the one-use authorization token after a blocked/finished dispatch."""
    _CURRENT_AUTHORIZATION.set(None)


def _binding_is_active(binding: ProtectedApiRunBinding) -> bool:
    try:
        return bool(binding.store.is_binding_current(binding))
    except Exception:
        return False


def require_protected_api_run_approval(
    tool_name: str,
    args: Mapping[str, Any],
    *,
    policy: Optional[ProtectedApiRunApprovalPolicy] = None,
    store: Optional[ProtectedApiRunApprovalStore] = None,
) -> Dict[str, Any]:
    """Require one protected approval, or no-op when no policy is attached.

    A Calendar adapter may call this from ``pre_tool_call``.  Core dispatch also
    calls it after request/pre-tool transformations and regardless of skip flags,
    so the adapter hook is not an authorization boundary.
    """
    def blocked(message: str) -> Dict[str, Any]:
        clear_current_protected_dispatch_authorization()
        return {"approved": False, "message": message}

    try:
        binding = _resolve_current_protected_binding()
    except Exception:
        return blocked("BLOCKED: protected policy resolution failed")
    if policy is not None:
        if binding is None:
            try:
                binding = attach_protected_api_run_policy(policy, store=store)
            except Exception:
                return blocked("BLOCKED: protected policy attachment failed")
        else:
            try:
                expected = policy.for_run(
                    run_id=binding.policy.run_id or "",
                    approval_session=binding.policy.approval_session or "",
                )
            except Exception:
                expected = None
            if expected != binding.policy:
                return blocked("BLOCKED: protected policy context changed")
    if binding is None:
        # A stale token must not survive a legacy/no-policy early return.
        clear_current_protected_dispatch_authorization()
        context = _CURRENT_API_RUN_CONTEXT.get()
        if context is not None and _run_is_retired(context.run_id):
            return blocked("BLOCKED: protected API run has retired")
        return {"approved": True, "protected": False}
    if not _binding_is_active(binding):
        return blocked("BLOCKED: protected policy is not active for this run")
    if not binding.policy.allows_tool(tool_name):
        return blocked(f"BLOCKED: tool '{tool_name}' is not in the protected allowlist")
    authorization = _CURRENT_AUTHORIZATION.get()
    if authorization is not None:
        if authorization.binding != binding or authorization.policy_epoch != binding.policy_epoch:
            clear_current_protected_dispatch_authorization()
            authorization = None
        else:
            try:
                digest = canonical_action_digest(tool_name, args)
                args_digest = canonical_args_digest(args)
            except CanonicalActionError:
                return blocked("BLOCKED: protected action could not be canonicalized")
            if (
                authorization.tool_name == tool_name
                and hmac.compare_digest(authorization.action_digest, digest)
                and hmac.compare_digest(authorization.args_digest, args_digest)
            ):
                return {
                    "approved": True,
                    "protected": True,
                    "request_id": authorization.request_id,
                    "action_digest": digest,
                }
            return blocked("BLOCKED: protected action changed after approval")
    try:
        return binding.store.request(
            binding.policy,
            tool_name,
            args,
            pending_callback=binding.pending_callback,
        )
    except Exception:
        return blocked("BLOCKED: protected approval request failed")


def verify_protected_dispatch(tool_name: str, args: Mapping[str, Any], *, consume: bool = True) -> bool:
    """Verify the exact action immediately before core/inline dispatch.

    No active protected policy means legacy callers retain their existing path.
    A retired binding, an allowlisted call with no authorization, or any changed
    tool/arguments fails closed. ``consume=False`` is the outer agent preflight;
    the registry boundary uses the default one-use consume operation.
    """
    had_binding = _CURRENT_BINDING.get() is not None
    try:
        binding = _resolve_current_protected_binding()
    except Exception:
        clear_current_protected_dispatch_authorization()
        return False
    if binding is None:
        clear_current_protected_dispatch_authorization()
        return not had_binding
    if not _binding_is_active(binding):
        clear_current_protected_dispatch_authorization()
        return False
    authorization = _CURRENT_AUTHORIZATION.get()
    if authorization is None or authorization.policy_epoch != binding.policy_epoch:
        clear_current_protected_dispatch_authorization()
        return False
    try:
        digest = canonical_action_digest(tool_name, args)
        args_digest = canonical_args_digest(args)
    except CanonicalActionError:
        clear_current_protected_dispatch_authorization()
        return False
    valid = (
        authorization.binding == binding
        and authorization.tool_name == tool_name
        and hmac.compare_digest(authorization.action_digest, digest)
        and hmac.compare_digest(authorization.args_digest, args_digest)
        and binding.policy.allows_tool(tool_name)
    )
    if not valid:
        clear_current_protected_dispatch_authorization()
        return False
    if consume:
        clear_current_protected_dispatch_authorization()
    return True


def is_protected_api_run(run_id: str, approval_session: Optional[str] = None) -> bool:
    """Return whether any active protected policy owns the run."""
    return bool(_active_stores_for(run_id, approval_session))


def protected_api_run_status_payload(
    run_id: str, approval_session: str, request_id: str
) -> Dict[str, Any]:
    """Build the protected approval status from the server-held pending record."""
    stores = _active_stores_for(run_id, approval_session)
    if not stores:
        stores = (_DEFAULT_STORE,)
    if len(stores) != 1:
        raise ProtectedApprovalError(
            "Protected approval status has conflicting policy stores",
            code="approval_status_unavailable",
            status=500,
        )
    payload = stores[0].status_payload(
        run_id=run_id, approval_session=approval_session, request_id=request_id
    )
    if payload is None:
        raise ProtectedApprovalError(
            "Protected approval request is not pending",
            code="approval_not_pending",
        )
    return payload


def submit_protected_api_approval(
    *, run_id: str, approval_session: str, body: Mapping[str, Any]
) -> Dict[str, Any]:
    """Resolve a strict API approval on the run's active policy store."""
    stores = _active_stores_for(run_id, approval_session)
    if not stores:
        return _DEFAULT_STORE.submit_approval(
            run_id=run_id, approval_session=approval_session, body=body
        )
    if len(stores) != 1:
        raise ProtectedApprovalError(
            "Protected approval has conflicting active policy stores",
            code="approval_store_conflict",
            status=500,
        )
    return stores[0].submit_approval(
        run_id=run_id, approval_session=approval_session, body=body
    )


def retire_protected_api_run(run_id: str, approval_session: Optional[str] = None) -> None:
    """Retire every active policy store for a protected run."""
    _mark_run_retired(run_id)
    _clear_current_run_authorization(run_id, approval_session)
    stores = set(_active_stores_for(run_id, approval_session))
    stores.add(_DEFAULT_STORE)
    for store in stores:
        store.retire_run(run_id, approval_session)
