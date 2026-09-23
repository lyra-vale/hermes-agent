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

__all__ = [
    "CanonicalActionError", "ProtectedApprovalError", "ProtectedApiRunApprovalPolicy",
    "ProtectedApiRunApprovalStore", "ProtectedApiRunBinding", "MAX_CANONICAL_ACTION_BYTES", "MAX_CANONICAL_ACTION_DEPTH",
    "bind_protected_api_run", "attach_protected_api_run_policy",
    "require_protected_api_run_approval", "verify_protected_dispatch", "canonical_action_json",
    "canonical_action_digest", "canonical_args_json", "canonical_args_digest",
    "normalize_registered_tool_name", "set_current_api_run_context", "reset_current_api_run_context",
    "submit_protected_api_approval", "is_protected_api_run", "retire_protected_api_run",
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
        object.__setattr__(self, "allowed_tool_names", names)
        object.__setattr__(self, "user_session_id", user_session_id)
        object.__setattr__(self, "hermes_session_id", hermes_session_id)
        object.__setattr__(self, "conversation_id", conversation_id)
        object.__setattr__(self, "expires_at", checked_expiry)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "approval_session", approval_session)

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


_CURRENT_API_RUN_CONTEXT: contextvars.ContextVar[Optional[_ApiRunContext]] = contextvars.ContextVar(
    "hermes_protected_api_run_context", default=None
)
_CURRENT_BINDING: contextvars.ContextVar[Optional[ProtectedApiRunBinding]] = contextvars.ContextVar(
    "hermes_protected_api_run_binding", default=None
)
_CURRENT_AUTHORIZATION: contextvars.ContextVar[Optional[_DispatchAuthorization]] = contextvars.ContextVar(
    "hermes_protected_api_run_authorization", default=None
)


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
    return {
        "approval_type": "protected_api_run",
        "run_id": pending.run_id,
        "request_id": pending.request_id,
        "tool_name": pending.tool_name,
        "args_digest": pending.args_digest,
        "action_digest": pending.action_digest,
        "expires_at": pending.expires_at,
        "choices": ["once", "deny"],
    }


class ProtectedApiRunApprovalStore:
    """Thread-safe in-memory ledger for protected API-run approvals."""

    def __init__(self, *, clock: Optional[Callable[[], float]] = None):
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._policies: Dict[tuple[str, str], tuple[ProtectedApiRunApprovalPolicy, Optional[Callable]]] = {}
        self._records: Dict[tuple[str, str, str], _PendingApproval] = {}
        self._action_records: Dict[tuple[str, str, str, str], _PendingApproval] = {}

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

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
        key = (policy.run_id, policy.approval_session)
        with self._lock:
            existing = self._policies.get(key)
            if existing is not None:
                if existing[0] != policy:
                    raise ProtectedApprovalError(
                        "A different protected policy is already bound to this run",
                        code="approval_policy_conflict",
                    )
                if pending_callback is not None and existing[1] is not pending_callback:
                    self._policies[key] = (existing[0], pending_callback)
                return existing[0]
            self._policies[key] = (policy, pending_callback)
        return policy

    def is_protected_run(self, run_id: str, approval_session: Optional[str] = None) -> bool:
        with self._lock:
            if approval_session is not None:
                return (run_id, approval_session) in self._policies
            return any(key[0] == run_id for key in self._policies)

    def _expire_locked(self, record: _PendingApproval, *, message: str = "Protected approval expired") -> None:
        if record.consumed:
            return
        record.choice = "deny"
        record.consumed = True
        record.failure_message = message
        self._action_records.pop(
            (record.run_id, record.approval_session, record.tool_name, record.args_digest), None
        )
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
            registered = self._policies.get(key)
            if registered is None or registered[0] != policy:
                return {"approved": False, "message": "BLOCKED: protected policy is not active for this run"}
            action_key = (policy.run_id, policy.approval_session, normalized_tool, args_digest)
            record = self._action_records.get(action_key)
            if record is not None:
                if record.consumed:
                    return {"approved": False, "message": "BLOCKED: protected approval was already used"}
                created = False
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
            if binding is None or binding.policy != policy or binding.store is not self:
                return {"approved": False, "message": "BLOCKED: protected approval context changed"}
            _CURRENT_AUTHORIZATION.set(
                _DispatchAuthorization(
                    binding=binding,
                    request_id=record.request_id,
                    tool_name=record.tool_name,
                    args_digest=record.args_digest,
                    action_digest=record.action_digest,
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
            self._action_records.pop(
                (record.run_id, record.approval_session, record.tool_name, record.args_digest), None
            )
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
        with self._lock:
            keys = [
                key for key in self._policies
                if key[0] == run_id and (approval_session is None or key[1] == approval_session)
            ]
            for key in keys:
                self._policies.pop(key, None)
            for record in self._records.values():
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


def attach_protected_api_run_policy(
    policy: ProtectedApiRunApprovalPolicy,
    *,
    store: Optional[ProtectedApiRunApprovalStore] = None,
    pending_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ProtectedApiRunBinding:
    """Attach a policy to the current API run; intended for a local adapter hook."""
    context = _CURRENT_API_RUN_CONTEXT.get()
    if context is None:
        if not policy.run_id or not policy.approval_session:
            raise ProtectedApprovalError(
                "Protected policy must be attached inside an API-run context",
                code="approval_context_missing",
            )
        context = _ApiRunContext(policy.run_id, policy.approval_session, pending_callback)
        _CURRENT_API_RUN_CONTEXT.set(context)
    bound = policy.for_run(run_id=context.run_id, approval_session=context.approval_session)
    target_store = store or _DEFAULT_STORE
    callback = pending_callback if pending_callback is not None else context.pending_callback
    registered = target_store.register_policy(bound, pending_callback=callback)
    binding = ProtectedApiRunBinding(registered, target_store, callback)
    current = _CURRENT_BINDING.get()
    if current is not None and current != binding:
        raise ProtectedApprovalError(
            "A different protected policy is already active in this API-run context",
            code="approval_policy_conflict",
        )
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


def current_protected_api_run_binding() -> Optional[ProtectedApiRunBinding]:
    """Return the current binding for core dispatch and narrow adapters."""
    return _CURRENT_BINDING.get()


def protected_observer_args(tool_name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return no raw arguments while a protected action is being observed."""
    binding = _CURRENT_BINDING.get()
    if binding is not None and binding.policy.allows_tool(tool_name):
        return {}
    return dict(args)


def has_current_protected_dispatch_authorization() -> bool:
    """Return whether a protected approval token is awaiting final dispatch."""
    return _CURRENT_AUTHORIZATION.get() is not None


def clear_current_protected_dispatch_authorization() -> None:
    """Clear the one-use authorization token after a blocked/finished dispatch."""
    _CURRENT_AUTHORIZATION.set(None)


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
    binding = _CURRENT_BINDING.get()
    if policy is not None:
        if binding is None:
            binding = attach_protected_api_run_policy(policy, store=store)
        else:
            try:
                expected = policy.for_run(
                    run_id=binding.policy.run_id or "",
                    approval_session=binding.policy.approval_session or "",
                )
            except Exception:
                expected = None
            if expected != binding.policy:
                return {"approved": False, "message": "BLOCKED: protected policy context changed"}
    if binding is None:
        return {"approved": True, "protected": False}
    if not binding.policy.allows_tool(tool_name):
        return {"approved": False, "message": f"BLOCKED: tool '{tool_name}' is not in the protected allowlist"}
    authorization = _CURRENT_AUTHORIZATION.get()
    if authorization is not None and authorization.binding == binding:
        try:
            digest = canonical_action_digest(tool_name, args)
            args_digest = canonical_args_digest(args)
        except CanonicalActionError:
            clear_current_protected_dispatch_authorization()
            return {"approved": False, "message": "BLOCKED: protected action could not be canonicalized"}
        if (
            authorization.tool_name == tool_name
            and hmac.compare_digest(authorization.action_digest, digest)
            and hmac.compare_digest(authorization.args_digest, args_digest)
        ):
            return {"approved": True, "protected": True, "request_id": authorization.request_id, "action_digest": digest}
        clear_current_protected_dispatch_authorization()
        return {"approved": False, "message": "BLOCKED: protected action changed after approval"}
    return binding.store.request(
        binding.policy,
        tool_name,
        args,
        pending_callback=binding.pending_callback,
    )


def verify_protected_dispatch(tool_name: str, args: Mapping[str, Any], *, consume: bool = True) -> bool:
    """Verify the exact action immediately before core/inline dispatch.

    No active protected policy means legacy callers retain their existing path.
    An allowlisted call with no authorization, or any changed tool/arguments,
    fails closed.  ``consume=False`` is the outer agent preflight; the registry
    boundary uses the default one-use consume operation.
    """
    binding = _CURRENT_BINDING.get()
    if binding is None:
        return True
    authorization = _CURRENT_AUTHORIZATION.get()
    if authorization is None:
        return not binding.policy.allows_tool(tool_name)
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
    """Return whether the default API approval ledger owns a protected run."""
    return _DEFAULT_STORE.is_protected_run(run_id, approval_session)


def submit_protected_api_approval(
    *, run_id: str, approval_session: str, body: Mapping[str, Any]
) -> Dict[str, Any]:
    """Resolve a strict API approval on the default server ledger."""
    return _DEFAULT_STORE.submit_approval(
        run_id=run_id, approval_session=approval_session, body=body
    )


def retire_protected_api_run(run_id: str, approval_session: Optional[str] = None) -> None:
    """Retire a protected run from the default server ledger."""
    _DEFAULT_STORE.retire_run(run_id, approval_session)
