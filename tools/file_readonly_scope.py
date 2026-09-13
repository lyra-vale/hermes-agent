"""Run-local confinement for API ``file_readonly`` tools.

The policy is intentionally a ContextVar: API runs execute in a shared process and
may overlap, so a process-global root list would let one run affect another.  The
API adapter validates and captures server-owned roots before admission; this module
only binds that captured tuple and applies canonical realpath/commonpath checks.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterable, Optional


class FileReadonlyPolicyError(ValueError):
    """A read-only root policy or path cannot be used safely."""


_FILE_READONLY_ROOTS: ContextVar[Optional[tuple[str, ...]]] = ContextVar(
    "file_readonly_roots", default=None
)


def _canonical_directory(value: Any) -> str:
    if not isinstance(value, (str, os.PathLike)) or isinstance(value, (bytes, bytearray)):
        raise FileReadonlyPolicyError("file_readonly_roots must contain absolute directory paths")
    raw = os.fspath(value)
    if isinstance(raw, bytes) or not raw or not os.path.isabs(raw):
        raise FileReadonlyPolicyError("file_readonly_roots must contain absolute directory paths")
    try:
        canonical = os.path.realpath(raw)
        if not os.path.isabs(canonical) or not os.path.isdir(canonical):
            raise FileReadonlyPolicyError("file_readonly_roots must contain existing directories")
        # realpath() normally does not stat every component on all platforms.  A
        # separate stat makes permission/disappearing-root failures fail closed.
        os.stat(canonical)
    except FileReadonlyPolicyError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise FileReadonlyPolicyError("file_readonly_roots could not be resolved") from exc
    return canonical


def resolve_file_readonly_roots(values: Any) -> tuple[str, ...]:
    """Validate and canonicalize server-owned roots, failing closed on any error."""
    if not isinstance(values, (list, tuple)) or isinstance(values, (str, bytes)) or not values:
        raise FileReadonlyPolicyError("file_readonly requires at least one valid absolute root")
    roots: list[str] = []
    for value in values:
        roots.append(_canonical_directory(value))
    # Preserve order for deterministic diagnostics while avoiding redundant work.
    return tuple(dict.fromkeys(roots))


def bind_file_readonly_roots(values: Iterable[str]) -> Token:
    """Bind validated roots for one synchronous run and return its reset token."""
    return _FILE_READONLY_ROOTS.set(resolve_file_readonly_roots(values))


def reset_file_readonly_roots(token: Token) -> None:
    _FILE_READONLY_ROOTS.reset(token)


def current_file_readonly_roots() -> Optional[tuple[str, ...]]:
    """Return the current run policy; ``None`` means ordinary file behavior."""
    return _FILE_READONLY_ROOTS.get()


def enforce_file_readonly_path(path: str | os.PathLike[str]) -> Optional[str]:
    """Return a canonical allowed path, or ``None`` when no policy is bound.

    ``realpath`` resolves both ``..`` and symlinks before ``commonpath``.  Any
    resolution/type/commonpath failure raises rather than falling back to a raw
    path, which is the security boundary's fail-closed behavior.
    """
    roots = _FILE_READONLY_ROOTS.get()
    if roots is None:
        return None
    if not roots:
        raise FileReadonlyPolicyError("file_readonly has no configured roots")
    try:
        raw = os.fspath(path)
        if isinstance(raw, bytes) or not raw:
            raise FileReadonlyPolicyError("file path could not be resolved")
        canonical = os.path.realpath(raw)
        if not os.path.isabs(canonical):
            raise FileReadonlyPolicyError("file path could not be resolved")
        if not any(os.path.commonpath((canonical, root)) == root for root in roots):
            raise FileReadonlyPolicyError("path is outside configured read-only roots")
        return canonical
    except FileReadonlyPolicyError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise FileReadonlyPolicyError("file path could not be resolved") from exc


__all__ = [
    "FileReadonlyPolicyError",
    "bind_file_readonly_roots",
    "current_file_readonly_roots",
    "enforce_file_readonly_path",
    "reset_file_readonly_roots",
    "resolve_file_readonly_roots",
]
