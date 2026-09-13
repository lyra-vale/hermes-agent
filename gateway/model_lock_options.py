"""Canonical, non-secret options persisted in browser model locks."""

from typing import Any, Dict


_MAX_OPTION_KEYS = 3
_MAX_REASONING_KEYS = 2
_MAX_OPTION_STRING_LENGTH = 32
_ALLOWED_OPTIONS = frozenset({"reasoning", "service_tier", "fast"})
_ALLOWED_REASONING = frozenset({"enabled", "effort"})
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})


def _invalid() -> ValueError:
    """Return one value-free validation error for all rejected option shapes."""
    return ValueError("model_options contains unsupported or malformed values")


def _bounded_string(value: Any) -> str:
    if not isinstance(value, str):
        raise _invalid()
    value = value.strip()
    if len(value) > _MAX_OPTION_STRING_LENGTH or any(ord(ch) < 32 for ch in value):
        raise _invalid()
    return value


def canonicalize_model_options(value: Any) -> Dict[str, Any]:
    """Validate and canonicalize lock options without retaining secrets or arbitrary data.

    The persisted shape contains only reasoning ``enabled``/``effort`` and an explicit
    ``service_tier``. ``fast`` is accepted as input for compatibility and becomes
    ``service_tier`` (including an empty tier for an explicit normal/disabled choice).
    """
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > _MAX_OPTION_KEYS:
        raise _invalid()
    if set(value) - _ALLOWED_OPTIONS:
        raise _invalid()

    result: Dict[str, Any] = {}
    reasoning = value.get("reasoning")
    if "reasoning" in value:
        if not isinstance(reasoning, dict) or not reasoning or len(reasoning) > _MAX_REASONING_KEYS:
            raise _invalid()
        if set(reasoning) - _ALLOWED_REASONING:
            raise _invalid()
        enabled = reasoning.get("enabled")
        if "enabled" in reasoning and not isinstance(enabled, bool):
            raise _invalid()
        effort = reasoning.get("effort")
        if "effort" in reasoning:
            effort = _bounded_string(effort).lower()
            if not effort or effort not in _REASONING_EFFORTS:
                raise _invalid()
        if effort == "none" or enabled is False:
            if effort not in (None, "none"):
                raise _invalid()
            result["reasoning"] = {"enabled": False}
        elif effort is not None:
            result["reasoning"] = {"enabled": True, "effort": effort}
        elif enabled is True:
            result["reasoning"] = {"enabled": True}
        else:
            raise _invalid()

    has_tier = "service_tier" in value
    has_fast = "fast" in value
    if has_tier and has_fast:
        raise _invalid()
    if has_tier:
        result["service_tier"] = _bounded_string(value["service_tier"]).lower()
    elif has_fast:
        fast = value["fast"]
        if not isinstance(fast, bool):
            raise _invalid()
        result["service_tier"] = "priority" if fast else ""
    return result
