from __future__ import annotations

from enum import Enum


class ProviderMode(str, Enum):
    """Режим выбора источника данных: официальный API и/или SDK."""

    API_FIRST = "api_first"
    API_ONLY = "api_only"
    SDK_ONLY = "sdk_only"
    PARALLEL = "parallel"

    @classmethod
    def from_value(cls, value: str | None, *, default: "ProviderMode" = API_FIRST) -> "ProviderMode":
        if not value:
            return default
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return default


__all__ = ["ProviderMode"]
