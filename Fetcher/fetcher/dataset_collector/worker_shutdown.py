"""Cooperative shutdown flag for dataset collector queue workers."""

from __future__ import annotations

_shutdown_requested = False


def request_shutdown() -> None:
    global _shutdown_requested
    _shutdown_requested = True


def should_stop() -> bool:
    return _shutdown_requested


def reset_shutdown() -> None:
    global _shutdown_requested
    _shutdown_requested = False
