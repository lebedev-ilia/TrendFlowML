"""Реестр активных агентов (heartbeat). Чтобы каждый агент знал, сколько их работает сейчас,
и учитывал, что общий лимит Claude делится между всеми.
"""
from __future__ import annotations
import json
import os
import time

import config

TTL = 90  # сек: агент считается активным, если пинговал за последние TTL


def _load() -> dict:
    if config.AGENTS_FILE.exists():
        try:
            return json.loads(config.AGENTS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save(d: dict) -> None:
    config.AGENTS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def heartbeat(role: str, model: str = "") -> None:
    d = _load()
    d[str(os.getpid())] = {"role": role, "model": model, "last_seen": time.time()}
    # чистим протухшие
    now = time.time()
    d = {k: v for k, v in d.items() if now - v.get("last_seen", 0) < TTL}
    _save(d)


def unregister(role: str = "") -> None:
    d = _load()
    d.pop(str(os.getpid()), None)
    _save(d)


def active() -> list[dict]:
    now = time.time()
    return [v for v in _load().values() if now - v.get("last_seen", 0) < TTL]


def count() -> int:
    return len(active())


def text() -> str:
    a = active()
    if not a:
        return "Активных агентов: 0"
    who = ", ".join(f"{x['role']}({x.get('model', '?')})" for x in a)
    return f"Активных агентов: {len(a)} — {who}"
