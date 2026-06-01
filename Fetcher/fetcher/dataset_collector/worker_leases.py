from __future__ import annotations

import json
import os
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from fetcher.dataset_collector.state import DatasetState, atomic_write_json, file_lock, utcnow


class WorkerLeaseError(RuntimeError):
    pass


def _leases_path(state: DatasetState) -> Path:
    return state.state_dir / "worker_leases.json"


def _read_leases(state: DatasetState) -> dict:
    path = _leases_path(state)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def acquire_lease(
    state: DatasetState,
    *,
    lease_name: str,
    owner: str,
    ttl_seconds: int = 600,
    metadata: dict | None = None,
) -> dict:
    now = utcnow()
    expires_at = now + timedelta(seconds=max(ttl_seconds, 60))
    with file_lock(state.state_dir / "worker_leases.lock"):
        leases = _read_leases(state)
        current = leases.get(lease_name)
        if current:
            current_expires = current.get("expires_at")
            try:
                still_valid = bool(current_expires and now < datetime.fromisoformat(current_expires))
            except ValueError:
                still_valid = False
            if still_valid and current.get("owner") != owner:
                raise WorkerLeaseError(
                    f"lease {lease_name!r} is held by {current.get('owner')} until {current_expires}"
                )
        record = {
            "name": lease_name,
            "owner": owner,
            "pid": os.getpid(),
            "acquired_at": now.isoformat(),
            "heartbeat_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "metadata": metadata or {},
        }
        leases[lease_name] = record
        atomic_write_json(_leases_path(state), leases)
        return record


def heartbeat_lease(
    state: DatasetState,
    *,
    lease_name: str,
    owner: str,
    ttl_seconds: int = 600,
) -> dict:
    now = utcnow()
    with file_lock(state.state_dir / "worker_leases.lock"):
        leases = _read_leases(state)
        current = leases.get(lease_name)
        if not current or current.get("owner") != owner:
            raise WorkerLeaseError(f"lease {lease_name!r} is no longer owned by {owner}")
        current["heartbeat_at"] = now.isoformat()
        current["expires_at"] = (now + timedelta(seconds=max(ttl_seconds, 60))).isoformat()
        current["pid"] = os.getpid()
        leases[lease_name] = current
        atomic_write_json(_leases_path(state), leases)
        return current


def release_lease(state: DatasetState, *, lease_name: str, owner: str) -> None:
    with file_lock(state.state_dir / "worker_leases.lock"):
        leases = _read_leases(state)
        current = leases.get(lease_name)
        if current and current.get("owner") == owner:
            leases.pop(lease_name, None)
            atomic_write_json(_leases_path(state), leases)
