from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@dataclass
class Level1StateStore:
    """
    MVP durable state for the global scheduler (Level 1).

    We keep:
    - snapshot: state_level1_global.json
    - append-only journal: state_level1_events.jsonl
    """

    state_dir: str

    @property
    def snapshot_path(self) -> str:
        return os.path.join(self.state_dir, "state_level1_global.json")

    @property
    def events_path(self) -> str:
        return os.path.join(self.state_dir, "state_level1_events.jsonl")

    def init(self) -> None:
        if os.path.exists(self.snapshot_path):
            return
        payload = {
            "schema_version": "state_level1_global_v1",
            "updated_at": _utc_iso_now(),
            "runs": {},
        }
        _atomic_write_json(self.snapshot_path, payload)

    def emit_event(self, event: Dict[str, Any]) -> None:
        e = dict(event)
        e.setdefault("ts", _utc_iso_now())
        _append_jsonl(self.events_path, e)

    def update_run(self, run_key: str, patch: Dict[str, Any]) -> None:
        """
        Best-effort snapshot update (single-writer assumption for MVP).
        """
        self.init()
        try:
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                cur = json.load(f) or {}
        except Exception:
            cur = {}
        runs = cur.get("runs") if isinstance(cur, dict) else None
        if not isinstance(runs, dict):
            runs = {}
        prev = runs.get(run_key)
        if not isinstance(prev, dict):
            prev = {}
        prev.update(patch)
        runs[run_key] = prev
        cur = cur if isinstance(cur, dict) else {}
        cur["schema_version"] = "state_level1_global_v1"
        cur["updated_at"] = _utc_iso_now()
        cur["runs"] = runs
        _atomic_write_json(self.snapshot_path, cur)


