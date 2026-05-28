"""Публикация событий прогресса/run в Redis (WebSocket и др.)."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..models import RunLog
from ..services.events import publish_run_event


def _utcnow() -> datetime:
    return datetime.utcnow()


def _append_log(db: Session, run_id: str, level: str, message: str) -> None:
    """Добавляет лог в legacy таблицу run_logs (для обратной совместимости)."""
    db.add(RunLog(run_id=run_id, level=level, message=message))
    db.flush()


def _publish(run_id: str, payload: Dict[str, Any]) -> None:
    """Публикует событие через Redis pubsub."""
    try:
        asyncio.run(publish_run_event(run_id, payload))
    except Exception:
        pass


def _emit_status(
    run_id: str,
    status: str,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    _publish(
        run_id,
        {
            "type": "run.status_changed",
            "run_id": run_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            "payload": {"status": status, "error_code": error_code, "error_message": error_message},
        },
    )


def _emit_stage(run_id: str, stage: str) -> None:
    _publish(
        run_id,
        {
            "type": "run.stage_changed",
            "run_id": run_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            "payload": {"stage": stage},
        },
    )


def _emit_component(
    run_id: str,
    component_name: str,
    status: str,
    empty_reason: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "component_name": component_name,
        "status": status,
    }
    if empty_reason:
        payload["empty_reason"] = empty_reason
    _publish(
        run_id,
        {
            "type": "component.finished" if status in {"ok", "empty", "error"} else "component.started",
            "run_id": run_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        },
    )


def _tail_state_events(run_id: str, path: Path, stop_flag: Dict[str, bool]) -> None:
    """Tail state_events.jsonl и публикует события."""
    offset = 0
    last_processor = None
    while not stop_flag.get("stop"):
        if not path.exists():
            time.sleep(0.5)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    offset = f.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    scope = event.get("scope")
                    if scope == "processor":
                        proc = event.get("processor")
                        status = event.get("status")
                        if proc and status:
                            if proc != last_processor and status == "running":
                                last_processor = proc
                                _emit_stage(run_id, proc)
                            _emit_component(run_id, proc, status)
                    elif scope == "component":
                        comp = event.get("component")
                        status = event.get("status")
                        if comp and status:
                            _emit_component(run_id, comp, status)
        except Exception:
            time.sleep(0.5)
            continue
        time.sleep(0.2)
