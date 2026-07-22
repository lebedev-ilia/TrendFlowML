"""Персистентность Claude Agent SDK session_id между перезапусками процесса.

Проблема (найдена 2026-07-19): deepdive_agent.py/models_agent.py при рестарте (краш, обновление
кода, ручная перезагрузка systemd-юнита) открывали ClaudeSDKClient БЕЗ resume — весь контекст
разговора и текущей автономной задачи терялся, агент начинал заново с FIRST_TASK_MESSAGE/
AUTONOMOUS_KICKOFF_MESSAGE, как будто ничего не делал. Файлы на диске (FINAL_REPORT.md,
DatasetBuilder и т.д.) не терялись, но живой контекст рассуждений — терялся.

Решение: ClaudeAgentOptions.resume=<session_id> заставляет SDK подгрузить историю разговора из
локального стора Claude Code CLI (тот же, что использует `claude --resume`) — агент буквально
помнит, что делал. session_id приходит в каждом AssistantMessage/ResultMessage, пока идёт ход;
сохраняем его на диск при каждом сообщении (дёшево — маленький JSON), читаем при следующем
запуске процесса ИЛИ при внутреннем переоткрытии сессии после обрыва (см. main() обоих агентов).
"""
from __future__ import annotations
import json
import datetime as dt
from pathlib import Path

import config


def _path(agent_name: str) -> Path:
    return config.STATE_DIR / f"{agent_name}_session.json"


def save(agent_name: str, session_id: str | None) -> None:
    if not session_id:
        return
    try:
        _path(agent_name).write_text(
            json.dumps({
                "session_id": session_id,
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }),
            encoding="utf-8",
        )
    except Exception:
        pass


def load(agent_name: str) -> str | None:
    p = _path(agent_name)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("session_id") or None
    except Exception:
        return None
