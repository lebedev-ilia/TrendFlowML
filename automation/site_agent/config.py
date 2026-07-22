"""Конфигурация Site Agent — отдельный VK-бот, ведущий каркас сайта TrendFlow
(/home/ilya/Рабочий стол/site, NEXT независимый репозиторий от TrendFlowML).

НАМЕРЕННО ОТДЕЛЬНАЯ от automation/runner/config.py и automation/fetcher/config.py — свой .env,
свой VK-токен, своя рабочая директория (SITE_DIR указывает НА ДРУГОЙ репозиторий, не на этот).

.env читается из automation/site_agent/.env (свой файл, отдельный от runner/.env и fetcher/.env).
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except Exception:
    pass

AGENT_DIR = Path(__file__).resolve().parent
AUTOMATION_DIR = AGENT_DIR.parent
TRENDFLOW_REPO_DIR = AUTOMATION_DIR.parent  # .../TrendFlowML — только для чтения контекста
# ml/PROJECT docs (CLAUDE.md, backend/docs/API.md, Models/docs/contracts/, DataProcessor/docs/)
# по прямому требованию владельца (2026-07-22): агент должен сверяться с реальным функционалом
# ML-системы и backend API, а не только с SITE_SPECIFICATION.md.

# --- Рабочая директория агента: САЙТ, не TrendFlowML ---
SITE_DIR = Path(os.environ.get("SITE_DIR", str(Path.home() / "Рабочий стол" / "site"))).expanduser()

STATE_DIR = AGENT_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
OUTBOX_DIR = STATE_DIR / "site_agent_outbox"
OUTBOX_SENT_DIR = OUTBOX_DIR / "sent"
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
OUTBOX_SENT_DIR.mkdir(parents=True, exist_ok=True)

# messenger.py (скопирован из automation/runner/messenger.py) ожидает это имя переменной.
AGENT1_CHAT_LOG = STATE_DIR / "site_agent_chat.log"

# --- VK: свой отдельный бот/токен ---
VK_TOKEN = os.environ.get("VK_TOKEN", "")
VK_TOKEN2 = ""  # messenger.py::send_assistant() — тут не используется, оставлено для совместимости
VK_OWNER_ID = int(os.environ.get("VK_OWNER_ID", "0") or 0)
VK_API_VERSION = os.environ.get("VK_API_VERSION", "5.199")

AGENT_MODEL = os.environ.get("SITE_AGENT_MODEL", "claude-opus-4-8")


def require(name: str, value) -> None:
    if not value:
        raise RuntimeError(f"{name} не задан — проверь automation/site_agent/.env")
