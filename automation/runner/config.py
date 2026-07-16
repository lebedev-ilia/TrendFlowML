"""Конфигурация раннера. Значения берутся из окружения (.env)."""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except Exception:
    pass  # dotenv не обязателен, если переменные уже в окружении

# --- Пути проекта ---
RUNNER_DIR = Path(__file__).resolve().parent
AUTOMATION_DIR = RUNNER_DIR.parent
REPO_DIR = AUTOMATION_DIR.parent  # .../TrendFlowML
CHECKLIST = REPO_DIR / "DataProcessor" / "docs" / "COMPONENT_VALIDATION_CHECKLIST.md"

STATE_DIR = RUNNER_DIR / "state"
STATE_DIR.mkdir(exist_ok=True)
PROGRESS_DIR = STATE_DIR / "progress"
PROGRESS_DIR.mkdir(exist_ok=True)
QUEUE_FILE = STATE_DIR / "queue.json"
CLAIM_FILE = STATE_DIR / "claim.json"
DONE_FILE = STATE_DIR / "done.json"              # компоненты, закрытые раннером (advance очереди)
SETTINGS_FILE = STATE_DIR / "settings.json"      # изменяемые настройки (/settings)
LAST_SESSION = STATE_DIR / "last_session.md"     # краткий контекст прошлой сессии (для /ssc и авто-продолжения)
USAGE_LOG = STATE_DIR / "usage_events.csv"       # общий лог токенов ВСЕХ агентов (лимиты Claude)
LIMITS_SNAPSHOT = STATE_DIR / "limits_snapshot.json"  # последние точные значения (браузер claude.ai)
AGENTS_FILE = STATE_DIR / "agents.json"          # реестр активных агентов (heartbeat)
SPEND_LOG = STATE_DIR / "spend_log.csv"          # стоимость модели по компонентам (учёт, не лимит)
POD_LEDGER = STATE_DIR / "pod_ledger.csv"        # аренда подов: gpu, цена/час, время, $
SESSIONS_LOG = STATE_DIR / "sessions.csv"        # все сессии: время, компонент, модель$, gpu$
OPEN_PODS = STATE_DIR / "open_pods.json"         # текущие поднятые поды (для начисления)
PAUSE_FLAG = STATE_DIR / "PAUSED"                # общий флаг ручной/аварийной паузы (владелец, крэш, агент 2)
AGENT1_CHAT_LOG = STATE_DIR / "agent1_chat.log"  # исходящие/входящие сообщения чата агента 1 (для агента 2)
AGENT2_CHAT_LOG = STATE_DIR / "agent2_chat.log"  # исходящие/входящие сообщения чата агента 2 (для владельца/истории)
AGENT2_BUSY_FLAG = STATE_DIR / "agent2_busy.json"        # агент 2 сейчас отвечает агенту 1 (не владельцу)
ASSISTANT_STOP_REQUEST = STATE_DIR / "assistant_stop_request.json"   # агент 2 просит аккуратный стоп агента 1
ASSISTANT_START_REQUEST = STATE_DIR / "assistant_start_request.json"  # агент 2 просит рестарт агента 1
LIVE_NOTE_FILE = STATE_DIR / "live_note.json"    # сообщение агенту 1 БЕЗ остановки (см. hooks.py)
HOOK_DECISIONS_LOG = STATE_DIR / "hook_decisions.log"     # вердикты супервайзера по опасным bash-командам

# --- VK ---
VK_TOKEN = os.environ.get("VK_TOKEN", "")
VK_OWNER_ID = int(os.environ.get("VK_OWNER_ID", "0") or 0)
VK_API_VERSION = os.environ.get("VK_API_VERSION", "5.199")
# Второй бот-ассистент (разговорный помощник, полный доступ).
VK_TOKEN2 = os.environ.get("VK_TOKEN2", "")
# Модель второго агента (отвечает Первому + мониторит/чинит/перезапускает + чат с владельцем).
# Sonnet по умолчанию — ему нужно судить о качестве работы и править код, не только болтать.
ASSISTANT_MODEL = os.environ.get("ASSISTANT_MODEL", "claude-sonnet-4-6")
# Быстрая дешёвая модель ТОЛЬКО для мгновенных авто-ответов на вопросы Первого агента (supervisor.py) —
# латентность важнее глубины, Первый агент ждёт синхронно.
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "claude-haiku-4-5-20251001")

# --- RunPod ---
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_API = "https://rest.runpod.io/v1"
# Необязательно: закрепить конкретный под для start/status (тот, у которого наш Network Volume).
# Если пусто — берётся первый остановленный под из списка.
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "").strip()
# Network Volume, к датацентру которого привязаны поды (для точной проверки цен/наличия GPU).
RUNPOD_VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "").strip()
RUNPOD_VOLUME_NAME = os.environ.get("RUNPOD_VOLUME_NAME", "polite_magenta_coral").strip()
RUNPOD_GPUS_JSON = STATE_DIR / "runpod_gpus.json"  # точные цены/наличие GPU (Chrome-парсер console.runpod.io)

# --- Лимиты (сознательно МИНИМАЛЬНЫЕ — см. automation/runner/AGENT_CONTEXT.md раздел 0) ---
# Единственные лимиты системы: (1) % Claude 5ч/неделя (settings: limit_pct_stop/limit_pct_hard),
# (2) баланс RunPod (предупреждение, НЕ стоп — settings: runpod_balance_warn_usd),
# (3) ценовой потолок пода (settings: max_pod_hourly_usd / max_pod_hourly_usd_short).
# Дневной $-бюджет НАМЕРЕННО убран (раньше DAILY_BUDGET_USD/GPU_SAFETY) — не блокирует работу агента.
#
# Порог бездействия пода (сек): предупреждение и авто-стоп (в 3× от него). Это НЕ лимит бюджета —
# просто не оставляем GPU крутиться без дела.
POD_IDLE_STOP_SEC = int(os.environ.get("POD_IDLE_STOP_SEC", "1200"))
# Цена/час-ОЦЕНКА для существующих подов, если не удалось определить по типу GPU (фолбэк, не лимит).
POD_HOURLY_USD = float(os.environ.get("POD_HOURLY_USD", "0.25"))

# --- Модель ---
# Аутентификация идёт через bundled Claude Code CLI (твой Max-логин). Модель по умолчанию:
AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")
# Порт remote-debugging твоего Chrome для парсера лимитов (пусто = отдельный профиль).
CLAUDE_CDP_PORT = os.environ.get("CLAUDE_CDP_PORT", "").strip()
MAX_TURNS_PER_COMPONENT = int(os.environ.get("MAX_TURNS_PER_COMPONENT", "200"))


def require(name: str, value) -> None:
    if not value:
        raise SystemExit(f"[config] Не задана переменная окружения {name}. Заполни .env (см. .env.example).")
