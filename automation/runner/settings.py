"""Изменяемые в рантайме настройки (через VK /settings). Персист в state/settings.json.

Значения по умолчанию берутся из config (env). budget/agent_runner читают отсюда, чтобы /settings
и /model менялись без перезапуска.
"""
from __future__ import annotations
import json

import config

# Единственные лимиты системы (см. AGENT_CONTEXT.md раздел 0): % Claude (limit_pct_*), цена пода
# (max_pod_hourly_usd*), баланс RunPod (runpod_balance_warn_usd, ТОЛЬКО предупреждение). НЕТ дневного
# $-бюджета — раньше был (daily_budget_usd/soft_limit_frac/gpu_safety), сознательно убран.
_DEFAULTS = {
    "pod_hourly_usd": config.POD_HOURLY_USD,  # оценка цены пода, если API её не отдал (не лимит)
    "model": config.AGENT_MODEL,
    "five_hour_token_cap": 0,   # 0 = неизвестно (Anthropic не публикует); можно задать ориентир
    "weekly_token_cap": 0,
    "limit_pct_stop": 95,       # рабочий лимит: с него — только сворачивание (сохранить/погасить/закончить)
    "limit_pct_hard": 97,       # после сворачивания жёсткий стоп; 97-100% — неприкосновенный резерв (3%)
    "limits_poll_sec": 30,      # частота фонового опроса claude.ai (каждые 30с)
    "auto_resume": 1,           # 1 = после стопа по ЛИМИТУ раннер сам ждёт сброса и продолжает (без /sas)
    "auto_answer": 1,           # 1 = Второй агент отвечает за владельца, чтобы Первый не простаивал
    "auto_answer_wait_sec": 180,  # окно на переопределение владельцем перед авто-ответом
    "max_pod_hourly_usd": 0.30,       # ЖЁСТКИЙ гейт для обычной работы: не брать под дороже $/час
    "max_pod_hourly_usd_short": 0.60,  # гейт для КОРОТКИХ/быстрых прогонов (тесты, smoke) — до $/час
    "runpod_balance_warn_usd": 1.0,   # баланс RunPod ниже этого — предупреждение в VK (не стоп)
    "autonomous_pods": 1,       # 1 = поды создаются/гасятся автономно (без подтверждения в VK)
    "assistant_monitor_interval_sec": 1200,  # раз в сколько сек Второй агент проверяет Первого (20 мин)
    "assistant_monitor_enabled": 1,          # 1 = периодический контроль включён
}

# Дружелюбные алиасы моделей -> строка модели для SDK.
MODELS = {
    "opus": "claude-opus-4-8", "opus 4.8": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6", "sonnet 4.6": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001", "haiku 4.5": "claude-haiku-4-5-20251001",
    "fable": "fable", "fable 5": "fable",
}

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        d = dict(_DEFAULTS)
        if config.SETTINGS_FILE.exists():
            try:
                d.update(json.loads(config.SETTINGS_FILE.read_text()))
            except Exception:
                pass
        _cache = d
    return _cache


def _save() -> None:
    config.SETTINGS_FILE.write_text(json.dumps(_load(), ensure_ascii=False, indent=2))


def get(key: str):
    return _load().get(key, _DEFAULTS.get(key))


def set_(key: str, value) -> None:
    _load()[key] = value
    _save()


def resolve_model(name: str) -> str:
    """Дружелюбное имя/алиас/сырой id -> строка модели."""
    n = (name or "").strip().lower()
    return MODELS.get(n, name.strip())


def model() -> str:
    return str(get("model"))


def as_text() -> str:
    d = _load()
    return (
        f"⚙️ Настройки (лимиты):\n"
        f"• limit_pct_stop/hard = {d['limit_pct_stop']}/{d['limit_pct_hard']}% (Claude 5ч/неделя)\n"
        f"• max_pod_hourly_usd = ${d['max_pod_hourly_usd']}/ч (обычная работа), "
        f"${d['max_pod_hourly_usd_short']}/ч (короткие/быстрые прогоны)\n"
        f"• runpod_balance_warn_usd = ${d['runpod_balance_warn_usd']} (предупреждение, не стоп)\n"
        f"• pod_hourly_usd = {d['pod_hourly_usd']} (оценка цены, фолбэк)\n"
        f"• model = {d['model']}\n"
        f"Изменить: /settings <ключ> <значение>. Модель: /model <opus|sonnet|haiku|fable|...>"
    )
