"""Быстрый авто-ответчик за Второго агента (agent 2) на вопросы Первого агента (component-runner),
чтобы работа не простаивала. Технически — отдельный дешёвый одноразовый вызов (латентность важнее
глубины: Первый агент ждёт синхронно), но по СМЫСЛУ это Второй агент отвечает "во-первых на вопросы
Первого" (см. AGENT_CONTEXT.md раздел 0). Пока идёт ответ — выставлен state/agent2_busy.json, чтобы
assistant.py (разговорный процесс Второго агента) знал, что занят, и не молчал владельцу.

Решает консервативно, по духу проекта. Необратимое (деньги, доступы, продуктовые решения) — эскалирует
явному ответу владельца через NEEDS_OWNER (см. tools.ask_human).
"""
from __future__ import annotations
import datetime as dt
import json

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage

import config
import limits

SYSTEM = (
    "Ты — Второй агент проекта TrendFlow, отвечаешь ЗА ВЛАДЕЛЬЦА на вопросы Первого агента (Opus), который "
    "валидирует компоненты и иногда задаёт вопросы; владелец часто недоступен. Твоя задача — дать КОНКРЕТНОЕ "
    "решение, чтобы работа не останавливалась. Решай консервативно, приоритет — качество и безопасность. "
    "Контекст читай при необходимости (у тебя есть Bash/Read/Grep/Glob): "
    "automation/runner/AGENT_CONTEXT.md (раздел 5 — устройство автоматизации, раздел 7 — решения и уроки), "
    "automation/runner/state/agent1_chat.log (свежие сообщения переписки Первого агента с владельцем — "
    "хвост файла, `tail -80`, чтобы понимать текущий контекст СЕССИИ, а не только вопрос), "
    "automation/runner/state/last_session.md, CLAUDE.md, DataProcessor/docs/COMPONENT_VALIDATION_PROTOCOL.md. "
    "Отвечай на РУССКОМ, коротко, одним конкретным решением/указанием. "
    "git push в репозиторий проекта РАЗРЕШЁН (не требует владельца). "
    "Если вопрос реально требует ЛИЧНО владельца (необратимое: крупная трата денег, удаление важных данных, "
    "выбор весов/доступов, продуктовый приоритет) — ответь строго 'NEEDS_OWNER: <кратко почему>' и ничего больше."
)


def _set_busy(question: str) -> None:
    try:
        config.AGENT2_BUSY_FLAG.write_text(json.dumps(
            {"since": dt.datetime.now().isoformat(), "question": (question or "")[:200]},
            ensure_ascii=False, indent=2))
    except Exception:
        pass


def _clear_busy() -> None:
    try:
        if config.AGENT2_BUSY_FLAG.exists():
            config.AGENT2_BUSY_FLAG.unlink()
    except Exception:
        pass


async def answer(question: str, context: str = "") -> str:
    _set_busy(question)
    try:
        opts = ClaudeAgentOptions(
            model=config.SUPERVISOR_MODEL, system_prompt=SYSTEM, cwd=str(config.REPO_DIR),
            allowed_tools=["Read", "Grep", "Glob", "Bash"], permission_mode="bypassPermissions", max_turns=8,
        )
        prompt = f"Вопрос Первого агента:\n{question}\n\nКонтекст: {context}\n\nДай конкретное решение."
        parts, cost, tin, tout = [], 0.0, 0, 0
        async for msg in query(prompt=prompt, options=opts):
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        parts.append(b.text.strip())
            elif isinstance(msg, ResultMessage):
                cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
                u = getattr(msg, "usage", None)
                if isinstance(u, dict):
                    tin, tout = int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)
        limits.record("supervisor", tin, tout, cost)
        return ("\n".join(parts)).strip() or "NEEDS_OWNER: пустой ответ супервайзера"
    finally:
        _clear_busy()
