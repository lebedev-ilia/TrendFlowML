#!/usr/bin/env python3
"""Свободный диалоговый агент для глубокого разбора компонентов (Final Report), см.
DataProcessor/docs/COMPONENT_DEEP_DIVE_PROTOCOL.md.

Это ТОТ ЖЕ БОТ ("TrendFlow Bot", VK_TOKEN), что работал с очередью компонентов через
agent_runner.py — тот же чат, та же история переписки у владельца в VK. НО: это ОТДЕЛЬНЫЙ,
ПРОСТОЙ процесс — свободный диалог вместо очереди/структурированных сессий, и он НЕ требует
assistant.py/supervisor.py (Второго агента). Запускать вместо agent_runner.py, не вместе с ним
(один и тот же VK_TOKEN — если оба слушают long poll одновременно, будут конкурировать за апдейты).

Запуск:
    cd automation/runner && source .venv/bin/activate
    python deepdive_agent.py                  # модель по умолчанию (Opus, как в config.AGENT_MODEL)
    python deepdive_agent.py --model sonnet    # или любой алиас из settings.MODELS

Работа: владелец пишет в VK как в обычном чате (не нужны спецкоманды типа /start-session) —
например «разбери core_clip» или просто вопрос. Одна долгоживущая Claude-сессия держит контекст
всего разговора (в отличие от agent_runner.py, где на каждый компонент — новая сессия). Если
агент решит показать график/визуализацию — сохраняет PNG в state/deepdive_outbox/, и он
автоматически уходит в VK как фото (см. _flush_outbox_photos).
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage,
)

import config
import messenger
import settings

PROTOCOL_PATH = config.REPO_DIR / "DataProcessor" / "docs" / "COMPONENT_DEEP_DIVE_PROTOCOL.md"
OUTBOX_DIR = config.STATE_DIR / "deepdive_outbox"
OUTBOX_SENT_DIR = OUTBOX_DIR / "sent"
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
OUTBOX_SENT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

VK_MAX_LEN = 4000  # запас от лимита VK ~4096 символов на сообщение


def _system_prompt() -> str:
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8") if PROTOCOL_PATH.is_file() else (
        "(файл протокола не найден: " + str(PROTOCOL_PATH) + ")"
    )
    return (
        "Ты — TrendFlow Bot. Раньше ты работал по жёсткой очереди валидации ML-компонентов "
        "(automation/runner/agent_runner.py) — ТЕПЕРЬ это отдельная роль: свободный диалог с "
        "владельцем проекта в VK для ГЛУБОКОГО разбора компонентов DataProcessor (Final Report), "
        "не очередь и не структурированные брифинг/итог-сессии. Владелец просто пишет тебе "
        "как в обычном чате — задача, вопрос, уточнение — отвечай по существу и по-русски.\n\n"
        "Когда владелец называет компонент (например «разбери core_clip» или «сделай отчёт по "
        "asr_extractor») — следуй протоколу ниже целиком: найди всё связанное с компонентом по "
        "всей кодовой базе (не только очевидную папку), прочитай подробно (включая связанные "
        "компоненты, если нужно понять контракты входа/выхода), напиши финальный отчёт в "
        "DataProcessor/docs/component_reports/<component>/FINAL_REPORT.md и отметь прогресс "
        "(статус, дата, обе оценки, ссылка) в DataProcessor/docs/COMPONENT_DEEP_DIVE_CHECKLIST.md.\n\n"
        "=== ПРОТОКОЛ (DataProcessor/docs/COMPONENT_DEEP_DIVE_PROTOCOL.md) ===\n" + protocol + "\n"
        "=== КОНЕЦ ПРОТОКОЛА ===\n\n"
        "ВИЗУАЛИЗАЦИИ ДЛЯ ВЛАДЕЛЬЦА: если можешь построить полезный график/схему (matplotlib, "
        "PIL, что угодно, что даёт PNG) — сохрани файл в " + str(OUTBOX_DIR) + " с осмысленным "
        "уникальным именем. Он АВТОМАТИЧЕСКИ будет отправлен владельцу в VK как фото сразу после "
        "твоего ответа — не нужно ничего для этого вызывать отдельно, просто положи готовый PNG "
        "в эту папку. Не клади туда ничего, кроме изображений, реально готовых к отправке.\n\n"
        "В чате отвечай кратко и по делу — это переписка, а не отчёт-в-VK. Развёрнутое содержание "
        "уходит в FINAL_REPORT.md, в чат — суть, прогресс, вопросы, выводы."
    )


def _flush_outbox_photos() -> None:
    """Отправить в VK все новые PNG/JPG из OUTBOX_DIR, появившиеся с прошлого раза, затем убрать
    их в sent/, чтобы не отправить повторно."""
    for path in sorted(OUTBOX_DIR.iterdir()):
        if path.is_dir() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            messenger.send_photo(path, caption=path.stem.replace("_", " "))
            path.rename(OUTBOX_SENT_DIR / path.name)
        except Exception as e:
            print(f"[deepdive] не удалось отправить фото {path.name}: {e}", flush=True)


def _send_long(text: str) -> None:
    """VK режет сообщения ~4096 символов — режем сами по абзацам, чтобы не обрубало на полуслове."""
    text = text.strip()
    if not text:
        return
    while text:
        if len(text) <= VK_MAX_LEN:
            messenger.send(text)
            return
        cut = text.rfind("\n\n", 0, VK_MAX_LEN)
        if cut <= 0:
            cut = text.rfind(" ", 0, VK_MAX_LEN)
        if cut <= 0:
            cut = VK_MAX_LEN
        messenger.send(text[:cut])
        text = text[cut:].strip()


async def _handle_turn(client: ClaudeSDKClient, text: str) -> None:
    messenger.log_chat("OWNER->", text)
    await client.query(text)
    parts: list[str] = []
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    parts.append(block.text.strip())
        elif isinstance(msg, ResultMessage):
            cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
            print(f"[deepdive] ход завершён, cost=${cost:.4f}", flush=True)
    if parts:
        _send_long("\n\n".join(parts))
    _flush_outbox_photos()


async def main(model_name: str | None) -> None:
    config.require("VK_TOKEN", config.VK_TOKEN)
    config.require("VK_OWNER_ID", config.VK_OWNER_ID)
    model = settings.resolve_model(model_name) if model_name else config.AGENT_MODEL
    print(f"[deepdive] запуск, модель={model}...", flush=True)
    lp = messenger.LongPoll()
    messenger.send(
        "🧭 TrendFlow Bot переключён на свободный разбор компонентов (Final Report), модель "
        f"{model}. Пиши как в обычном чате — спецкоманды не нужны. Например: «разбери core_clip»."
    )
    print("[deepdive] готов, слушаю VK", flush=True)
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=_system_prompt(),
        cwd=str(config.REPO_DIR),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
        permission_mode="bypassPermissions",
        max_turns=500,
    )
    while True:
        try:
            async with ClaudeSDKClient(options=options) as client:
                while True:
                    texts = await asyncio.to_thread(lp.poll_once, 25)
                    for t in texts:
                        try:
                            await _handle_turn(client, t)
                        except Exception as e:
                            print(f"[deepdive] ошибка обработки сообщения: {e}", flush=True)
                            try:
                                messenger.send(f"❗ Ошибка: {e}")
                            except Exception:
                                pass
        except Exception as e:
            print(f"[deepdive] сессия оборвалась, переоткрываю: {e}", flush=True)
            try:
                messenger.send(
                    f"⚠️ Сессия оборвалась ({e}), переоткрываю. Контекст разговора начнётся заново, "
                    "но всё, что уже записано в FINAL_REPORT.md/CHECKLIST, не потеряно."
                )
            except Exception:
                pass
            await asyncio.sleep(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="Алиас модели (opus/sonnet/haiku/fable) или сырой id.")
    args = ap.parse_args()
    try:
        asyncio.run(main(args.model))
    except KeyboardInterrupt:
        print("\n[deepdive] стоп")
