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

АВТОПРОДОЛЖЕНИЕ ОЧЕРЕДИ (добавлено 2026-07-18): раньше после каждого компонента владелец вручную
писал «Продолжай со следующим компонентом». Теперь это делает сам скрипт — после каждого хода
сверяет DataProcessor/docs/COMPONENT_DEEP_DIVE_CHECKLIST.md ДО и ПОСЛЕ (не читает намерения LLM,
чистый факт: появилась ли новая строка ✅, которой не было). Если появилась и в чеклисте ещё есть
⬜ — сам шлёт себе то же сообщение «Продолжай со следующим компонентом» и продолжает, пока
компоненты не кончатся. При запуске, если в чеклисте уже есть ⬜, сразу стартует сам — не нужно
даже первое «разбери X». Если владелец пишет что-то своё, пока идёт автопродолжение — оно всегда
в приоритете (см. _run_component_turn: перед авто-сообщением коротко проверяется VK на реальные
сообщения от владельца).

ПОЛНАЯ АВТОНОМИЯ ПО ВСЕМ РАБОТАМ (добавлено 2026-07-18): владелец решил — тот же бот (TrendFlow Bot)
ведёт ВСЕ дальнейшие работы автономно, не только разбор компонентов. Теперь у него есть те же
инструменты управления GPU-подом, что и у agent_runner.py (pod_control/manager/budget_status/
limits_status через tools.build_server(), см. automation/runner/tools.py) — может сам поднимать/
гасить поды и пересчитывать батчи. Когда очередь разбора компонентов пуста (см. §комментарий выше),
скрипт сам направляет его на автономные задачи из DataProcessor/docs/
COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md (§7.1/§9) и DataProcessor/docs/
AUTOMATED_TEST_CORPUS_PROTOCOL.md. Правило из системного промпта: к владельцу — только за
РЕАЛЬНЫМИ решениями (данные, деньги/лицензии, юридические вопросы, продуктовые развилки), не за
подтверждением рутинных технических действий."""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions, HookMatcher, AssistantMessage, TextBlock, ResultMessage,
)

import config
import messenger
import settings
import tools
import hooks

PROTOCOL_PATH = config.REPO_DIR / "DataProcessor" / "docs" / "COMPONENT_DEEP_DIVE_PROTOCOL.md"
CHECKLIST_PATH = config.REPO_DIR / "DataProcessor" / "docs" / "COMPONENT_DEEP_DIVE_CHECKLIST.md"
PORTFOLIO_PATH = config.REPO_DIR / "DataProcessor" / "docs" / "COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md"
CORPUS_PROTOCOL_PATH = config.REPO_DIR / "DataProcessor" / "docs" / "AUTOMATED_TEST_CORPUS_PROTOCOL.md"

AUTONOMOUS_KICKOFF_MESSAGE = (
    "Разбор компонентов из чеклиста завершён. Переходи к автономной работе: сначала пункты §7.1 "
    "и §9 из DataProcessor/docs/COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md (пересчёт батчей после "
    "уже сделанных фиксов, выравнивание штампов video_pacing/voice_quality_extractor, консолидация "
    "избыточных фич — предложи план и делай), затем подбор реального тест-корпуса по "
    "DataProcessor/docs/AUTOMATED_TEST_CORPUS_PROTOCOL.md. Работай сама, без вопросов на каждый шаг — "
    "спрашивай меня только если решение реально не в твоей власти (данные, деньги/лицензии, "
    "юридические вопросы, продуктовая развилка)."
)
OUTBOX_DIR = config.STATE_DIR / "deepdive_outbox"
OUTBOX_SENT_DIR = OUTBOX_DIR / "sent"
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
OUTBOX_SENT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

VK_MAX_LEN = 4000  # запас от лимита VK ~4096 символов на сообщение

CONTINUE_MESSAGE = "Продолжай со следующим компонентом"
_STATUS_EMOJI = {"⬜", "🔄", "✅"}


def _checklist_rows() -> list[tuple[str, str]]:
    """(имя_компонента, статус) для каждой строки таблицы в CHECKLIST_PATH, в порядке файла.
    Не читает намерения LLM — чистый факт по markdown-таблице (тот же принцип, что и
    component_queue.py::stamped_components() для очереди валидации)."""
    if not CHECKLIST_PATH.is_file():
        return []
    rows = []
    for line in CHECKLIST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        name = cells[0].replace("`", "").strip()
        status = cells[1].strip()
        if not name or name.lower() == "компонент" or set(name) <= {"-"}:
            continue
        if status not in _STATUS_EMOJI:
            continue
        rows.append((name, status))
    return rows


def _done_components() -> set[str]:
    return {name for name, status in _checklist_rows() if status == "✅"}


def _has_pending_components() -> bool:
    rows = _checklist_rows()
    return bool(rows) and any(status != "✅" for _, status in rows)


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
        "АВТОПРОДОЛЖЕНИЕ: сообщение «" + CONTINUE_MESSAGE + "» может прийти не от владельца лично, "
        "а автоматически от обвязки скрипта (она следит за чеклистом и сама шлёт эту фразу, когда "
        "видит, что предыдущий компонент только что стал ✅) — реагируй на него ТАК ЖЕ, как если бы "
        "владелец написал это сам: открой DataProcessor/docs/COMPONENT_DEEP_DIVE_CHECKLIST.md, "
        "возьми первый компонент со статусом ⬜ (сверху вниз, порядок таблицы = порядок работы: "
        "сначала Visual, потом Audio, потом Text) и веди его по протоколу целиком. Если ⬜ в "
        "чеклисте не осталось — так и скажи одной строкой, обвязка сама остановит автопродолжение.\n\n"
        "ВИЗУАЛИЗАЦИИ ДЛЯ ВЛАДЕЛЬЦА: если можешь построить полезный график/схему (matplotlib, "
        "PIL, что угодно, что даёт PNG) — сохрани файл в " + str(OUTBOX_DIR) + " с осмысленным "
        "уникальным именем. Он АВТОМАТИЧЕСКИ будет отправлен владельцу в VK как фото сразу после "
        "твоего ответа — не нужно ничего для этого вызывать отдельно, просто положи готовый PNG "
        "в эту папку. Не клади туда ничего, кроме изображений, реально готовых к отправке.\n\n"
        "ПОЛНАЯ АВТОНОМИЯ (с 2026-07-18): когда разбор компонентов закончен, ты САМА ведёшь дальнейшую "
        "работу проекта — не только текстовый анализ. У тебя есть mcp__trendflow__pod_control/manager "
        "(поднять/погасить/пересоздать GPU-под RunPod, тот же контур, что у agent_runner.py — можешь "
        "сама пересчитывать батчи, никого не спрашивая на каждый чих) и mcp__trendflow__budget_status/"
        "limits_status (проверяй баланс RunPod и лимит Claude ПЕРЕД длинными GPU-прогонами). Актуальные "
        "автономные задачи — DataProcessor/docs/COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md (§7.1 — что "
        "можно чинить самой; §8 — уже принятые владельцем решения, не пересматривай их без причины; "
        "§9 — ссылка на протокол подбора корпуса) и DataProcessor/docs/AUTOMATED_TEST_CORPUS_PROTOCOL.md "
        "(скачивание HF-данных, отбор кандидатов, разбор на кадры и САМОСТОЯТЕЛЬНОЕ описание кадров "
        "через Read — ты умеешь смотреть на изображения нативно, отдельный vision-сервис не нужен).\n\n"
        "КОГДА ОБРАЩАТЬСЯ К ВЛАДЕЛЬЦУ (и только тогда): нужны данные/ресурсы, которых физически нет "
        "(например датасет, которого не существует), решение про деньги/лицензии/подписки, юридический "
        "вопрос (например GDPR/приватность), или развилка, где оба пути валидны и цена ошибки высокая. "
        "НЕ спрашивай подтверждения на рутинные технические действия (запустить скрипт, починить баг, "
        "пересчитать батч, выбрать порог) — просто делай и коротко отчитайся результатом. Из уже принятых "
        "решений (см. §8 портфолио-оценки): базы брендов/машин/франшиз пока не трогать (данных нет и не "
        "будет в ближайшее время — это ожидаемо, не поднимай вопрос снова); `core_object_detections` НЕ "
        "трогать вообще — идёт независимая ручная переразметка YOLO-классов (cvat_yolo/yolo_dataset/"
        "SESSION_SUMMARY.md), любой пере-прогон сейчас будет впустую; для similarity_metrics/"
        "topk_similar_titles_extractor эталон сравнения — категория + канал + похожие метрики + топ "
        "периода, реализовывать все оси сразу, не выбирать одну.\n\n"
        "В чате отвечай кратко и по делу — это переписка, а не отчёт-в-VK. Развёрнутое содержание "
        "уходит в FINAL_REPORT.md/соответствующие доки, в чат — суть, прогресс, вопросы, выводы."
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


async def _run_component_turn(client: ClaudeSDKClient, lp: "messenger.LongPoll", text: str) -> None:
    """Один ход + автопродолжение очереди компонентов. Сверяет чеклист до/после хода: если
    появился новый ✅ и есть ещё ⬜ — сама шлёт CONTINUE_MESSAGE и рекурсивно продолжает, пока
    компоненты не кончатся или пока не появится реальное сообщение от владельца (проверяется
    коротким non-blocking-подобным poll_once(1) ПЕРЕД авто-сообщением — реальное сообщение всегда
    приоритетнее автопродолжения)."""
    before = _done_components()
    await _handle_turn(client, text)
    after = _done_components()
    newly_done = sorted(after - before)
    if not newly_done:
        return  # ничего не завершилось в этом ходе — это не наша забота продолжать, ждём владельца
    if not _has_pending_components():
        send_text = "🎉 Все компоненты чеклиста разобраны! Жду новых задач."
        messenger.send(send_text)
        messenger.log_chat("AGENT1(auto)", send_text)
        return
    # Реальное сообщение владельца — всегда приоритетнее автопродолжения.
    owner_msgs = await asyncio.to_thread(lp.poll_once, 1)
    if owner_msgs:
        for m in owner_msgs:
            await _run_component_turn(client, lp, m)
        return
    await asyncio.sleep(2)
    messenger.log_chat("AGENT1(auto)", f"[авто] {', '.join(newly_done)} готов -> {CONTINUE_MESSAGE}")
    await _run_component_turn(client, lp, CONTINUE_MESSAGE)


async def main(model_name: str | None) -> None:
    config.require("VK_TOKEN", config.VK_TOKEN)
    config.require("VK_OWNER_ID", config.VK_OWNER_ID)
    model = settings.resolve_model(model_name) if model_name else config.AGENT_MODEL
    print(f"[deepdive] запуск, модель={model}...", flush=True)
    lp = messenger.LongPoll()
    messenger.send(
        "🧭 TrendFlow Bot переключён на свободный разбор компонентов (Final Report), модель "
        f"{model}. Пиши как в обычном чате — спецкоманды не нужны. Например: «разбери core_clip». "
        "Очередь компонентов идёт автоматически — после каждого готового отчёта сама берётся за "
        "следующий, не нужно писать «продолжай»."
    )
    print("[deepdive] готов, слушаю VK", flush=True)
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=_system_prompt(),
        cwd=str(config.REPO_DIR),
        mcp_servers={"trendflow": tools.build_server()},
        allowed_tools=[
            "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch",
            "mcp__trendflow__pod_control", "mcp__trendflow__manager",
            "mcp__trendflow__budget_status", "mcp__trendflow__limits_status",
        ],
        permission_mode="bypassPermissions",
        max_turns=500,
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[hooks.guard_bash])]},
    )
    while True:
        try:
            async with ClaudeSDKClient(options=options) as client:
                if _has_pending_components():
                    messenger.send("▶️ В чеклисте есть неразобранные компоненты — начинаю сама.")
                    await _run_component_turn(client, lp, CONTINUE_MESSAGE)
                else:
                    messenger.send("▶️ Очередь разбора компонентов пуста — перехожу к автономным задачам "
                                   "из портфолио-оценки.")
                    await _run_component_turn(client, lp, AUTONOMOUS_KICKOFF_MESSAGE)
                while True:
                    texts = await asyncio.to_thread(lp.poll_once, 25)
                    for t in texts:
                        try:
                            await _run_component_turn(client, lp, t)
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
                    "но всё, что уже записано в FINAL_REPORT.md/CHECKLIST, не потеряно — очередь "
                    "продолжится с того места, где остановилась."
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
