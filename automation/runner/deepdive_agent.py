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
    ClaudeSDKClient, ClaudeAgentOptions, HookMatcher, AssistantMessage, TextBlock, ToolUseBlock,
    ResultMessage,
)

import config
import messenger
import settings
import tools
import hooks
import session_state

AGENT_NAME = "deepdive_agent"
CONTINUE_AFTER_RESTART_MESSAGE = (
    "Перезапуск (сбой сессии или рестарт процесса) — контекст восстановлен автоматически из "
    "прошлой сессии, ничего не потеряно. Продолжай с того места, на котором остановился(-ась) "
    "до перезапуска."
)

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


async def _safe_interrupt(client: ClaudeSDKClient) -> None:
    """Вызывается фоновой задачей из _producer (см. баг №5 там) — НИКОГДА не await'ится напрямую
    в цикле опроса VK, чтобы долгий/подвисший control-request к CLI не блокировал приём сообщений."""
    try:
        await client.interrupt()
    except Exception as e:
        print(f"[deepdive] interrupt не сработал: {e}", flush=True)


_TOOL_INPUT_KEYS = ("command", "file_path", "pattern", "path", "prompt", "url", "description")


def _summarize_tool_use(block: "ToolUseBlock") -> str:
    """Короткая строка-эхо на каждый вызов инструмента — см. _handle_turn докстринг, баг №4."""
    value = ""
    for key in _TOOL_INPUT_KEYS:
        raw = (block.input or {}).get(key)
        if raw:
            value = str(raw)
            break
    if not value and block.input:
        value = str(block.input)
    value = value.strip().replace("\n", " ")
    if len(value) > 140:
        value = value[:140] + "…"
    return f"🔧 {block.name}: {value}" if value else f"🔧 {block.name}"


async def _handle_turn(client: ClaudeSDKClient, text: str) -> None:
    """Баг найден 2026-07-19: раньше весь текст хода копился в parts и уходил в VK ОДНИМ
    сообщением только в конце — при долгой работе (разбор компонента, DatasetBuilder, обучение)
    в чате была полная тишина по 10+ минут, владелец решал, что агент завис (тот же баг, что
    чинили в watchdog.py). Теперь каждый TextBlock уходит в VK сразу, короткой пометкой — это и
    есть промежуточный прогресс; в конце хода финальный текст всё равно приходит целиком
    (небольшое дублирование последней мысли — приемлемая цена за отсутствие тишины).

    Второй баг найден 2026-07-19 (в этом же разговоре): если во время хода приходит НОВОЕ
    сообщение от владельца, скрипт физически не мог его увидеть — poll_once() вызывался только
    ПОСЛЕ полного завершения текущего хода (см. main()). Теперь ход может быть прерван снаружи
    через client.interrupt() (см. _producer в main()) — except ниже ловит обрыв стрима, который
    interrupt вызывает, и не считает это ошибкой. session_id хода сохраняется на диск при каждом
    сообщении — это то, что позволяет resume при рестарте (см. session_state.py).

    Третий баг найден 2026-07-20 (владелец): каждый TextBlock уходил в VK СРАЗУ урезанным до 200
    символов ("⏳ ..."), а в конце хода ВЕСЬ накопленный текст уходил ЕЩЁ РАЗ целиком через
    _send_long — то есть каждая мысль дублировалась дважды, а полный текст владелец видел только
    во втором, отложенном сообщении (создавало ощущение, что бот "завис и потом резко высыпал
    сообщения" — на деле это второй, дублирующий проход). Теперь каждый блок уходит СРАЗУ и
    ПОЛНОСТЬЮ (через _send_long, режет только если реально длиннее лимита VK) — финального
    повторного прохода по parts больше нет.

    Четвёртый баг найден 2026-07-21 (владелец, снова "зависает"): после того, как боту дали
    многодневную работу с длинными сериями Bash/Read/Write БЕЗ TextBlock между ними (git, pod_control,
    анализ файлов), в VK не уходило вообще ничего на протяжении всей такой серии — процесс жив и
    работает, но со стороны VK неотличимо от зависания. Теперь каждый ToolUseBlock тоже шлёт короткую
    строку-эхо (см. _summarize_tool_use) — тишины между текстовыми мыслями больше не бывает, даже если
    ход целиком состоит из вызовов инструментов."""
    messenger.log_chat("OWNER->", text)
    await client.query(text)
    try:
        async for msg in client.receive_response():
            sid = getattr(msg, "session_id", None)
            if sid:
                session_state.save(AGENT_NAME, sid)
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        _send_long(f"⏳ {block.text.strip()}")
                    elif isinstance(block, ToolUseBlock):
                        try:
                            messenger.send(_summarize_tool_use(block))
                        except Exception as e:
                            print(f"[deepdive] tool-echo не отправился: {e}", flush=True)
            elif isinstance(msg, ResultMessage):
                cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
                print(f"[deepdive] ход завершён, cost=${cost:.4f}", flush=True)
    except Exception as e:
        print(f"[deepdive] ход прерван: {e}", flush=True)
    _flush_outbox_photos()


async def _run_component_turn(client: ClaudeSDKClient, queue: "asyncio.Queue[str]", text: str) -> None:
    """Один ход + автопродолжение очереди компонентов. Сверяет чеклист до/после хода: если
    появился новый ✅ и есть ещё ⬜ — сама шлёт CONTINUE_MESSAGE и рекурсивно продолжает, пока
    компоненты не кончатся или пока не появится реальное сообщение от владельца.

    Раньше проверка «не написал ли владелец» шла отдельным прямым lp.poll_once(1) — теперь ЕСТЬ
    только один VK-слушатель на процесс (_producer в main()), поэтому здесь просто смотрим в общую
    очередь non-blocking (queue.get_nowait()): два конкурентных poll_once на одном long-poll ts
    ломали бы курсор VK. Реальное сообщение владельца всегда приоритетнее автопродолжения."""
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
    try:
        m = queue.get_nowait()
        await _run_component_turn(client, queue, m)
        return
    except asyncio.QueueEmpty:
        pass
    await asyncio.sleep(2)
    messenger.log_chat("AGENT1(auto)", f"[авто] {', '.join(newly_done)} готов -> {CONTINUE_MESSAGE}")
    await _run_component_turn(client, queue, CONTINUE_MESSAGE)


async def main(model_name: str | None) -> None:
    config.require("VK_TOKEN", config.VK_TOKEN)
    config.require("VK_OWNER_ID", config.VK_OWNER_ID)
    model = settings.resolve_model(model_name) if model_name else config.AGENT_MODEL
    print(f"[deepdive] запуск, модель={model}...", flush=True)
    lp = messenger.LongPoll()
    resume_id = session_state.load(AGENT_NAME)
    messenger.send(
        f"🧭 TrendFlow Bot на связи (модель {model})." + (
            " Восстанавливаю прошлую сессию — контекст не потерян."
            if resume_id else
            " Пиши как в обычном чате — спецкоманды не нужны. Например: «разбери core_clip». "
            "Очередь компонентов идёт автоматически — после каждого готового отчёта сама "
            "берётся за следующий, не нужно писать «продолжай»."
        )
    )
    print(f"[deepdive] готов, слушаю VK (resume={resume_id!r})", flush=True)
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
        # Ограничение на Bash (guard_bash: supervisor-проверка опасных команд, лимит на тяжёлые
        # прогоны при высоком % Claude) СНЯТО по прямой просьбе владельца 2026-07-19 — полный
        # доступ ко всем командам без исключений (в т.ч. нужно для ночной автономной задачи:
        # чистка старых данных на Network Volume, куда явно дано разрешение).
        resume=resume_id,
    )
    while True:
        try:
            async with ClaudeSDKClient(options=options) as client:
                queue: asyncio.Queue[str] = asyncio.Queue()

                if options.resume:
                    await _run_component_turn(client, queue, CONTINUE_AFTER_RESTART_MESSAGE)
                elif _has_pending_components():
                    messenger.send("▶️ В чеклисте есть неразобранные компоненты — начинаю сама.")
                    await _run_component_turn(client, queue, CONTINUE_MESSAGE)
                else:
                    messenger.send("▶️ Очередь разбора компонентов пуста — перехожу к автономным задачам "
                                   "из портфолио-оценки.")
                    await _run_component_turn(client, queue, AUTONOMOUS_KICKOFF_MESSAGE)

                # Баг найден 2026-07-19: poll_once() раньше вызывался ТОЛЬКО между полностью
                # завершёнными ходами (включая рекурсивное автопродолжение очереди компонентов) —
                # если владелец писал во время долгого хода, сообщение физически не читалось, пока
                # ход сам не закончится. Теперь VK слушается НЕПРЕРЫВНО отдельной задачей
                # (_producer): если приходит сообщение, а бот занят (busy) — зовём client.interrupt()
                # (штатная возможность SDK для streaming-режима), текущий receive_response()
                # обрывается, и новое сообщение уходит в обработку сразу.
                busy = asyncio.Event()

                async def _producer() -> None:
                    """Баг найден 2026-07-19: раньше был fire-and-forget asyncio.create_task —
                    если lp.poll_once() кидал ЛЮБОЕ исключение мимо requests.RequestException
                    (например json.JSONDecodeError на кривом ответе VK), задача молча умирала.
                    Внешний try/except в main() её не видел (это отдельная Task, не await в
                    основном потоке) — VK переставал слушаться НАВСЕГДА до ручного рестарта, при
                    этом консьюмер продолжал жить на том, что уже было в очереди (автопродолжение
                    очереди компонентов), создавая иллюзию, что бот просто "не реагирует", хотя на
                    деле не мог физически увидеть новые сообщения. Подтверждено на практике:
                    настоящее сообщение владельца НИ РАЗУ не попало в agent1_chat.log за ~8 минут
                    непрерывной фоновой работы. Теперь опрос обёрнут в try/except, который никогда
                    не даёт задаче умереть."""
                    while True:
                        try:
                            texts = await asyncio.to_thread(lp.poll_once, 25)
                        except Exception as e:
                            print(f"[deepdive] poll_once упал, жду и пробую снова: {e}", flush=True)
                            await asyncio.sleep(3)
                            continue
                        for t in texts:
                            if busy.is_set():
                                messenger.send("⏸ Прерываю текущую работу — читаю новое сообщение.")
                                # Баг найден 2026-07-21: раньше `await client.interrupt()` стоял ПРЯМО
                                # тут, внутри _producer — единственного цикла, который вообще читает
                                # VK. SDK ждёт ответа CLI-подпроцесса на control-request до 60с; если
                                # CLI в этот момент занят долгим Bash (git/pod_control/анализ файлов —
                                # именно то, чем боты теперь занимаются часами), interrupt() блокировал
                                # ВЕСЬ _producer на эти секунды/до минуты — новые сообщения физически
                                # не вычитывались из VK, не только не обрабатывались. Со стороны
                                # владельца выглядело как "зависает, пока не напишу ещё раз" — на самом
                                # деле не interrupt чинил зависание повторным сообщением, а просто
                                # совпадало по времени с окончанием этой блокировки. Теперь interrupt
                                # запускается фоновой задачей — _producer НИКОГДА не блокируется на
                                # нём и продолжает опрашивать VK без остановки.
                                asyncio.create_task(_safe_interrupt(client))
                            await queue.put(t)

                producer_task = asyncio.create_task(_producer())
                try:
                    while True:
                        t = await queue.get()
                        busy.set()
                        try:
                            await _run_component_turn(client, queue, t)
                        except Exception as e:
                            print(f"[deepdive] ошибка обработки сообщения: {e}", flush=True)
                            try:
                                messenger.send(f"❗ Ошибка: {e}")
                            except Exception:
                                pass
                        finally:
                            busy.clear()
                finally:
                    producer_task.cancel()
        except Exception as e:
            print(f"[deepdive] сессия оборвалась, переоткрываю: {e}", flush=True)
            try:
                messenger.send(
                    f"⚠️ Сессия оборвалась ({e}), переоткрываю с тем же контекстом (не с нуля). "
                    "Прогресс в файлах тоже не теряется."
                )
            except Exception:
                pass
            # Резюмируем с последнего известного session_id — переоткрытие сессии ВНУТРИ процесса
            # тоже не должно терять контекст, не только полный рестарт процесса.
            options.resume = session_state.load(AGENT_NAME) or options.resume
            await asyncio.sleep(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="Алиас модели (opus/sonnet/haiku/fable) или сырой id.")
    args = ap.parse_args()
    try:
        asyncio.run(main(args.model))
    except KeyboardInterrupt:
        print("\n[deepdive] стоп")
