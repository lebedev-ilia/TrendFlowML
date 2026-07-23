#!/usr/bin/env python3
"""Site Agent — отдельный VK-бот, строящий ГОЛЫЙ КАРКАС сайта TrendFlow (кнопки/страницы/блоки,
без дизайна — ч/б или минимальные базовые стили). Работает в /home/ilya/Рабочий стол/site
(независимый репозиторий от TrendFlowML, см. config.SITE_DIR).

Архитектура — тот же паттерн, что у deepdive_agent.py/models_agent.py (VK long-poll в отдельной
asyncio-задаче + очередь + client.interrupt() для приоритета свежего сообщения владельца), но С
УЖЕ ВСТРОЕННЫМИ фиксами багов, найденных на них 2026-07-19..21 (не наступаем на те же грабли):
  - client.interrupt() всегда фоновая задача (asyncio.create_task), НИКОГДА не await'ится внутри
    _producer — иначе на долгом Bash блокируется весь приём VK-сообщений.
  - Каждый ToolUseBlock тоже шлёт короткую строку-эхо в VK — тишины во время долгих серий
    Bash/Read/Write не бывает.
  - Отправка каждого сообщения обёрнута в try/except — сетевой сбой (DNS и т.п.) на ОДНОМ
    сообщении не обрывает весь ход.

ГЛАВНОЕ ОТЛИЧИЕ от deepdive_agent.py: там есть автопродолжение (_run_component_turn сама шлёт
следующее сообщение и идёт дальше). ЗДЕСЬ автопродолжения НЕТ НАРОЧНО — это жёсткий стоп-гейт по
прямому требованию владельца (2026-07-22): после каждой страницы/блока агент присылает скриншот
и текстовое резюме, затем ФИЗИЧЕСКИ останавливается (просто возвращает управление, ход
заканчивается) и ждёт следующего сообщения от владельца — цикл `while True: await queue.get()`
сам по себе это обеспечивает, если система-промпт СТРОГО запрещает начинать следующую
страницу/блок в том же ходу.

Запуск:
    cd automation/site_agent && python3 -m venv .venv && source .venv/bin/activate
    pip install -U claude-agent-sdk python-dotenv requests
    python site_agent.py                  # модель по умолчанию (SITE_AGENT_MODEL из .env)
    python site_agent.py --model sonnet    # или другой алиас
"""
from __future__ import annotations
import argparse
import asyncio
import time
from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ToolUseBlock, ResultMessage,
)

import config
import messenger
import session_state

AGENT_NAME = "site_agent"

VK_MAX_LEN = 4000
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

MODELS = {
    "opus": "claude-opus-4-8", "opus 4.8": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6", "sonnet 4.6": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001", "haiku 4.5": "claude-haiku-4-5-20251001",
}

CONTINUE_AFTER_RESTART_MESSAGE = (
    "Перезапуск (сбой сессии или рестарт процесса) — контекст восстановлен автоматически из "
    "прошлой сессии, ничего не потеряно. Если на момент обрыва ждал(а) моего ответа/одобрения — "
    "просто напиши, на чём остановился(-ась), и жди, как предписывает стоп-гейт. Ничего нового "
    "сама не начинай."
)

FIRST_TASK_MESSAGE = (
    "Начни с ознакомления, ПЕРЕД тем как писать хоть одну строчку кода:\n\n"
    "1. Прочитай (в TrendFlowML, доступно для чтения, но НЕ рабочая директория — рабочая "
    f"директория для всех правок и запуска сайта: {config.SITE_DIR}):\n"
    "   - TrendFlowML/CLAUDE.md — общий контекст проекта TrendFlow (что за продукт, аудитория, "
    "архитектура MLService/Site/backend).\n"
    "   - site/SITE_SPECIFICATION.md (в самой рабочей директории) — уже существующая полная "
    "спецификация UI/UX (дизайн отдельно, тебе НУЖНА структура: какие страницы, блоки, шаги).\n"
    "   - TrendFlowML/backend/docs/API.md и OVERVIEW.md — какие эндпоинты реально есть/планируются, "
    "какие данные отдаёт backend (это должно определять, какие поля/блоки реально имеет смысл "
    "закладывать в каркас страницы результатов/дашборда — не выдумывай поля, которых нет ни в API, "
    "ни в контрактах).\n"
    "   - TrendFlowML/Models/docs/contracts/BASELINE_MODEL.md и TARGETS_SPLITS_METRICS.md — что "
    "именно предсказывает ML (views/likes на 14/21 день и т.д.), в каком виде — это должно "
    "напрямую определять каркас страницы прогноза/результатов.\n"
    "   - TrendFlowML/DataProcessor/docs/MAIN_INDEX.md (бегло) — какие компоненты анализа "
    "существуют (визуальные/аудио/текстовые), чтобы каркас 'разбора по компонентам' на странице "
    "результатов отражал реальные категории, а не абстрактные.\n\n"
    "2. Осмотри уже существующий код в src/ — landing (app/page.tsx), auth "
    "(app/(auth)/login,register), docs (app/docs), дашборд-заглушки (app/(dashboard)/*). Пойми, что "
    "уже есть, прежде чем что-то менять.\n\n"
    "3. (По желанию, если полезно) поищи в интернете 1-2 референса структуры (НЕ дизайна) у "
    "популярных продуктов близкого класса (аналитика видео/YouTube, SaaS-дашборды с мастером "
    "создания отчёта — напр. как устроена информационная архитектура у VidIQ/TubeBuddy/аналогов, "
    "или как в целом устроены wizard-флоу создания отчёта в популярных SaaS) — только чтобы "
    "свериться со структурой (сколько шагов, что на каждом), НЕ копировать визуальный стиль.\n\n"
    "4. Предложи владельцу порядок работы (список страниц/блоков и в каком порядке ты предлагаешь "
    "их каркасить) — коротким сообщением, и ОСТАНОВИСЬ, жди подтверждения или правок порядка, "
    "прежде чем писать первую строчку кода. Это первый чек-поинт стоп-гейта."
)


def _system_prompt() -> str:
    return f"""Ты — Site Agent, отдельный VK-бот-разработчик. Задача: строить ГОЛЫЙ КАРКАС сайта
TrendFlow — НЕ дизайн. Рабочая директория (единственная, где ты правишь код и откуда запускаешь
сайт): {config.SITE_DIR}. Это Next.js 15 (App Router) + TypeScript + Tailwind + shadcn/ui (уже
настроено, components.json/tailwind.config.ts на месте) + NextAuth v5 + Prisma + Three.js/R3F —
стек уже выбран, не меняй его без явной причины и согласования.

## Что значит "каркас, не дизайн" (владелец, дословно)
Кнопки, элементы, страницы, блоки — БЕЗ стилей, чёрно-белое или максимально простые базовые
стили (структура/отступы/типографика по умолчанию Tailwind — не бренд-палитра, не 3D-сцены, не
анимации). Это касается ВСЕГО сайта, включая уже частично задизайненные landing/auth/docs
(владелец подтвердил: их тоже упростить до каркаса — единый проход по всему сайту, дизайн будет
накладываться позже одним отдельным проходом). SITE_SPECIFICATION.md остаётся источником истины
для СТРУКТУРЫ (какие блоки/шаги/поля), не для визуального стиля — визуальные детали (§1 палитра,
3D-фон и т.п.) сейчас игнорируй.

## Жёсткий стоп-гейт (обязательно, не автопродолжение)
После КАЖДОЙ страницы или крупного блока: (1) сам запусти дев-сервер (`npm run dev`, если ещё не
запущен — проверяй, не плоди процессы), (2) сделай скриншот результата (см. ниже — Playwright),
(3) положи PNG в {config.OUTBOX_DIR} (уйдёт в VK автоматически как фото сразу после твоего ответа —
ничего сверх этого делать не нужно), (4) коротко резюмируй текстом, что сделано, (5) ОСТАНОВИСЬ.
НЕ начинай следующую страницу/блок в этом же ходу и не пиши себе следующее сообщение сама —
дождись реального ответа владельца (одобрение, правки, или новое задание) прежде чем продолжать.
Это осознанное отличие от других твоих коллег-ботов (TrendFlow Bot/Models Bot) — у них
автопродолжение, у тебя НЕТ, по прямому требованию владельца.

## Скриншоты (Playwright)
Если Playwright ещё не установлен в {config.SITE_DIR}: `npm install -D playwright` (или
`@playwright/test`), затем один раз `npx playwright install chromium`. Дальше для каждого чек-поинта:
`npx playwright screenshot http://localhost:3000/<путь> {config.OUTBOX_DIR}/<осмысленное_имя>.png
--viewport-size=1280,900` (сервер `npm run dev` должен уже быть поднят и отвечать). Держи дев-сервер
поднятым между ходами (фон, не блокирующий процесс) — не поднимай второй параллельно.

## Тесное взаимодействие с реальным функционалом (владелец, 2026-07-22)
Каркас должен отражать РЕАЛЬНЫЙ функционал ML-системы и backend, а не абстрактные заглушки:
- Читай TrendFlowML/backend/docs/API.md — какие эндпоинты/поля реально есть, прежде чем закладывать
  блок на странице (напр. страница результатов — какие метрики/поля backend реально может отдать).
- Читай TrendFlowML/Models/docs/contracts/ — что именно предсказывает модель (views/likes @14/21д и
  т.д.) — это должно НАПРЯМУЮ определять структуру блоков прогноза.
- Читай TrendFlowML/DataProcessor/docs/ — какие компоненты анализа (визуальные/аудио/текстовые)
  реально существуют — раздел "разбор по компонентам" на странице результатов должен отражать
  реальные категории (см. также свежие DataProcessor/docs/corpus_run_report/*.md — там актуальные
  находки по метрикам качества компонентов, если что-то там помечено как "мёртвое"/mock — не
  закладывай под это отдельный видный UI-блок).
- Если backend/ML ещё не готовы под что-то из SITE_SPECIFICATION.md — не выдумывай точную форму
  данных, закладывай разумный плейсхолдер-тип и явно спроси владельца/оставь TODO-комментарий в
  коде, а не гадай молча.

## Референсы (структура, НЕ дизайн)
Можно и нужно искать в интернете (WebSearch/WebFetch) референсы ИНФОРМАЦИОННОЙ АРХИТЕКТУРЫ
популярных сайтов близкого класса (видео-аналитика, SaaS с多step-мастером создания отчёта/анализа,
дашборды) — сколько шагов в мастере, как группируются разделы личного кабинета, как устроена
страница результатов у аналитических инструментов. Цель — не упустить очевидный структурный блок,
не скопировать чей-то визуальный стиль (стиль сейчас не нужен вообще).

## Продакшен-архитектура с самого начала (владелец, дословно: "качество, безопасность")
- Файлы/папки/компоненты — по уже заданной конвенции Next.js App Router проекта (route groups
  (auth)/(dashboard), src/components/{{ui,layout,sections,auth,...}}) — не изобретай новую структуру.
  Пере используй shadcn/ui примитивы в src/components/ui, не дублируй.
  Компоненты — TypeScript, типизированные props, никаких `any` без крайней необходимости.
- Формы — react-hook-form + zod (уже зависимости) — валидация на клиенте с самого начала каркаса,
  даже если поля пока пустые/плейсхолдер.
- Безопасность: не хардкодь секреты/токены в коде, только через .env.local (не коммить — проверь
  .gitignore). NextAuth-конфигурация — не ослабляй существующие настройки без согласования.
  Проверяй `npm run lint`/`npm run build` перед тем как считать страницу готовой к показу.
- Коммить в git часто, маленькими логическими шагами (по странице/блоку — как раз совпадает со
  стоп-гейтом), понятные сообщения коммитов. `git pull --rebase` перед push, если вдруг были
  правки не от тебя.

## Общение с владельцем
Всегда на связи, коротко и по делу. Вопросы задавай, когда реально не хватает информации для
решения (не для мелочей, которые можно решить самостоятельно и показать результат на скриншоте —
показать и спросить "так?" часто быстрее, чем спрашивать заранее). После каждой правки по замечанию
владельца — тоже стоп-гейт (скриншот + резюме + жди дальше).

Модель: {{model}}."""


async def _handle_turn(client: ClaudeSDKClient, text: str) -> None:
    """Один ход, без автопродолжения (см. докстринг модуля — стоп-гейт нарочно). Паттерны отправки
    в VK (эхо каждого TextBlock/ToolUseBlock сразу, устойчивость к сетевым сбоям на отправке) —
    те же уже отлаженные фиксы, что в deepdive_agent.py/models_agent.py (найдены 2026-07-19..21)."""
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
                        try:
                            _send_long(f"⏳ {block.text.strip()}")
                        except Exception as e:
                            print(f"[site_agent] сообщение не отправилось (сеть?): {e}", flush=True)
                    elif isinstance(block, ToolUseBlock):
                        print(f"[site_agent] tool: {_summarize_tool_use(block)}", flush=True)
                        if _should_send_tool_echo():
                            try:
                                messenger.send(_summarize_tool_use(block))
                            except Exception as e:
                                print(f"[site_agent] tool-echo не отправился: {e}", flush=True)
            elif isinstance(msg, ResultMessage):
                cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
                print(f"[site_agent] ход завершён, cost=${cost:.4f}", flush=True)
    except Exception as e:
        print(f"[site_agent] ход прерван: {e}", flush=True)
    _flush_outbox_photos()


def _send_long(text: str) -> None:
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


# Превентивно (2026-07-23): на TrendFlow Bot/Models Bot нашли, что эхо на КАЖДЫЙ ToolUseBlock при
# интенсивной работе топит реальный статус в потоке сообщений + вероятный троттлинг VK. У Site
# Agent пока мало вызовов (стоп-гейт ограничивает объём хода), но фикс ставим сразу, чтобы не
# наступить на те же грабли, когда работы станет больше. Троттлим только tool-эхо, не TextBlock.
TOOL_ECHO_MIN_INTERVAL_SEC = 2.5
_last_tool_echo_at = 0.0


def _should_send_tool_echo() -> bool:
    global _last_tool_echo_at
    now = time.monotonic()
    if now - _last_tool_echo_at < TOOL_ECHO_MIN_INTERVAL_SEC:
        return False
    _last_tool_echo_at = now
    return True


_TOOL_INPUT_KEYS = ("command", "file_path", "pattern", "path", "prompt", "url", "description")


def _summarize_tool_use(block: "ToolUseBlock") -> str:
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


def _flush_outbox_photos() -> None:
    for path in sorted(config.OUTBOX_DIR.iterdir()):
        if path.is_dir() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            messenger.send_photo(path, caption=path.stem.replace("_", " "))
            path.rename(config.OUTBOX_SENT_DIR / path.name)
        except Exception as e:
            print(f"[site_agent] не удалось отправить фото {path.name}: {e}", flush=True)


async def _safe_interrupt(client: ClaudeSDKClient) -> None:
    """НИКОГДА не await'ится напрямую в цикле опроса VK (см. докстринг модуля) — фоновая задача."""
    try:
        await client.interrupt()
    except Exception as e:
        print(f"[site_agent] interrupt не сработал: {e}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config.require("VK_TOKEN", config.VK_TOKEN)
    config.require("VK_OWNER_ID", config.VK_OWNER_ID)
    if not config.SITE_DIR.is_dir():
        raise RuntimeError(f"SITE_DIR не найден: {config.SITE_DIR} — проверь .env")

    model = MODELS.get((args.model or "").lower(), args.model) or config.AGENT_MODEL

    resume_id = session_state.load(AGENT_NAME)
    lp = messenger.LongPoll()

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=_system_prompt().replace("{model}", model),
        cwd=str(config.SITE_DIR),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
        permission_mode="bypassPermissions",
        max_turns=500,
        # Баг найден 2026-07-23: дефолтный лимит SDK на JSON-сообщение от CLI-подпроцесса — 1MB
        # (claude_agent_sdk/_internal/transport/subprocess_cli.py: _DEFAULT_MAX_BUFFER_SIZE). Один
        # большой результат инструмента (вывод Bash — npm install, playwright screenshot логи,
        # большой Read/WebFetch) легко превышает это и роняет ВЕСЬ клиент с фатальной ошибкой
        # ("Fatal error in message reader"). Подняли с запасом — не убирает риск совсем при
        # экстремально больших выводах, но покрывает реалистичные случаи для этого бота.
        max_buffer_size=20 * 1024 * 1024,  # 20MB
        resume=resume_id,
    )
    print(f"[site_agent] готов, слушаю VK (resume={resume_id!r}), cwd={config.SITE_DIR}", flush=True)

    while True:
        try:
            async with ClaudeSDKClient(options=options) as client:
                queue: asyncio.Queue[str] = asyncio.Queue()
                busy = asyncio.Event()

                if options.resume:
                    await _handle_turn(client, CONTINUE_AFTER_RESTART_MESSAGE)
                else:
                    messenger.send("▶️ Site Agent на связи. Начинаю с ознакомления с проектом и "
                                   "предложу порядок работы — без правок кода, пока не подтвердишь.")
                    await _handle_turn(client, FIRST_TASK_MESSAGE)

                async def _producer() -> None:
                    while True:
                        try:
                            texts = await asyncio.to_thread(lp.poll_once, 25)
                        except Exception as e:
                            print(f"[site_agent] poll_once упал, жду и пробую снова: {e}", flush=True)
                            await asyncio.sleep(3)
                            continue
                        for t in texts:
                            if busy.is_set():
                                messenger.send("⏸ Прерываю текущую работу — читаю новое сообщение.")
                                asyncio.create_task(_safe_interrupt(client))
                            await queue.put(t)

                producer_task = asyncio.create_task(_producer())
                try:
                    while True:
                        t = await queue.get()
                        busy.set()
                        try:
                            await _handle_turn(client, t)
                        except Exception as e:
                            print(f"[site_agent] ошибка обработки сообщения: {e}", flush=True)
                            try:
                                messenger.send(f"❗ Ошибка: {e}")
                            except Exception:
                                pass
                        finally:
                            busy.clear()
                finally:
                    producer_task.cancel()
        except Exception as e:
            print(f"[site_agent] сессия оборвалась, переподключаюсь через 5с: {e}", flush=True)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
