#!/usr/bin/env python3
"""Models Bot — Агент B из Models/docs/MULTI_AGENT_TRAINING_STRATEGY.md.

Отдельный бот (VK_TOKEN2 — тот же токен, что раньше был у "Второго агента"/assistant.py; его
исходная задача — присматривать за agent_runner.py — сейчас неактуальна, т.к. очередь валидации
компонентов закрыта и agent_runner.py заменён на deepdive_agent.py, см. AGENT_CONTEXT.md). Работает
ПАРАЛЛЕЛЬНО с TrendFlow Bot (deepdive_agent.py, VK_TOKEN) — разные VK-токены специально, чтобы не
конкурировать за long poll одного и того же бота.

Задача: довести до конца сборку датасета и первый честный прогон baseline-модели на реальных данных —
DatasetBuilder (сейчас есть только описание в докax, кода нет) → baseline train → eval, по методологии
из Models/docs/MULTI_AGENT_TRAINING_STRATEGY.md. Свободный диалог, как у deepdive_agent.py — не нужны
спецкоманды, автономная работа, обращение к владельцу только за реальными решениями.

Запуск:
    cd automation/runner && source .venv/bin/activate
    python models_agent.py                  # модель по умолчанию (Opus)
    python models_agent.py --model sonnet    # или другой алиас из settings.MODELS
"""
from __future__ import annotations
import argparse
import asyncio
import datetime as dt
import time
import mimetypes
import random
from pathlib import Path

import requests
from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions, HookMatcher, AssistantMessage, TextBlock, ToolUseBlock,
    ResultMessage,
)

import config
import settings
import tools
import hooks
import session_state

VK = "https://api.vk.com/method"
AGENT_NAME = "models_agent"

CONTINUE_AFTER_RESTART_MESSAGE = (
    "Перезапуск (сбой сессии или рестарт процесса) — контекст восстановлен автоматически из "
    "прошлой сессии, ничего не потеряно. Продолжай с того места, на котором остановился(-ась) "
    "до перезапуска; если на момент обрыва ждал(а) моего ответа — просто напиши, на чём "
    "остановился(-ась), и жди."
)

MODELS_DIR = config.REPO_DIR / "Models"
STRATEGY_PATH = MODELS_DIR / "docs" / "MULTI_AGENT_TRAINING_STRATEGY.md"
BASELINE_CONTRACT_PATH = MODELS_DIR / "docs" / "contracts" / "BASELINE_MODEL.md"
TARGETS_CONTRACT_PATH = MODELS_DIR / "docs" / "contracts" / "TARGETS_SPLITS_METRICS.md"
ROADMAP_PATH = MODELS_DIR / "docs" / "roadmaps" / "BASELINE_TO_TRAINING_ROADMAP.md"

CHAT_LOG = config.STATE_DIR / "models_agent_chat.log"
OUTBOX_DIR = config.STATE_DIR / "models_agent_outbox"
OUTBOX_SENT_DIR = OUTBOX_DIR / "sent"
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
OUTBOX_SENT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
VK_MAX_LEN = 4000

FIRST_TASK_MESSAGE = (
    "Начни с самого узкого места: `DataProcessor/DatasetBuilder/` описан в контрактах, но кода нет "
    "вообще ни одного файла. Первый конкретный результат, который от тебя нужен:\n"
    "1) `feature_spec.yaml` — схема v0-real: возьми список фичей из BASELINE_MODEL.md (вход "
    "система-промпта ниже), но ИСКЛЮЧИ core_brand_semantics/core_car_semantics/core_place_semantics/"
    "core_face_identity (заглушки, не трогаем) и семантику text/logo из core_object_detections "
    "(веса не те, идёт переразметка) — про это подробнее в DataProcessor/docs/"
    "COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md §8. Также сверься с 75 FINAL_REPORT.md "
    "(DataProcessor/docs/component_reports/) — исключи уже названные там мёртвые/дублирующие фичи, "
    "не включай их в v0 просто чтобы потом чистить руками.\n"
    "2) `build_training_table.py` — реально проходит по `result_store`, собирает по каждому "
    "видео плоский вектор согласно feature_spec.yaml (используй как основу существующий "
    "Models/baseline/common/npz_features.py, не изобретай парсинг NPZ заново).\n"
    "3) `add_targets.py` — подтягивает снапшоты просмотров/лайков (HF-датасет владельца, см. "
    "Fetcher/fetcher/dataset_collector/ и FETCHER_DATASET_COLLECTOR_HANDOFF.md) и считает "
    "y=log(1+Δ) по контракту.\n"
    "Прогони на маленьком реальном наборе видео, какой есть в storage/result_store сейчас (не жди "
    "100k) — цель первого прохода: пайплайн работает и метрики не NaN, а не идеальное качество. "
    "Явно проверь на утечку данных (ничего из будущего относительно snapshot_0). Дальше — "
    "Models/baseline/Training/train_baseline.py уже написан и ни разу не запускался — прогони как "
    "есть, чини только то, что реально сломается. Отчитывайся кратко по ходу, полный разбор — "
    "в файлы/README рядом с DatasetBuilder."
)


def _system_prompt() -> str:
    strategy = STRATEGY_PATH.read_text(encoding="utf-8") if STRATEGY_PATH.is_file() else "(нет файла)"
    baseline = BASELINE_CONTRACT_PATH.read_text(encoding="utf-8") if BASELINE_CONTRACT_PATH.is_file() else "(нет файла)"
    targets = TARGETS_CONTRACT_PATH.read_text(encoding="utf-8") if TARGETS_CONTRACT_PATH.is_file() else "(нет файла)"
    return (
        "Ты — Models Bot, Агент B проекта TrendFlow (см. Models/docs/MULTI_AGENT_TRAINING_STRATEGY.md "
        "— документ ниже целиком). Твоя зона: Models/ — сборка датасета из реальных фичей DataProcessor "
        "и первое обучение baseline-модели предсказания просмотров/лайков видео. Работаешь ПАРАЛЛЕЛЬНО "
        "с TrendFlow Bot (automation/runner/deepdive_agent.py, отдельный VK-бот) — он занимается "
        "DataProcessor-стороной (дозревание компонентов, пересчёт батчей), ты — Models-стороной. Не "
        "лезь чинить компоненты DataProcessor сам — если для твоей работы не хватает конкретной "
        "фичи/батча, читай прогресс TrendFlow Bot (DataProcessor/docs/COMPONENT_DEEP_DIVE_CHECKLIST.md, "
        "COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md) и работай с тем, что реально есть сейчас — не "
        "жди, пока всё дозреет.\n\n"
        "=== СТРАТЕГИЯ (Models/docs/MULTI_AGENT_TRAINING_STRATEGY.md) ===\n" + strategy + "\n"
        "=== КОНЕЦ СТРАТЕГИИ ===\n\n"
        "=== BASELINE_MODEL.md ===\n" + baseline + "\n=== КОНЕЦ ===\n\n"
        "=== TARGETS_SPLITS_METRICS.md ===\n" + targets + "\n=== КОНЕЦ ===\n\n"
        "Более подробный (497 строк) план — Models/docs/roadmaps/BASELINE_TO_TRAINING_ROADMAP.md — "
        "прочитай его сам через Read, когда понадобятся детали по конкретному этапу, не полагайся "
        "только на выжимку выше. Контракт Encoder'а (для v1, НЕ для сегодняшней задачи) — "
        "Models/docs/contracts/ENCODER_CONTRACT.md, читать только если станет актуально.\n\n"
        "У тебя есть Bash/Read/Write/Edit/Glob/Grep/WebSearch/WebFetch + управление GPU-подом "
        "(mcp__trendflow__pod_control/manager — тот же контур, что у остальных агентов, "
        "mcp__trendflow__budget_status/limits_status — проверяй перед длинными прогонами).\n\n"
        "ВИЗУАЛИЗАЦИИ: полезный график (распределение фичей, feature importance, метрики по "
        "возрастным бакетам) — сохрани PNG в " + str(OUTBOX_DIR) + ", уйдёт в VK автоматически как "
        "фото после ответа.\n\n"
        "ПРОМЕЖУТОЧНЫЕ СООБЩЕНИЯ: работа долгая (сборка датасета, обучение) — присылай короткие "
        "пометки по ходу (что делаешь сейчас), не молчи по 10+ минут без единой строки.\n\n"
        "К владельцу — только за реальными решениями (данные, деньги/лицензии, юридические вопросы, "
        "продуктовая развилка с высокой ценой ошибки). Рутинные технические решения — сама, без "
        "вопросов. Пиши по-русски, кратко в чате — развёрнутое содержание в файлы/README."
    )


def _log_chat(direction: str, text: str) -> None:
    try:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {direction} {text[:1500]}\n"
        with open(CHAT_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def send(text: str) -> None:
    _log_chat("MODELSBOT->", text)
    try:
        requests.post(f"{VK}/messages.send", data={
            "user_id": config.VK_OWNER_ID, "random_id": random.randint(1, 2_000_000_000),
            "message": text[:4000], "access_token": config.VK_TOKEN2, "v": config.VK_API_VERSION,
        }, timeout=40)
    except requests.RequestException:
        pass


def send_photo(path, caption: str = "") -> None:
    """Тот же флоу, что messenger.send_photo, но для VK_TOKEN2 (свой бот)."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise RuntimeError(f"send_photo: файл отсутствует или пуст: {p}")
    def _api(method, **params):
        params.setdefault("access_token", config.VK_TOKEN2)
        params.setdefault("v", config.VK_API_VERSION)
        r = requests.post(f"{VK}/{method}", data=params, timeout=60)
        return r.json().get("response")
    upload = _api("photos.getMessagesUploadServer", peer_id=config.VK_OWNER_ID)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    with open(p, "rb") as f:
        r = requests.post(upload["upload_url"], files={"photo": (p.name, f, mime)}, timeout=60)
    uploaded = r.json()
    if not uploaded.get("photo"):
        raise RuntimeError(f"send_photo: невалидный ответ upload-сервера: {uploaded}")
    saved = _api("photos.saveMessagesPhoto", photo=uploaded["photo"], server=uploaded["server"], hash=uploaded["hash"])
    photo = saved[0] if isinstance(saved, list) else saved
    _log_chat("MODELSBOT->", f"[фото] {p}" + (f" — {caption}" if caption else ""))
    _api("messages.send", user_id=config.VK_OWNER_ID, random_id=random.randint(1, 2_000_000_000),
        message=caption[:4000], attachment=f"photo{photo['owner_id']}_{photo['id']}")


def _flush_outbox_photos() -> None:
    for path in sorted(OUTBOX_DIR.iterdir()):
        if path.is_dir() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            send_photo(path, caption=path.stem.replace("_", " "))
            path.rename(OUTBOX_SENT_DIR / path.name)
        except Exception as e:
            print(f"[models_agent] не удалось отправить фото {path.name}: {e}", flush=True)


def _send_long(text: str) -> None:
    text = text.strip()
    if not text:
        return
    while text:
        if len(text) <= VK_MAX_LEN:
            send(text)
            return
        cut = text.rfind("\n\n", 0, VK_MAX_LEN)
        if cut <= 0:
            cut = text.rfind(" ", 0, VK_MAX_LEN)
        if cut <= 0:
            cut = VK_MAX_LEN
        send(text[:cut])
        text = text[cut:].strip()


class LP:
    def __init__(self):
        r = self._api("groups.getById")
        g = r["groups"][0] if isinstance(r, dict) and "groups" in r else (r[0] if isinstance(r, list) else r)
        self.gid = int(g["id"])
        self._refresh()

    def _api(self, method, **p):
        p.setdefault("access_token", config.VK_TOKEN2)
        p.setdefault("v", config.VK_API_VERSION)
        return requests.post(f"{VK}/{method}", data=p, timeout=40).json()["response"]

    def _refresh(self):
        s = self._api("groups.getLongPollServer", group_id=self.gid)
        self.server, self.key, self.ts = s["server"], s["key"], s["ts"]

    def poll_once(self, wait: int = 25) -> list[str]:
        try:
            d = requests.get(self.server, params={"act": "a_check", "key": self.key,
                             "ts": self.ts, "wait": wait}, timeout=wait + 10).json()
        except requests.RequestException:
            return []
        if "failed" in d:
            if d["failed"] == 1:
                self.ts = d.get("new_ts", self.ts)
            else:
                self._refresh()
            return []
        self.ts = d.get("ts", self.ts)
        out = []
        for u in d.get("updates", []):
            if u.get("type") == "message_new":
                m = u["object"]["message"]
                if int(m.get("from_id", 0)) == config.VK_OWNER_ID and m.get("text"):
                    out.append(m["text"].strip())
        return out


async def _safe_interrupt(client: ClaudeSDKClient) -> None:
    """Вызывается фоновой задачей из _producer (см. баг там) — НИКОГДА не await'ится напрямую в
    цикле опроса VK, чтобы долгий/подвисший control-request к CLI не блокировал приём сообщений."""
    try:
        await client.interrupt()
    except Exception as e:
        print(f"[models_agent] interrupt не сработал: {e}", flush=True)


_TOOL_INPUT_KEYS = ("command", "file_path", "pattern", "path", "prompt", "url", "description")

# Баг найден 2026-07-23 (владелец, симметрично deepdive_agent.py): эхо на КАЖДЫЙ ToolUseBlock при
# интенсивной автономной работе (десятки вызовов в ход) топит реальный статус в потоке сообщений
# + вероятный троттлинг VK на слишком частую отправку. Троттлим только tool-эхо (не TextBlock —
# это реальный контент).
TOOL_ECHO_MIN_INTERVAL_SEC = 2.5
_last_tool_echo_at = 0.0


def _should_send_tool_echo() -> bool:
    global _last_tool_echo_at
    now = time.monotonic()
    if now - _last_tool_echo_at < TOOL_ECHO_MIN_INTERVAL_SEC:
        return False
    _last_tool_echo_at = now
    return True


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
    """Баг найден 2026-07-19 (тот же, что в deepdive_agent.py/watchdog.py): раньше весь текст хода
    копился и уходил в VK одним сообщением в конце — при долгой работе (сборка датасета, обучение)
    в чате была тишина по 10+ минут. Теперь каждый TextBlock уходит сразу короткой пометкой.

    Второй баг найден 2026-07-19 (в этом же разговоре): если во время хода приходит НОВОЕ сообщение
    от владельца, скрипт физически не мог его увидеть — poll_once() вызывался только ПОСЛЕ полного
    завершения текущего хода (см. main()). Теперь ход может быть прерван снаружи через
    client.interrupt() (см. _producer в main()) — этот except ловит обрыв стрима, который interrupt
    вызывает, и не считает это ошибкой.

    Третий баг найден 2026-07-20 (владелец, тот же в deepdive_agent.py): каждый блок уходил в VK
    урезанным до 200 симв., а в конце хода ВЕСЬ текст уходил ЕЩЁ РАЗ целиком — дублирование каждой
    мысли. Теперь блок уходит сразу и полностью (через _send_long), финального повторного прохода
    нет.

    Четвёртый баг найден 2026-07-21 (владелец, снова "зависает" — после того, как боту дали
    многодневную работу с длинными сериями Bash/Read/Write без единого TextBlock между ними). В VK не
    уходило вообще ничего на протяжении такой серии — процесс жив, но со стороны VK неотличимо от
    зависания. Теперь каждый ToolUseBlock тоже шлёт короткую строку-эхо."""
    _log_chat("OWNER->", text)
    await client.query(text)
    try:
        async for msg in client.receive_response():
            sid = getattr(msg, "session_id", None)
            if sid:
                session_state.save(AGENT_NAME, sid)
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        # Баг найден 2026-07-21 (симметрично deepdive_agent.py): DNS/сетевой сбой
                        # api.vk.com внутри _send_long раньше улетал в общий except ниже и обрывал
                        # ВЕСЬ остаток хода, не только этот один месседж. Теперь не фатально.
                        try:
                            _send_long(f"⏳ {block.text.strip()}")
                        except Exception as e:
                            print(f"[models_agent] сообщение не отправилось (сеть?): {e}", flush=True)
                    elif isinstance(block, ToolUseBlock):
                        print(f"[models_agent] tool: {_summarize_tool_use(block)}", flush=True)
                        if _should_send_tool_echo():
                            try:
                                send(_summarize_tool_use(block))
                            except Exception as e:
                                print(f"[models_agent] tool-echo не отправился: {e}", flush=True)
            elif isinstance(msg, ResultMessage):
                cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
                print(f"[models_agent] ход завершён, cost=${cost:.4f}", flush=True)
    except Exception as e:
        print(f"[models_agent] ход прерван: {e}", flush=True)
    _flush_outbox_photos()


async def main(model_name: str | None) -> None:
    config.require("VK_TOKEN2", config.VK_TOKEN2)
    config.require("VK_OWNER_ID", config.VK_OWNER_ID)
    model = settings.resolve_model(model_name) if model_name else config.AGENT_MODEL
    print(f"[models_agent] запуск, модель={model}...", flush=True)
    lp = LP()
    resume_id = session_state.load(AGENT_NAME)
    send(f"🧮 Models Bot на связи (модель {model})." + (
        " Восстанавливаю прошлую сессию — контекст не потерян."
        if resume_id else
        " Собираю DatasetBuilder и первый baseline на реальных данных."
    ) + " Пиши как в обычном чате.")
    print(f"[models_agent] готов, слушаю VK (resume={resume_id!r})", flush=True)
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
        # guard_bash СНЯТ 2026-07-21 (симметрично с deepdive_agent.py, снято там 2026-07-19 по прямой
        # просьбе владельца): вызывал supervisor.answer() на "опасные" паттерны (pip install torch,
        # huggingface-cli download без --include, snapshot_download без allow_patterns и т.д.) —
        # ровно то, что понадобится Agent B для анализа фичей (pip install pandas/sklearn/xgboost,
        # монтирование Network Volume, возможные HF-загрузки). supervisor.py — часть СТАРОГО
        # assistant.py-флоу ("Второй агент" в комментариях hooks.py), который сейчас не запущен как
        # сервис -> вызов рисковал зависать так же, как раньше зависал deepdive_agent.py. Полный
        # автономный доступ ко всем командам без исключений.
        # Баг найден 2026-07-23 (симметрично deepdive_agent.py): дефолтный лимит SDK на JSON-
        # сообщение от CLI-подпроцесса — 1MB ("Fatal error in message reader" — роняет клиент).
        # Подняли с запасом.
        max_buffer_size=20 * 1024 * 1024,  # 20MB
        resume=resume_id,
    )
    while True:
        try:
            async with ClaudeSDKClient(options=options) as client:
                if options.resume:
                    await _handle_turn(client, CONTINUE_AFTER_RESTART_MESSAGE)
                else:
                    send("▶️ Начинаю с DatasetBuilder — подробности в системном промпте.")
                    await _handle_turn(client, FIRST_TASK_MESSAGE)

                # Баг найден 2026-07-19: poll_once() раньше вызывался ТОЛЬКО между полностью
                # завершёнными ходами — если владелец писал во время долгого хода (сборка
                # датасета, обучение — сотни tool-вызовов подряд без остановки), сообщение
                # физически не читалось, пока ход сам не закончится. Теперь VK слушается
                # НЕПРЕРЫВНО отдельной задачей (_producer): если приходит сообщение, а бот занят
                # (busy) — зовём client.interrupt() (штатная возможность SDK для streaming-режима),
                # текущий receive_response() в _handle_turn обрывается, и новое сообщение уходит в
                # обработку сразу, а не после неопределённо долгого ожидания.
                queue: asyncio.Queue[str] = asyncio.Queue()
                busy = asyncio.Event()

                async def _producer() -> None:
                    """Баг найден 2026-07-19: раньше был fire-and-forget asyncio.create_task —
                    если lp.poll_once() кидал ЛЮБОЕ исключение мимо requests.RequestException
                    (например json.JSONDecodeError на кривом ответе VK), задача молча умирала.
                    Внешний try/except в main() её не видел (это отдельная Task, не await в
                    основном потоке) — VK переставал слушаться НАВСЕГДА до ручного рестарта, при
                    этом консьюмер продолжал жить на том, что уже было в очереди, создавая
                    иллюзию, что бот просто "не реагирует", хотя на деле не мог физически увидеть
                    новые сообщения. Теперь опрос обёрнут в try/except, который никогда не даёт
                    задаче умереть."""
                    while True:
                        try:
                            texts = await asyncio.to_thread(lp.poll_once, 25)
                        except Exception as e:
                            print(f"[models_agent] poll_once упал, жду и пробую снова: {e}", flush=True)
                            await asyncio.sleep(3)
                            continue
                        for t in texts:
                            if busy.is_set():
                                send("⏸ Прерываю текущую работу — читаю новое сообщение.")
                                # Баг найден 2026-07-21 (симметрично deepdive_agent.py): раньше
                                # `await client.interrupt()` стоял ПРЯМО тут, внутри _producer —
                                # единственного цикла, который вообще читает VK. SDK ждёт ответа
                                # CLI-подпроцесса на control-request до 60с; если CLI занят долгим
                                # Bash (git/pod_control — именно то, чем бот теперь занимается часами),
                                # interrupt() блокировал ВЕСЬ _producer на эти секунды/до минуты —
                                # новые сообщения физически не вычитывались из VK. Выглядело как
                                # "зависает, пока не напишу ещё раз" — на деле не повторное сообщение
                                # чинило зависание, а просто совпадало по времени с концом блокировки.
                                # Теперь interrupt — фоновая задача, _producer никогда на нём не стоит.
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
                            print(f"[models_agent] ошибка обработки сообщения: {e}", flush=True)
                            try:
                                send(f"❗ Ошибка: {e}")
                            except Exception:
                                pass
                        finally:
                            busy.clear()
                finally:
                    producer_task.cancel()
        except Exception as e:
            print(f"[models_agent] сессия оборвалась, переоткрываю: {e}", flush=True)
            try:
                send(f"⚠️ Сессия оборвалась ({e}), переоткрываю с тем же контекстом.")
            except Exception:
                pass
            # Резюмируем с последнего известного session_id (мог обновиться за время работы) —
            # переоткрытие сессии ВНУТРИ процесса тоже не должно терять контекст, не только
            # полный рестарт процесса.
            options.resume = session_state.load(AGENT_NAME) or options.resume
            await asyncio.sleep(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="Алиас модели (opus/sonnet/haiku/fable) или сырой id.")
    args = ap.parse_args()
    try:
        asyncio.run(main(args.model))
    except KeyboardInterrupt:
        print("\n[models_agent] стоп")
