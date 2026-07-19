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
import mimetypes
import random
from pathlib import Path

import requests
from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions, HookMatcher, AssistantMessage, TextBlock, ResultMessage,
)

import config
import settings
import tools
import hooks

VK = "https://api.vk.com/method"

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


async def _handle_turn(client: ClaudeSDKClient, text: str) -> None:
    _log_chat("OWNER->", text)
    await client.query(text)
    parts: list[str] = []
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    parts.append(block.text.strip())
        elif isinstance(msg, ResultMessage):
            cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
            print(f"[models_agent] ход завершён, cost=${cost:.4f}", flush=True)
    if parts:
        _send_long("\n\n".join(parts))
    _flush_outbox_photos()


async def main(model_name: str | None) -> None:
    config.require("VK_TOKEN2", config.VK_TOKEN2)
    config.require("VK_OWNER_ID", config.VK_OWNER_ID)
    model = settings.resolve_model(model_name) if model_name else config.AGENT_MODEL
    print(f"[models_agent] запуск, модель={model}...", flush=True)
    lp = LP()
    send(f"🧮 Models Bot на связи (модель {model}). Собираю DatasetBuilder и первый baseline "
        "на реальных данных. Пиши как в обычном чате.")
    print("[models_agent] готов, слушаю VK", flush=True)
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
    first_run = True
    while True:
        try:
            async with ClaudeSDKClient(options=options) as client:
                if first_run:
                    send("▶️ Начинаю с DatasetBuilder — подробности в системном промпте.")
                    await _handle_turn(client, FIRST_TASK_MESSAGE)
                    first_run = False
                while True:
                    texts = await asyncio.to_thread(lp.poll_once, 25)
                    for t in texts:
                        try:
                            await _handle_turn(client, t)
                        except Exception as e:
                            print(f"[models_agent] ошибка обработки сообщения: {e}", flush=True)
                            try:
                                send(f"❗ Ошибка: {e}")
                            except Exception:
                                pass
        except Exception as e:
            print(f"[models_agent] сессия оборвалась, переоткрываю: {e}", flush=True)
            try:
                send(f"⚠️ Сессия оборвалась ({e}), переоткрываю. Прогресс в файлах не теряется.")
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
        print("\n[models_agent] стоп")
