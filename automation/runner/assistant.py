#!/usr/bin/env python3
"""Второй агент — разговорный + контролирующий (VK_TOKEN2). Полный доступ: bash/файлы/интернет/поды.

Роль (см. AGENT_CONTEXT.md раздел 0):
  1. ГЛАВНЫЙ ПРИОРИТЕТ — отвечать на вопросы Первого агента (component-runner). Технически это делает
     supervisor.answer() СИНХРОННО внутри процесса agent_runner.py (латентность критична — Первый
     агент ждёт), но это тот же "Второй агент" по смыслу: пока идёт такой ответ, здесь виден
     busy-флаг (state/agent2_busy.json), и владельцу вместо тишины прилетает заготовленный ответ.
  2. Периодически (по таймеру) — контролирует Первого: логи, чат, штампы, поды, бюджет; если видит
     проблему — просит Первого аккуратно остановиться (сохранив контекст), чинит код, перезапускает.
     Сообщает владельцу только в критичных случаях — по умолчанию старается решить сама.
  3. Разговорный ассистент владельцу: код/доки/задачи параллельно с работой Первого агента.

Отдельный процесс от основного раннера. Команды в чате: /model <opus|sonnet|haiku|fable|...>,
/new (сбросить контекст), /monitor [on|off|<минуты>] (управление периодическим контролем), /help.

Запуск: python assistant.py   (нужен VK_TOKEN2 в .env; напиши боту 'start' один раз)
"""
from __future__ import annotations
import asyncio
import datetime as dt
import json
import random
import requests

from claude_agent_sdk import (
    query, ClaudeSDKClient, ClaudeAgentOptions, HookMatcher,
    AssistantMessage, TextBlock, ResultMessage,
)

import config
import settings
import limits
import agents
import hooks

VK = "https://api.vk.com/method"

BUSY_REPLY = "Я сейчас отвечаю основному агенту, вам отвечу сразу после."

SYSTEM = (
    "Ты — Второй агент проекта TrendFlow, общаешься с владельцем в VK. Полный доступ: "
    "Bash/Read/Write/Edit, WebSearch/WebFetch, PodManager (CLI: `python podmanager.py list|health <id>`). "
    "ГЛАВНАЯ РОЛЬ (по приоритету): (1) быстрые ответы Первому агенту на его вопросы — технически это "
    "делает supervisor.py синхронно внутри процесса agent_runner.py, пока он занят — ты видишь "
    "state/agent2_busy.json и владелец получает заготовленный ответ вместо тишины; (2) периодически "
    "контролировать Первого (логи/чат/штампы/поды/бюджет — automation/runner/AGENT_CONTEXT.md раздел 6-7, "
    "state/agent1_chat.log, state/last_session.md, state/sessions.csv, DataProcessor/docs/"
    "COMPONENT_VALIDATION_CHECKLIST.md, state/hook_decisions.log, state/spend_log.csv, state/pod_ledger.csv) "
    "и если что-то не так — попросить Первого аккуратно остановиться (запиши JSON "
    "{\"reason\": \"...\"} в state/assistant_stop_request.json — Первый сохранит контекст и погасит под), "
    "починить код, затем перезапустить (запиши {\"text\": \"...\"} в state/assistant_start_request.json, "
    "пустой text = следующий компонент из очереди). ЕСТЬ ТРЕТИЙ, МЯГКИЙ вариант для несрочных указаний "
    "(предупредить о чём-то, попросить учесть нюанс, не требующий остановки): запиши "
    "{\"text\": \"...\", \"from\": \"Второго агента\"} в state/live_note.json — доставится Первому через "
    "system-подсказку на следующем Bash-вызове (обычно секунды), БЕЗ остановки/потери контекста. "
    "Используй stop/start ТОЛЬКО когда реально нужно вмешаться в работу (баг, зависание), а не для "
    "просто «сказать что-то»; (3) быстрые правки КОДА и ЛОГИКИ Первого агента — "
    "файлы в automation/runner/ (AGENT_CONTEXT.md, agent_runner.py, tools.py, budget.py, limits.py и т.д.) и "
    "код компонентов DataProcessor. Также помогаешь с любыми вопросами по проекту (контекст — CLAUDE.md, "
    "MAIN_INDEX.md, automation/runner/AGENT_CONTEXT.md). "
    "ЧТО ДЕЛАЕТ ПЕРВЫЙ АГЕНТ: узнавай из файлов — `tail -100 state/agent1_chat.log`, `cat state/last_session.md`, "
    "`ls state/progress/`, `cat state/agents.json` (кто активен), `tail state/sessions.csv`. НЕ говори "
    "«не вижу его» — просто прочитай эти файлы. "
    "ВАЖНО: Первый агент работает НЕПРЕРЫВНО по ВСЕМ компонентам очереди без ожидания владельца между "
    "ними — это нормально, не повод для тревоги. Вмешивайся (стоп/фикс/рестарт), только если РЕАЛЬНО "
    "что-то сломано (зацикливание, повторяющиеся ошибки, штамп без прогона, аномальные траты, опасные "
    "команды) — старайся решать сама, владельцу пиши редко, только в критичных случаях. "
    "Опасные bash-команды (rm -rf, git push принудительный и т.п.) — как у Первого агента, через хук. "
    "КОМАНДА: рядом работает Первый агент (валидирует компоненты на GPU). Общий лимит Claude ДЕЛИТСЯ между "
    "вами — учитывай чужую нагрузку (limits_status/`python limits.py`). "
    "ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ, кратко, экономь токены. Действуй сама; подтверждение владельца — лишь на "
    "необратимое (удаление важных данных, публикации)."
)

MONITOR_PROMPT = (
    "Проведи периодическую проверку Первого агента (component-runner). Прочитай: "
    "`tail -150 state/agent1_chat.log`, `cat state/last_session.md`, `tail -20 state/sessions.csv`, "
    "`tail -20 state/spend_log.csv state/pod_ledger.csv`, `tail -30 state/hook_decisions.log` (если есть), "
    "недавние изменения в `state/progress/*.md`, статусы в `DataProcessor/docs/COMPONENT_VALIDATION_CHECKLIST.md`, "
    "`git status`/`git diff --stat` на некоммиченные изменения. "
    "Реши: всё штатно (в т.ч. штамп после 100%-PASS без вопроса владельцу — это НОРМА) — тогда НИЧЕГО не делай "
    "и просто заверши коротким итогом (1 строка, НЕ пиши владельцу). Есть РЕАЛЬНАЯ проблема (зацикливание на "
    "одном компоненте много сессий подряд, повторяющаяся одна и та же ошибка, штамп без реального прогона/чисел, "
    "аномальный расход GPU-часов, признаки того, что агент сломан/завис) — тогда: "
    "(1) запиши {\"reason\": \"<кратко>\"} в state/assistant_stop_request.json (Первый сам аккуратно "
    "остановится: сохранит контекст, погасит под — подожди ~60-90 секунд и проверь по chat log, что "
    "остановка произошла); "
    "(2) разберись в коде/логах, почини, если нашла причину; "
    "(3) перезапусти: запиши {\"text\": \"\"} (или ctx:/verify:<comp> при необходимости) в "
    "state/assistant_start_request.json; "
    "(4) У тебя НЕТ инструмента отправки в VK — сообщить владельцу можно только так: если ситуация "
    "реально критичная (потеряны деньги, риск данных, нужен продуктовый выбор), НАЧНИ свой самый "
    "последний ответ строкой 'КРИТИЧНО: <суть в 1-2 предложения>' — код снаружи прочитает это и "
    "перешлёт в VK сам. Рутинный автономный фикс НЕ требует этой пометки — просто резюме без неё. "
    "Пиши по-русски, кратко."
)

HELP = ("Я Второй агент. Команды: /model <имя>, /limits, /monitor [on|off|<минуты>], /new (сброс контекста), "
       "/help. Просто пиши вопрос/задачу — отвечу, если не занята ответом Первому агенту.")


def send(text: str):
    try:
        requests.post(f"{VK}/messages.send", data={
            "user_id": config.VK_OWNER_ID, "random_id": random.randint(1, 2_000_000_000),
            "message": text[:4000], "access_token": config.VK_TOKEN2, "v": config.VK_API_VERSION,
        }, timeout=40)
    except requests.RequestException:
        pass


def _api(method, **p):
    p.setdefault("access_token", config.VK_TOKEN2)
    p.setdefault("v", config.VK_API_VERSION)
    return requests.post(f"{VK}/{method}", data=p, timeout=40).json()


class LP:
    def __init__(self):
        r = _api("groups.getById")["response"]
        g = r["groups"][0] if isinstance(r, dict) and "groups" in r else (r[0] if isinstance(r, list) else r)
        self.gid = int(g["id"])
        self._refresh()

    def _refresh(self):
        s = _api("groups.getLongPollServer", group_id=self.gid)["response"]
        self.server, self.key, self.ts = s["server"], s["key"], s["ts"]

    def poll(self, wait=25):
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


# --------------------------------------------------------------- busy-флаг (агент 1 сейчас отвечает)
def _agent1_busy() -> bool:
    """True, если сейчас идёт синхронный ответ Первому агенту (supervisor.answer() в другом процессе).
    Флаг с TTL — если протух (процесс упал не почистив), не считаем busy вечно."""
    if not config.AGENT2_BUSY_FLAG.exists():
        return False
    try:
        data = json.loads(config.AGENT2_BUSY_FLAG.read_text(encoding="utf-8"))
        since = dt.datetime.fromisoformat(data.get("since"))
        return (dt.datetime.now() - since).total_seconds() < 300  # 5 мин TTL — защита от протухшего флага
    except Exception:
        return False


# --------------------------------------------------------------------- разговор с владельцем
_conv_lock = asyncio.Lock()


def _opts(model: str) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=model, system_prompt=SYSTEM, cwd=str(config.REPO_DIR),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
        permission_mode="bypassPermissions", max_turns=60,
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[hooks.guard_bash])]},
    )


async def _answer(model: str, history: list[str], text: str) -> str:
    # Короткая память: подставляем последние реплики в промпт (одноразовый query).
    ctx = ("\n".join(history[-6:]) + "\n\n") if history else ""
    prompt = f"{ctx}Пользователь: {text}"
    parts = []
    cost, tin, tout = 0.0, 0, 0
    async with _conv_lock:
        async for msg in query(prompt=prompt, options=_opts(model)):
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        parts.append(b.text.strip())
            elif isinstance(msg, ResultMessage):
                cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
                u = getattr(msg, "usage", None)
                if isinstance(u, dict):
                    tin, tout = int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)
    limits.record(f"assistant:{model}", tin, tout, cost)  # общий учёт лимитов
    return "\n".join(parts) if parts else "(готово)"


# --------------------------------------------------------------------- периодический контроль Первого
_last_monitor_msg = ""


async def _monitor_pass(model: str) -> str:
    """Один проход контроля: читает состояние Первого агента, при необходимости просит стоп,
    правит код, перезапускает. Возвращает короткий итог (для логов, НЕ обязательно для VK)."""
    parts = []
    cost, tin, tout = 0.0, 0, 0
    async with _conv_lock:
        async with ClaudeSDKClient(options=ClaudeAgentOptions(
            model=model, system_prompt=SYSTEM, cwd=str(config.REPO_DIR),
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
            permission_mode="bypassPermissions", max_turns=40,
            hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[hooks.guard_bash])]},
        )) as client:
            await client.query(MONITOR_PROMPT)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock) and b.text.strip():
                            parts.append(b.text.strip())
                elif isinstance(msg, ResultMessage):
                    cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
                    u = getattr(msg, "usage", None)
                    if isinstance(u, dict):
                        tin, tout = int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)
    limits.record(f"assistant-monitor:{model}", tin, tout, cost)
    return "\n".join(parts) if parts else "(проверка без замечаний)"


async def _monitor_loop(model_getter):
    """Фоновая задача: раз в assistant_monitor_interval_sec проверяет Первого агента, если
    assistant_monitor_enabled=1 и не идёт разговор/ответ Первому прямо сейчас."""
    global _last_monitor_msg
    while True:
        interval = int(settings.get("assistant_monitor_interval_sec") or 1200)
        await asyncio.sleep(max(60, interval))
        if not int(settings.get("assistant_monitor_enabled") or 0):
            continue
        if _agent1_busy():
            continue  # не мешаем синхронному ответу Первому
        try:
            _last_monitor_msg = await _monitor_pass(model_getter())
            print(f"[assistant][monitor] {_last_monitor_msg[:300]}", flush=True)
            # Единственный канал "сообщить владельцу" у контролирующей сессии — маркер в тексте
            # (у неё нет VK-инструмента); код сам решает, слать в VK или нет.
            if _last_monitor_msg.strip().upper().startswith("КРИТИЧНО"):
                send(f"🚨 Второй агент (контроль): {_last_monitor_msg[:1000]}")
        except Exception as e:
            print(f"[assistant][monitor] ошибка: {e}", flush=True)


def _handle_monitor_command(t: str) -> str:
    parts = t.split(None, 1)
    arg = parts[1].strip().lower() if len(parts) > 1 else ""
    if not arg:
        en = int(settings.get("assistant_monitor_enabled") or 0)
        iv = int(settings.get("assistant_monitor_interval_sec") or 1200)
        last = f"\nПоследняя проверка: {_last_monitor_msg[:300]}" if _last_monitor_msg else ""
        return f"Контроль: {'вкл' if en else 'выкл'}, раз в {iv//60} мин.{last}"
    if arg == "on":
        settings.set_("assistant_monitor_enabled", 1)
        return "✅ Периодический контроль включён."
    if arg == "off":
        settings.set_("assistant_monitor_enabled", 0)
        return "🔕 Периодический контроль выключен."
    if arg.isdigit():
        settings.set_("assistant_monitor_interval_sec", int(arg) * 60)
        return f"✅ Интервал контроля: {arg} мин."
    return "Формат: /monitor [on|off|<минуты>]"


async def main():
    if not config.VK_TOKEN2:
        raise SystemExit("Нет VK_TOKEN2 в .env")
    print("[assistant] запуск, инициализирую long poll…", flush=True)
    lp = LP()
    model = config.ASSISTANT_MODEL
    history: list[str] = []
    pending: list[str] = []  # сообщения владельца, пришедшие пока была занята ответом Первому
    send("🤝 Второй агент на связи. /help — команды.")
    print("[assistant] готов, слушаю VK", flush=True)

    monitor_task = asyncio.create_task(_monitor_loop(lambda: model))

    try:
        while True:
            agents.heartbeat("assistant", model)
            msgs = await asyncio.to_thread(lp.poll, 25)
            for t in msgs:
                print(f"[assistant] < {t[:120]}", flush=True)
                low = t.lower()
                if low.startswith("/model"):
                    arg = t.split(None, 1)[1].strip() if len(t.split(None, 1)) > 1 else ""
                    if arg:
                        model = settings.resolve_model(arg)
                        send(f"✅ Модель: {model}")
                    else:
                        send(f"Текущая: {model}")
                elif low == "/new":
                    history.clear()
                    send("🧹 Контекст сброшен.")
                elif low == "/limits":
                    send(limits.status_text())
                elif low.startswith("/monitor"):
                    send(_handle_monitor_command(t))
                elif low == "/scanlimits":
                    import subprocess, sys as _s
                    send("🌐 Снимаю лимиты с claude.ai…")
                    cmd = [_s.executable, str(config.RUNNER_DIR / "claude_limits_scraper.py")]
                    cmd += (["--cdp", config.CLAUDE_CDP_PORT] if config.CLAUDE_CDP_PORT else ["--headless"])
                    try:
                        await asyncio.to_thread(lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=150))
                    except Exception as e:
                        send(f"Ошибка парсера: {e}")
                    send(limits.status_text())
                elif low == "/help":
                    send(HELP)
                elif _agent1_busy():
                    # Не LLM — заготовленный код-ответ, чтобы владелец не молчал в тишину.
                    send(BUSY_REPLY)
                    pending.append(t)
                else:
                    pending.append(t)

            # Обрабатываем накопленные сообщения владельца, если сейчас свободна.
            if pending and not _agent1_busy():
                to_process, pending = pending, []
                for t in to_process:
                    try:
                        ans = await _answer(model, history, t)
                    except Exception as e:
                        ans = f"❗ Ошибка: {e}"
                    history.append(f"Пользователь: {t}")
                    history.append(f"Ты: {ans}")
                    send(ans)
                    print(f"[assistant] > {ans[:120]}", flush=True)
    finally:
        monitor_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[assistant] стоп")
