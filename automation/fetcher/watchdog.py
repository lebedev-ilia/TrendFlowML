#!/usr/bin/env python3
"""Агент-наблюдатель Fetcher (Третий бот, VK_TOKEN3, модель Haiku).

Раз в час (WATCHDOG_INTERVAL_SEC): SSH на все 3 пода, смотрит логи (discover.log, workers_*.log)
и state/inventory/summary.json (см. deploy.py), решает — всё штатно (молчит) или есть проблема
(зацикливание, ошибка ключей/квоты сверх нормы, аномальный рост queue_dead_letter, процесс не
отвечает и т.п.) — тогда пишет брифинг с диагнозом и рекомендацией, а если может — сама чинит код
локально (в этом репозитории, источник истины) и перезапускает процесс на поде (git push -> git pull
на поде -> kill+restart через deploy.py).

Также слушает VK на простые команды (/status, /help) — не полноценный диалог (для этого нет задачи),
но минимальная отзывчивость.

НИКОГДА не трогает поды ML-раннера (automation/runner/) — работает только со своими тремя.
"""
from __future__ import annotations
import asyncio
import datetime as dt
import random
import time

import requests
from claude_agent_sdk import (
    query, ClaudeSDKClient, ClaudeAgentOptions, HookMatcher,
    AssistantMessage, TextBlock, ResultMessage,
)

import config

VK = "https://api.vk.com/method"

SYSTEM = (
    "Ты — Третий агент проекта TrendFlow: наблюдатель ЗА FETCHER DATASET COLLECTOR (сбор датасета "
    "YouTube на 3 постоянных CPU-подах RunPod: fetcher-main — discover+свой шард workers, "
    "fetcher-worker-b/-c — только workers). Это ОТДЕЛЬНАЯ система от ML-валидации компонентов "
    "(automation/runner/) — НИКОГДА не трогай ML-поды/файлы automation/runner/, это не твоя зона. "
    "У тебя есть: Bash/Read/Write/Edit в этом репозитории (automation/fetcher/ — твой код: "
    "deploy.py содержит готовые функции ssh_run/tail_logs/read_inventory_summary/kill_processes/"
    "pull_latest_code/launch — используй их через `python3 -c \"import deploy; ...\"` из директории "
    "automation/fetcher/, не изобретай SSH-команды заново). Секреты — automation/fetcher/.env. "
    "SSH-ключ — automation/fetcher/ssh/id_ed25519 (тот же паблик-ключ уже прописан RunPod автоматически "
    "во все поды аккаунта). Пароли/ID подов — automation/fetcher/state/provision_result.json. "
    "ПРОТОКОЛ ПРОВЕРКИ (раз в час): для каждого из 3 подов — deploy.tail_logs(pod) на ошибки "
    "(QuotaExceededError — это НОРМА, код сам ждёт сброса, не считай багом; реальные проблемы — "
    "traceback без восстановления, растущий queue_dead_letter, процесс не пишет в лог часами, "
    "'yt-dlp enrich failed' резкий скачок, bot-detection постоянный) и deploy.read_inventory_summary(pod) "
    "на lag_* метрики (растущий lag без движения — подозрительно). Всё штатно — НИЧЕГО не делай, "
    "заверши коротким итогом (1 строка). Есть проблема: (1) разберись в причине по логам/коду; "
    "(2) если можешь починить — правь код В ЭТОМ РЕПО (Fetcher/fetcher/dataset_collector/...), "
    "закоммить и запушь (git push — разрешено), затем deploy.pull_latest_code(pod) + "
    "deploy.kill_processes(pod) + deploy.launch(pod, hf_token) для применения фикса; "
    "(3) сообщить владельцу можно ТОЛЬКО через маркер — НАЧНИ последний ответ строкой "
    "'БРИФИНГ: <диагноз в 2-4 предложения + рекомендация>' если ситуация требует внимания владельца "
    "(нужен новый набор YouTube-ключей, кампания зависла необъяснимо, RunPod-инфра сломана) — рутинный "
    "автономный фикс НЕ требует брифинга, просто резюме без маркера. "
    "Пиши по-русски, кратко."
)

HELP = "Я наблюдатель Fetcher. /status — проверить сейчас, /help — справка."


def _log_chat(direction: str, text: str) -> None:
    try:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {direction} {text[:1500]}\n"
        with open(config.STATE_DIR / "fetcher_chat.log", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def send(text: str) -> None:
    _log_chat("WATCHDOG->", text)
    try:
        requests.post(f"{VK}/messages.send", data={
            "user_id": config.VK_OWNER_ID, "random_id": random.randint(1, 2_000_000_000),
            "message": text[:4000], "access_token": config.VK_TOKEN3, "v": config.VK_API_VERSION,
        }, timeout=40)
    except requests.RequestException:
        pass


def _api(method, **p):
    p.setdefault("access_token", config.VK_TOKEN3)
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


CHECK_PROMPT = (
    "Проведи часовую проверку Fetcher dataset collector по протоколу из системного промпта — "
    "все 3 пода (fetcher-main, fetcher-worker-b, fetcher-worker-c). Начни с "
    "`cd automation/fetcher && python3 -c \"import deploy; print(deploy.tail_logs('fetcher-main'))\"` "
    "и аналогично для остальных подов + summary.json."
)


async def _check_pass(model: str) -> str:
    parts = []
    cost, tin, tout = 0.0, 0, 0
    async with ClaudeSDKClient(options=ClaudeAgentOptions(
        model=model, system_prompt=SYSTEM, cwd=str(config.REPO_DIR),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions", max_turns=40,
    )) as client:
        await client.query(CHECK_PROMPT)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        parts.append(b.text.strip())
                        _log_chat("WATCHDOG(raw)", b.text.strip()[:300])
            elif isinstance(msg, ResultMessage):
                cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
    print(f"[watchdog] проверка завершена, cost=${cost:.4f}", flush=True)
    return "\n".join(parts) if parts else "(без замечаний)"


async def _check_loop(model_getter):
    while True:
        await asyncio.sleep(max(60, config.WATCHDOG_INTERVAL_SEC))
        try:
            result = await _check_pass(model_getter())
            print(f"[watchdog] {result[:300]}", flush=True)
            if result.strip().upper().startswith("БРИФИНГ"):
                send(f"🔍 Наблюдатель Fetcher: {result[:1500]}")
        except Exception as e:
            print(f"[watchdog] ошибка проверки: {e}", flush=True)


async def main():
    if not config.VK_TOKEN3:
        raise SystemExit("Нет VK_TOKEN3 в automation/fetcher/.env")
    print("[watchdog] запуск...", flush=True)
    lp = LP()
    model = config.WATCHDOG_MODEL
    send("🤖 Наблюдатель Fetcher на связи. /help — команды. Проверяю каждый час.")
    print("[watchdog] готов, слушаю VK", flush=True)
    check_task = asyncio.create_task(_check_loop(lambda: model))
    try:
        while True:
            msgs = await asyncio.to_thread(lp.poll, 25)
            for t in msgs:
                _log_chat("OWNER->", t)
                low = t.lower()
                if low == "/help":
                    send(HELP)
                elif low == "/status":
                    send("⏳ Проверяю сейчас...")
                    try:
                        result = await _check_pass(model)
                        send(f"📋 {result[:1500]}")
                    except Exception as e:
                        send(f"❗ Ошибка проверки: {e}")
                else:
                    send(HELP)
    finally:
        check_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[watchdog] стоп")
