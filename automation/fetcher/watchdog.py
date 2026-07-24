#!/usr/bin/env python3
"""Агент-наблюдатель Fetcher (Третий бот, VK_TOKEN3, модель Sonnet 4.6 — см. config.WATCHDOG_MODEL;
сменено с Haiku 2026-07-17, т.к. диагностика реальных проблем на подах требует более сильной модели).

Работает НА ТВОЁМ ПК (не на поде) — использует ту же Claude-подписку (OAuth-сессия `claude login`),
что и agent_runner.py/assistant.py в automation/runner/. Постоянно поднят через systemd
(fetcher-watchdog.service, см. README.md) — переживает перезагрузку ПК: systemd сам стартует сервис
при загрузке (WantedBy=multi-user.target) и перезапускает при падении (Restart=on-failure). Внутреннего
состояния между перезапусками почти нет (проверка раз в час читает живое состояние подов по SSH), так
что просто продолжает работать как ни в чём не бывало.

Раз в час (WATCHDOG_INTERVAL_SEC): SSH на все 3 пода, смотрит логи (discover.log, workers_*.log)
и state/inventory/summary.json (см. deploy.py), решает — всё штатно (молчит) или есть проблема
(зацикливание, ошибка ключей/квоты сверх нормы, аномальный рост queue_dead_letter, процесс не
отвечает и т.п.) — тогда пишет брифинг с диагнозом и рекомендацией, а если может — сама чинит код
локально (в этом репозитории, источник истины) и перезапускает процесс на поде (git push -> git pull
на поде -> kill+restart через deploy.py).

Также ведёт полноценный свободный диалог в VK — любое сообщение (кроме /status, /help) обрабатывается
как реальная задача (см. _handle_owner_message), с короткими промежуточными сообщениями по ходу работы.

ПОЛНОСТЬЮ АВТОНОМЕН (с 2026-07-17): `git config --global credential.helper store` настроен для
пользователя ОС (см. README.md, разово) — `git push` работает без ручного вмешательства владельца.
`/etc/sudoers.d/fetcher-watchdog` даёт passwordless sudo ТОЛЬКО на restart/status этого сервиса и
fetcher-report — агент может сам применить и перезапустить собственный фикс, не дожидаясь владельца.

НИКОГДА не трогает поды ML-раннера (automation/runner/) — работает только со своими тремя.
"""
from __future__ import annotations
import asyncio
import datetime as dt
import random

import requests
from claude_agent_sdk import (
    query, ClaudeSDKClient, ClaudeAgentOptions, HookMatcher,
    AssistantMessage, TextBlock, ResultMessage,
)

import config
import hourly_report

VK = "https://api.vk.com/method"

SYSTEM = (
    "Ты — Третий агент проекта TrendFlow: наблюдатель ЗА FETCHER DATASET COLLECTOR (сбор датасета "
    "YouTube). Активный под на RunPod — ТОЛЬКО fetcher-main (discover + workers шард 0/5). "
    "fetcher-worker-b и fetcher-worker-c намеренно удалены с RunPod (2026-07) — их download-workers "
    "переехали на 4 Colab-аккаунта (шарды 1..4). НЕ пытайся подключаться к b/c по SSH — они вернут "
    "404. Реестр активных подов: config.PODS (итерируй его, не provision_result.json). "
    "Это ОТДЕЛЬНАЯ система от ML-валидации компонентов "
    "(automation/runner/) — НИКОГДА не трогай ML-поды/файлы automation/runner/, это не твоя зона. "
    "У тебя есть: Bash/Read/Write/Edit в этом репозитории (automation/fetcher/ — твой код: "
    "deploy.py содержит готовые функции ssh_run/tail_logs/read_inventory_summary/kill_processes/"
    "pull_latest_code/launch — используй их через `python3 -c \"import deploy; ...\"` из директории "
    "automation/fetcher/, не изобретай SSH-команды заново). ВАЖНО: kill_processes() использует "
    "bracket-trick в pkill-паттернах ('[d]ataset_collector') — НЕ убирай скобки, без них pkill "
    "матчит свою же командную строку и рвёт SSH-сессию сигналом (баг был найден и исправлен "
    "2026-07-16, не повторяй его в собственных SSH-командах: если сам пишешь pkill/grep по SSH, "
    "тоже используй bracket-trick на искомом слове). Секреты — automation/fetcher/.env. "
    "SSH-ключ — automation/fetcher/ssh/id_ed25519 (тот же паблик-ключ уже прописан RunPod автоматически "
    "во все поды аккаунта). ID пода fetcher-main — automation/fetcher/state/provision_result.json. "
    "ПРОТОКОЛ ПРОВЕРКИ (раз в час): только fetcher-main — deploy.tail_logs('fetcher-main') на ошибки "
    "(QuotaExceededError — это НОРМА, код сам ждёт сброса, не считай багом; HF 429 Too Many Requests — "
    "тоже норма, ретраится сам; bot_detection — штатная пауза; реальные проблемы — traceback без "
    "восстановления, растущий queue_dead_letter, процесс не пишет в лог часами, ModuleNotFoundError/"
    "FileNotFoundError, резкий скачок 'yt-dlp enrich failed') и deploy.read_inventory_summary('fetcher-main') на "
    "lag_* метрики (растущий lag без движения — подозрительно). Всё штатно — НИЧЕГО не делай, "
    "заверши коротким итогом (1 строка). Есть проблема: (1) разберись в причине по логам/коду; "
    "(2) если можешь починить — правь код В ЭТОМ РЕПО (Fetcher/fetcher/dataset_collector/...), "
    "закоммить и запушь (обычный `git commit`+`git push` — credential.helper store настроен "
    "2026-07-17, авторизация уже работает, НЕ нужно городить URL с токеном/просить владельца), "
    "затем deploy.pull_latest_code(pod) + deploy.kill_processes(pod) + deploy.launch(pod, hf_token) "
    "для применения фикса на подах. Если фикс касается ТВОЕГО СОБСТВЕННОГО кода (watchdog.py/"
    "hourly_report.py/deploy.py — то, что исполняется здесь, на этом ПК, а не на подах) — "
    "после коммита+пуша сам перезапусти сервис: `sudo systemctl restart fetcher-watchdog` (или "
    "fetcher-report) — passwordless sudo настроен ТОЛЬКО на эти 2 команды restart и 2 status "
    "(см. automation/fetcher/fetcher-watchdog.sudoers), ничего больше через sudo не делай и не "
    "пытайся — команды за пределами этого списка запросят пароль и зависнут; "
    "(3) сообщить владельцу можно ТОЛЬКО через маркер — НАЧНИ последний ответ строкой "
    "'БРИФИНГ: <диагноз в 2-4 предложения + рекомендация>' если ситуация требует внимания владельца "
    "(нужен новый набор YouTube-ключей, кампания зависла необъяснимо, RunPod-инфра сломана) — рутинный "
    "автономный фикс НЕ требует брифинга, просто резюме без маркера. "
    "Пиши по-русски, кратко."
)

HELP = ("Я наблюдатель Fetcher. /status — проверить сейчас, /help — справка. Любой другой текст — "
       "реальная задача мне (спроси про логи, попроси починить, разобраться в проблеме и т.д.).")


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
    "активный под fetcher-main (fetcher-worker-b/c намеренно удалены, работают на Colab). Начни с "
    "`cd automation/fetcher && python3 -c \"import deploy; print(deploy.tail_logs('fetcher-main'))\"` "
    "и затем deploy.read_inventory_summary('fetcher-main')."
)


async def _run_llm(
    prompt: str, model: str, *, max_turns: int = 40, progress: bool = False,
    client_holder: dict | None = None,
) -> str:
    """Общий раннер LLM-сессии для часовой проверки И для ответа на произвольное сообщение
    владельца — один и тот же агент с одним и тем же системным промптом/доступом к deploy.py.

    progress=True — слать каждый промежуточный шаг агента короткой строкой в VK СРАЗУ, а не только
    финальный ответ в конце. Баг найден 2026-07-17: обычный разбор занимает 3-5 минут (SSH на 3
    пода + анализ), всё это время чат молчал — владелец решил, что бот завис, и слал сообщение
    повторно. Для часовой автопроверки (_check_pass) progress НЕ включаем — она и так молчит,
    когда всё штатно, это осознанное поведение, не нужно.

    client_holder — опциональный dict {"client": None}, куда кладётся текущий активный
    ClaudeSDKClient, пока сессия идёт (см. main()/_producer: 2026-07-19, тот же баг, что чинили
    в deepdive_agent.py/models_agent.py — если во время долгого разбора приходит НОВОЕ сообщение
    владельца, нужен способ прервать текущий client.receive_response() через interrupt())."""
    parts = []
    cost = 0.0
    async with ClaudeSDKClient(options=ClaudeAgentOptions(
        model=model, system_prompt=SYSTEM, cwd=str(config.REPO_DIR),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions", max_turns=max_turns,
    )) as client:
        if client_holder is not None:
            client_holder["client"] = client
        await client.query(prompt)
        try:
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock) and b.text.strip():
                            text = b.text.strip()
                            parts.append(text)
                            _log_chat("WATCHDOG(raw)", text[:300])
                            if progress:
                                # Баг найден 2026-07-20 (владелец): раньше блок уходил урезанным
                                # до 200 симв., а вызывающий код (_process_owner_text) ЕЩЁ РАЗ
                                # слал этот же текст целиком через result — двойная отправка
                                # каждой мысли. Теперь шлём сразу полностью (send() сам режет
                                # длинные сообщения по 4000 символов), а вызывающий код result
                                # больше не пересылает (см. _process_owner_text).
                                send(f"⏳ {text}")
                elif isinstance(msg, ResultMessage):
                    cost = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)
        except Exception as e:
            print(f"[watchdog] сессия прервана: {e}", flush=True)
        finally:
            if client_holder is not None:
                client_holder["client"] = None
    print(f"[watchdog] сессия завершена, cost=${cost:.4f}", flush=True)
    return "\n".join(parts) if parts else "(без замечаний)"


async def _check_pass(model: str) -> str:
    return await _run_llm(CHECK_PROMPT, model)


async def _handle_owner_message(text: str, model: str, client_holder: dict | None = None) -> str:
    """Реальный диалог с владельцем в том же VK-чате, где идут часовые отчёты — не просто /status
    и /help, а любой текст трактуется как задача агенту (аналог того, как assistant.py общается с
    владельцем по ML-раннеру). Баг найден 2026-07-17: раньше любое сообщение, кроме /status и
    /help, получало шаблонный HELP — владелец физически не мог попросить агента разобраться
    в чём-то или починить, приходилось разбирать логи вручную вместе со мной в Cowork."""
    prompt = (
        f"Владелец написал в VK: «{text}»\n\n"
        "Это НЕ команда из фиксированного списка — обработай как реальную задачу/вопрос по "
        "Fetcher dataset collector (3 пода: fetcher-main, fetcher-worker-b, fetcher-worker-c). "
        "Если нужно — посмотри логи/summary.json по протоколу из системного промпта, разберись "
        "в причине, при возможности почини код и перезапусти процесс на поде. "
        "Активный RunPod-под — только fetcher-main; b/c переехали на Colab (не пытайся к ним подключаться). "
        "Ответь по существу и кратко — это уйдёт напрямую владельцу в VK."
    )
    return await _run_llm(prompt, model, max_turns=60, progress=True, client_holder=client_holder)


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


async def _process_owner_text(t: str, model: str, client_holder: dict) -> None:
    _log_chat("OWNER->", t)
    low = t.lower()
    if low == "/help":
        send(HELP)
    elif low == "/status":
        send("⏳ Проверяю сейчас...")
        # Шаг 1 — детерминированный отчёт с цифрами (быстро, без LLM).
        try:
            nums = await asyncio.to_thread(hourly_report.build_report)
            send(nums[:4000])
        except Exception as e:
            send(f"❗ Ошибка чтения метрик: {e}")
        # Шаг 2 — LLM-диагностика (ищет аномалии в логах; только если есть проблемы).
        try:
            diag = await _run_llm(CHECK_PROMPT, model, client_holder=client_holder)
            low_diag = diag.lower()
            if any(w in low_diag for w in ("брифинг", "ошибк", "проблем", "зависл", "traceback", "dead_letter")):
                send(f"🔍 Диагностика: {diag[:1500]}")
        except Exception as e:
            send(f"❗ Ошибка диагностики: {e}")
    else:
        # Любой другой текст — реальная задача агенту, не заглушка HELP (см.
        # _handle_owner_message: баг найден 2026-07-17, владелец не мог обратиться к
        # агенту напрямую в VK).
        send("⏳ Разбираюсь...")
        try:
            result = await _handle_owner_message(t, model, client_holder=client_holder)
            if not result:
                send("(агент не дал ответа)")
            # иначе result уже полностью ушёл в VK живьём по ходу (progress=True) — повторно не
            # шлём (баг найден 2026-07-20: раньше дублировали каждую мысль).
        except Exception as e:
            send(f"❗ Ошибка: {e}")


async def main():
    if not config.VK_TOKEN3:
        raise SystemExit("Нет VK_TOKEN3 в automation/fetcher/.env")
    print("[watchdog] запуск...", flush=True)
    lp = LP()
    model = config.WATCHDOG_MODEL
    send("🤖 Наблюдатель Fetcher на связи (модель " + model + "). /help — команды. Проверяю каждый час, "
        "но можешь и просто написать мне — разберусь.")
    print("[watchdog] готов, слушаю VK", flush=True)
    check_task = asyncio.create_task(_check_loop(lambda: model))

    # Баг найден 2026-07-19 (тот же класс, что чинили в deepdive_agent.py/models_agent.py):
    # 1) lp.poll() был ничем не защищён от исключений мимо requests.RequestException (например
    #    json.JSONDecodeError на кривом ответе VK) — такое падение рвало ВЕСЬ while-цикл, и
    #    поскольку вокруг него не было try/except (только внешний try/finally на cancel
    #    check_task), процесс просто завершался; ждать, что systemd успеет перезапустить и
    #    владелец это заметит, не самая быстрая диагностика.
    # 2) `/status` и произвольные сообщения разбирались ПОСЛЕДОВАТЕЛЬНО в одном цикле —
    #    poll() не вызывался, пока текущее сообщение не обработано целиком (3-5+ минут на
    #    SSH-разбор). Новое сообщение владельца в это время просто не читалось.
    # Теперь: poll — в защищённой фоновой задаче, которая никогда не падает; при новом
    # сообщении, если бот занят — прерываем текущую LLM-сессию через client.interrupt().
    queue: asyncio.Queue[str] = asyncio.Queue()
    busy = asyncio.Event()
    client_holder: dict = {"client": None}

    async def _producer() -> None:
        while True:
            try:
                msgs = await asyncio.to_thread(lp.poll, 25)
            except Exception as e:
                print(f"[watchdog] poll упал, жду и пробую снова: {e}", flush=True)
                await asyncio.sleep(3)
                continue
            for t in msgs:
                if busy.is_set() and client_holder["client"] is not None:
                    send("⏸ Прерываю текущую работу — читаю новое сообщение.")
                    try:
                        await client_holder["client"].interrupt()
                    except Exception as e:
                        print(f"[watchdog] interrupt не сработал: {e}", flush=True)
                await queue.put(t)

    producer_task = asyncio.create_task(_producer())
    try:
        while True:
            t = await queue.get()
            busy.set()
            try:
                await _process_owner_text(t, model, client_holder)
            finally:
                busy.clear()
    finally:
        check_task.cancel()
        producer_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[watchdog] стоп")
