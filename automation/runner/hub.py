"""Центральный VK-хаб: единственный потребитель group long poll.

Разводит входящие сообщения:
  - начинающиеся с '/'  -> команды управления (logs/stop-logs/stop-session/start-session/status)
  - остальной текст     -> ответы владельца (для ask_human и подтверждений pod_control)

Так решается конфликт: если бы ask_human и слушатель команд опрашивали long poll независимо,
они бы «воровали» сообщения друг у друга. Здесь приём в одном месте.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field

import messenger
import settings

# Человекочитаемые метки моделей для консоли/VK.
MODEL_LABELS = {
    "claude-opus-4-8": "Opus 4.8",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "fable": "Fable 5",
}


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


@dataclass
class RuntimeState:
    model_label: str = "agent"
    tokens_in: int = 0
    tokens_out: int = 0
    session_start: float = 0.0
    last_activity: float = 0.0
    logs_enabled: bool = False
    pod_running: bool = False
    session_active: bool = False
    manual_pause: bool = False  # True после стоп-команды/лимита: ждём старт
    pause_reason: str = ""      # "user" | "limit" | "error" | "assistant" — почему на паузе
    auto_left: int = 0          # ночной режим: сколько ещё авто-сессий пройти после сброса лимита
    hit_limit: bool = False     # сессия завершилась из-за ЛИМИТА Claude (по тексту)
    session_failed: bool = False  # сессия не сделала работы по ДРУГОЙ причине (ошибка/неверная модель)
    reset_at: str = ""          # время сброса лимита из сообщения Claude (напр. "12:20am")
    dry_pod: bool = False       # тест без реального управления подами
    stop_kind: str = "save"     # как останавливать текущую сессию: save (стоп пода) | plain (без стопа)
    # Событие остановки текущей сессии (создаётся заново на каждую сессию).
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Очереди для межзадачного обмена.
    reply_queue: "asyncio.Queue[str]" = field(default_factory=asyncio.Queue)
    start_queue: "asyncio.Queue[str]" = field(default_factory=asyncio.Queue)

    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    def elapsed(self) -> int:
        return int(time.time() - self.session_start) if self.session_start else 0

    def tag(self) -> str:
        return f"[{self.model_label}][~{_fmt_tok(self.tokens_total())}][{self.elapsed()}s]"


def _fmt_tok(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


# Единый глобальный state.
STATE = RuntimeState()

HELP = (
    "Команды VK:\n"
    "/logs, /stop-logs — логи модели в чат вкл/выкл\n"
    "/sss — стоп сессии + сохранение + стоп пода (= /stop-session)\n"
    "/sps — стоп сессии без сохранения и БЕЗ стопа пода\n"
    "/start-session [текст] — старт (пусто = следующий компонент из очереди)\n"
    "/ssc [текст] — старт с чтением контекста прошлой сессии\n"
    "/sas <N> [%] [-m 'сообщение'] — ночной режим: N авто-сессий, стоп на % лимита\n"
    "/sas-stop — выключить ночной режим\n"
    "/verify <компонент> — независимая верификация\n"
    "/model [opus|sonnet|haiku|fable|...] — сменить модель (со следующей сессии)\n"
    "/settings [ключ значение] — показать/изменить настройки (лимиты и т.д.)\n"
    "/limits — лимиты Claude (5ч/неделя, все агенты)\n"
    "/scanlimits — обновить лимиты с claude.ai (браузер-парсер)\n"
    "/status — статус (сессия, токены, бюджет, поды)\n"
    "/help — справка"
)


async def _handle_command(text: str):
    low = text.strip().lower()
    if low == "/logs":
        STATE.logs_enabled = True
        messenger.send("📡 Логи включены — буду дублировать сообщения модели сюда. /stop-logs чтобы выключить.")
    elif low == "/stop-logs":
        STATE.logs_enabled = False
        messenger.send("🔕 Логи выключены.")
    elif low in ("/stop-session", "/sss"):
        if STATE.session_active:
            STATE.stop_kind = "save"
            STATE.stop_event.set()
            messenger.send("🛑 Стоп: сохраняю состояние и гашу под.")
        else:
            messenger.send("Активной сессии нет. /start-session чтобы запустить.")
    elif low == "/sps":
        if STATE.session_active:
            STATE.stop_kind = "plain"
            STATE.stop_event.set()
            messenger.send("🛑 Стоп без сохранения. Под НЕ гашу.")
        else:
            messenger.send("Активной сессии нет.")
    elif low.startswith("/start-session"):
        prompt = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        await STATE.start_queue.put(prompt)
        messenger.send(f"▶️ Старт: {prompt or '(следующий компонент из очереди)'}")
    elif low.startswith("/ssc"):
        prompt = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        await STATE.start_queue.put("ctx:" + prompt)
        messenger.send(f"▶️ Старт с контекстом прошлой сессии: {prompt or '(следующий компонент)'}")
    elif low.startswith("/model"):
        arg = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        if not arg:
            messenger.send(f"Текущая модель: {settings.model()}. Сменить: /model opus|sonnet|haiku|fable")
        else:
            mid = settings.resolve_model(arg)
            settings.set_("model", mid)
            messenger.send(f"✅ Модель со следующей сессии: {model_label(mid)} ({mid}).")
    elif low.startswith("/settings"):
        parts = text.split()
        if len(parts) == 1:
            messenger.send(settings.as_text())
        elif len(parts) >= 3:
            key, val = parts[1], " ".join(parts[2:])
            try:
                val = float(val) if key != "model" else val
            except ValueError:
                pass
            settings.set_(key, val)
            messenger.send(f"✅ {key} = {val}")
        else:
            messenger.send("Формат: /settings <ключ> <значение>. Показать всё: /settings")
    elif low.startswith("/sas"):
        parts = text.split()
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2
        pct_stop = settings.get('limit_pct_stop')
        msg = ""

        # Парсим аргументы: /sas <N> [%] [-m "сообщение"]
        idx = 2
        if idx < len(parts) and parts[idx].isdigit():
            pct_stop = int(parts[idx])
            settings.set_("limit_pct_stop", pct_stop)
            idx += 1

        # Проверяем флаг -m
        if idx < len(parts) and parts[idx] == "-m":
            if idx + 1 < len(parts):
                msg = " ".join(parts[idx + 1:])

        STATE.auto_left = n
        if msg:
            # "auto:" — заметка к компоненту, а НЕ разовая ручная задача (иначе сессия встанет
            # на паузу после одного компонента и ночной режим не будет работать непрерывно).
            await STATE.start_queue.put("auto:" + msg)
        else:
            await STATE.start_queue.put("")  # запустить первую сессию
        messenger.send(f"🌙 Ночной режим: {n} авто-сессий, стоп на {pct_stop}% лимита. "
                       f"Работаю до лимита → жду сброса 5ч-окна → продолжаю.\n"
                       f"{('📝 Сообщение: ' + msg) if msg else ''}\n/sas-stop для отмены.")
    elif low == "/sas-stop":
        STATE.auto_left = 0
        messenger.send("🌙 Ночной режим выключен (текущая сессия доработает).")
    elif low.startswith("/verify"):
        comp = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        if not comp:
            messenger.send("Укажи компонент: /verify <component>")
        else:
            await STATE.start_queue.put(f"verify:{comp}")
            messenger.send(f"🔎 Принял запрос на верификацию: {comp}")
    elif low == "/limits":
        import limits
        messenger.send(limits.status_text())
    elif low == "/scanlimits":
        import limits, asyncio as _a, subprocess, sys as _s, config
        messenger.send("🌐 Снимаю лимиты с claude.ai (headless)…")

        def _run():
            cmd = [_s.executable, str(config.RUNNER_DIR / "claude_limits_scraper.py")]
            cmd += (["--cdp", config.CLAUDE_CDP_PORT] if config.CLAUDE_CDP_PORT else ["--headless"])
            return subprocess.run(cmd, capture_output=True, text=True, timeout=150)
        try:
            await _a.to_thread(_run)
        except Exception as e:
            messenger.send(f"Не смог запустить парсер: {e}")
        messenger.send(limits.status_text())
    elif low == "/status":
        messenger.send(_status_text())
    elif low == "/help":
        messenger.send(HELP)
    else:
        messenger.send(f"Неизвестная команда. {HELP}")


def _status_text() -> str:
    import runpod_api
    import budget
    pod = "активна" if STATE.session_active else "нет"
    try:
        pods = runpod_api.status_summary()
    except Exception as e:
        pods = f"(не смог получить: {e})"
    return (f"Сессия: {pod} {STATE.tag() if STATE.session_active else ''}\n"
            f"Логи в чат: {'вкл' if STATE.logs_enabled else 'выкл'}\n"
            f"{budget.status_text()}\n"
            f"Поды:\n{pods}")


async def listener():
    """Фоновая задача: постоянно читает long poll и разводит сообщения. Запускать один раз."""
    lp = await asyncio.to_thread(messenger.LongPoll)
    messenger.send("🤖 Хаб на связи. /help — список команд.")
    while True:
        try:
            texts = await asyncio.to_thread(lp.poll_once, 25)
        except Exception:
            await asyncio.sleep(3)
            continue
        for t in texts:
            messenger.log_chat("OWNER->", t)
            if t.startswith("/"):
                await _handle_command(t)
            else:
                await STATE.reply_queue.put(t)


async def get_reply(prompt: str, timeout_sec: int = 6 * 3600,
                    reping_every: int = 1800, stop_pods_after: int = 1800) -> str:
    """Отправить владельцу вопрос/запрос подтверждения и дождаться НЕ-командного ответа.

    Если ответа нет дольше stop_pods_after (по умолчанию 30 мин) и есть поднятые поды —
    гасим их (деньги!), а к пришедшему ответу добавляем пометку поднять под заново.
    """
    import runpod_api
    # Сбрасываем «старые» сообщения, чтобы не считать их ответом на этот вопрос.
    while not STATE.reply_queue.empty():
        try:
            STATE.reply_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    messenger.send(prompt)
    waited = 0
    tick = 30
    last_reping = 0
    pods_stopped = False
    while waited < timeout_sec:
        try:
            reply = await asyncio.wait_for(STATE.reply_queue.get(), timeout=tick)
            if pods_stopped:
                reply += ("\n[Система: поды были остановлены из-за паузы >30 мин. Если для продолжения "
                          "нужен GPU — подними под заново через pod_control start.]")
            return reply
        except asyncio.TimeoutError:
            waited += tick
            if (not pods_stopped) and stop_pods_after and waited >= stop_pods_after \
                    and STATE.pod_running and not STATE.dry_pod:
                try:
                    stopped = runpod_api.stop_all_and_bill()
                    STATE.pod_running = False
                    pods_stopped = True
                    messenger.send(f"⏸️ >30 мин без ответа — погасил поды {stopped or ''} до твоего ответа, "
                                   f"чтобы не тратить деньги. Отвечай — подниму заново и продолжу.")
                except Exception as e:
                    messenger.send(f"⚠️ Хотел погасить поды из-за долгой паузы, но не смог: {e}")
                    pods_stopped = True  # не долбить каждые 30с
            if waited - last_reping >= reping_every and waited < timeout_sec:
                last_reping = waited
                messenger.send("⏳ Жду твой ответ на вопрос выше.")
    return "__TIMEOUT__"


async def wait_owner(prompt: str, timeout_sec: int = 180) -> str | None:
    """Отправить сообщение и подождать НЕ-командный ответ владельца короткое время. None если молчит."""
    while not STATE.reply_queue.empty():
        try:
            STATE.reply_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    messenger.send(prompt)
    try:
        return await asyncio.wait_for(STATE.reply_queue.get(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        return None


def emit_log(tag: str, text: str):
    """Печать в консоль, дублирование в chat log (для агента 2) и (если включено) в VK."""
    print(f"{tag} {text}")
    messenger.log_chat("AGENT1(raw)", f"{tag} {text[:300]}")
    if STATE.logs_enabled:
        messenger.send(f"{tag}\n{text[:1500]}")
