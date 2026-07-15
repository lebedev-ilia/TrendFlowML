#!/usr/bin/env python3
"""Главный сервис автономного раннера TrendFlow (один агент, Claude Agent SDK).

Управляется из VK (см. hub.py): /logs /stop-logs /stop-session /start-session /status.
В консоли каждая строка модели помечается тегом вида [Opus 4.8][~3.2k][345s]
(модель · приблизит. токены · секунды сессии). При /logs те же строки летят в VK.

Запуск:
  python agent_runner.py --once            # один компонент и стоп
  python agent_runner.py --once --dry-pod  # без реального управления подом (тест VK/логики)
  python agent_runner.py -m "промпт"       # сессия с начальным промптом (verify:/ctx:/ручное задание)
  python agent_runner.py                    # непрерывный сервис, управляемый из VK
"""
from __future__ import annotations
import argparse
import asyncio
import datetime as dt
import json
import re
import subprocess
import sys
import time
import traceback

from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions, HookMatcher,
    AssistantMessage, TextBlock, ResultMessage,
)

import config
import budget
import settings
import limits
import agents
import component_queue as q
import messenger
import tools
import hooks
import hub
import runpod_api

STATE = hub.STATE


def _extract_section(text: str, name: str) -> str:
    """Вырезать секцию, помеченную <!-- SECTION:name:start/end --> в AGENT_CONTEXT.md."""
    start_tag, end_tag = f"<!-- SECTION:{name}:start -->", f"<!-- SECTION:{name}:end -->"
    i, j = text.find(start_tag), text.find(end_tag)
    if i == -1 or j == -1:
        raise ValueError(f"Секция {name!r} не найдена в AGENT_CONTEXT.md (проверь HTML-якоря)")
    return text[i + len(start_tag):j].strip()


def _system_prompt(component: str, mode: str) -> str:
    """Системный промпт собирается из automation/runner/AGENT_CONTEXT.md: секция 'common'
    (язык/краткость/лимиты/документы) + секция 'worker' или 'verify' в зависимости от режима."""
    full = (config.RUNNER_DIR / "AGENT_CONTEXT.md").read_text(encoding="utf-8")
    common = _extract_section(full, "common")
    role = _extract_section(full, "verify" if mode == "verify" else "worker")
    tmpl = f"{common}\n\n---\n\n{role}"
    return tmpl.replace("{COMPONENT}", component or "(задача задана владельцем вручную через /start-session)")


def _options(component: str, mode: str) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=settings.model(),
        system_prompt=_system_prompt(component, mode),
        cwd=str(config.REPO_DIR),
        mcp_servers={"trendflow": tools.build_server()},
        allowed_tools=[
            "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch",
            "mcp__trendflow__ask_human", "mcp__trendflow__pod_control", "mcp__trendflow__manager",
            "mcp__trendflow__budget_status", "mcp__trendflow__limits_status",
        ],
        permission_mode="bypassPermissions",
        max_turns=config.MAX_TURNS_PER_COMPONENT,
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[hooks.guard_bash_worker])]},
    )


def _accum_usage(msg, final: bool):
    u = getattr(msg, "usage", None)
    if not isinstance(u, dict):
        return
    itok = int(u.get("input_tokens", 0) or 0)
    otok = int(u.get("output_tokens", 0) or 0)
    if final:  # ResultMessage.usage — авторитетное кумулятивное
        STATE.tokens_in = itok or STATE.tokens_in
        STATE.tokens_out = otok or STATE.tokens_out
    else:
        STATE.tokens_in = max(STATE.tokens_in, itok)
        STATE.tokens_out += otok


async def _consume(client, holder):
    async for msg in client.receive_response():
        STATE.last_activity = time.time()
        if isinstance(msg, AssistantMessage):
            _accum_usage(msg, final=False)
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    t = block.text.strip()
                    if re.search(r"session limit|hit your (usage|session)|usage limit|rate limit", t, re.I):
                        holder["limited"] = True
                        mr = re.search(r"resets?\s+(\d{1,2}:\d{2}\s*[ap]m)", t, re.I)
                        if mr:
                            holder["reset_at"] = mr.group(1).replace(" ", "").lower()
                    hub.emit_log(STATE.tag(), t[:800])
        elif isinstance(msg, ResultMessage):
            _accum_usage(msg, final=True)
            if getattr(msg, "is_error", False):
                holder["error"] = True
            holder["cost"] = float(getattr(msg, "total_cost_usd", 0.0) or 0.0)


async def run_session(prompt: str, component: str | None, mode: str = "component") -> tuple[float, bool]:
    """Отработать одну сессию (компонент/ручное/verify).
    Возвращает (cost_usd, stopped_by_user)."""
    STATE.stop_event = asyncio.Event()
    STATE.stop_kind = "save"
    STATE.model_label = hub.model_label(settings.model())
    STATE.session_active = True
    STATE.session_start = time.time()
    STATE.last_activity = time.time()
    STATE.tokens_in = STATE.tokens_out = 0
    STATE.hit_limit = False
    STATE.session_failed = False
    gpu_before = budget.gpu_spent_today()
    if component and mode == "component":
        q.set_claim(component)
    holder = {"cost": 0.0}
    stopped = False
    try:
        async with ClaudeSDKClient(options=_options(component or "", mode)) as client:
            await client.query(prompt)
            consume = asyncio.create_task(_consume(client, holder))
            stopper = asyncio.create_task(STATE.stop_event.wait())
            done, _ = await asyncio.wait({consume, stopper}, return_when=asyncio.FIRST_COMPLETED)
            if stopper in done and not consume.done():
                stopped = True
                consume.cancel()
                try:
                    await consume
                except asyncio.CancelledError:
                    pass
            else:
                stopper.cancel()
    finally:
        STATE.session_active = False
    # Лимит — ТОЛЬКО по тексту («session limit» и т.п.). Пусто/ошибка — это СБОЙ, не лимит.
    limited = bool(holder.get("limited"))
    empty = not stopped and holder["cost"] < 0.001 and STATE.tokens_out < 20
    STATE.hit_limit = limited and not stopped
    STATE.reset_at = holder.get("reset_at", "") if limited else ""
    STATE.session_failed = (empty or bool(holder.get("error"))) and not limited and not stopped
    if component and mode == "component" and not stopped and not STATE.hit_limit and not STATE.session_failed:
        q.clear_claim()
    label = component or mode
    budget.record(label, holder["cost"], note="stopped" if stopped else "done")
    gpu_usd = max(0.0, budget.gpu_spent_today() - gpu_before)
    budget.record_session(label, mode, holder["cost"], gpu_usd, STATE.tokens_total(),
                          STATE.session_start, note="stopped" if stopped else "done")
    limits.record(f"runner:{STATE.model_label}", STATE.tokens_in, STATE.tokens_out, holder["cost"])
    return holder["cost"], stopped


def _component_prompt(comp: str, with_context: bool = False, extra_note: str = "") -> str:
    prog = config.PROGRESS_DIR / f"{comp.replace('/', '__')}.md"
    ctx = (f"Кратко прочитай {config.LAST_SESSION} (контекст прошлой сессии), затем " if with_context else "")
    note = (f"Указание владельца (учти, но это НЕ повод останавливаться после этого компонента): "
            f"{extra_note}\n" if extra_note else "")
    return (
        f"{note}{ctx}работай над компонентом «{comp}».\n"
        f"ФАЙЛ ПРОГРЕССА: {prog} — СНАЧАЛА прочитай (если есть) и продолжи с места остановки, не с нуля; "
        f"веди его по ходу.\n"
        f"Процесс: логика → оптимизация → инфра → прогон → REPORT → предложить вердикт. "
        f"В конце обнови {config.LAST_SESSION} (2–4 строки: что сделал, где остановился, что дальше), "
        f"допиши урок в AGENT_CONTEXT.md (раздел 7), погаси под.\n"
        f"КРАТКОСТЬ: пиши сжато, минимум сообщений владельцу, по делу. Только на русском."
    )


def _parse_verify(text: str) -> str | None:
    t = (text or "").strip()
    for pref in ("verify:", "verify ", "проверь ", "verify_"):
        if t.lower().startswith(pref):
            return t[len(pref):].strip() or None
    return None


def _verify_run_prompt(comp: str) -> str:
    return (f"Проведи НЕЗАВИСИМУЮ верификацию компонента «{comp}» по AGENT_CONTEXT.md (раздел 4): "
            f"универсальные хард-гейты A1–A6 + критерии CRITERIA.md, с сырыми числами. Запиши "
            f"VERIFICATION_YYYY-MM-DD.md, пришли владельцу вердикт, погаси под. Логику компонента не меняй.\n"
            f"Пиши ВСЁ только на русском — включая рассуждения и пояснения к действиям.")


def _from_manual(p: str):
    """Разобрать ручной ввод: ctx: -> с контекстом; auto: -> непрерывное продолжение очереди
    компонентов с доп. заметкой владельца (см. ниже почему это отдельный префикс); verify ->
    верификация; иначе разовое ручное задание."""
    # /ssc -> префикс "ctx:"
    if p.startswith("ctx:"):
        rest = p[len("ctx:"):].strip()
        if not rest:
            comp = q.next_component()
            if comp:
                return (_component_prompt(comp, with_context=True), comp, "component")
            return (f"Кратко прочитай {config.LAST_SESSION} и предложи следующий шаг.", None, "manual")
        return (f"Кратко прочитай {config.LAST_SESSION} для контекста, затем: {rest}\nТолько на русском.", None, "manual")
    # /sas <N> -m "текст" -> префикс "auto:" (см. hub.py). ВАЖНО: это НЕ разовая ручная задача —
    # /sas запускает непрерывный ночной режим, поэтому сообщение владельца должно идти как заметка
    # к текущему компоненту, а сессия должна остаться в mode="component" (иначе один компонент
    # выполнится и раннер встанет на паузу, ожидая владельца — баг, который был исправлен).
    if p.startswith("auto:"):
        note = p[len("auto:"):].strip()
        comp = q.next_component()
        if comp:
            return (_component_prompt(comp, extra_note=note), comp, "component")
        return (f"{note}\nОчередь компонентов пуста — предложи следующий шаг.\nТолько на русском.", None, "manual")
    v = _parse_verify(p)
    if v:
        return (_verify_run_prompt(v), v, "verify")
    return (p + "\nТолько на русском, кратко.", None, "manual")


def _scan_limits():
    cmd = [sys.executable, str(config.RUNNER_DIR / "claude_limits_scraper.py")]
    cmd += (["--cdp", config.CLAUDE_CDP_PORT] if config.CLAUDE_CDP_PORT else ["--headless"])
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=150)
    except Exception:
        pass


def _seconds_until(clock: str) -> int:
    """Секунд до ближайшего наступления времени вида '12:20am' (по локальному времени ПК)."""
    m = re.match(r"(\d{1,2}):(\d{2})(am|pm)", clock.strip().lower())
    if not m:
        return 0
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    now = dt.datetime.now()
    target = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return int((target - now).total_seconds())


async def _wait_limit_ok(pct: float) -> bool:
    """Дождаться сброса лимита. НЕ полагаемся на браузерный % (при session-лимите он врёт):
    ждём до времени сброса из сообщения Claude, иначе фиксированные 30 минут."""
    secs = _seconds_until(STATE.reset_at) if STATE.reset_at else 0
    if secs and secs < 6 * 3600:  # известно время сброса и оно правдоподобно
        mins = secs // 60
        msg = f"🌙 Лимит Claude. Жду сброса до {STATE.reset_at} (~{mins} мин)."
        messenger.send(msg); messenger.send_assistant(msg)
        await asyncio.sleep(secs + 120)  # +2 мин запас
    else:
        msg = "🌙 Лимит Claude (время сброса неизвестно). Жду 30 мин до повтора."
        messenger.send(msg); messenger.send_assistant(msg)
        await asyncio.sleep(1800)
    STATE.reset_at = ""
    return True


async def _wait_start_command(poll_sec: int = 5) -> str:
    """Ждать команду старта: из VK (STATE.start_queue) ИЛИ из файла config.ASSISTANT_START_REQUEST,
    который пишет Второй агент (assistant.py), когда сам решает перезапустить Первого после починки
    кода. Файл — межпроцессный канал (assistant.py — отдельный процесс, не может достучаться до
    asyncio.Queue этого процесса напрямую)."""
    while True:
        if not STATE.start_queue.empty():
            return await STATE.start_queue.get()
        if config.ASSISTANT_START_REQUEST.exists():
            try:
                data = json.loads(config.ASSISTANT_START_REQUEST.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            try:
                config.ASSISTANT_START_REQUEST.unlink()
            except Exception:
                pass
            text = (data.get("text") or "").strip()
            messenger.send(f"🔁 Рестарт по запросу Второго агента" + (f": {text}" if text else " (следующий компонент)."))
            return text
        try:
            return await asyncio.wait_for(STATE.start_queue.get(), timeout=poll_sec)
        except asyncio.TimeoutError:
            continue


def _assistant_stop_requested() -> str | None:
    """Проверить файл config.ASSISTANT_STOP_REQUEST (Второй агент просит аккуратно остановить
    Первого — с сохранением состояния и гашением пода). Возвращает причину или None."""
    if not config.ASSISTANT_STOP_REQUEST.exists():
        return None
    try:
        data = json.loads(config.ASSISTANT_STOP_REQUEST.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    try:
        config.ASSISTANT_STOP_REQUEST.unlink()
    except Exception:
        pass
    return (data.get("reason") or "без причины").strip()


async def _acquire_task(once: bool):
    """Определить следующую работу. Возвращает (prompt, component, mode). Может блокироваться."""
    # 1) Явный запрос из VK (в т.ч. путь возобновления после /stop-session).
    if not STATE.start_queue.empty():
        p = await STATE.start_queue.get()
        STATE.manual_pause = False
        budget.clear_pause()
        if p:
            return _from_manual(p)

    # 1b) Пауза из-за ЛИМИТА — не ждём владельца: ждём сброса 5ч-окна и продолжаем сами.
    #     Активно, если включён auto_resume ИЛИ задан ночной режим (/sas auto_left>0).
    if (not once and STATE.manual_pause and STATE.pause_reason == "limit"
            and (STATE.auto_left > 0 or int(settings.get("auto_resume") or 0))):
        pct = float(settings.get("limit_pct_stop"))
        ok = await _wait_limit_ok(pct)
        STATE.manual_pause = False
        STATE.pause_reason = ""
        budget.clear_pause()
        if STATE.auto_left > 0:
            STATE.auto_left -= 1
        messenger.send("🌙 Лимит сброшен, продолжаю работу." if ok else
                       "🌙 Не дождался сброса за 8ч — жду /start-session.")
        if not ok:
            STATE.manual_pause = True
            STATE.pause_reason = "user"

    # 2) Остановлено вручную (/stop-session) ИЛИ по аварийной паузе (крэш / запрос Второго агента):
    #    ждём /start-session ИЛИ файловый рестарт-запрос от assistant.py. Под НЕ трогаем (уже погашен,
    #    если это требовалось — см. _safe_stop_pods в местах, где выставляется пауза).
    if not once and (STATE.manual_pause or budget.is_paused()):
        if STATE.pause_reason == "assistant":
            messenger.send("⏸️ Остановлено по запросу Второго агента (см. выше причину). Он сам пришлёт "
                           "рестарт, когда разберётся; можешь и ты прислать /start-session раньше.")
        elif budget.is_paused():
            messenger.send("⏸️ Пауза (авария). Пришли /start-session чтобы продолжить.")
        else:
            messenger.send("⏸️ Остановлено по команде. Пришли /start-session (или /start-session <текст>, "
                           "или /verify <компонент>) чтобы продолжить. Под не тронут.")
        p = await _wait_start_command()
        STATE.manual_pause = False
        budget.clear_pause()
        if p:
            return _from_manual(p)

    # 3) Следующий компонент из очереди (или текущий заклейменный — продолжение).
    comp = q.next_component()
    if comp:
        return (_component_prompt(comp), comp, "component")

    # 4) Очередь пуста — ВСЕ компоненты заштампованы. Ждём ручную задачу.
    if once:
        return (None, None, None)
    messenger.send("✅ Очередь компонентов пуста — все компоненты заштампованы. "
                   "Пришли /start-session <текст> для ручной задачи.")
    p = await _wait_start_command()
    if p:
        return _from_manual(p)
    return ("Оцени состояние проекта и предложи следующий шаг.", None, "manual")


def _pods_actually_running() -> bool:
    """Есть ли реально поднятые поды по данным RunPod (ловит поды, поднятые в обход pod_control)."""
    if STATE.dry_pod:
        return False
    try:
        return len(runpod_api.running_pods()) > 0
    except Exception:
        return STATE.pod_running


async def pod_watchdog():
    """Сторож: простой (>20м пред., >60м стоп) + лимит Claude (95%/97%) + баланс RunPod
    (предупреждение, НЕ стоп) + запрос аккуратного стопа от Второго агента (assistant.py).
    Проверяет РЕАЛЬНОЕ наличие подов через API раз в ~5 мин (ловит поды в обход pod_control)."""
    warned_idle = False
    warned_limit = False
    warned_balance = False
    api_check = 0
    while True:
        await asyncio.sleep(30)
        agents.heartbeat("component-runner", STATE.model_label)

        # --- запрос Второго агента: аккуратно остановить (сохранить контекст, погасить под) ---
        if STATE.session_active:
            reason = _assistant_stop_requested()
            if reason:
                STATE.stop_kind = "save"
                STATE.stop_event.set()
                STATE.manual_pause = True
                STATE.pause_reason = "assistant"
                messenger.send(f"🧭 Второй агент попросил остановиться: {reason}. Сохраняю состояние, гашу под.")

        # --- лимит Claude (браузер): 95% (work) — сворачиваться; 97% (hard) — жёсткий стоп, 3% резерв ---
        pct = limits.max_used_pct()
        if pct is not None and STATE.session_active:
            work = float(settings.get("limit_pct_stop"))
            hard = float(settings.get("limit_pct_hard"))
            if pct >= hard:
                STATE.stop_kind = "save"
                STATE.stop_event.set()
                STATE.manual_pause = True
                STATE.pause_reason = "limit"
                _safe_stop_pods()
                messenger.send(f"🧯 Лимит Claude {pct:.0f}% ≥ {hard:.0f}% — жёсткий стоп сессии и пода "
                               f"(неприкосновенный резерв {100-hard:.0f}% выше не трогаем).")
            elif pct >= work and not warned_limit:
                warned_limit = True
                STATE.pause_reason = "limit"  # чтобы ночной режим продолжил после сброса
                messenger.send(f"⚠️ Лимит Claude {pct:.0f}% ≥ {work:.0f}% — СВОРАЧИВАЙСЯ: доведи шаг, "
                               f"зафиксируй состояние/доки, погаси под, заверши сессию.")
            elif pct < work:
                warned_limit = False

        # --- баланс RunPod: только предупреждение, раз в ~10 мин (не при каждом тике) ---
        if api_check % 20 == 0:
            if budget.runpod_balance_low():
                if not warned_balance:
                    warned_balance = True
                    bal = budget.runpod_balance()
                    bal_txt = f"${bal:.2f}" if bal is not None else "0 или недоступен"
                    messenger.send(f"💳 Баланс RunPod низкий ({bal_txt}) — пополни, иначе поды скоро "
                                   f"перестанут подниматься. Работу НЕ останавливаю.")
            else:
                warned_balance = False

        if STATE.dry_pod:
            api_check += 1
            continue
        api_check += 1
        real_running = STATE.pod_running or (api_check % 5 == 0 and _pods_actually_running())
        if not real_running:
            warned_idle = False
            continue
        STATE.pod_running = True  # синхронизируем флаг, если под подняли в обход
        # --- простой ---
        idle = time.time() - STATE.last_activity
        if idle > 3600:
            messenger.send(f"⏱️ Под простаивает >1ч — гашу.")
            _safe_stop_pods()
            warned_idle = False
        elif idle > 1200 and not warned_idle:
            messenger.send(f"⏱️ Под простаивает >20 мин. Авто-стоп на 60-й минуте.")
            warned_idle = True


async def main_service(once: bool):
    STATE.model_label = hub.model_label(settings.model())
    agents.heartbeat("component-runner", STATE.model_label)
    listener_task = asyncio.create_task(hub.listener())
    watchdog_task = asyncio.create_task(pod_watchdog())
    messenger.send(f"🤖 Раннер запущен ({STATE.model_label}). /help — команды.")
    try:
        while True:
            prompt, component, mode = await _acquire_task(once)
            if prompt is None:
                messenger.send("☑️ Работать не над чем. Останавливаюсь.")
                break
            label = component or ("верификация" if mode == "verify" else "ручное задание")
            if mode == "verify" and component:
                label = f"verify:{component}"
            messenger.send(f"▶️ Старт: {label}")
            try:
                cost, stopped = await run_session(prompt, component, mode)
            except Exception as e:
                tb = traceback.format_exc()[-1200:]
                messenger.send(f"❗ Ошибка на «{label}»: {e}\n{tb}")
                _safe_stop_pods()
                budget.set_pause(f"crash on {label}: {e}")
                if once:
                    break
                continue

            if stopped:
                # Стоп по команде/лимиту. НЕ затираем "limit" (иначе авто-возобновление не сработает).
                STATE.manual_pause = True
                if STATE.pause_reason != "limit":
                    STATE.pause_reason = "user"
                if STATE.stop_kind == "save":
                    _safe_stop_pods()
                    messenger.send(f"🛑 «{label}» остановлено (сохранено, под погашен). /start-session или /ssc для продолжения.")
                else:
                    messenger.send(f"🛑 «{label}» остановлено (без сохранения, под оставлен). /ssc для продолжения с контекстом.")
            elif STATE.hit_limit:
                # Лимит Claude исчерпан — РЕАЛЬНОЙ работы не было. НЕ помечаем компонент готовым!
                STATE.pause_reason = "limit"
                _safe_stop_pods()
                messenger.send(f"🧯 Лимит Claude исчерпан — «{label}» НЕ выполнен (0 работы), остаётся в очереди.")
                if not once and (STATE.auto_left > 0 or int(settings.get("auto_resume") or 0)):
                    ok = await _wait_limit_ok(float(settings.get("limit_pct_stop")))
                    if STATE.auto_left > 0:
                        STATE.auto_left -= 1
                    STATE.pause_reason = ""
                    if not ok:
                        STATE.manual_pause = True
                    # ok → цикл продолжится, повторно возьмёт тот же (заклейменный) компонент
                else:
                    STATE.manual_pause = True
            elif STATE.session_failed:
                # Сессия НЕ сделала работы (ошибка/неверная модель) — НЕ помечаем готовым, встаём на паузу.
                STATE.manual_pause = True
                STATE.pause_reason = "error"
                _safe_stop_pods()
                messenger.send(f"❗ Сессия «{label}» не выполнила работу (0 токенов) — вероятно ошибка или "
                               f"неверная модель (сейчас {settings.model()}). Пауза. Проверь /model, затем /start-session.")
            else:
                # Компонент завершён — помечаем закрытым и АВТОМАТИЧЕСКИ идём дальше (не ждём владельца:
                # непрерывная работа по ВСЕМ компонентам — единственные стопы: лимит Claude, запрос
                # Второго агента, ручная команда владельца).
                if mode == "component" and component:
                    q.mark_done(component)
                messenger.send(f"🏁 «{label}» готово (≈${cost:.2f}). Итого сегодня ${budget.total_spent_today():.2f}.")
                if mode != "component":
                    # verify/manual — разовая задача, ждём владельца, что дальше.
                    STATE.manual_pause = True
                    messenger.send("Готово. /start-session для следующего или /verify/ручное задание.")
                # иначе (component) -> цикл продолжится к следующему компоненту сам, непрерывно
            if once:
                if not stopped and STATE.stop_kind == "save":
                    _safe_stop_pods()
                messenger.send("☑️ Режим --once: одна сессия готова, останавливаюсь.")
                break
    finally:
        listener_task.cancel()
        watchdog_task.cancel()
        agents.unregister()


def _safe_stop_pods():
    STATE.pod_running = False
    if STATE.dry_pod:
        return
    try:
        killed = runpod_api.terminate_all_and_bill()
        messenger.send(f"🗑️ Поды УДАЛЕНЫ {killed or 'нет'}. GPU сегодня ${budget.gpu_spent_today():.2f}.")
    except Exception as e:
        messenger.send(f"⚠️ Не смог удалить поды: {e}. Проверь runpod.io вручную!")


def _try_pod_recovery():
    """Попытка самовосстановления при 2+ EXITED подах (ночной режим). Тихий вызов."""
    if STATE.dry_pod or STATE.auto_left <= 0:  # только в ночном режиме
        return
    try:
        import podmanager
        msg = podmanager.self_recovery_check()
        if "✓" in msg:
            messenger.send(f"🔄 {msg}")
    except Exception as e:
        pass  # молча игнорируем — это не критично


def _preflight():
    config.require("VK_TOKEN", config.VK_TOKEN)
    config.require("VK_OWNER_ID", config.VK_OWNER_ID)
    config.require("RUNPOD_API_KEY", config.RUNPOD_API_KEY)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="одна сессия и стоп")
    ap.add_argument("--dry-pod", action="store_true", help="не трогать RunPod (тест VK/логики)")
    ap.add_argument("-m", "--message", type=str, help="начальный промпт для сессии")
    ap.add_argument("--sas", type=int, default=0, help="ночной режим: N авто-сессий (auto-resume после сброса лимита)")
    args = ap.parse_args()

    hub.STATE.dry_pod = args.dry_pod
    hub.STATE.auto_left = args.sas   # ночной режим прямо при запуске
    _preflight()
    budget.clear_pause()

    # Если задан начальный промпт, поместить его в очередь. В связке с --sas (ночной режим) —
    # это заметка к текущему компоненту ("auto:"), а не разовая задача, иначе после неё раннер
    # встанет на паузу и ночной режим не продолжится непрерывно (та же логика, что в hub.py /sas).
    if args.message:
        prefix = "auto:" if args.sas > 0 else ""
        hub.STATE.start_queue.put_nowait(prefix + args.message)

    try:
        asyncio.run(main_service(once=args.once))
    except KeyboardInterrupt:
        print("\n[runner] прервано вручную")
        _safe_stop_pods()
        sys.exit(0)
