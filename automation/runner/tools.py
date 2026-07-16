"""Кастомные SDK-инструменты для агента: ask_human, pod_control.

Оба завязаны на VK: вопросы и подтверждения действий с подом идут владельцу в личные сообщения.
"""
from __future__ import annotations

from claude_agent_sdk import tool, create_sdk_mcp_server

import messenger
import runpod_api
import hub
import budget
import config
import settings


def _pod_price(pod: dict) -> tuple[str, float]:
    """Определить (имя GPU, цена/час) для существующего пода. Фолбэк — config.POD_HOURLY_USD."""
    name = (pod.get("machine") or {}).get("gpuType") or pod.get("gpuTypeId") or pod.get("gpuTypeIds") or ""
    if isinstance(name, list):
        name = name[0] if name else ""
    try:
        for g in runpod_api.list_gpu_types():
            gid = g.get("id") or ""
            if gid and (gid in str(name) or str(name) in gid) and g.get("price"):
                return gid, float(g["price"])
    except Exception:
        pass
    return (str(name) or "unknown"), float(settings.get("pod_hourly_usd"))


def _is_yes(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in ("да", "yes", "y", "ок", "ok", "давай", "+", "го", "подтверждаю")


# ---------------------------------------------------------------------------
@tool(
    "ask_human",
    "Задать владельцу вопрос в VK и дождаться ответа. Используй ТОЛЬКО если ответа нет в "
    "интернете/документации и нужно продуктовое решение владельца (веса, доступы, приоритет). "
    "Всегда прикладывай краткий контекст и, если возможно, варианты.",
    {"question": str, "context": str},
)
async def ask_human(args):
    q = args["question"].strip()
    ctx = (args.get("context") or "").strip()
    text = f"❓ Вопрос от Opus:\n{q}" + (f"\nКонтекст: {ctx}" if ctx else "")

    # Старое поведение (если авто-ответ выключен): ждём владельца.
    if not int(settings.get("auto_answer") or 0):
        reply = await hub.get_reply(text)
        if reply == "__TIMEOUT__":
            return {"content": [{"type": "text", "text": "Владелец не ответил. Прими безопасный дефолт и отметь в отчёте."}]}
        return {"content": [{"type": "text", "text": f"Ответ владельца: {reply}"}]}

    # Авто-ответ супервайзера (Haiku), чтобы работа не простаивала.
    import supervisor
    auto = await supervisor.answer(q, ctx)
    qs = q if len(q) < 300 else q[:300] + "…"

    if auto.upper().startswith("NEEDS_OWNER"):
        # Реально нужен владелец — эскалируем надолго (с обычной логикой пауз/стопа пода).
        reply = await hub.get_reply(f"❓ Opus (нужно ТВОЁ решение):\n{qs}\n🧭 Супервайзер: {auto}")
        final = reply if reply != "__TIMEOUT__" else "Владелец недоступен — прими безопасный консервативный дефолт и отметь в отчёте."
        return {"content": [{"type": "text", "text": f"Ответ: {final}"}]}

    # Обычный вопрос: показываем владельцу Q&A + короткое окно на переопределение.
    wait = int(settings.get("auto_answer_wait_sec") or 180)
    reply = await hub.wait_owner(
        f"💬 Opus↔Супервайзер\n❓ {qs}\n🤖 {auto}\n(ответь в {wait//60} мин, чтобы переопределить)", wait)
    final = reply or auto
    return {"content": [{"type": "text", "text": f"Ответ: {final}"}]}


# ---------------------------------------------------------------------------
@tool(
    "pod_control",
    "Управление GPU-подом RunPod. action ∈ {status, start, stop_all, migrate, create}. "
    "start: пробует поднять под, при недоступности GPU автоматически пробует другой под и сообщает "
    "владельцу в VK. migrate/create: автономно берёт самую дешёвую доступную GPU ≤ ценового потолка. "
    "Потолок ЗАВИСИТ от arg: если работа короткая (тест/smoke/быстрый прогон) — передай arg='short' "
    "(или 'тест'/'быстро') — тогда потолок выше (см. settings max_pod_hourly_usd_short), иначе обычный "
    "потолок (max_pod_hourly_usd). Прогнозируй длительность внимательно — короткий потолок только для "
    "реально коротких прогонов. stop_all: гасит все поды. status — без подтверждения. Гаси под "
    "(stop_all) после каждого компонента и при паузах.",
    {"action": str, "arg": str},
)
async def pod_control(args):
    import config
    action = (args.get("action") or "").strip().lower()
    arg = (args.get("arg") or "").strip().lower()
    is_short = any(w in arg for w in ("short", "quick", "тест", "быстр", "smoke"))

    if action == "status":
        return {"content": [{"type": "text", "text": runpod_api.status_summary()}]}

    if action == "start":
        pods = runpod_api.own_pods()  # чужая инфра (напр. Fetcher) исключена
        if not pods:
            return {"content": [{"type": "text",
                    "text": "Подов в аккаунте нет. Используй action=migrate, чтобы создать новый под "
                            "с Network Volume (после подтверждения владельца)."}]}
        # Порядок попыток: закреплённый POD_ID (если задан), иначе все остановленные, затем остальные.
        order = []
        if config.RUNPOD_POD_ID:
            order = [p for p in pods if p.get("id") == config.RUNPOD_POD_ID] or pods
        else:
            stopped = [p for p in pods if str(p.get("desiredStatus") or p.get("status")).upper() != "RUNNING"]
            order = stopped + [p for p in pods if p not in stopped]

        if hub.STATE.dry_pod:
            return {"content": [{"type": "text", "text": f"[dry-pod] Пропускаю реальный старт. Кандидаты: {[p.get('id') for p in order]}"}]}

        tried = []
        for p in order:
            pid = p.get("id")
            ok, msg = runpod_api.start_pod(pid)
            tried.append(f"{pid}: {'старт' if ok else msg}")
            if not ok:
                messenger.send(f"⚠️ Под {pid}: не стартовал (вероятно нет свободного GPU). Пробую следующий…")
                continue
            ep = runpod_api.wait_running(pid, timeout=150)
            if ep:
                gpu_name, price = _pod_price(p)
                budget.pod_open(pid, gpu_name, price)
                hub.STATE.pod_running = True
                hub.STATE.last_activity = __import__("time").time()
                hrs = budget.affordable_hours(price)
                hrs_txt = f"~{hrs:.1f} ч" if hrs is not None else "баланс RunPod н/д"
                messenger.send(f"✅ Под {pid} поднят: {ep} (${price:.3f}/ч). Баланса RunPod хватит на {hrs_txt}.")
                return {"content": [{"type": "text",
                        "text": f"Под {pid} RUNNING, SSH {ep}. Ставка ${price:.3f}/ч, "
                                f"баланса RunPod хватит на {hrs_txt} — планируй прогоны с запасом "
                                f"(тормози заранее, проверяй budget_status). Работай по нему."}]}
            messenger.send(f"⚠️ Под {pid}: не поднялся за 150с (нет GPU?). Гашу его и пробую следующий…")
            runpod_api.stop_pod(pid)
        # Ни один не поднялся
        messenger.send("❌ Ни один под не поднялся — вероятно, в дата-центре нет свободного A4500. "
                       "Могу пересоздать под (migrate). Данные на Network Volume целы.")
        return {"content": [{"type": "text",
                "text": "GPU недоступен ни на одном поде: " + "; ".join(tried) +
                        ". Сообщи владельцу и вызови pod_control action=migrate для пересоздания."}]}

    if action in ("migrate", "create"):
        return await _create_with_menu(short=is_short)

    if action == "stop_all":
        if hub.STATE.dry_pod:
            return {"content": [{"type": "text", "text": "[dry-pod] Пропускаю реальную остановку подов."}]}
        killed = runpod_api.terminate_all_and_bill()  # УДАЛЯЕМ (не останавливаем)
        hub.STATE.pod_running = False
        return {"content": [{"type": "text",
                "text": f"Поды УДАЛЕНЫ: {killed or 'нет'}. GPU сегодня ${budget.gpu_spent_today():.2f}."}]}

    return {"content": [{"type": "text", "text": f"Неизвестный action={action!r}. Разрешено: status|start|stop_all|migrate|create."}]}


async def _create_with_menu(short: bool = False):
    """Создать новый под с Network Volume, автономно выбирая самую дешёвую GPU ≤ потолка.
    short=True — это КОРОТКИЙ/быстрый прогон (тест, smoke) → используем более высокий потолок
    (max_pod_hourly_usd_short), иначе обычный (max_pod_hourly_usd). Прогнозировать длительность —
    ответственность вызывающего агента; ошибка в прогнозе стоит реальных денег."""
    import time as _t
    pods = runpod_api.own_pods()  # чужая инфра (напр. Fetcher) исключена — не годится как шаблон
    # Образец конфигурации (image/диск/порты/volume) берём из существующего пода, если он есть.
    template = {}
    src_id = None
    if pods:
        src_id = pods[0].get("id")
        try:
            full = runpod_api.get_pod(src_id)
            template = runpod_api.clone_spec(full)
        except Exception:
            template = runpod_api.clone_spec(pods[0])

    try:
        gpus = runpod_api.list_gpu_types()
    except Exception as e:
        messenger.send(f"❗ Не смог получить список GPU: {e}. Создай/мигрируй под вручную на runpod.io.")
        return {"content": [{"type": "text", "text": f"list_gpu_types упал: {e}. Нужна ручная миграция владельцем."}]}

    cap_key = "max_pod_hourly_usd_short" if short else "max_pod_hourly_usd"
    cap = float(settings.get(cap_key) or (0.60 if short else 0.30))
    avail = [g for g in gpus if g.get("price") is not None and g["price"] <= cap]
    avail.sort(key=lambda g: g["price"])  # дешёвые первыми
    if not avail:
        messenger.send(f"❗ Нет GPU дешевле ${cap:.2f}/ч{' (короткий прогон)' if short else ''}. Жду появления доступных.")
        return {"content": [{"type": "text", "text": f"Нет доступных GPU ≤ ${cap:.2f}/ч. Повтори позже."}]}

    if hub.STATE.dry_pod:
        menu = "\n".join(f"{g['displayName']} ${g['price']:.3f}/ч" for g in avail[:8])
        return {"content": [{"type": "text", "text": "[dry-pod] Кандидаты ≤ cap:\n" + menu}]}

    # ЖЁСТКО: под ДОЛЖЕН монтировать НАШ Network Volume в ЕГО датацентре, иначе пустой volume/чужой DC.
    vi = runpod_api.volume_info()
    if not vi.get("id") or not vi.get("dc"):
        messenger.send("❗ Не нашёл наш Network Volume — НЕ создаю под (был бы пустой/чужой датацентр).")
        return {"content": [{"type": "text", "text": "volume_info пуст — создание отменено, чтобы не поднять бесполезный под."}]}

    # АВТОНОМНО: берём самую дешёвую доступную ≤ cap, пробуем по возрастанию цены. Без подтверждения.
    tried = []
    for chosen in avail[:8]:
        body = dict(template)
        body["name"] = "trendflow-" + str(int(_t.time()))
        body["gpuTypeIds"] = [chosen["id"]]
        body["gpuCount"] = 1
        body["cloudType"] = chosen["cloud"]
        body["networkVolumeId"] = vi["id"]     # НАШ том
        body["dataCenterIds"] = [vi["dc"]]     # его датацентр (обязательно! RunPod API: массив)
        body["volumeMountPath"] = "/workspace"
        if not body.get("imageName"):
            body["imageName"] = config_default_image()
        messenger.send(f"⏳ Пробую {chosen['displayName']} (${chosen['price']:.3f}/ч)…")
        try:
            new = runpod_api.create_pod(body)
        except Exception as e:
            tried.append(f"{chosen['displayName']}: {str(e)[:120]}")
            continue
        new_id = new.get("id") or (new.get("pod") or {}).get("id")
        ep = runpod_api.wait_running(new_id, timeout=180) if new_id else None
        if new_id:
            budget.pod_open(new_id, chosen["displayName"], chosen["price"])
        hub.STATE.pod_running = True
        hub.STATE.last_activity = _t.time()
        messenger.send(f"✅ Создан под {new_id} на {chosen['displayName']} (${chosen['price']:.3f}/ч)" +
                       (f", SSH {ep}" if ep else " (ждёт готовности)"))
        return {"content": [{"type": "text",
                "text": f"Новый под {new_id} на {chosen['displayName']}" +
                        (f", SSH {ep}. Работай по нему." if ep else ", проверь pod_control status.")}]}
    messenger.send("❗ Ни одна из выбранных GPU не создалась:\n" + "\n".join(tried[:6]))
    return {"content": [{"type": "text", "text": "Не удалось создать под ни на одной из выбранных GPU: " + "; ".join(tried)}]}


def _parse_gpu_choice(reply: str, n: int) -> list[int]:
    """Разобрать ответ владельца в список 0-based индексов (по порядку попыток)."""
    import re
    t = (reply or "").strip().lower()
    # «любую/любой/подешевле/первую доступную» -> все по порядку (avail уже отсортирован по цене)
    if any(w in t for w in ("любую", "любой", "подешевле", "дешёв", "дешев", "первую", "по порядку", "все")):
        return list(range(n))
    # диапазон: «1-6», «1 - 6», «от 1 до 6», «с 1 по 6»
    m = re.search(r"(\d+)\s*(?:-|–|до|по)\s*(\d+)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        return [i - 1 for i in range(lo, hi + 1) if 1 <= i <= n]
    # список чисел: «1,3,5» или «1 3 5»
    nums = [int(x) for x in re.findall(r"\d+", t)]
    order, seen = [], set()
    for x in nums:
        if 1 <= x <= n and x - 1 not in seen:
            order.append(x - 1)
            seen.add(x - 1)
    return order


def config_default_image() -> str:
    import os
    return os.environ.get("RUNPOD_DEFAULT_IMAGE",
                          "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")


@tool(
    "manager",
    "PodManager — единый учёт/управление ВСЕМИ машинами (GPU-поды сейчас; ноды/сайт/БД позже). "
    "action ∈ {list, health <id>, create, migrate, stop <id|all>, register <id> <kind> <purpose>, forget <id>}. "
    "list/health/register/forget — без подтверждения. create/migrate/stop — деньги, спрашивают владельца в VK. "
    "Через arg передавай аргументы действия строкой (напр. 'health abc123' или 'stop all').",
    {"action": str, "arg": str},
)
async def manager(args):
    import podmanager
    action = (args.get("action") or "").strip().lower()
    arg = (args.get("arg") or "").strip()

    if action in ("list", "status"):
        return {"content": [{"type": "text", "text": podmanager.summary_text()}]}
    if action == "health":
        return {"content": [{"type": "text", "text": podmanager.health(arg)}]}
    if action == "register":
        parts = arg.split(None, 2)
        mid = parts[0] if parts else ""
        kind = parts[1] if len(parts) > 1 else "other"
        purpose = parts[2] if len(parts) > 2 else ""
        if not mid:
            return {"content": [{"type": "text", "text": "Формат: register <id> <kind> <purpose>"}]}
        podmanager.register(mid, kind=kind, purpose=purpose)
        return {"content": [{"type": "text", "text": f"Зарегистрирована машина {mid} ({kind})."}]}
    if action == "forget":
        ok = podmanager.forget(arg)
        return {"content": [{"type": "text", "text": f"{'Удалена' if ok else 'Не найдена'}: {arg}"}]}
    if action in ("create", "migrate"):
        is_short = any(w in arg for w in ("short", "quick", "тест", "быстр", "smoke"))
        return await _create_with_menu(short=is_short)  # автономный выбор GPU ≤ ценового потолка
    if action == "stop":
        if hub.STATE.dry_pod:
            return {"content": [{"type": "text", "text": "[dry-pod] Пропускаю стоп."}]}
        target = arg or "all"
        if target == "all":
            stopped = runpod_api.terminate_all_and_bill()
            hub.STATE.pod_running = False
        else:
            budget.pod_close(target)
            runpod_api.delete_pod(target)
            stopped = [target]
        return {"content": [{"type": "text", "text": f"Удалено: {stopped}"}]}
    return {"content": [{"type": "text", "text": f"Неизвестный action={action!r}. См. описание инструмента."}]}


@tool(
    "limits_status",
    "Показать лимиты Claude аккаунта (5-часовое окно и недельный, суммарно по всем агентам). "
    "Вызывай, чтобы решить, хватит ли лимита на длинный прогон/сессию.",
    {},
)
async def limits_status(args):
    import limits
    return {"content": [{"type": "text", "text": limits.status_text()}]}


@tool(
    "budget_status",
    "Показать траты за сегодня (учёт, не лимит), ставку текущих подов, РЕАЛЬНЫЙ баланс аккаунта RunPod "
    "и на сколько ЧАСОВ пода его хватит на текущей ставке. Нет дневного $-лимита — единственные реальные "
    "ограничители: баланс RunPod (пополни, если предупреждение) и ценовой потолок пода. Вызывай ПЕРЕД "
    "длинными GPU-прогонами, чтобы не остаться без баланса посреди прогона.",
    {},
)
async def budget_status(args):
    return {"content": [{"type": "text", "text": budget.status_text()}]}


def build_server():
    return create_sdk_mcp_server(name="trendflow", version="1.0.0",
                                 tools=[ask_human, pod_control, manager, budget_status, limits_status])
