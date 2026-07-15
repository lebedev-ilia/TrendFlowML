"""Учёт расходов: модель (токены→$) + аренда подов ($/час) + журнал сессий.

ЭТО УЧЁТ, НЕ ЛИМИТ: нет дневного $-бюджета, который бы ставил сессию на паузу. Единственные реальные
ограничители — цена пода (settings.max_pod_hourly_usd[_short]) и фактический баланс RunPod
(runpod_api.account_balance(), предупреждение при settings.runpod_balance_warn_usd — см. AGENT_CONTEXT.md
раздел 0). budget_status показывает траты за день + сколько ЧАСОВ пода вытянет текущий баланс RunPod.
"""
from __future__ import annotations
import csv
import json
import datetime as dt
from pathlib import Path

import config
import settings


def _today() -> str:
    return dt.date.today().isoformat()


def _now() -> float:
    return dt.datetime.now().timestamp()


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


# ---------------------------------------------------------------- модель ($)
def _ensure(path: Path, header: list[str]):
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def record(component: str, model_usd: float, note: str = "") -> None:
    _ensure(config.SPEND_LOG, ["date", "component", "model_usd", "note"])
    with open(config.SPEND_LOG, "a", newline="") as f:
        csv.writer(f).writerow([_today(), component, f"{model_usd:.4f}", note])


def model_spent_today() -> float:
    return _sum_today(config.SPEND_LOG, "model_usd")


# ------------------------------------------------------------------ поды ($)
def _load_open() -> dict:
    if config.OPEN_PODS.exists():
        try:
            return json.loads(config.OPEN_PODS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_open(d: dict) -> None:
    config.OPEN_PODS.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def pod_open(pod_id: str, gpu: str, price_per_hr: float) -> None:
    d = _load_open()
    d[pod_id] = {"gpu": gpu, "price": float(price_per_hr or 0), "start": _now()}
    _save_open(d)


def pod_close(pod_id: str) -> float:
    d = _load_open()
    rec = d.pop(pod_id, None)
    _save_open(d)
    if not rec:
        return 0.0
    hours = max(0.0, (_now() - rec["start"]) / 3600.0)
    usd = hours * rec["price"]
    _ensure(config.POD_LEDGER, ["date", "pod_id", "gpu", "price_per_hr", "start", "stop", "hours", "usd"])
    with open(config.POD_LEDGER, "a", newline="") as f:
        csv.writer(f).writerow([_today(), pod_id, rec["gpu"], f"{rec['price']:.4f}",
                                _iso(rec["start"]), _iso(_now()), f"{hours:.3f}", f"{usd:.4f}"])
    return usd


def open_pod_accrued() -> float:
    """Ещё не закрытая стоимость поднятых прямо сейчас подов."""
    total = 0.0
    for rec in _load_open().values():
        total += max(0.0, (_now() - rec["start"]) / 3600.0) * rec["price"]
    return total


def current_hourly() -> float:
    """Суммарная ставка $/час поднятых сейчас подов (0 если подов нет)."""
    return sum(r.get("price", 0.0) for r in _load_open().values())


def gpu_spent_today() -> float:
    return _sum_today(config.POD_LEDGER, "usd") + open_pod_accrued()


# ------------------------------------------------------------------- общее
def _sum_today(path: Path, col: str) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("date") == _today():
                try:
                    total += float(row.get(col) or 0)
                except ValueError:
                    pass
    return total


def total_spent_today() -> float:
    return model_spent_today() + gpu_spent_today()


def spent_today() -> float:  # обратная совместимость
    return total_spent_today()


def runpod_balance() -> float | None:
    """Текущий баланс аккаунта RunPod в $ (None если не удалось получить — не считать это нулём!)."""
    import runpod_api
    return runpod_api.account_balance()


def runpod_balance_low() -> bool:
    bal = runpod_balance()
    return bal is not None and bal < float(settings.get("runpod_balance_warn_usd"))


def affordable_hours(price_per_hr: float | None = None) -> float | None:
    """Сколько ЧАСОВ пода ещё вытянет РЕАЛЬНЫЙ баланс RunPod (не дневной бюджет — его больше нет).
    None если баланс не удалось получить (не блокируем работу, просто не знаем число)."""
    price = price_per_hr if price_per_hr else (current_hourly() or settings.get("pod_hourly_usd"))
    if price <= 0:
        return None
    bal = runpod_balance()
    if bal is None:
        return None
    return max(0.0, bal) / price


def status_text() -> str:
    bal = runpod_balance()
    bal_txt = f"${bal:.2f}" if bal is not None else "н/д (не смог получить)"
    warn = " ⚠️ НИЗКИЙ БАЛАНС" if (bal is not None and bal < float(settings.get("runpod_balance_warn_usd"))) else ""
    hrs = affordable_hours()
    hrs_txt = f"~{hrs:.1f} ч" if hrs is not None else "н/д"
    return (
        f"Потрачено сегодня ${total_spent_today():.2f} (модель ${model_spent_today():.2f} + "
        f"GPU ${gpu_spent_today():.2f}) — это УЧЁТ, не лимит. "
        f"Ставка подов сейчас ${current_hourly():.3f}/ч. "
        f"Баланс RunPod: {bal_txt}{warn}. На текущей ставке хватит на {hrs_txt} пода."
    )


# --------------------------------------------------------------- сессии
def record_session(label: str, mode: str, model_usd: float, gpu_usd: float,
                   tokens: int, started: float, note: str = "") -> None:
    _ensure(config.SESSIONS_LOG,
            ["date", "start", "end", "label", "mode", "model_usd", "gpu_usd", "tokens", "note"])
    with open(config.SESSIONS_LOG, "a", newline="") as f:
        csv.writer(f).writerow([_today(), _iso(started), _iso(_now()), label, mode,
                                f"{model_usd:.4f}", f"{gpu_usd:.4f}", tokens, note])


# ---------------------------------------------------------------- пауза
def set_pause(reason: str) -> None:
    Path(config.PAUSE_FLAG).write_text(f"{dt.datetime.now().isoformat()} {reason}\n")


def clear_pause() -> None:
    if config.PAUSE_FLAG.exists():
        config.PAUSE_FLAG.unlink()


def is_paused() -> bool:
    return config.PAUSE_FLAG.exists()
