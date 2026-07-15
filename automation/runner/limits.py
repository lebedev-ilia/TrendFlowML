"""Единый модуль лимитов Claude — общий для ВСЕХ агентов (component-runner, assistant, main).

Anthropic НЕ публикует точные квоты Max (5ч-окно + недельный кап в «часах активного компьюта»).
Поэтому модуль сводит ТРИ источника в одну картину:
  1) НАШ учёт — суммарные токены/стоимость всех агентов в окнах 5ч и 7дней (общий usage_events.csv).
     Каждый процесс дописывает сюда через record(); это надёжный НИЖНИЙ предел потребления.
  2) Заголовки rate-limit — если агент словил ответ с anthropic-ratelimit-* (record_headers).
  3) Браузер claude.ai — точный «остаток %» и время сброса, если снят парсером (record_browser).

Использование:
  import limits; limits.record("opus-runner", tin, tout, cost)
  print(limits.status_text())          # или: python limits.py
"""
from __future__ import annotations
import csv
import json
import datetime as dt

import config
import settings

_H = 3600.0


def _now() -> float:
    return dt.datetime.now().timestamp()


# ------------------------------------------------------------ наш учёт (1)
def record(source: str, tokens_in: int = 0, tokens_out: int = 0, cost_usd: float = 0.0) -> None:
    new = not config.USAGE_LOG.exists()
    with open(config.USAGE_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "source", "tokens_in", "tokens_out", "cost_usd"])
        w.writerow([f"{_now():.0f}", source, int(tokens_in or 0), int(tokens_out or 0), f"{cost_usd or 0:.4f}"])


def _window(hours: float) -> dict:
    since = _now() - hours * _H
    tin = tout = 0
    cost = 0.0
    earliest = None
    per_src: dict[str, int] = {}
    if config.USAGE_LOG.exists():
        with open(config.USAGE_LOG) as f:
            for r in csv.DictReader(f):
                try:
                    ts = float(r["ts"])
                except (ValueError, KeyError):
                    continue
                if ts >= since:
                    ti, to = int(r.get("tokens_in") or 0), int(r.get("tokens_out") or 0)
                    tin += ti
                    tout += to
                    cost += float(r.get("cost_usd") or 0)
                    per_src[r.get("source", "?")] = per_src.get(r.get("source", "?"), 0) + ti + to
                    earliest = ts if earliest is None else min(earliest, ts)
    reset_in = (earliest + hours * _H - _now()) if earliest else 0.0
    return {"tokens": tin + tout, "tokens_in": tin, "tokens_out": tout, "cost": cost,
            "reset_in_h": max(0.0, reset_in / _H), "by_source": per_src}


# ---------------------------------------------------- заголовки/браузер (2,3)
def _load_snap() -> dict:
    if config.LIMITS_SNAPSHOT.exists():
        try:
            return json.loads(config.LIMITS_SNAPSHOT.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_snap(d: dict) -> None:
    config.LIMITS_SNAPSHOT.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def record_headers(headers: dict) -> None:
    """Сохранить все anthropic-ratelimit-* из ответа API (если агент их видит)."""
    rl = {k.lower(): v for k, v in (headers or {}).items() if k.lower().startswith("anthropic-ratelimit-")}
    if rl:
        snap = _load_snap()
        snap["headers"] = {"ts": _now(), "values": rl}
        _save_snap(snap)


def set_browser(metrics: dict, raw: str | None = None) -> None:
    """Записать произвольный набор метрик, снятых парсером claude.ai (гибко, любые ключи)."""
    snap = _load_snap()
    m = dict(metrics or {})
    m["ts"] = _now()
    if raw is not None:
        m["raw"] = raw[:2000]
    snap["browser"] = m
    _save_snap(snap)


def record_browser(five_hour_used_pct=None, weekly_used_pct=None,
                   five_hour_reset=None, weekly_reset=None,
                   credits_used_usd=None, credits_total_usd=None, raw=None) -> None:
    """Совместимость: маппит на set_browser."""
    set_browser({"five_hour_pct": five_hour_used_pct, "weekly_pct": weekly_used_pct,
                 "five_hour_reset": five_hour_reset, "weekly_reset": weekly_reset,
                 "credits_used_usd": credits_used_usd, "credits_total_usd": credits_total_usd}, raw)


def _age(ts: float) -> str:
    m = (_now() - ts) / 60.0
    return f"{m:.0f} мин назад" if m < 90 else f"{m/60:.1f} ч назад"


# ------------------------------------------------------------------ сводка
def browser() -> dict:
    return _load_snap().get("browser", {}) or {}


def browser_age_min() -> float | None:
    b = browser()
    return (_now() - b["ts"]) / 60.0 if b.get("ts") else None


STALE_AFTER_MIN = 10.0  # старше — НЕ доверяем цифре (см. ниже почему)


def max_used_pct() -> float | None:
    """Максимальный использованный % (5ч или неделя) из браузера — главный сигнал стопа (95%/97%).

    ВАЖНО (баг от 2026-07-16, исправлено): если снимок старше STALE_AFTER_MIN — считаем его
    НЕИЗВЕСТНЫМ (None), а не доверяем цифре. Иначе снимок из прошлой сессии (даже двухдневной
    давности — например, из-за упавшего Chrome/CDP) может вызвать ФАНТОМНЫЙ жёсткий стоп в первые
    секунды после старта раннера, пока limits_daemon.py ещё не сделал первый свежий скан. pod_watchdog
    и hooks.py уже трактуют None как «неизвестно, не блокируем» — это безопаснее, чем слепо доверять
    устаревшим данным."""
    age = browser_age_min()
    if age is not None and age > STALE_AFTER_MIN:
        return None
    b = browser()
    vals = [b.get("five_hour_pct"), b.get("weekly_pct")]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return max(vals) if vals else None


def status() -> dict:
    w5 = _window(5)
    w7 = _window(24 * 7)
    snap = _load_snap()
    cap5 = settings.get("five_hour_token_cap")
    cap7 = settings.get("weekly_token_cap")
    return {"w5": w5, "w7": w7, "snap": snap, "cap5": cap5, "cap7": cap7}


def status_text() -> str:
    import agents
    s = status()
    w5, w7 = s["w5"], s["w7"]
    lines = ["📊 Лимиты Claude:"]
    # Браузер claude.ai — ГЛАВНЫЙ источник (проценты), ведём первым.
    br = s["snap"].get("browser")
    if br:
        age = browser_age_min()
        stale = " ⚠️устарело" if (age is not None and age > 5) else ""

        def g(k):
            v = br.get(k)
            return v if v is not None else "?"
        lines.append(f"🌐 claude.ai ({_age(br['ts'])}{stale}):")
        lines.append(f"  5ч: {g('five_hour_pct')}% · сброс {g('five_hour_reset')}")
        lines.append(f"  Неделя: {g('weekly_pct')}% · сброс {g('weekly_reset')}")
        if br.get("fable_pct") is not None:
            lines.append(f"  Fable: {g('fable_pct')}%")
        if br.get("credits_used_usd") is not None or br.get("credits_pct") is not None:
            lines.append(f"  💳 Кредиты: ${g('credits_used_usd')} ({g('credits_pct')}%) · сброс {g('credits_reset')}")
    else:
        lines.append("🌐 claude.ai: данных нет — запусти парсер (/scanlimits).")
    lines.append(agents.text() + " — лимит делится между ними, учитывай чужую нагрузку.")
    lines.append(f"(наш учёт: 5ч {w5['tokens']:,} ток/${w5['cost']:.2f}, 7д ${w7['cost']:.2f})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(status_text())
