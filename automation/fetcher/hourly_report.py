#!/usr/bin/env python3
"""Часовой отчёт метрик Fetcher — ЧИСТЫЙ КОД, без LLM (дешевле/надёжнее детерминированной сводки
чисел). SSH на все 3 пода -> читает state/inventory/summary.json (уже готовые lag-метрики, см.
FETCHER_DATASET_COLLECTOR_HANDOFF.md §13) -> шлёт агрегированную сводку в VK (Третий бот).

Пишет в тот же state/agent3_chat.log, что и watchdog.py — Второй агент (в этой же переписке) видит
отчёт наравне с владельцем, как просил владелец ("независимо от агента, но он должен это видеть").

Запуск: python hourly_report.py            # разовый прогон
        python hourly_report.py --loop      # бесконечный цикл раз в HOURLY_REPORT_INTERVAL_SEC
"""
from __future__ import annotations
import argparse
import datetime as dt
import random
import time

import requests

import config
import deploy

VK = "https://api.vk.com/method"


def _log_chat(direction: str, text: str) -> None:
    try:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {direction} {text[:2000]}\n"
        with open(config.STATE_DIR / "fetcher_chat.log", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def send(text: str) -> None:
    _log_chat("REPORT->", text)
    if not config.VK_TOKEN3:
        return
    try:
        requests.post(f"{VK}/messages.send", data={
            "user_id": config.VK_OWNER_ID, "random_id": random.randint(1, 2_000_000_000),
            "message": text[:4000], "access_token": config.VK_TOKEN3, "v": config.VK_API_VERSION,
        }, timeout=40)
    except requests.RequestException:
        pass


def _lifecycle(summary: dict | None) -> dict:
    """Реальная схема summary.json (подтверждена на живом поде 2026-07-16):
    {"totals": {"lifecycle": {accepted, enriched, downloaded_or_on_hf, lag_download, lag_enrich,
    lag_hf_video, lag_hf_enrich, ...}, "videos": {...}, "shards": {...}}, "by_category": {name: {...}}}
    НЕ "counters"/"category_counters" верхнего уровня — это была ошибочная догадка при первом
    написании отчёта (без живых данных под рукой), из-за которой отчёт слал одни "?"."""
    if not summary:
        return {}
    return (summary.get("totals") or {}).get("lifecycle") or {}


def _fmt_summary(pod_name: str, summary: dict | None) -> str:
    if summary is None:
        return f"  {pod_name}: нет данных (кампания ещё не запущена или SSH недоступен)"
    c = _lifecycle(summary)
    if not c:
        return f"  {pod_name}: summary.json есть, но пустой/непривычной формы"
    accepted = c.get("accepted", "?")
    enriched = c.get("enriched", "?")
    downloaded = c.get("downloaded_or_on_hf", "?")
    lag_dl = c.get("lag_download", "?")
    lag_en = c.get("lag_enrich", "?")
    lag_hf_v = c.get("lag_hf_video", "?")
    lag_hf_e = c.get("lag_hf_enrich", "?")
    return (f"  {pod_name}: accepted={accepted} enriched={enriched} downloaded/on_hf={downloaded} | "
           f"lag: download={lag_dl} enrich={lag_en} hf_video={lag_hf_v} hf_enrich={lag_hf_e}")


def _category_breakdown(summary: dict | None) -> str:
    if not summary:
        return ""
    cats = summary.get("by_category")
    if not cats:
        return ""
    def _accepted(item):
        lc = (item[1] or {}).get("lifecycle") or {}
        try:
            return -int(lc.get("accepted") or 0)
        except Exception:
            return 0
    lines = []
    for name, entry in sorted(cats.items(), key=_accepted)[:6]:
        lc = (entry or {}).get("lifecycle") or {}
        lines.append(f"    {name}: accepted={lc.get('accepted', '?')} enriched={lc.get('enriched', '?')} "
                    f"on_hf={lc.get('downloaded_or_on_hf', '?')}")
    return "\n".join(lines)


def _fmt_snapshot_status(status: dict | None) -> str:
    """Добавлено 2026-07-24 (владелец): отчёт должен показывать прогресс повторных снапшотов
    (views/likes через 7/14/21/28д от discover), не только download/enrich/upload lifecycle."""
    if not status:
        return "  снапшоты: нет данных (процесс не запущен или SSH недоступен)"
    schedule_size = status.get("schedule_size", 0)
    per_index = status.get("per_index") or {}
    if not schedule_size:
        return "  снапшоты: расписание пустое (видео ещё не набралось)"
    lines = [f"  снапшоты (расписание {schedule_size} видео):"]
    labels = {"1": "+7д", "2": "+14д", "3": "+21д", "4": "+28д"}
    for idx in sorted(per_index.keys(), key=lambda x: int(x)):
        e = per_index[idx]
        label = labels.get(idx, f"#{idx}")
        due_now = e.get("due_now", 0)
        done = e.get("done", 0)
        pending = e.get("pending", 0)
        flag = " ⏳ ЖДЁТ СБОРА" if due_now else ""
        lines.append(f"    {label}: готово {done}/{schedule_size}, ждут своего срока {pending - done}, "
                     f"пора собирать сейчас {due_now}{flag}")
    next_due = status.get("next_due_at")
    if next_due:
        lines.append(f"    следующий дедлайн: {next_due[:16].replace('T', ' ')} UTC")
    return "\n".join(lines)


def build_report() -> str:
    lines = [f"📊 Fetcher — часовой отчёт ({dt.datetime.now().strftime('%Y-%m-%d %H:%M')} локальное)"]
    total_accepted = 0
    any_data = False
    for pod_name in config.PODS:
        try:
            summary = deploy.read_inventory_summary(pod_name)
        except Exception as e:
            summary = None
            lines.append(f"  {pod_name}: ошибка SSH/чтения ({e})")
            continue
        lines.append(_fmt_summary(pod_name, summary))
        if summary:
            any_data = True
            c = _lifecycle(summary)
            try:
                total_accepted = max(total_accepted, int(c.get("accepted") or 0))
            except Exception:
                pass
    if not any_data:
        lines.append("")
        lines.append("Кампания ещё не запущена ни на одном поде (нет summary.json) — это ожидаемо, "
                     "пока не переданы секреты (HF_TOKEN/keys.txt/cookies) и не выполнен launch.")
        return "\n".join(lines)
    # По категориям — с main-пода (там актуальнее всего discovery-счётчики).
    try:
        main_summary = deploy.read_inventory_summary("fetcher-main")
        cat_lines = _category_breakdown(main_summary)
        if cat_lines:
            lines.append("")
            lines.append("  Топ категорий (по main):")
            lines.append(cat_lines)
    except Exception:
        pass
    lines.append("")
    lines.append(f"Всего discovered (по последнему видимому счётчику): ~{total_accepted}")
    # Снапшоты (views/likes через 7/14/21/28д) — читается с fetcher-main, где крутится snapshot-poll.
    try:
        snap_status = deploy.read_snapshot_status("fetcher-main")
    except Exception:
        snap_status = None
    lines.append("")
    lines.append(_fmt_snapshot_status(snap_status))
    return "\n".join(lines)


def run_once() -> None:
    report = build_report()
    print(report, flush=True)
    send(report)


def run_loop() -> None:
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[hourly_report] ошибка: {e}", flush=True)
        time.sleep(max(60, config.HOURLY_REPORT_INTERVAL_SEC))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    if args.loop:
        run_loop()
    else:
        run_once()
