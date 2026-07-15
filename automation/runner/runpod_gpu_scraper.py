#!/usr/bin/env python3
"""Парсер console.runpod.io/deploy через Chrome (CDP) — ТОЧНЫЕ цены и наличие GPU для нашего
Network Volume (GraphQL врёт глобальными ценами). По аналогии с claude_limits_scraper.

Логика: открыть /deploy → выбрать наш Network Volume (обновляет список GPU в его датацентре) →
снять карточки GPU (имя, $/час, наличие) → записать в state/runpod_gpus.json.

Запуск: python runpod_gpu_scraper.py --cdp 9222   (тот же залогиненный Chrome, что и для лимитов)
"""
from __future__ import annotations
import re
import json
import argparse

import config

URL = "https://console.runpod.io/deploy"
RAW = config.STATE_DIR / "runpod_gpus_raw.txt"

_NAME = re.compile(r"\b(RTX\s?(?:PRO\s)?\d{3,4}(?:\s?(?:Ti|Ada|SUPER))?|A\d{3,4}|A100|A40|A30|H100|H200|L4|L40S?|"
                   r"MI300X?|B200|B300|V100|RTX\s?A\d{4})\b", re.I)
_PRICE = re.compile(r"\$\s*([\d.]+)\s*/?\s*(?:hr|hour|ч)", re.I)
_STOCK = re.compile(r"\b(high|medium|low|unavailable|out of stock|available)\b", re.I)


def parse(text: str) -> list[dict]:
    lines = [l.strip() for l in re.split(r"\n+", text) if l.strip()]
    gpus = []
    cur = None
    for ln in lines:
        nm = _NAME.search(ln)
        if nm:
            cur = nm.group(0)
        pm = _PRICE.search(ln)
        if pm and cur:
            sm = _STOCK.search(ln)
            gpus.append({"name": cur, "price": float(pm.group(1)),
                         "stock": (sm.group(1).lower() if sm else "available")})
            cur = None
    # dedup по имени (min цена)
    best = {}
    for g in gpus:
        if g["name"] not in best or g["price"] < best[g["name"]]["price"]:
            best[g["name"]] = g
    out = sorted(best.values(), key=lambda x: x["price"])
    return out


def _select_volume(page) -> bool:
    """Открыть дропдаун Network volume и выбрать наш том. Несколько стратегий."""
    vol = config.RUNPOD_VOLUME_NAME
    openers = [
        lambda: page.get_by_role("button", name=re.compile(r"network volume", re.I)).first.click(timeout=5000),
        lambda: page.get_by_text(re.compile(r"network volume", re.I)).first.click(timeout=5000),
        lambda: page.get_by_role("combobox").first.click(timeout=5000),
    ]
    for op in openers:
        try:
            op()
            page.wait_for_timeout(1200)
            page.get_by_text(vol, exact=False).first.click(timeout=5000)
            page.wait_for_timeout(5000)  # список GPU обновляется под датацентр тома
            return True
        except Exception:
            continue
    print(f"[warn] не смог выбрать volume '{vol}' — сниму страницу как есть (пришли raw для настройки).")
    return False


def scrape(cdp_port: int) -> str:
    from playwright.sync_api import sync_playwright
    from browser_lock import browser_lock
    with browser_lock(), sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://localhost:{cdp_port}")
        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
        page = ctx.new_page()
        try:
            # SPA RunPod не даёт "networkidle" → ждём domcontentloaded + фикс-паузу.
            page.goto(URL, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(8000)
            _select_volume(page)
            return page.inner_text("body")
        finally:
            page.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdp", type=int, default=int(config.CLAUDE_CDP_PORT or 9222))
    args = ap.parse_args()
    try:
        text = scrape(args.cdp)
    except Exception as e:
        print(f"Playwright/Chrome недоступен: {e}. Запусти Chrome с --remote-debugging-port и войди в RunPod.")
        return
    RAW.write_text(text)
    gpus = parse(text)
    config.RUNPOD_GPUS_JSON.write_text(json.dumps({"gpus": gpus, "ts": __import__("time").time()},
                                                  ensure_ascii=False, indent=2))
    if gpus:
        print(f"Датацентр volume '{config.RUNPOD_VOLUME_NAME}'. Доступные GPU:")
        for g in gpus:
            print(f"  {g['name']:<20} ${g['price']}/ч  stock={g['stock']}")
    else:
        print("Ничего не распознал — см. state/runpod_gpus_raw.txt, пришли его мне для настройки.")


if __name__ == "__main__":
    main()
