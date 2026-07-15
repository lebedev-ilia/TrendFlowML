#!/usr/bin/env python3
"""Парсер claude.ai — снимает реальные лимиты (%) и кредиты ($) через Playwright, пишет в limits.

ВАЖНО (капча): Playwright-Chromium детектится как бот → claude.ai кидает капчу при логине. Мы капчу
НЕ обходим. Логинься в СВОЁМ обычном Chrome (там капчи нет), а парсер лишь ЧИТАЕТ уже открытую сессию.

Рекомендуемый режим — CDP (подключение к твоему Chrome, без повторного логина):
  1) Закрой Chrome. Запусти его с портом отладки (профиль твой, сессия claude.ai сохранена):
       google-chrome --remote-debugging-port=9222        (Linux)
     Убедись, что в этом Chrome ты залогинен в claude.ai.
  2) python claude_limits_scraper.py --cdp 9222

Резервный режим — отдельный профиль на реальном Chrome (channel=chrome, тише детект):
  python claude_limits_scraper.py             # окно: залогинься один раз (капча возможна)
  python claude_limits_scraper.py --headless  # потом фоном

Селекторы claude.ai меняются, поэтому берём весь текст, разбираем эвристикой и сохраняем сырой текст в
state/claude_usage_raw.txt (для донастройки).
"""
from __future__ import annotations
import os
import re
import argparse

import config
import limits

PROFILE = os.environ.get("CLAUDE_PROFILE_DIR", str(config.STATE_DIR / "chrome_profile"))
URLS = os.environ.get(
    "CLAUDE_USAGE_URLS",
    "https://claude.ai/settings/usage,https://claude.ai/settings/billing",
).split(",")
RAW = config.STATE_DIR / "claude_usage_raw.txt"


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


LABELS = {
    # 5-часовое окно на claude.ai подписано как «Current session» / «5-hour».
    "five_hour": re.compile(r"5[\s-]?hour|current session|\bsession\b", re.I),
    "weekly":    re.compile(r"\bweek", re.I),
    "fable":     re.compile(r"fable", re.I),
    "credits":   re.compile(r"credit", re.I),
}
_PCT = re.compile(r"(\d{1,3})\s*%")
_RESET = re.compile(r"reset[s]?\b[^\n]*", re.I)
_DOLLAR = re.compile(r"\$([\d.,]+)")


def parse(text: str) -> dict:
    """Построчный проход: метка секции (5-hour/Weekly/Fable/Credits) задаёт, к чему относятся
    следующие проценты / 'Resets …' / '$…'. Устойчиво к тому, что значения на отдельных строках."""
    res: dict = {}
    section = None
    for ln in re.split(r"\n+", text):
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        for key, rx in LABELS.items():
            if rx.search(low):
                section = key
                break
        if not section:
            continue
        m = _PCT.search(s)
        if m:
            res.setdefault(f"{section}_pct", int(m.group(1)))
        r = _RESET.search(s)
        if r:
            res.setdefault(f"{section}_reset", r.group(0).strip()[:40])
        if section == "credits":
            d = _DOLLAR.search(s)
            if d:
                res.setdefault("credits_used_usd", _num(d.group(1)))
    return res


def _read_pages(page) -> str:
    texts = []
    for url in URLS:
        try:
            page.goto(url.strip(), wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3500)  # Дольше ждём — элементы могут быть динамическими
            # Нажать кнопку обновления актуальных данных, если есть.
            try:
                page.get_by_role("button", name=re.compile(r"refresh|update|reload|обнов", re.I)).first.click(timeout=4000)
                page.wait_for_timeout(3500)
            except Exception:
                pass
            # Пролистай вниз, может быть лимиты в скролле
            try:
                page.evaluate("window.scrollBy(0, 1000)")
                page.wait_for_timeout(1500)
            except Exception:
                pass
            texts.append(f"### {url}\n" + page.inner_text("body"))
        except Exception as e:
            texts.append(f"### {url}\n[ошибка: {e}]")
    return "\n\n".join(texts)


def scrape(headless: bool, cdp_port: int | None = None) -> str:
    from playwright.sync_api import sync_playwright
    from browser_lock import browser_lock
    with browser_lock(), sync_playwright() as p:
        if cdp_port:  # подключаемся к УЖЕ запущенному Chrome пользователя (без логина/капчи)
            browser = p.chromium.connect_over_cdp(f"http://localhost:{cdp_port}")
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            # Переиспользуем первую открытую вкладку вместо создания новой
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            try:
                return _read_pages(page)
            finally:
                # Вкладку не закрываем — переиспользуем при следующем запуске
                pass
        # свой профиль на РЕАЛЬНОМ Chrome (тише детект), без флага автоматизации
        kwargs = dict(headless=headless, viewport={"width": 1200, "height": 900},
                      args=["--disable-blink-features=AutomationControlled"])
        try:
            ctx = p.chromium.launch_persistent_context(PROFILE, channel="chrome", **kwargs)
        except Exception:
            ctx = p.chromium.launch_persistent_context(PROFILE, **kwargs)  # fallback на bundled
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            return _read_pages(page)
        finally:
            ctx.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--cdp", type=int, default=None, help="порт remote-debugging твоего Chrome (напр. 9222)")
    args = ap.parse_args()
    try:
        text = scrape(args.headless, cdp_port=args.cdp)
    except Exception as e:
        print(f"Playwright/браузер недоступен: {e}\n"
              f"Установи: pip install playwright && playwright install chromium")
        return
    RAW.write_text(text)
    logged = not re.search(r"\b(log in|sign in|войти)\b", text.lower())
    if not logged:
        print("⚠️ Не залогинен в claude.ai. Запусти БЕЗ --headless и войди в открывшемся окне, затем повтори.")
    res = parse(text)
    limits.set_browser(res, raw=text)
    print("Распознано:", res or "(ничего — см. state/claude_usage_raw.txt, пришли его мне для настройки)")
    print(limits.status_text())


if __name__ == "__main__":
    main()
