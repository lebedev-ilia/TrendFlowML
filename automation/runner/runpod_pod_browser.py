#!/usr/bin/env python3
"""ФОЛБЭК: управление подами через Chrome (CDP), когда RunPod API не срабатывает.

CLI:
  python runpod_pod_browser.py list                 # список подов (текст страницы /pods)
  python runpod_pod_browser.py terminate <pod_id>   # удалить под (⋮ → Terminate → подтвердить)
  python runpod_pod_browser.py create <gpu_name>    # создать под (выбрать volume+GPU → Deploy On-Demand → SSH)

Нужен тот же залогиненный Chrome с CDP (console.runpod.io). Селекторы заданы по описанию владельца —
при расхождении вёрстки пришли raw-текст, подгоню.
"""
from __future__ import annotations
import re
import sys

import config
import settings
import runpod_api
from browser_lock import browser_lock

PODS = "https://console.runpod.io/pods"
DEPLOY = "https://console.runpod.io/deploy"
_SSH = re.compile(r"(?:ssh\s+)?root@([\d.]+)\s+-p\s+(\d+)", re.I)


def _page(p, cdp):
    browser = p.chromium.connect_over_cdp(f"http://localhost:{cdp}")
    ctx = browser.contexts[0] if browser.contexts else browser.new_context()
    return ctx.new_page()


def list_pods(cdp: int) -> str:
    from playwright.sync_api import sync_playwright
    with browser_lock(), sync_playwright() as p:
        page = _page(p, cdp)
        try:
            page.goto(PODS, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(6000)
            return page.inner_text("body")
        finally:
            page.close()


def terminate(cdp: int, pod_id: str) -> str:
    from playwright.sync_api import sync_playwright
    with browser_lock(), sync_playwright() as p:
        page = _page(p, cdp)
        try:
            page.goto(PODS, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(6000)
            # Карточка пода по id → кнопка ⋮ рядом → Terminate → подтвердить.
            row = page.get_by_text(pod_id, exact=False).first
            row.scroll_into_view_if_needed(timeout=5000)
            # три точки: ищем ближайшую кнопку-меню в той же карточке
            page.get_by_role("button", name=re.compile(r"more|menu|⋮|options|\.\.\.", re.I)).last.click(timeout=5000)
            page.wait_for_timeout(800)
            page.get_by_text(re.compile(r"terminate", re.I)).first.click(timeout=5000)
            page.wait_for_timeout(800)
            # подтверждение (кнопка Terminate/Yes/Confirm/Delete)
            page.get_by_role("button", name=re.compile(r"terminate|confirm|yes|delete|удалить", re.I)).last.click(timeout=5000)
            page.wait_for_timeout(3000)
            return f"terminate {pod_id}: отправлено (проверь /pods)"
        except Exception as e:
            return f"terminate {pod_id}: ошибка UI — {e}"
        finally:
            page.close()


# React слушает клики делегированно на корне → bubbling-событие с самого <p> триггерит onClick карточки.
_JS_CLICK = """e => {
  e.scrollIntoView({block:'center'});
  const opts = {bubbles:true, cancelable:true, view:window};
  e.dispatchEvent(new MouseEvent('pointerdown', opts));
  e.dispatchEvent(new MouseEvent('mousedown', opts));
  e.dispatchEvent(new MouseEvent('mouseup', opts));
  e.dispatchEvent(new MouseEvent('click', opts));
  return true;
}"""


def _click(page, loc):
    """Клик: обычный, иначе JS bubbling-событие (React ловит его на любом дочернем узле)."""
    try:
        loc.scroll_into_view_if_needed(timeout=3000)
        loc.click(timeout=3000)
        return
    except Exception:
        pass
    loc.first.evaluate(_JS_CLICK)


def create(cdp: int, gpu_name: str) -> str:
    # Повторная проверка доступности + цены ≤ cap по свежему списку (Chrome-парсер).
    cap = float(settings.get("max_pod_hourly_usd") or 0.30)
    try:
        avail = runpod_api.list_gpu_types()  # только доступные
    except Exception:
        avail = []
    match = next((g for g in avail if gpu_name.lower() in (g.get("displayName", "").lower())), None)
    if avail and not match:
        names = ", ".join(f"{g['displayName']}(${g['price']})" for g in avail[:6])
        return f"GPU '{gpu_name}' сейчас недоступна. Доступные ≤${cap}/ч: {names}"
    if match and match.get("price") and match["price"] > cap:
        return f"GPU '{gpu_name}' ${match['price']}/ч > лимита ${cap}. Возьми дешевле."

    from playwright.sync_api import sync_playwright
    with browser_lock(), sync_playwright() as p:
        page = _page(p, cdp)
        try:
            page.goto(DEPLOY, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(8000)
            # 1) выбрать наш Network Volume
            for op in (lambda: page.get_by_role("button", name=re.compile(r"network volume", re.I)).first.click(timeout=5000),
                       lambda: page.get_by_text(re.compile(r"network volume", re.I)).first.click(timeout=5000)):
                try:
                    op(); page.wait_for_timeout(1000)
                    _click(page, page.get_by_text(config.RUNPOD_VOLUME_NAME, exact=False).first)
                    page.wait_for_timeout(4000)
                    break
                except Exception:
                    continue
            # 2) выбрать GPU — точная кнопка-карточка по data-атрибуту (надёжно).
            gpu_btn = page.locator(f'button[data-ph-capture-attribute-gpu-card-selected*="{gpu_name}"]').first
            try:
                gpu_btn.scroll_into_view_if_needed(timeout=5000)
                gpu_btn.click(timeout=6000)
            except Exception:
                # фолбэк: любая gpu-card с текстом
                _click(page, page.locator('button[data-testid^="gpu-card-"]').filter(has_text=gpu_name).first)
            page.wait_for_timeout(1500)
            # 3) кнопка Deploy On-Demand
            _click(page, page.get_by_role("button", name=re.compile(r"deploy on-?demand", re.I)).first)
            page.wait_for_timeout(8000)
            # 4) забрать SSH из сайдбара Connect на /pods
            try:
                page.get_by_role("button", name=re.compile(r"connect", re.I)).first.click(timeout=6000)
                page.wait_for_timeout(2000)
            except Exception:
                pass
            m = _SSH.search(page.inner_text("body"))
            if m:
                return f"created: ssh root@{m.group(1)} -p {m.group(2)}"
            return "created: под создаётся, SSH пока не виден (проверь /pods → Connect)"
        except Exception as e:
            return f"create {gpu_name}: ошибка UI — {e}"
        finally:
            page.close()


if __name__ == "__main__":
    cdp = int(config.CLAUDE_CDP_PORT or 9222)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "terminate" and len(sys.argv) > 2:
        print(terminate(cdp, sys.argv[2]))
    elif cmd == "create" and len(sys.argv) > 2:
        print(create(cdp, " ".join(sys.argv[2:])))
    else:
        print(list_pods(cdp))
