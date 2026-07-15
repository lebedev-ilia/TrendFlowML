#!/usr/bin/env python3
"""Фоновый опрос лимитов claude.ai каждые N секунд (settings.limits_poll_sec, по умолч. 120).

Обновляет общий snapshot, чтобы ВСЕ агенты видели свежие проценты и не упирались в лимит.
Запуск: python limits_daemon.py   (нужен залогиненный Chrome с CDP — см. CLAUDE_CDP_PORT в .env)
"""
from __future__ import annotations
import subprocess
import sys
import time
import logging

import config
import settings
import limits

SCRAPER = str(config.RUNNER_DIR / "claude_limits_scraper.py")
LOG_FILE = config.STATE_DIR / "limits_daemon.log"

# Логирование в файл
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("limits_daemon")


def scan_once():
    """Попытка обновить данные через CDP (если Chrome запущен с --remote-debugging-port).
    Если CDP недоступен — gracefully используем старые данные (не зависаем на headless)."""
    if not config.CLAUDE_CDP_PORT:
        log.warning(f"⚠️ CLAUDE_CDP_PORT не задан в .env. Добавь строку: CLAUDE_CDP_PORT=9222")
        return False

    cmd = [sys.executable, SCRAPER, "--cdp", config.CLAUDE_CDP_PORT]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            log.info("✓ Данные обновлены из Chrome CDP")
            return True
        else:
            log.warning(f"Скрейпер CDP вернул ошибку (код {result.returncode}). "
                       f"Chrome может не быть запущен на порту {config.CLAUDE_CDP_PORT}")
            if result.stderr:
                log.debug(f"stderr: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"❌ Chrome не отвечает на порту {config.CLAUDE_CDP_PORT} (timeout 60s). "
                 f"Запусти Chrome: google-chrome --remote-debugging-port={config.CLAUDE_CDP_PORT}")
        return False
    except Exception as e:
        log.error(f"❌ Ошибка подключения к Chrome: {e}")
        return False


def main():
    log.info("========== СТАРТ limits_daemon ==========")
    log.info(f"Режим: {'CDP' if config.CLAUDE_CDP_PORT else 'headless'}")
    consecutive_fails = 0

    while True:
        try:
            success = scan_once()
            b = limits.browser()
            age = limits.browser_age_min()

            if success:
                consecutive_fails = 0
            else:
                consecutive_fails += 1

            stale = f" (устарело {age:.0f}м назад)" if age and age > 5 else ""
            log.info(f"5ч={b.get('five_hour_pct')}% неделя={b.get('weekly_pct')}% "
                    f"кредиты={b.get('credits_pct')}%{stale} [fails={consecutive_fails}]")

            if consecutive_fails >= 3:
                log.warning(f"⚠️ 3 подряд ошибки парсинга! Проверь Chrome с CDP или логин в claude.ai")

        except Exception as e:
            log.error(f"Критическая ошибка в main: {e}")

        time.sleep(int(settings.get("limits_poll_sec") or 120))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[limits_daemon] стоп")
