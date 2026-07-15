#!/usr/bin/env python3
"""Extra-watchdog: следит за основным nightwatch.py и перезапускает его если упал.
Запускается каждые 10 минут, не конкурирует с основным watchdog.

Запуск: python extra_watchdog.py
"""
import subprocess
import time
from pathlib import Path
from datetime import datetime

RUNNER_DIR = Path(__file__).parent
LOG_FILE = Path("/tmp/extra_watchdog.log")

def log(msg: str):
    ts = datetime.now().isoformat(timespec='seconds')
    line = f"[{ts}] [EXTRA] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def is_nightwatch_running() -> bool:
    """Проверить, запущен ли nightwatch.py."""
    result = subprocess.run(
        ["pgrep", "-f", "nightwatch.py"],
        capture_output=True, text=True
    )
    return bool(result.stdout.strip())

def restart_nightwatch():
    """Перезапустить nightwatch.py."""
    try:
        log("🔴 Ночной watchdog не запущен! Перезапускаю...")
        result = subprocess.run(
            ["pkill", "-f", "nightwatch.py"],
            capture_output=True, timeout=5
        )
        time.sleep(1)

        subprocess.Popen(
            ["python", str(RUNNER_DIR / "nightwatch.py")],
            cwd=str(RUNNER_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        log("✅ Nightwatch перезапущен")
        return True
    except Exception as e:
        log(f"❌ Ошибка при перезапуске: {e}")
        return False

def main():
    log("🛡️  Extra-watchdog запущен (проверка каждые 10 минут)")

    while True:
        try:
            if not is_nightwatch_running():
                restart_nightwatch()
            else:
                log("✅ Nightwatch работает")
        except Exception as e:
            log(f"❌ Ошибка в цикле: {e}")

        time.sleep(10 * 60)  # Проверяем каждые 10 минут

if __name__ == "__main__":
    main()
