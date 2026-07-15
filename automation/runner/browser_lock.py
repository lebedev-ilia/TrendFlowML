"""Единый лок на Chrome (CDP): только ОДИН скрапер работает с браузером за раз.
Решает конфликт, когда лимиты и RunPod-парсер лезут в один браузер и «съедают» вкладки друг друга.

Использование:
    from browser_lock import browser_lock
    with browser_lock():
        ...playwright CDP...
"""
from __future__ import annotations
import fcntl
import time
from contextlib import contextmanager

import config

LOCK_FILE = config.STATE_DIR / "browser.lock"


@contextmanager
def browser_lock(timeout: int = 120):
    """Эксклюзивный лок. Ждёт до timeout сек, пока другой скрапер освободит браузер."""
    f = open(LOCK_FILE, "w")
    start = time.time()
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.time() - start > timeout:
                f.close()
                raise TimeoutError("browser занят другим скрапером >timeout")
            time.sleep(2)
    try:
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()
