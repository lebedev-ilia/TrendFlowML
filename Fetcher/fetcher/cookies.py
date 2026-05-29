from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Iterator, Optional

from fetcher.config import settings


class CookieFileRotator:
    def __init__(self, cookie_files: list[Path]) -> None:
        self.cookie_files = [path for path in cookie_files if path.is_file()]
        self._iterator: Iterator[Path] | None = cycle(self.cookie_files) if self.cookie_files else None

    @classmethod
    def from_settings(cls) -> "CookieFileRotator":
        if not settings.cookie_files_dir:
            return cls([])
        root = Path(settings.cookie_files_dir)
        return cls(sorted(root.glob(settings.cookie_file_glob or "*.txt")))

    def next(self) -> Optional[Path]:
        if self._iterator is None:
            return None
        return next(self._iterator)


_rotator: CookieFileRotator | None = None


def get_next_cookie_file() -> Optional[str]:
    global _rotator
    if _rotator is None:
        _rotator = CookieFileRotator.from_settings()
    cookie_file = _rotator.next()
    return str(cookie_file) if cookie_file else None


def apply_cookiefile(ydl_opts: dict) -> dict:
    cookie_file = get_next_cookie_file()
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
    return ydl_opts
