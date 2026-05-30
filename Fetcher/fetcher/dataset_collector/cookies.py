from __future__ import annotations

import http.cookiejar
import os
from itertools import cycle
from pathlib import Path
from typing import Iterator, Optional
from urllib import request

from fetcher.dataset_collector.schemas import CampaignConfig


class CookieRotator:
    def __init__(self, cookie_files: list[Path]) -> None:
        self.cookie_files = [path for path in cookie_files if path.is_file()]
        self._iterator: Iterator[Path] | None = cycle(self.cookie_files) if self.cookie_files else None

    @classmethod
    def from_config(cls, config: CampaignConfig) -> "CookieRotator":
        cookie_dir = config.cookie_files_dir or os.getenv("FETCHER_COOKIE_FILES_DIR")
        if not cookie_dir:
            return cls([])
        root = Path(cookie_dir)
        if not root.is_absolute():
            # Paths in campaign JSON are relative to Fetcher/ (cli cwd).
            fetcher_root = Path(__file__).resolve().parents[2]
            root = fetcher_root / root
        return cls(sorted(root.glob(config.cookie_file_glob or "*.txt")))

    @classmethod
    def from_directory(cls, directory: Path | str, *, glob_pattern: str = "*.txt") -> "CookieRotator":
        return cls(sorted(Path(directory).glob(glob_pattern)))

    def next(self) -> Optional[Path]:
        if self._iterator is None:
            return None
        return next(self._iterator)


def install_pytubefix_session(
    *,
    proxies: dict[str, str] | None = None,
    cookie_file: Path | None = None,
) -> None:
    """Configure urllib opener used by pytubefix (proxy + Netscape cookie file)."""
    handlers: list[object] = []
    if proxies:
        handlers.append(request.ProxyHandler(proxies))
    if cookie_file is not None and cookie_file.is_file():
        jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
        jar.load(ignore_discard=True, ignore_expires=True)
        handlers.append(request.HTTPCookieProcessor(jar))
    if handlers:
        request.install_opener(request.build_opener(*handlers))


def apply_cookiefile(ydl_opts: dict, rotator: CookieRotator | None) -> dict:
    if rotator is None:
        return ydl_opts
    cookie_file = rotator.next()
    if cookie_file is not None:
        ydl_opts["cookiefile"] = str(cookie_file)
    return ydl_opts
