from __future__ import annotations

import os
from itertools import cycle
from pathlib import Path
from typing import Iterator, Optional

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
        return cls(sorted(root.glob(config.cookie_file_glob or "*.txt")))

    def next(self) -> Optional[Path]:
        if self._iterator is None:
            return None
        return next(self._iterator)


def apply_cookiefile(ydl_opts: dict, rotator: CookieRotator | None) -> dict:
    if rotator is None:
        return ydl_opts
    cookie_file = rotator.next()
    if cookie_file is not None:
        ydl_opts["cookiefile"] = str(cookie_file)
    return ydl_opts
