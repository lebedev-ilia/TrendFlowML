from __future__ import annotations

import os
from itertools import cycle
from pathlib import Path
from typing import Iterator, Optional

from fetcher.config import settings
from fetcher.dataset_collector.schemas import CampaignConfig


def normalize_proxy_url(value: str, *, default_scheme: str = "http") -> str:
    proxy = value.strip()
    if not proxy:
        return ""
    if "://" not in proxy:
        proxy = f"{default_scheme}://{proxy}"
    return proxy


def is_local_proxy(value: str) -> bool:
    lowered = value.lower()
    return "://127." in lowered or "://localhost" in lowered or "://0.0.0.0" in lowered


def load_proxy_file(
    path: str | Path,
    *,
    default_scheme: str = "http",
    include_local: bool = False,
) -> list[str]:
    proxy_path = Path(path)
    if not proxy_path.exists():
        return []
    proxies: list[str] = []
    for raw_line in proxy_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        proxy = normalize_proxy_url(parts[0], default_scheme=default_scheme)
        role_tokens = {part.lower() for part in parts[1:]}
        if "download_only" in role_tokens or "download-only" in role_tokens:
            if not include_local:
                continue
        if is_local_proxy(proxy) and not include_local:
            continue
        if proxy:
            proxies.append(proxy)
    return list(dict.fromkeys(proxies))


def configured_proxies(
    *,
    config: CampaignConfig | None = None,
    include_local: bool = False,
) -> list[str]:
    proxies = []
    if config is not None and config.proxies_file:
        proxies.extend(
            load_proxy_file(
                config.proxies_file,
                default_scheme=config.proxy_default_scheme,
                include_local=include_local or config.include_local_proxies_for_discovery,
            )
        )
    if getattr(settings, "enable_proxies", False):
        raw = getattr(settings, "proxies", [])
        if isinstance(raw, str):
            proxies.extend([normalize_proxy_url(item.strip()) for item in raw.split(",") if item.strip()])
        else:
            proxies.extend([normalize_proxy_url(str(item).strip()) for item in raw if str(item).strip()])
    env_proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if env_proxy:
        proxies.append(normalize_proxy_url(env_proxy))
    if not include_local:
        proxies = [proxy for proxy in proxies if not is_local_proxy(proxy)]
    return list(dict.fromkeys(proxies))


class ProxyRotator:
    def __init__(
        self,
        proxies: list[str] | None = None,
        *,
        config: CampaignConfig | None = None,
        include_local: bool = False,
    ) -> None:
        self.proxies = proxies if proxies is not None else configured_proxies(
            config=config,
            include_local=include_local,
        )
        self._iterator: Iterator[str] | None = cycle(self.proxies) if self.proxies else None

    def next(self) -> Optional[str]:
        if self._iterator is None:
            return None
        return next(self._iterator)
