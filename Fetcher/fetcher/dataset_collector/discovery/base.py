from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Protocol

from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot


@dataclass(frozen=True)
class DiscoveryCapabilities:
    search: bool
    metadata: bool
    snapshots: bool
    comments: bool
    downloads: bool


class DiscoveryAdapter(Protocol):
    platform: str
    capabilities: DiscoveryCapabilities

    def discover(
        self,
        *,
        category: str,
        query: str,
        limit: int,
        published_after: Optional[datetime] = None,
        published_before: Optional[datetime] = None,
        time_interval: Optional[str] = None,
    ) -> Iterable[CollectedVideo]:
        ...

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int) -> Snapshot:
        ...
