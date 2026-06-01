from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from fetcher.dataset_collector.config import CampaignConfig
    from fetcher.dataset_collector.discovery.youtube import YouTubeKeyPool
    from fetcher.dataset_collector.state import DatasetState


class ProgressReporter:
    """Compact progress lines for long discovery runs."""

    def __init__(
        self,
        config: CampaignConfig,
        state: DatasetState,
        *,
        key_pool: Optional[YouTubeKeyPool] = None,
    ) -> None:
        self.config = config
        self.state = state
        self.key_pool = key_pool
        self.session_accepted = 0
        self.session_rejected = 0
        self._session_quota_baseline = self._total_quota_used()
        self._session_started_monotonic = time.monotonic()

    def record_accept(self) -> None:
        self.session_accepted += 1

    def record_reject(self) -> None:
        self.session_rejected += 1

    def _total_quota_used(self) -> int:
        if self.key_pool is None:
            return 0
        return self.key_pool.quota_stats()["quota_used_total"]

    def session_quota_used(self) -> int:
        return max(0, self._total_quota_used() - self._session_quota_baseline)

    def snapshot(self) -> Dict[str, Any]:
        manifest = self.state.load_manifest()
        run_accepted = self.state.live_run_accepted()
        baseline = int(getattr(manifest, "baseline_accepted", 0) or self.config.baseline_accepted)
        key_stats = self.key_pool.quota_stats() if self.key_pool else {}
        session_from_manifest = int((manifest.session_counters or {}).get("accepted", 0))
        session_accepted = max(self.session_accepted, session_from_manifest)
        return {
            "total_with_baseline": baseline + run_accepted,
            "baseline_accepted": baseline,
            "run_accepted": run_accepted,
            "session_accepted": session_accepted,
            "session_rejected": self.session_rejected,
            "session_quota_units": self.session_quota_used(),
            "session_elapsed_seconds": time.monotonic() - self._session_started_monotonic,
            "keys_available": key_stats.get("keys_available", 0),
            "keys_total": key_stats.get("keys_total", 0),
        }

    @staticmethod
    def _short_keyword(keyword: str, *, max_len: int = 48) -> str:
        text = keyword.strip() or "(empty)"
        if len(text) <= max_len:
            return text
        return f"{text[: max_len - 1]}…"

    def log_keyword_start(
        self,
        *,
        category: str,
        category_accepted: int,
        category_target: int,
        keyword_index: int,
        keywords_total: int,
        keyword: str,
        min_unique: int,
    ) -> None:
        stats = self.snapshot()
        kw_label = self._short_keyword(keyword)
        print(
            f"[{self.config.name}] {category} {category_accepted:,}/{category_target:,}"
            f" | total {stats['total_with_baseline']:,}"
            f" | session +{stats['session_accepted']:,}"
            f" | quota {stats['session_quota_units']:,}"
            f" | keys {stats['keys_available']}/{stats['keys_total']}"
            f"\n  → kw {keyword_index + 1}/{keywords_total} \"{kw_label}\""
            f" (target ≥{min_unique} unique)",
            flush=True,
        )

    def log_keyword_skip(
        self,
        *,
        category: str,
        keyword_index: int,
        keywords_total: int,
        keyword: str,
    ) -> None:
        kw_label = self._short_keyword(keyword)
        print(
            f"⊘ kw {keyword_index + 1}/{keywords_total} \"{kw_label}\""
            f" — already done (see state/keyword_progress.jsonl)",
            flush=True,
        )

    def log_keyword_done(
        self,
        *,
        category: str,
        category_accepted: int,
        category_target: int,
        keyword_index: int,
        keywords_total: int,
        keyword: str,
        keyword_accepted: int,
        keyword_min: int,
        keyword_scanned: int,
        keyword_dup: int,
        keyword_rejected: int,
        warn: bool = False,
    ) -> None:
        stats = self.snapshot()
        kw_label = self._short_keyword(keyword)
        status = "OK" if keyword_accepted >= keyword_min else "LOW"
        prefix = "⚠ " if warn or keyword_accepted < keyword_min else "✓ "
        print(
            f"{prefix}kw {keyword_index + 1}/{keywords_total} \"{kw_label}\""
            f" | +{keyword_accepted}/{keyword_min} unique"
            f" | scanned {keyword_scanned} dup {keyword_dup} rejected {keyword_rejected}"
            f" | {category} {category_accepted:,}/{category_target:,}"
            f" | session +{stats['session_accepted']:,} quota {stats['session_quota_units']:,}",
            flush=True,
        )

    def log(
        self,
        *,
        category: str,
        category_accepted: int,
        category_target: int,
        keyword_index: int,
        keywords_total: int,
        keyword: str,
        keyword_accepted: int = 0,
        keyword_min: int = 0,
        force: bool = False,
        every_n: int = 25,
    ) -> None:
        if not force and self.session_accepted and self.session_accepted % every_n != 0:
            return
        stats = self.snapshot()
        kw_label = self._short_keyword(keyword)
        kw_part = ""
        if keyword_min:
            kw_part = f" | kw +{keyword_accepted}/{keyword_min} \"{kw_label}\""
        print(
            f"[{self.config.name}] {category} {category_accepted:,}/{category_target:,}"
            f" | session +{stats['session_accepted']:,}"
            f" | quota {stats['session_quota_units']:,}"
            f"{kw_part}",
            flush=True,
        )
