from __future__ import annotations

from typing import Any, Dict

from fetcher.dataset_collector.config import CampaignConfig
from fetcher.dataset_collector.discovery.youtube import YouTubeKeyPool
from fetcher.dataset_collector.state import DatasetState
from fetcher.dataset_collector.inventory import compute_inventory_stats, load_summary
from fetcher.dataset_collector.stats import aggregate_shard_distributions


def build_status_report(
    config: CampaignConfig,
    state: DatasetState,
    *,
    key_pool: YouTubeKeyPool | None = None,
) -> Dict[str, Any]:
    manifest = state.load_manifest()
    run_accepted = int(manifest.counters.get("accepted", 0))
    baseline = int(manifest.baseline_accepted or config.baseline_accepted)
    checkpoint = state.load_checkpoint()
    key_stats = key_pool.quota_stats() if key_pool else {}

    categories = {}
    for category in config.categories:
        counters = manifest.category_counters.get(category.name, {})
        accepted = int(counters.get("accepted", 0))
        categories[category.name] = {
            "accepted": accepted,
            "rejected": int(counters.get("rejected", 0)),
            "target": category.target_count,
            "collect_count": category.collect_count,
            "complete": accepted >= category.collect_count,
        }

    return {
        "campaign": config.name,
        "output_dir": str(state.root),
        "total_with_baseline": baseline + run_accepted,
        "baseline_accepted": baseline,
        "legacy_seen_imported": int(manifest.legacy_seen_imported or 0),
        "run_accepted": run_accepted,
        "run_rejected": int(manifest.counters.get("rejected", 0)),
        "session": {
            "started_at": manifest.session_started_at,
            "accepted": int((manifest.session_counters or {}).get("accepted", 0)),
            "rejected": int((manifest.session_counters or {}).get("rejected", 0)),
        },
        "keys": key_stats,
        "checkpoint": checkpoint.dict() if checkpoint is not None else None,
        "categories": categories,
        "distributions": aggregate_shard_distributions(state.root),
        "inventory": load_summary(state) or compute_inventory_stats(state),
    }
