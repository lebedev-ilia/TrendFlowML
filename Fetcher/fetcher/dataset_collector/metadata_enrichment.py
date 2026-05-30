from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yt_dlp
from yt_dlp.utils import DownloadError

from fetcher.dataset_collector.cookies import CookieRotator, apply_cookiefile
from fetcher.dataset_collector.config import merged_filters
from fetcher.dataset_collector.filters import VideoFilter
from fetcher.dataset_collector.proxy import ProxyRotator, configured_proxies
from fetcher.dataset_collector.schemas import CampaignConfig
from fetcher.dataset_collector.state import DatasetState, atomic_write_json, iter_jsonl
from fetcher.dataset_collector.training_format import (
    compact_training_metadata,
    merge_ytdlp_into_training_metadata,
    metadata_captions_are_bloated,
    training_entry_needs_ytdlp_enrichment,
)
from fetcher.dataset_collector.worker_logging import (
    count_jsonl_lines,
    log_kv_block,
    log_pass_footer,
    log_pass_header,
    worker_log,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _enriched_marker() -> dict[str, str]:
    return {
        "at": _utcnow().isoformat().replace("+00:00", "Z"),
        "source": "yt_dlp",
    }


def iter_metadata_enrich_queue(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def fetch_ytdlp_info(
    url: str,
    *,
    cookie_rotator: CookieRotator | None,
    proxy_rotator: ProxyRotator | None,
) -> dict[str, Any] | None:
    proxy = proxy_rotator.next() if proxy_rotator else None
    ydl_opts: dict[str, Any] = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "ignore_no_formats_error": True,
    }
    if proxy:
        ydl_opts["proxy"] = proxy
    apply_cookiefile(ydl_opts, cookie_rotator)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if proxy_rotator and proxy:
            proxy_rotator.record_success(proxy)
        if not info:
            return None
        return info
    except DownloadError as exc:
        if proxy_rotator and proxy:
            proxy_rotator.record_failure(proxy, exc)
        print(f"[enrich] yt-dlp failed {url}: {exc}", file=sys.stderr, flush=True)
        return None
    except Exception as exc:
        if proxy_rotator and proxy:
            proxy_rotator.record_failure(proxy, exc)
        print(f"[enrich] yt-dlp error {url}: {exc}", file=sys.stderr, flush=True)
        return None


def patch_training_entry(entry: dict, info: dict) -> None:
    metadata = entry.get("metadata") or {}
    entry["metadata"] = merge_ytdlp_into_training_metadata(metadata, info)
    entry["_enriched"] = _enriched_marker()


def _resolve_category_filters(config: CampaignConfig, category: str) -> dict:
    for cat in config.categories:
        if cat.name == category:
            return merged_filters(config, cat)
    return dict(config.default_filters or {})


def _reject_enriched_video(
    state: DatasetState,
    *,
    data: dict,
    shard_path: Path,
    video_id: str,
    category: str,
    entry: dict,
    reason: str,
) -> None:
    data.pop(video_id, None)
    atomic_write_json(shard_path, data)
    state.record_post_enrich_rejection(
        platform="youtube",
        video_id=video_id,
        category=category,
        query=str(entry.get("query") or ""),
        reason=reason,
        record=entry,
    )
    worker_log("enrich", f"REJECT {video_id}: {reason} (removed from shard)")


def compact_metadata_shards(
    state: DatasetState,
    *,
    category: str | None = None,
) -> dict[str, int]:
    """Rewrite shard metadata to drop bloated caption URLs (ru/en ext list only)."""
    metadata_root = state.shards_dir / "metadata"
    if not metadata_root.exists():
        return {"shards": 0, "entries": 0, "compacted": 0}

    pattern = f"category={category}/part_*.json" if category else "**/part_*.json"
    stats = {"shards": 0, "entries": 0, "compacted": 0}
    for shard_path in sorted(metadata_root.glob(pattern)):
        if shard_path.name.endswith(".tmp"):
            continue
        data = json.loads(shard_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        stats["shards"] += 1
        changed = False
        for video_id, entry in data.items():
            stats["entries"] += 1
            metadata = entry.get("metadata") or {}
            if not metadata_captions_are_bloated(metadata):
                continue
            entry["metadata"] = compact_training_metadata(metadata)
            changed = True
            stats["compacted"] += 1
        if changed:
            atomic_write_json(shard_path, data)
            worker_log("enrich", f"compacted captions in {shard_path.relative_to(state.root)}")
    return stats


def enrich_shard_video(
    state: DatasetState,
    config: CampaignConfig,
    *,
    shard_relpath: str,
    video_id: str,
    url: str,
    category: str,
    cookie_rotator: CookieRotator | None,
    proxy_rotator: ProxyRotator | None,
) -> str:
    """Return 'ok', 'rejected', or 'failed'."""
    shard_path = state.root / shard_relpath
    if not shard_path.exists():
        worker_log("enrich", f"shard missing: {shard_relpath} (discover may not have flushed yet)")
        return "failed"

    data = json.loads(shard_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        worker_log("enrich", f"unsupported shard format: {shard_relpath}")
        return "failed"

    entry = data.get(video_id)
    if entry is None:
        worker_log("enrich", f"video {video_id} not in {shard_relpath}")
        return "failed"

    if not training_entry_needs_ytdlp_enrichment(entry):
        metadata = entry.get("metadata") or {}
        if metadata_captions_are_bloated(metadata):
            entry["metadata"] = compact_training_metadata(metadata)
            atomic_write_json(shard_path, data)
            worker_log("enrich", f"compact {video_id} captions in {shard_relpath}")
        else:
            worker_log("enrich", f"skip {video_id}: metadata already has yt-dlp fields")
        return "ok"

    worker_log("enrich", f"yt-dlp fetch {video_id} …")
    info = fetch_ytdlp_info(url, cookie_rotator=cookie_rotator, proxy_rotator=proxy_rotator)
    if info is None:
        worker_log("enrich", f"FAIL {video_id}: yt-dlp returned no info")
        return "failed"

    rules = _resolve_category_filters(config, category)
    decision = VideoFilter(rules).decide_post_enrich(
        info=info,
        metadata=entry.get("metadata") or {},
    )
    if not decision.accepted:
        _reject_enriched_video(
            state,
            data=data,
            shard_path=shard_path,
            video_id=video_id,
            category=category,
            entry=entry,
            reason=decision.reason or "post_enrich_rejected",
        )
        return "rejected"

    patch_training_entry(entry, info)
    atomic_write_json(shard_path, data)
    worker_log("enrich", f"OK {video_id} patched in {shard_relpath}")
    return "ok"


def scan_shards_for_enrichment(
    state: DatasetState,
    *,
    category: str | None = None,
    done_keys: set[str] | None = None,
) -> int:
    """Enqueue videos from existing metadata shards that still need yt-dlp fields."""
    done_keys = done_keys or state.load_metadata_enrich_done()
    queued = 0
    seen: set[str] = set()
    metadata_root = state.shards_dir / "metadata"
    if not metadata_root.exists():
        return 0

    pattern = f"category={category}/part_*.json" if category else "**/part_*.json"
    for shard_path in sorted(metadata_root.glob(pattern)):
        if shard_path.name.endswith(".tmp"):
            continue
        data = json.loads(shard_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        shard_relpath = str(shard_path.relative_to(state.root))
        category_name = category or shard_path.parent.name.replace("category=", "")
        for video_id, entry in data.items():
            if not training_entry_needs_ytdlp_enrichment(entry):
                continue
            key = f"youtube:{video_id}"
            if key in done_keys or key in seen:
                continue
            seen.add(key)
            state.enqueue_metadata_enrichment(
                platform="youtube",
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                category=category_name,
                shard_relpath=shard_relpath,
            )
            queued += 1
    return queued


def run_metadata_enrich_queue(
    state: DatasetState,
    config: CampaignConfig,
    *,
    category: str | None = None,
    limit: int | None = None,
    cookie_rotator: CookieRotator | None = None,
) -> dict[str, int]:
    cookie_rotator = cookie_rotator or CookieRotator.from_config(config)
    enrich_proxies = configured_proxies(config=config)
    if enrich_proxies:
        worker_log("enrich", f"proxies: {', '.join(enrich_proxies)}")
    proxy_rotator = ProxyRotator(proxies=enrich_proxies, config=config)
    log_pass_header("enrich", "pass start")

    done_keys = state.load_metadata_enrich_done()
    queue_path = state.metadata_enrich_queue_path
    queue_lines = count_jsonl_lines(queue_path)

    log_kv_block(
        "enrich",
        [
            ("queue_file", queue_path),
            ("queue_lines", queue_lines),
            ("already_done", len(done_keys)),
            ("category_filter", category or "all"),
            ("limit_this_pass", limit if limit is not None else "none"),
        ],
    )

    if queue_lines == 0:
        worker_log("enrich", "queue empty — wait until discover writes a metadata shard")
        result = {"enriched": 0, "failed": 0, "skipped": 0, "rejected": 0, "attempted": 0}
        log_pass_footer("enrich", result)
        return result

    results = {"enriched": 0, "failed": 0, "skipped": 0, "rejected": 0}
    skip_reasons: dict[str, int] = {}
    attempted = 0

    for item in iter_metadata_enrich_queue(queue_path):
        if limit is not None and results["enriched"] + results["failed"] + results["rejected"] >= limit:
            worker_log("enrich", f"limit reached ({limit}), stopping this pass")
            break
        platform = item.get("platform") or "youtube"
        if platform != "youtube":
            results["skipped"] += 1
            skip_reasons["unsupported_platform"] = skip_reasons.get("unsupported_platform", 0) + 1
            continue
        if category and item.get("category") != category:
            results["skipped"] += 1
            skip_reasons["other_category"] = skip_reasons.get("other_category", 0) + 1
            continue

        video_id = item.get("video_id")
        url = item.get("url")
        shard_relpath = item.get("shard")
        if not video_id or not url or not shard_relpath:
            results["skipped"] += 1
            skip_reasons["invalid_row"] = skip_reasons.get("invalid_row", 0) + 1
            continue

        key = f"{platform}:{video_id}"
        if key in done_keys:
            results["skipped"] += 1
            skip_reasons["already_done"] = skip_reasons.get("already_done", 0) + 1
            continue

        attempted += 1
        worker_log(
            "enrich",
            f"({attempted}) {video_id} shard={shard_relpath} "
            f"(pass: +{results['enriched']} ok, {results['failed']} fail)",
        )

        ok = enrich_shard_video(
            state,
            config,
            shard_relpath=shard_relpath,
            video_id=video_id,
            url=url,
            category=item.get("category") or category or "unknown",
            cookie_rotator=cookie_rotator,
            proxy_rotator=proxy_rotator,
        )
        if ok == "ok":
            state.mark_metadata_enrich_done(key, video_id=video_id, shard=shard_relpath)
            done_keys.add(key)
            results["enriched"] += 1
        elif ok == "rejected":
            state.mark_metadata_enrich_done(key, video_id=video_id, shard=shard_relpath)
            done_keys.add(key)
            results["rejected"] += 1
        else:
            results["failed"] += 1

    if skip_reasons:
        log_kv_block("enrich", [(f"skipped_{k}", v) for k, v in sorted(skip_reasons.items())])
    if attempted == 0 and results["skipped"] > 0:
        worker_log("enrich", "no new items processed — all queue rows already done or filtered")

    from fetcher.dataset_collector.inventory import refresh_summary

    refresh_summary(state)
    results["attempted"] = attempted
    log_pass_footer("enrich", results)
    return results
