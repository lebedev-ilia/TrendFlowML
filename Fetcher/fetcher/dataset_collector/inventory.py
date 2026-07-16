from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set

from fetcher.dataset_collector.state import DatasetState, atomic_write_json, iter_jsonl


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def inventory_dir(state: DatasetState) -> Path:
    return state.state_dir / "inventory"


def shards_index_path(state: DatasetState) -> Path:
    return inventory_dir(state) / "shards.jsonl"


def videos_index_path(state: DatasetState) -> Path:
    return inventory_dir(state) / "videos.jsonl"


def summary_path(state: DatasetState) -> Path:
    return inventory_dir(state) / "summary.json"


def shard_video_ids(shard_path: Path) -> List[str]:
    if not shard_path.is_file():
        return []
    try:
        data = json.loads(shard_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict):
        return [str(k) for k in data.keys()]
    if isinstance(data, list):
        return [str(row.get("video_id")) for row in data if row.get("video_id")]
    return []


def register_shard(
    state: DatasetState,
    *,
    shard_relpath: str,
    category: str,
    video_ids: Iterable[str],
    platform: str = "youtube",
) -> None:
    """Append shard record with video IDs (metadata inventory)."""
    ids = list(video_ids)
    inventory_dir(state).mkdir(parents=True, exist_ok=True)
    record = {
        "shard": shard_relpath,
        "category": category,
        "platform": platform,
        "video_ids": ids,
        "count": len(ids),
        "registered_at": utcnow().isoformat(),
    }
    with shards_index_path(state).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")
    for video_id in ids:
        register_video_in_shard(
            state,
            video_id=video_id,
            category=category,
            shard_relpath=shard_relpath,
            platform=platform,
        )
    refresh_summary(state)


def register_video_in_shard(
    state: DatasetState,
    *,
    video_id: str,
    category: str,
    shard_relpath: str,
    platform: str = "youtube",
) -> None:
    inventory_dir(state).mkdir(parents=True, exist_ok=True)
    record = {
        "video_id": video_id,
        "category": category,
        "platform": platform,
        "shard": shard_relpath,
        "in_shard": True,
        "registered_at": utcnow().isoformat(),
    }
    with videos_index_path(state).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


def _video_key(platform: str, category: str, video_id: str) -> str:
    return f"{platform}:{category}:{video_id}"


def _shard_key(shard_relpath: str) -> str:
    return f"shard:{shard_relpath}"


def _iter_queue(path: Path) -> Iterator[dict]:
    """encoding=utf-8-sig + skip-on-JSONDecodeError: см. state.py::iter_jsonl (баг 2026-07-16) —
    одна битая строка (BOM/торн-запись) не должна ронять весь проход воркера."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _unique_queue_keys(
    queue_path: Path,
    *,
    key_fn,
    done_keys: Set[str],
) -> Set[str]:
    pending: Set[str] = set()
    for item in _iter_queue(queue_path):
        key = key_fn(item)
        if not key or key in done_keys:
            continue
        pending.add(key)
    return pending


def _count_local_downloads(state: DatasetState, *, category: str | None = None) -> int:
    videos_root = state.download_dir / "videos"
    if not videos_root.exists():
        return 0
    count = 0
    if category:
        roots = [videos_root / category]
    else:
        roots = [p for p in videos_root.iterdir() if p.is_dir()]
    for root in roots:
        if not root.is_dir():
            continue
        count += sum(1 for p in root.glob("*.mp4") if p.stat().st_size > 0)
    return count


def _load_shard_index(state: DatasetState) -> List[dict]:
    path = shards_index_path(state)
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _load_videos_in_shards(state: DatasetState) -> Set[str]:
    keys: Set[str] = set()
    for row in iter_jsonl(videos_index_path(state)):
        vid = row.get("video_id")
        cat = row.get("category")
        plat = row.get("platform") or "youtube"
        if vid and cat:
            keys.add(_video_key(plat, cat, vid))
    return keys


def _metadata_records_for_category(state: DatasetState, *, category: str | None = None) -> Iterator[dict]:
    metadata_root = state.shards_dir / "metadata"
    if not metadata_root.exists():
        return
    pattern = f"category={category}/part_*.json" if category else "**/part_*.json"
    for shard_path in sorted(metadata_root.glob(pattern)):
        if shard_path.name.endswith(".tmp"):
            continue
        try:
            data = json.loads(shard_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = data.values() if isinstance(data, dict) else data
        if not isinstance(rows, Iterable):
            continue
        for row in rows:
            if isinstance(row, dict):
                yield row


def _channel_stats(state: DatasetState, *, category: str | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in _metadata_records_for_category(state, category=category):
        metadata = row.get("metadata") or {}
        channel_id = (
            metadata.get("channel_id")
            or metadata.get("channelId")
            or row.get("channel_id")
            or metadata.get("channelTitle")
            or "unknown"
        )
        counts[str(channel_id)] += 1
    total = sum(counts.values())
    top_count = counts.most_common(1)[0][1] if counts else 0
    return {
        "unique": len(counts),
        "top_count": top_count,
        "top_share": (top_count / total) if total else 0.0,
        "average_videos_per_channel": (total / len(counts)) if counts else 0.0,
    }


def _snapshot_readiness(state: DatasetState, *, category: str | None = None) -> dict[str, Any]:
    scheduled: dict[str, str] = {}
    for row in iter_jsonl(state.schedule_path):
        key = f"{row.get('platform') or 'youtube'}:{row.get('video_id')}"
        if not row.get("video_id"):
            continue
        scheduled[key] = row.get("category") or "unknown"
    if category:
        scheduled = {key: cat for key, cat in scheduled.items() if cat == category}

    completed: dict[str, set[int]] = defaultdict(set)
    for row in iter_jsonl(state.snapshot_completion_path):
        key = str(row.get("key") or "")
        if not key or key not in scheduled:
            continue
        try:
            index = int(row.get("snapshot_index"))
        except (TypeError, ValueError):
            continue
        completed[key].add(index)

    by_index = {
        str(index): sum(1 for indexes in completed.values() if index in indexes)
        for index in (1, 2, 3, 4)
    }
    ready_14_21 = sum(1 for indexes in completed.values() if 2 in indexes and 3 in indexes)
    ready_7_14_21 = sum(
        1
        for indexes in completed.values()
        if 1 in indexes and 2 in indexes and 3 in indexes
    )
    return {
        "scheduled": len(scheduled),
        "snapshot_7d": by_index["1"],
        "snapshot_14d": by_index["2"],
        "snapshot_21d": by_index["3"],
        "snapshot_28d": by_index["4"],
        "training_ready_14_21": ready_14_21,
        "training_ready_7_14_21": ready_7_14_21,
    }


def rebuild_inventory_from_disk(
    state: DatasetState,
    *,
    category: str | None = None,
) -> dict[str, Any]:
    """Rescan metadata shards and rebuild shard/video index files."""
    inventory_dir(state).mkdir(parents=True, exist_ok=True)
    shards_out = shards_index_path(state)
    videos_out = videos_index_path(state)
    if shards_out.exists():
        shards_out.unlink()
    if videos_out.exists():
        videos_out.unlink()

    metadata_root = state.shards_dir / "metadata"
    shards_written = 0
    videos_written = 0
    if metadata_root.exists():
        pattern = f"category={category}/part_*.json" if category else "**/part_*.json"
        for shard_path in sorted(metadata_root.glob(pattern)):
            if shard_path.name.endswith(".tmp"):
                continue
            shard_relpath = str(shard_path.relative_to(state.root))
            category_name = category or shard_path.parent.name.replace("category=", "")
            video_ids = shard_video_ids(shard_path)
            register_shard(
                state,
                shard_relpath=shard_relpath,
                category=category_name,
                video_ids=video_ids,
            )
            shards_written += 1
            videos_written += len(video_ids)

    summary = refresh_summary(state)
    summary["rebuilt"] = True
    summary["shards_scanned"] = shards_written
    summary["videos_indexed"] = videos_written
    return summary


def compute_inventory_stats(
    state: DatasetState,
    *,
    category: str | None = None,
) -> Dict[str, Any]:
    download_done = state.load_download_done()
    hf_video_done = state.load_hf_video_upload_done()
    hf_shard_done = state.load_hf_shard_upload_done()
    enrich_done = state.load_metadata_enrich_done()
    for row in iter_jsonl(state.metadata_enrich_done_path):
        platform = row.get("platform") or "youtube"
        shard = str(row.get("shard") or "")
        category_name = row.get("category") or "unknown"
        for part in Path(shard).parts:
            if part.startswith("category="):
                category_name = part.split("=", 1)[1] or category_name
                break
        video_id = row.get("video_id")
        if video_id:
            enrich_done.add(_video_key(platform, category_name, str(video_id)))

    download_pending_keys = _unique_queue_keys(
        state.download_dir / "queue.jsonl",
        key_fn=lambda item: _video_key(
            item.get("platform") or "youtube",
            item.get("category") or "unknown",
            str(item.get("video_id") or ""),
        )
        if item.get("video_id")
        else "",
        done_keys=download_done,
    )

    hf_video_pending_keys = _unique_queue_keys(
        state.hf_video_upload_queue_path,
        key_fn=lambda item: _video_key(
            item.get("platform") or "youtube",
            item.get("category") or "unknown",
            str(item.get("video_id") or ""),
        )
        if item.get("video_id")
        else "",
        done_keys=hf_video_done,
    )

    hf_shard_pending_keys = _unique_queue_keys(
        state.hf_shard_upload_queue_path,
        key_fn=lambda item: _shard_key(str(item.get("shard") or ""))
        if item.get("shard")
        else "",
        done_keys=hf_shard_done,
    )

    def _enrich_key(item: dict) -> str:
        platform = item.get("platform") or "youtube"
        vid = item.get("video_id")
        cat = item.get("category") or "unknown"
        if vid:
            return _video_key(platform, cat, vid)
        return str(item.get("key") or "")

    enrich_pending_keys = _unique_queue_keys(
        state.metadata_enrich_queue_path,
        key_fn=_enrich_key,
        done_keys=enrich_done,
    )

    shard_rows = _load_shard_index(state)
    if category:
        shard_rows = [r for r in shard_rows if r.get("category") == category]

    shards_total = len(shard_rows)
    shards_on_hf = len(hf_shard_done)
    if category:
        shards_on_hf = sum(
            1
            for key in hf_shard_done
            if f"category={category}" in key
        )

    videos_in_shards = _load_videos_in_shards(state)
    if category:
        videos_in_shards = {k for k in videos_in_shards if f":{category}:" in k}

    videos_downloaded = len(download_done)
    videos_on_hf = len(hf_video_done)
    enrich_on_hf = len(state.load_hf_enrich_upload_done())
    enrich_root = state.shards_dir / "enrich"
    videos_enriched = 0
    if enrich_root.exists():
        enrich_pattern = f"category={category}/part_*.json" if category else "**/part_*.json"
        for enrich_path in enrich_root.glob(enrich_pattern):
            try:
                data = json.loads(enrich_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                videos_enriched += len(data)
    local_mp4 = _count_local_downloads(state, category=category)
    channel_stats = _channel_stats(state, category=category)
    snapshot_readiness = _snapshot_readiness(state, category=category)

    if category:
        videos_downloaded = sum(1 for k in download_done if f":{category}:" in k)
        videos_on_hf = sum(1 for k in hf_video_done if f":{category}:" in k)
        enrich_on_hf = sum(
            1
            for row in iter_jsonl(state.hf_enrich_upload_done_path)
            if row.get("category") == category
        )
        download_pending_keys = {k for k in download_pending_keys if f":{category}:" in k}
        hf_video_pending_keys = {k for k in hf_video_pending_keys if f":{category}:" in k}
        hf_shard_pending_keys = {
            k for k in hf_shard_pending_keys if f"category={category}" in k
        }
        enrich_pending_keys = {k for k in enrich_pending_keys if f":{category}:" in k}

    videos_in_shards_count = len(videos_in_shards)
    lifecycle = {
        "accepted": videos_in_shards_count,
        "enriched": videos_enriched,
        "downloaded_or_on_hf": max(videos_downloaded, videos_on_hf),
        "uploaded_metadata_shards": shards_on_hf,
        "uploaded_enrich": enrich_on_hf,
        "uploaded_video": videos_on_hf,
        "lag_enrich": max(0, videos_in_shards_count - videos_enriched),
        "lag_download": max(0, videos_in_shards_count - max(videos_downloaded, videos_on_hf)),
        "lag_hf_video": max(0, videos_in_shards_count - videos_on_hf),
        "lag_hf_enrich": max(0, videos_in_shards_count - enrich_on_hf),
        "training_ready_snapshot0": videos_in_shards_count,
        "training_ready_14_21": snapshot_readiness["training_ready_14_21"],
    }

    retry_dead_letters = list(iter_jsonl(state.queue_dead_letter_path))
    if category:
        retry_dead_letters = [row for row in retry_dead_letters if row.get("category") == category]

    return {
        "category": category,
        "shards": {
            "total": shards_total,
            "on_hf": shards_on_hf,
            "pending_hf_upload": len(hf_shard_pending_keys),
        },
        "videos": {
            "in_shards": len(videos_in_shards),
            "downloaded_local_files": local_mp4,
            "downloaded_marked_done": videos_downloaded,
            "on_hf": videos_on_hf,
            "enriched": videos_enriched,
            "enrich_on_hf": enrich_on_hf,
            "pending_download": len(download_pending_keys),
            "pending_hf_upload": len(hf_video_pending_keys),
            "pending_enrich": len(enrich_pending_keys),
        },
        "lifecycle": lifecycle,
        "snapshots": snapshot_readiness,
        "channels": channel_stats,
        "queues": {
            "dead_letter": len(retry_dead_letters),
        },
        "updated_at": utcnow().isoformat(),
    }


def refresh_summary(state: DatasetState) -> dict[str, Any]:
    stats = compute_inventory_stats(state)
    by_category: Dict[str, Any] = {}
    manifest_categories = {c.name for c in state.config.categories}
    shard_rows = _load_shard_index(state)
    categories_seen = manifest_categories | {r.get("category") for r in shard_rows if r.get("category")}
    for cat in sorted(categories_seen):
        if cat:
            by_category[cat] = compute_inventory_stats(state, category=cat)

    payload = {
        "campaign": state.config.name,
        "output_dir": str(state.root),
        "updated_at": stats["updated_at"],
        "totals": stats,
        "by_category": by_category,
        "shards_index": str(shards_index_path(state).relative_to(state.root)),
        "videos_index": str(videos_index_path(state).relative_to(state.root)),
    }
    atomic_write_json(summary_path(state), payload)
    return payload


def load_summary(state: DatasetState) -> dict[str, Any] | None:
    path = summary_path(state)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_shard_records(
    state: DatasetState,
    *,
    category: str | None = None,
    limit: int | None = None,
) -> List[dict]:
    rows = _load_shard_index(state)
    if category:
        rows = [r for r in rows if r.get("category") == category]
    if limit is not None:
        rows = rows[-limit:]
    return rows
