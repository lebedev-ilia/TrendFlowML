from __future__ import annotations

import json
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
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


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
    videos_enriched = len(enrich_done)
    local_mp4 = _count_local_downloads(state, category=category)

    if category:
        videos_downloaded = sum(1 for k in download_done if f":{category}:" in k)
        videos_on_hf = sum(1 for k in hf_video_done if f":{category}:" in k)
        videos_enriched = sum(
            1 for row in iter_jsonl(state.metadata_enrich_done_path)
            if row.get("shard") and f"category={category}" in str(row.get("shard"))
        )
        download_pending_keys = {k for k in download_pending_keys if f":{category}:" in k}
        hf_video_pending_keys = {k for k in hf_video_pending_keys if f":{category}:" in k}
        hf_shard_pending_keys = {
            k for k in hf_shard_pending_keys if f"category={category}" in k
        }
        enrich_pending_keys = {k for k in enrich_pending_keys if f":{category}:" in k}

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
            "pending_download": len(download_pending_keys),
            "pending_hf_upload": len(hf_video_pending_keys),
            "pending_enrich": len(enrich_pending_keys),
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
