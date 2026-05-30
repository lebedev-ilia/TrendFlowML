from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

from fetcher.dataset_collector.hf_upload import (
    HuggingFaceUploadError,
    remote_shard_path,
    remote_video_path,
    resolve_shards_repo_id,
    resolve_videos_repo_id,
    upload_local_file,
)
from fetcher.dataset_collector.schemas import CampaignConfig
from fetcher.dataset_collector.state import DatasetState, iter_jsonl
from fetcher.dataset_collector.worker_logging import (
    count_glob_files,
    count_jsonl_lines,
    log_kv_block,
    log_pass_footer,
    log_pass_header,
    worker_log,
)


def _iter_queue(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def scan_shards_for_hf_upload(
    state: DatasetState,
    *,
    category: str | None = None,
    done_keys: set[str] | None = None,
) -> int:
    done_keys = done_keys or state.load_hf_shard_upload_done()
    queued = 0
    seen: set[str] = set()
    metadata_root = state.shards_dir / "metadata"
    if not metadata_root.exists():
        return 0

    pattern = f"category={category}/part_*.json" if category else "**/part_*.json"
    for shard_path in sorted(metadata_root.glob(pattern)):
        if shard_path.name.endswith(".tmp"):
            continue
        shard_relpath = str(shard_path.relative_to(state.root))
        key = f"shard:{shard_relpath}"
        if key in done_keys or key in seen:
            continue
        seen.add(key)
        category_name = category or shard_path.parent.name.replace("category=", "")
        state.enqueue_hf_shard_upload(
            shard_relpath=shard_relpath,
            category=category_name,
        )
        queued += 1
    return queued


def scan_downloaded_videos_for_hf_upload(
    state: DatasetState,
    *,
    category: str | None = None,
    done_keys: set[str] | None = None,
) -> int:
    done_keys = done_keys or state.load_hf_video_upload_done()
    videos_root = state.download_dir / "videos"
    if not videos_root.exists():
        return 0

    queued = 0
    seen: set[str] = set()
    if category:
        roots = [videos_root / category]
    else:
        roots = [path for path in videos_root.iterdir() if path.is_dir()]

    for root in roots:
        if not root.is_dir():
            continue
        category_name = root.name
        for video_path in sorted(root.glob("*.mp4")):
            video_id = video_path.stem
            key = f"youtube:{category_name}:{video_id}"
            if key in done_keys or key in seen:
                continue
            seen.add(key)
            state.enqueue_hf_video_upload(
                platform="youtube",
                video_id=video_id,
                category=category_name,
                local_path=str(video_path.relative_to(state.root)),
            )
            queued += 1
    return queued


def run_hf_shard_upload_queue(
    state: DatasetState,
    config: CampaignConfig,
    *,
    category: str | None = None,
    limit: int | None = None,
    repo_id: str | None = None,
) -> dict[str, int]:
    log_pass_header("hf-shards", "pass start")

    try:
        target_repo = resolve_shards_repo_id(config, repo_id=repo_id)
    except HuggingFaceUploadError as exc:
        worker_log("hf-shards", str(exc))
        result = {"uploaded": 0, "failed": 0, "skipped": 0, "error": str(exc), "attempted": 0}
        log_pass_footer("hf-shards", result)
        return result

    done_keys = state.load_hf_shard_upload_done()
    queue_path = state.hf_shard_upload_queue_path
    queue_lines = count_jsonl_lines(queue_path)

    log_kv_block(
        "hf-shards",
        [
            ("repo", target_repo),
            ("queue_file", queue_path),
            ("queue_lines", queue_lines),
            ("already_on_hf", len(done_keys)),
            ("category_filter", category or "all"),
            ("hf_upload_enabled", config.hf_upload_enabled),
            ("limit_this_pass", limit if limit is not None else "none"),
        ],
    )

    if not config.hf_upload_enabled:
        worker_log("hf-shards", "hf_upload_enabled=false in config — enable to upload")

    if queue_lines == 0:
        worker_log("hf-shards", "queue empty — shards enqueue when part_*.json is written")

    results = {"uploaded": 0, "failed": 0, "skipped": 0}
    skip_reasons: dict[str, int] = {}
    attempted = 0

    for item in _iter_queue(queue_path):
        if limit is not None and results["uploaded"] + results["failed"] >= limit:
            worker_log("hf-shards", f"limit reached ({limit}), stopping this pass")
            break
        shard_relpath = item.get("shard")
        if not shard_relpath:
            results["skipped"] += 1
            skip_reasons["invalid_row"] = skip_reasons.get("invalid_row", 0) + 1
            continue
        if category and item.get("category") != category:
            results["skipped"] += 1
            skip_reasons["other_category"] = skip_reasons.get("other_category", 0) + 1
            continue

        key = f"shard:{shard_relpath}"
        if key in done_keys:
            results["skipped"] += 1
            skip_reasons["already_on_hf"] = skip_reasons.get("already_on_hf", 0) + 1
            continue

        local_path = state.root / shard_relpath
        if not local_path.is_file():
            worker_log("hf-shards", f"FAIL missing file {shard_relpath}")
            results["failed"] += 1
            continue

        attempted += 1
        size_mb = local_path.stat().st_size / (1024 * 1024)
        remote = remote_shard_path(config, shard_relpath)
        worker_log(
            "hf-shards",
            f"({attempted}) uploading {shard_relpath} ({size_mb:.1f} MiB) -> {target_repo}:{remote}",
        )
        try:
            upload_local_file(config, local_path, repo_id=target_repo, path_in_repo=remote)
            state.mark_hf_shard_upload_done(key, shard_relpath=shard_relpath)
            done_keys.add(key)
            results["uploaded"] += 1
            worker_log("hf-shards", f"OK {shard_relpath}")
        except Exception as exc:
            worker_log("hf-shards", f"FAIL {shard_relpath}: {exc}")
            results["failed"] += 1

    if skip_reasons:
        log_kv_block("hf-shards", [(f"skipped_{k}", v) for k, v in sorted(skip_reasons.items())])

    from fetcher.dataset_collector.inventory import refresh_summary

    refresh_summary(state)
    results["attempted"] = attempted
    log_pass_footer("hf-shards", results)
    return results


def run_hf_video_upload_queue(
    state: DatasetState,
    config: CampaignConfig,
    *,
    category: str | None = None,
    limit: int | None = None,
    repo_id: str | None = None,
) -> dict[str, int]:
    log_pass_header("hf-videos", "pass start")

    try:
        target_repo = resolve_videos_repo_id(config, repo_id=repo_id)
    except HuggingFaceUploadError as exc:
        worker_log("hf-videos", str(exc))
        result = {"uploaded": 0, "failed": 0, "skipped": 0, "error": str(exc), "attempted": 0}
        log_pass_footer("hf-videos", result)
        return result

    done_keys = state.load_hf_video_upload_done()
    queue_path = state.hf_video_upload_queue_path
    queue_lines = count_jsonl_lines(queue_path)
    videos_root = state.download_dir / "videos"
    local_mp4 = count_glob_files(videos_root, "**/*.mp4") if videos_root.exists() else 0

    log_kv_block(
        "hf-videos",
        [
            ("repo", target_repo),
            ("queue_file", queue_path),
            ("queue_lines", queue_lines),
            ("already_on_hf", len(done_keys)),
            ("local_mp4_on_disk", local_mp4),
            ("category_filter", category or "all"),
            ("limit_this_pass", limit if limit is not None else "none"),
        ],
    )

    if local_mp4 == 0:
        worker_log("hf-videos", "no local mp4 yet — wait for download worker (can take minutes per video)")
    if queue_lines == 0:
        worker_log("hf-videos", "HF video queue empty — filled after each successful download")

    results = {"uploaded": 0, "failed": 0, "skipped": 0}
    skip_reasons: dict[str, int] = {}
    attempted = 0

    for item in _iter_queue(queue_path):
        if limit is not None and results["uploaded"] + results["failed"] >= limit:
            worker_log("hf-videos", f"limit reached ({limit}), stopping this pass")
            break
        platform = item.get("platform") or "youtube"
        video_id = item.get("video_id")
        local_relpath = item.get("local_path")
        item_category = item.get("category") or "unknown"
        if not video_id or not local_relpath:
            results["skipped"] += 1
            skip_reasons["invalid_row"] = skip_reasons.get("invalid_row", 0) + 1
            continue
        if category and item_category != category:
            results["skipped"] += 1
            skip_reasons["other_category"] = skip_reasons.get("other_category", 0) + 1
            continue

        key = f"{platform}:{item_category}:{video_id}"
        if key in done_keys:
            results["skipped"] += 1
            skip_reasons["already_on_hf"] = skip_reasons.get("already_on_hf", 0) + 1
            continue

        local_path = state.root / local_relpath
        if not local_path.is_file():
            worker_log("hf-videos", f"FAIL missing {local_relpath} (download not finished?)")
            results["failed"] += 1
            continue

        attempted += 1
        size_mb = local_path.stat().st_size / (1024 * 1024)
        remote = remote_video_path(config, category=item_category, video_id=video_id)
        worker_log(
            "hf-videos",
            f"({attempted}) uploading {video_id} ({size_mb:.1f} MiB) -> {target_repo}:{remote}",
        )
        try:
            upload_local_file(config, local_path, repo_id=target_repo, path_in_repo=remote)
            state.mark_hf_video_upload_done(
                key,
                video_id=video_id,
                category=item_category,
                local_path=local_relpath,
            )
            done_keys.add(key)
            results["uploaded"] += 1
            worker_log("hf-videos", f"OK {video_id}")
        except Exception as exc:
            worker_log("hf-videos", f"FAIL {video_id}: {exc}")
            results["failed"] += 1

    if skip_reasons:
        log_kv_block("hf-videos", [(f"skipped_{k}", v) for k, v in sorted(skip_reasons.items())])

    from fetcher.dataset_collector.inventory import refresh_summary

    refresh_summary(state)
    results["attempted"] = attempted
    log_pass_footer("hf-videos", results)
    return results
