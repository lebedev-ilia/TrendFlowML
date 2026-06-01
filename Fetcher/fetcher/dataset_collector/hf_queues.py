from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

from fetcher.dataset_collector.hf_upload import (
    HuggingFaceUploadError,
    remote_enrich_path,
    remote_shard_path,
    remote_video_path,
    resolve_enrich_repo_id,
    resolve_shards_repo_id,
    resolve_videos_repo_id,
    upload_local_files_commit,
)
from fetcher.dataset_collector.schemas import CampaignConfig
from fetcher.dataset_collector.state import DatasetState, iter_jsonl
from fetcher.dataset_collector.queue_retries import (
    load_dead_letter_keys,
    queue_item_key,
    record_queue_failure,
)
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


def _batch(items: list[dict], size: int) -> Iterable[list[dict]]:
    size = max(1, size)
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


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


def scan_enrich_files_for_hf_upload(
    state: DatasetState,
    *,
    category: str | None = None,
    done_keys: set[str] | None = None,
) -> int:
    done_keys = done_keys or state.load_hf_enrich_upload_done()
    enrich_root = state.shards_dir / "enrich"
    if not enrich_root.exists():
        return 0
    queued = 0
    seen = state.load_hf_enrich_upload_queued()
    pattern = f"category={category}/part_*.json" if category else "**/part_*.json"
    for path in sorted(enrich_root.glob(pattern)):
        if path.name.endswith(".tmp"):
            continue
        category_name = category or path.parent.name.replace("category=", "")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for video_id in data:
            key = f"youtube:{video_id}"
            if key in done_keys or key in seen:
                continue
            seen.add(key)
            state.enqueue_hf_enrich_upload(
                platform="youtube",
                video_id=video_id,
                category=category_name,
                local_path=str(path.relative_to(state.root)),
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
    seen = state.load_hf_video_upload_queued()
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
    dead_letter_keys = load_dead_letter_keys(state, service="hf-shards")
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
    pending_uploads: list[dict] = []

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
        retry_key = queue_item_key("hf-shards", item)
        if retry_key in dead_letter_keys:
            results["skipped"] += 1
            skip_reasons["dead_letter"] = skip_reasons.get("dead_letter", 0) + 1
            continue
        if key in done_keys:
            results["skipped"] += 1
            skip_reasons["already_on_hf"] = skip_reasons.get("already_on_hf", 0) + 1
            continue

        local_path = state.root / shard_relpath
        if not local_path.is_file():
            worker_log("hf-shards", f"FAIL missing file {shard_relpath}")
            results["failed"] += 1
            if record_queue_failure(
                state,
                service="hf-shards",
                item=item,
                error=f"missing file {shard_relpath}",
            ):
                dead_letter_keys.add(retry_key)
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        remote = remote_shard_path(config, shard_relpath)
        pending_uploads.append(
            {
                "key": key,
                "shard_relpath": shard_relpath,
                "local_path": local_path,
                "remote": remote,
                "size_mb": size_mb,
            }
        )
        attempted += 1
        if limit is not None and attempted >= limit:
            break

    for batch in _batch(pending_uploads, config.hf_shard_upload_batch_files):
        label = ", ".join(item["shard_relpath"] for item in batch[:3])
        if len(batch) > 3:
            label += f", +{len(batch) - 3} more"
        worker_log("hf-shards", f"uploading batch {len(batch)} -> {target_repo}: {label}")
        try:
            upload_local_files_commit(
                config,
                [(item["local_path"], item["remote"]) for item in batch],
                repo_id=target_repo,
                commit_message=f"Upload {len(batch)} metadata shard(s)",
                state_dir=state.state_dir,
            )
            for item in batch:
                state.mark_hf_shard_upload_done(item["key"], shard_relpath=item["shard_relpath"])
                done_keys.add(item["key"])
                results["uploaded"] += 1
                worker_log("hf-shards", f"OK {item['shard_relpath']}")
            from fetcher.dataset_collector.metrics import record_hf_commit

            record_hf_commit("shards", len(batch))
        except Exception as exc:
            worker_log("hf-shards", f"FAIL batch: {exc}")
            for item in batch:
                if record_queue_failure(
                    state,
                    service="hf-shards",
                    item={"shard": item["shard_relpath"], "category": item.get("category")},
                    error=str(exc),
                ):
                    dead_letter_keys.add(queue_item_key("hf-shards", {"shard": item["shard_relpath"]}))
            results["failed"] += len(batch)

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
    dead_letter_keys = load_dead_letter_keys(state, service="hf-videos")
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
    pending_uploads: list[dict] = []

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
        retry_key = queue_item_key("hf-videos", item)
        if retry_key in dead_letter_keys:
            results["skipped"] += 1
            skip_reasons["dead_letter"] = skip_reasons.get("dead_letter", 0) + 1
            continue
        if key in done_keys:
            results["skipped"] += 1
            skip_reasons["already_on_hf"] = skip_reasons.get("already_on_hf", 0) + 1
            continue

        local_path = state.root / local_relpath
        if not local_path.is_file():
            worker_log("hf-videos", f"FAIL missing {local_relpath} (download not finished?)")
            results["failed"] += 1
            if record_queue_failure(
                state,
                service="hf-videos",
                item=item,
                error=f"missing local file {local_relpath}",
            ):
                dead_letter_keys.add(retry_key)
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        remote = remote_video_path(config, category=item_category, video_id=video_id)
        pending_uploads.append(
            {
                "key": key,
                "platform": platform,
                "video_id": video_id,
                "category": item_category,
                "local_relpath": local_relpath,
                "local_path": local_path,
                "remote": remote,
                "size_mb": size_mb,
            }
        )
        attempted += 1
        if limit is not None and attempted >= limit:
            break

    for batch in _batch(pending_uploads, config.hf_video_upload_batch_files):
        label = ", ".join(item["video_id"] for item in batch[:5])
        if len(batch) > 5:
            label += f", +{len(batch) - 5} more"
        worker_log("hf-videos", f"uploading batch {len(batch)} -> {target_repo}: {label}")
        try:
            upload_local_files_commit(
                config,
                [(item["local_path"], item["remote"]) for item in batch],
                repo_id=target_repo,
                commit_message=f"Upload {len(batch)} video file(s)",
                state_dir=state.state_dir,
            )
            for item in batch:
                state.mark_hf_video_upload_done(
                    item["key"],
                    video_id=item["video_id"],
                    category=item["category"],
                    local_path=item["local_relpath"],
                )
                state.mark_download_done(
                    item["key"],
                    video_id=item["video_id"],
                    category=item["category"],
                    local_path=item["local_relpath"],
                )
                item["local_path"].unlink(missing_ok=True)
                done_keys.add(item["key"])
                results["uploaded"] += 1
                worker_log("hf-videos", f"OK {item['video_id']} (uploaded; local file removed)")
            from fetcher.dataset_collector.metrics import record_hf_commit

            record_hf_commit("videos", len(batch))
        except Exception as exc:
            worker_log("hf-videos", f"FAIL batch: {exc}")
            for item in batch:
                if record_queue_failure(
                    state,
                    service="hf-videos",
                    item={
                        "platform": item["platform"],
                        "category": item["category"],
                        "video_id": item["video_id"],
                        "local_path": item["local_relpath"],
                    },
                    error=str(exc),
                ):
                    dead_letter_keys.add(queue_item_key("hf-videos", item))
            results["failed"] += len(batch)

    if skip_reasons:
        log_kv_block("hf-videos", [(f"skipped_{k}", v) for k, v in sorted(skip_reasons.items())])

    from fetcher.dataset_collector.inventory import refresh_summary

    refresh_summary(state)
    results["attempted"] = attempted
    log_pass_footer("hf-videos", results)
    return results


def run_hf_enrich_upload_queue(
    state: DatasetState,
    config: CampaignConfig,
    *,
    category: str | None = None,
    limit: int | None = None,
    repo_id: str | None = None,
) -> dict[str, int]:
    log_pass_header("hf-enrich", "pass start")

    try:
        target_repo = resolve_enrich_repo_id(config, repo_id=repo_id)
    except HuggingFaceUploadError as exc:
        worker_log("hf-enrich", str(exc))
        result = {"uploaded": 0, "failed": 0, "skipped": 0, "error": str(exc), "attempted": 0}
        log_pass_footer("hf-enrich", result)
        return result

    done_keys = state.load_hf_enrich_upload_done()
    dead_letter_keys = load_dead_letter_keys(state, service="hf-enrich")
    queue_path = state.hf_enrich_upload_queue_path
    queue_lines = count_jsonl_lines(queue_path)
    enrich_root = state.shards_dir / "enrich"
    local_json = count_glob_files(enrich_root, "**/part_*.json") if enrich_root.exists() else 0

    log_kv_block(
        "hf-enrich",
        [
            ("repo", target_repo),
            ("queue_file", queue_path),
            ("queue_lines", queue_lines),
            ("already_on_hf", len(done_keys)),
            ("local_enrich_json", local_json),
            ("category_filter", category or "all"),
            ("limit_this_pass", limit if limit is not None else "none"),
        ],
    )

    if queue_lines == 0:
        worker_log("hf-enrich", "HF enrich queue empty — filled after successful yt-dlp enrich")

    results = {"uploaded": 0, "failed": 0, "skipped": 0}
    skip_reasons: dict[str, int] = {}
    attempted = 0
    pending_uploads: list[dict] = []

    for item in _iter_queue(queue_path):
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

        key = f"{platform}:{video_id}"
        retry_key = queue_item_key("hf-enrich", item)
        if retry_key in dead_letter_keys:
            results["skipped"] += 1
            skip_reasons["dead_letter"] = skip_reasons.get("dead_letter", 0) + 1
            continue
        if key in done_keys:
            results["skipped"] += 1
            skip_reasons["already_on_hf"] = skip_reasons.get("already_on_hf", 0) + 1
            continue

        local_path = state.root / local_relpath
        if not local_path.is_file():
            if str(local_relpath).endswith("part_pending.json"):
                results["skipped"] += 1
                skip_reasons["stale_pending_path"] = skip_reasons.get("stale_pending_path", 0) + 1
                continue
            worker_log("hf-enrich", f"FAIL missing {local_relpath}")
            results["failed"] += 1
            if record_queue_failure(
                state,
                service="hf-enrich",
                item=item,
                error=f"missing local file {local_relpath}",
            ):
                dead_letter_keys.add(retry_key)
            continue

        pending_uploads.append(
            {
                "key": key,
                "video_id": video_id,
                "category": item_category,
                "local_relpath": local_relpath,
                "local_path": local_path,
                "remote": remote_enrich_path(config, local_relpath),
            }
        )
        attempted += 1
        if limit is not None and attempted >= limit:
            break

    for batch in _batch(pending_uploads, config.hf_enrich_upload_batch_files):
        label = ", ".join(item["video_id"] for item in batch[:5])
        if len(batch) > 5:
            label += f", +{len(batch) - 5} more"
        unique_files: dict[str, tuple[Path, str]] = {}
        for item in batch:
            unique_files[item["local_relpath"]] = (item["local_path"], item["remote"])
        worker_log(
            "hf-enrich",
            f"uploading batch {len(batch)} video refs / {len(unique_files)} file(s) -> {target_repo}: {label}",
        )
        try:
            upload_local_files_commit(
                config,
                list(unique_files.values()),
                repo_id=target_repo,
                commit_message=f"Upload {len(unique_files)} enrich shard file(s)",
                state_dir=state.state_dir,
            )
            for item in batch:
                state.mark_hf_enrich_upload_done(
                    item["key"],
                    video_id=item["video_id"],
                    category=item["category"],
                    local_path=item["local_relpath"],
                )
                state.mark_metadata_enrich_done(
                    item["key"],
                    video_id=item["video_id"],
                    shard=item["local_relpath"],
                )
                results["uploaded"] += 1
                worker_log("hf-enrich", f"OK {item['video_id']}")
            from fetcher.dataset_collector.metrics import record_hf_commit

            record_hf_commit("enrich", len(unique_files))
        except Exception as exc:
            worker_log("hf-enrich", f"FAIL batch: {exc}")
            for item in batch:
                if record_queue_failure(
                    state,
                    service="hf-enrich",
                    item={
                        "video_id": item["video_id"],
                        "category": item["category"],
                        "local_path": item["local_relpath"],
                    },
                    error=str(exc),
                ):
                    dead_letter_keys.add(queue_item_key("hf-enrich", item))
            results["failed"] += len(batch)

    if skip_reasons:
        log_kv_block("hf-enrich", [(f"skipped_{k}", v) for k, v in sorted(skip_reasons.items())])

    from fetcher.dataset_collector.inventory import refresh_summary

    refresh_summary(state)
    results["attempted"] = attempted
    log_pass_footer("hf-enrich", results)
    return results
