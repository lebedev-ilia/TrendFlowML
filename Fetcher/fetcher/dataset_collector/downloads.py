from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable

from pytubefix import YouTube

from fetcher.dataset_collector.cookies import CookieRotator, install_pytubefix_session
from fetcher.dataset_collector.proxy import ProxyRotator, configured_proxies, pytubefix_proxy_dict
from fetcher.dataset_collector.schemas import CampaignConfig
from fetcher.dataset_collector.state import DatasetState
from fetcher.dataset_collector.worker_shutdown import should_stop
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

MAX_DOWNLOAD_HEIGHT = 1080
# Try highest cap first; on failure retry with lower caps (merge / stream errors).
DOWNLOAD_HEIGHT_TIERS = (1080, 720, 480, 360, 240, 144)


def iter_download_queue(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def scan_metadata_shards_for_downloads(
    state: DatasetState,
    *,
    category: str | None = None,
) -> int:
    """Enqueue discover metadata videos not downloaded/uploaded yet."""
    done = state.load_download_done() | state.load_hf_video_upload_done()
    queued = state.load_hf_video_upload_queued()
    existing_queue = {
        f"{row.get('platform') or 'youtube'}:{row.get('category') or 'unknown'}:{row.get('video_id')}"
        for row in iter_download_queue(state.download_dir / "queue.jsonl")
        if row.get("video_id")
    }
    metadata_root = state.shards_dir / "metadata"
    if not metadata_root.exists():
        return 0
    pattern = f"category={category}/part_*.json" if category else "**/part_*.json"
    added = 0
    for shard_path in sorted(metadata_root.glob(pattern)):
        if shard_path.name.endswith(".tmp"):
            continue
        data = json.loads(shard_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        category_name = category or shard_path.parent.name.replace("category=", "")
        for video_id, entry in data.items():
            platform = entry.get("platform") or "youtube"
            key = f"{platform}:{category_name}:{video_id}"
            if key in done or key in queued or key in existing_queue:
                continue
            state.enqueue_download_item(
                platform=platform,
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                category=category_name,
            )
            existing_queue.add(key)
            added += 1
    return added


def local_video_path(state: DatasetState, *, category: str, video_id: str) -> Path:
    return state.download_dir / "videos" / category / f"{video_id}.mp4"


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _parse_resolution_height(resolution: str | None) -> int:
    if not resolution:
        return 0
    digits = "".join(ch for ch in resolution if ch.isdigit())
    return int(digits) if digits else 0


def _pick_progressive_stream(yt: YouTube, max_height: int):
    streams = yt.streams.filter(progressive=True, file_extension="mp4")
    for height in range(max_height, 0, -1):
        stream = streams.filter(res=f"{height}p").first()
        if stream:
            return stream
    return streams.get_highest_resolution()


def _pick_adaptive_streams(yt: YouTube, *, max_height: int):
    video = None
    for stream in yt.streams.filter(adaptive=True, only_video=True).order_by("resolution").desc():
        height = _parse_resolution_height(stream.resolution)
        if height > max_height:
            continue
        if stream.subtype == "mp4":
            video = stream
            break
    if video is None:
        for stream in yt.streams.filter(adaptive=True, only_video=True).order_by("resolution").desc():
            if _parse_resolution_height(stream.resolution) <= max_height:
                video = stream
                break
    audio = yt.streams.filter(only_audio=True, mime_type="audio/mp4").order_by("abr").desc().first()
    if audio is None:
        audio = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    if video is None or audio is None:
        return None, None
    return video, audio


def _check_shutdown() -> None:
    if should_stop():
        raise KeyboardInterrupt("worker shutdown requested")


def _cleanup_download_temps(target: Path) -> None:
    for suffix in (".video.tmp", ".audio.tmp"):
        target.with_name(f"{target.stem}{suffix}").unlink(missing_ok=True)
    if target.exists() and target.stat().st_size == 0:
        target.unlink(missing_ok=True)


def _merge_av_with_ffmpeg(video_path: Path, audio_path: Path, output_path: Path) -> None:
    video_path = video_path.resolve()
    audio_path = audio_path.resolve()
    output_path = output_path.resolve()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while proc.poll() is None:
        _check_shutdown()
        time.sleep(0.25)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=stdout,
            stderr=stderr,
        )


def _download_youtube_mp4_at_height(
    yt: YouTube,
    *,
    target: Path,
    max_height: int,
    log: Callable[[str], None],
) -> str:
    """Download best mp4 up to max_height; ffmpeg-merge adaptive when higher than progressive."""
    progressive = _pick_progressive_stream(yt, max_height)
    progressive_height = _parse_resolution_height(progressive.resolution if progressive else None)

    video_stream, audio_stream = _pick_adaptive_streams(yt, max_height=max_height)
    adaptive_height = _parse_resolution_height(video_stream.resolution if video_stream else None)

    if video_stream and audio_stream and adaptive_height > progressive_height:
        tmp_video = target.with_name(f"{target.stem}.video.tmp")
        tmp_audio = target.with_name(f"{target.stem}.audio.tmp")
        log(
            f"  adaptive {video_stream.resolution} + {audio_stream.abr} "
            f"({video_stream.filesize_mb:.1f}+{audio_stream.filesize_mb:.1f} MiB) -> ffmpeg merge"
        )
        try:
            video_stream.download(output_path=str(target.parent), filename=tmp_video.name)
            _check_shutdown()
            audio_stream.download(output_path=str(target.parent), filename=tmp_audio.name)
            _check_shutdown()
            _merge_av_with_ffmpeg(tmp_video, tmp_audio, target)
        finally:
            tmp_video.unlink(missing_ok=True)
            tmp_audio.unlink(missing_ok=True)
        return f"{video_stream.resolution} (merged)"

    if progressive is None:
        raise RuntimeError(f"no stream available up to {max_height}p")

    log(f"  progressive {progressive.resolution} ({progressive.filesize_mb:.1f} MiB)")
    progressive.download(output_path=str(target.parent), filename=target.name)
    return f"{progressive.resolution} progressive"


def download_youtube_mp4(
    yt: YouTube,
    *,
    target: Path,
    max_height: int = MAX_DOWNLOAD_HEIGHT,
    log: Callable[[str], None] | None = None,
) -> str:
    """Pick best quality ≤1080, stepping down 720/480/… on hard failures."""
    emit = log or (lambda _msg: None)
    tiers = [h for h in DOWNLOAD_HEIGHT_TIERS if h <= max_height]
    if not tiers:
        tiers = list(DOWNLOAD_HEIGHT_TIERS)

    last_error: Exception | None = None
    for tier in tiers:
        _check_shutdown()
        try:
            return _download_youtube_mp4_at_height(yt, target=target, max_height=tier, log=emit)
        except KeyboardInterrupt:
            _cleanup_download_temps(target)
            raise
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            last_error = exc
            _cleanup_download_temps(target)
            if tier != tiers[-1]:
                emit(f"  tier ≤{tier}p failed ({exc}); trying lower resolution …")
            continue

    raise RuntimeError(str(last_error) if last_error else "no stream available")


def _pick_stream(yt: YouTube, *, max_height: int = MAX_DOWNLOAD_HEIGHT):
    return _pick_progressive_stream(yt, max_height)


def _progress_callback(video_id: str) -> Callable:
    last_pct = -1

    def _on_progress(stream, chunk, bytes_remaining) -> None:  # noqa: ARG001
        nonlocal last_pct
        _check_shutdown()
        total = stream.filesize or 0
        if total <= 0:
            return
        pct = int(100 * (total - bytes_remaining) / total)
        if pct >= last_pct + 5 or pct >= 99:
            last_pct = pct
            worker_log("download", f"  {video_id} {pct}%")

    return _on_progress


def download_video_local(
    state: DatasetState,
    config: CampaignConfig,
    *,
    platform: str,
    video_id: str,
    url: str,
    category: str,
    proxy_rotator: ProxyRotator | None,
    cookie_rotator: CookieRotator | None = None,
) -> Path | None:
    if platform != "youtube":
        worker_log("download", f"skip {video_id}: platform {platform!r} not supported")
        return None

    target = local_video_path(state, category=category, video_id=video_id)
    if target.exists() and target.stat().st_size > 0:
        worker_log(
            "download",
            f"already on disk {video_id} -> {target.relative_to(state.root)} "
            f"({_format_bytes(target.stat().st_size)})",
        )
        return target

    if target.exists() and target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        worker_log("download", f"removed empty partial file {video_id}")

    target.parent.mkdir(parents=True, exist_ok=True)
    proxy_url = proxy_rotator.next() if proxy_rotator else None
    proxies = pytubefix_proxy_dict(proxy_url)
    cookie_file = cookie_rotator.next() if cookie_rotator else None
    install_pytubefix_session(proxies=proxies, cookie_file=cookie_file)

    worker_log(
        "download",
        f"pytubefix start {video_id} category={category} proxy={proxy_url or 'direct'} "
        f"cookie={cookie_file.name if cookie_file else 'none'}",
    )
    worker_log("download", f"  url={url}")
    started_at = time.perf_counter()

    try:
        yt = YouTube(url, on_progress_callback=_progress_callback(video_id))
        label = download_youtube_mp4(
            yt,
            target=target,
            max_height=MAX_DOWNLOAD_HEIGHT,
            log=lambda msg: worker_log("download", msg),
        )
        worker_log("download", f"  saved as {label}")

        if proxy_rotator and proxy_url:
            proxy_rotator.record_success(proxy_url)

        if target.exists() and target.stat().st_size > 0:
            elapsed = time.perf_counter() - started_at
            size_bytes = target.stat().st_size
            state.record_performance_event(
                "download",
                {
                    "platform": platform,
                    "video_id": video_id,
                    "category": category,
                    "seconds": round(elapsed, 3),
                    "size_bytes": size_bytes,
                    "local_path": str(target.relative_to(state.root)),
                },
            )
            worker_log(
                "download",
                f"OK {video_id} -> {target.relative_to(state.root)} "
                f"({_format_bytes(size_bytes)}, {elapsed:.1f}s)",
            )
            return target

        worker_log("download", f"FAIL {video_id}: file missing after download")
        if proxy_rotator and proxy_url:
            proxy_rotator.record_download_failure(proxy_url)
        return None
    except KeyboardInterrupt:
        _cleanup_download_temps(target)
        worker_log("download", f"STOP {video_id}: shutdown requested")
        raise
    except RuntimeError as exc:
        worker_log("download", f"FAIL {video_id}: {exc}")
        if proxy_rotator and proxy_url:
            proxy_rotator.record_download_failure(proxy_url)
        return None
    except subprocess.CalledProcessError as exc:
        worker_log("download", f"FAIL {video_id}: ffmpeg merge failed: {exc.stderr.decode(errors='replace')[:300]}")
        if proxy_rotator and proxy_url:
            proxy_rotator.record_download_failure(proxy_url)
        return None
    except Exception as exc:
        if proxy_rotator and proxy_url:
            proxy_rotator.record_download_failure(proxy_url)
        worker_log("download", f"FAIL {video_id}: {type(exc).__name__}: {exc}")
        return None


def run_download_queue(
    state: DatasetState,
    config: CampaignConfig,
    queue_path: Path,
    *,
    limit: int | None = None,
    cookie_rotator=None,
) -> dict[str, int]:
    cookie_rotator = cookie_rotator or CookieRotator.from_config(config)
    if cookie_rotator.cookie_files:
        worker_log(
            "download",
            f"cookies: {', '.join(p.name for p in cookie_rotator.cookie_files)}",
        )

    log_pass_header("download", "pass start")
    queued_from_metadata = scan_metadata_shards_for_downloads(state)
    if queued_from_metadata:
        worker_log("download", f"queued_from_metadata_shards: {queued_from_metadata}")

    proxy_list = configured_proxies(config=config, download_only=True)
    if proxy_list:
        worker_log("download", f"proxies (download_only): {', '.join(proxy_list)}")
    else:
        worker_log("download", "no download_only proxies — direct connection")
    proxy_rotator = ProxyRotator(proxies=proxy_list)

    done_keys = state.load_download_done()
    hf_done_keys = state.load_hf_video_upload_done()
    hf_queued_keys = state.load_hf_video_upload_queued()
    post_enrich_rejected = state.load_post_enrich_rejected_video_ids()
    dead_letter_keys = load_dead_letter_keys(state, service="download")
    queue_lines = count_jsonl_lines(queue_path)
    videos_dir = state.download_dir / "videos"
    local_mp4 = count_glob_files(videos_dir, "**/*.mp4") if videos_dir.exists() else 0

    log_kv_block(
        "download",
        [
            ("queue_file", queue_path),
            ("queue_lines", queue_lines),
            ("already_done", len(done_keys)),
            ("already_on_hf", len(hf_done_keys)),
            ("pending_hf_upload", len(hf_queued_keys - hf_done_keys)),
            ("local_mp4_on_disk", local_mp4),
            ("limit_this_pass", limit if limit is not None else "none (until queue end)"),
        ],
    )

    if queue_lines == 0:
        worker_log("download", "queue empty — nothing to download (wait for discover)")
        result = {"downloaded": 0, "failed": 0, "skipped": 0, "attempted": 0}
        log_pass_footer("download", result)
        return result

    results = {"downloaded": 0, "failed": 0, "skipped": 0}
    skip_reasons: dict[str, int] = {}
    attempted = 0

    for item in iter_download_queue(queue_path):
        if should_stop():
            worker_log("download", "shutdown requested — stopping pass")
            break
        if limit is not None and results["downloaded"] + results["failed"] >= limit:
            worker_log("download", f"limit reached ({limit}), stopping this pass")
            break

        platform = item.get("platform")
        url = item.get("url")
        video_id = item.get("video_id")
        category = item.get("category") or "unknown"
        if not platform or not url or not video_id:
            results["skipped"] += 1
            skip_reasons["invalid_row"] = skip_reasons.get("invalid_row", 0) + 1
            continue

        key = f"{platform}:{category}:{video_id}"
        retry_key = queue_item_key("download", item)
        if retry_key in dead_letter_keys:
            results["skipped"] += 1
            skip_reasons["dead_letter"] = skip_reasons.get("dead_letter", 0) + 1
            continue
        if key in done_keys or key in hf_done_keys:
            results["skipped"] += 1
            skip_reasons["already_done"] = skip_reasons.get("already_done", 0) + 1
            continue
        target = local_video_path(state, category=category, video_id=video_id)
        if key in hf_queued_keys and target.exists() and target.stat().st_size > 0:
            results["skipped"] += 1
            skip_reasons["pending_hf_upload"] = skip_reasons.get("pending_hf_upload", 0) + 1
            continue
        if video_id in post_enrich_rejected:
            results["skipped"] += 1
            skip_reasons["post_enrich_rejected"] = skip_reasons.get("post_enrich_rejected", 0) + 1
            continue

        attempted += 1
        worker_log(
            "download",
            f"({attempted}) pending {video_id} [{category}] "
            f"(pass: +{results['downloaded']} ok, {results['failed']} fail)",
        )

        try:
            path = download_video_local(
                state,
                config,
                platform=platform,
                video_id=video_id,
                url=url,
                category=category,
                proxy_rotator=proxy_rotator,
                cookie_rotator=cookie_rotator,
            )
        except KeyboardInterrupt:
            worker_log("download", "shutdown requested — exiting pass")
            break
        if path is None:
            results["failed"] += 1
            if record_queue_failure(
                state,
                service="download",
                item=item,
                error="download returned no local file",
            ):
                dead_letter_keys.add(retry_key)
            continue

        rel = str(path.relative_to(state.root))
        state.enqueue_hf_video_upload(
            platform=platform,
            video_id=video_id,
            category=category,
            local_path=rel,
        )
        hf_queued_keys.add(key)
        results["downloaded"] += 1
        worker_log("download", f"  -> enqueued HF video upload for {video_id}; done after HF commit")

    if skip_reasons:
        log_kv_block("download", [(f"skipped_{k}", v) for k, v in sorted(skip_reasons.items())])

    local_mp4_after = count_glob_files(videos_dir, "**/*.mp4") if videos_dir.exists() else 0
    worker_log("download", f"local_mp4_on_disk now: {local_mp4_after} (+{local_mp4_after - local_mp4} this pass)")

    from fetcher.dataset_collector.inventory import refresh_summary

    refresh_summary(state)
    results["attempted"] = attempted
    log_pass_footer("download", results)
    return results
