"""Smoke-test: parse metadata via YouTube Data API, pytubefix, yt-dlp (+ optional download)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FETCHER_ROOT = ROOT / "Fetcher"
sys.path.insert(0, str(FETCHER_ROOT))

from pytubefix import YouTube

from fetcher.dataset_collector.cli import load_youtube_keys
from fetcher.dataset_collector.config import load_campaign_config
from fetcher.dataset_collector.cookies import CookieRotator, install_pytubefix_session
from fetcher.dataset_collector.discovery.youtube import YouTubeDiscoveryAdapter, YouTubeKeyPool
from fetcher.dataset_collector.downloads import download_youtube_mp4
from fetcher.dataset_collector.metadata_enrichment import fetch_ytdlp_info
from fetcher.dataset_collector.proxy import ProxyRotator, configured_proxies, pytubefix_proxy_dict
from fetcher.dataset_collector.schemas import CampaignConfig
from fetcher.dataset_collector.training_format import (
    extract_ytdlp_formats,
    merge_ytdlp_into_training_metadata,
)

DEFAULT_URL = "https://www.youtube.com/watch?v=-A8CX3ISjpA"
DEFAULT_CONFIG = FETCHER_ROOT / "dataset_campaign.json"


def _download_proxy(config: CampaignConfig, override: str | None) -> str | None:
    if override:
        return override
    proxies = configured_proxies(config=config, download_only=True)
    return proxies[0] if proxies else None


def _video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Cannot parse video id from URL: {url}")
    return match.group(1)


def _resolve_fetcher_path(config: CampaignConfig, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return FETCHER_ROOT / path


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _ok(label: str, payload: dict[str, Any]) -> None:
    print(f"OK {label}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _fail(label: str, exc: Exception) -> None:
    print(f"FAIL {label}: {type(exc).__name__}: {exc}")


def parse_youtube_api(
    *,
    config: CampaignConfig,
    video_id: str,
    comments_limit: int = 5,
) -> dict[str, Any]:
    keys_path = _resolve_fetcher_path(config, config.youtube_keys_file)
    keys = load_youtube_keys(str(keys_path) if keys_path else None)
    if not keys:
        raise RuntimeError(
            "No YouTube API keys — set FETCHER_YOUTUBE_DATA_API_KEY or "
            f"create {keys_path or 'keys file'}"
        )

    pool = YouTubeKeyPool(
        keys,
        proxy_rotator=ProxyRotator(proxies=configured_proxies(config=config, download_only=False)),
    )
    adapter = YouTubeDiscoveryAdapter(pool)
    client = pool.get_client()
    meta = client.get_video_metadata(video_id)
    pool.record_success(client.api_key, client.quota_tracker.used_units)

    channel = None
    if meta.channel_id:
        channels = client.get_channels_metadata_batch([meta.channel_id])
        channel = channels[0] if channels else None

    snapshot = adapter._snapshot_from_metadata(meta, channel, snapshot_index=0)  # noqa: SLF001
    internal = adapter._metadata_dict(meta, channel)  # noqa: SLF001

    comments: list[dict[str, Any]] = []
    try:
        comments = [
            {
                "author": c.author_display_name,
                "likes": c.like_count,
                "text": (c.text_original or "")[:120],
            }
            for c in client.iter_comments(video_id, max_count=comments_limit)
        ]
        pool.record_success(client.api_key, client.quota_tracker.used_units)
    except Exception as exc:
        comments = [{"error": str(exc)[:200]}]

    return {
        "source": "youtube_data_api",
        "video_id": video_id,
        "title": meta.title,
        "channel": meta.channel_title,
        "duration_seconds": meta.duration_seconds,
        "published_at": meta.published_at.isoformat(),
        "views": meta.view_count,
        "likes": meta.like_count,
        "comments_count": meta.comment_count,
        "snapshot_0": {
            "viewCount": snapshot.viewCount,
            "likeCount": snapshot.likeCount,
            "commentCount": snapshot.commentCount,
            "subscriberCount": snapshot.subscriberCount,
        },
        "metadata_keys": sorted(internal.keys()),
        "comments_sample": comments,
        "quota_used": client.quota_tracker.used_units,
    }


def parse_pytubefix(*, url: str, proxy: str | None, cookie_dir: Path) -> dict[str, Any]:
    rotator = CookieRotator.from_directory(cookie_dir)
    cookie_file = rotator.next()
    if cookie_file is None:
        raise RuntimeError(f"No cookies in {cookie_dir}")

    proxies = pytubefix_proxy_dict(proxy)
    install_pytubefix_session(proxies=proxies, cookie_file=cookie_file)
    yt = YouTube(url)

    progressive = list(yt.streams.filter(progressive=True, file_extension="mp4"))
    adaptive_video = list(yt.streams.filter(adaptive=True, only_video=True))
    audio = list(yt.streams.filter(only_audio=True))

    resolutions = sorted(
        {s.resolution for s in progressive + adaptive_video if s.resolution},
        key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or "0"),
        reverse=True,
    )

    return {
        "source": "pytubefix",
        "video_id": yt.video_id,
        "title": yt.title,
        "author": yt.author,
        "length_seconds": yt.length,
        "views": yt.views,
        "publish_date": str(yt.publish_date) if yt.publish_date else None,
        "cookie": cookie_file.name,
        "proxy": proxy or "direct",
        "progressive_streams": len(progressive),
        "adaptive_video_streams": len(adaptive_video),
        "audio_streams": len(audio),
        "resolutions": resolutions[:8],
        "best_progressive": progressive[0].resolution if progressive else None,
        "best_adaptive": adaptive_video[0].resolution if adaptive_video else None,
    }


def parse_ytdlp(
    *,
    url: str,
    config: CampaignConfig,
) -> dict[str, Any]:
    cookie_rotator = CookieRotator.from_config(config)
    discovery_proxies = configured_proxies(config=config, download_only=False)
    proxy_rotator = ProxyRotator(proxies=discovery_proxies) if discovery_proxies else None

    info = fetch_ytdlp_info(url, cookie_rotator=cookie_rotator, proxy_rotator=proxy_rotator)
    if info is None:
        raise RuntimeError("yt-dlp returned no info")

    merged = merge_ytdlp_into_training_metadata({}, info)
    formats = extract_ytdlp_formats(info)

    return {
        "source": "yt_dlp",
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "duration_seconds": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "language": info.get("language"),
        "tags_count": len(info.get("tags") or []),
        "tags_sample": (info.get("tags") or [])[:5],
        "is_live": info.get("is_live"),
        "formats_count": len(info.get("formats") or []),
        "video_formats_summary": formats[:6],
        "automatic_caption_langs": sorted((info.get("automatic_captions") or {}).keys())[:10],
        "subtitle_langs": sorted((info.get("subtitles") or {}).keys())[:10],
        "merged_metadata_keys": sorted(merged.keys()),
        "merged_duration": merged.get("duration_seconds"),
        "proxy": discovery_proxies[0] if discovery_proxies else "direct",
    }


def maybe_download(*, url: str, proxy: str | None, cookie_dir: Path, output_dir: Path, max_height: int) -> None:
    _section("DOWNLOAD (pytubefix + ffmpeg merge)")
    rotator = CookieRotator.from_directory(cookie_dir)
    cookie_file = rotator.next()
    install_pytubefix_session(proxies=pytubefix_proxy_dict(proxy), cookie_file=cookie_file)
    yt = YouTube(url)
    target = output_dir / f"{yt.video_id}.mp4"
    started = time.time()
    label = download_youtube_mp4(yt, target=target, max_height=max_height, log=print)
    print(f"saved {target} — {label} ({time.time() - started:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test YouTube parsing (API / pytubefix / yt-dlp)")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--proxy", default=None, help="Override download_only proxy for pytubefix")
    parser.add_argument("--download", action="store_true", help="Also download mp4 after parsing")
    parser.add_argument("--max-height", type=int, default=1080)
    parser.add_argument("--comments", type=int, default=5, help="API comments sample size")
    args = parser.parse_args()

    config = load_campaign_config(args.config)
    vid = _video_id(args.url)
    download_proxy = _download_proxy(config, args.proxy)
    discovery_proxies = configured_proxies(config=config, download_only=False)

    print(f"URL: {args.url}")
    print(f"video_id: {vid}")
    print(f"download proxy (nodpi): {download_proxy or 'none'}")
    print(f"discover/enrich proxies: {discovery_proxies or ['direct']}")
    print(f"config: {args.config}")

    _section("1) YouTube Data API (discover)")
    try:
        _ok("youtube_api", parse_youtube_api(config=config, video_id=vid, comments_limit=args.comments))
    except Exception as exc:
        _fail("youtube_api", exc)

    # _section("2) pytubefix (download worker metadata)")
    # try:
    #     cookie_dir = _resolve_fetcher_path(config, config.cookie_files_dir)
    #     if cookie_dir is None:
    #         raise RuntimeError("cookie_files_dir is not configured")
    #     _ok("pytubefix", parse_pytubefix(url=args.url, proxy=download_proxy, cookie_dir=cookie_dir))
    # except Exception as exc:
    #     _fail("pytubefix", exc)

    # _section("3) yt-dlp (enrich-metadata worker)")
    # try:
    #     _ok("yt_dlp", parse_ytdlp(url=args.url, config=config))
    # except Exception as exc:
    #     _fail("yt_dlp", exc)

    # if args.download:
    #     cookie_dir = _resolve_fetcher_path(config, config.cookie_files_dir)
    #     if cookie_dir is None:
    #         raise RuntimeError("cookie_files_dir is not configured")
    #     maybe_download(
    #         url=args.url,
    #         proxy=download_proxy,
    #         cookie_dir=cookie_dir,
    #         output_dir=ROOT,
    #         max_height=args.max_height,
    #     )


if __name__ == "__main__":
    main()
