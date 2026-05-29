from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

from fetcher.dataset_collector.collector import DatasetCollector
from fetcher.dataset_collector.config import load_campaign_config, write_campaign_template
from fetcher.dataset_collector.cookies import CookieRotator
from fetcher.dataset_collector.discovery.base import DiscoveryAdapter
from fetcher.dataset_collector.discovery.rutube import RutubeDiscoveryAdapter
from fetcher.dataset_collector.discovery.tiktok import TikTokDiscoveryAdapter
from fetcher.dataset_collector.discovery.twitch import TwitchDiscoveryAdapter
from fetcher.dataset_collector.discovery.youtube import YouTubeDiscoveryAdapter, YouTubeKeyPool
from fetcher.dataset_collector.downloads import run_download_queue
from fetcher.dataset_collector.export import export_legacy_json, validate_export
from fetcher.dataset_collector.hf_upload import upload_paths
from fetcher.dataset_collector.legacy_import import import_seen_ids
from fetcher.dataset_collector.proxy import ProxyRotator, configured_proxies
from fetcher.dataset_collector.snapshots import SnapshotRunner
from fetcher.dataset_collector.state import DatasetState
from fetcher.config import settings


def load_youtube_keys(path: str | None) -> list[str]:
    if path:
        keys_path = Path(path)
        raw = keys_path.read_text(encoding="utf-8")
        if keys_path.suffix.lower() == ".json":
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(item) for item in data if item]
            return [str(item) for item in data.get("keys", []) if item]
        keys = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keys.extend([token.strip() for token in line.replace(",", " ").split() if token.strip()])
        return list(dict.fromkeys(keys))
    env_value = os.getenv("FETCHER_YOUTUBE_DATA_API_KEYS") or os.getenv("FETCHER_YOUTUBE_DATA_API_KEY")
    return [item.strip() for item in (env_value or "").split(",") if item.strip()]


def build_adapters(state: DatasetState, args: argparse.Namespace) -> Dict[str, DiscoveryAdapter]:
    adapters: Dict[str, DiscoveryAdapter] = {}
    config = getattr(args, "_campaign_config", None)
    cookie_rotator = CookieRotator.from_config(config) if config is not None else None
    keys_path = getattr(args, "youtube_keys", None) or (config.youtube_keys_file if config else None)
    youtube_keys = load_youtube_keys(keys_path)
    if youtube_keys:
        key_pool = YouTubeKeyPool(
            youtube_keys,
            state_path=state.api_keys_path,
            proxy_rotator=ProxyRotator(config=config, include_local=False),
        )
        adapters["youtube"] = YouTubeDiscoveryAdapter(key_pool)
    if getattr(args, "enable_tiktok", False):
        adapters["tiktok"] = TikTokDiscoveryAdapter(
            proxy_rotator=ProxyRotator(config=config, include_local=False),
            cookie_rotator=cookie_rotator,
        )
    twitch_client_id = os.getenv("FETCHER_TWITCH_CLIENT_ID")
    twitch_token = os.getenv("FETCHER_TWITCH_ACCESS_TOKEN")
    if getattr(args, "enable_twitch", False) and twitch_client_id and twitch_token:
        adapters["twitch"] = TwitchDiscoveryAdapter(client_id=twitch_client_id, access_token=twitch_token)
    if getattr(args, "enable_rutube", False):
        adapters["rutube"] = RutubeDiscoveryAdapter()
    return adapters


def command_init(args: argparse.Namespace) -> None:
    path = write_campaign_template(args.config, overwrite=args.force)
    print(f"Wrote campaign template: {path}")


def command_discover(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config)
    args._campaign_config = config
    state = DatasetState(config)
    state.initialize()
    collector = DatasetCollector(config, state, build_adapters(state, args))
    categories = [args.category] if args.category else [category.name for category in config.categories]
    total = {"accepted": 0, "rejected": 0}
    for category in categories:
        result = collector.discover_category(category, limit=args.limit)
        total["accepted"] += result["accepted"]
        total["rejected"] += result["rejected"]
        print(f"{category}: accepted={result['accepted']} rejected={result['rejected']}")
    print(json.dumps(total, ensure_ascii=False))


def command_snapshot(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config)
    args._campaign_config = config
    state = DatasetState(config)
    state.initialize()
    runner = SnapshotRunner(
        state,
        build_adapters(state, args),
        comments_limit=config.comments_per_snapshot,
    )
    result = runner.collect_due(snapshot_index=args.snapshot_index, limit=args.limit)
    print(json.dumps({"snapshots": len(result)}, ensure_ascii=False))


def command_download(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config)
    if config.cookie_files_dir:
        settings.cookie_files_dir = config.cookie_files_dir
        settings.cookie_file_glob = config.cookie_file_glob
    download_proxies = configured_proxies(config=config, include_local=True)
    if download_proxies:
        settings.enable_proxies = True
        settings.proxies = download_proxies
    state = DatasetState(config)
    queue_path = state.download_dir / "queue.jsonl"
    print(json.dumps(run_download_queue(queue_path), ensure_ascii=False))


def command_export(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config)
    result = export_legacy_json(config.output_dir, args.output, split_count=args.split_count)
    print(json.dumps(result, ensure_ascii=False))


def command_status(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config)
    state = DatasetState(config)
    manifest = state.initialize()
    print(manifest.json(indent=2, ensure_ascii=False))


def command_validate(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config)
    print(json.dumps(validate_export(config.output_dir, required_snapshots=args.required_snapshots), ensure_ascii=False))


def command_import_seen(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config)
    state = DatasetState(config)
    state.initialize()
    imported = import_seen_ids(state, args.input, platform=args.platform, category=args.category)
    print(json.dumps({"imported": imported}, ensure_ascii=False))


def command_upload_hf(args: argparse.Namespace) -> None:
    config = load_campaign_config(args.config)
    paths = [Path(path) for path in args.paths]
    print(json.dumps(upload_paths(config, paths, repo_id=args.repo_id), ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fetcher.dataset_collector.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init-campaign")
    init_cmd.add_argument("config")
    init_cmd.add_argument("--force", action="store_true")
    init_cmd.set_defaults(func=command_init)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("config")
        command.add_argument("--youtube-keys")
        command.add_argument("--enable-tiktok", action="store_true")
        command.add_argument("--enable-twitch", action="store_true")
        command.add_argument("--enable-rutube", action="store_true")

    discover = sub.add_parser("discover")
    add_common(discover)
    discover.add_argument("--category")
    discover.add_argument("--limit", type=int)
    discover.set_defaults(func=command_discover)

    snapshot = sub.add_parser("snapshot")
    add_common(snapshot)
    snapshot.add_argument("--snapshot-index", type=int, required=True)
    snapshot.add_argument("--limit", type=int)
    snapshot.set_defaults(func=command_snapshot)

    download = sub.add_parser("download")
    download.add_argument("config")
    download.set_defaults(func=command_download)

    export = sub.add_parser("export")
    export.add_argument("config")
    export.add_argument("output")
    export.add_argument("--split-count", type=int, default=20)
    export.set_defaults(func=command_export)

    status = sub.add_parser("status")
    status.add_argument("config")
    status.set_defaults(func=command_status)

    validate = sub.add_parser("validate")
    validate.add_argument("config")
    validate.add_argument("--required-snapshots", type=int, default=1)
    validate.set_defaults(func=command_validate)

    import_seen = sub.add_parser("import-seen")
    import_seen.add_argument("config")
    import_seen.add_argument("input")
    import_seen.add_argument("--platform", default="youtube")
    import_seen.add_argument("--category", default="legacy")
    import_seen.set_defaults(func=command_import_seen)

    upload_hf = sub.add_parser("upload-hf")
    upload_hf.add_argument("config")
    upload_hf.add_argument("paths", nargs="+")
    upload_hf.add_argument("--repo-id")
    upload_hf.set_defaults(func=command_upload_hf)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
