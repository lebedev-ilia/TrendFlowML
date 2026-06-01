from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fetcher_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_runtime_config(args: argparse.Namespace) -> Path:
    fetcher_root = _fetcher_root()
    template_path = fetcher_root / args.template
    config = _read_json(template_path)
    output_dir = Path(args.output_dir).expanduser()
    config["output_dir"] = str(output_dir)
    config["name"] = args.run_name
    config["campaign_profile"] = config.get("campaign_profile") or "20k-colab-free-popularity-v1"
    if args.hf_repo_prefix:
        prefix = args.hf_repo_prefix.rstrip("/")
        config["hf_repo_id"] = f"{prefix}/dataset_20k_colab"
        config["hf_shards_repo_id"] = f"{prefix}/dataset_20k_colab_shards"
        config["hf_videos_repo_id"] = f"{prefix}/dataset_20k_colab_videos"
        config["hf_enrich_repo_id"] = f"{prefix}/dataset_20k_colab_enrich"
    if args.youtube_keys_file:
        config["youtube_keys_file"] = args.youtube_keys_file
    if args.cookie_files_dir:
        config["cookie_files_dir"] = args.cookie_files_dir
    if args.proxies_file:
        config["proxies_file"] = args.proxies_file
    if args.disable_hf_upload:
        config["hf_upload_enabled"] = False
    runtime_config = output_dir / "runtime_dataset_campaign_20k.json"
    _write_json(runtime_config, config)
    return runtime_config


def run_command(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_fetcher_root()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Google Colab bootstrap for the 20k dataset collector run.")
    parser.add_argument("--template", default="dataset_campaign_20k.json")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/dataset_runs/20k-test")
    parser.add_argument("--run-name", default="dataset-20k-colab")
    parser.add_argument("--role", choices=["discover", "workers", "snapshot", "status", "inventory-rebuild"], default="workers")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--snapshot-index", type=int)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--metrics-port", type=int, default=0, help="0 disables Prometheus server for Colab.")
    parser.add_argument("--lease-name", help="Shared lease name for multi-Colab coordination.")
    parser.add_argument("--lease-owner", default=os.getenv("COLAB_RELEASE_TAG") or os.getenv("HOSTNAME"))
    parser.add_argument("--hf-repo-prefix", help="HF namespace, e.g. Ilialebedev.")
    parser.add_argument("--youtube-keys-file")
    parser.add_argument("--cookie-files-dir")
    parser.add_argument("--proxies-file")
    parser.add_argument("--disable-hf-upload", action="store_true")
    args = parser.parse_args(argv)

    runtime_config = build_runtime_config(args)
    py = sys.executable
    base = [py, "-m", "fetcher.dataset_collector.cli"]

    if args.role == "discover":
        cmd = [*base, "discover", str(runtime_config)]
        if args.category:
            cmd += ["--category", args.category]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.metrics_port:
            cmd += ["--metrics-port", str(args.metrics_port)]
        return run_command(cmd)

    if args.role == "workers":
        cmd = [*base, "run-workers", str(runtime_config), "--interval", str(args.interval)]
        if args.category:
            cmd += ["--category", args.category]
        if args.metrics_port:
            cmd += ["--metrics-port", str(args.metrics_port)]
        else:
            cmd += ["--metrics-port", "0"]
        if args.lease_name:
            cmd += ["--lease-name", args.lease_name]
            if args.lease_owner:
                cmd += ["--lease-owner", args.lease_owner]
        return run_command(cmd)

    if args.role == "snapshot":
        if args.snapshot_index is None:
            raise SystemExit("--snapshot-index is required for role=snapshot")
        cmd = [*base, "snapshot", str(runtime_config), "--snapshot-index", str(args.snapshot_index)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        return run_command(cmd)

    if args.role == "inventory-rebuild":
        cmd = [*base, "inventory-rebuild", str(runtime_config)]
        if args.category:
            cmd += ["--category", args.category]
        return run_command(cmd)

    return run_command([*base, "status", str(runtime_config)])


if __name__ == "__main__":
    raise SystemExit(main())
