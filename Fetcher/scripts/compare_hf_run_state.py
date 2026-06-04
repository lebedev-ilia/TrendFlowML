#!/usr/bin/env python3
"""Compare local dataset run state with Hugging Face dataset repos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_metadata_video_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    meta_root = root / "shards" / "metadata"
    if not meta_root.exists():
        return ids
    for path in meta_root.glob("category=*/part_*.json"):
        if path.name.endswith(".tmp"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            ids.update(data.keys())
    return ids


def load_done_video_ids(state_dir: Path) -> set[str]:
    done: set[str] = set()
    for row in read_jsonl(state_dir / "hf_video_upload_done.jsonl"):
        vid = row.get("video_id")
        if vid:
            done.add(str(vid))
    return done


def hf_video_ids(repo_id: str) -> tuple[set[str], list[str]]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    ids: set[str] = set()
    pollution: list[str] = []
    for rel in files:
        if rel.startswith("state/") or "/coordination/" in rel:
            pollution.append(rel)
        if rel.endswith(".mp4"):
            ids.add(Path(rel).stem)
    return ids, pollution


def hf_shard_paths(repo_id: str) -> tuple[list[str], list[str]]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    shards = [f for f in files if f.startswith("shards/metadata/") and f.endswith(".json")]
    pollution = [f for f in files if f.startswith("state/") or "/coordination/" in f]
    return shards, pollution


def find_run_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "manifest.json").exists():
        return path
    if (path / "state" / "inventory" / "summary.json").exists():
        return path
    for child in path.iterdir():
        if child.is_dir() and (child / "manifest.json").exists():
            return child
    return path


def compare_run(root: Path, *, check_hf: bool = True) -> dict[str, Any]:
    root = find_run_root(root)
    runtime_files = sorted(root.glob("runtime_*.json"))
    runtime = {}
    if runtime_files:
        runtime = json.loads(runtime_files[0].read_text(encoding="utf-8"))

    meta_ids = load_metadata_video_ids(root)
    done_ids = load_done_video_ids(root / "state")
    local_mp4 = {
        p.stem
        for p in (root / "downloads" / "videos").rglob("*.mp4")
        if p.is_file()
    } if (root / "downloads" / "videos").exists() else set()

    report: dict[str, Any] = {
        "root": str(root),
        "runtime": {
            "hf_videos_repo_id": runtime.get("hf_videos_repo_id"),
            "hf_shards_repo_id": runtime.get("hf_shards_repo_id"),
            "hf_enrich_repo_id": runtime.get("hf_enrich_repo_id"),
            "hf_parallel_colab_count": runtime.get("hf_parallel_colab_count"),
        },
        "local": {
            "metadata_video_ids": len(meta_ids),
            "hf_video_upload_done": len(done_ids),
            "local_mp4_files": len(local_mp4),
            "done_not_in_metadata": len(done_ids - meta_ids),
            "local_mp4_not_done": len(local_mp4 - done_ids),
            "metadata_not_done": len(meta_ids - done_ids),
        },
    }

    if not check_hf:
        return report

    videos_repo = runtime.get("hf_videos_repo_id") or runtime.get("hf_repo_id")
    shards_repo = runtime.get("hf_shards_repo_id") or runtime.get("hf_repo_id")
    if videos_repo:
        hf_ids, mp4_pollution = hf_video_ids(videos_repo)
        report["hf_videos"] = {
            "repo_id": videos_repo,
            "mp4_count": len(hf_ids),
            "only_hf_not_local_done": len(hf_ids - done_ids),
            "only_local_done_not_hf": len(done_ids - hf_ids),
            "metadata_not_on_hf": len(meta_ids - hf_ids),
        }
    if shards_repo:
        shard_paths, shard_pollution = hf_shard_paths(shards_repo)
        local_shards = [
            str(p.relative_to(root)).replace("\\", "/")
            for p in (root / "shards" / "metadata").glob("**/part_*.json")
            if not p.name.endswith(".tmp")
        ]
        report["hf_shards"] = {
            "repo_id": shards_repo,
            "shard_files_on_hf": len(shard_paths),
            "local_shard_files": len(local_shards),
            "coord_pollution_files": shard_pollution,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="dataset_runs output dir or full_results export folder")
    parser.add_argument("--out", type=Path, help="Write JSON report to this path")
    parser.add_argument("--no-hf", action="store_true", help="Skip Hugging Face API calls")
    args = parser.parse_args(argv)

    report = compare_run(args.run_dir, check_hf=not args.no_hf)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
