from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def numeric_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values = sorted(values)

    def pct(p: float) -> int:
        return values[min(len(values) - 1, max(0, round((p / 100) * (len(values) - 1))))]

    return {
        "count": len(values),
        "min": values[0],
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": values[-1],
        "mean": round(sum(values) / len(values), 2),
    }


def audit(root: Path) -> dict[str, Any]:
    metadata_files = sorted((root / "shards" / "metadata").glob("**/part_*.json"))
    enrich_files = sorted((root / "shards" / "enrich").glob("**/part_*.json"))
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    nums: dict[str, list[int]] = defaultdict(list)
    channels: Counter[str] = Counter()
    accepted = 0

    for path in metadata_files:
        category = next((part.split("=", 1)[1] for part in path.parts if part.startswith("category=")), "unknown")
        data = read_json(path)
        rows = data.values() if isinstance(data, dict) else data
        if not isinstance(rows, list) and not hasattr(rows, "__iter__"):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            accepted += 1
            metadata = row.get("metadata") or {}
            snapshot = row.get("snapshot_0") or {}
            counts["category"][category] += 1
            counts["language"][str(metadata.get("language") or "unknown")] += 1
            counts["country"][str(metadata.get("country") or "unknown")] += 1
            counts["time_interval"][str(row.get("time_interval") or "unknown")] += 1
            channel_id = metadata.get("channel_id") or metadata.get("channelId") or metadata.get("channelTitle") or "unknown"
            channels[str(channel_id)] += 1
            for name, raw in {
                "duration_seconds": metadata.get("duration_seconds") or metadata.get("duration"),
                "view_count": snapshot.get("viewCount") or metadata.get("view_count"),
                "like_count": snapshot.get("likeCount") or metadata.get("like_count"),
                "comment_count": snapshot.get("commentCount") or metadata.get("comment_count"),
                "subscriber_count": snapshot.get("subscriberCount"),
            }.items():
                value = to_int(raw)
                if value is not None:
                    nums[name].append(value)

    enrich_count = 0
    subtitle_presence = Counter()
    resolutions = Counter()
    for path in enrich_files:
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        for payload in data.values():
            if not isinstance(payload, dict):
                continue
            enrich_count += 1
            formats = payload.get("formats") or []
            best = (0, 0, "unknown")
            for item in formats:
                if not isinstance(item, dict):
                    continue
                resolution = item.get("resolution")
                width = height = 0
                if isinstance(resolution, str) and "x" in resolution:
                    left, right = resolution.lower().split("x", 1)
                    width = to_int(left) or 0
                    height = to_int(right) or 0
                else:
                    width = to_int(item.get("width")) or 0
                    height = to_int(item.get("height")) or 0
                if width * height > best[0] * best[1]:
                    best = (width, height, str(resolution or f"{width}x{height}"))
            resolutions[best[2]] += 1
            langs = set()
            for key in ("subtitles", "automatic_captions"):
                block = payload.get(key) or {}
                if isinstance(block, dict):
                    langs.update(str(lang) for lang, value in block.items() if value)
            subtitle_presence["with_subtitles" if langs else "without_subtitles"] += 1

    summary = read_json(root / "state" / "inventory" / "summary.json") or {}
    rejected = Counter()
    for path in sorted((root / "rejected").glob("part_*.json")):
        rows = read_json(path)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    rejected[str(row.get("reason") or "unknown")] += 1

    hf_commit_rows = read_jsonl(root / "state" / "hf_commit_log.jsonl")
    commits_by_repo = Counter(str(row.get("repo_id") or "unknown") for row in hf_commit_rows)

    top_channel_count = channels.most_common(1)[0][1] if channels else 0
    return {
        "root": str(root),
        "accepted_metadata": accepted,
        "metadata_shards": len(metadata_files),
        "enrich_records": enrich_count,
        "enrich_shards": len(enrich_files),
        "top_counts": {key: value.most_common(20) for key, value in counts.items()},
        "numeric_stats": {key: numeric_stats(value) for key, value in nums.items()},
        "rejected_reasons": rejected.most_common(),
        "enrich": {
            "resolution": resolutions.most_common(20),
            "subtitle_presence": subtitle_presence.most_common(),
        },
        "channels": {
            "unique": len(channels),
            "top_count": top_channel_count,
            "top_share": round(top_channel_count / accepted, 4) if accepted else 0,
        },
        "hf_commits_by_repo": commits_by_repo.most_common(),
        "queue_failures": len(read_jsonl(root / "state" / "queue_failures.jsonl")),
        "queue_dead_letter": len(read_jsonl(root / "state" / "queue_dead_letter.jsonl")),
        "inventory_summary": summary.get("totals", summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a dataset collector run after smoke/checkpoint/full collection.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--check-hf",
        action="store_true",
        help="Also compare local state with Hugging Face repos (requires huggingface_hub and HF_TOKEN).",
    )
    args = parser.parse_args()
    result = audit(args.run_dir)
    if args.check_hf:
        import importlib.util

        compare_path = Path(__file__).resolve().parent / "compare_hf_run_state.py"
        spec = importlib.util.spec_from_file_location("compare_hf_run_state", compare_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {compare_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result["hf_compare"] = module.compare_run(args.run_dir, check_hf=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
