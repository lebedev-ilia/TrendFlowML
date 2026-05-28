from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _guess_content_type(path: str) -> Optional[str]:
    p = path.lower()
    if p.endswith(".json"):
        return "application/json"
    if p.endswith(".jsonl"):
        return "application/x-ndjson"
    if p.endswith(".html"):
        return "text/html"
    if p.endswith(".txt") or p.endswith(".log") or p.endswith(".md"):
        return "text/plain"
    if p.endswith(".npz"):
        return "application/octet-stream"
    if p.endswith(".png"):
        return "image/png"
    if p.endswith(".jpg") or p.endswith(".jpeg"):
        return "image/jpeg"
    if p.endswith(".svg"):
        return "image/svg+xml"
    return None


def _iter_run_dirs(results_root: str, *, max_runs: Optional[int]) -> Iterable[str]:
    found = 0
    for root, _dirs, files in os.walk(results_root):
        if "manifest.json" in files:
            yield root
            found += 1
            if max_runs is not None and found >= max_runs:
                return


@dataclass(frozen=True)
class RunIdentity:
    platform_id: str
    video_id: str
    run_id: str


def _read_run_identity(run_dir: str) -> Optional[RunIdentity]:
    mp = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(mp):
        return None
    m = _load_json(mp)
    run = m.get("run") if isinstance(m.get("run"), dict) else {}
    platform_id = str(run.get("platform_id") or "").strip()
    video_id = str(run.get("video_id") or "").strip()
    run_id = str(run.get("run_id") or "").strip()
    if not platform_id or not video_id or not run_id:
        return None
    return RunIdentity(platform_id=platform_id, video_id=video_id, run_id=run_id)


def _copy_run_dir(
    *,
    run_dir: str,
    storage: Any,
    key_layout: Any,
    index_jsonl_path: Optional[str],
    skip_existing: bool,
) -> Tuple[int, int]:
    ident = _read_run_identity(run_dir)
    if ident is None:
        raise RuntimeError(f"cannot parse run identity from manifest.json in {run_dir}")

    dst_prefix = key_layout.result_store_run_prefix(ident.platform_id, ident.video_id, ident.run_id)

    uploaded = 0
    skipped = 0

    run_dir_abs = os.path.abspath(run_dir)
    for root, _dirs, files in os.walk(run_dir_abs):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, run_dir_abs).replace(os.sep, "/")
            key = f"{dst_prefix}/{rel}".lstrip("/")

            if skip_existing and storage.exists(key):
                skipped += 1
                continue

            data = Path(src).read_bytes()
            storage.atomic_write_bytes(key, data, content_type=_guess_content_type(src))
            uploaded += 1

    if index_jsonl_path:
        os.makedirs(os.path.dirname(index_jsonl_path) or ".", exist_ok=True)
        line = json.dumps(
            {
                "platform_id": ident.platform_id,
                "video_id": ident.video_id,
                "run_id": ident.run_id,
                "storage_run_prefix": dst_prefix,
                "source_run_dir": run_dir_abs,
            },
            ensure_ascii=False,
        )
        with open(index_jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return uploaded, skipped


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sync local dp_results runs into Storage (FS or S3/MinIO).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", default="", help="Path to a single run dir containing manifest.json")
    src.add_argument("--results-root", default="", help="Scan this directory recursively for manifest.json and sync all runs")
    p.add_argument("--max-runs", type=int, default=0, help="Safety cap when scanning (0 = no cap)")
    p.add_argument("--skip-existing", action="store_true", help="Skip objects that already exist in Storage")
    p.add_argument(
        "--index-jsonl",
        default="",
        help="Optional local JSONL file to append one line per synced run (for later DB import).",
    )
    p.add_argument(
        "--no-write-index-objects",
        action="store_true",
        help="Disable writing per-run index objects into Storage under result_store/_indexes/runs/...",
    )

    args = p.parse_args(argv)

    # Make DataProcessor/ importable as top-level modules (storage.*, state.*, ...)
    dp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if dp_root not in sys.path:
        sys.path.insert(0, dp_root)

    # Use the same Storage settings/env as the API (TREND_STORAGE_BACKEND, S3_ENDPOINT, etc.)
    from storage.settings import load_storage_settings
    from storage.fs import FileSystemStorage
    from storage.s3 import S3Storage
    from storage.paths import KeyLayout

    settings = load_storage_settings()
    storage = (
        S3Storage(endpoint_url=settings.s3_endpoint, bucket=settings.s3_bucket, region=settings.aws_region)
        if settings.backend == "s3"
        else FileSystemStorage(root_dir=settings.fs_root)
    )
    key_layout = KeyLayout(prefix=settings.s3_prefix if settings.backend == "s3" else "")

    index_jsonl_path = args.index_jsonl.strip() or None
    skip_existing = bool(args.skip_existing)

    run_dirs: List[str] = []
    if args.run_dir:
        run_dirs = [args.run_dir]
    else:
        max_runs = int(args.max_runs) if int(args.max_runs) > 0 else None
        run_dirs = list(_iter_run_dirs(args.results_root, max_runs=max_runs))

    total_uploaded = 0
    total_skipped = 0
    for rd in run_dirs:
        u, s = _copy_run_dir(
            run_dir=rd,
            storage=storage,
            key_layout=key_layout,
            index_jsonl_path=index_jsonl_path,
            skip_existing=skip_existing,
        )
        total_uploaded += u
        total_skipped += s
        ident = _read_run_identity(rd)
        ident_s = f"{ident.platform_id}/{ident.video_id}/{ident.run_id}" if ident else os.path.abspath(rd)
        print(f"synced {ident_s}: uploaded={u} skipped={s}")

        # Write per-run index object (S3-friendly alternative to JSONL appends).
        if ident and not args.no_write_index_objects:
            index_key = (
                f"{key_layout.result_store_prefix()}/_indexes/runs/"
                f"{ident.platform_id}/{ident.video_id}/{ident.run_id}.json"
            ).lstrip("/")
            payload = {
                "platform_id": ident.platform_id,
                "video_id": ident.video_id,
                "run_id": ident.run_id,
                "storage_run_prefix": key_layout.result_store_run_prefix(ident.platform_id, ident.video_id, ident.run_id),
                "source_run_dir": os.path.abspath(rd),
            }
            storage.atomic_write_bytes(index_key, json.dumps(payload, ensure_ascii=False).encode("utf-8"), content_type="application/json")

    print(f"TOTAL uploaded={total_uploaded} skipped={total_skipped} backend={settings.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

