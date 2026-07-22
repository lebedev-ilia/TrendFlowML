#!/usr/bin/env python3
"""
s3_corpus.py — RunPod S3 helper for Agent A's corpus_out NPZ (Network Volume vuiq0iq3yf).

Agent A's 300-video corpus run wrote NPZ directly to the 120GB Network Volume
(EU-RO-1) under `corpus_out/<video_id>/rs/<component>/*.npz`, WITHOUT a per-run
manifest.json. This module reads them over RunPod's S3-compatible API (no pod needed).

Credentials are read from `storage/.s3creds` (gitignored) or env vars
(S3_AK/S3_SK/S3_ENDPOINT/S3_REGION/S3_BUCKET).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]
CREDS_FILE = REPO_ROOT / "storage" / ".s3creds"


def _load_creds() -> Dict[str, str]:
    c: Dict[str, str] = {}
    if CREDS_FILE.exists():
        for line in CREDS_FILE.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                c[k.strip()] = v.strip()
    return {
        "AK": os.environ.get("S3_AK", c.get("AK", "")),
        "SK": os.environ.get("S3_SK", c.get("SK", "")),
        "ENDPOINT": os.environ.get("S3_ENDPOINT", c.get("ENDPOINT", "https://s3api-eu-ro-1.runpod.io")),
        "REGION": os.environ.get("S3_REGION", c.get("REGION", "eu-ro-1")),
        "BUCKET": os.environ.get("S3_BUCKET", c.get("BUCKET", "vuiq0iq3yf")),
    }


def client():
    import boto3
    from botocore.config import Config

    c = _load_creds()
    cli = boto3.client(
        "s3", endpoint_url=c["ENDPOINT"], aws_access_key_id=c["AK"],
        aws_secret_access_key=c["SK"], region_name=c["REGION"],
        config=Config(signature_version="s3v4", connect_timeout=20,
                      read_timeout=180, retries={"max_attempts": 3}),
    )
    return cli, c["BUCKET"]


def list_videos_under_prefix(s3, bucket: str, prefix: str = "corpus_out") -> List[str]:
    """Enumerate video_ids (dir names) under <prefix>/ on the volume."""
    pag = s3.get_paginator("list_objects_v2")
    vids: List[str] = []
    for page in pag.paginate(Bucket=bucket, Prefix=f"{prefix}/", Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            vids.append(p["Prefix"].split("/")[1])
    return vids


def list_video_npz(s3, bucket: str, vid: str, include_depth: bool = False,
                   prefix: str = "corpus_out") -> List[dict]:
    """Return [{key,size}] of rs/*.npz for one video."""
    r = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/{vid}/rs/")
    out = []
    for o in r.get("Contents", []):
        k = o["Key"]
        if not k.endswith(".npz"):
            continue
        if not include_depth and "/core_depth_midas/" in k:
            continue
        out.append({"key": k, "size": o["Size"]})
    return out


def download_video(s3, bucket: str, vid: str, dest_root: Path,
                   include_depth: bool = False) -> int:
    """Download a video's rs/*.npz under dest_root/<vid>/rs/... Returns files fetched."""
    n = 0
    for o in list_video_npz(s3, bucket, vid, include_depth=include_depth):
        rel = o["key"].split(f"{vid}/", 1)[-1]  # rs/<comp>/<file>.npz
        local = dest_root / vid / rel
        if local.exists() and local.stat().st_size == o["size"]:
            n += 1
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, o["key"], str(local))
        n += 1
    return n


import threading

_TLS = threading.local()


def _tls_client():
    if not hasattr(_TLS, "s3"):
        _TLS.s3, _TLS.bucket = client()
    return _TLS.s3, _TLS.bucket


def download_corpus_fast(video_ids: List[str], dest_root: Path, *,
                         include_depth: bool = False, list_workers: int = 24,
                         dl_workers: int = 32, prefix: str = "corpus_out") -> Dict[str, int]:
    """File-level parallel download with thread-local (reused) clients.
    1) list all videos in parallel -> flat file task list
    2) download all files in parallel, skipping already-complete ones."""
    dest_root.mkdir(parents=True, exist_ok=True)

    # ---- parallel listing ----
    def _list(vid: str):
        s3, bk = _tls_client()
        try:
            return vid, list_video_npz(s3, bk, vid, include_depth=include_depth, prefix=prefix)
        except Exception as e:  # noqa: BLE001
            return vid, f"ERR:{type(e).__name__}"

    tasks = []  # (vid, key, size)
    listed = 0
    with ThreadPoolExecutor(max_workers=list_workers) as ex:
        for vid, res in ex.map(_list, video_ids):
            listed += 1
            if isinstance(res, str):
                continue
            for o in res:
                tasks.append((vid, o["key"], o["size"]))
    print(f"  listed {listed} videos -> {len(tasks)} files", flush=True)

    # ---- parallel download ----
    results: Dict[str, int] = {v: 0 for v in video_ids}
    lock = threading.Lock()

    def _dl(t):
        vid, key, size = t
        rel = key.split(f"{vid}/", 1)[-1]
        local = dest_root / vid / rel
        if local.exists() and local.stat().st_size == size:
            return vid, True
        local.parent.mkdir(parents=True, exist_ok=True)
        s3, bk = _tls_client()
        try:
            s3.download_file(bk, key, str(local))
            return vid, True
        except Exception:
            return vid, False

    done = 0
    with ThreadPoolExecutor(max_workers=dl_workers) as ex:
        for vid, ok in ex.map(_dl, tasks):
            done += 1
            if ok:
                with lock:
                    results[vid] += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(tasks)} files", flush=True)
    return results


def download_corpus(video_ids: List[str], dest_root: Path, *,
                    include_depth: bool = False, workers: int = 8) -> Dict[str, int]:
    s3, bucket = client()
    dest_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, int] = {}

    def _one(vid: str) -> tuple:
        # each thread its own client (botocore clients are not thread-safe for all ops)
        cli, bk = client()
        try:
            return vid, download_video(cli, bk, vid, dest_root, include_depth=include_depth)
        except Exception as e:  # noqa: BLE001
            return vid, f"ERR:{type(e).__name__}:{e}"

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, v): v for v in video_ids}
        for f in as_completed(futs):
            vid, res = f.result()
            results[vid] = res
            done += 1
            if done % 25 == 0:
                ok = sum(1 for v in results.values() if isinstance(v, int))
                print(f"  ... {done}/{len(video_ids)} videos (ok={ok})", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Download <prefix>/<vid>/rs/*.npz from RunPod S3")
    ap.add_argument("--corpus-json", default=str(REPO_ROOT / "DataProcessor/docs/corpus_run_report/corpus300.json"))
    ap.add_argument("--prefix", default="corpus_out", help="S3 volume prefix (e.g. corpus_out, corpus_smoke)")
    ap.add_argument("--from-s3", action="store_true", help="enumerate video_ids from <prefix>/ on S3 instead of --corpus-json")
    ap.add_argument("--dest", default=str(REPO_ROOT / "storage/corpus_npz"))
    ap.add_argument("--include-depth", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.from_s3:
        s3, bk = client()
        vids = list_videos_under_prefix(s3, bk, prefix=args.prefix)
    else:
        vids = [c["video_id"] for c in json.loads(Path(args.corpus_json).read_text())]
    if args.limit:
        vids = vids[: args.limit]
    print(f"downloading {len(vids)} videos from '{args.prefix}/' -> {args.dest} (depth={args.include_depth})")
    res = download_corpus_fast(vids, Path(args.dest), include_depth=args.include_depth, prefix=args.prefix)
    ok = sum(1 for v in res.values() if isinstance(v, int))
    errs = {k: v for k, v in res.items() if not isinstance(v, int)}
    print(f"[done] ok={ok}/{len(vids)} files_total={sum(v for v in res.values() if isinstance(v,int))}")
    if errs:
        print(f"[errors] {len(errs)}: {list(errs.items())[:5]}")
