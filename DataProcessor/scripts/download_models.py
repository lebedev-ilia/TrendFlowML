#!/usr/bin/env python3
"""
Download all TrendFlow models/artifacts from the unified Hugging Face repo and
lay them out into their canonical local paths.

Source of truth: ``configs/models_manifest.json`` (repo root). Every manifest
entry ``path`` is BOTH the path inside the HF dataset AND the local path
relative to the TrendFlowML repo root, so this script is essentially a verified
mirror download.

Design goals:
  * zero third-party deps (stdlib only) -> works before any pip install
  * idempotent: skip files already present with matching sha256
  * integrity: verify sha256 after every download (LFS oid == sha256)
  * parallel downloads, atomic writes, resumable (re-run to finish)
  * private-repo friendly: token from HF_TOKEN / HUGGINGFACE_HUB_TOKEN

Examples:
  # dry-run: show what would be downloaded
  python DataProcessor/scripts/download_models.py --dry-run

  # download everything (token only needed if repo is private)
  HF_TOKEN=hf_xxx python DataProcessor/scripts/download_models.py

  # only audio + visual weights (skip the 409 semantic-base images)
  python DataProcessor/scripts/download_models.py --groups audio visual

  # pin a versioned snapshot
  python DataProcessor/scripts/download_models.py --revision v1.0.0
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

HF_BASE = "https://huggingface.co"


def repo_root_default() -> Path:
    # .../TrendFlowML/DataProcessor/scripts/download_models.py -> TrendFlowML
    return Path(__file__).resolve().parents[2]


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}" if u != "B" else f"{n}B"
        n /= 1024


def token_from_env() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_url(repo_id: str, repo_type: str, revision: str, rel_path: str) -> str:
    kind = "datasets" if repo_type == "dataset" else ("models" if repo_type == "model" else repo_type + "s")
    quoted = urllib.parse.quote(rel_path)
    return f"{HF_BASE}/{kind}/{repo_id}/resolve/{revision}/{quoted}"


def download_one(entry: dict, *, repo_id, repo_type, revision, root: Path,
                 token: str | None, force: bool, verify: bool) -> tuple[str, str]:
    """Returns (status, path). status in {ok, skip, fail}."""
    rel = entry["path"]
    dest = root / rel
    sha = entry.get("sha256")
    size = entry.get("size", 0)

    if dest.exists() and not force:
        if not verify or not sha:
            return ("skip", rel)
        if dest.stat().st_size == size and sha256_file(dest) == sha:
            return ("skip", rel)
        # present but mismatched -> redownload

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Per-entry source repo override (multi-source manifest); falls back to manifest repo.
    eff_repo = entry.get("repo") or repo_id
    eff_type = entry.get("repo_type") or repo_type
    url = resolve_url(eff_repo, eff_type, revision, rel)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=".dl_")
                try:
                    with os.fdopen(fd, "wb") as out:
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                    if verify and sha:
                        got = sha256_file(Path(tmp))
                        if got != sha:
                            os.unlink(tmp)
                            return ("fail", f"{rel} (sha mismatch: {got[:12]}!={sha[:12]})")
                    os.replace(tmp, dest)
                    return ("ok", rel)
                except Exception:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return ("fail", f"{rel} (HTTP {e.code} - set HF_TOKEN for private repo)")
            last_err = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    return ("fail", f"{rel} ({last_err})")


def main() -> int:
    root = repo_root_default()
    ap = argparse.ArgumentParser(description="Download TrendFlow models from the unified HF repo.")
    ap.add_argument("--manifest", default=str(root / "configs" / "models_manifest.json"))
    ap.add_argument("--repo-root", default=str(root))
    ap.add_argument("--repo-id", default=None, help="override manifest repo_id")
    ap.add_argument("--revision", default=None, help="branch/tag/sha (default: manifest revision)")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="subset of groups to fetch (e.g. audio visual semantic_bases)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="redownload even if present")
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 verification")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    root = Path(args.repo_root)
    repo_id = args.repo_id or manifest["repo_id"]
    repo_type = manifest.get("repo_type", "dataset")
    revision = args.revision or manifest.get("revision", "main")
    verify = not args.no_verify
    token = token_from_env()

    entries = manifest["entries"]
    if args.groups:
        wanted = set(args.groups)
        entries = [e for e in entries if e.get("group") in wanted]

    total_bytes = sum(e.get("size", 0) for e in entries)
    print(f"[download_models] repo={repo_id}@{revision}  root={root}")
    print(f"[download_models] entries={len(entries)}  size={human(total_bytes)}  "
          f"jobs={args.jobs}  verify={verify}  token={'yes' if token else 'no'}")

    if args.dry_run:
        by_group: dict[str, list[int]] = {}
        present = 0
        for e in entries:
            g = e.get("group", "other")
            by_group.setdefault(g, [0, 0])
            by_group[g][0] += 1
            by_group[g][1] += e.get("size", 0)
            if (root / e["path"]).exists():
                present += 1
        for g, (c, b) in sorted(by_group.items()):
            print(f"  - {g:<16} {c:>4} files  {human(b)}")
        print(f"[dry-run] already present locally: {present}/{len(entries)}")
        return 0

    ok = skip = fail = 0
    failures: list[str] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(download_one, e, repo_id=repo_id, repo_type=repo_type,
                          revision=revision, root=root, token=token,
                          force=args.force, verify=verify) for e in entries]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            status, rel = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failures.append(rel)
            if done % 25 == 0 or done == len(entries):
                print(f"  ... {done}/{len(entries)}  ok={ok} skip={skip} fail={fail}", flush=True)

    dt = time.time() - t0
    print(f"[download_models] done in {dt:.1f}s  ok={ok} skip={skip} fail={fail}")
    if failures:
        print("[download_models] FAILURES:")
        for f in failures[:50]:
            print("   -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
