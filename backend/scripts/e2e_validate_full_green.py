#!/usr/bin/env python3
"""
Проверка критериев полного зелёного E2E (backend/docs/E2E_RUNBOOK.md §0.1).

Usage:
  python backend/scripts/e2e_validate_full_green.py \\
    --run-id b166b63c-3424-4870-8ebd-b8c9a78736ec \\
    --platform-id youtube --video-id -Q6fnPIybEI

  python backend/scripts/e2e_validate_full_green.py --latest-e2e-artifact
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _storage_root(repo: Path) -> Path:
    return repo / "storage"


def _find_run_id_in_dir(artifact_dir: Path) -> Optional[str]:
    pat = re.compile(r'"run_id":\s*"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"')
    for log in sorted(artifact_dir.rglob("*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def _latest_e2e_artifact(repo: Path) -> Path:
    base = _storage_root(repo) / "e2e_full_max"
    dirs = [p for p in base.iterdir() if p.is_dir() and p.name != "active_global_config"]
    if not dirs:
        raise FileNotFoundError(f"No e2e_full_max artifacts in {base}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def _check_fetcher_7of7(events_path: Path) -> Tuple[bool, str]:
    if not events_path.is_file():
        return True, "skip (no orchestrator_events.jsonl)"
    last_fetcher: Optional[Dict[str, Any]] = None
    for line in events_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        fetcher = ev.get("fetcher")
        if isinstance(fetcher, dict):
            last_fetcher = fetcher
    if not last_fetcher:
        return True, "skip (no fetcher events)"
    cs = last_fetcher.get("completed_stages") or []
    tot = last_fetcher.get("total_stages")
    if isinstance(cs, list) and tot is not None and len(cs) >= int(tot):
        return True, f"fetcher {len(cs)}/{tot} COMPLETED"
    return False, f"fetcher incomplete: {len(cs) if isinstance(cs, list) else 0}/{tot}"


def validate(
    *,
    repo: Path,
    run_id: str,
    platform_id: str,
    video_id: str,
    e2e_artifact_dir: Optional[Path] = None,
) -> List[str]:
    errors: List[str] = []
    rs_path = _storage_root(repo) / "state" / platform_id / video_id / run_id / "run_state.json"
    mf_path = (
        _storage_root(repo) / "result_store" / platform_id / video_id / run_id / "manifest.json"
    )

    if not rs_path.is_file():
        errors.append(f"missing run_state: {rs_path}")
    if not mf_path.is_file():
        errors.append(f"missing manifest: {mf_path}")

    if mf_path.is_file():
        manifest = _load_json(mf_path)
        run = manifest.get("run") or {}
        if run.get("status") != "success":
            errors.append(f"manifest run.status={run.get('status')!r} (expected success)")
        for comp in manifest.get("components") or []:
            if not isinstance(comp, dict):
                continue
            st = comp.get("status")
            name = comp.get("name", "?")
            if st == "error":
                errors.append(f"manifest component {name}: status=error ({comp.get('error')})")

    if rs_path.is_file():
        rs = _load_json(rs_path)
        procs = (rs.get("processors") or {})
        for pname in ("segmenter", "audio", "visual", "text"):
            pdata = procs.get(pname) or {}
            if pdata.get("status") != "success":
                errors.append(f"run_state processor {pname}: status={pdata.get('status')!r}")
        vis = procs.get("visual") or {}
        vcomps = vis.get("components") or {}
        if isinstance(vcomps, dict):
            for cname, cdata in vcomps.items():
                if isinstance(cdata, dict) and cdata.get("status") == "error":
                    errors.append(f"run_state visual.{cname}: error")

    if e2e_artifact_dir:
        events = e2e_artifact_dir / "orchestrator_events.jsonl"
        ok, msg = _check_fetcher_7of7(events)
        if not ok:
            errors.append(msg)
        else:
            print(f"OK: {msg}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate full green E2E criteria (§0.1)")
    ap.add_argument("--run-id")
    ap.add_argument("--platform-id", default="youtube")
    ap.add_argument("--video-id", default="-Q6fnPIybEI")
    ap.add_argument(
        "--latest-e2e-artifact",
        action="store_true",
        help="Resolve run_id from newest storage/e2e_full_max/*",
    )
    ap.add_argument("--e2e-artifact-dir", type=Path, default=None)
    args = ap.parse_args()

    repo = _repo_root()
    artifact_dir = args.e2e_artifact_dir

    if args.latest_e2e_artifact:
        artifact_dir = artifact_dir or _latest_e2e_artifact(repo)
        run_id = args.run_id or _find_run_id_in_dir(artifact_dir)
        if not run_id:
            print(f"FAIL: could not find run_id in {artifact_dir}", file=sys.stderr)
            return 2
    else:
        run_id = args.run_id
        if not run_id:
            print("FAIL: --run-id or --latest-e2e-artifact required", file=sys.stderr)
            return 2

    print(f"Validating run_id={run_id} platform={args.platform_id} video={args.video_id}")
    if artifact_dir:
        print(f"E2E artifact dir: {artifact_dir}")

    errors = validate(
        repo=repo,
        run_id=run_id,
        platform_id=args.platform_id,
        video_id=args.video_id,
        e2e_artifact_dir=artifact_dir,
    )

    if errors:
        print("FAIL: full green validation:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("PASS: full green E2E (§0.1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
