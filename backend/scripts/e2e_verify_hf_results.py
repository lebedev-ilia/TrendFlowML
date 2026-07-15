#!/usr/bin/env python3
"""
Повторная §0.1/§0.2 проверка сохранённых HF videos11 прогонов без полного E2E.

Читает backend/.e2e/state/hf_videos11_results.json и для каждой записи с run_id
запускает e2e_validate_full_green.py и e2e_validate_output_quality.py.

Usage:
  cd backend && source scripts/e2e_env.sh && source .venv/bin/activate
  python scripts/e2e_verify_hf_results.py
  python scripts/e2e_verify_hf_results.py --results path/to/hf_videos11_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest_durations(manifest_path: Path) -> Dict[str, float]:
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[str, float] = {}
    for v in data.get("videos") or []:
        if isinstance(v, dict) and v.get("video_id"):
            try:
                out[str(v["video_id"])] = float(v.get("duration_sec") or 0)
            except (TypeError, ValueError):
                pass
    return out


def _verify_audio_duration(root: Path, video_id: str, expected_sec: float) -> Optional[str]:
    if expected_sec <= 0:
        return None
    meta = root / "storage" / "frames_dir" / video_id / "audio" / "metadata.json"
    if not meta.is_file():
        return f"segmenter metadata missing: {meta}"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        dur = float(data.get("duration_sec") or 0)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return f"invalid segmenter metadata: {meta}"
    if dur < expected_sec * 0.85:
        return f"segmenter audio duration {dur:.2f}s << expected {expected_sec:.2f}s"
    return None


def main() -> int:
    root = _repo_root()
    backend = root / "backend"
    ap = argparse.ArgumentParser(description="Re-verify HF videos11 E2E results (§0.1 + §0.2)")
    ap.add_argument(
        "--results",
        type=Path,
        default=backend / ".e2e" / "state" / "hf_videos11_results.json",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=root / "example" / "hf_videos11" / "manifest.json",
    )
    ap.add_argument("--skip-core-identity", action="store_true", default=True)
    ap.add_argument("--with-core-identity", action="store_true")
    ap.add_argument("--strict-quality", action="store_true", help="Treat §0.2 warnings as errors")
    ap.add_argument(
        "--skip-missing-runs",
        action="store_true",
        help="Skip entries when storage/result_store run dir is absent (CI without artifacts)",
    )
    args = ap.parse_args()

    skip_ci = not args.with_core_identity
    if not args.results.is_file():
        print(f"FAIL: results file not found: {args.results}", file=sys.stderr)
        return 2

    try:
        payload = json.loads(args.results.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"FAIL: cannot read {args.results}: {e}", file=sys.stderr)
        return 2

    results: List[Dict[str, Any]] = [
        r for r in (payload.get("results") or []) if isinstance(r, dict)
    ]
    if not results:
        print("FAIL: no results entries", file=sys.stderr)
        return 2

    durations = _load_manifest_durations(args.manifest)
    quality_py = backend / "scripts" / "e2e_validate_output_quality.py"
    green_py = backend / "scripts" / "e2e_validate_full_green.py"
    dp_py = root / "DataProcessor" / ".data_venv" / "bin" / "python"
    py = sys.executable

    worst = 0
    print(f"Re-verifying {len(results)} HF run(s) from {args.results}\n")

    for entry in results:
        video_id = str(entry.get("video_id") or "")
        run_id = str(entry.get("run_id") or "")
        if not video_id or not run_id:
            print(f"SKIP {video_id or '?'}: missing run_id")
            worst = max(worst, 2)
            continue

        print(f"=== {video_id} run_id={run_id} ===")
        code = 0

        run_dir = root / "storage" / "result_store" / "youtube" / video_id / run_id
        if args.skip_missing_runs and not run_dir.is_dir():
            print(f"  SKIP: run dir missing ({run_dir})")
            print()
            continue

        expected = durations.get(video_id, 0.0)
        audio_err = _verify_audio_duration(root, video_id, expected)
        if audio_err:
            print(f"  FAIL audio_check: {audio_err}")
            code = 1
        else:
            print("  audio_check: ok")

        g_cmd = [py, str(green_py), "--run-id", run_id, "--video-id", video_id]
        g = subprocess.run(g_cmd, cwd=str(backend))
        print(f"  green_exit: {g.returncode}")
        if g.returncode != 0:
            code = g.returncode

        if dp_py.is_file():
            q_cmd = [
                str(dp_py),
                str(quality_py),
                "--run-id",
                run_id,
                "--video-id",
                video_id,
                "--real-video",
            ]
            if skip_ci:
                q_cmd.append("--skip-core-identity")
            if args.strict_quality:
                q_cmd.append("--strict")
            q = subprocess.run(q_cmd, cwd=str(backend))
            print(f"  quality_exit: {q.returncode}")
            if q.returncode != 0:
                code = q.returncode
        else:
            print(f"  WARN: {dp_py} missing, skip §0.2")

        print(f"  overall: {'PASS' if code == 0 else 'FAIL'} (exit={code})\n")
        worst = max(worst, code)

    print(f"Summary: worst_exit={worst}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
