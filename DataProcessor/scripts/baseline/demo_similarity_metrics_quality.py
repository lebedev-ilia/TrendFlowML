#!/usr/bin/env python3
"""
Legacy wrapper (moved).

The beautiful HTML quality report now lives in:
  VisualProcessor/modules/similarity_metrics/quality_report/demo_similarity_metrics_quality.py
"""

def main() -> int:
    import argparse
    import subprocess
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to similarity_metrics/results.npz")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # Execute the new script in-place.
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "VisualProcessor" / "modules" / "similarity_metrics" / "quality_report" / "demo_similarity_metrics_quality.py"
    cmd = [sys.executable, str(script), "--npz", str(Path(args.npz).resolve()), "--out-dir", str(Path(args.out_dir).resolve())]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())


