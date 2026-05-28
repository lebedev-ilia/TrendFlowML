#!/usr/bin/env python3
"""Анализ результатов pitch_extractor."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, Any

def analyze(rs_base: str = "dp_results/youtube", run_id_prefix: str = "test_pitch_", component_name: str = "pitch_extractor", npz_name: str = "pitch_extractor_features.npz") -> Dict[str, Any]:
    rs = Path(rs_base) / "youtube"
    if not rs.exists():
        return {"total_videos": 0, "per_video": [], "summary": {}}
    stats = []
    for run_dir in sorted(rs.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(run_id_prefix):
            continue
        npz = run_dir / run_dir.name / component_name / npz_name
        if not npz.exists():
            continue
        stats.append({"video_id": run_dir.name, "valid": True})
    return {"total_videos": len(stats), "per_video": stats, "summary": {"valid_count": len(stats), "total_count": len(stats)}}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rs-base", default="dp_results/youtube")
    p.add_argument("--run-id-prefix", default="test_pitch_")
    p.add_argument("--component-name", default="pitch_extractor")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    r = analyze(args.rs_base, args.run_id_prefix, args.component_name)
    print(json.dumps(r, indent=2, ensure_ascii=False) if args.json else f"Всего: {r['total_videos']}, валидных: {r['summary'].get('valid_count', 0)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
