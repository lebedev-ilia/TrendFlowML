#!/usr/bin/env python3
"""
Анализ всех результатов тестирования optical_flow компонента.

Цели:
- сводная статистика по video-level feature_values (mean/median/p90/variance/...)
- сводная статистика по длинам осей и NaN-coverage
- поиск аномалий по z-score (best-effort)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

import numpy as np

# Add VisualProcessor to path
vp_root = Path(__file__).resolve().parent.parent.parent
if str(vp_root) not in sys.path:
    sys.path.insert(0, str(vp_root))

from utils.renderer import load_npz, extract_meta


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "count": int(arr.size),
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": float(std / (abs(mean) + 1e-12)),
    }


def analyze_optical_flow_results(results_base_path: str) -> Dict[str, Any]:
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}

    per_video: List[Dict[str, Any]] = []
    features_by_name: Dict[str, List[float]] = defaultdict(list)
    nan_ratios_motion: List[float] = []
    N_frames: List[float] = []

    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_optical_flow"):
            continue
        run_dir = video_dir / video_dir.name
        optical_flow_dir = run_dir / "optical_flow"
        if not optical_flow_dir.exists():
            continue

        npz_path = optical_flow_dir / "optical_flow.npz"
        render_path = optical_flow_dir / "_render" / "render_context.json"
        if not npz_path.exists():
            continue

        try:
            npz_data = load_npz(str(npz_path))
            meta = extract_meta(npz_data)
            render_data = {}
            if render_path.exists():
                with open(render_path, "r", encoding="utf-8") as f:
                    render_data = json.load(f)

            video_id = video_dir.name
            status = meta.get("status", "unknown")

            fi = npz_data.get("frame_indices")
            motion = npz_data.get("motion_norm_per_sec_mean")

            stats: Dict[str, Any] = {
                "video_id": video_id,
                "status": status,
                "empty_reason": meta.get("empty_reason"),
            }

            if isinstance(fi, np.ndarray) and fi.ndim == 1:
                stats["num_frames"] = int(len(fi))
                N_frames.append(float(len(fi)))

            if isinstance(motion, np.ndarray) and motion.ndim == 1:
                nan_ratio = float(np.mean(~np.isfinite(motion))) if motion.size else 0.0
                stats["motion_nan_ratio"] = nan_ratio
                nan_ratios_motion.append(nan_ratio)
                finite = motion[np.isfinite(motion)]
                if finite.size:
                    stats["motion_mean"] = float(np.mean(finite))
                    stats["motion_p90"] = float(np.percentile(finite, 90))

            # video-level feature table
            fn = npz_data.get("feature_names")
            fv = npz_data.get("feature_values")
            if isinstance(fn, np.ndarray) and isinstance(fv, np.ndarray) and fn.ndim == 1 and fv.ndim == 1:
                try:
                    names = [str(x) for x in fn.reshape(-1).tolist()]
                    vals = fv.reshape(-1).astype(np.float64)
                    for name, val in zip(names, vals):
                        if np.isfinite(val):
                            features_by_name[name].append(float(val))
                            stats[f"feature_{name}"] = float(val)
                except Exception:
                    pass

            # from meta (if present)
            for k in ["total_frames", "processed_frames", "analysis_fps", "analysis_width", "analysis_height"]:
                if k in meta:
                    v = _safe_float(meta.get(k))
                    if v is not None:
                        stats[k] = v

            per_video.append(stats)
        except Exception:
            continue

    # global report
    report: Dict[str, Any] = {
        "total_videos": len(per_video),
        "motion_nan_ratio": _summary_stats(nan_ratios_motion),
        "num_frames": _summary_stats(N_frames),
        "features": {name: _summary_stats(vals) for name, vals in sorted(features_by_name.items())},
        "per_video": per_video,
    }

    # best-effort anomaly detection (z-score on feature_* where enough samples)
    anomalies: List[Dict[str, Any]] = []
    for name, vals in features_by_name.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size < 8:
            continue
        mu = float(np.mean(arr))
        sd = float(np.std(arr)) + 1e-12
        z = (arr - mu) / sd
        # mark anything beyond 3.5
        if np.any(np.abs(z) > 3.5):
            anomalies.append({"feature": name, "mean": mu, "std": sd, "max_abs_z": float(np.max(np.abs(z)))})
    report["anomalies"] = sorted(anomalies, key=lambda x: x["max_abs_z"], reverse=True)

    return report


def print_analysis_report(report: Dict[str, Any]) -> None:
    if "error" in report:
        print(f"Error: {report['error']}")
        return

    print("=" * 60)
    print("Optical Flow Component Analysis Report")
    print("=" * 60)
    print(f"Total videos: {report.get('total_videos', 0)}")
    print()

    print("Axis stats:")
    print(f"- num_frames: {report.get('num_frames')}")
    print(f"- motion_nan_ratio: {report.get('motion_nan_ratio')}")
    print()

    print("Video-level features (aggregate across videos):")
    feats = report.get("features", {})
    for name, st in feats.items():
        print(f"- {name}: {st}")
    print()

    anomalies = report.get("anomalies", [])
    if anomalies:
        print("Potential anomalies (z-score > 3.5):")
        for a in anomalies[:20]:
            print(f"- {a['feature']}: max_abs_z={a['max_abs_z']:.2f} (mean={a['mean']:.4g}, std={a['std']:.4g})")
    else:
        print("✅ No obvious anomalies found (best-effort).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze optical_flow component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    args = parser.parse_args()

    report = analyze_optical_flow_results(args.results_base)
    print_analysis_report(report)
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())


