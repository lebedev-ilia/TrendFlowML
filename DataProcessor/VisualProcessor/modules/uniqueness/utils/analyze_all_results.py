#!/usr/bin/env python3
"""
Анализ всех результатов тестирования uniqueness компонента.

Цели:
- сводная статистика по длинам осей (N кадров)
- распределения по max_sim_to_other, cos_dist_next
- базовые summary по ключевым фичам (repetition_ratio, diversity_score, pairwise_sim_mean, temporal_change_mean)
- поиск грубых аномалий (z-score по некоторым агрегатам)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add VisualProcessor to path
vp_root = Path(__file__).resolve().parent.parent.parent
if str(vp_root) not in sys.path:
    sys.path.insert(0, str(vp_root))

from utils.renderer import load_npz, extract_meta  # type: ignore


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


def analyze_uniqueness_results(results_base_path: str) -> Dict[str, Any]:
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}

    per_video: List[Dict[str, Any]] = []

    num_frames: List[float] = []
    max_sim_all: List[float] = []
    cos_dist_all: List[float] = []

    # Key features
    repetition_ratio_all: List[float] = []
    diversity_score_all: List[float] = []
    pairwise_sim_mean_all: List[float] = []
    temporal_change_mean_all: List[float] = []
    max_sim_to_other_mean_all: List[float] = []
    effective_unique_ratio_all: List[float] = []

    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_uniqueness"):
            continue
        run_dir = video_dir / video_dir.name
        uniq_dir = run_dir / "uniqueness"
        if not uniq_dir.exists():
            continue

        npz_path = uniq_dir / "uniqueness.npz"
        render_path = uniq_dir / "_render" / "render_context.json"
        if not npz_path.exists():
            continue

        try:
            npz_data = load_npz(str(npz_path))
            meta = extract_meta(npz_data)
            render_data: Dict[str, Any] = {}
            if render_path.exists():
                with open(render_path, "r", encoding="utf-8") as f:
                    render_data = json.load(f)

            video_id = video_dir.name
            status = meta.get("status", "unknown")

            fi = npz_data.get("frame_indices")
            max_sim = npz_data.get("max_sim_to_other")
            cos_dist = npz_data.get("cos_dist_next")
            fn = npz_data.get("feature_names")
            fv = npz_data.get("feature_values")

            stats: Dict[str, Any] = {
                "video_id": video_id,
                "status": status,
                "empty_reason": meta.get("empty_reason"),
            }

            if isinstance(fi, np.ndarray) and fi.ndim == 1:
                n = int(fi.shape[0])
                stats["num_frames"] = n
                num_frames.append(float(n))

            # per-frame arrays
            if isinstance(max_sim, np.ndarray) and max_sim.ndim == 1:
                max_sim_float = max_sim.astype(np.float32)
                finite = max_sim_float[np.isfinite(max_sim_float)]
                if finite.size:
                    stats["max_sim_mean"] = float(np.mean(finite))
                    stats["max_sim_std"] = float(np.std(finite))
                    max_sim_all.extend(finite.tolist())

            if isinstance(cos_dist, np.ndarray) and cos_dist.ndim == 1:
                cos_dist_float = cos_dist.astype(np.float32)
                finite = cos_dist_float[np.isfinite(cos_dist_float)]
                if finite.size:
                    stats["cos_dist_mean"] = float(np.mean(finite))
                    stats["cos_dist_std"] = float(np.std(finite))
                    cos_dist_all.extend(finite.tolist())

            # features dict
            features: Dict[str, Any] = {}
            if isinstance(fn, np.ndarray) and isinstance(fv, np.ndarray):
                try:
                    names = [str(x) for x in fn.reshape(-1).tolist()]
                    vals = fv.reshape(-1).tolist()
                    features = dict(zip(names, vals))
                except Exception:
                    features = {}

            if features:
                # core scalars
                for k in [
                    "repetition_ratio",
                    "diversity_score",
                    "pairwise_sim_mean",
                    "temporal_change_mean",
                    "max_sim_to_other_mean",
                    "effective_unique_ratio",
                ]:
                    v = features.get(k)
                    if v is not None and np.isfinite(v):
                        stats[k] = float(v)
                        if k == "repetition_ratio":
                            repetition_ratio_all.append(float(v))
                        elif k == "diversity_score":
                            diversity_score_all.append(float(v))
                        elif k == "pairwise_sim_mean":
                            pairwise_sim_mean_all.append(float(v))
                        elif k == "temporal_change_mean":
                            temporal_change_mean_all.append(float(v))
                        elif k == "max_sim_to_other_mean":
                            max_sim_to_other_mean_all.append(float(v))
                        elif k == "effective_unique_ratio":
                            effective_unique_ratio_all.append(float(v))

            per_video.append(stats)

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue

    # Aggregate statistics
    summary: Dict[str, Any] = {
        "total_videos": len(per_video),
        "num_frames": _summary_stats(num_frames),
        "max_sim_to_other": _summary_stats(max_sim_all),
        "cos_dist_next": _summary_stats(cos_dist_all),
        "repetition_ratio": _summary_stats(repetition_ratio_all),
        "diversity_score": _summary_stats(diversity_score_all),
        "pairwise_sim_mean": _summary_stats(pairwise_sim_mean_all),
        "temporal_change_mean": _summary_stats(temporal_change_mean_all),
        "max_sim_to_other_mean": _summary_stats(max_sim_to_other_mean_all),
        "effective_unique_ratio": _summary_stats(effective_unique_ratio_all),
    }

    # Anomaly detection (z-score > 3)
    anomalies: List[Dict[str, Any]] = []
    for key in ["repetition_ratio", "diversity_score", "pairwise_sim_mean", "temporal_change_mean"]:
        values = []
        video_ids = []
        for v in per_video:
            if key in v:
                values.append(v[key])
                video_ids.append(v["video_id"])
        if len(values) > 3:
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size > 3:
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                if std > 1e-6:
                    z_scores = (arr - mean) / std
                    for i, z in enumerate(z_scores):
                        if abs(z) > 3.0:
                            anomalies.append(
                                {
                                    "video_id": video_ids[i],
                                    "feature": key,
                                    "value": float(arr[i]),
                                    "z_score": float(z),
                                }
                            )

    if anomalies:
        summary["anomalies"] = anomalies[:10]  # Limit to 10

    return {
        "summary": summary,
        "per_video": per_video,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze uniqueness component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    args = parser.parse_args()

    result = analyze_uniqueness_results(args.results_base)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("=" * 60)
    print("Uniqueness Component Analysis Report")
    print("=" * 60)
    print()

    summary = result["summary"]
    print(f"Total videos: {summary['total_videos']}")
    print()

    def _p(name: str) -> None:
        s = summary.get(name, {"count": 0})
        print(f"{name}:")
        print(f"  count: {s.get('count', 0)}, mean: {s.get('mean', 0):.4f}, std: {s.get('std', 0):.4f}")
        if s.get("count", 0) > 0:
            print(f"  min/max: {s.get('min', 0):.4f} / {s.get('max', 0):.4f}")
        print()

    for key in [
        "num_frames",
        "max_sim_to_other",
        "cos_dist_next",
        "repetition_ratio",
        "diversity_score",
        "pairwise_sim_mean",
        "temporal_change_mean",
        "max_sim_to_other_mean",
        "effective_unique_ratio",
    ]:
        _p(key)

    # Check for anomalies
    if "anomalies" in summary:
        anomalies = summary["anomalies"]
        print(f"⚠️  Anomalies (z > 3): {len(anomalies)}")
        for a in anomalies[:5]:
            print(f"  - {a['video_id']}: {a['feature']}={a['value']:.4f}, z={a['z_score']:.2f}")
        print()
    else:
        print("✅ No anomalies detected (z-score > 3)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

