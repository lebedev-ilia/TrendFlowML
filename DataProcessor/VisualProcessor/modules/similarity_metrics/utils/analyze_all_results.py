#!/usr/bin/env python3
"""
Анализ всех результатов тестирования similarity_metrics компонента.

Цели:
- сводная статистика по длинам осей (кадры)
- распределения по centroid_sims и temporal_sim_next
- статистики по feature_names/feature_values (intra-video coherence)
- опциональные reference similarity метрики
- поиск аномалий по z-score (best-effort)
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


def analyze_similarity_metrics_results(results_base_path: str) -> Dict[str, Any]:
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}

    per_video: List[Dict[str, Any]] = []

    num_frames: List[float] = []
    centroid_sim_all: List[float] = []
    temporal_sim_all: List[float] = []
    reference_present_count = 0
    
    # Feature-level aggregates
    centroid_sim_mean_all: List[float] = []
    centroid_sim_std_all: List[float] = []
    temporal_sim_mean_all: List[float] = []
    temporal_sim_std_all: List[float] = []
    reference_similarity_max_all: List[float] = []
    reference_similarity_mean_topn_all: List[float] = []

    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_similarity_metrics"):
            continue
        run_dir = video_dir / video_dir.name
        sim_dir = run_dir / "similarity_metrics"
        if not sim_dir.exists():
            continue

        npz_path = sim_dir / "results.npz"
        render_path = sim_dir / "_render" / "render_context.json"
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
            centroid_sims = npz_data.get("centroid_sims")
            temporal_sim_next = npz_data.get("temporal_sim_next")
            reference_present = npz_data.get("reference_present")
            fn = npz_data.get("feature_names")
            fv = npz_data.get("feature_values")

            stats: Dict[str, Any] = {
                "video_id": video_id,
                "status": status,
                "empty_reason": meta.get("empty_reason"),
            }

            # frame-axis
            if isinstance(fi, np.ndarray) and fi.ndim == 1:
                n = int(fi.shape[0])
                stats["num_frames"] = n
                num_frames.append(float(n))

            # reference_present
            if reference_present is not None:
                if isinstance(reference_present, np.ndarray):
                    ref_present = bool(reference_present.item())
                else:
                    ref_present = bool(reference_present)
                stats["reference_present"] = ref_present
                if ref_present:
                    reference_present_count += 1

            # centroid_sims (per-frame)
            if isinstance(centroid_sims, np.ndarray) and centroid_sims.ndim == 1:
                finite = centroid_sims[np.isfinite(centroid_sims)]
                if finite.size:
                    stats["centroid_sim_mean"] = float(np.mean(finite))
                    stats["centroid_sim_std"] = float(np.std(finite))
                    stats["centroid_sim_min"] = float(np.min(finite))
                    stats["centroid_sim_max"] = float(np.max(finite))
                    centroid_sim_all.extend(finite.tolist())
                    centroid_sim_mean_all.append(float(np.mean(finite)))
                    centroid_sim_std_all.append(float(np.std(finite)))

            # temporal_sim_next (per-frame, N-1)
            if isinstance(temporal_sim_next, np.ndarray) and temporal_sim_next.ndim == 1:
                finite = temporal_sim_next[np.isfinite(temporal_sim_next)]
                if finite.size:
                    stats["temporal_sim_mean"] = float(np.mean(finite))
                    stats["temporal_sim_std"] = float(np.std(finite))
                    stats["temporal_sim_min"] = float(np.min(finite))
                    stats["temporal_sim_max"] = float(np.max(finite))
                    temporal_sim_all.extend(finite.tolist())
                    temporal_sim_mean_all.append(float(np.mean(finite)))
                    temporal_sim_std_all.append(float(np.std(finite)))

            # feature_names / feature_values (video-level aggregates)
            if isinstance(fn, np.ndarray) and isinstance(fv, np.ndarray):
                try:
                    names = [str(x) for x in fn.reshape(-1).tolist()]
                    values = fv.reshape(-1).tolist()
                    features = dict(zip(names, values))
                    
                    # Extract expected features
                    if "centroid_sim_mean" in features:
                        val = features["centroid_sim_mean"]
                        if np.isfinite(val):
                            stats["feature_centroid_sim_mean"] = float(val)
                    if "centroid_sim_std" in features:
                        val = features["centroid_sim_std"]
                        if np.isfinite(val):
                            stats["feature_centroid_sim_std"] = float(val)
                    if "temporal_sim_mean" in features:
                        val = features["temporal_sim_mean"]
                        if np.isfinite(val):
                            stats["feature_temporal_sim_mean"] = float(val)
                    if "temporal_sim_std" in features:
                        val = features["temporal_sim_std"]
                        if np.isfinite(val):
                            stats["feature_temporal_sim_std"] = float(val)
                    
                    # Reference similarity (optional)
                    if "reference_similarity_max" in features:
                        val = features["reference_similarity_max"]
                        if np.isfinite(val):
                            stats["reference_similarity_max"] = float(val)
                            reference_similarity_max_all.append(float(val))
                    if "reference_similarity_mean_topn" in features:
                        val = features["reference_similarity_mean_topn"]
                        if np.isfinite(val):
                            stats["reference_similarity_mean_topn"] = float(val)
                            reference_similarity_mean_topn_all.append(float(val))
                except Exception as e:
                    stats["feature_parse_error"] = str(e)

            per_video.append(stats)

        except Exception as e:
            print(f"Error processing {video_dir.name}: {e}")
            continue

    # Aggregate statistics
    summary: Dict[str, Any] = {
        "total_videos": len(per_video),
        "reference_present_count": reference_present_count,
        "num_frames": _summary_stats(num_frames),
        "centroid_sim_all": _summary_stats(centroid_sim_all),
        "temporal_sim_all": _summary_stats(temporal_sim_all),
        "centroid_sim_mean_per_video": _summary_stats(centroid_sim_mean_all),
        "centroid_sim_std_per_video": _summary_stats(centroid_sim_std_all),
        "temporal_sim_mean_per_video": _summary_stats(temporal_sim_mean_all),
        "temporal_sim_std_per_video": _summary_stats(temporal_sim_std_all),
    }

    if reference_similarity_max_all:
        summary["reference_similarity_max"] = _summary_stats(reference_similarity_max_all)
    if reference_similarity_mean_topn_all:
        summary["reference_similarity_mean_topn"] = _summary_stats(reference_similarity_mean_topn_all)

    # Anomaly detection (z-score > 3)
    if centroid_sim_mean_all:
        arr = np.asarray(centroid_sim_mean_all)
        mean = np.mean(arr)
        std = np.std(arr)
        if std > 0:
            z_scores = np.abs((arr - mean) / std)
            anomalies = np.where(z_scores > 3.0)[0]
            if anomalies.size > 0:
                summary["centroid_sim_mean_anomalies"] = [
                    {
                        "video_id": per_video[int(i)]["video_id"],
                        "value": float(arr[i]),
                        "z_score": float(z_scores[i]),
                    }
                    for i in anomalies
                ]

    if temporal_sim_mean_all:
        arr = np.asarray(temporal_sim_mean_all)
        mean = np.mean(arr)
        std = np.std(arr)
        if std > 0:
            z_scores = np.abs((arr - mean) / std)
            anomalies = np.where(z_scores > 3.0)[0]
            if anomalies.size > 0:
                summary["temporal_sim_mean_anomalies"] = [
                    {
                        "video_id": per_video[int(i)]["video_id"],
                        "value": float(arr[i]),
                        "z_score": float(z_scores[i]),
                    }
                    for i in anomalies
                ]

    return {
        "summary": summary,
        "per_video": per_video,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze similarity_metrics component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    args = parser.parse_args()

    result = analyze_similarity_metrics_results(args.results_base)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("=" * 60)
    print("Similarity Metrics Component Analysis Report")
    print("=" * 60)
    print()

    summary = result["summary"]
    print(f"Total videos: {summary['total_videos']}")
    print(f"Videos with reference: {summary['reference_present_count']}")
    print()

    print("Num frames:")
    nf = summary["num_frames"]
    print(f"  count: {nf['count']}, mean: {nf.get('mean', 0):.1f}, std: {nf.get('std', 0):.1f}")
    print()

    print("Centroid similarity (all frames):")
    cs = summary["centroid_sim_all"]
    print(f"  count: {cs['count']}, mean: {cs.get('mean', 0):.4f}, std: {cs.get('std', 0):.4f}")
    print(f"  range: [{cs.get('min', 0):.4f}, {cs.get('max', 0):.4f}]")
    print()

    print("Temporal similarity (all frame pairs):")
    ts = summary["temporal_sim_all"]
    print(f"  count: {ts['count']}, mean: {ts.get('mean', 0):.4f}, std: {ts.get('std', 0):.4f}")
    print(f"  range: [{ts.get('min', 0):.4f}, {ts.get('max', 0):.4f}]")
    print()

    print("Centroid similarity (per-video mean):")
    csm = summary["centroid_sim_mean_per_video"]
    print(f"  count: {csm['count']}, mean: {csm.get('mean', 0):.4f}, std: {csm.get('std', 0):.4f}")
    print()

    print("Temporal similarity (per-video mean):")
    tsm = summary["temporal_sim_mean_per_video"]
    print(f"  count: {tsm['count']}, mean: {tsm.get('mean', 0):.4f}, std: {tsm.get('std', 0):.4f}")
    print()

    if "reference_similarity_max" in summary:
        print("Reference similarity (max):")
        rsm = summary["reference_similarity_max"]
        print(f"  count: {rsm['count']}, mean: {rsm.get('mean', 0):.4f}, std: {rsm.get('std', 0):.4f}")
        print()

    if "centroid_sim_mean_anomalies" in summary:
        print(f"⚠️  Centroid sim mean anomalies (z-score > 3): {len(summary['centroid_sim_mean_anomalies'])}")
        for a in summary["centroid_sim_mean_anomalies"][:5]:
            print(f"  - {a['video_id']}: value={a['value']:.4f}, z={a['z_score']:.2f}")
        print()

    if "temporal_sim_mean_anomalies" in summary:
        print(f"⚠️  Temporal sim mean anomalies (z-score > 3): {len(summary['temporal_sim_mean_anomalies'])}")
        for a in summary["temporal_sim_mean_anomalies"][:5]:
            print(f"  - {a['video_id']}: value={a['value']:.4f}, z={a['z_score']:.2f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

