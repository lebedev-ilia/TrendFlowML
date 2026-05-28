#!/usr/bin/env python3
"""
Анализ всех результатов тестирования video_pacing компонента.

Цели:
- сводная статистика по длинам осей (N кадров, S shot boundaries)
- распределения по motion_norm_per_sec_mean, semantic_change_rate_per_sec, color_change_rate_per_sec
- базовые summary по ключевым фичам (shots_count, shot_duration_mean, cuts_per_10s, motion_speed_median, и т.д.)
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


def analyze_video_pacing_results(results_base_path: str) -> Dict[str, Any]:
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}

    per_video: List[Dict[str, Any]] = []

    num_frames: List[float] = []
    num_shots: List[float] = []
    motion_all: List[float] = []
    semantic_all: List[float] = []
    color_all: List[float] = []

    # Key features
    shots_count_all: List[float] = []
    shot_duration_mean_all: List[float] = []
    cuts_per_10s_all: List[float] = []
    motion_speed_median_all: List[float] = []
    color_change_rate_mean_all: List[float] = []
    semantic_change_rate_mean_all: List[float] = []
    video_length_seconds_all: List[float] = []

    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_video_pacing"):
            continue
        run_dir = video_dir / video_dir.name
        vp_dir = run_dir / "video_pacing"
        if not vp_dir.exists():
            continue

        npz_path = vp_dir / "video_pacing_features.npz"
        render_path = vp_dir / "_render" / "render_context.json"
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
            shot_bounds = npz_data.get("shot_boundary_frame_indices")
            motion = npz_data.get("motion_norm_per_sec_mean")
            semantic = npz_data.get("semantic_change_rate_per_sec")
            color = npz_data.get("color_change_rate_per_sec")
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

            if isinstance(shot_bounds, np.ndarray) and shot_bounds.ndim == 1:
                s = int(shot_bounds.shape[0])
                stats["num_shots"] = s
                num_shots.append(float(s))

            # per-frame arrays
            if isinstance(motion, np.ndarray) and motion.ndim == 1:
                motion_float = motion.astype(np.float32)
                finite = motion_float[np.isfinite(motion_float)]
                if finite.size:
                    stats["motion_mean"] = float(np.mean(finite))
                    stats["motion_std"] = float(np.std(finite))
                    motion_all.extend(finite.tolist())

            if isinstance(semantic, np.ndarray) and semantic.ndim == 1:
                semantic_float = semantic.astype(np.float32)
                finite = semantic_float[np.isfinite(semantic_float)]
                if finite.size:
                    stats["semantic_mean"] = float(np.mean(finite))
                    stats["semantic_std"] = float(np.std(finite))
                    semantic_all.extend(finite.tolist())

            if isinstance(color, np.ndarray) and color.ndim == 1:
                color_float = color.astype(np.float32)
                finite = color_float[np.isfinite(color_float)]
                if finite.size:
                    stats["color_mean"] = float(np.mean(finite))
                    stats["color_std"] = float(np.std(finite))
                    color_all.extend(finite.tolist())

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
                    "video_length_seconds",
                    "shots_count",
                    "shot_duration_mean",
                    "cuts_per_10s",
                    "motion_speed_median",
                    "color_change_rate_mean",
                    "semantic_change_rate_mean",
                ]:
                    v = features.get(k)
                    if v is not None and np.isfinite(v):
                        stats[k] = float(v)
                        if k == "shots_count":
                            shots_count_all.append(float(v))
                        elif k == "shot_duration_mean":
                            shot_duration_mean_all.append(float(v))
                        elif k == "cuts_per_10s":
                            cuts_per_10s_all.append(float(v))
                        elif k == "motion_speed_median":
                            motion_speed_median_all.append(float(v))
                        elif k == "color_change_rate_mean":
                            color_change_rate_mean_all.append(float(v))
                        elif k == "semantic_change_rate_mean":
                            semantic_change_rate_mean_all.append(float(v))
                        elif k == "video_length_seconds":
                            video_length_seconds_all.append(float(v))

            per_video.append(stats)

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue

    # Aggregate statistics
    summary: Dict[str, Any] = {
        "total_videos": len(per_video),
        "num_frames": _summary_stats(num_frames),
        "num_shots": _summary_stats(num_shots),
        "motion_norm_per_sec_mean": _summary_stats(motion_all),
        "semantic_change_rate_per_sec": _summary_stats(semantic_all),
        "color_change_rate_per_sec": _summary_stats(color_all),
        "shots_count": _summary_stats(shots_count_all),
        "shot_duration_mean": _summary_stats(shot_duration_mean_all),
        "cuts_per_10s": _summary_stats(cuts_per_10s_all),
        "motion_speed_median": _summary_stats(motion_speed_median_all),
        "color_change_rate_mean": _summary_stats(color_change_rate_mean_all),
        "semantic_change_rate_mean": _summary_stats(semantic_change_rate_mean_all),
        "video_length_seconds": _summary_stats(video_length_seconds_all),
    }

    # Anomaly detection (z-score > 3)
    anomalies: List[Dict[str, Any]] = []
    for key in ["shots_count", "shot_duration_mean", "cuts_per_10s", "motion_speed_median"]:
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
    parser = argparse.ArgumentParser(description="Analyze video_pacing component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    args = parser.parse_args()

    result = analyze_video_pacing_results(args.results_base)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("=" * 60)
    print("Video Pacing Component Analysis Report")
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
        "num_shots",
        "motion_norm_per_sec_mean",
        "semantic_change_rate_per_sec",
        "color_change_rate_per_sec",
        "shots_count",
        "shot_duration_mean",
        "cuts_per_10s",
        "motion_speed_median",
        "color_change_rate_mean",
        "semantic_change_rate_mean",
        "video_length_seconds",
    ]:
        _p(key)

    if summary.get("anomalies"):
        print("Anomalies (z-score > 3):")
        for a in summary["anomalies"]:
            print(f"  {a['video_id']}: {a['feature']} = {a['value']:.4f} (z={a['z_score']:.2f})")
        print()
    else:
        print("✅ No anomalies detected (z-score > 3)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

