#!/usr/bin/env python3
"""
Анализ всех результатов тестирования micro_emotion компонента.

Цели:
- сводная статистика по длинам осей (N кадров, K событий)
- распределения по frame_features, compact22
- базовые summary по ключевым фичам (microexpr_count, au_intensity_mean, и т.д.)
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


def analyze_micro_emotion_results(results_base_path: str) -> Dict[str, Any]:
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}

    per_video: List[Dict[str, Any]] = []

    num_frames: List[float] = []
    num_face_frames: List[float] = []
    num_events: List[float] = []
    compact22_mean: List[float] = []

    # Key features
    microexpr_count_all: List[float] = []
    au_intensity_mean_all: List[float] = []
    face_present_ratio_all: List[float] = []
    gaze_centered_ratio_all: List[float] = []

    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_micro_emotion"):
            continue
        run_dir = video_dir / video_dir.name
        me_dir = run_dir / "micro_emotion"
        if not me_dir.exists():
            continue

        npz_path = me_dir / "micro_emotion.npz"
        render_path = me_dir / "_render" / "render_context.json"
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
            face_present = npz_data.get("face_present_any")
            compact22 = npz_data.get("compact22")
            events = npz_data.get("event_times_s")
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

            if isinstance(face_present, np.ndarray) and face_present.ndim == 1:
                face_count = int(np.sum(face_present))
                stats["num_face_frames"] = face_count
                num_face_frames.append(float(face_count))
                if n > 0:
                    face_ratio = float(face_count) / float(n)
                    stats["face_present_ratio"] = face_ratio
                    face_present_ratio_all.append(face_ratio)

            if isinstance(compact22, np.ndarray) and compact22.ndim == 2:
                compact22_float = compact22.astype(np.float32)
                finite = compact22_float[np.isfinite(compact22_float)]
                if finite.size:
                    stats["compact22_mean"] = float(np.mean(finite))
                    compact22_mean.append(float(np.mean(finite)))

            if isinstance(events, np.ndarray) and events.ndim == 1:
                k = int(events.shape[0])
                stats["num_events"] = k
                num_events.append(float(k))

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
                    "microexpr_count",
                    "au_intensity_mean",
                    "gaze_centered_ratio",
                ]:
                    v = features.get(k)
                    if v is not None and np.isfinite(v):
                        stats[k] = float(v)
                        if k == "microexpr_count":
                            microexpr_count_all.append(float(v))
                        elif k == "au_intensity_mean":
                            au_intensity_mean_all.append(float(v))
                        elif k == "gaze_centered_ratio":
                            gaze_centered_ratio_all.append(float(v))

            per_video.append(stats)

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue

    # Aggregate statistics
    summary: Dict[str, Any] = {
        "total_videos": len(per_video),
        "num_frames": _summary_stats(num_frames),
        "num_face_frames": _summary_stats(num_face_frames),
        "num_events": _summary_stats(num_events),
        "compact22_mean": _summary_stats(compact22_mean),
        "microexpr_count": _summary_stats(microexpr_count_all),
        "au_intensity_mean": _summary_stats(au_intensity_mean_all),
        "face_present_ratio": _summary_stats(face_present_ratio_all),
        "gaze_centered_ratio": _summary_stats(gaze_centered_ratio_all),
    }

    # Anomaly detection (z-score > 3)
    anomalies: List[Dict[str, Any]] = []
    for key in ["microexpr_count", "au_intensity_mean", "face_present_ratio"]:
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
    parser = argparse.ArgumentParser(description="Analyze micro_emotion component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    args = parser.parse_args()

    result = analyze_micro_emotion_results(args.results_base)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("=" * 60)
    print("Micro Emotion Component Analysis Report")
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
        "num_face_frames",
        "num_events",
        "compact22_mean",
        "microexpr_count",
        "au_intensity_mean",
        "face_present_ratio",
        "gaze_centered_ratio",
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

