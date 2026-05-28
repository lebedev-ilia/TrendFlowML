#!/usr/bin/env python3
"""
Анализ всех результатов тестирования text_scoring компонента.

Цели:
- сводная статистика по длинам осей (N кадров)
- распределения по text_presence, text_count_per_frame
- базовые summary по ключевым фичам (text_present, cta_presence, continuity, sync scores)
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


def analyze_text_scoring_results(results_base_path: str) -> Dict[str, Any]:
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}

    per_video: List[Dict[str, Any]] = []

    num_frames: List[float] = []
    text_presence_all: List[float] = []
    text_count_all: List[float] = []

    # Key features
    text_present_all: List[float] = []
    text_frames_ratio_all: List[float] = []
    num_unique_texts_all: List[float] = []
    cta_presence_all: List[float] = []
    text_on_screen_continuity_all: List[float] = []
    text_action_sync_score_all: List[float] = []
    text_motion_alignment_all: List[float] = []

    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_text_scoring"):
            continue
        run_dir = video_dir / video_dir.name
        ts_dir = run_dir / "text_scoring"
        if not ts_dir.exists():
            continue

        npz_path = ts_dir / "text_scoring.npz"
        render_path = ts_dir / "_render" / "render_context.json"
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
            text_presence = npz_data.get("text_presence")
            text_count = npz_data.get("text_count_per_frame")
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
            if isinstance(text_presence, np.ndarray) and text_presence.ndim == 1:
                text_presence_float = text_presence.astype(np.float32)
                finite = text_presence_float[np.isfinite(text_presence_float)]
                if finite.size:
                    stats["text_presence_mean"] = float(np.mean(finite))
                    text_presence_all.extend(finite.tolist())

            if isinstance(text_count, np.ndarray) and text_count.ndim == 1:
                text_count_float = text_count.astype(np.float32)
                finite = text_count_float[np.isfinite(text_count_float)]
                if finite.size:
                    stats["text_count_mean"] = float(np.mean(finite))
                    stats["text_count_max"] = float(np.max(finite))
                    text_count_all.extend(finite.tolist())

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
                    "text_present",
                    "text_frames_ratio",
                    "num_unique_texts",
                    "cta_presence",
                    "text_on_screen_continuity",
                    "text_action_sync_score",
                    "text_motion_alignment",
                ]:
                    v = features.get(k)
                    if v is not None and np.isfinite(v):
                        stats[k] = float(v)
                        if k == "text_present":
                            text_present_all.append(float(v))
                        elif k == "text_frames_ratio":
                            text_frames_ratio_all.append(float(v))
                        elif k == "num_unique_texts":
                            num_unique_texts_all.append(float(v))
                        elif k == "cta_presence":
                            cta_presence_all.append(float(v))
                        elif k == "text_on_screen_continuity":
                            text_on_screen_continuity_all.append(float(v))
                        elif k == "text_action_sync_score":
                            text_action_sync_score_all.append(float(v))
                        elif k == "text_motion_alignment":
                            text_motion_alignment_all.append(float(v))

            per_video.append(stats)

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue

    # Aggregate statistics
    summary: Dict[str, Any] = {
        "total_videos": len(per_video),
        "num_frames": _summary_stats(num_frames),
        "text_presence": _summary_stats(text_presence_all),
        "text_count": _summary_stats(text_count_all),
        "text_present": _summary_stats(text_present_all),
        "text_frames_ratio": _summary_stats(text_frames_ratio_all),
        "num_unique_texts": _summary_stats(num_unique_texts_all),
        "cta_presence": _summary_stats(cta_presence_all),
        "text_on_screen_continuity": _summary_stats(text_on_screen_continuity_all),
        "text_action_sync_score": _summary_stats(text_action_sync_score_all),
        "text_motion_alignment": _summary_stats(text_motion_alignment_all),
    }

    # Anomaly detection (z-score > 3)
    anomalies: List[Dict[str, Any]] = []
    for key in ["cta_presence", "text_action_sync_score", "text_motion_alignment"]:
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
        summary[f"{key}_anomalies"] = anomalies[:10]  # Limit to 10

    return {
        "summary": summary,
        "per_video": per_video,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze text_scoring component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    args = parser.parse_args()

    result = analyze_text_scoring_results(args.results_base)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("=" * 60)
    print("Text Scoring Component Analysis Report")
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
        "text_presence",
        "text_count",
        "text_present",
        "text_frames_ratio",
        "num_unique_texts",
        "cta_presence",
        "text_on_screen_continuity",
        "text_action_sync_score",
        "text_motion_alignment",
    ]:
        _p(key)

    # Check for anomalies
    anomaly_keys = [k for k in summary.keys() if k.endswith("_anomalies")]
    if anomaly_keys:
        for key in anomaly_keys:
            anomalies = summary[key]
            print(f"⚠️  {key} (z > 3): {len(anomalies)}")
            for a in anomalies[:5]:
                print(f"  - {a['video_id']}: {a['feature']}={a['value']:.4f}, z={a['z_score']:.2f}")
            print()
    else:
        print("✅ No anomalies detected (z-score > 3)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

