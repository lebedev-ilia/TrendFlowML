#!/usr/bin/env python3
"""
Анализ всех результатов тестирования story_structure компонента.

Цели:
- сводная статистика по длинам осей (N кадров)
- распределения по story_energy_curve, motion_norm_per_sec_mean, embedding_change_rate_per_sec
- базовые summary по hook/climax/character features
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


def analyze_story_structure_results(results_base_path: str) -> Dict[str, Any]:
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}

    per_video: List[Dict[str, Any]] = []

    num_frames: List[float] = []
    video_length_sec: List[float] = []
    energy_all: List[float] = []
    motion_all: List[float] = []
    emb_rate_all: List[float] = []

    hook_score_all: List[float] = []
    climax_time_all: List[float] = []
    main_char_screen_time_all: List[float] = []

    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_story_structure"):
            continue
        run_dir = video_dir / video_dir.name
        ss_dir = run_dir / "story_structure"
        if not ss_dir.exists():
            continue

        npz_path = ss_dir / "story_structure.npz"
        render_path = ss_dir / "_render" / "render_context.json"
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
            energy = npz_data.get("story_energy_curve")
            motion = npz_data.get("motion_norm_per_sec_mean")
            emb_rate = npz_data.get("embedding_change_rate_per_sec")
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

            # per-frame curves
            if isinstance(energy, np.ndarray) and energy.ndim == 1:
                finite = energy[np.isfinite(energy)]
                if finite.size:
                    stats["energy_mean"] = float(np.mean(finite))
                    stats["energy_std"] = float(np.std(finite))
                    energy_all.extend(finite.tolist())

            if isinstance(motion, np.ndarray) and motion.ndim == 1:
                finite = motion[np.isfinite(motion)]
                if finite.size:
                    stats["motion_mean"] = float(np.mean(finite))
                    stats["motion_std"] = float(np.std(finite))
                    motion_all.extend(finite.tolist())

            if isinstance(emb_rate, np.ndarray) and emb_rate.ndim == 1:
                finite = emb_rate[np.isfinite(emb_rate)]
                if finite.size:
                    stats["emb_rate_mean"] = float(np.mean(finite))
                    stats["emb_rate_std"] = float(np.std(finite))
                    emb_rate_all.extend(finite.tolist())

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
                    "hook_visual_surprise_score",
                    "climax_time_sec",
                    "main_character_screen_time",
                    "number_of_peaks",
                ]:
                    if k in features and np.isfinite(features[k]):
                        stats[k] = float(features[k])

                if "video_length_seconds" in features and np.isfinite(features["video_length_seconds"]):
                    video_length_sec.append(float(features["video_length_seconds"]))

                if "hook_visual_surprise_score" in features and np.isfinite(features["hook_visual_surprise_score"]):
                    hook_score_all.append(float(features["hook_visual_surprise_score"]))

                if "climax_time_sec" in features and np.isfinite(features["climax_time_sec"]):
                    climax_time_all.append(float(features["climax_time_sec"]))

                if "main_character_screen_time" in features and np.isfinite(features["main_character_screen_time"]):
                    main_char_screen_time_all.append(float(features["main_character_screen_time"]))

            per_video.append(stats)

        except Exception as e:  # pragma: no cover - best-effort
            print(f"Error processing {video_dir.name}: {e}")
            continue

    summary: Dict[str, Any] = {
        "total_videos": len(per_video),
        "num_frames": _summary_stats(num_frames),
        "video_length_seconds": _summary_stats(video_length_sec),
        "story_energy_curve": _summary_stats(energy_all),
        "motion_norm_per_sec_mean": _summary_stats(motion_all),
        "embedding_change_rate_per_sec": _summary_stats(emb_rate_all),
        "hook_visual_surprise_score": _summary_stats(hook_score_all),
        "climax_time_sec": _summary_stats(climax_time_all),
        "main_character_screen_time": _summary_stats(main_char_screen_time_all),
    }

    # простейший поиск аномалий по z-score для hook_visual_surprise_score
    if hook_score_all:
        arr = np.asarray(hook_score_all, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std > 0:
            z = np.abs((arr - mean) / std)
            idx = np.where(z > 3.0)[0]
            if idx.size > 0:
                summary["hook_visual_surprise_score_anomalies"] = [
                    {
                        "video_id": per_video[int(i)]["video_id"],
                        "value": float(arr[i]),
                        "z_score": float(z[i]),
                    }
                    for i in idx
                ]

    return {"summary": summary, "per_video": per_video}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze story_structure component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    args = parser.parse_args()

    result = analyze_story_structure_results(args.results_base)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("=" * 60)
    print("Story Structure Component Analysis Report")
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
        "video_length_seconds",
        "story_energy_curve",
        "motion_norm_per_sec_mean",
        "embedding_change_rate_per_sec",
        "hook_visual_surprise_score",
        "climax_time_sec",
        "main_character_screen_time",
    ]:
        _p(key)

    if "hook_visual_surprise_score_anomalies" in summary:
        anomalies = summary["hook_visual_surprise_score_anomalies"]
        print(f"⚠️  hook_visual_surprise_score anomalies (z > 3): {len(anomalies)}")
        for a in anomalies[:5]:
            print(f"  - {a['video_id']}: value={a['value']:.4f}, z={a['z_score']:.2f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
