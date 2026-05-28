#!/usr/bin/env python3
"""
Анализ всех результатов тестирования shot_quality компонента.

Цели:
- сводная статистика по длинам осей (кадры, шоты)
- распределения по confidence/entropy (frame-level и shot-level)
- базовые статистики по frame_feature_present_ratio и shot_frame_feature_present_ratio
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


def analyze_shot_quality_results(results_base_path: str) -> Dict[str, Any]:
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}

    per_video: List[Dict[str, Any]] = []

    num_frames: List[float] = []
    num_shots: List[float] = []
    frame_conf_all: List[float] = []
    frame_entropy_all: List[float] = []
    shot_conf_all: List[float] = []
    shot_entropy_all: List[float] = []
    feature_present_ratios: List[float] = []
    shot_feature_present_ratios: List[float] = []

    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_shot_quality"):
            continue
        run_dir = video_dir / video_dir.name
        sq_dir = run_dir / "shot_quality"
        if not sq_dir.exists():
            continue

        npz_path = sq_dir / "shot_quality.npz"
        render_path = sq_dir / "_render" / "render_context.json"
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
            shot_ids = npz_data.get("shot_ids")
            ffr = npz_data.get("frame_feature_present_ratio")
            sffr = npz_data.get("shot_frame_feature_present_ratio")
            shot_conf = npz_data.get("shot_quality_conf_mean")
            shot_ent = npz_data.get("shot_quality_entropy_mean")

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

            # shot-axis
            if isinstance(shot_ids, np.ndarray) and shot_ids.ndim == 1:
                s = int(np.max(shot_ids) + 1) if shot_ids.size else 0
                stats["num_shots"] = s
                num_shots.append(float(s))

            # feature present ratios
            if isinstance(ffr, np.ndarray) and ffr.ndim == 1:
                finite = ffr[np.isfinite(ffr)]
                if finite.size:
                    stats["frame_feature_present_ratio_mean"] = float(np.mean(finite))
                    feature_present_ratios.extend(finite.tolist())

            if isinstance(sffr, np.ndarray) and sffr.ndim == 2:
                finite = sffr[np.isfinite(sffr)]
                if finite.size:
                    stats["shot_frame_feature_present_ratio_mean"] = float(np.mean(finite))
                    shot_feature_present_ratios.extend(finite.tolist())

            # from ui_payload if present
            ui = meta.get("ui_payload") or {}
            if isinstance(ui, dict):
                q = ui.get("quality") or {}
                if isinstance(q, dict):
                    frame_conf = np.asarray(q.get("frame_confidence"), dtype=np.float32)
                    frame_ent = np.asarray(q.get("frame_entropy"), dtype=np.float32)
                    if frame_conf.ndim == 1 and frame_conf.size:
                        finite = frame_conf[np.isfinite(frame_conf)]
                        if finite.size:
                            stats["frame_confidence_mean"] = float(np.mean(finite))
                            frame_conf_all.extend(finite.tolist())
                    if frame_ent.ndim == 1 and frame_ent.size:
                        finite = frame_ent[np.isfinite(frame_ent)]
                        if finite.size:
                            stats["frame_entropy_mean"] = float(np.mean(finite))
                            frame_entropy_all.extend(finite.tolist())

            # shot-level quality aggregates
            if isinstance(shot_conf, np.ndarray) and shot_conf.ndim == 1:
                finite = shot_conf[np.isfinite(shot_conf)]
                if finite.size:
                    stats["shot_conf_mean"] = float(np.mean(finite))
                    shot_conf_all.extend(finite.tolist())
            if isinstance(shot_ent, np.ndarray) and shot_ent.ndim == 1:
                finite = shot_ent[np.isfinite(shot_ent)]
                if finite.size:
                    stats["shot_entropy_mean"] = float(np.mean(finite))
                    shot_entropy_all.extend(finite.tolist())

            per_video.append(stats)
        except Exception:
            continue

    report: Dict[str, Any] = {
        "total_videos": len(per_video),
        "per_video": per_video,
        "num_frames": _summary_stats(num_frames),
        "num_shots": _summary_stats(num_shots),
        "frame_confidence": _summary_stats(frame_conf_all),
        "frame_entropy": _summary_stats(frame_entropy_all),
        "shot_confidence": _summary_stats(shot_conf_all),
        "shot_entropy": _summary_stats(shot_entropy_all),
        "frame_feature_present_ratio": _summary_stats(feature_present_ratios),
        "shot_frame_feature_present_ratio": _summary_stats(shot_feature_present_ratios),
    }

    # best-effort anomaly detection по num_frames/num_shots
    anomalies: List[Dict[str, Any]] = []
    for key in ("num_frames", "num_shots"):
        st = report.get(key) or {}
        if st.get("count", 0) < 5:
            continue
        mu = float(st.get("mean", 0.0))
        sd = float(st.get("std", 0.0)) + 1e-12
        for v in per_video:
            if key not in v:
                continue
            z = abs((float(v[key]) - mu) / sd)
            if z > 3.5:
                anomalies.append(
                    {"metric": key, "video_id": v["video_id"], "value": float(v[key]), "z_score": float(z)}
                )
    report["anomalies"] = anomalies
    return report


def print_analysis_report(report: Dict[str, Any]) -> None:
    if "error" in report:
        print(f"Error: {report['error']}")
        return

    print("=" * 60)
    print("Shot Quality Component Analysis Report")
    print("=" * 60)
    print(f"Total videos: {report.get('total_videos', 0)}")
    print()

    print("Axis stats:")
    print(f"- num_frames: {report.get('num_frames')}")
    print(f"- num_shots: {report.get('num_shots')}")
    print()

    print("Frame-level quality:")
    print(f"- frame_confidence: {report.get('frame_confidence')}")
    print(f"- frame_entropy: {report.get('frame_entropy')}")
    print()

    print("Shot-level quality:")
    print(f"- shot_confidence: {report.get('shot_confidence')}")
    print(f"- shot_entropy: {report.get('shot_entropy')}")
    print()

    print("Feature coverage:")
    print(f"- frame_feature_present_ratio: {report.get('frame_feature_present_ratio')}")
    print(f"- shot_frame_feature_present_ratio: {report.get('shot_frame_feature_present_ratio')}")
    print()

    anomalies = report.get("anomalies") or []
    if anomalies:
        print("Potential anomalies (z-score > 3.5):")
        for a in anomalies:
            print(
                f"- {a['metric']} | {a['video_id']}: value={a['value']:.2f}, z_score={a['z_score']:.2f}"
            )
    else:
        print("✅ No obvious anomalies found (best-effort).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze shot_quality component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    args = parser.parse_args()

    report = analyze_shot_quality_results(args.results_base)
    print_analysis_report(report)
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())


