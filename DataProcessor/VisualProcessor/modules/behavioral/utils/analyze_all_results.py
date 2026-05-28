#!/usr/bin/env python3
"""
Анализ всех результатов тестирования behavioral компонента.
Проверяет целостность, нормальность и информативность значений.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add VisualProcessor to path
vp_root = Path(__file__).resolve().parent.parent.parent
if str(vp_root) not in sys.path:
    sys.path.insert(0, str(vp_root))

from utils.renderer import load_npz, extract_meta


def analyze_behavioral_results(results_base_path: str) -> Dict[str, Any]:
    """Анализирует все результаты behavioral тестов."""
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}
    
    # Находим все behavioral результаты
    all_results = []
    video_stats = []
    
    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_behavioral"):
            continue
        
        run_dir = video_dir / video_dir.name
        behavioral_dir = run_dir / "behavioral"
        
        if not behavioral_dir.exists():
            continue
        
        npz_path = behavioral_dir / "behavioral_features.npz"
        render_path = behavioral_dir / "_render" / "render_context.json"
        
        if not npz_path.exists():
            continue
        
        try:
            # Загружаем NPZ
            npz_data = load_npz(str(npz_path))
            meta = extract_meta(npz_data)
            
            # Загружаем render
            render_data = {}
            if render_path.exists():
                with open(render_path, 'r', encoding='utf-8') as f:
                    render_data = json.load(f)
            
            video_id = video_dir.name
            
            # Извлекаем статистики
            stats = {
                "video_id": video_id,
                "status": meta.get("status", "unknown"),
                "empty_reason": meta.get("empty_reason"),
            }
            
            # Из aggregated
            aggregated = npz_data.get("aggregated")
            if aggregated is not None:
                if isinstance(aggregated, np.ndarray) and aggregated.dtype == object:
                    try:
                        aggregated = aggregated.item()
                    except Exception:
                        aggregated = {}
                
                if isinstance(aggregated, dict):
                    for key in [
                        "avg_engagement", "avg_confidence", "avg_stress",
                        "gesture_rate_per_sec", "hands_visibility_ratio", "face_visibility_ratio",
                        "early_engagement_mean", "late_engagement_mean",
                        "early_confidence_mean", "late_confidence_mean",
                        "early_stress_mean", "late_stress_mean",
                    ]:
                        if key in aggregated:
                            val = aggregated[key]
                            if isinstance(val, (int, float)) and np.isfinite(val):
                                stats[key] = float(val)
            
            # Из render summary
            if render_data:
                summary = render_data.get("summary", {})
                if summary:
                    stats["frames_count"] = summary.get("frames_count", 0)
                    stats["landmarks_present_ratio"] = summary.get("landmarks_present_ratio", 0.0)
            
            # Проверяем sequence features
            seq_features = {}
            for key in [
                "seq_speech_activity_proxy", "seq_arm_openness", "seq_body_lean_angle",
                "seq_hand_motion_energy", "seq_blink_rate_short", "seq_fidgeting_energy",
            ]:
                arr = npz_data.get(key)
                if arr is not None:
                    if isinstance(arr, (list, np.ndarray)):
                        arr = np.asarray(arr, dtype=np.float32)
                        valid = arr[np.isfinite(arr)]
                        if valid.size > 0:
                            seq_features[key] = {
                                "mean": float(np.mean(valid)),
                                "std": float(np.std(valid)),
                                "min": float(np.min(valid)),
                                "max": float(np.max(valid)),
                                "median": float(np.median(valid)),
                                "valid_ratio": float(np.sum(np.isfinite(arr)) / len(arr)),
                            }
            
            stats["seq_features"] = seq_features
            
            # Проверяем целостность
            issues = []
            
            # Проверка обязательных ключей
            required_keys = ["frame_indices", "times_s", "landmarks_present", "aggregated"]
            missing = [k for k in required_keys if k not in npz_data]
            if missing:
                issues.append(f"Missing keys: {missing}")
            
            # Проверка размеров
            frame_indices = npz_data.get("frame_indices")
            times_s = npz_data.get("times_s")
            if frame_indices is not None and times_s is not None:
                fi_len = len(frame_indices) if isinstance(frame_indices, (list, np.ndarray)) else 0
                ts_len = len(times_s) if isinstance(times_s, (list, np.ndarray)) else 0
                if fi_len != ts_len:
                    issues.append(f"Size mismatch: frame_indices={fi_len}, times_s={ts_len}")
            
            # Проверка диапазонов
            if "avg_engagement" in stats:
                if not (0.0 <= stats["avg_engagement"] <= 1.0):
                    issues.append(f"avg_engagement out of range: {stats['avg_engagement']}")
            
            if "avg_confidence" in stats:
                if not (0.0 <= stats["avg_confidence"] <= 1.0):
                    issues.append(f"avg_confidence out of range: {stats['avg_confidence']}")
            
            if "avg_stress" in stats:
                if not (0.0 <= stats["avg_stress"] <= 1.0):
                    issues.append(f"avg_stress out of range: {stats['avg_stress']}")
            
            stats["issues"] = issues
            stats["has_issues"] = len(issues) > 0
            
            video_stats.append(stats)
            all_results.append({
                "video_id": video_id,
                "npz_data": npz_data,
                "meta": meta,
                "render": render_data,
                "stats": stats,
            })
            
        except Exception as e:
            print(f"Error processing {video_dir.name}: {e}")
            continue
    
    # Сравнительный анализ
    comparison = {}
    
    # Собираем метрики для сравнения
    metrics_to_compare = [
        "avg_engagement", "avg_confidence", "avg_stress",
        "gesture_rate_per_sec", "landmarks_present_ratio",
        "hands_visibility_ratio", "face_visibility_ratio",
    ]
    
    for metric in metrics_to_compare:
        values = []
        for stats in video_stats:
            if metric in stats:
                val = stats[metric]
                if isinstance(val, (int, float)) and np.isfinite(val):
                    values.append({
                        "video_id": stats["video_id"],
                        "value": float(val),
                    })
        
        if len(values) >= 2:
            vals = [v["value"] for v in values]
            comparison[metric] = {
                "count": len(values),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
                "values": values,
                "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0,  # coefficient of variation
            }
            
            # Выявляем аномалии (z-score > 2)
            mean_val = comparison[metric]["mean"]
            std_val = comparison[metric]["std"]
            if std_val > 0:
                for v in values:
                    z_score = abs((v["value"] - mean_val) / std_val)
                    if z_score > 2.0:
                        if "anomalies" not in comparison[metric]:
                            comparison[metric]["anomalies"] = []
                        comparison[metric]["anomalies"].append({
                            "video_id": v["video_id"],
                            "value": v["value"],
                            "z_score": float(z_score),
                        })
    
    return {
        "total_videos": len(video_stats),
        "videos_with_issues": sum(1 for s in video_stats if s["has_issues"]),
        "video_stats": video_stats,
        "comparison": comparison,
        "all_results": all_results,
    }


def print_analysis_report(analysis: Dict[str, Any]):
    """Выводит детальный отчет об анализе."""
    print("=" * 100)
    print("BEHAVIORAL COMPONENT - COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 100)
    print()
    
    print(f"Total videos analyzed: {analysis['total_videos']}")
    print(f"Videos with issues: {analysis['videos_with_issues']}")
    print()
    
    # Детальная статистика по каждому видео
    print("=" * 100)
    print("DETAILED VIDEO STATISTICS")
    print("=" * 100)
    
    for stats in analysis["video_stats"]:
        print(f"\nVideo: {stats['video_id']}")
        print(f"  Status: {stats.get('status', 'unknown')}")
        if stats.get('empty_reason'):
            print(f"  Empty reason: {stats['empty_reason']}")
        
        print(f"  Frames: {stats.get('frames_count', 'N/A')}")
        print(f"  Landmarks present ratio: {stats.get('landmarks_present_ratio', 0.0):.2%}")
        
        print(f"\n  Aggregated Metrics:")
        if "avg_engagement" in stats:
            print(f"    Avg engagement: {stats['avg_engagement']:.4f}")
        if "avg_confidence" in stats:
            print(f"    Avg confidence: {stats['avg_confidence']:.4f}")
        if "avg_stress" in stats:
            print(f"    Avg stress: {stats['avg_stress']:.4f}")
        if "gesture_rate_per_sec" in stats:
            print(f"    Gesture rate: {stats['gesture_rate_per_sec']:.4f} per sec")
        if "hands_visibility_ratio" in stats:
            print(f"    Hands visibility: {stats['hands_visibility_ratio']:.2%}")
        if "face_visibility_ratio" in stats:
            print(f"    Face visibility: {stats['face_visibility_ratio']:.2%}")
        
        # Temporal dynamics
        if "early_engagement_mean" in stats and "late_engagement_mean" in stats:
            early = stats["early_engagement_mean"]
            late = stats["late_engagement_mean"]
            change = late - early
            print(f"    Engagement change (early→late): {early:.4f} → {late:.4f} (Δ={change:+.4f})")
        
        if "early_confidence_mean" in stats and "late_confidence_mean" in stats:
            early = stats["early_confidence_mean"]
            late = stats["late_confidence_mean"]
            change = late - early
            print(f"    Confidence change (early→late): {early:.4f} → {late:.4f} (Δ={change:+.4f})")
        
        # Sequence features summary
        if stats.get("seq_features"):
            print(f"\n  Sequence Features (valid ratios):")
            for key, feat in stats["seq_features"].items():
                print(f"    {key}: valid={feat['valid_ratio']:.2%}, mean={feat['mean']:.4f}, std={feat['std']:.4f}")
        
        # Issues
        if stats.get("issues"):
            print(f"\n  ⚠️  Issues:")
            for issue in stats["issues"]:
                print(f"    - {issue}")
    
    # Сравнительный анализ
    if analysis["comparison"]:
        print("\n" + "=" * 100)
        print("CROSS-VIDEO COMPARISON")
        print("=" * 100)
        
        for metric, comp in analysis["comparison"].items():
            print(f"\n{metric}:")
            print(f"  Count: {comp['count']}")
            print(f"  Mean: {comp['mean']:.4f}")
            print(f"  Std: {comp['std']:.4f}")
            print(f"  Min: {comp['min']:.4f}")
            print(f"  Max: {comp['max']:.4f}")
            print(f"  Median: {comp['median']:.4f}")
            print(f"  CV (coefficient of variation): {comp['cv']:.4f}")
            
            if comp.get("anomalies"):
                print(f"  ⚠️  Anomalies (z-score > 2):")
                for anomaly in comp["anomalies"]:
                    print(f"    - {anomaly['video_id']}: value={anomaly['value']:.4f}, z-score={anomaly['z_score']:.2f}")
            
            # Показываем все значения для визуального сравнения
            print(f"  All values:")
            for v in comp["values"]:
                z_score = abs((v["value"] - comp["mean"]) / comp["std"]) if comp["std"] > 0 else 0
                marker = "⚠️" if z_score > 2.0 else "  "
                print(f"    {marker} {v['video_id']}: {v['value']:.4f}")
    
    # Оценка информативности
    print("\n" + "=" * 100)
    print("INFORMATIVENESS ASSESSMENT")
    print("=" * 100)
    
    # Проверяем вариативность метрик
    informative_metrics = []
    non_informative_metrics = []
    
    for metric, comp in analysis["comparison"].items():
        cv = comp.get("cv", 0.0)
        if cv > 0.1:  # CV > 10% считается информативным
            informative_metrics.append((metric, cv))
        else:
            non_informative_metrics.append((metric, cv))
    
    print(f"\nInformative metrics (CV > 10%):")
    for metric, cv in sorted(informative_metrics, key=lambda x: x[1], reverse=True):
        print(f"  ✅ {metric}: CV={cv:.4f}")
    
    if non_informative_metrics:
        print(f"\nLow-variability metrics (CV <= 10%):")
        for metric, cv in sorted(non_informative_metrics, key=lambda x: x[1]):
            print(f"  ⚠️  {metric}: CV={cv:.4f} (may be less informative)")
    
    # Проверяем покрытие landmarks
    landmarks_ratios = [s.get("landmarks_present_ratio", 0.0) for s in analysis["video_stats"]]
    if landmarks_ratios:
        avg_landmarks = np.mean(landmarks_ratios)
        print(f"\nLandmarks coverage:")
        print(f"  Average landmarks present ratio: {avg_landmarks:.2%}")
        print(f"  Min: {min(landmarks_ratios):.2%}")
        print(f"  Max: {max(landmarks_ratios):.2%}")
        
        if avg_landmarks < 0.1:
            print(f"  ⚠️  WARNING: Low average landmarks coverage ({avg_landmarks:.2%})")
            print(f"      This may indicate issues with face/pose detection or video content")
    
    print("\n" + "=" * 100)


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze all behavioral test results")
    parser.add_argument(
        "--results-base",
        type=str,
        default="DataProcessor/dp_results",
        help="Base path to results"
    )
    
    args = parser.parse_args()
    
    print("Analyzing behavioral test results...")
    analysis = analyze_behavioral_results(args.results_base)
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return 1
    
    print_analysis_report(analysis)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

