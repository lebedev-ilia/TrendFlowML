#!/usr/bin/env python3
"""
Анализ всех результатов тестирования frames_composition компонента.
Проверяет целостность, нормальность и информативность значений.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import argparse

# Add VisualProcessor to path
vp_root = Path(__file__).resolve().parent.parent.parent
if str(vp_root) not in sys.path:
    sys.path.insert(0, str(vp_root))

from utils.renderer import load_npz, extract_meta


def analyze_frames_composition_results(results_base_path: str) -> Dict[str, Any]:
    """Анализирует все результаты frames_composition тестов."""
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}
    
    # Находим все frames_composition результаты
    all_results = []
    video_stats = []
    
    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_frames_composition"):
            continue
        
        run_dir = video_dir / video_dir.name
        frames_composition_dir = run_dir / "frames_composition"
        
        if not frames_composition_dir.exists():
            continue
        
        npz_path = frames_composition_dir / "frames_composition.npz"
        render_path = frames_composition_dir / "_render" / "render_context.json"
        
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
            
            # Из meta
            if "total_frames" in meta:
                stats["total_frames"] = float(meta["total_frames"])
            if "processed_frames" in meta:
                stats["processed_frames"] = float(meta["processed_frames"])
            
            # Из feature_values (video-level features)
            feature_values = npz_data.get("feature_values")
            feature_names = npz_data.get("feature_names")
            if isinstance(feature_values, np.ndarray) and isinstance(feature_names, np.ndarray):
                # Извлекаем некоторые ключевые метрики
                for i, name in enumerate(feature_names):
                    if i < len(feature_values):
                        val = feature_values[i]
                        if isinstance(name, (str, np.str_)):
                            name_str = str(name)
                            if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                                stats[f"feature_{name_str}"] = float(val)
            
            # Из frame_feature_values (per-frame features)
            frame_feature_values = npz_data.get("frame_feature_values")
            frame_feature_names = npz_data.get("frame_feature_names")
            if isinstance(frame_feature_values, np.ndarray) and isinstance(frame_feature_names, np.ndarray):
                if frame_feature_values.ndim == 2:
                    # Статистики по кадрам
                    stats["num_frames"] = int(frame_feature_values.shape[0])
                    stats["num_features"] = int(frame_feature_values.shape[1])
                    
                    # Вычисляем средние значения по некоторым ключевым фичам
                    for i, name in enumerate(frame_feature_names):
                        if i < frame_feature_values.shape[1]:
                            col = frame_feature_values[:, i]
                            finite_mask = np.isfinite(col)
                            if np.any(finite_mask):
                                mean_val = float(np.nanmean(col))
                                if isinstance(name, (str, np.str_)):
                                    name_str = str(name)
                                    stats[f"frame_feature_{name_str}_mean"] = mean_val
            
            # Из frame_feature_present_ratio
            frame_feature_present_ratio = npz_data.get("frame_feature_present_ratio")
            if isinstance(frame_feature_present_ratio, np.ndarray):
                stats["avg_feature_present_ratio"] = float(np.nanmean(frame_feature_present_ratio))
            
            video_stats.append(stats)
            all_results.append({
                "video_id": video_id,
                "npz_data": npz_data,
                "meta": meta,
                "render": render_data,
            })
        except Exception as e:
            print(f"Error processing {video_id}: {e}", file=sys.stderr)
            continue
    
    return {
        "total_videos": len(all_results),
        "successful_videos": len([s for s in video_stats if s.get("status") == "ok"]),
        "video_stats": video_stats,
        "all_results": all_results,
    }


def print_analysis_report(results: Dict[str, Any]):
    """Выводит отчет анализа."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    total = results["total_videos"]
    successful = results["successful_videos"]
    video_stats = results["video_stats"]
    
    print("=" * 80)
    print("Frames Composition Component - Comprehensive Analysis Report")
    print("=" * 80)
    print()
    print(f"Total videos processed: {total}")
    print(f"Successful videos: {successful}")
    print()
    
    if not video_stats:
        print("No video statistics available.")
        return
    
    # Собираем все метрики
    metrics_values = defaultdict(list)
    
    for stats in video_stats:
        if stats.get("status") != "ok":
            continue
        
        video_id = stats["video_id"]
        for key, value in stats.items():
            if key not in ["video_id", "status", "empty_reason"] and isinstance(value, (int, float)) and np.isfinite(value):
                metrics_values[key].append((video_id, value))
    
    # Группируем метрики по категориям
    categories = {
        "Frame statistics": ["total_frames", "processed_frames", "num_frames", "num_features"],
        "Feature statistics": [k for k in metrics_values.keys() if k.startswith("feature_")],
        "Frame feature statistics": [k for k in metrics_values.keys() if k.startswith("frame_feature_")],
        "Other": [k for k in metrics_values.keys() if k not in ["total_frames", "processed_frames", "num_frames", "num_features", "avg_feature_present_ratio"] and not k.startswith("feature_") and not k.startswith("frame_feature_")],
    }
    
    metrics_analysis = {}
    
    for metric, values_list in metrics_values.items():
        if not values_list:
            continue
        
        values = [v[1] for v in values_list]
        values_arr = np.array(values)
        
        # Фильтруем NaN и Inf
        valid_mask = np.isfinite(values_arr)
        if not np.any(valid_mask):
            continue
        
        valid_values = values_arr[valid_mask]
        
        if len(valid_values) == 0:
            continue
        
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        median_val = np.median(valid_values)
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        
        # Коэффициент вариации
        cv = std_val / mean_val if mean_val != 0 else None
        
        # Z-scores для аномалий
        z_scores = []
        anomalies = []
        if len(valid_values) > 2 and std_val > 0:
            z_scores = np.abs((valid_values - mean_val) / std_val)
            anomaly_threshold = 2.0
            for i, (video_id, value) in enumerate(values_list):
                if valid_mask[i] and z_scores[np.where(valid_mask)[0][i]] > anomaly_threshold:
                    anomalies.append({
                        "video_id": video_id,
                        "value": float(value),
                        "z_score": float(z_scores[np.where(valid_mask)[0][i]])
                    })
        
        metrics_analysis[metric] = {
            "count": len(valid_values),
            "mean": float(mean_val),
            "std": float(std_val),
            "median": float(median_val),
            "min": float(min_val),
            "max": float(max_val),
            "cv": float(cv) if cv is not None else None,
            "anomalies": anomalies[:5] if anomalies else [],
        }
    
    # Выводим метрики по категориям
    print("=" * 80)
    print("Metrics Analysis")
    print("=" * 80)
    print()
    
    for category, metric_keys in categories.items():
        if not any(k in metrics_analysis for k in metric_keys):
            continue
        
        print(category + ":")
        print("-" * 80)
        
        for metric in metric_keys:
            if metric not in metrics_analysis:
                continue
            
            m = metrics_analysis[metric]
            cv = m["cv"]
            
            if cv is None:
                cv_str = "N/A"
                status = "❌"
            elif cv < 0.1:
                cv_str = f"{cv:.4f}"
                status = "❌"
            elif cv < 0.5:
                cv_str = f"{cv:.4f}"
                status = "✅"
            else:
                cv_str = f"{cv:.4f}"
                status = "✅✅"
            
            print(f"  {status} {metric}:")
            print(f"    Range: [{m['min']:.4f}, {m['max']:.4f}]")
            print(f"    Mean: {m['mean']:.4f} ± {m['std']:.4f} (CV: {cv_str})")
            print(f"    Median: {m['median']:.4f}")
            
            if m['anomalies']:
                print(f"    ⚠️  Anomalies: {len(m['anomalies'])}")
                for anomaly in m['anomalies'][:3]:  # Показываем топ-3
                    print(f"      - {anomaly['video_id']}: {anomaly['value']:.4f} (z={anomaly['z_score']:.2f})")
        print()
    
    # Топ информативных метрик
    print("=" * 80)
    print("Most Informative Metrics (CV > 10%)")
    print("=" * 80)
    
    informative_metrics = []
    for metric, m in metrics_analysis.items():
        if m['cv'] is not None and m['cv'] > 0.1:
            informative_metrics.append((metric, m['cv'], m['min'], m['max']))
    
    informative_metrics.sort(key=lambda x: x[1], reverse=True)
    
    for metric, cv, min_val, max_val in informative_metrics[:10]:
        print(f"  {metric}: CV={cv:.4f}, Range=[{min_val:.4f}, {max_val:.4f}]")
    
    print()
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze frames_composition component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    
    args = parser.parse_args()
    
    results = analyze_frames_composition_results(args.results_base)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    print_analysis_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

