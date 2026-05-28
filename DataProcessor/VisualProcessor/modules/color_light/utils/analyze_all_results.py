#!/usr/bin/env python3
"""
Анализ всех результатов тестирования color_light компонента.
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


def analyze_color_light_results(results_base_path: str) -> Dict[str, Any]:
    """Анализирует все результаты color_light тестов."""
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}
    
    # Находим все color_light результаты
    all_results = []
    video_stats = []
    
    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_color_light"):
            continue
        
        run_dir = video_dir / video_dir.name
        color_light_dir = run_dir / "color_light"
        
        if not color_light_dir.exists():
            continue
        
        npz_path = color_light_dir / "color_light_features.npz"
        render_path = color_light_dir / "_render" / "render_context.json"
        
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
            
            # Из video_features
            video_features = npz_data.get("video_features", {})
            if isinstance(video_features, np.ndarray) and video_features.dtype == object:
                try:
                    video_features = video_features.item()
                except Exception:
                    video_features = {}
            
            if isinstance(video_features, dict):
                for key in [
                    "color_distribution_entropy", "color_distribution_gini",
                    "global_brightness_change_speed", "global_color_change_speed",
                    "cinematic_lighting_score", "professional_look_score",
                    "style_teal_orange_prob", "style_film_prob", "style_vintage_prob",
                ]:
                    if key in video_features:
                        val = video_features[key]
                        if isinstance(val, (int, float)) and np.isfinite(val):
                            stats[key] = float(val)
            
            # Из aggregated
            aggregated = npz_data.get("aggregated", {})
            if isinstance(aggregated, np.ndarray) and aggregated.dtype == object:
                try:
                    aggregated = aggregated.item()
                except Exception:
                    aggregated = {}
            
            if isinstance(aggregated, dict):
                # Статистики по компактным фичам
                for stat_key in ["mean", "std", "min", "max", "median"]:
                    if stat_key in aggregated:
                        stat_dict = aggregated[stat_key]
                        if isinstance(stat_dict, dict):
                            for feature_key in [
                                "hue_mean_norm", "hue_std_norm", "sat_mean_norm", "val_mean_norm",
                                "L_mean_norm", "global_contrast_norm", "colorfulness_norm",
                            ]:
                                if feature_key in stat_dict:
                                    val = stat_dict[feature_key]
                                    if isinstance(val, (int, float)) and np.isfinite(val):
                                        stats[f"{feature_key}_{stat_key}"] = float(val)
            
            # Из render summary
            if render_data:
                summary = render_data.get("summary", {})
                if summary:
                    stats["frames_count"] = summary.get("frames_count", 0)
                    stats["scenes_count"] = summary.get("scenes_count", 0)
            
            # Проверяем frame_compact_features
            frame_compact_features = npz_data.get("frame_compact_features")
            if frame_compact_features is not None:
                if isinstance(frame_compact_features, np.ndarray):
                    stats["compact_features_shape"] = list(frame_compact_features.shape)
                    # Проверка диапазона [0, 1]
                    valid_mask = np.isfinite(frame_compact_features)
                    if np.any(valid_mask):
                        stats["compact_features_min"] = float(np.nanmin(frame_compact_features))
                        stats["compact_features_max"] = float(np.nanmax(frame_compact_features))
                        stats["compact_features_mean"] = float(np.nanmean(frame_compact_features))
                        stats["compact_features_std"] = float(np.nanstd(frame_compact_features))
            
            # Проверяем scenes
            scenes = npz_data.get("scenes", {})
            if isinstance(scenes, np.ndarray) and scenes.dtype == object:
                try:
                    scenes = scenes.item()
                except Exception:
                    scenes = {}
            
            if isinstance(scenes, dict):
                stats["scenes_count_actual"] = len(scenes)
            
            video_stats.append(stats)
            all_results.append({
                "video_id": video_id,
                "npz_data": npz_data,
                "meta": meta,
                "render": render_data,
                "stats": stats,
            })
            
        except Exception as e:
            print(f"Error processing {video_id}: {e}", file=sys.stderr)
            continue
    
    # Анализ метрик
    metrics_analysis = {}
    
    # Собираем все значения для каждой метрики
    metric_values = defaultdict(list)
    for stats in video_stats:
        if stats.get("status") != "ok":
            continue
        for key, value in stats.items():
            if key not in ["video_id", "status", "empty_reason"] and isinstance(value, (int, float)):
                metric_values[key].append(value)
    
    # Вычисляем статистики для каждой метрики
    for metric_name, values in metric_values.items():
        if not values:
            continue
        
        values_arr = np.array(values)
        valid_values = values_arr[np.isfinite(values_arr)]
        
        if len(valid_values) == 0:
            continue
        
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        cv = std_val / mean_val if mean_val != 0 else np.inf
        
        metrics_analysis[metric_name] = {
            "count": len(valid_values),
            "mean": float(mean_val),
            "std": float(std_val),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "median": float(np.median(valid_values)),
            "cv": float(cv) if np.isfinite(cv) else None,
            "is_informative": abs(cv) > 0.1 if np.isfinite(cv) else False,
        }
        
        # Z-scores для выявления аномалий
        if len(valid_values) > 1 and std_val > 0:
            z_scores = np.abs((valid_values - mean_val) / std_val)
            anomalies = np.where(z_scores > 2.0)[0]
            if len(anomalies) > 0:
                metrics_analysis[metric_name]["anomalies"] = [
                    {
                        "video_id": video_stats[i]["video_id"],
                        "value": float(valid_values[i]),
                        "z_score": float(z_scores[i]),
                    }
                    for i in anomalies
                ]
    
    return {
        "total_videos": len(all_results),
        "successful_videos": len([v for v in video_stats if v.get("status") == "ok"]),
        "video_stats": video_stats,
        "metrics_analysis": metrics_analysis,
    }


def print_report(analysis: Dict[str, Any]):
    """Выводит отчет по анализу."""
    print("=" * 80)
    print("Color Light Component - Comprehensive Analysis Report")
    print("=" * 80)
    print()
    
    print(f"Total videos processed: {analysis['total_videos']}")
    print(f"Successful videos: {analysis['successful_videos']}")
    print()
    
    # Анализ метрик
    metrics = analysis.get("metrics_analysis", {})
    if metrics:
        print("=" * 80)
        print("Metrics Analysis")
        print("=" * 80)
        print()
        
        # Группируем по категориям
        categories = {
            "Video-level features": [
                "color_distribution_entropy", "color_distribution_gini",
                "global_brightness_change_speed", "global_color_change_speed",
                "cinematic_lighting_score", "professional_look_score",
            ],
            "Style probabilities": [
                "style_teal_orange_prob", "style_film_prob", "style_vintage_prob",
            ],
            "Frame compact features (mean)": [
                "hue_mean_norm_mean", "sat_mean_norm_mean", "val_mean_norm_mean",
                "L_mean_norm_mean", "global_contrast_norm_mean", "colorfulness_norm_mean",
            ],
        }
        
        for category, metric_list in categories.items():
            print(f"\n{category}:")
            print("-" * 80)
            for metric_name in metric_list:
                if metric_name in metrics:
                    m = metrics[metric_name]
                    informative = "✅" if m.get("is_informative") else "⚠️"
                    print(f"  {informative} {metric_name}:")
                    print(f"    Range: [{m['min']:.4f}, {m['max']:.4f}]")
                    cv_str = f"{m['cv']:.4f}" if m['cv'] is not None else "N/A"
                    print(f"    Mean: {m['mean']:.4f} ± {m['std']:.4f} (CV: {cv_str})")
                    print(f"    Median: {m['median']:.4f}")
                    if "anomalies" in m:
                        print(f"    ⚠️  Anomalies: {len(m['anomalies'])}")
                        for anomaly in m["anomalies"][:3]:  # Show first 3
                            print(f"      - {anomaly['video_id']}: {anomaly['value']:.4f} (z={anomaly['z_score']:.2f})")
        
        # Информативные метрики
        informative_metrics = [
            (name, m) for name, m in metrics.items()
            if m.get("is_informative") and m.get("cv") is not None and m["cv"] > 0.1
        ]
        informative_metrics.sort(key=lambda x: x[1]["cv"], reverse=True)
        
        if informative_metrics:
            print("\n" + "=" * 80)
            print("Most Informative Metrics (CV > 10%)")
            print("=" * 80)
            for name, m in informative_metrics[:10]:
                print(f"  {name}: CV={m['cv']:.4f}, Range=[{m['min']:.4f}, {m['max']:.4f}]")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze all color_light test results")
    parser.add_argument("--results-base", type=str, required=True,
                        help="Base path to results (e.g., DataProcessor/dp_results)")
    
    args = parser.parse_args()
    
    analysis = analyze_color_light_results(args.results_base)
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}", file=sys.stderr)
        return 1
    
    print_report(analysis)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

