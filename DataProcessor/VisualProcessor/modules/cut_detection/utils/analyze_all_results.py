#!/usr/bin/env python3
"""
Анализ всех результатов тестирования cut_detection компонента.
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


def analyze_cut_detection_results(results_base_path: str) -> Dict[str, Any]:
    """Анализирует все результаты cut_detection тестов."""
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}
    
    # Находим все cut_detection результаты
    all_results = []
    video_stats = []
    
    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_cut_detection"):
            continue
        
        run_dir = video_dir / video_dir.name
        cut_detection_dir = run_dir / "cut_detection"
        
        if not cut_detection_dir.exists():
            continue
        
        # Ищем последний NPZ файл
        npz_files = list(cut_detection_dir.glob("cut_detection_features_*.npz"))
        if not npz_files:
            continue
        
        npz_path = max(npz_files, key=lambda p: p.stat().st_mtime)
        render_path = cut_detection_dir / "_render" / "render_context.json"
        
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
            
            # Из features
            features = npz_data.get("features", {})
            if isinstance(features, np.ndarray) and features.ndim == 0:
                try:
                    features = features.item()
                except Exception:
                    features = {}
            
            if isinstance(features, dict):
                # Ключевые метрики для анализа
                key_metrics = [
                    "hard_cuts_count", "hard_cuts_per_minute", "hard_cut_strength_mean",
                    "fade_in_count", "fade_out_count", "dissolve_count", "avg_fade_duration",
                    "motion_cuts_count", "motion_cut_intensity_score", "flow_spike_ratio",
                    "whip_pan_transitions_count", "zoom_transition_count", "speed_ramp_cuts_count",
                    "jump_cuts_count", "jump_cut_intensity", "jump_cut_ratio_per_minute",
                    "cuts_per_minute", "median_cut_interval", "min_cut_interval", "max_cut_interval",
                    "cut_interval_std", "cut_interval_cv", "cut_interval_entropy",
                    "cut_rhythm_uniformity_score", "avg_shot_length", "median_shot_length",
                    "short_shots_ratio", "long_shots_ratio", "very_long_shots_count",
                    "scene_count", "avg_scene_length_shots", "scene_to_shot_ratio",
                    "audio_cut_alignment_score", "audio_spike_cut_ratio",
                    "edit_style_hard_cut_prob", "edit_style_fade_prob", "edit_style_dissolve_prob",
                    "edit_style_fast_prob", "edit_style_slow_prob", "edit_style_cinematic_prob",
                    "edit_style_meme_prob", "edit_style_social_prob", "edit_style_high_action_prob",
                ]
                
                for key in key_metrics:
                    if key in features:
                        val = features[key]
                        if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                            stats[key] = float(val)
            
            # Из detections
            detections = npz_data.get("detections", {})
            if isinstance(detections, np.ndarray) and detections.ndim == 0:
                try:
                    detections = detections.item()
                except Exception:
                    detections = {}
            
            if isinstance(detections, dict):
                # Подсчитываем детекции
                for key in ["hard_cut_pos", "motion_cut_pos", "jump_cut_pos"]:
                    if key in detections:
                        val = detections[key]
                        if isinstance(val, (list, np.ndarray)):
                            stats[f"{key}_count"] = len(val)
                        elif isinstance(val, np.ndarray) and val.ndim == 0:
                            stats[f"{key}_count"] = 0
                
                # Soft events
                if "soft_events" in detections:
                    soft_events = detections["soft_events"]
                    if isinstance(soft_events, (list, np.ndarray)):
                        stats["soft_events_count"] = len(soft_events)
            
            # Из frame_indices и times_s
            frame_indices = npz_data.get("frame_indices")
            times_s = npz_data.get("times_s")
            
            if frame_indices is not None and isinstance(frame_indices, np.ndarray):
                stats["frames_count"] = len(frame_indices)
            
            if times_s is not None and isinstance(times_s, np.ndarray) and len(times_s) > 0:
                stats["duration_s"] = float(times_s[-1] - times_s[0])
                stats["duration_min"] = stats["duration_s"] / 60.0
            
            # Из render summary
            if render_data:
                summary = render_data.get("summary", {})
                if summary:
                    if "frames_count" not in stats:
                        stats["frames_count"] = summary.get("frames_count", 0)
            
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
        for key, value in stats.items():
            if key not in ["video_id", "status", "empty_reason"] and isinstance(value, (int, float)):
                metric_values[key].append((stats["video_id"], value))
    
    # Вычисляем статистики для каждой метрики
    for metric, values_list in metric_values.items():
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
                if valid_mask[i] and z_scores[valid_mask][i] > anomaly_threshold:
                    anomalies.append({
                        "video_id": video_id,
                        "value": float(value),
                        "z_score": float(z_scores[valid_mask][i])
                    })
        
        metrics_analysis[metric] = {
            "count": len(valid_values),
            "mean": float(mean_val),
            "std": float(std_val),
            "median": float(median_val),
            "min": float(min_val),
            "max": float(max_val),
            "cv": float(cv) if cv is not None else None,
            "anomalies": anomalies[:5] if anomalies else [],  # Топ-5 аномалий
        }
    
    return {
        "total_videos": len(all_results),
        "successful_videos": len([r for r in all_results if r["stats"].get("status") == "ok"]),
        "video_stats": video_stats,
        "metrics_analysis": metrics_analysis,
    }


def print_analysis_report(results: Dict[str, Any]):
    """Вывести отчет анализа."""
    print("=" * 80)
    print("Cut Detection Component - Comprehensive Analysis Report")
    print("=" * 80)
    print()
    print(f"Total videos processed: {results['total_videos']}")
    print(f"Successful videos: {results['successful_videos']}")
    print()
    
    metrics_analysis = results["metrics_analysis"]
    
    # Группируем метрики по категориям
    categories = {
        "Hard cuts": ["hard_cuts_count", "hard_cuts_per_minute", "hard_cut_strength_mean"],
        "Soft transitions": ["fade_in_count", "fade_out_count", "dissolve_count", "avg_fade_duration"],
        "Motion cuts": ["motion_cuts_count", "motion_cut_intensity_score", "flow_spike_ratio"],
        "Jump cuts": ["jump_cuts_count", "jump_cut_intensity", "jump_cut_ratio_per_minute"],
        "Cut timing": ["cuts_per_minute", "median_cut_interval", "cut_interval_std", "cut_interval_cv", "cut_interval_entropy"],
        "Shot statistics": ["avg_shot_length", "median_shot_length", "short_shots_ratio", "long_shots_ratio"],
        "Scene statistics": ["scene_count", "avg_scene_length_shots", "scene_to_shot_ratio"],
        "Edit style": ["edit_style_hard_cut_prob", "edit_style_fade_prob", "edit_style_dissolve_prob", 
                       "edit_style_fast_prob", "edit_style_slow_prob", "edit_style_cinematic_prob"],
    }
    
    print("=" * 80)
    print("Metrics Analysis")
    print("=" * 80)
    print()
    
    for category, metric_keys in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        
        found_metrics = []
        for metric in metric_keys:
            if metric in metrics_analysis:
                found_metrics.append(metric)
        
        if not found_metrics:
            print("  (No metrics found)")
            continue
        
        for metric in found_metrics:
            m = metrics_analysis[metric]
            cv_str = f"{m['cv']:.4f}" if isinstance(m['cv'], (float, int)) and np.isfinite(m['cv']) else "N/A"
            
            # Определяем статус информативности
            if m['cv'] is not None and m['cv'] > 0.1:
                status = "✅"
            elif m['cv'] is not None and m['cv'] > 0.05:
                status = "⚠️"
            else:
                status = "❌"
            
            print(f"  {status} {metric}:")
            print(f"    Range: [{m['min']:.4f}, {m['max']:.4f}]")
            print(f"    Mean: {m['mean']:.4f} ± {m['std']:.4f} (CV: {cv_str})")
            print(f"    Median: {m['median']:.4f}")
            
            if m['anomalies']:
                print(f"    ⚠️  Anomalies: {len(m['anomalies'])}")
                for anomaly in m['anomalies'][:3]:  # Показываем топ-3
                    print(f"      - {anomaly['video_id']}: {anomaly['value']:.4f} (z={anomaly['z_score']:.2f})")
    
    # Топ информативных метрик
    print()
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
    parser = argparse.ArgumentParser(description="Analyze cut_detection component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    
    args = parser.parse_args()
    
    results = analyze_cut_detection_results(args.results_base)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    print_analysis_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

