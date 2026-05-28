#!/usr/bin/env python3
"""
Анализ всех результатов тестирования detalize_face компонента.
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


def analyze_detalize_face_results(results_base_path: str) -> Dict[str, Any]:
    """Анализирует все результаты detalize_face тестов."""
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}
    
    # Находим все detalize_face результаты
    all_results = []
    video_stats = []
    
    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_detalize_face"):
            continue
        
        run_dir = video_dir / video_dir.name
        detalize_face_dir = run_dir / "detalize_face"
        
        if not detalize_face_dir.exists():
            continue
        
        npz_path = detalize_face_dir / "detalize_face.npz"
        render_path = detalize_face_dir / "_render" / "render_context.json"
        
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
            aggregated = npz_data.get("aggregated", {})
            if isinstance(aggregated, np.ndarray) and aggregated.ndim == 0:
                try:
                    aggregated = aggregated.item()
                except Exception:
                    aggregated = {}
            
            if isinstance(aggregated, dict):
                # Ключевые метрики для анализа
                key_metrics = [
                    "valid_frames", "axis_frames", "face_present_ratio", "processed_ratio",
                    "primary_valid_ratio", "compact_dim", "compact_l2_mean", "compact_l2_std"
                ]
                
                for key in key_metrics:
                    if key in aggregated:
                        val = aggregated[key]
                        if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                            stats[key] = float(val)
            
            # Из summary
            summary = npz_data.get("summary", {})
            if isinstance(summary, np.ndarray) and summary.ndim == 0:
                try:
                    summary = summary.item()
                except Exception:
                    summary = {}
            
            if isinstance(summary, dict):
                for key in ["total_frames", "processed_frames", "frames_with_faces", "total_faces", "primary_faces", "avg_faces_per_frame"]:
                    if key in summary:
                        val = summary[key]
                        if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                            stats[key] = float(val)
            
            # Из frame_indices и times_s
            frame_indices = npz_data.get("frame_indices")
            times_s = npz_data.get("times_s")
            
            if frame_indices is not None and isinstance(frame_indices, np.ndarray):
                stats["frames_count"] = len(frame_indices)
            
            if times_s is not None and isinstance(times_s, np.ndarray) and len(times_s) > 0:
                stats["duration_s"] = float(times_s[-1] - times_s[0])
                stats["duration_min"] = stats["duration_s"] / 60.0
            
            # Из primary_compact_features
            primary_compact_features = npz_data.get("primary_compact_features")
            if primary_compact_features is not None and isinstance(primary_compact_features, np.ndarray):
                if primary_compact_features.ndim == 2:
                    stats["compact_features_shape"] = list(primary_compact_features.shape)
                    # Статистики по компактным фичам
                    valid_mask = np.isfinite(primary_compact_features)
                    if np.any(valid_mask):
                        stats["compact_features_min"] = float(np.nanmin(primary_compact_features))
                        stats["compact_features_max"] = float(np.nanmax(primary_compact_features))
                        stats["compact_features_mean"] = float(np.nanmean(primary_compact_features))
                        stats["compact_features_std"] = float(np.nanstd(primary_compact_features))
            
            # Из face_present и primary_valid
            face_present = npz_data.get("face_present")
            primary_valid = npz_data.get("primary_valid")
            
            if face_present is not None and isinstance(face_present, np.ndarray):
                stats["face_present_count"] = int(np.sum(face_present))
                stats["face_present_ratio_actual"] = float(np.mean(face_present))
            
            if primary_valid is not None and isinstance(primary_valid, np.ndarray):
                stats["primary_valid_count"] = int(np.sum(primary_valid))
                stats["primary_valid_ratio_actual"] = float(np.mean(primary_valid))
            
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
    print("Detalize Face Component - Comprehensive Analysis Report")
    print("=" * 80)
    print()
    print(f"Total videos processed: {results['total_videos']}")
    print(f"Successful videos: {results['successful_videos']}")
    print()
    
    metrics_analysis = results["metrics_analysis"]
    
    # Группируем метрики по категориям
    categories = {
        "Frame statistics": ["frames_count", "total_frames", "processed_frames", "frames_with_faces"],
        "Face detection": ["total_faces", "primary_faces", "avg_faces_per_frame", "face_present_ratio", "face_present_ratio_actual"],
        "Processing statistics": ["processed_ratio", "primary_valid_ratio", "primary_valid_ratio_actual", "valid_frames"],
        "Compact features": ["compact_features_mean", "compact_features_std", "compact_l2_mean", "compact_l2_std"],
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
    parser = argparse.ArgumentParser(description="Analyze detalize_face component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    
    args = parser.parse_args()
    
    results = analyze_detalize_face_results(args.results_base)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    print_analysis_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

