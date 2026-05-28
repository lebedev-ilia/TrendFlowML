#!/usr/bin/env python3
"""
Анализ всех результатов тестирования emotion_face компонента.
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


def analyze_emotion_face_results(results_base_path: str) -> Dict[str, Any]:
    """Анализирует все результаты emotion_face тестов."""
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}
    
    # Находим все emotion_face результаты
    all_results = []
    video_stats = []
    
    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_emotion_face"):
            continue
        
        run_dir = video_dir / video_dir.name
        emotion_face_dir = run_dir / "emotion_face"
        
        if not emotion_face_dir.exists():
            continue
        
        npz_path = emotion_face_dir / "emotion_face.npz"
        render_path = emotion_face_dir / "_render" / "render_context.json"
        
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
            
            # Из summary
            summary = npz_data.get("summary", {})
            if isinstance(summary, np.ndarray) and summary.ndim == 0:
                try:
                    summary = summary.item()
                except Exception:
                    summary = {}
            
            if isinstance(summary, dict):
                for key in ["frames_count", "processed_frames", "faces_found_frames", "keyframes_count", "transitions_count"]:
                    if key in summary:
                        val = summary[key]
                        if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                            stats[key] = float(val)
                
                # Статистики по эмоциям
                for key in ["valence_mean", "arousal_mean", "intensity_mean", "emotion_confidence_mean"]:
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
            
            # Из valence, arousal, intensity, emotion_confidence
            for key in ["valence", "arousal", "intensity", "emotion_confidence"]:
                arr = npz_data.get(key)
                if arr is not None and isinstance(arr, np.ndarray):
                    valid_mask = np.isfinite(arr)
                    if np.any(valid_mask):
                        valid_arr = arr[valid_mask]
                        stats[f"{key}_mean"] = float(np.mean(valid_arr))
                        stats[f"{key}_std"] = float(np.std(valid_arr))
                        stats[f"{key}_min"] = float(np.min(valid_arr))
                        stats[f"{key}_max"] = float(np.max(valid_arr))
                        stats[f"{key}_median"] = float(np.median(valid_arr))
            
            # Из emotion_probs
            emotion_probs = npz_data.get("emotion_probs")
            if emotion_probs is not None and isinstance(emotion_probs, np.ndarray) and emotion_probs.ndim == 2:
                valid_mask = np.isfinite(emotion_probs)
                if np.any(valid_mask):
                    # Средние вероятности по каждому классу эмоций
                    emotion_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
                    for i, emotion_name in enumerate(emotion_names):
                        if i < emotion_probs.shape[1]:
                            emotion_col = emotion_probs[:, i]
                            valid_col = emotion_col[np.isfinite(emotion_col)]
                            if len(valid_col) > 0:
                                stats[f"emotion_{emotion_name.lower()}_mean"] = float(np.mean(valid_col))
            
            # Из dominant_emotion_id
            dominant_emotion_id = npz_data.get("dominant_emotion_id")
            if dominant_emotion_id is not None and isinstance(dominant_emotion_id, np.ndarray):
                valid_mask = (dominant_emotion_id >= 0) & (dominant_emotion_id < 8)
                if np.any(valid_mask):
                    valid_ids = dominant_emotion_id[valid_mask]
                    emotion_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
                    for i, emotion_name in enumerate(emotion_names):
                        count = int(np.sum(valid_ids == i))
                        if count > 0:
                            stats[f"dominant_{emotion_name.lower()}_count"] = count
            
            # Из face_present и processed_mask
            face_present = npz_data.get("face_present")
            processed_mask = npz_data.get("processed_mask")
            
            if face_present is not None and isinstance(face_present, np.ndarray):
                stats["face_present_count"] = int(np.sum(face_present))
                stats["face_present_ratio"] = float(np.mean(face_present))
            
            if processed_mask is not None and isinstance(processed_mask, np.ndarray):
                stats["processed_count"] = int(np.sum(processed_mask))
                stats["processed_ratio"] = float(np.mean(processed_mask))
            
            # Из keyframes
            keyframes = npz_data.get("keyframes")
            if keyframes is not None:
                if isinstance(keyframes, np.ndarray) and keyframes.ndim == 0:
                    keyframes = keyframes.item()
                if isinstance(keyframes, (list, np.ndarray)):
                    stats["keyframes_count"] = len(keyframes)
                    # Подсчет типов keyframes
                    peaks = sum(1 for kf in keyframes if isinstance(kf, dict) and kf.get("type") == "emotion_peak")
                    transitions = sum(1 for kf in keyframes if isinstance(kf, dict) and kf.get("type") == "transition")
                    stats["keyframes_peaks"] = peaks
                    stats["keyframes_transitions"] = transitions
            
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
    print("Emotion Face Component - Comprehensive Analysis Report")
    print("=" * 80)
    print()
    print(f"Total videos processed: {results['total_videos']}")
    print(f"Successful videos: {results['successful_videos']}")
    print()
    
    metrics_analysis = results["metrics_analysis"]
    
    # Группируем метрики по категориям
    categories = {
        "Frame statistics": ["frames_count", "processed_frames", "faces_found_frames"],
        "Face detection": ["face_present_count", "face_present_ratio", "processed_count", "processed_ratio"],
        "Emotion metrics": ["valence_mean", "arousal_mean", "intensity_mean", "emotion_confidence_mean"],
        "Emotion variance": ["valence_std", "arousal_std", "intensity_std", "emotion_confidence_std"],
        "Keyframes": ["keyframes_count", "keyframes_peaks", "keyframes_transitions"],
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
    parser = argparse.ArgumentParser(description="Analyze emotion_face component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    
    args = parser.parse_args()
    
    results = analyze_emotion_face_results(args.results_base)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    print_analysis_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

