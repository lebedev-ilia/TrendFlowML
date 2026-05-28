#!/usr/bin/env python3
"""
Анализ всех результатов тестирования scene_classification компонента.
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


def analyze_scene_classification_results(results_base_path: str) -> Dict[str, Any]:
    """Анализирует все результаты scene_classification тестов."""
    results_base = Path(results_base_path)
    youtube_dir = results_base / "youtube"
    
    if not youtube_dir.exists():
        return {"error": "Results directory not found"}
    
    # Находим все scene_classification результаты
    all_results = []
    video_stats = []
    
    # Собираем метрики по всем видео
    features_by_name = defaultdict(list)
    num_frames_list = []
    num_scenes_list = []
    scene_duration_list = []
    frame_top1_prob_list = []
    frame_entropy_list = []
    
    for video_dir in youtube_dir.iterdir():
        if not video_dir.is_dir() or not video_dir.name.startswith("test_scene_classification"):
            continue
        
        run_dir = video_dir / video_dir.name
        scene_classification_dir = run_dir / "scene_classification"
        
        if not scene_classification_dir.exists():
            continue
        
        npz_path = scene_classification_dir / "scene_classification_features.npz"
        render_path = scene_classification_dir / "_render" / "render_context.json"
        
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
            status = meta.get("status", "unknown")
            
            fi = npz_data.get("frame_indices")
            frame_top1_prob = npz_data.get("frame_top1_prob")
            frame_entropy = npz_data.get("frame_entropy")
            scene_ids = npz_data.get("scene_ids")
            length_seconds = npz_data.get("length_seconds")
            
            stats: Dict[str, Any] = {
                "video_id": video_id,
                "status": status,
                "empty_reason": meta.get("empty_reason"),
            }
            
            # Из meta
            if "total_frames" in meta:
                stats["total_frames"] = float(meta["total_frames"])
            if "processed_frames" in meta:
                stats["processed_frames"] = float(meta["processed_frames"])
            
            # Frame-level статистики
            if isinstance(fi, np.ndarray) and fi.ndim == 1:
                stats["num_frames"] = int(len(fi))
                num_frames_list.append(float(len(fi)))
            
            if isinstance(frame_top1_prob, np.ndarray) and frame_top1_prob.ndim == 1:
                finite_mask = np.isfinite(frame_top1_prob)
                if np.any(finite_mask):
                    stats["frame_top1_prob_mean"] = float(np.nanmean(frame_top1_prob))
                    stats["frame_top1_prob_std"] = float(np.nanstd(frame_top1_prob))
                    stats["frame_top1_prob_min"] = float(np.nanmin(frame_top1_prob))
                    stats["frame_top1_prob_max"] = float(np.nanmax(frame_top1_prob))
                    frame_top1_prob_list.extend(frame_top1_prob[finite_mask].tolist())
            
            if isinstance(frame_entropy, np.ndarray) and frame_entropy.ndim == 1:
                finite_mask = np.isfinite(frame_entropy)
                if np.any(finite_mask):
                    stats["frame_entropy_mean"] = float(np.nanmean(frame_entropy))
                    stats["frame_entropy_std"] = float(np.nanstd(frame_entropy))
                    frame_entropy_list.extend(frame_entropy[finite_mask].tolist())
            
            # Scene-level статистики
            if isinstance(scene_ids, np.ndarray):
                stats["num_scenes"] = int(len(scene_ids))
                num_scenes_list.append(float(len(scene_ids)))
            
            if isinstance(length_seconds, np.ndarray) and length_seconds.ndim == 1:
                finite_mask = np.isfinite(length_seconds)
                if np.any(finite_mask):
                    stats["scene_duration_mean"] = float(np.nanmean(length_seconds))
                    stats["scene_duration_std"] = float(np.nanstd(length_seconds))
                    stats["scene_duration_min"] = float(np.nanmin(length_seconds))
                    stats["scene_duration_max"] = float(np.nanmax(length_seconds))
                    scene_duration_list.extend(length_seconds[finite_mask].tolist())
            
            # Scene-level features (per-scene aggregates)
            scene_features = [
                "mean_score", "class_entropy_mean", "top1_prob_mean",
                "top1_vs_top2_gap_mean", "fraction_high_confidence_frames",
                "mean_aesthetic_score", "aesthetic_std", "aesthetic_frac_high",
                "mean_luxury_score",
                "mean_cozy", "mean_scary", "mean_epic", "mean_neutral", "atmosphere_entropy",
                "scene_change_score", "label_stability",
            ]
            
            for feat_name in scene_features:
                arr = npz_data.get(feat_name)
                if isinstance(arr, np.ndarray) and arr.ndim == 1:
                    finite_mask = np.isfinite(arr)
                    if np.any(finite_mask):
                        mean_val = float(np.nanmean(arr))
                        stats[f"scene_{feat_name}_mean"] = mean_val
                        features_by_name[feat_name].append(mean_val)
            
            # label_fusion
            label_fusion = npz_data.get("label_fusion")
            if label_fusion is not None:
                stats["label_fusion"] = str(label_fusion)
            
            # min_scene_seconds
            min_scene_seconds = npz_data.get("min_scene_seconds")
            if min_scene_seconds is not None:
                stats["min_scene_seconds"] = float(min_scene_seconds)
            
            video_stats.append(stats)
            all_results.append({
                "video_id": video_id,
                "stats": stats,
                "npz_data": npz_data,
                "meta": meta,
            })
            
        except Exception as e:
            print(f"Error processing {video_dir.name}: {e}", file=sys.stderr)
            continue
    
    # Агрегированная статистика
    result = {
        "total_videos": len(video_stats),
        "video_stats": video_stats,
    }
    
    # Статистики по метрикам
    if num_frames_list:
        result["num_frames"] = {
            "mean": float(np.mean(num_frames_list)),
            "std": float(np.std(num_frames_list)),
            "median": float(np.median(num_frames_list)),
            "min": float(np.min(num_frames_list)),
            "max": float(np.max(num_frames_list)),
        }
    
    if num_scenes_list:
        result["num_scenes"] = {
            "mean": float(np.mean(num_scenes_list)),
            "std": float(np.std(num_scenes_list)),
            "median": float(np.median(num_scenes_list)),
            "min": float(np.min(num_scenes_list)),
            "max": float(np.max(num_scenes_list)),
        }
    
    if scene_duration_list:
        result["scene_duration_seconds"] = {
            "mean": float(np.mean(scene_duration_list)),
            "std": float(np.std(scene_duration_list)),
            "median": float(np.median(scene_duration_list)),
            "min": float(np.min(scene_duration_list)),
            "max": float(np.max(scene_duration_list)),
        }
    
    if frame_top1_prob_list:
        result["frame_top1_prob"] = {
            "mean": float(np.mean(frame_top1_prob_list)),
            "std": float(np.std(frame_top1_prob_list)),
            "median": float(np.median(frame_top1_prob_list)),
            "min": float(np.min(frame_top1_prob_list)),
            "max": float(np.max(frame_top1_prob_list)),
        }
    
    if frame_entropy_list:
        result["frame_entropy"] = {
            "mean": float(np.mean(frame_entropy_list)),
            "std": float(np.std(frame_entropy_list)),
            "median": float(np.median(frame_entropy_list)),
            "min": float(np.min(frame_entropy_list)),
            "max": float(np.max(frame_entropy_list)),
        }
    
    # Статистики по scene-level features
    result["scene_features"] = {}
    for feat_name, values in features_by_name.items():
        if values:
            arr = np.array(values)
            finite_mask = np.isfinite(arr)
            if np.any(finite_mask):
                finite_arr = arr[finite_mask]
                result["scene_features"][feat_name] = {
                    "mean": float(np.mean(finite_arr)),
                    "std": float(np.std(finite_arr)),
                    "median": float(np.median(finite_arr)),
                    "min": float(np.min(finite_arr)),
                    "max": float(np.max(finite_arr)),
                }
    
    return result


def print_analysis_report(analysis_result: Dict[str, Any]):
    """Вывод отчета анализа."""
    print("=" * 60)
    print("Scene Classification Component Analysis Report")
    print("=" * 60)
    print(f"Total videos: {analysis_result.get('total_videos', 0)}")
    print()
    
    if "error" in analysis_result:
        print(f"Error: {analysis_result['error']}")
        return
    
    # Статистики по кадрам
    if "num_frames" in analysis_result:
        nf = analysis_result["num_frames"]
        print("Frames per video:")
        print(f"  Mean: {nf['mean']:.1f}, Std: {nf['std']:.1f}, Median: {nf['median']:.1f}")
        print(f"  Range: [{nf['min']:.0f}, {nf['max']:.0f}]")
        print()
    
    # Статистики по сценам
    if "num_scenes" in analysis_result:
        ns = analysis_result["num_scenes"]
        print("Scenes per video:")
        print(f"  Mean: {ns['mean']:.1f}, Std: {ns['std']:.1f}, Median: {ns['median']:.1f}")
        print(f"  Range: [{ns['min']:.0f}, {ns['max']:.0f}]")
        print()
    
    # Длительность сцен
    if "scene_duration_seconds" in analysis_result:
        sd = analysis_result["scene_duration_seconds"]
        print("Scene duration (seconds):")
        print(f"  Mean: {sd['mean']:.2f}, Std: {sd['std']:.2f}, Median: {sd['median']:.2f}")
        print(f"  Range: [{sd['min']:.2f}, {sd['max']:.2f}]")
        print()
    
    # Frame-level метрики
    if "frame_top1_prob" in analysis_result:
        ftp = analysis_result["frame_top1_prob"]
        print("Frame top1 probability:")
        print(f"  Mean: {ftp['mean']:.3f}, Std: {ftp['std']:.3f}, Median: {ftp['median']:.3f}")
        print(f"  Range: [{ftp['min']:.3f}, {ftp['max']:.3f}]")
        print()
    
    if "frame_entropy" in analysis_result:
        fe = analysis_result["frame_entropy"]
        print("Frame entropy:")
        print(f"  Mean: {fe['mean']:.3f}, Std: {fe['std']:.3f}, Median: {fe['median']:.3f}")
        print(f"  Range: [{fe['min']:.3f}, {fe['max']:.3f}]")
        print()
    
    # Scene-level features
    if "scene_features" in analysis_result and analysis_result["scene_features"]:
        print("Scene-level features (mean across videos):")
        for feat_name, stats in sorted(analysis_result["scene_features"].items()):
            print(f"  {feat_name}:")
            print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, Median: {stats['median']:.3f}")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print()
    
    # Проверка на аномалии (z-score > 3)
    print("Anomaly detection (z-score > 3):")
    anomalies_found = False
    
    if "num_frames" in analysis_result:
        nf = analysis_result["num_frames"]
        mean, std = nf["mean"], nf["std"]
        if std > 0:
            for video_stat in analysis_result.get("video_stats", []):
                if "num_frames" in video_stat:
                    z_score = abs((video_stat["num_frames"] - mean) / std)
                    if z_score > 3:
                        print(f"  ⚠️ {video_stat['video_id']}: num_frames z-score={z_score:.2f}")
                        anomalies_found = True
    
    if not anomalies_found:
        print("  ✅ No obvious anomalies found (best-effort).")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze scene_classification component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    
    args = parser.parse_args()
    
    result = analyze_scene_classification_results(args.results_base)
    print_analysis_report(result)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

