#!/usr/bin/env python3
"""
Персональный валидатор для cut_detection компонента.

Проверяет качество данных на основе прогонов на нескольких видео.
Сравнивает результаты между разными видео и выявляет аномалии.
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


class CutDetectionValidator:
    """Валидатор для cut_detection компонента."""
    
    def __init__(self, results_base_path: str):
        """
        Args:
            results_base_path: Базовый путь к результатам (например, DataProcessor/dp_results)
        """
        self.results_base_path = Path(results_base_path)
        self.videos: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []
    
    def load_video_results(self, platform_id: str, video_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        """Загрузить результаты для одного видео."""
        cut_detection_dir = self.results_base_path / platform_id / video_id / run_id / "cut_detection"
        
        # Ищем последний NPZ файл (по timestamp)
        npz_files = list(cut_detection_dir.glob("cut_detection_features_*.npz"))
        if not npz_files:
            return None
        
        npz_path = max(npz_files, key=lambda p: p.stat().st_mtime)
        render_path = cut_detection_dir / "_render" / "render_context.json"
        
        try:
            npz_data = load_npz(str(npz_path))
            meta = extract_meta(npz_data)
            
            render_data = {}
            if render_path.exists():
                with open(render_path, 'r', encoding='utf-8') as f:
                    render_data = json.load(f)
            
            return {
                "video_id": video_id,
                "run_id": run_id,
                "npz_data": npz_data,
                "meta": meta,
                "render": render_data,
                "npz_path": str(npz_path),
            }
        except Exception as e:
            print(f"Error loading {video_id}: {e}")
            return None
    
    def validate_single_video(self, video_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Валидация одного видео."""
        issues = []
        video_id = video_data["video_id"]
        npz_data = video_data["npz_data"]
        meta = video_data["meta"]
        
        # 1. Проверка статуса
        status = meta.get("status", "unknown")
        if status != "ok":
            issues.append({
                "type": "status",
                "severity": "error",
                "video_id": video_id,
                "message": f"Status is not 'ok': {status}",
                "empty_reason": meta.get("empty_reason"),
            })
            return issues
        
        # 2. Проверка обязательных ключей
        required_keys = ["frame_indices", "times_s", "features", "detections", "meta"]
        for key in required_keys:
            if key not in npz_data:
                issues.append({
                    "type": "missing_key",
                    "severity": "error",
                    "video_id": video_id,
                    "message": f"Missing required key: {key}",
                })
        
        # 3. Проверка размерностей
        if "frame_indices" in npz_data:
            frame_indices = npz_data["frame_indices"]
            # Конвертируем в numpy array если нужно
            if not isinstance(frame_indices, np.ndarray):
                try:
                    frame_indices = np.asarray(frame_indices)
                except Exception:
                    pass
            if not isinstance(frame_indices, np.ndarray) or frame_indices.ndim != 1:
                issues.append({
                    "type": "invalid_shape",
                    "severity": "error",
                    "video_id": video_id,
                    "message": f"frame_indices must be 1D array, got {type(frame_indices)}, shape: {getattr(frame_indices, 'shape', 'N/A')}",
                })
        
        if "times_s" in npz_data:
            times_s = npz_data["times_s"]
            # Конвертируем в numpy array если нужно
            if not isinstance(times_s, np.ndarray):
                try:
                    times_s = np.asarray(times_s)
                except Exception:
                    pass
            if not isinstance(times_s, np.ndarray) or times_s.ndim != 1:
                issues.append({
                    "type": "invalid_shape",
                    "severity": "error",
                    "video_id": video_id,
                    "message": f"times_s must be 1D array, got {type(times_s)}, shape: {getattr(times_s, 'shape', 'N/A')}",
                })
            
            # Проверка монотонности
            if len(times_s) > 1:
                diffs = np.diff(times_s)
                if np.any(diffs < 0):
                    issues.append({
                        "type": "invalid_value",
                        "severity": "error",
                        "video_id": video_id,
                        "message": "times_s is not monotonically increasing",
                    })
        
        # 4. Проверка features
        if "features" in npz_data:
            features = npz_data["features"]
            if isinstance(features, np.ndarray) and features.ndim == 0:
                features = features.item()
            
            if not isinstance(features, dict):
                issues.append({
                    "type": "invalid_type",
                    "severity": "error",
                    "video_id": video_id,
                    "message": f"features must be dict, got {type(features)}",
                })
            else:
                # Проверка ключевых метрик
                key_metrics = [
                    "hard_cuts_count", "hard_cuts_per_minute",
                    "cuts_per_minute", "scene_count"
                ]
                for metric in key_metrics:
                    if metric not in features:
                        issues.append({
                            "type": "missing_stat",
                            "severity": "warning",
                            "video_id": video_id,
                            "message": f"Missing key metric: {metric}",
                        })
                    else:
                        value = features[metric]
                        if not isinstance(value, (int, float, np.number)):
                            issues.append({
                                "type": "invalid_value",
                                "severity": "warning",
                                "video_id": video_id,
                                "message": f"{metric} must be numeric, got {type(value)}",
                            })
                        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
                            issues.append({
                                "type": "invalid_value",
                                "severity": "warning",
                                "video_id": video_id,
                                "message": f"{metric} is not finite: {value}",
                            })
                        elif value < 0:
                            issues.append({
                                "type": "value_range",
                                "severity": "warning",
                                "video_id": video_id,
                                "message": f"{metric} is negative: {value}",
                            })
        
        # 5. Проверка detections
        if "detections" in npz_data:
            detections = npz_data["detections"]
            if isinstance(detections, np.ndarray) and detections.ndim == 0:
                detections = detections.item()
            
            if not isinstance(detections, dict):
                issues.append({
                    "type": "invalid_type",
                    "severity": "error",
                    "video_id": video_id,
                    "message": f"detections must be dict, got {type(detections)}",
                })
            else:
                # Проверка ключевых детекций
                key_detections = ["hard_cut_pos", "soft_events", "motion_cut_pos"]
                for key in key_detections:
                    if key not in detections:
                        issues.append({
                            "type": "missing_stat",
                            "severity": "warning",
                            "video_id": video_id,
                            "message": f"Missing detection key: {key}",
                        })
        
        return issues
    
    def validate_all(self, platform_id: str = "youtube") -> Dict[str, Any]:
        """Валидация всех видео."""
        platform_dir = self.results_base_path / platform_id
        if not platform_dir.exists():
            return {"error": f"Platform directory not found: {platform_dir}"}
        
        # Находим все test_cut_detection_* директории
        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith("test_cut_detection"):
                continue
            
            video_id = video_dir.name
            run_id = video_id  # Обычно run_id совпадает с video_id
            
            video_data = self.load_video_results(platform_id, video_id, run_id)
            if video_data is None:
                continue
            
            self.videos.append(video_data)
            issues = self.validate_single_video(video_data)
            self.issues.extend(issues)
        
        # Группировка проблем
        issues_by_type = defaultdict(int)
        issues_by_severity = defaultdict(int)
        
        for issue in self.issues:
            issues_by_type[issue["type"]] += 1
            issues_by_severity[issue["severity"]] += 1
        
        # Статистика по метрикам
        summary_stats = {}
        if self.videos:
            # Собираем статистику по ключевым метрикам
            key_metrics = [
                "hard_cuts_count", "hard_cuts_per_minute", "cuts_per_minute",
                "scene_count", "fade_in_count", "fade_out_count", "dissolve_count"
            ]
            
            for metric in key_metrics:
                values = []
                for video_data in self.videos:
                    features = video_data["npz_data"].get("features")
                    if isinstance(features, np.ndarray) and features.ndim == 0:
                        features = features.item()
                    if isinstance(features, dict) and metric in features:
                        value = features[metric]
                        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                            values.append(float(value))
                
                if values:
                    summary_stats[metric] = {
                        "count": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "range": [float(np.min(values)), float(np.max(values))],
                        "median": float(np.median(values)),
                    }
        
        return {
            "total_videos": len(self.videos),
            "total_issues": len(self.issues),
            "issues_by_type": dict(issues_by_type),
            "issues_by_severity": dict(issues_by_severity),
            "summary_statistics": summary_stats,
        }
    
    def print_report(self, results: Dict[str, Any]):
        """Вывести отчет."""
        print("=" * 60)
        print("Cut Detection Component Validation Report")
        print("=" * 60)
        print(f"Total videos: {results['total_videos']}")
        print(f"Total issues: {results['total_issues']}")
        print()
        
        print("Issues by severity:")
        for severity, count in results["issues_by_severity"].items():
            print(f"  {severity}: {count}")
        print()
        
        print("Issues by type:")
        for issue_type, count in results["issues_by_type"].items():
            print(f"  {issue_type}: {count}")
        print()
        
        print("Summary statistics:")
        for metric, stats in results["summary_statistics"].items():
            print(f"  {metric}:")
            print(f"    count: {stats['count']}")
            print(f"    mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    range: [{stats['range'][0]:.4f}, {stats['range'][1]:.4f}]")
            print(f"    median: {stats['median']:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Validate cut_detection component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    parser.add_argument("--platform-id", default="youtube", help="Platform ID")
    
    args = parser.parse_args()
    
    validator = CutDetectionValidator(args.results_base)
    results = validator.validate_all(args.platform_id)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    validator.print_report(results)
    return 0 if results["total_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

