#!/usr/bin/env python3
"""
Персональный валидатор для detalize_face компонента.

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


class DetalizeFaceValidator:
    """Валидатор для detalize_face компонента."""
    
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
        detalize_face_dir = self.results_base_path / platform_id / video_id / run_id / "detalize_face"
        npz_path = detalize_face_dir / "detalize_face.npz"
        render_path = detalize_face_dir / "_render" / "render_context.json"
        
        if not npz_path.exists():
            return None
        
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
        if status not in ["ok", "empty"]:
            issues.append({
                "type": "status",
                "severity": "error",
                "video_id": video_id,
                "message": f"Status is not 'ok' or 'empty': {status}",
                "empty_reason": meta.get("empty_reason"),
            })
            if status == "error":
                return issues
        
        # 2. Проверка обязательных ключей
        required_keys = ["frame_indices", "times_s", "face_present", "processed_mask", "primary_valid", "face_count", "primary_tracking_id", "primary_compact_features", "aggregated", "summary", "meta"]
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
                    "message": f"frame_indices must be 1D array, got {type(frame_indices)}",
                })
        
        if "times_s" in npz_data:
            times_s = npz_data["times_s"]
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
                    "message": f"times_s must be 1D array, got {type(times_s)}",
                })
            elif len(times_s) > 1:
                diffs = np.diff(times_s)
                if np.any(diffs < 0):
                    issues.append({
                        "type": "invalid_value",
                        "severity": "error",
                        "video_id": video_id,
                        "message": "times_s is not monotonically increasing",
                    })
        
        # 4. Проверка primary_compact_features
        if "primary_compact_features" in npz_data:
            compact = npz_data["primary_compact_features"]
            if isinstance(compact, np.ndarray):
                if compact.ndim != 2:
                    issues.append({
                        "type": "invalid_shape",
                        "severity": "error",
                        "video_id": video_id,
                        "message": f"primary_compact_features must be 2D array (N, 40), got shape {compact.shape}",
                    })
                elif compact.shape[1] != 40:
                    issues.append({
                        "type": "invalid_shape",
                        "severity": "warning",
                        "video_id": video_id,
                        "message": f"primary_compact_features expected 40 dims, got {compact.shape[1]}",
                    })
        
        # 5. Проверка aggregated
        if "aggregated" in npz_data:
            aggregated = npz_data["aggregated"]
            if isinstance(aggregated, np.ndarray) and aggregated.ndim == 0:
                aggregated = aggregated.item()
            
            if not isinstance(aggregated, dict):
                issues.append({
                    "type": "invalid_type",
                    "severity": "error",
                    "video_id": video_id,
                    "message": f"aggregated must be dict, got {type(aggregated)}",
                })
            else:
                # Проверка ключевых метрик
                key_metrics = ["valid_frames", "axis_frames", "face_present_ratio", "processed_ratio", "primary_valid_ratio"]
                for metric in key_metrics:
                    if metric not in aggregated:
                        issues.append({
                            "type": "missing_stat",
                            "severity": "warning",
                            "video_id": video_id,
                            "message": f"Missing key metric in aggregated: {metric}",
                        })
        
        # 6. Проверка summary
        if "summary" in npz_data:
            summary = npz_data["summary"]
            if isinstance(summary, np.ndarray) and summary.ndim == 0:
                summary = summary.item()
            
            if not isinstance(summary, dict):
                issues.append({
                    "type": "invalid_type",
                    "severity": "error",
                    "video_id": video_id,
                    "message": f"summary must be dict, got {type(summary)}",
                })
        
        return issues
    
    def validate_all(self, platform_id: str = "youtube") -> Dict[str, Any]:
        """Валидация всех видео."""
        platform_dir = self.results_base_path / platform_id
        if not platform_dir.exists():
            return {"error": f"Platform directory not found: {platform_dir}"}
        
        # Находим все test_detalize_face_* директории
        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith("test_detalize_face"):
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
            # Собираем статистику по ключевым метрикам из aggregated
            key_metrics = [
                "valid_frames", "axis_frames", "face_present_ratio", "processed_ratio", "primary_valid_ratio"
            ]
            
            for metric in key_metrics:
                values = []
                for video_data in self.videos:
                    aggregated = video_data["npz_data"].get("aggregated")
                    if isinstance(aggregated, np.ndarray) and aggregated.ndim == 0:
                        aggregated = aggregated.item()
                    if isinstance(aggregated, dict) and metric in aggregated:
                        value = aggregated[metric]
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
        print("Detalize Face Component Validation Report")
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
    parser = argparse.ArgumentParser(description="Validate detalize_face component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    parser.add_argument("--platform-id", default="youtube", help="Platform ID")
    
    args = parser.parse_args()
    
    validator = DetalizeFaceValidator(args.results_base)
    results = validator.validate_all(args.platform_id)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    validator.print_report(results)
    return 0 if results["total_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

