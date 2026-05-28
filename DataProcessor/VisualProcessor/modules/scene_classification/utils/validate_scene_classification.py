#!/usr/bin/env python3
"""
Валидатор для scene_classification компонента.

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


class SceneClassificationValidator:
    """Валидатор для scene_classification компонента."""
    
    def __init__(self, results_base_path: str):
        """
        Args:
            results_base_path: Базовый путь к результатам (например, DataProcessor/dp_results)
        """
        self.results_base_path = Path(results_base_path)
        self.videos: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []
    
    def _issue(self, issue_type: str, severity: str, video_id: str, message: str, **kwargs):
        """Добавить проблему."""
        self.issues.append({
            "type": issue_type,
            "severity": severity,
            "video_id": video_id,
            "message": message,
            **kwargs
        })
    
    def load_video_results(self, platform_id: str, video_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        """Загрузить результаты для одного видео."""
        scene_classification_dir = self.results_base_path / platform_id / video_id / run_id / "scene_classification"
        npz_path = scene_classification_dir / "scene_classification_features.npz"
        render_path = scene_classification_dir / "_render" / "render_context.json"
        
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
    
    def validate_single_video(self, video_data: Dict[str, Any]) -> None:
        """Валидация одного видео."""
        video_id = video_data["video_id"]
        npz_data = video_data["npz_data"]
        meta = video_data["meta"]
        
        # 1. Проверка статуса
        status = meta.get("status", "unknown")
        if status not in ["ok", "empty"]:
            self._issue(
                issue_type="status",
                severity="error",
                video_id=video_id,
                message=f"Status is not 'ok' or 'empty': {status}",
                empty_reason=meta.get("empty_reason"),
            )
            if status == "error":
                return
        
        if status == "empty" and not meta.get("empty_reason"):
            self._issue(
                issue_type="meta_empty_reason",
                severity="warning",
                video_id=video_id,
                message="meta.status is 'empty' but meta.empty_reason is missing/empty",
            )
        
        # 2. Проверка обязательных ключей (основные)
        required_keys = [
            "frame_indices", "times_s",
            "frame_topk_ids", "frame_topk_probs", "frame_entropy", "frame_top1_prob",
            "frame_top1_top2_gap", "frame_scene_id",
            "label_fusion", "min_scene_seconds",
            "scene_ids", "scene_label", "fusion_mode",
            "start_frame", "end_frame", "start_time_s", "end_time_s",
            "length_frames", "length_seconds",
            "mean_score", "class_entropy_mean", "top1_prob_mean",
            "top1_vs_top2_gap_mean", "fraction_high_confidence_frames",
            "mean_aesthetic_score", "aesthetic_std", "aesthetic_frac_high",
            "mean_luxury_score",
            "mean_cozy", "mean_scary", "mean_epic", "mean_neutral", "atmosphere_entropy",
            "scene_change_score", "label_stability",
            "indices", "dominant_places_topk_ids", "dominant_places_topk_probs",
            "scenes", "scenes_raw",
            "scene_aesthetic_prompts", "scene_luxury_prompts", "scene_atmosphere_prompts", "places365_prompts",
            "summary", "meta"
        ]
        
        for key in required_keys:
            if key not in npz_data:
                self._issue(
                    issue_type="missing_key",
                    severity="error",
                    video_id=video_id,
                    message=f"Missing required key: {key}",
                )
        
        # Если критичные ключи отсутствуют, прекращаем глубокую проверку
        if any(k not in npz_data for k in ["frame_indices", "times_s", "frame_scene_id", "scene_ids"]):
            return
        
        # 3. Конвертация в numpy arrays если нужно
        fi = npz_data.get("frame_indices")
        ts = npz_data.get("times_s")
        fsi = npz_data.get("frame_scene_id")
        
        if fi is not None and not isinstance(fi, np.ndarray):
            try:
                fi = np.asarray(fi)
            except Exception:
                pass
        if ts is not None and not isinstance(ts, np.ndarray):
            try:
                ts = np.asarray(ts)
            except Exception:
                pass
        if fsi is not None and not isinstance(fsi, np.ndarray):
            try:
                fsi = np.asarray(fsi)
            except Exception:
                pass
        
        # 4. Проверка размерностей frame-level
        if not isinstance(fi, np.ndarray) or fi.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"frame_indices must be 1D array, got {type(fi)}",
            )
            return
        
        if len(fi) < 2:
            self._issue(
                issue_type="invalid_value",
                severity="error",
                video_id=video_id,
                message=f"frame_indices must have >= 2 frames, got {len(fi)}",
            )
            return
        
        N = int(len(fi))
        
        # Проверка sorted + unique для frame_indices
        if len(fi) > 1 and not np.all(np.diff(fi) > 0):
            self._issue(
                issue_type="invalid_value",
                severity="error",
                video_id=video_id,
                message="frame_indices must be sorted and unique",
            )
        
        # times_s
        if not isinstance(ts, np.ndarray) or ts.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"times_s must be 1D array, got {type(ts)}",
            )
        else:
            if len(ts) != N:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"times_s length ({len(ts)}) != frame_indices length ({N})",
                )
            if len(ts) > 1 and np.any(np.diff(ts) < 0):
                self._issue(
                    issue_type="invalid_value",
                    severity="error",
                    video_id=video_id,
                    message="times_s is not monotonically increasing",
                )
        
        # frame_scene_id
        if not isinstance(fsi, np.ndarray) or fsi.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"frame_scene_id must be 1D array, got {type(fsi)}",
            )
        else:
            if len(fsi) != N:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"frame_scene_id length ({len(fsi)}) != frame_indices length ({N})",
                )
            if np.any(fsi < 0):
                self._issue(
                    issue_type="invalid_value",
                    severity="error",
                    video_id=video_id,
                    message="frame_scene_id contains negative values (invalid scene index)",
                )
        
        # frame_topk_ids, frame_topk_probs
        ftk_ids = npz_data.get("frame_topk_ids")
        ftk_probs = npz_data.get("frame_topk_probs")
        
        if ftk_ids is not None and not isinstance(ftk_ids, np.ndarray):
            try:
                ftk_ids = np.asarray(ftk_ids)
            except Exception:
                pass
        if ftk_probs is not None and not isinstance(ftk_probs, np.ndarray):
            try:
                ftk_probs = np.asarray(ftk_probs)
            except Exception:
                pass
        
        if isinstance(ftk_ids, np.ndarray) and isinstance(ftk_probs, np.ndarray):
            if ftk_ids.ndim != 2 or ftk_ids.shape[1] != 5:
                self._issue(
                    issue_type="invalid_shape",
                    severity="error",
                    video_id=video_id,
                    message=f"frame_topk_ids must be (N, 5), got shape {ftk_ids.shape}",
                )
            elif ftk_ids.shape[0] != N:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"frame_topk_ids rows ({ftk_ids.shape[0]}) != N ({N})",
                )
            
            if ftk_probs.ndim != 2 or ftk_probs.shape[1] != 5:
                self._issue(
                    issue_type="invalid_shape",
                    severity="error",
                    video_id=video_id,
                    message=f"frame_topk_probs must be (N, 5), got shape {ftk_probs.shape}",
                )
            elif ftk_probs.shape[0] != N:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"frame_topk_probs rows ({ftk_probs.shape[0]}) != N ({N})",
                )
            elif ftk_ids.shape[0] == ftk_probs.shape[0]:
                # Проверка вероятностей в [0, 1]
                finite_probs = ftk_probs[np.isfinite(ftk_probs)]
                if len(finite_probs) > 0:
                    if np.any(finite_probs < 0) or np.any(finite_probs > 1):
                        self._issue(
                            issue_type="invalid_value",
                            severity="warning",
                            video_id=video_id,
                            message="frame_topk_probs contains values outside [0, 1]",
                        )
        
        # 5. Проверка scene-level данных
        scene_ids = npz_data.get("scene_ids")
        scene_label = npz_data.get("scene_label")
        
        if scene_ids is not None and not isinstance(scene_ids, np.ndarray):
            try:
                scene_ids = np.asarray(scene_ids, dtype=object)
            except Exception:
                pass
        
        if isinstance(scene_ids, np.ndarray):
            S = int(len(scene_ids))
            
            # Проверка согласованности scene-level массивов
            scene_arrays = {
                "scene_label": scene_label,
                "start_frame": npz_data.get("start_frame"),
                "end_frame": npz_data.get("end_frame"),
                "start_time_s": npz_data.get("start_time_s"),
                "end_time_s": npz_data.get("end_time_s"),
                "length_frames": npz_data.get("length_frames"),
                "length_seconds": npz_data.get("length_seconds"),
            }
            
            for key, arr in scene_arrays.items():
                if arr is not None:
                    if not isinstance(arr, np.ndarray):
                        try:
                            arr = np.asarray(arr)
                        except Exception:
                            pass
                    if isinstance(arr, np.ndarray) and arr.ndim == 1:
                        if len(arr) != S:
                            self._issue(
                                issue_type="dimension_mismatch",
                                severity="error",
                                video_id=video_id,
                                message=f"{key} length ({len(arr)}) != num scenes ({S})",
                            )
            
            # Проверка frame_scene_id согласованности
            if isinstance(fsi, np.ndarray):
                unique_scene_ids = np.unique(fsi)
                if len(unique_scene_ids) != S:
                    self._issue(
                        issue_type="dimension_mismatch",
                        severity="warning",
                        video_id=video_id,
                        message=f"frame_scene_id unique count ({len(unique_scene_ids)}) != num scenes ({S})",
                    )
        
        # 6. Проверка scenes dict
        scenes = npz_data.get("scenes")
        if scenes is None or not isinstance(scenes, dict):
            self._issue(
                issue_type="missing_key",
                severity="error",
                video_id=video_id,
                message="scenes must be a dict",
            )
        elif isinstance(scene_ids, np.ndarray):
            # Проверка что все scene_ids присутствуют в scenes
            for sid in scene_ids:
                sid_str = str(sid)
                if sid_str not in scenes:
                    self._issue(
                        issue_type="missing_key",
                        severity="warning",
                        video_id=video_id,
                        message=f"scene_id {sid_str} not found in scenes dict",
                    )
    
    def validate_all(self, platform_id: str = "youtube") -> Dict[str, Any]:
        """Валидация всех видео."""
        platform_dir = self.results_base_path / platform_id
        
        if not platform_dir.exists():
            return {"error": f"Platform directory not found: {platform_dir}"}
        
        # Находим все test_scene_classification_* директории
        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith("test_scene_classification"):
                continue
            
            video_id = video_dir.name
            run_id = video_id
            
            video_data = self.load_video_results(platform_id, video_id, run_id)
            if video_data is None:
                continue
            
            self.videos.append(video_data)
            self.validate_single_video(video_data)
        
        return {
            "total_videos": len(self.videos),
            "total_issues": len(self.issues),
            "issues_by_severity": self._group_issues_by_severity(),
            "issues_by_type": self._group_issues_by_type(),
        }
    
    def _group_issues_by_severity(self) -> Dict[str, int]:
        """Группировка проблем по серьезности."""
        result = defaultdict(int)
        for issue in self.issues:
            result[issue["severity"]] += 1
        return dict(result)
    
    def _group_issues_by_type(self) -> Dict[str, int]:
        """Группировка проблем по типу."""
        result = defaultdict(int)
        for issue in self.issues:
            result[issue["type"]] += 1
        return dict(result)
    
    def print_report(self):
        """Вывод отчета."""
        print("=" * 60)
        print("Scene Classification Component Validation Report")
        print("=" * 60)
        print(f"Total videos: {len(self.videos)}")
        print(f"Total issues: {len(self.issues)}")
        print()
        
        if self.issues:
            print("Issues by severity:")
            for severity, count in self._group_issues_by_severity().items():
                print(f"  {severity}: {count}")
            print()
            
            print("Issues by type:")
            for issue_type, count in self._group_issues_by_type().items():
                print(f"  {issue_type}: {count}")
            print()
            
            print("Sample issues:")
            for issue in self.issues[:10]:
                print(f"- [{issue['severity']}] {issue['video_id']} | {issue['type']}: {issue['message']}")
        else:
            print("✅ No issues found!")
            print()


def main():
    parser = argparse.ArgumentParser(description="Validate scene_classification component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    parser.add_argument("--platform-id", default="youtube", help="Platform ID (default: youtube)")
    
    args = parser.parse_args()
    
    validator = SceneClassificationValidator(args.results_base)
    result = validator.validate_all(args.platform_id)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    
    validator.print_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())

