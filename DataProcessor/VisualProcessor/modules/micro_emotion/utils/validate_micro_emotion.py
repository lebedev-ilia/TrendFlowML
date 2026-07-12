#!/usr/bin/env python3
"""
Валидатор для micro_emotion компонента.

Проверяет:
- наличие обязательных ключей NPZ (micro_emotion_npz_v3)
- размерности, dtype, монотонность осей, согласованность N/F
- базовый meta-контракт и статус
- базовые sanity-checks по frame_features, compact22, feature_values
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add VisualProcessor root to path
vp_root = Path(__file__).resolve().parent.parent.parent
if str(vp_root) not in sys.path:
    sys.path.insert(0, str(vp_root))

from utils.renderer import load_npz, extract_meta  # type: ignore


class MicroEmotionValidator:
    def __init__(self, results_base_path: str):
        self.results_base_path = Path(results_base_path)
        self.videos: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []

    def _issue(
        self,
        *,
        issue_type: str,
        severity: str,
        video_id: str,
        message: str,
        **extra: Any,
    ) -> None:
        self.issues.append(
            {
                "type": issue_type,
                "severity": severity,
                "video_id": video_id,
                "message": message,
                **extra,
            }
        )

    def load_video_results(
        self, platform_id: str, video_id: str, run_id: str
    ) -> Optional[Dict[str, Any]]:
        me_dir = self.results_base_path / platform_id / video_id / run_id / "micro_emotion"
        npz_path = me_dir / "micro_emotion.npz"
        render_path = me_dir / "_render" / "render_context.json"

        if not npz_path.exists():
            return None

        try:
            npz_data = load_npz(str(npz_path))
            meta = extract_meta(npz_data)
            render_data: Dict[str, Any] = {}
            if render_path.exists():
                with open(render_path, "r", encoding="utf-8") as f:
                    render_data = json.load(f)
            return {
                "video_id": video_id,
                "run_id": run_id,
                "npz_data": npz_data,
                "meta": meta,
                "render": render_data,
                "npz_path": str(npz_path),
            }
        except Exception as e:  # pragma: no cover - best-effort
            print(f"Error loading {video_id}: {e}")
            return None

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _as_array(x: Any, *, dtype: Optional[np.dtype] = None) -> Optional[np.ndarray]:
        if isinstance(x, np.ndarray):
            if dtype is not None and x.dtype != dtype:
                return x.astype(dtype)
            return x
        try:
            return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
        except Exception:
            return None

    # --- per-video validation -------------------------------------------

    def validate_single_video(self, video_data: Dict[str, Any]) -> None:
        video_id = video_data["video_id"]
        npz_data = video_data["npz_data"]
        meta = video_data["meta"]

        # 1) meta / status
        status = meta.get("status", "unknown")
        if status not in ("ok", "empty"):
            self._issue(
                issue_type="status",
                severity="error",
                video_id=video_id,
                message=f"Status is not 'ok' or 'empty': {status}",
                empty_reason=meta.get("empty_reason"),
            )
            if status == "error":
                return

        # базовый meta-контракт
        for k in ("producer", "producer_version", "schema_version"):
            if k not in meta:
                self._issue(
                    issue_type="meta_missing_key",
                    severity="warning",
                    video_id=video_id,
                    message=f"meta missing key: {k}",
                )

        if meta.get("schema_version") not in ("micro_emotion_npz_v3", None):
            self._issue(
                issue_type="schema_version",
                severity="warning",
                video_id=video_id,
                message=f"Unexpected schema_version: {meta.get('schema_version')}",
            )

        # 2) required top-level keys from schema
        required_keys = [
            "frame_indices",
            "times_s",
            "face_present_any",
            "frame_feature_names",
            "frame_features",
            "compact22",
            "compact22_feature_names",
            "event_times_s",
            "event_type_id",
            "event_strength",
            "feature_names",
            "feature_values",
            "meta",
        ]
        for k in required_keys:
            if k not in npz_data:
                self._issue(
                    issue_type="missing_key",
                    severity="error",
                    video_id=video_id,
                    message=f"Missing required key: {k}",
                )

        # Критичные оси
        if any(
            k not in npz_data
            for k in (
                "frame_indices",
                "times_s",
                "face_present_any",
                "frame_features",
                "compact22",
            )
        ):
            return

        fi = self._as_array(npz_data.get("frame_indices"), dtype=np.int32)
        ts = self._as_array(npz_data.get("times_s"), dtype=np.float32)
        face_present = self._as_array(npz_data.get("face_present_any"), dtype=bool)
        frame_features = self._as_array(npz_data.get("frame_features"), dtype=np.float32)
        compact22 = self._as_array(npz_data.get("compact22"), dtype=np.float32)

        # frame_indices / times_s
        if fi is None or fi.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"frame_indices must be 1D array, got {type(fi)}",
            )
            return

        N = int(fi.shape[0])
        if N == 0:
            self._issue(
                issue_type="invalid_value",
                severity="error",
                video_id=video_id,
                message="frame_indices is empty",
            )
            return

        if np.any(fi < 0):
            self._issue(
                issue_type="invalid_value",
                severity="error",
                video_id=video_id,
                message="frame_indices contains negative values",
            )

        if fi.dtype != np.int32:
            self._issue(
                issue_type="invalid_dtype",
                severity="warning",
                video_id=video_id,
                message=f"frame_indices dtype should be int32, got {fi.dtype}",
            )

        if N > 1 and not np.all(np.diff(fi) > 0):
            self._issue(
                issue_type="invalid_value",
                severity="error",
                video_id=video_id,
                message="frame_indices must be strictly increasing",
            )

        if ts is None or ts.ndim != 1 or ts.shape[0] != N:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"times_s must be 1D array of length N, got {None if ts is None else ts.shape}",
            )
        else:
            if ts.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"times_s dtype should be float32, got {ts.dtype}",
                )
            if ts.size > 1 and np.any(np.diff(ts) < -1e-3):
                self._issue(
                    issue_type="invalid_value",
                    severity="error",
                    video_id=video_id,
                    message="times_s is not monotonically non-decreasing",
                )

        # face_present_any (N,) bool
        if face_present is None or face_present.ndim != 1 or face_present.shape[0] != N:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"face_present_any must be 1D array of length N, got {None if face_present is None else face_present.shape}",
            )

        # frame_features (N, F) float32.
        # Валидный частный случай: пустой набор wide-фич (F=0), напр. на status=empty
        # (нет лиц → wide-фичи не строятся). При этом load_npz схлопывает zero-size массив
        # (N,0) в [], т.е. читается как 1D shape (0,). Это НЕ ошибка формы — принимаем.
        ff_empty = frame_features is None or frame_features.size == 0
        if ff_empty and status == "empty":
            pass
        elif frame_features is None or frame_features.ndim != 2 or frame_features.shape[0] != N:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"frame_features must be 2D array of shape (N, F), got {None if frame_features is None else frame_features.shape}",
            )
        else:
            if frame_features.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"frame_features dtype should be float32, got {frame_features.dtype}",
                )

        # compact22 (N, 22) float32
        if compact22 is None or compact22.ndim != 2 or compact22.shape[0] != N or compact22.shape[1] != 22:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"compact22 must be 2D array of shape (N, 22), got {None if compact22 is None else compact22.shape}",
            )
        else:
            if compact22.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"compact22 dtype should be float32, got {compact22.dtype}",
                )

        # feature_names / feature_values
        fn = npz_data.get("feature_names")
        fv = self._as_array(npz_data.get("feature_values"), dtype=np.float32)
        if isinstance(fn, np.ndarray):
            fn = fn.tolist()
        if not isinstance(fn, (list, tuple)) or len(fn) == 0:
            self._issue(
                issue_type="invalid_value",
                severity="error",
                video_id=video_id,
                message=f"feature_names must be non-empty list/tuple, got {type(fn)}",
            )
        else:
            F = len(fn)
            if fv is None or fv.ndim != 1 or fv.shape[0] != F:
                self._issue(
                    issue_type="invalid_shape",
                    severity="error",
                    video_id=video_id,
                    message=f"feature_values must be 1D array of length F ({F}), got {None if fv is None else fv.shape}",
                )
            else:
                if fv.dtype != np.float32:
                    self._issue(
                        issue_type="invalid_dtype",
                        severity="warning",
                        video_id=video_id,
                        message=f"feature_values dtype should be float32, got {fv.dtype}",
                    )

    # --- batch validation -----------------------------------------------

    def validate_all(self, platform_id: str = "youtube") -> Dict[str, Any]:
        platform_dir = self.results_base_path / platform_id
        if not platform_dir.exists():
            return {"error": f"Platform directory not found: {platform_dir}"}

        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith("test_micro_emotion"):
                continue
            run_dir = video_dir / video_dir.name
            video_data = self.load_video_results(platform_id, video_dir.name, video_dir.name)
            if video_data:
                self.videos.append(video_data)
                self.validate_single_video(video_data)

        return {
            "total_videos": len(self.videos),
            "total_issues": len(self.issues),
        }

    def _group_issues_by_severity(self) -> Dict[str, int]:
        grouped = defaultdict(int)
        for issue in self.issues:
            grouped[issue["severity"]] += 1
        return dict(grouped)

    def _group_issues_by_type(self) -> Dict[str, int]:
        grouped = defaultdict(int)
        for issue in self.issues:
            grouped[issue["type"]] += 1
        return dict(grouped)

    def print_report(self) -> None:
        print("=" * 60)
        print("Micro Emotion Component Validation Report")
        print("=" * 60)
        print(f"Total videos: {len(self.videos)}")
        print(f"Total issues: {len(self.issues)}")
        print()

        if not self.issues:
            print("✅ No issues found!")
            return

        by_severity = self._group_issues_by_severity()
        by_type = self._group_issues_by_type()

        print("Issues by severity:")
        for severity, count in sorted(by_severity.items()):
            print(f"  {severity}: {count}")
        print()

        print("Issues by type:")
        for issue_type, count in sorted(by_type.items()):
            print(f"  {issue_type}: {count}")
        print()

        print("Details:")
        for issue in self.issues[:20]:  # Show first 20
            print(f"  [{issue['severity']}] {issue['video_id']}: {issue['message']}")
        if len(self.issues) > 20:
            print(f"  ... and {len(self.issues) - 20} more issues")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate micro_emotion component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    parser.add_argument("--platform-id", default="youtube", help="Platform ID (default: youtube)")
    args = parser.parse_args()

    validator = MicroEmotionValidator(args.results_base)
    validator.validate_all(args.platform_id)
    validator.print_report()

    return 0 if len(validator.issues) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

