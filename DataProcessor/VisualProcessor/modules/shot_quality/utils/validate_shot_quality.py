#!/usr/bin/env python3
"""
Персональный валидатор для shot_quality компонента.

Проверяет:
- наличие обязательных ключей NPZ (shot_quality_npz_v3)
- размерности, dtype, монотонность осей, согласованность N/F/S/K
- базовый meta-контракт и статус
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


class ShotQualityValidator:
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
        sq_dir = self.results_base_path / platform_id / video_id / run_id / "shot_quality"
        npz_path = sq_dir / "shot_quality.npz"
        render_path = sq_dir / "_render" / "render_context.json"

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

        if status == "empty":
            # shot_quality по схеме не должен быть empty — но на всякий случай
            if not meta.get("empty_reason"):
                self._issue(
                    issue_type="meta_empty_reason",
                    severity="warning",
                    video_id=video_id,
                    message="meta.status is 'empty' but meta.empty_reason is missing/empty",
                )

        # базовый meta-контракт
        for k in ("producer", "producer_version", "schema_version"):
            if k not in meta:
                self._issue(
                    issue_type="meta_missing_key",
                    severity="warning",
                    video_id=video_id,
                    message=f"meta missing key: {k}",
                )

        if meta.get("schema_version") not in ("shot_quality_npz_v3", None):
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
            "feature_names",
            "frame_features",
            "frame_feature_present_ratio",
            "quality_probs",
            "shot_ids",
            "shot_start_frame",
            "shot_end_frame",
            "shot_frame_count",
            "shot_features_mean",
            "shot_features_std",
            "shot_features_min",
            "shot_features_max",
            "shot_frame_feature_present_ratio",
            "shot_quality_topk_ids",
            "shot_quality_topk_probs",
            "shot_quality_conf_mean",
            "shot_quality_entropy_mean",
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
        if any(k not in npz_data for k in ("frame_indices", "times_s", "feature_names", "frame_features")):
            return

        # 3) ось кадров
        fi = self._as_array(npz_data.get("frame_indices"))
        ts = self._as_array(npz_data.get("times_s"))

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
                message="frame_indices must be strictly increasing (sorted+unique)",
            )

        if ts is None or ts.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"times_s must be 1D array, got {type(ts)}",
            )
        else:
            if ts.shape[0] != N:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"times_s length ({ts.shape[0]}) != frame_indices length ({N})",
                )
            if ts.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"times_s dtype should be float32, got {ts.dtype}",
                )
            if ts.shape[0] > 1 and np.any(np.diff(ts) < 0):
                self._issue(
                    issue_type="invalid_value",
                    severity="error",
                    video_id=video_id,
                    message="times_s is not monotonically increasing",
                )

        # 4) frame_features / feature_names
        feature_names = self._as_array(npz_data.get("feature_names"), dtype=object)
        frame_features = self._as_array(npz_data.get("frame_features"))

        if feature_names is None or feature_names.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"feature_names must be 1D array, got {None if feature_names is None else feature_names.shape}",
            )
            return
        if frame_features is None or frame_features.ndim != 2:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"frame_features must be 2D array (N,F), got {type(frame_features)}",
            )
            return

        F = int(feature_names.shape[0])
        if frame_features.shape[0] != N:
            self._issue(
                issue_type="dimension_mismatch",
                severity="error",
                video_id=video_id,
                message=f"frame_features rows ({frame_features.shape[0]}) != N ({N})",
            )
        if frame_features.shape[1] != F:
            self._issue(
                issue_type="dimension_mismatch",
                severity="error",
                video_id=video_id,
                message=f"frame_features cols ({frame_features.shape[1]}) != len(feature_names) ({F})",
            )
        if frame_features.dtype != np.float32:
            self._issue(
                issue_type="invalid_dtype",
                severity="warning",
                video_id=video_id,
                message=f"frame_features dtype should be float32, got {frame_features.dtype}",
            )

        # ratio per feature
        ffr = self._as_array(npz_data.get("frame_feature_present_ratio"))
        if ffr is None or ffr.ndim != 1 or ffr.shape[0] != F:
            self._issue(
                issue_type="dimension_mismatch",
                severity="error",
                video_id=video_id,
                message=f"frame_feature_present_ratio must be (F,), got {None if ffr is None else ffr.shape}, F={F}",
            )
        else:
            if ffr.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"frame_feature_present_ratio dtype should be float32, got {ffr.dtype}",
                )
            finite = ffr[np.isfinite(ffr)]
            if finite.size and (np.any(finite < 0.0) or np.any(finite > 1.0)):
                self._issue(
                    issue_type="invalid_value",
                    severity="warning",
                    video_id=video_id,
                    message="frame_feature_present_ratio contains values outside [0,1]",
                )

        # 5) quality_probs
        qp = self._as_array(npz_data.get("quality_probs"))
        if qp is None or qp.ndim != 2:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"quality_probs must be 2D array (N,P), got {type(qp)}",
            )
        else:
            if qp.shape[0] != N:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"quality_probs rows ({qp.shape[0]}) != N ({N})",
                )
            if qp.dtype != np.float16:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"quality_probs dtype should be float16, got {qp.dtype}",
                )
            finite = qp[np.isfinite(qp)]
            if finite.size and (np.any(finite < 0.0) or np.any(finite > 1.0)):
                self._issue(
                    issue_type="invalid_value",
                    severity="warning",
                    video_id=video_id,
                    message="quality_probs contains values outside [0,1]",
                )

        # 6) shot-level ось
        shot_ids = self._as_array(npz_data.get("shot_ids"))
        ss = self._as_array(npz_data.get("shot_start_frame"))
        se = self._as_array(npz_data.get("shot_end_frame"))
        sc = self._as_array(npz_data.get("shot_frame_count"))

        if shot_ids is None or shot_ids.ndim != 1 or shot_ids.shape[0] != N:
            self._issue(
                issue_type="dimension_mismatch",
                severity="error",
                video_id=video_id,
                message=f"shot_ids must be (N,), got {None if shot_ids is None else shot_ids.shape}, N={N}",
            )
        else:
            if shot_ids.dtype != np.int32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"shot_ids dtype should be int32, got {shot_ids.dtype}",
                )

        if ss is None or se is None or sc is None:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message="shot_start_frame / shot_end_frame / shot_frame_count must be present",
            )
            return

        if not (ss.ndim == se.ndim == sc.ndim == 1):
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message="shot_start_frame, shot_end_frame, shot_frame_count must be 1D arrays",
            )
            return

        S = int(ss.shape[0])
        if not (se.shape[0] == sc.shape[0] == S):
            self._issue(
                issue_type="dimension_mismatch",
                severity="error",
                video_id=video_id,
                message=f"shot_* arrays must have same length S, got start={ss.shape[0]}, end={se.shape[0]}, count={sc.shape[0]}",
            )

        # проверка границ шотов
        if S > 0:
            if ss.dtype != np.int32 or se.dtype != np.int32 or sc.dtype != np.int32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"shot_* dtypes should be int32, got {ss.dtype}/{se.dtype}/{sc.dtype}",
                )
            if np.any(se < ss):
                self._issue(
                    issue_type="invalid_value",
                    severity="error",
                    video_id=video_id,
                    message="shot_end_frame < shot_start_frame for some shots",
                )

        # shot-level feature матрицы
        shot_mean = self._as_array(npz_data.get("shot_features_mean"))
        shot_std = self._as_array(npz_data.get("shot_features_std"))
        shot_min = self._as_array(npz_data.get("shot_features_min"))
        shot_max = self._as_array(npz_data.get("shot_features_max"))

        for name, arr in [
            ("shot_features_mean", shot_mean),
            ("shot_features_std", shot_std),
            ("shot_features_min", shot_min),
            ("shot_features_max", shot_max),
        ]:
            if arr is None or arr.ndim != 2:
                self._issue(
                    issue_type="invalid_shape",
                    severity="error",
                    video_id=video_id,
                    message=f"{name} must be 2D array (S,F), got {type(arr)}",
                )
            else:
                if arr.shape[0] != S or arr.shape[1] != F:
                    self._issue(
                        issue_type="dimension_mismatch",
                        severity="error",
                        video_id=video_id,
                        message=f"{name} shape {arr.shape} != (S={S}, F={F})",
                    )

        sffr = self._as_array(npz_data.get("shot_frame_feature_present_ratio"))
        if sffr is None or sffr.ndim != 2 or sffr.shape != (S, F):
            self._issue(
                issue_type="dimension_mismatch",
                severity="error",
                video_id=video_id,
                message=f"shot_frame_feature_present_ratio must be (S,F), got {None if sffr is None else sffr.shape}",
            )

        # per-shot quality агрегаты
        sq_ids = self._as_array(npz_data.get("shot_quality_topk_ids"))
        sq_probs = self._as_array(npz_data.get("shot_quality_topk_probs"))
        sq_conf = self._as_array(npz_data.get("shot_quality_conf_mean"))
        sq_ent = self._as_array(npz_data.get("shot_quality_entropy_mean"))

        if sq_ids is None or sq_probs is None or sq_conf is None or sq_ent is None:
            self._issue(
                issue_type="missing_key",
                severity="error",
                video_id=video_id,
                message="shot_quality_* arrays must all be present",
            )
            return

        if sq_ids.ndim != 2 or sq_probs.ndim != 2:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message="shot_quality_topk_ids / shot_quality_topk_probs must be 2D (S,K)",
            )
        else:
            if sq_ids.shape != sq_probs.shape:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"shot_quality_topk_ids shape {sq_ids.shape} != shot_quality_topk_probs shape {sq_probs.shape}",
                )
            if sq_ids.shape[0] != S:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"shot_quality_topk_* rows ({sq_ids.shape[0]}) != S ({S})",
                )

        if sq_conf.ndim != 1 or sq_ent.ndim != 1 or sq_conf.shape[0] != S or sq_ent.shape[0] != S:
            self._issue(
                issue_type="dimension_mismatch",
                severity="error",
                video_id=video_id,
                message="shot_quality_conf_mean / shot_quality_entropy_mean must be (S,)",
            )

    # --- aggregation -----------------------------------------------------

    def _group_issues_by_severity(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for it in self.issues:
            out[it["severity"]] += 1
        return dict(out)

    def _group_issues_by_type(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for it in self.issues:
            out[it["type"]] += 1
        return dict(out)

    def validate_all(self, platform_id: str = "youtube") -> Dict[str, Any]:
        platform_dir = self.results_base_path / platform_id
        if not platform_dir.exists():
            return {"error": f"Platform directory not found: {platform_dir}"}

        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith("test_shot_quality"):
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

    def print_report(self) -> None:
        print("=" * 60)
        print("Shot Quality Component Validation Report")
        print("=" * 60)
        print(f"Total videos: {len(self.videos)}")
        print(f"Total issues: {len(self.issues)}")
        print()

        if not self.issues:
            print("✅ No issues found!")
            print()
            return

        print("Issues by severity:")
        for severity, count in self._group_issues_by_severity().items():
            print(f"  {severity}: {count}")
        print()

        print("Issues by type:")
        for t, count in self._group_issues_by_type().items():
            print(f"  {t}: {count}")
        print()

        print("Sample issues:")
        for issue in self.issues[:20]:
            print(f"- [{issue['severity']}] {issue['video_id']} | {issue['type']}: {issue['message']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate shot_quality component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    parser.add_argument("--platform-id", default="youtube", help="Platform ID (default: youtube)")
    args = parser.parse_args()

    v = ShotQualityValidator(args.results_base)
    result = v.validate_all(args.platform_id)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    v.print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


