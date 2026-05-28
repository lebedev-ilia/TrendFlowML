#!/usr/bin/env python3
"""
Персональный валидатор для optical_flow компонента.

Проверяет:
- наличие обязательных ключей NPZ
- размерности, dtype, монотонность осей, согласованность N/D/F
- meta.status и базовые метаданные
- фиксированный набор feature_names и базовые sanity-checks значений
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

# Add VisualProcessor to path
vp_root = Path(__file__).resolve().parent.parent.parent
if str(vp_root) not in sys.path:
    sys.path.insert(0, str(vp_root))

from utils.renderer import load_npz, extract_meta


FIXED_FEATURE_NAMES = [
    "motion_curve_mean",
    "motion_curve_median",
    "motion_curve_p90",
    "motion_curve_variance",
    "missing_frame_ratio",
    "cam_shake_std_mean",
    "cam_rotation_abs_mean",
    "cam_translation_abs_mean",
    "flow_consistency_mean",
]


class OpticalFlowValidator:
    def __init__(self, results_base_path: str):
        self.results_base_path = Path(results_base_path)
        self.videos: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []

    def load_video_results(self, platform_id: str, video_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        optical_flow_dir = self.results_base_path / platform_id / video_id / run_id / "optical_flow"
        npz_path = optical_flow_dir / "optical_flow.npz"
        render_path = optical_flow_dir / "_render" / "render_context.json"

        if not npz_path.exists():
            return None

        try:
            npz_data = load_npz(str(npz_path))
            meta = extract_meta(npz_data)
            render_data = {}
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
        except Exception as e:
            print(f"Error loading {video_id}: {e}")
            return None

    def _issue(self, *, issue_type: str, severity: str, video_id: str, message: str, **extra):
        self.issues.append(
            {
                "type": issue_type,
                "severity": severity,
                "video_id": video_id,
                "message": message,
                **extra,
            }
        )

    def validate_single_video(self, video_data: Dict[str, Any]) -> None:
        video_id = video_data["video_id"]
        npz_data = video_data["npz_data"]
        meta = video_data["meta"]

        # 1) meta.status
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

        # 2) required keys
        required_keys = [
            "frame_indices",
            "times_s",
            "motion_norm_per_sec_mean",
            "frame_feature_names",
            "frame_feature_values",
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

        # If critical keys missing, stop deep checks
        if any(k not in npz_data for k in ["frame_indices", "times_s", "motion_norm_per_sec_mean"]):
            return

        # 3) shapes + types
        fi = npz_data.get("frame_indices")
        ts = npz_data.get("times_s")
        motion = npz_data.get("motion_norm_per_sec_mean")
        ffn = npz_data.get("frame_feature_names")
        ffv = npz_data.get("frame_feature_values")
        fn = npz_data.get("feature_names")
        fv = npz_data.get("feature_values")

        # Convert to numpy arrays if needed
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
        if motion is not None and not isinstance(motion, np.ndarray):
            try:
                motion = np.asarray(motion)
            except Exception:
                pass

        # frame_indices
        if not isinstance(fi, np.ndarray) or fi.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"frame_indices must be 1D ndarray, got {type(fi)}",
            )
            return
        if fi.dtype != np.int32:
            self._issue(
                issue_type="invalid_dtype",
                severity="warning",
                video_id=video_id,
                message=f"frame_indices dtype should be int32, got {fi.dtype}",
            )
        if len(fi) > 1 and not np.all(np.diff(fi) > 0):
            self._issue(
                issue_type="invalid_value",
                severity="error",
                video_id=video_id,
                message="frame_indices must be sorted and unique",
            )
        if np.any(fi < 0):
            self._issue(
                issue_type="invalid_value",
                severity="error",
                video_id=video_id,
                message="frame_indices contains negative values",
            )

        N = int(len(fi))

        # times_s
        if not isinstance(ts, np.ndarray) or ts.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"times_s must be 1D ndarray, got {type(ts)}",
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
            if ts.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"times_s dtype should be float32, got {ts.dtype}",
                )

        # motion curve
        if not isinstance(motion, np.ndarray) or motion.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"motion_norm_per_sec_mean must be 1D ndarray, got {type(motion)}",
            )
        else:
            if len(motion) != N:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"motion_norm_per_sec_mean length ({len(motion)}) != frame_indices length ({N})",
                )
            if motion.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"motion_norm_per_sec_mean dtype should be float32, got {motion.dtype}",
                )
            # allow NaNs (missing frames), but sanity: motion should not be negative
            finite_motion = motion[np.isfinite(motion)]
            if finite_motion.size > 0 and np.any(finite_motion < 0):
                self._issue(
                    issue_type="invalid_value",
                    severity="warning",
                    video_id=video_id,
                    message="motion_norm_per_sec_mean has negative finite values (unexpected for norm)",
                )

        # frame_feature_names / values
        # Convert to numpy array if needed (object arrays can be saved as lists)
        if ffn is not None and not isinstance(ffn, np.ndarray):
            try:
                ffn = np.asarray(ffn, dtype=object)
            except Exception:
                pass
        
        if isinstance(ffn, np.ndarray):
            if ffn.ndim != 1:
                self._issue(
                    issue_type="invalid_shape",
                    severity="error",
                    video_id=video_id,
                    message=f"frame_feature_names must be 1D array, got shape {ffn.shape}",
                )
        else:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"frame_feature_names must be ndarray, got {type(ffn)}",
            )

        # Convert to numpy array if needed
        if ffv is not None and not isinstance(ffv, np.ndarray):
            try:
                ffv = np.asarray(ffv, dtype=np.float32)
            except Exception:
                pass
        
        if isinstance(ffv, np.ndarray):
            if ffv.ndim != 2:
                self._issue(
                    issue_type="invalid_shape",
                    severity="error",
                    video_id=video_id,
                    message=f"frame_feature_values must be 2D array (N,D), got shape {ffv.shape}",
                )
            else:
                if ffv.shape[0] != N:
                    self._issue(
                        issue_type="dimension_mismatch",
                        severity="error",
                        video_id=video_id,
                        message=f"frame_feature_values rows ({ffv.shape[0]}) != N ({N})",
                    )
                if isinstance(ffn, np.ndarray) and ffv.shape[1] != len(ffn):
                    self._issue(
                        issue_type="dimension_mismatch",
                        severity="error",
                        video_id=video_id,
                        message=f"frame_feature_values cols ({ffv.shape[1]}) != len(frame_feature_names) ({len(ffn)})",
                    )
                if ffv.dtype != np.float32:
                    self._issue(
                        issue_type="invalid_dtype",
                        severity="warning",
                        video_id=video_id,
                        message=f"frame_feature_values dtype should be float32, got {ffv.dtype}",
                    )
        else:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"frame_feature_values must be ndarray, got {type(ffv)}",
            )

        # feature_names / values (video-level)
        # Convert to numpy arrays if needed
        if fn is not None and not isinstance(fn, np.ndarray):
            try:
                fn = np.asarray(fn, dtype=object)
            except Exception:
                pass
        if fv is not None and not isinstance(fv, np.ndarray):
            try:
                fv = np.asarray(fv, dtype=np.float32)
            except Exception:
                pass
        
        if not isinstance(fn, np.ndarray) or not isinstance(fv, np.ndarray):
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"feature_names and feature_values must be ndarrays, got {type(fn)} / {type(fv)}",
            )
        elif isinstance(fn, np.ndarray) and isinstance(fv, np.ndarray):
            if fn.ndim != 1 or fv.ndim != 1:
                self._issue(
                    issue_type="invalid_shape",
                    severity="error",
                    video_id=video_id,
                    message=f"feature_names/feature_values must be 1D arrays, got shapes {getattr(fn,'shape',None)} / {getattr(fv,'shape',None)}",
                )
            else:
                if len(fn) != len(fv):
                    self._issue(
                        issue_type="dimension_mismatch",
                        severity="error",
                        video_id=video_id,
                        message=f"len(feature_names) ({len(fn)}) != len(feature_values) ({len(fv)})",
                    )
                if fv.dtype != np.float32:
                    self._issue(
                        issue_type="invalid_dtype",
                        severity="warning",
                        video_id=video_id,
                        message=f"feature_values dtype should be float32, got {fv.dtype}",
                    )

                # fixed set check
                try:
                    names = [str(x) for x in fn.reshape(-1).tolist()]
                    missing = [n for n in FIXED_FEATURE_NAMES if n not in names]
                    if missing:
                        self._issue(
                            issue_type="missing_feature_name",
                            severity="error",
                            video_id=video_id,
                            message=f"feature_names missing expected items: {missing}",
                        )
                    # sanity for missing_frame_ratio
                    if "missing_frame_ratio" in names:
                        idx = names.index("missing_frame_ratio")
                        val = float(fv.reshape(-1)[idx])
                        if not (0.0 <= val <= 1.0) and np.isfinite(val):
                            self._issue(
                                issue_type="invalid_value",
                                severity="warning",
                                video_id=video_id,
                                message=f"missing_frame_ratio should be in [0,1], got {val}",
                            )
                except Exception:
                    self._issue(
                        issue_type="feature_names_parse",
                        severity="warning",
                        video_id=video_id,
                        message="Failed to parse feature_names as strings for fixed-set validation",
                    )
        else:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"feature_names and feature_values must be ndarrays, got {type(fn)} / {type(fv)}",
            )

    def validate_all(self, platform_id: str = "youtube") -> Dict[str, Any]:
        platform_dir = self.results_base_path / platform_id
        if not platform_dir.exists():
            return {"error": f"Platform directory not found: {platform_dir}"}

        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith("test_optical_flow"):
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
        out = defaultdict(int)
        for it in self.issues:
            out[it["severity"]] += 1
        return dict(out)

    def _group_issues_by_type(self) -> Dict[str, int]:
        out = defaultdict(int)
        for it in self.issues:
            out[it["type"]] += 1
        return dict(out)

    def print_report(self) -> None:
        print("=" * 60)
        print("Optical Flow Component Validation Report")
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

        # Print a few concrete errors for faster debugging
        print("Sample issues:")
        for issue in self.issues[:20]:
            print(f"- [{issue['severity']}] {issue['video_id']} | {issue['type']}: {issue['message']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate optical_flow component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    parser.add_argument("--platform-id", default="youtube", help="Platform ID (default: youtube)")
    args = parser.parse_args()

    v = OpticalFlowValidator(args.results_base)
    result = v.validate_all(args.platform_id)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    v.print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


