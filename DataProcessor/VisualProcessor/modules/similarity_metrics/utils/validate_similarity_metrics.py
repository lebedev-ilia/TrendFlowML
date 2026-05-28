#!/usr/bin/env python3
"""
Персональный валидатор для similarity_metrics компонента.

Проверяет:
- наличие обязательных ключей NPZ (similarity_metrics_npz_v3)
- размерности, dtype, монотонность осей, согласованность N/F
- базовый meta-контракт и статус
- диапазоны значений для similarity метрик ([-1, 1])
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


class SimilarityMetricsValidator:
    def __init__(self, results_base_path: str):
        self.results_base_path = Path(results_base_path)
        self.videos: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []

    def load_video_results(self, platform_id: str, video_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        sim_dir = self.results_base_path / platform_id / video_id / run_id / "similarity_metrics"
        npz_path = sim_dir / "results.npz"
        render_path = sim_dir / "_render" / "render_context.json"

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
            "centroid_sims",
            "temporal_sim_next",
            "reference_present",
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
        if any(k not in npz_data for k in ["frame_indices", "times_s", "centroid_sims", "temporal_sim_next"]):
            return

        # 3) shapes + types
        fi = npz_data.get("frame_indices")
        ts = npz_data.get("times_s")
        centroid_sims = npz_data.get("centroid_sims")
        temporal_sim_next = npz_data.get("temporal_sim_next")
        reference_present = npz_data.get("reference_present")
        fn = npz_data.get("feature_names")
        fv = npz_data.get("feature_values")

        # Convert to numpy arrays if needed
        if fi is not None and not isinstance(fi, np.ndarray):
            try:
                fi = np.asarray(fi, dtype=np.int32)
            except Exception:
                pass
        if ts is not None and not isinstance(ts, np.ndarray):
            try:
                ts = np.asarray(ts, dtype=np.float32)
            except Exception:
                pass
        if centroid_sims is not None and not isinstance(centroid_sims, np.ndarray):
            try:
                centroid_sims = np.asarray(centroid_sims, dtype=np.float32)
            except Exception:
                pass
        if temporal_sim_next is not None and not isinstance(temporal_sim_next, np.ndarray):
            try:
                temporal_sim_next = np.asarray(temporal_sim_next, dtype=np.float32)
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

        # centroid_sims
        if not isinstance(centroid_sims, np.ndarray) or centroid_sims.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"centroid_sims must be 1D ndarray, got {type(centroid_sims)}",
            )
        else:
            if len(centroid_sims) != N:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"centroid_sims length ({len(centroid_sims)}) != frame_indices length ({N})",
                )
            if centroid_sims.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"centroid_sims dtype should be float32, got {centroid_sims.dtype}",
                )
            # cosine similarity should be in [-1, 1]
            finite_centroid = centroid_sims[np.isfinite(centroid_sims)]
            if finite_centroid.size > 0:
                if np.any(finite_centroid < -1.0) or np.any(finite_centroid > 1.0):
                    self._issue(
                        issue_type="invalid_value",
                        severity="warning",
                        video_id=video_id,
                        message=f"centroid_sims should be in [-1, 1] (cosine similarity), got range [{np.min(finite_centroid):.4f}, {np.max(finite_centroid):.4f}]",
                    )

        # temporal_sim_next
        if not isinstance(temporal_sim_next, np.ndarray) or temporal_sim_next.ndim != 1:
            self._issue(
                issue_type="invalid_shape",
                severity="error",
                video_id=video_id,
                message=f"temporal_sim_next must be 1D ndarray, got {type(temporal_sim_next)}",
            )
        else:
            expected_len = N - 1 if N > 1 else 0
            if len(temporal_sim_next) != expected_len:
                self._issue(
                    issue_type="dimension_mismatch",
                    severity="error",
                    video_id=video_id,
                    message=f"temporal_sim_next length ({len(temporal_sim_next)}) != N-1 ({expected_len})",
                )
            if temporal_sim_next.dtype != np.float32:
                self._issue(
                    issue_type="invalid_dtype",
                    severity="warning",
                    video_id=video_id,
                    message=f"temporal_sim_next dtype should be float32, got {temporal_sim_next.dtype}",
                )
            # cosine similarity should be in [-1, 1]
            finite_temporal = temporal_sim_next[np.isfinite(temporal_sim_next)]
            if finite_temporal.size > 0:
                if np.any(finite_temporal < -1.0) or np.any(finite_temporal > 1.0):
                    self._issue(
                        issue_type="invalid_value",
                        severity="warning",
                        video_id=video_id,
                        message=f"temporal_sim_next should be in [-1, 1] (cosine similarity), got range [{np.min(finite_temporal):.4f}, {np.max(finite_temporal):.4f}]",
                    )

        # reference_present
        if reference_present is None:
            self._issue(
                issue_type="missing_key",
                severity="error",
                video_id=video_id,
                message="reference_present is missing",
            )
        else:
            # Convert to bool if needed (npz can store as numpy bool)
            if isinstance(reference_present, np.ndarray):
                reference_present = bool(reference_present.item())
            elif not isinstance(reference_present, bool):
                try:
                    reference_present = bool(reference_present)
                except Exception:
                    self._issue(
                        issue_type="invalid_dtype",
                        severity="error",
                        video_id=video_id,
                        message=f"reference_present must be bool, got {type(reference_present)}",
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
                    message=f"feature_names/feature_values must be 1D arrays, got shapes {fn.shape} / {fv.shape}",
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

                # Check for expected feature names (intra-video coherence metrics)
                try:
                    names = [str(x) for x in fn.reshape(-1).tolist()]
                    expected_intra_video = [
                        "n_frames",
                        "centroid_sim_mean",
                        "centroid_sim_std",
                        "centroid_sim_p10",
                        "centroid_sim_p90",
                        "temporal_sim_mean",
                        "temporal_sim_std",
                    ]
                    # These should always be present (intra-video coherence)
                    missing = [n for n in expected_intra_video if n not in names]
                    if missing:
                        self._issue(
                            issue_type="missing_feature_name",
                            severity="warning",
                            video_id=video_id,
                            message=f"feature_names missing expected intra-video coherence items: {missing}",
                        )
                except Exception:
                    self._issue(
                        issue_type="feature_names_parse",
                        severity="warning",
                        video_id=video_id,
                        message="Failed to parse feature_names as strings",
                    )

    def validate_all(self, platform_id: str = "youtube") -> Dict[str, Any]:
        platform_dir = self.results_base_path / platform_id
        if not platform_dir.exists():
            return {"error": f"Platform directory not found: {platform_dir}"}

        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith("test_similarity_metrics"):
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
        print("Similarity Metrics Component Validation Report")
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
    parser = argparse.ArgumentParser(description="Validate similarity_metrics component results")
    parser.add_argument("--results-base", required=True, help="Base path to results directory")
    parser.add_argument("--platform-id", default="youtube", help="Platform ID (default: youtube)")
    args = parser.parse_args()

    v = SimilarityMetricsValidator(args.results_base)
    result = v.validate_all(args.platform_id)
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    v.print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

