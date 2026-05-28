#!/usr/bin/env python3
"""Валидатор emotion_face NPZ: --struct, --qa; батч --results-base (test_emotion_face_*)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SCHEMA = "emotion_face_npz_v3"
_REQUIRED = (
    "frame_indices",
    "times_s",
    "face_present",
    "processed_mask",
    "face_count",
    "valence",
    "arousal",
    "intensity",
    "emotion_confidence",
    "emotion_probs",
    "dominant_emotion_id",
    "sequence_features",
    "keyframes",
    "summary",
    "features",
    "advanced_features",
    "meta",
)


def load_npz(npz_path: str) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=True)
    try:
        out: Dict[str, Any] = {}
        for k in z.files:
            v = z[k]
            if isinstance(v, np.ndarray) and v.dtype == object and getattr(v, "shape", None) == ():
                try:
                    out[k] = v.item()
                except Exception:
                    out[k] = v
            else:
                out[k] = v
        return out
    finally:
        try:
            z.close()
        except Exception:
            pass


def extract_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    m = d.get("meta")
    if m is None:
        return {}
    if isinstance(m, np.ndarray) and m.dtype == object and m.shape == ():
        m = m.item()
    return m if isinstance(m, dict) else {}


def validate_schema(npz_path: str) -> bool:
    try:
        d = load_npz(npz_path)
        for k in _REQUIRED:
            if k not in d:
                return False
        meta = extract_meta(d)
        sv = str(meta.get("schema_version", ""))
        return _SCHEMA in sv
    except Exception:
        return False


def _as_f32_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).ravel()


def _validate_data_dict(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k in _REQUIRED:
        if k not in npz_data:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    st = str(meta.get("status") or "")
    if st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    fi = np.asarray(npz_data.get("frame_indices"), dtype=np.int32).ravel()
    ts = _as_f32_1d(npz_data.get("times_s"))
    N = int(fi.size)
    if int(ts.size) != N:
        out.append(f"len(times_s) != N ({ts.size} != {N})")
    elif N > 1 and np.any(np.diff(ts) < -1e-6):
        out.append("times_s не неубывает")

    fp = np.asarray(npz_data.get("face_present")).ravel()
    pm = np.asarray(npz_data.get("processed_mask")).ravel()
    fc = np.asarray(npz_data.get("face_count")).ravel()
    va = _as_f32_1d(npz_data.get("valence"))
    ar = _as_f32_1d(npz_data.get("arousal"))
    it = _as_f32_1d(npz_data.get("intensity"))
    ec = _as_f32_1d(npz_data.get("emotion_confidence"))
    de = np.asarray(npz_data.get("dominant_emotion_id")).ravel()
    ep = np.asarray(npz_data.get("emotion_probs"), dtype=np.float32)

    for name, arr in (
        ("face_present", fp),
        ("processed_mask", pm),
        ("face_count", fc),
        ("valence", va),
        ("arousal", ar),
        ("intensity", it),
        ("emotion_confidence", ec),
        ("dominant_emotion_id", de),
    ):
        if int(arr.size) != N:
            out.append(f"{name}: len != N ({arr.size} != {N})")

    if ep.ndim != 2 or int(ep.shape[0]) != N:
        out.append("emotion_probs: ож. форма (N, 8)")
    elif int(ep.shape[1]) != 8:
        out.append("emotion_probs: ож. 8 классов (Ekman)")

    summ = npz_data.get("summary")
    if isinstance(summ, np.ndarray) and summ.dtype == object and summ.shape == ():
        summ = summ.item()
    if not isinstance(summ, dict):
        out.append("summary: ож. dict")

    kf = npz_data.get("keyframes")
    if isinstance(kf, np.ndarray) and kf.dtype == object and kf.shape == ():
        kf = kf.item()
    if not isinstance(kf, (list, np.ndarray)):
        out.append("keyframes: ож. list или ndarray")

    for name in ("sequence_features", "features", "advanced_features"):
        v = npz_data.get(name)
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            v = v.item()
        if not isinstance(v, dict):
            out.append(f"{name}: ож. dict, факт {type(v)}")

    if not meta:
        out.append("meta пустой")
    return out


def validate_structure(npz_path: str) -> List[str]:
    d = load_npz(npz_path)
    m = extract_meta(d)
    if m.get("status") == "error":
        return []
    return _validate_data_dict(d, m)


def _load_qa_config() -> Tuple[Any, Path]:
    dp = Path(__file__).resolve().parents[4]
    r = str(dp)
    if r not in sys.path:
        sys.path.insert(0, r)
    from qa.component_feature_qa import find_repo_root_from_path, load_qa_config

    root = find_repo_root_from_path(Path(__file__))
    if root is None:
        raise FileNotFoundError("view_csv_feature_qa.json (repo root not found)")
    path = root / "storage" / "result_store" / "view_csv_feature_qa.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return load_qa_config(path), path


def validate_qa_rows(npz_path: str, qa: Any) -> List[str]:
    from qa.component_feature_qa import flatten_meta

    d = load_npz(npz_path)
    meta = extract_meta(d)
    flat: Dict[str, Any] = dict(flatten_meta(meta, prefix="meta_"))
    warnings: List[str] = []
    comp = "emotion_face"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


class EmotionFaceValidator:
    def __init__(self, results_base_path: str) -> None:
        self.results_base_path = Path(results_base_path)
        self.videos: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []

    def load_video_results(
        self, platform_id: str, video_id: str, run_id: str
    ) -> Optional[Dict[str, Any]]:
        emotion_face_dir = (
            self.results_base_path / platform_id / video_id / run_id / "emotion_face"
        )
        npz_path = emotion_face_dir / "emotion_face.npz"
        render_path = emotion_face_dir / "_render" / "render_context.json"
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
        except Exception as e:  # pragma: no cover
            print(f"Error loading {video_id}: {e}")
            return None

    def validate_single_video(self, video_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        video_id = video_data["video_id"]
        npz_data = video_data["npz_data"]
        meta = video_data["meta"]
        st = str(meta.get("status") or "")
        if st == "error":
            return [
                {
                    "type": "status",
                    "severity": "error",
                    "video_id": video_id,
                    "message": f"status=error: {meta.get('empty_reason')}",
                }
            ]
        msgs = _validate_data_dict(npz_data, meta)
        return [
            {
                "type": "structure",
                "severity": "error",
                "video_id": video_id,
                "message": m,
            }
            for m in msgs
        ]

    def validate_all(self, platform_id: str = "youtube") -> Dict[str, Any]:
        platform_dir = self.results_base_path / platform_id
        if not platform_dir.exists():
            return {"error": f"Platform directory not found: {platform_dir}"}

        for video_dir in platform_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith("test_emotion_face"):
                continue
            video_id = video_dir.name
            run_id = video_id
            video_data = self.load_video_results(platform_id, video_id, run_id)
            if video_data is None:
                continue
            self.videos.append(video_data)
            self.issues.extend(self.validate_single_video(video_data))

        issues_by_type: Dict[str, int] = defaultdict(int)
        issues_by_severity: Dict[str, int] = defaultdict(int)
        for issue in self.issues:
            issues_by_type[issue["type"]] += 1
            issues_by_severity[issue["severity"]] += 1

        summary_stats: Dict[str, Any] = {}
        if self.videos:
            key_metrics = [
                "frames_count",
                "processed_frames",
                "face_present_ratio",
                "processed_ratio",
            ]
            for metric in key_metrics:
                values: List[float] = []
                for video_data in self.videos:
                    summary = video_data["npz_data"].get("summary")
                    if isinstance(summary, np.ndarray) and summary.ndim == 0:
                        summary = summary.item()
                    if (
                        isinstance(summary, dict)
                        and metric in summary
                        and isinstance(
                            summary[metric], (int, float, np.number)
                        )
                        and np.isfinite(float(summary[metric]))
                    ):
                        values.append(float(summary[metric]))
                if values:
                    arr = np.asarray(values, dtype=np.float32)
                    summary_stats[metric] = {
                        "count": len(values),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "range": [float(np.min(arr)), float(np.max(arr))],
                        "median": float(np.median(arr)),
                    }

        return {
            "total_videos": len(self.videos),
            "total_issues": len(self.issues),
            "issues_by_type": dict(issues_by_type),
            "issues_by_severity": dict(issues_by_severity),
            "summary_statistics": summary_stats,
        }

    def print_report(self, results: Dict[str, Any]) -> None:
        print("=" * 60)
        print("Emotion Face Component Validation Report")
        print("=" * 60)
        print(f"Total videos: {results['total_videos']}")
        print(f"Total issues: {results['total_issues']}")
        print()
        print("Issues by severity:")
        for sev, count in results["issues_by_severity"].items():
            print(f"  {sev}: {count}")
        print()
        print("Issues by type:")
        for t, count in results["issues_by_type"].items():
            print(f"  {t}: {count}")
        print()
        print("Summary statistics:")
        for metric, stats in results["summary_statistics"].items():
            print(f"  {metric}:")
            print(f"    count: {stats['count']}")
            print(f"    mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    range: [{stats['range'][0]:.4f}, {stats['range'][1]:.4f}]")
            print(f"    median: {stats['median']:.4f}")
        print()


def main() -> int:
    p = argparse.ArgumentParser(
        description=f"validate emotion_face NPZ (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к emotion_face.npz")
    p.add_argument(
        "--results-base",
        type=str,
        help="Каталог результатов (батч: test_emotion_face_* / emotion_face.npz)",
    )
    p.add_argument("--platform-id", default="youtube")
    p.add_argument("--qa", action="store_true")
    p.add_argument("--struct", action="store_true")
    args = p.parse_args()

    if args.results_base:
        validator = EmotionFaceValidator(args.results_base)
        results = validator.validate_all(args.platform_id)
        if "error" in results:
            print(f"Error: {results['error']}")
            return 1
        validator.print_report(results)
        return 0 if results["total_issues"] == 0 else 1

    if not args.npz_path:
        p.error("нужен npz_path или --results-base")
        return 1

    ok = validate_schema(args.npz_path)
    print("✅ VALID schema" if ok else "❌ INVALID schema")
    if not ok:
        return 1
    ex = 0
    if args.struct:
        st = validate_structure(args.npz_path)
        if st:
            print("⚠️  structure:")
            for s in st:
                print("  -", s)
            ex = max(ex, 2)
    if args.qa:
        try:
            qa, path = _load_qa_config()
        except Exception as e:
            print(f"QA: пропуск ({e})", flush=True)
            return ex or 0
        warns = validate_qa_rows(args.npz_path, qa)
        if warns:
            print(f"⚠️  QA warnings ({path}):")
            for w in warns:
                print("  -", w)
            ex = max(ex, 2)
        else:
            print(f"✅ QA OK (rules {path})")
    return ex


if __name__ == "__main__":
    sys.exit(main())
