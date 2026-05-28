#!/usr/bin/env python3
"""Валидатор color_light NPZ: --struct, --qa; батч --results-base (отчёт по run-ам)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Синхрон с utils/processor.py — FRAME_COMPACT_KEYS
_FRAME_COMPACT_KEYS: Tuple[str, ...] = (
    "hue_mean_norm",
    "hue_std_norm",
    "hue_entropy_weighted",
    "sat_mean_norm",
    "val_mean_norm",
    "L_mean_norm",
    "global_contrast_norm",
    "local_contrast_mean_norm",
    "colorfulness_norm",
    "skin_tone_ratio",
    "overexposed_ratio",
    "underexposed_ratio",
    "vignetting_score_norm",
    "soft_light_prob",
    "dominant_lab_a_norm",
    "dominant_lab_b_norm",
)

_SCHEMA = "color_light_npz_v2"
_REQUIRED = (
    "frame_indices",
    "times_s",
    "sequence_frame_indices",
    "sequence_times_s",
    "sequence_inputs",
    "frame_compact_features",
    "frame_compact_feature_names",
    "frame_compact_frame_indices",
    "video_features",
    "aggregated",
    "scenes",
    "frames",
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
        return _SCHEMA in sv or "color_light" in sv
    except Exception:
        return False


def _as_f32_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).ravel()


def _as_i32_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int32).ravel()


def _validate_data_dict(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k in _REQUIRED:
        if k not in npz_data:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    st = meta.get("status", "")
    if st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    for name in ("video_features", "aggregated", "scenes", "frames", "sequence_inputs"):
        v = npz_data.get(name)
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            v = v.item()
        if not isinstance(v, dict):
            out.append(f"{name} должен быть dict (после unbox), факт: {type(v)}")

    fi = _as_i32_1d(npz_data.get("frame_indices"))
    ts = _as_f32_1d(npz_data.get("times_s"))
    if int(fi.size) != int(ts.size):
        out.append(f"len(frame_indices) != len(times_s) ({fi.size} != {ts.size})")

    sqi = _as_i32_1d(npz_data.get("sequence_frame_indices"))
    sqt = _as_f32_1d(npz_data.get("sequence_times_s"))
    if int(sqi.size) != int(sqt.size):
        out.append("len(sequence_frame_indices) != len(sequence_times_s)")

    M = int(sqi.size)
    fcf = np.asarray(npz_data.get("frame_compact_features"), dtype=np.float32)
    if fcf.ndim != 2:
        out.append("frame_compact_features: ож. 2D")
    else:
        if int(fcf.shape[0]) != M:
            out.append("frame_compact_features: первая ось != len(sequence_frame_indices)")
        d = int(fcf.shape[1]) if fcf.ndim == 2 else 0
        if d != len(_FRAME_COMPACT_KEYS):
            out.append(
                f"frame_compact_features: ож. {len(_FRAME_COMPACT_KEYS)} компонент, факт {d}"
            )

    fci = _as_i32_1d(npz_data.get("frame_compact_frame_indices"))
    if int(fci.size) != M:
        out.append("len(frame_compact_frame_indices) != M(sequence)")

    fn = np.asarray(npz_data.get("frame_compact_feature_names"), dtype=object).ravel()
    if int(fn.size) != len(_FRAME_COMPACT_KEYS):
        out.append("frame_compact_feature_names: длина != FRAME_COMPACT_KEYS")
    else:
        got = [str(x) for x in fn.tolist()]
        if got != list(_FRAME_COMPACT_KEYS):
            out.append("frame_compact_feature_names: порядок/состав != FRAME_COMPACT_KEYS")

    # `dominant_lab_a_norm` / `dominant_lab_b_norm` — не [0,1], см. docs/FEATURE_DESCRIPTION.md;
    # NaN в video_level метриках допустим при узких сценах/гистограммах.

    if not meta:
        out.append("meta пустой")
    return out


def validate_structure(npz_path: str) -> List[str]:
    d = load_npz(npz_path)
    m = extract_meta(d)
    st = m.get("status", "")
    if st == "error":
        return [f"meta.status=error: {m.get('empty_reason')!r} (struct не валидирует payload)"]
    return _validate_data_dict(d, m)


def _load_qa_config() -> Tuple[Any, Path]:
    from qa.component_feature_qa import find_repo_root_from_path, load_qa_config

    root = find_repo_root_from_path(Path(__file__))
    if root is None:
        raise FileNotFoundError("view_csv_feature_qa.json (repo root not found)")
    dp = root / "DataProcessor"
    r = str(dp)
    if r not in sys.path:
        sys.path.insert(0, r)
    path = root / "storage" / "result_store" / "view_csv_feature_qa.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return load_qa_config(path), path


def validate_ranges(npz_path: str) -> List[str]:
    """См. docs/FEATURE_DESCRIPTION.md; мягкие диапазоны по компактному вектору."""
    out: List[str] = []
    d = load_npz(npz_path)
    fi = _as_i32_1d(d.get("frame_indices"))
    ts = _as_f32_1d(d.get("times_s"))
    n = int(fi.size)
    if n > 1 and int(ts.size) == n and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд (union)")

    sqi = _as_i32_1d(d.get("sequence_frame_indices"))
    sqt = _as_f32_1d(d.get("sequence_times_s"))
    M = int(sqi.size)
    if M > 1 and int(sqt.size) == M and np.any(np.diff(sqt) < -1e-4):
        out.append("sequence_times_s: не неубывающий ряд")

    fcf = np.asarray(d.get("frame_compact_features"), dtype=np.float32)
    if fcf.ndim == 2 and fcf.shape[0] and fcf.shape[1] == len(_FRAME_COMPACT_KEYS):
        dcol = fcf.shape[1]
        # Явные доли / вероятности — [0,1]. Энтропии и «norm*» в ячейке могут выходить за 1
        for j, key in enumerate(_FRAME_COMPACT_KEYS):
            if key not in (
                "skin_tone_ratio",
                "overexposed_ratio",
                "underexposed_ratio",
                "vignetting_score_norm",
                "soft_light_prob",
            ):
                continue
            col = fcf[:, j]
            m = np.isfinite(col)
            if m.any():
                t = col[m]
                if np.any(t < -0.01) or np.any(t > 1.01):
                    out.append(
                        f"frame_compact {key}: вне [0,1] (finite), min={float(np.min(t)):.4f} max={float(np.max(t)):.4f}"
                    )
        for j in (dcol - 2, dcol - 1):
            col = fcf[:, j]
            m = np.isfinite(col)
            if m.any():
                t = col[m]
                if np.any(t < -2.5) or np.any(t > 2.5):
                    out.append(
                        f"frame_compact {_FRAME_COMPACT_KEYS[j]}: вне пилотного [-2.5, 2.5], "
                        f"min={float(np.min(t)):.4f} max={float(np.max(t)):.4f}"
                    )

    meta = extract_meta(d)
    tf = meta.get("total_frames")
    pf = meta.get("processed_frames")
    if tf is not None and pf is not None:
        try:
            if int(pf) > int(tf) >= 0:
                out.append("meta.processed_frames > meta.total_frames")
        except (TypeError, ValueError):
            pass
    return out


def validate_qa_rows(npz_path: str, qa: Any) -> List[str]:
    from qa.component_feature_qa import flatten_meta

    d = load_npz(npz_path)
    meta = extract_meta(d)
    flat: Dict[str, Any] = dict(flatten_meta(meta, prefix="meta_"))
    warnings: List[str] = []
    comp = "color_light"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("color_light/color_light_features.npz")):
        n += 1
        d = load_npz(str(npz))
        m = extract_meta(d)
        ok = validate_schema(str(npz))
        stl: List[str] = []
        if not ok:
            stl = ["INVALID schema"]
        elif m.get("status") == "error":
            stl = [f"meta.status=error: {m.get('empty_reason')}"]
        else:
            stl = _validate_data_dict(d, m)
        if not ok or stl:
            ex = max(ex, 2)
        status = "OK" if ok and not stl else "ISSUES"
        print(f"[{status}] {npz}", flush=True)
        for line in stl:
            print(f"    - {line}", flush=True)
    print(f"Проверено файлов: {n}", flush=True)
    return ex if n else 1


# --- батч: прежняя агрегация по run-ам ---------------------------------


class ColorLightValidator:
    def __init__(self, results_base_path: str) -> None:
        self.results_base_path = Path(results_base_path)
        self.videos: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []

    def load_video_results(
        self, platform_id: str, video_id: str, run_id: str
    ) -> Optional[Dict[str, Any]]:
        npz_path = (
            self.results_base_path
            / platform_id
            / video_id
            / run_id
            / "color_light"
            / "color_light_features.npz"
        )
        render_path = (
            self.results_base_path
            / platform_id
            / video_id
            / run_id
            / "color_light"
            / "_render"
            / "render_context.json"
        )
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

    def validate_all(
        self, platform_id: str = "youtube", video_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if video_ids is None:
            base_dir = self.results_base_path / platform_id
            if not base_dir.exists():
                return {"error": f"Base directory not found: {base_dir}"}
            video_ids = []
            for video_dir in base_dir.iterdir():
                if video_dir.is_dir():
                    run_dir = video_dir / video_dir.name
                    if (run_dir / "color_light" / "color_light_features.npz").exists():
                        video_ids.append(video_dir.name)

        all_issues: List[Dict[str, Any]] = []
        all_videos: List[Dict[str, Any]] = []

        for video_id in video_ids:
            video_data = self.load_video_results(platform_id, video_id, video_id)
            if video_data is None:
                continue
            all_videos.append(video_data)
            all_issues.extend(self.validate_single_video(video_data))

        self.videos = all_videos
        self.issues = all_issues

        return {
            "total_videos": len(all_videos),
            "total_issues": len(all_issues),
            "issues_by_type": self._group_issues_by_type(),
            "issues_by_severity": self._group_issues_by_severity(),
            "videos": [v["video_id"] for v in all_videos],
        }

    def _group_issues_by_type(self) -> Dict[str, int]:
        groups: Dict[str, int] = defaultdict(int)
        for issue in self.issues:
            groups[issue["type"]] += 1
        return dict(groups)

    def _group_issues_by_severity(self) -> Dict[str, int]:
        groups: Dict[str, int] = defaultdict(int)
        for issue in self.issues:
            groups[issue["severity"]] += 1
        return dict(groups)

    def get_summary_stats(self) -> Dict[str, Any]:
        if not self.videos:
            return {}
        stats: Dict[str, List[float]] = {
            "frames_count": [],
            "scenes_count": [],
            "color_distribution_entropy": [],
            "color_distribution_gini": [],
            "global_brightness_change_speed": [],
            "global_color_change_speed": [],
        }
        for video_data in self.videos:
            npz_data = video_data["npz_data"]
            frame_indices = npz_data.get("frame_indices")
            if frame_indices is not None:
                stats["frames_count"].append(len(np.asarray(frame_indices).ravel()))
            scenes = npz_data.get("scenes", {})
            if isinstance(scenes, np.ndarray) and scenes.dtype == object and scenes.shape == ():
                scenes = scenes.item()
            if isinstance(scenes, dict):
                stats["scenes_count"].append(len(scenes))
            video_features = npz_data.get("video_features", {})
            if isinstance(video_features, np.ndarray) and video_features.dtype == object:
                video_features = video_features.item() if video_features.size == 1 else {}
            if isinstance(video_features, dict):
                for key in list(stats.keys()):
                    if key in video_features and key not in ("frames_count", "scenes_count"):
                        val = video_features[key]
                        if isinstance(val, (int, float)) and np.isfinite(val):
                            stats[key].append(float(val))
        summary: Dict[str, Any] = {}
        for key, values in stats.items():
            if values:
                arr = np.asarray(values, dtype=np.float32)
                summary[key] = {
                    "count": len(values),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                }
        return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description=f"validate color_light NPZ (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument(
        "npz_path",
        nargs="?",
        help="Путь к color_light_features.npz (если не задан --results-base)",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; обход **/color_light/color_light_features.npz",
    )
    p.add_argument("--platform-id", type=str, default="youtube")
    p.add_argument(
        "--legacy-report",
        action="store_true",
        help="Старый отчёт ColorLightValidator (test_* run_id==video_id), а не rglob-сканер",
    )
    p.add_argument("--video-ids", type=str, nargs="+", default=None)
    p.add_argument("--qa", action="store_true")
    p.add_argument("--struct", action="store_true")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Временные оси, компакт 14×[0,1] + 2×LAB, processed≤total (см. docs/FEATURE_DESCRIPTION.md).",
    )
    args = p.parse_args()

    if args.results_base:
        if args.legacy_report:
            validator = ColorLightValidator(args.results_base)
            result = validator.validate_all(args.platform_id, args.video_ids)
            if "error" in result:
                print(f"Error: {result['error']}")
                return 1
            print("=" * 60)
            print("Color Light Component Validation Report (legacy)")
            print("=" * 60)
            print(f"Total videos: {result.get('total_videos', 0)}")
            print(f"Total issues: {result.get('total_issues', 0)}")
            print()
            if result.get("issues_by_severity"):
                print("Issues by severity:")
                for severity, count in result["issues_by_severity"].items():
                    print(f"  {severity}: {count}")
                print()
            if result.get("issues_by_type"):
                print("Issues by type:")
                for issue_type, count in result["issues_by_type"].items():
                    print(f"  {issue_type}: {count}")
                print()
            summary = validator.get_summary_stats()
            if summary:
                print("Summary statistics:")
                for key, s in summary.items():
                    print(f"  {key}:")
                    print(f"    count: {s['count']}")
                    print(f"    mean: {s['mean']:.4f} ± {s['std']:.4f}")
                    print(f"    range: [{s['min']:.4f}, {s['max']:.4f}]")
                    print(f"    median: {s['median']:.4f}")
                print()
            errors = [i for i in validator.issues if i["severity"] == "error"]
            if errors:
                print("Errors:")
                for error in errors[:20]:
                    print(f"  {error['video_id']}: {error['message']}")
                if len(errors) > 20:
                    print(f"  ... and {len(errors) - 20} more errors")
                print()
            return 0 if result.get("total_issues", 0) == 0 else 1
        return _run_batch_rglob(results_base=args.results_base, platform_id=args.platform_id or "youtube")

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
        else:
            print("✅ Structure OK (M×16, FRAME_COMPACT_KEYS)")

    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (time axes, [0,1] vs LAB, meta)")

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
