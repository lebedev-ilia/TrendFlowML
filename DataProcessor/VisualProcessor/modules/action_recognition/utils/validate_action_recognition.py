#!/usr/bin/env python3
"""Валидатор action_recognition NPZ: схема, --struct, --qa (view_csv_feature_qa.json), опционально --legacy (полный отчёт)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCHEMA = "action_recognition_npz_v2"
_REQUIRED = ("tracks", "embeddings", "results_json", "meta")


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


def _as_results_json_list(d: Dict[str, Any]) -> List[Any]:
    rj = d.get("results_json")
    if rj is None:
        return []
    if isinstance(rj, np.ndarray) and rj.dtype == object:
        return [rj[i] for i in range(int(rj.shape[0]))]
    if isinstance(rj, (list, tuple)):
        return list(rj)
    return [rj]


def _as_embeddings_list(d: Dict[str, Any]) -> List[Any]:
    e = d.get("embeddings")
    if e is None:
        return []
    if isinstance(e, np.ndarray) and e.dtype == object:
        return [e[i] for i in range(int(e.shape[0]))]
    if isinstance(e, (list, tuple)):
        return list(e)
    return [e]


def validate_structure(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    meta = extract_meta(d)
    st = meta.get("status", "")
    if st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    tracks = d.get("tracks")
    if isinstance(tracks, list):
        tracks = np.asarray(tracks, dtype=np.int32)
    elif isinstance(tracks, np.ndarray):
        tracks = np.asarray(tracks, dtype=np.int32).ravel()
    else:
        out.append("tracks: невалидный тип")
        return out

    tracks_count = int(tracks.size)
    emb_list = _as_embeddings_list(d)
    rj_list = _as_results_json_list(d)
    embeddings_count = len(emb_list)
    results_count = len(rj_list)

    if tracks_count != embeddings_count:
        out.append(f"число tracks ({tracks_count}) != числу embeddings ({embeddings_count})")
    if tracks_count != results_count:
        out.append(f"число tracks ({tracks_count}) != len(results_json) ({results_count})")

    for i in range(min(tracks_count, embeddings_count, results_count)):
        track_id = int(tracks[i])

        emb = emb_list[i]
        if isinstance(emb, np.ndarray):
            if emb.ndim != 2:
                out.append(f"track {track_id}: embedding ож. 2D, факт {emb.ndim}D")
            elif emb.shape[1] != 256:
                out.append(f"track {track_id}: ож. dim=256, факт {emb.shape[1]}")

        rj = rj_list[i]
        if hasattr(rj, "item"):
            rj = rj.item()
        if not isinstance(rj, dict):
            out.append(f"track {track_id}: results_json[{i}] не dict")
            continue

        required_fields = (
            "embedding_normed_256d",
            "max_temporal_jump",
            "mean_temporal_jump",
            "stability",
            "stability_centroid_dist",
            "num_switches",
            "num_clips",
            "track_frame_count",
        )
        for field in required_fields:
            if field not in rj:
                out.append(f"track {track_id}: нет поля {field!r} в results_json")

        num_clips = rj.get("num_clips", 0)
        if not isinstance(num_clips, (int, np.integer)) or int(num_clips) < 0:
            out.append(f"track {track_id}: невалидный num_clips: {num_clips!r}")

    if not meta:
        out.append("meta пустой")
    return out


def _dataprocessor_on_path() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_qa_config() -> Tuple[Any, Path]:
    dp = _dataprocessor_on_path()
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
    comp = "action_recognition"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


# --- legacy: полный отчёт (errors/warnings/metrics distribution) -----------------


def _validate_schema_strict(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in _REQUIRED:
        if key not in npz_data:
            errors.append(f"Missing required key: {key}")
    required_meta_keys = [
        "producer",
        "producer_version",
        "schema_version",
        "status",
        "created_at",
        "platform_id",
        "video_id",
        "run_id",
    ]
    for key in required_meta_keys:
        if key not in meta:
            errors.append(f"Missing required meta key: {key}")
    if meta.get("schema_version") != "action_recognition_npz_v2":
        errors.append(
            f"Invalid schema_version: {meta.get('schema_version')}, expected 'action_recognition_npz_v2'"
        )
    return errors


def _consistency_core(npz_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    tracks = npz_data.get("tracks")
    embeddings = _as_embeddings_list(npz_data)
    results_json = _as_results_json_list(npz_data)

    if isinstance(tracks, list):
        tracks = np.asarray(tracks, dtype=np.int32)
    elif isinstance(tracks, np.ndarray):
        tracks = np.asarray(tracks, dtype=np.int32).ravel()
    else:
        return (["tracks is not a valid array"], [])

    tracks_count = int(tracks.size)
    embeddings_count = len(embeddings)
    results_count = len(results_json)

    if tracks_count != embeddings_count:
        errors.append(f"Tracks count ({tracks_count}) != embeddings count ({embeddings_count})")
    if tracks_count != results_count:
        errors.append(f"Tracks count ({tracks_count}) != results_json count ({results_count})")

    for i in range(min(tracks_count, embeddings_count, results_count)):
        track_id = int(tracks[i])
        emb = embeddings[i]
        if isinstance(emb, np.ndarray):
            if emb.ndim != 2:
                errors.append(f"Track {track_id}: embedding should be 2D array, got {emb.ndim}D")
            elif emb.shape[1] != 256:
                errors.append(f"Track {track_id}: embedding dimension should be 256, got {emb.shape[1]}")
            norms = np.linalg.norm(emb, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                warnings.append(
                    f"Track {track_id}: embeddings not properly L2-normalized (norms: {norms.min():.3f}-{norms.max():.3f})"
                )

        rj = results_json[i]
        if hasattr(rj, "item"):
            rj = rj.item()
        if not isinstance(rj, dict):
            errors.append(f"Track {track_id}: results_json[{i}] is not a dict")
            continue

        required_fields = [
            "embedding_normed_256d",
            "max_temporal_jump",
            "mean_temporal_jump",
            "stability",
            "stability_centroid_dist",
            "num_switches",
            "num_clips",
            "track_frame_count",
        ]
        for field in required_fields:
            if field not in rj:
                errors.append(f"Track {track_id}: missing field '{field}' in results_json")

        num_clips = rj.get("num_clips", 0)
        if not isinstance(num_clips, (int, np.integer)) or int(num_clips) < 0:
            errors.append(f"Track {track_id}: invalid num_clips: {num_clips}")

        stability = rj.get("stability")
        if stability is not None and isinstance(stability, (int, float, np.floating)):
            if not (0.0 <= float(stability) <= 1.0) and not np.isnan(stability):
                warnings.append(f"Track {track_id}: stability out of range [0,1]: {stability}")

        temporal_jumps = rj.get("temporal_jumps", [])
        if isinstance(temporal_jumps, (list, np.ndarray)) and len(temporal_jumps) > 0:
            expected_jumps = int(num_clips) - 1
            actual_jumps = len(temporal_jumps)
            if actual_jumps != expected_jumps:
                warnings.append(
                    f"Track {track_id}: temporal_jumps length ({actual_jumps}) != num_clips-1 ({expected_jumps})"
                )

    return errors, warnings


def _validate_metrics_distribution(npz_data: Dict[str, Any]) -> Dict[str, Any]:
    results_json = _as_results_json_list(npz_data)
    if not results_json:
        return {}

    metrics: Dict[str, List[float]] = {
        "stability": [],
        "stability_centroid_dist": [],
        "max_temporal_jump": [],
        "mean_temporal_jump": [],
        "num_clips": [],
        "num_switches": [],
    }

    for rj in results_json:
        if hasattr(rj, "item"):
            rj = rj.item()
        if not isinstance(rj, dict):
            continue
        for key in metrics:
            val = rj.get(key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                try:
                    metrics[key].append(float(val))
                except (TypeError, ValueError):
                    pass

    stats: Dict[str, Any] = {}
    for key, values in metrics.items():
        if values:
            arr = np.asarray(values, dtype=np.float32)
            stats[key] = {
                "count": len(values),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }
    return stats


def validate_action_recognition(npz_path: str, *, verbose: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "npz_path": str(npz_path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "stats": {},
        "metrics_distribution": {},
    }
    try:
        npz_data = load_npz(npz_path)
        meta = extract_meta(npz_data)
        result["errors"].extend(_validate_schema_strict(npz_data, meta))
        ce, cw = _consistency_errors_warnings(npz_data)
        result["errors"].extend(ce)
        result["warnings"].extend(cw)
        result["metrics_distribution"] = _validate_metrics_distribution(npz_data)
        tracks = npz_data.get("tracks", [])
        tr = np.asarray(tracks, dtype=np.int32).ravel() if isinstance(tracks, (list, np.ndarray)) else []
        tracks_count = int(tr.size) if tr.size else 0
        results_json = _as_results_json_list(npz_data)
        total_clips = 0
        for rj in results_json:
            if hasattr(rj, "item"):
                rj = rj.item()
            if isinstance(rj, dict):
                total_clips += int(rj.get("num_clips", 0) or 0)
        result["stats"] = {
            "tracks_count": tracks_count,
            "total_clips": total_clips,
            "status": meta.get("status", "unknown"),
            "schema_version": meta.get("schema_version", "unknown"),
            "producer_version": meta.get("producer_version", "unknown"),
        }
        result["valid"] = len(result["errors"]) == 0
    except Exception as e:
        result["errors"].append(f"Exception during validation: {e}")
        if verbose:
            import traceback

            result["errors"].append(traceback.format_exc())
    return result


def _run_batch(*, results_base: str, platform_id: str) -> int:
    """
    Батч по каталогу (scripts/wait_and_analyze.sh): ищет action_recognition_features.npz,
    для каждого — schema + struct.
    """
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("action_recognition/action_recognition_features.npz")):
        n += 1
        ok = validate_schema(str(npz))
        st = validate_structure(str(npz)) if ok else ["INVALID schema"]
        if not ok or st:
            ex = max(ex, 2)
        status = "OK" if ok and not st else "ISSUES"
        print(f"[{status}] {npz}", flush=True)
        for line in st:
            print(f"    - {line}", flush=True)
    print(f"Проверено файлов: {n}", flush=True)
    return ex if n else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=f"validate action_recognition NPZ (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument(
        "--results-base",
        help="[батч] корень result_store; обход **/action_recognition/action_recognition_features.npz",
    )
    p.add_argument("--platform-id", default="youtube", help="[батч] субкаталог платформы")
    p.add_argument("--qa", action="store_true")
    p.add_argument("--struct", action="store_true")
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Полный отчёт (строгий meta, warnings, distribution) в стиле старого CLI",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--json", action="store_true", help="Только с --legacy: вывод JSON")
    args = p.parse_args()

    if args.results_base:
        return _run_batch(results_base=args.results_base, platform_id=args.platform_id or "youtube")

    if not args.npz_path:
        p.error("нужен npz_path или --results-base")
        return 1

    if args.legacy:
        result = validate_action_recognition(args.npz_path, verbose=args.verbose)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Validation result for: {result['npz_path']}")
            print(f"Status: {'✅ VALID' if result['valid'] else '❌ INVALID'}")
            print(f"\nStats:")
            for k, v in result["stats"].items():
                print(f"  {k}: {v}")
            if result["errors"]:
                print(f"\n❌ Errors ({len(result['errors'])}):")
                for err in result["errors"]:
                    print(f"  - {err}")
            if result["warnings"]:
                print(f"\n⚠️  Warnings ({len(result['warnings'])}):")
                for warn in result["warnings"]:
                    print(f"  - {warn}")
            if result["metrics_distribution"]:
                print(f"\n📊 Metrics Distribution:")
                for metric, s in result["metrics_distribution"].items():
                    print(f"  {metric}:")
                    print(
                        f"    count={s['count']}, mean={s['mean']:.3f}, std={s['std']:.3f}"
                    )
                    print(f"    range=[{s['min']:.3f}, {s['max']:.3f}]")
        return 0 if result["valid"] else 1

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
