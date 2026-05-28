#!/usr/bin/env python3
"""Валидатор core_clip/embeddings.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REQUIRED = (
    "frame_indices",
    "times_s",
    "frame_embeddings",
    "shot_quality_prompts",
    "shot_quality_text_embeddings",
    "scene_aesthetic_prompts",
    "scene_aesthetic_text_embeddings",
    "scene_luxury_prompts",
    "scene_luxury_text_embeddings",
    "scene_atmosphere_prompts",
    "scene_atmosphere_text_embeddings",
    "cut_detection_transition_prompts",
    "cut_detection_transition_text_embeddings",
    "popularity_topic_prompts",
    "popularity_topic_text_embeddings",
    "places365_prompts",
    "places365_text_embeddings",
    "consecutive_cosine_prev",
    "shot_quality_scores",
    "scene_aesthetic_scores",
    "scene_luxury_scores",
    "scene_atmosphere_scores",
    "cut_detection_transition_scores",
    "popularity_topic_scores",
    "places365_topk_indices",
    "places365_topk_scores",
    "places365_video_topk_indices",
    "places365_video_topk_scores",
    "meta",
)

_N_ROW_SCORES = (
    "shot_quality_scores",
    "scene_aesthetic_scores",
    "scene_luxury_scores",
    "scene_atmosphere_scores",
    "cut_detection_transition_scores",
    "popularity_topic_scores",
    "places365_topk_indices",
    "places365_topk_scores",
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
        return "core_clip_npz_v2" in sv or "core_clip" in sv
    except Exception:
        return False


def _load_qa_config() -> Tuple[Any, Path]:
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
    comp = "core_clip"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def _validate_data_dict(d: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    st = str(meta.get("status", "") or "")
    if st and st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int32).reshape(-1)
    n = int(fi.size)
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    if int(ts.size) != n:
        out.append(f"len(times_s)={ts.size} != N={n}")
    fe = np.asarray(d["frame_embeddings"], dtype=np.float32)
    if fe.ndim != 2 or int(fe.shape[0]) != n:
        out.append(f"frame_embeddings: ожидается (N,D), N={n}, факт {getattr(fe, 'shape', None)}")

    cc = np.asarray(d["consecutive_cosine_prev"], dtype=np.float32).reshape(-1)
    if int(cc.size) != n:
        out.append("consecutive_cosine_prev: длина != N")

    for k in _N_ROW_SCORES:
        a = np.asarray(d[k])
        if a.ndim >= 1 and int(a.shape[0]) != n:
            out.append(f"{k}: первая ось {a.shape[0]} != N={n}")

    vk = int(np.asarray(d["places365_video_topk_indices"]).size)
    vs = int(np.asarray(d["places365_video_topk_scores"]).size)
    if vk != vs:
        out.append("places365_video_topk_indices / _scores: разная длина")

    if not meta:
        out.append("meta пустой")
    return out


def validate_structure(npz_path: str) -> List[str]:
    d = load_npz(npz_path)
    m = extract_meta(d)
    if m.get("status") == "error":
        return [
            f"meta.status=error: {m.get('empty_reason')!r} (struct не валидирует payload)"
        ]
    return _validate_data_dict(d, m)


def validate_ranges(npz_path: str) -> List[str]:
    """Диапазоны softmax/cosine/эмбеддингов, times_s (см. docs/FEATURE_DESCRIPTION.md)."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"], dtype=np.int32).size)
    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    if int(ts.size) == n and n > 1 and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд (union)")

    cc = np.asarray(d["consecutive_cosine_prev"], dtype=np.float64).reshape(-1)
    m = np.isfinite(cc)
    if m.any():
        t = cc[m]
        if np.any(t < -1.0 - 1e-3) or np.any(t > 1.0 + 1e-3):
            out.append("consecutive_cosine_prev: вне [-1, 1] (finite)")

    score_keys = (
        "shot_quality_scores",
        "scene_aesthetic_scores",
        "scene_luxury_scores",
        "scene_atmosphere_scores",
        "cut_detection_transition_scores",
        "popularity_topic_scores",
        "places365_topk_scores",
    )
    for k in score_keys:
        a = np.asarray(d[k], dtype=np.float64)
        if a.size and np.isfinite(a).any():
            t = a[np.isfinite(a)]
            if np.any(t < -1e-3) or np.any(t > 1.0 + 0.02):
                out.append(f"{k}: вне [0, 1] (finite, softmax)")

    pv = np.asarray(d["places365_video_topk_scores"], dtype=np.float64).reshape(-1)
    if pv.size and np.isfinite(pv).any():
        t = pv[np.isfinite(pv)]
        if np.any(t < -1e-3) or np.any(t > 1.0 + 0.02):
            out.append("places365_video_topk_scores: вне [0, 1] (finite)")

    fe = np.asarray(d["frame_embeddings"], dtype=np.float64)
    if fe.ndim == 2 and int(fe.shape[0]) > 0 and int(fe.shape[1]) > 0:
        nr = np.linalg.norm(fe, axis=1)
        m2 = np.isfinite(nr)
        if m2.any():
            lo, hi = float(np.min(nr[m2])), float(np.max(nr[m2]))
            if lo < 0.85 or hi > 1.15:
                out.append(f"frame_embeddings: L2-норма строк вне [0.85, 1.15] (min={lo:.4f}, max={hi:.4f})")

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("core_clip/embeddings.npz")):
        n += 1
        d = load_npz(str(npz))
        m = extract_meta(d)
        ok = validate_schema(str(npz))
        stl: List[str] = []
        if not ok:
            stl = ["INVALID schema"]
        elif m.get("status") == "error":
            stl = [f"meta.status=error: {m.get('empty_reason')!r}"]
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


def main() -> int:
    p = argparse.ArgumentParser(
        description="validate core_clip/embeddings.npz (core_clip_npz_v2)"
    )
    p.add_argument(
        "npz_path",
        nargs="?",
        help="Путь к embeddings.npz (если не задан --results-base)",
    )
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument(
        "--struct", action="store_true", help="Ключи, N, первые оси score-таблиц, D=embed."
    )
    p.add_argument(
        "--ranges",
        action="store_true",
        help="times_s, cosine/scores/норма эмбеддингов, N vs total_frames (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; обход **/core_clip/embeddings.npz",
    )
    p.add_argument("--platform-id", type=str, default="youtube")
    args = p.parse_args()

    if args.results_base:
        return _run_batch_rglob(
            results_base=args.results_base, platform_id=args.platform_id or "youtube"
        )

    if not args.npz_path:
        p.error("нужен npz_path или --results-base")
        return 1

    ok = validate_schema(args.npz_path)
    print("✅ VALID schema" if ok else "❌ INVALID schema")
    if not ok:
        return 1
    ex = 0
    d_once: Dict[str, Any] | None = None
    if args.struct or args.ranges:
        d_once = load_npz(args.npz_path)
    if args.struct:
        st = validate_structure(args.npz_path)
        if st:
            print("⚠️  structure:")
            for s in st:
                print("  -", s)
            ex = max(ex, 2)
        else:
            d0 = d_once if d_once is not None else load_npz(args.npz_path)
            n = int(np.asarray(d0["frame_indices"], dtype=np.int32).ravel().size)
            fe = np.asarray(d0["frame_embeddings"], dtype=np.float32)
            d_dim = int(fe.shape[1]) if fe.ndim == 2 else 0
            print(
                f"✅ Structure OK (N={n}, D={d_dim}, core_clip_npz_v2, prompts+score matrices)"
            )
    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (times_s, cosine, softmax [0,1], ‖frame_emb‖≈1, total_frames)")
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
