#!/usr/bin/env python3
"""
Component-scoped валидатор для action_recognition_features.npz (schema v3).

Design R5: §0.2 (e2e_validate_output_quality) требует full-run manifest и не работает
пер-компонентно. Этот валидатор проверяет контракт v3 на одном npz — быстро, локально.

CLI:
  python validate_action_recognition_npz.py <path/to/action_recognition_features.npz>
Возвращает 0 если ok/valid-empty, иначе 1 и печатает список проблем.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import numpy as np

EMB_DIM_MIN = 32          # v3.1: эмбеддинг dim-гибкий (penultimate features), проверяем только >0/finite/L2
NUM_CLASSES = 400


def _load(npz_path: str) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _meta(d: Dict[str, Any]) -> Dict[str, Any]:
    mj = d.get("meta_json")
    if mj is not None:
        s = mj.item() if getattr(mj, "shape", None) == () else np.asarray(mj).reshape(-1)[0]
        try:
            return json.loads(str(s))
        except Exception:
            pass
    m = d.get("meta")
    if m is not None:
        try:
            return m.item() if getattr(m, "shape", None) == () else dict(np.asarray(m).tolist())
        except Exception:
            return {}
    return {}


def validate(npz_path: str) -> List[str]:
    out: List[str] = []
    d = _load(npz_path)
    meta = _meta(d)

    required = [
        "clip_embeddings", "clip_track_id", "clip_frame_indices", "clip_times_s",
        "clip_topk_action_ids", "clip_topk_probs", "clip_count", "class_names",
        "video_action_hist", "num_tracks",
    ]
    for k in required:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    C = int(np.asarray(d["clip_count"]).reshape(-1)[0])
    emb = np.asarray(d["clip_embeddings"], dtype=np.float32)
    tid = np.asarray(d["clip_track_id"]).reshape(-1)
    cfi = np.asarray(d["clip_frame_indices"]).reshape(-1)
    ts = np.asarray(d["clip_times_s"], dtype=np.float32).reshape(-1)

    status = str(meta.get("status", "") or "")
    # valid empty
    if C == 0:
        if emb.shape[0] != 0:
            out.append(f"clip_count=0, но clip_embeddings не пустой ({emb.shape})")
        if status not in ("empty", "ok", ""):
            out.append(f"пустой поток, meta.status={status!r}")
        return out

    # shapes (dim-гибкий: penultimate features могут быть 256/768/2304/…)
    if emb.ndim != 2 or emb.shape[0] != C or emb.shape[1] < EMB_DIM_MIN:
        out.append(f"clip_embeddings: ожидается (C={C}, D≥{EMB_DIM_MIN}), факт {emb.shape}")
    for name, a in [("clip_track_id", tid), ("clip_frame_indices", cfi), ("clip_times_s", ts)]:
        if a.shape[0] != C:
            out.append(f"{name}: len={a.shape[0]} != C={C}")

    # embeddings finite + L2
    if not np.all(np.isfinite(emb)):
        out.append("clip_embeddings: есть не-finite значения")
    else:
        norms = np.linalg.norm(emb, axis=1)
        nonzero = norms > 1e-6
        if np.any(nonzero) and not np.allclose(norms[nonzero], 1.0, atol=1e-2):
            out.append(f"clip_embeddings: не L2-нормированы (min={norms.min():.3f}, max={norms.max():.3f})")

    # время монотонно возрастает (клипы отсортированы)
    if ts.shape[0] == C and np.any(np.diff(ts) < -1e-3):
        out.append("clip_times_s: не отсортированы по возрастанию")

    # topk probs в [0,1] и по убыванию
    tp = np.asarray(d["clip_topk_probs"], dtype=np.float32)
    ti = np.asarray(d["clip_topk_action_ids"])
    if tp.shape[0] != C or ti.shape[0] != C:
        out.append("clip_topk_*: первая ось != C")
    classes_avail = bool(np.asarray(d.get("classes_available", np.bool_(False))).reshape(-1)[0]) if "classes_available" in d else True
    if classes_avail and tp.size:
        if np.any(tp < -1e-6) or np.any(tp > 1.0 + 1e-3):
            out.append(f"clip_topk_probs: вне [0,1] (min={tp.min():.3f}, max={tp.max():.3f})")
        if tp.shape[1] > 1 and np.any(np.diff(tp, axis=1) > 1e-3):
            out.append("clip_topk_probs: не по убыванию внутри клипа")
        if np.any((ti < 0) | (ti >= NUM_CLASSES)):
            out.append("clip_topk_action_ids: вне [0,400)")

    # class_names
    cn = np.asarray(d["class_names"], dtype=object)
    if cn.size != NUM_CLASSES:
        out.append(f"class_names: ожидается {NUM_CLASSES}, факт {cn.size}")

    # union-consistency (если meta несёт union длину — опц.)
    ut = meta.get("union_len") or meta.get("total_frames")
    if ut and cfi.size and (np.any(cfi < 0) or np.any(cfi >= int(ut))):
        out.append(f"clip_frame_indices: вне [0,{ut})")

    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: validate_action_recognition_npz.py <features.npz>", file=sys.stderr)
        return 2
    problems = validate(sys.argv[1])
    if problems:
        print("❌ action_recognition v3 контракт НАРУШЕН:")
        for p in problems:
            print("  -", p)
        return 1
    print("✅ action_recognition_features.npz соответствует schema v3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
