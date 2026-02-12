#!/usr/bin/env python3
"""
Micro-benchmark / smoke for TitleEmbedder.extract_batch().

Compares:
- loop: for each doc -> embedder.extract(doc)
- batch: embedder.extract_batch(docs)

This script is intentionally lightweight and does not attempt a deep equality check
of all timings; it validates that embeddings are written and tp_artifacts are set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _load_doc_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--artifacts-dir", type=str, default=".artifacts_bench")
    args = parser.parse_args()

    # Allow running as a standalone script (so `import src.*` works).
    tp_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(tp_root))

    from src.schemas.models import video_document_from_dict
    from src.extractors.title_embedder.main import TitleEmbedder

    base = Path(args.artifacts_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    d0 = _load_doc_dict(args.input_json)

    # Build docs
    docs = [video_document_from_dict(d0) for _ in range(max(1, int(args.n)))]
    for i, d in enumerate(docs):
        # per-doc artifacts dir (Stage-1 layout)
        per = base / f"doc_{i:05d}"
        per.mkdir(parents=True, exist_ok=True)
        setattr(d, "_tp_artifacts_dir", str(per))
        setattr(d, "tp_artifacts", {})

    embedder = TitleEmbedder(device=args.device, artifacts_dir=str(base / "doc_00000"))

    # loop
    t0 = time.perf_counter()
    loop_res = [embedder.extract(d) for d in docs]
    t_loop = time.perf_counter() - t0

    # reset artifacts registry (do not delete files)
    for d in docs:
        setattr(d, "tp_artifacts", {})

    # batch
    t1 = time.perf_counter()
    batch_res = embedder.extract_batch(docs)
    t_batch = time.perf_counter() - t1

    # Validate artifacts existence for first/last doc
    def _check(i: int) -> bool:
        d = docs[i]
        per = Path(getattr(d, "_tp_artifacts_dir"))
        p = per / "title_embedding.npy"
        ok_file = p.exists()
        tp = getattr(d, "tp_artifacts", {})
        ok_tp = isinstance(tp, dict) and isinstance(tp.get("embeddings"), dict) and isinstance(tp["embeddings"].get("title"), dict)
        return bool(ok_file and ok_tp)

    ok = _check(0) and _check(len(docs) - 1) and len(loop_res) == len(batch_res) == len(docs)

    print(f"n={len(docs)} device={args.device}")
    print(f"loop  : {t_loop:.3f}s  ({(t_loop/len(docs)):.4f}s/doc)")
    print(f"batch : {t_batch:.3f}s  ({(t_batch/len(docs)):.4f}s/doc)")
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


