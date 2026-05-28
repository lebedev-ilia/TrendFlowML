#!/usr/bin/env python3
"""
Per-extractor smoke for Audit v3: VideoDocument from `audit_v3_20_scenarios.json` (inference), minimal dependency chain, CPU.

Usage (from TextProcessor root):
  DP_MODELS_ROOT=.../dp_models/bundled_models \\
    ./.tp_venv/bin/python scripts/smoke_each_extractor_audit_v3.py

All 20 scenarios × 22 extractors (440 runs):
  ./.tp_venv/bin/python scripts/smoke_each_extractor_audit_v3.py --all-scenarios

Subset (debug):
  ./.tp_venv/bin/python scripts/smoke_each_extractor_audit_v3.py --all-scenarios --limit-scenarios 2

Single scenario (legacy):
  ./.tp_venv/bin/python scripts/smoke_each_extractor_audit_v3.py --scenario-index 0

Requires:
  - intfloat/multilingual-e5-large under bundled_models/text/embeddings/intfloat_multilingual-e5-large
  - Temp union DP_MODELS_ROOT adds similar_titles embeddings + semantic_cluster PCA fixtures.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Repo layout: TextProcessor/scripts/this_file → TP_ROOT, DP_ROOT
TP_ROOT = Path(__file__).resolve().parents[1]
DP_ROOT = TP_ROOT.parent
REPO_ROOT = DP_ROOT.parent
SCENARIOS_JSON = REPO_ROOT / "example" / "text_audit_v3_smoke" / "scenarios" / "audit_v3_20_scenarios.json"


# Mirrors src/core/main_processor.py DEPENDENCIES (transitive closure for "needs").
DEPS: Dict[str, List[str]] = {
    "TagsExtractor": [],
    "LexicalStatsExtractor": ["TagsExtractor"],
    "ASRTextProxyExtractor": ["TagsExtractor"],
    "TitleEmbedder": ["TagsExtractor"],
    "DescriptionEmbedder": ["TagsExtractor"],
    "HashtagEmbedder": ["TagsExtractor"],
    "TranscriptChunkEmbedder": [],
    "CommentsEmbedder": [],
    "SpeakerTurnEmbeddingsAggregatorExtractor": [],
    "TranscriptAggregatorExtractor": ["TranscriptChunkEmbedder"],
    "CommentsAggregationExtractor": ["CommentsEmbedder"],
    "QAEmbeddingPairsExtractor": ["TranscriptChunkEmbedder"],
    "EmbeddingPairTopKExtractor": ["TitleEmbedder", "DescriptionEmbedder"],
    "SemanticTopicExtractor": ["TranscriptChunkEmbedder"],
    "EmbeddingStatsExtractor": ["TranscriptChunkEmbedder"],
    "CosineMetricsExtractor": [
        "TitleEmbedder",
        "DescriptionEmbedder",
        "TranscriptAggregatorExtractor",
        "CommentsEmbedder",
    ],
    "TitleEmbeddingClusterEntropyExtractor": ["TitleEmbedder"],
    "TitleToHashtagCosineExtractor": ["TitleEmbedder", "HashtagEmbedder"],
    "SemanticClusterExtractor": ["TitleEmbedder", "DescriptionEmbedder", "HashtagEmbedder"],
    "TopKSimilarCorpusTitlesExtractor": ["TitleEmbedder"],
    "EmbeddingShiftIndicatorExtractor": ["TranscriptChunkEmbedder"],
    "EmbeddingSourceIdExtractor": ["TitleEmbedder", "DescriptionEmbedder", "TranscriptAggregatorExtractor"],
}


FULL_ORDER: List[str] = [
    "TagsExtractor",
    "LexicalStatsExtractor",
    "ASRTextProxyExtractor",
    "TitleEmbedder",
    "DescriptionEmbedder",
    "HashtagEmbedder",
    "TranscriptChunkEmbedder",
    "CommentsEmbedder",
    "SpeakerTurnEmbeddingsAggregatorExtractor",
    "TranscriptAggregatorExtractor",
    "CommentsAggregationExtractor",
    "QAEmbeddingPairsExtractor",
    "EmbeddingPairTopKExtractor",
    "SemanticTopicExtractor",
    "EmbeddingStatsExtractor",
    "CosineMetricsExtractor",
    "TitleEmbeddingClusterEntropyExtractor",
    "TitleToHashtagCosineExtractor",
    "SemanticClusterExtractor",
    "TopKSimilarCorpusTitlesExtractor",
    "EmbeddingShiftIndicatorExtractor",
    "EmbeddingSourceIdExtractor",
]


def _closure(target: str) -> Set[str]:
    out: Set[str] = set()
    stack = [target]
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        for d in DEPS.get(n, []):
            stack.append(d)
    return out


def ordered_chain(target: str) -> List[str]:
    need = _closure(target)
    return [n for n in FULL_ORDER if n in need]


def _l2n_rows(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def build_union_dp_models_root(bundled: Path) -> Path:
    """
    Symlink everything under bundled_models into a temp root, except patch
    similar_titles_v1 (add embeddings.npy) and semantic_clusters_v1 (add pca/centroids, trim jsonl).
    """
    root = Path(tempfile.mkdtemp(prefix="tp_smoke_dp_models_"))
    # Shared caches & packs (offline HF, etc.) — same layout as bundled_models.
    for name in ("hf_cache", "cache", "torch_cache", "semantics", "visual", "audio", "clip_cache"):
        p = bundled / name
        if p.exists():
            link = root / name
            if not link.exists():
                link.symlink_to(p.resolve(), target_is_directory=True)

    src_text = bundled / "text"
    dst_text = root / "text"
    dst_text.mkdir(parents=True, exist_ok=True)

    # Copy/symlink sibling packages under text/
    if src_text.is_dir():
        for child in src_text.iterdir():
            name = child.name
            if name in ("similar_titles_v1", "semantic_clusters_v1"):
                continue
            link = dst_text / name
            if not link.exists():
                link.symlink_to(child.resolve(), target_is_directory=True)

    # --- similar_titles_v1: small randomized corpus, dim 1024 ---
    st_src = src_text / "similar_titles_v1"
    st = dst_text / "similar_titles_v1"
    st.mkdir(parents=True, exist_ok=True)
    n_corpus, dim = 64, 1024
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    emb = _l2n_rows(emb)
    np.save(st / "embeddings.npy", emb)
    ids_path = st / "ids.json"
    if st_src.is_dir() and (st_src / "ids.json").is_file():
        shutil.copy2(st_src / "ids.json", ids_path)
        data = json.loads(ids_path.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) >= n_corpus:
            ids_path.write_text(json.dumps(data[:n_corpus], ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            ids_path.write_text(
                json.dumps([f"smoke_{i:06d}" for i in range(n_corpus)], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        ids_path.write_text(
            json.dumps([f"smoke_{i:06d}" for i in range(n_corpus)], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # --- semantic_clusters_v1: tiny PCA + centroids ---
    sc_src = src_text / "semantic_clusters_v1"
    sc = dst_text / "semantic_clusters_v1"
    sc.mkdir(parents=True, exist_ok=True)
    n_clusters, red = 3, 8
    pca = rng.standard_normal((dim, red)).astype(np.float32)
    cent = rng.standard_normal((n_clusters, red)).astype(np.float32)
    cent = _l2n_rows(cent)
    np.save(sc / "pca.npy", pca)
    np.save(sc / "centroids.npy", cent)
    jsonl_lines = []
    if sc_src.is_dir() and (sc_src / "clusters.jsonl").is_file():
        raw = (sc_src / "clusters.jsonl").read_text(encoding="utf-8").splitlines()
        for line in raw[:n_clusters]:
            if line.strip():
                jsonl_lines.append(line.strip())
    while len(jsonl_lines) < n_clusters:
        i = len(jsonl_lines)
        jsonl_lines.append(json.dumps({"cluster_id": i, "name": f"cluster_{i:03d}", "group": "smoke"}, ensure_ascii=False))
    (sc / "clusters.jsonl").write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")

    return root


def load_scenarios_bundle(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios") or []
    if not scenarios:
        raise ValueError("no scenarios in bundle")
    return scenarios, len(scenarios)


def load_scenario_video_document(path: Path, scenario_index: int) -> Tuple[Dict[str, Any], str]:
    scenarios, n = load_scenarios_bundle(path)
    if scenario_index < 0 or scenario_index >= n:
        raise IndexError(f"No scenario at index {scenario_index} (len={n})")
    scen = scenarios[scenario_index]
    sid = str(scen.get("id", f"idx_{scenario_index}"))
    inf = scen.get("inference") or {}
    vd = inf.get("video_document")
    if not isinstance(vd, dict):
        raise ValueError(f"scenario {sid} missing inference.video_document")
    return vd, sid


def default_cpu_extractor_params() -> Dict[str, Dict[str, Any]]:
    """Force CPU for embedding extractors (device group is 'cpu' anyway; some still read kwargs)."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in (
        "TitleEmbedder",
        "DescriptionEmbedder",
        "HashtagEmbedder",
        "TranscriptChunkEmbedder",
        "CommentsEmbedder",
        "SpeakerTurnEmbeddingsAggregatorExtractor",
        "QAEmbeddingPairsExtractor",
        "EmbeddingPairTopKExtractor",
        "SemanticTopicExtractor",
    ):
        out[name] = {"device": "cpu", "fp16": False}
    return out


def validate_payload(target: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
    if payload.get("status") == "error":
        e = payload.get("error") or payload.get("errors_by_extractor") or {}
        return False, f"orchestrator status=error: {e}"
    rbe = payload.get("results_by_extractor") or {}
    if target not in rbe:
        return False, f"missing results_by_extractor[{target}]"
    block = rbe[target]
    if not isinstance(block, dict):
        return False, "result block not a dict"
    ff = block.get("features_flat")
    if not isinstance(ff, dict) or not ff:
        return False, "empty or missing features_flat"
    err_map = payload.get("errors_by_extractor") or {}
    if isinstance(err_map, dict) and err_map.get(target):
        return False, f"errors_by_extractor[{target}]={err_map[target]}"
    return True, "ok"


def ensure_import_paths() -> None:
    if str(TP_ROOT) not in sys.path:
        sys.path.insert(0, str(TP_ROOT))
    if str(DP_ROOT) not in sys.path:
        sys.path.insert(0, str(DP_ROOT))
    vp_root = REPO_ROOT / "VisualProcessor"
    if vp_root.is_dir() and str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))


def run_one(
    *,
    target: str,
    doc_path: Path,
    artifacts_dir: Path,
    extractor_params: Dict[str, Dict[str, Any]],
) -> Tuple[bool, str, float]:
    from src.core.main_processor import MainProcessor, load_document_from_json  # type: ignore

    chain = ordered_chain(target)
    devices_config = {"cpu": chain}
    doc = load_document_from_json(str(doc_path))
    t0 = __import__("time").perf_counter()
    proc = MainProcessor(
        devices_config=devices_config,
        extractor_params=extractor_params,
        strict=True,
        artifacts_dir=str(artifacts_dir),
    )
    payload = proc.run(doc) or {}
    elapsed = __import__("time").perf_counter() - t0
    ok, msg = validate_payload(target, payload)
    return ok, msg, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Smoke TextProcessor extractors on Audit v3 scenarios (CPU, isolated dependency chains)."
    )
    ap.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run every scenario in audit_v3_20_scenarios.json for each extractor (overrides --scenario-index)",
    )
    ap.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Single scenario index when --all-scenarios is not set (default: 0)",
    )
    ap.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        metavar="N",
        help="With --all-scenarios: only first N scenarios (indices 0..N-1)",
    )
    ap.add_argument(
        "--dp-models-root",
        type=str,
        default=os.environ.get("DP_MODELS_ROOT", str(DP_ROOT / "dp_models" / "bundled_models")),
        help="Base dp_models bundle (e5-large must exist here)",
    )
    ap.add_argument("--keep-union-root", action="store_true", help="Print union root path and do not delete it")
    ap.add_argument("--quiet", action="store_true", help="Only print failures and final summary")
    args = ap.parse_args()

    bundled = Path(args.dp_models_root).resolve()
    e5 = bundled / "text" / "embeddings" / "intfloat_multilingual-e5-large"
    if not e5.is_dir():
        print(f"FAIL: missing SentenceTransformer bundle: {e5}", file=sys.stderr)
        return 2

    if not SCENARIOS_JSON.is_file():
        print(f"FAIL: scenarios not found: {SCENARIOS_JSON}", file=sys.stderr)
        return 2

    _, n_scen = load_scenarios_bundle(SCENARIOS_JSON)
    if args.all_scenarios:
        n_run = n_scen
        if args.limit_scenarios is not None:
            n_run = max(0, min(n_scen, int(args.limit_scenarios)))
        scenario_indices = list(range(n_run))
    else:
        scenario_indices = [int(args.scenario_index)]

    union_root = build_union_dp_models_root(bundled)
    os.environ["DP_MODELS_ROOT"] = str(union_root)
    if args.keep_union_root:
        print(f"# union DP_MODELS_ROOT={union_root}", file=sys.stderr)

    ensure_import_paths()
    work = Path(tempfile.mkdtemp(prefix="tp_smoke_doc_"))
    doc_paths: List[Tuple[int, str, Path]] = []
    for si in scenario_indices:
        vd, sid = load_scenario_video_document(SCENARIOS_JSON, si)
        p = work / f"scenario_{si:02d}_{sid}.json"
        p.write_text(json.dumps(vd, ensure_ascii=False, indent=2), encoding="utf-8")
        doc_paths.append((si, sid, p))

    params = default_cpu_extractor_params()
    failures: List[Tuple[str, int, str, str]] = []
    total_elapsed = 0.0
    total_ok = 0
    total_runs = len(FULL_ORDER) * len(doc_paths)
    run_i = 0

    for target in FULL_ORDER:
        for si, sid, doc_path in doc_paths:
            run_i += 1
            art = work / "artifacts" / target / f"scen_{si:02d}"
            art.mkdir(parents=True, exist_ok=True)
            try:
                ok, msg, elapsed = run_one(
                    target=target,
                    doc_path=doc_path,
                    artifacts_dir=art,
                    extractor_params=params,
                )
                total_elapsed += elapsed
                if ok:
                    total_ok += 1
                    if not args.quiet:
                        print(f"OK [{run_i:4d}/{total_runs}] {target:48s} {sid:22s} {elapsed:7.2f}s")
                else:
                    print(f"FAIL [{run_i:4d}/{total_runs}] {target:48s} {sid:22s} {msg}", file=sys.stderr)
                    failures.append((target, si, sid, msg))
            except Exception as e:
                print(f"FAIL [{run_i:4d}/{total_runs}] {target:48s} {sid:22s} {e}", file=sys.stderr)
                failures.append((target, si, sid, str(e)))

    try:
        shutil.rmtree(work, ignore_errors=True)
        if not args.keep_union_root:
            shutil.rmtree(union_root, ignore_errors=True)
    except Exception:
        pass

    print(
        f"\nSummary: {total_ok}/{total_runs} runs OK, "
        f"{len(failures)} failed, {total_elapsed:.1f}s total wall-time in extractors"
    )
    if failures:
        print(f"\nFailed {len(failures)}:", file=sys.stderr)
        for name, si, sid, msg in failures:
            print(f"  - {name} scenario_index={si} id={sid!r}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
