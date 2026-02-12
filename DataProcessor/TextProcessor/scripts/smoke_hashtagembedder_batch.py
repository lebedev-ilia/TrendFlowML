#!/usr/bin/env python3
"""
Smoke test for HashtagEmbedder batch processing.
Compares extract() loop vs extract_batch() on multiple documents with different hashtags.
"""

import sys
import os
from pathlib import Path

# Add TextProcessor to path
script_dir = Path(__file__).parent
tp_root = script_dir.parent
sys.path.insert(0, str(tp_root))

import json
import numpy as np
from src.core.main_processor import MainProcessor, load_document_from_json
from src.schemas.models import VideoDocument


def create_doc_with_hashtags(title: str, hashtags: list) -> VideoDocument:
    """Create a VideoDocument with hashtags."""
    doc_dict = {
        "title": title,
        "description": "",
        "transcripts": {},
        "comments": [],
    }
    doc = VideoDocument(**doc_dict)
    doc.hashtags = hashtags
    doc.tp_artifacts = {}
    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, help="Example JSON to use as template")
    parser.add_argument("--n", type=int, default=4, help="Number of documents")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--artifacts-dir", type=str, default=".artifacts_smoke_hashtag", help="Artifacts directory")
    args = parser.parse_args()

    # Create test documents with different hashtags
    docs = []
    hashtags_sets = [
        ["python", "programming", "coding"],
        ["ai", "machinelearning", "deeplearning"],
        ["tech", "innovation", "startup"],
        ["data", "analytics", "science"],
    ]
    for i in range(args.n):
        tags = hashtags_sets[i % len(hashtags_sets)]
        doc = create_doc_with_hashtags(f"Test video {i+1}", tags)
        docs.append(doc)

    print(f"n={args.n} device={args.device}")

    # Ensure TagsExtractor ran (or simulate it)
    # For smoke test, we'll just set hashtags directly
    for doc in docs:
        if not hasattr(doc, "hashtags") or not doc.hashtags:
            doc.hashtags = []

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create processor
    processor = MainProcessor(
        devices_config={"gpu": ["HashtagEmbedder"]} if args.device == "cuda" else {"cpu": ["HashtagEmbedder"]},
        artifacts_dir=str(artifacts_dir),
    )

    # Test 1: Loop (single extract)
    import time
    t0 = time.perf_counter()
    results_loop = []
    for i, doc in enumerate(docs):
        doc_artifacts = artifacts_dir / f"doc_loop_{i:05d}"
        doc_artifacts.mkdir(parents=True, exist_ok=True)
        doc._tp_artifacts_dir = str(doc_artifacts)
        res = processor.run(doc, artifacts_dir_override=str(doc_artifacts))
        results_loop.append(res)
    t_loop = time.perf_counter() - t0

    # Test 2: Batch
    for doc in docs:
        doc.tp_artifacts = {}  # Reset
    t0 = time.perf_counter()
    results_batch = processor.run_batch(docs)
    t_batch = time.perf_counter() - t0

    # Compare results
    print(f"loop : {t_loop:.3f}s  ({t_loop/args.n:.4f}s/doc)")
    print(f"batch: {t_batch:.3f}s  ({t_batch/args.n:.4f}s/doc)")

    # Check equivalence: compare features_flat for each document
    ok = True
    for i, (r_loop, r_batch) in enumerate(zip(results_loop, results_batch)):
        f_loop = r_loop.get("result", {}).get("features_flat", {})
        f_batch = r_batch.get("result", {}).get("features_flat", {})
        
        # Key fields to compare
        keys = ["tp_hashemb_present", "tp_hashemb_dim", "tp_hashemb_tag_count", "tp_hashemb_n_unique_tags"]
        for k in keys:
            v_loop = f_loop.get(k)
            v_batch = f_batch.get(k)
            if v_loop != v_batch:
                print(f"doc {i}: {k} mismatch: loop={v_loop} batch={v_batch}")
                ok = False

        # Compare embeddings if present
        if f_loop.get("tp_hashemb_present", 0) > 0.5:
            # Check artifacts exist
            loop_art = artifacts_dir / f"doc_loop_{i:05d}" / "hashtag_embedding.npy"
            batch_art = artifacts_dir / f"doc_{i:05d}" / "hashtag_embedding.npy"
            if loop_art.exists() and batch_art.exists():
                emb_loop = np.load(loop_art)
                emb_batch = np.load(batch_art)
                if not np.allclose(emb_loop, emb_batch, rtol=1e-5):
                    print(f"doc {i}: embedding mismatch (max diff: {np.abs(emb_loop - emb_batch).max()})")
                    ok = False

    if ok:
        print("OK")
        return 0
    else:
        print("FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

