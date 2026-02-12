#!/usr/bin/env python3
"""
Smoke test for TranscriptChunkEmbedder batch processing.
Compares extract() loop vs extract_batch() on multiple documents with different transcripts.
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


def create_doc_with_transcript(title: str, transcript_text: str, source: str = "whisper") -> VideoDocument:
    """Create a VideoDocument with transcript."""
    doc_dict = {
        "title": title,
        "description": "",
        "transcripts": {},
        "comments": [],
    }
    doc = VideoDocument(**doc_dict)
    
    # Add transcript via asr.segments (for whisper) or transcripts dict (for youtube_auto)
    if source == "whisper":
        doc.asr = {
            "segments": [
                {"text": part, "confidence": 0.95}
                for part in transcript_text.split(". ") if part.strip()
            ]
        }
    elif source == "youtube_auto":
        doc.transcripts = {"youtube_auto": transcript_text}
    
    doc.tp_artifacts = {}
    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, help="Example JSON to use as template")
    parser.add_argument("--n", type=int, default=4, help="Number of documents")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--artifacts-dir", type=str, default=".artifacts_smoke_transcript", help="Artifacts directory")
    args = parser.parse_args()

    # Create test documents with different transcripts
    docs = []
    transcripts = [
        "This is a test transcript. It contains multiple sentences. Each sentence should be chunked separately.",
        "Another transcript with different content. This one is shorter. But still has multiple sentences.",
        "A longer transcript that will produce more chunks. " * 3 + "This should test batching with variable chunk counts.",
        "Short one.",
    ]
    for i in range(args.n):
        transcript = transcripts[i % len(transcripts)]
        doc = create_doc_with_transcript(f"Test video {i+1}", transcript, source="whisper")
        docs.append(doc)

    print(f"n={args.n} device={args.device}")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create processor
    processor = MainProcessor(
        devices_config={"gpu": ["TranscriptChunkEmbedder"]} if args.device == "cuda" else {"cpu": ["TranscriptChunkEmbedder"]},
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
        keys = ["tp_tchunk_present", "tp_tchunk_sources_count", "tp_tchunk_whisper_chunks", "tp_tchunk_embedding_dim"]
        for k in keys:
            v_loop = f_loop.get(k)
            v_batch = f_batch.get(k)
            if v_loop != v_batch:
                print(f"doc {i}: {k} mismatch: loop={v_loop} batch={v_batch}")
                ok = False

        # Compare embeddings if present
        if f_loop.get("tp_tchunk_present", 0) > 0.5:
            # Check artifacts exist
            loop_art = artifacts_dir / f"doc_loop_{i:05d}" / "transcript_whisper_chunk_embeddings.npy"
            batch_art = artifacts_dir / f"doc_{i:05d}" / "transcript_whisper_chunk_embeddings.npy"
            if loop_art.exists() and batch_art.exists():
                emb_loop = np.load(loop_art)
                emb_batch = np.load(batch_art)
                if emb_loop.shape != emb_batch.shape:
                    print(f"doc {i}: embedding shape mismatch: loop={emb_loop.shape} batch={emb_batch.shape}")
                    ok = False
                elif not np.allclose(emb_loop, emb_batch, rtol=1e-5):
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

