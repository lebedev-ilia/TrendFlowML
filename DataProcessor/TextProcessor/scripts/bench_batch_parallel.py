#!/usr/bin/env python3
"""
Benchmark для проверки Stage 4: CPU parallelism и GPU batching.
Сравнивает последовательную обработку vs параллельную/batch обработку.
"""

import sys
import time
from pathlib import Path

# Add TextProcessor to path
script_dir = Path(__file__).parent
tp_root = script_dir.parent
sys.path.insert(0, str(tp_root))

from src.core.main_processor import MainProcessor, load_document_from_json
from src.schemas.models import VideoDocument


def create_dummy_doc(title: str, doc_id: str) -> VideoDocument:
    """Create a dummy VideoDocument for testing."""
    doc = VideoDocument(
        title=title,
        description=f"Description for {title}",
        transcripts={"youtube_auto": f"Transcript for {title}. " * 10},
        comments=[{"text": f"Comment {i} for {title}"} for i in range(3)],
    )
    doc.tp_artifacts = {}
    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark batch processing with parallelism")
    parser.add_argument("--n", type=int, default=8, help="Number of documents")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--max-workers", type=int, default=None, help="Max workers for CPU parallelism")
    parser.add_argument("--disable-gpu-batching", action="store_true", help="Disable GPU batching")
    parser.add_argument("--disable-cpu-parallel", action="store_true", help="Disable CPU parallelism")
    args = parser.parse_args()

    num_docs = args.n
    device = args.device
    
    print(f"n={num_docs} device={device} max_workers={args.max_workers}")
    print(f"GPU batching: {not args.disable_gpu_batching}")
    print(f"CPU parallel: {not args.disable_cpu_parallel}")

    # Create test documents
    docs = []
    for i in range(num_docs):
        docs.append(create_dummy_doc(f"Test Title {i}", f"video_{i}"))

    # Configure extractors
    devices_config = {
        "cpu": ["LexicalStatsExtractor", "TagsExtractor"],  # ASRTextProxyExtractor requires audio_duration_sec
    }
    if device == "cuda":
        devices_config["cuda"] = ["TitleEmbedder", "HashtagEmbedder", "TranscriptChunkEmbedder", "CommentsEmbedder"]
    else:
        devices_config["cpu"].extend(["TitleEmbedder", "HashtagEmbedder", "TranscriptChunkEmbedder", "CommentsEmbedder"])

    processor = MainProcessor(
        devices_config=devices_config,
        artifacts_dir=".artifacts_bench_parallel"
    )

    # Test 1: Sequential (baseline)
    print("\n--- Sequential (baseline) ---")
    t0 = time.perf_counter()
    results_seq = []
    for i, doc in enumerate(docs):
        doc.tp_artifacts = {}
        doc_artifacts_dir = Path(processor.artifacts_dir) / f"doc_seq_{i:05d}"
        doc_artifacts_dir.mkdir(parents=True, exist_ok=True)
        setattr(doc, "_tp_artifacts_dir", str(doc_artifacts_dir))
        results_seq.append(processor.run(doc, artifacts_dir_override=str(doc_artifacts_dir)))
    t_seq = time.perf_counter() - t0
    print(f"Sequential: {t_seq:.3f}s  ({t_seq/num_docs:.4f}s/doc)")

    # Test 2: Batch with optimizations
    print("\n--- Batch with optimizations ---")
    for doc in docs:
        doc.tp_artifacts = {}
    t0 = time.perf_counter()
    results_batch = processor.run_batch(
        docs,
        max_workers=args.max_workers,
        enable_gpu_batching=not args.disable_gpu_batching,
        enable_cpu_parallel=not args.disable_cpu_parallel,
    )
    t_batch = time.perf_counter() - t0
    print(f"Batch:     {t_batch:.3f}s  ({t_batch/num_docs:.4f}s/doc)")
    
    if t_seq > 0:
        speedup = t_seq / t_batch
        print(f"Speedup:   {speedup:.2f}x")

    # Verify results equivalence (check status and key features)
    print("\n--- Verification ---")
    all_ok = True
    for i, (r_seq, r_batch) in enumerate(zip(results_seq, results_batch)):
        if r_seq.get("status") != r_batch.get("status"):
            print(f"doc {i}: status mismatch: seq={r_seq.get('status')} batch={r_batch.get('status')}")
            all_ok = False
        
        # Check features_flat count
        ff_seq = r_seq.get("features_flat", {})
        ff_batch = r_batch.get("features_flat", {})
        if len(ff_seq) != len(ff_batch):
            print(f"doc {i}: features count mismatch: seq={len(ff_seq)} batch={len(ff_batch)}")
            all_ok = False

    if all_ok:
        print("OK: Results are equivalent")
        return 0
    else:
        print("FAILED: Results differ")
        return 1


if __name__ == "__main__":
    sys.exit(main())

