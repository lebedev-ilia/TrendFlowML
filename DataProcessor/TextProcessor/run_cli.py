#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import uuid
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _timestamp_now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def _sha256_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _atomic_save_npz(path: str, **arrays: Any) -> None:
    target_dir = os.path.dirname(path)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=Path(path).name + ".", suffix=".npz", dir=target_dir)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _meta(*, producer_version: str, status: str, schema_version: str, extra: Dict[str, Any]) -> np.ndarray:
    d = {
        "producer": "text_processor",
        "producer_version": producer_version,
        "schema_version": schema_version,
        "status": status,
        "created_at": _utc_iso_now(),
        **(extra or {}),
    }
    # PR-3: model system baseline
    try:
        from src.utils.meta_builder import apply_models_meta  # type: ignore

        d = apply_models_meta(d, models_used=d.get("models_used"))
    except Exception:
        d.setdefault("models_used", [])
        d.setdefault("model_signature", "")
    return np.asarray(d, dtype=object)


def _flatten_scalars(d: Any, prefix: str = "") -> Dict[str, float]:
    """
    Extract numeric scalars into a flat dict. Non-numeric values are ignored.
    """
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_scalars(v, key))
        return out
    if isinstance(d, list):
        return out
    if d is None:
        return out
    if isinstance(d, bool):
        out[prefix] = 1.0 if d else 0.0
        return out
    if isinstance(d, (int, float, np.integer, np.floating)):
        try:
            out[prefix] = float(d)
        except Exception:
            pass
        return out
    return out


def _safe_text(s: Any, limit: int = 20000) -> str:
    """
    Best-effort string normalization for hashing/summaries.
    Never returns raw content into artifacts unless explicitly allowed by flags.
    """
    try:
        txt = str(s or "")
    except Exception:
        txt = ""
    txt = " ".join(txt.split())
    if len(txt) > limit:
        txt = txt[:limit]
    return txt


def _content_hash_for_document(doc: Any) -> str:
    """
    Privacy-safe stable hash for document content (title/description/transcripts/comments).
    Raw text is NOT stored; only the hash.
    """
    try:
        title = _safe_text(getattr(doc, "title", ""))
        desc = _safe_text(getattr(doc, "description", ""))
        transcripts = getattr(doc, "transcripts", {}) or {}
        transcript_join = ""
        if isinstance(transcripts, dict):
            transcript_join = " ".join(_safe_text(v) for v in transcripts.values() if v)
        comments = getattr(doc, "comments", []) or []
        c_join = ""
        if isinstance(comments, list):
            parts = []
            for c in comments[:2000]:
                if isinstance(c, dict):
                    parts.append(_safe_text(c.get("text")))
                else:
                    parts.append(_safe_text(getattr(c, "text", c)))
            c_join = " ".join([p for p in parts if p])
        payload = "\n".join([title, desc, transcript_join, c_join]).strip()
        return _sha256_text(payload)
    except Exception:
        return ""


def _process_batch(
    input_paths: List[str],
    devices_config: Dict[str, List[str]],
    extractor_params: Dict[str, Dict[str, Any]],
    strict: bool,
    base_artifacts_dir: str,
    logger: logging.Logger,
    batch_max_workers: Optional[int],
    batch_enable_gpu_batching: bool,
    batch_enable_cpu_parallel: bool,
    rs_base: str,
    run_rs_path: Optional[str],
    platform_id: str,
    run_id: str,
    sampling_policy_version: str,
    config_hash: Optional[str],
    dataprocessor_version: str,
    enable_embeddings: bool,
    include_primary_embedding: bool,
    store_raw_payload: bool,
) -> int:
    """
    Обрабатывает несколько документов в batch режиме.
    Для каждого документа создается отдельная директория result_store.
    """
    # Импорты уже выполнены в main(), но повторяем для ясности
    from src.core.main_processor import MainProcessor, load_document_from_json
    from utils.artifact_validator import validate_npz
    
    logger.info(f"Batch mode: processing {len(input_paths)} documents")
    
    # Загружаем все документы
    documents = []
    doc_names = []
    for input_path in input_paths:
        try:
            doc = load_document_from_json(input_path)
            documents.append(doc)
            doc_name = Path(input_path).stem  # имя без расширения
            doc_names.append(doc_name)
            logger.info(f"  ✓ Loaded: {Path(input_path).name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {input_path}: {e}")
            return 1
    
    if not documents:
        logger.error("No documents loaded")
        return 1
    
    # Создаем processor
    processor = MainProcessor(
        devices_config=devices_config,
        extractor_params=extractor_params,
        strict=strict,
        artifacts_dir=base_artifacts_dir,
        logger=logger,
        batch_max_workers=batch_max_workers,
        batch_enable_gpu_batching=batch_enable_gpu_batching,
        batch_enable_cpu_parallel=batch_enable_cpu_parallel,
    )
    
    # Обрабатываем batch
    logger.info("Processing documents in batch mode...")
    t0 = time.time()
    try:
        results = processor.run_batch(documents)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        return 1
    
    duration = time.time() - t0
    logger.info(f"Batch processing completed in {duration:.2f}s ({duration/len(documents):.2f}s/doc)")
    
    # Сохраняем результаты для каждого документа
    success_count = 0
    for i, (doc_name, doc, result) in enumerate(zip(doc_names, documents, results)):
        try:
            # Создаем отдельную директорию для каждого документа
            if run_rs_path:
                # Используем базовый путь и добавляем doc_name
                doc_rs_path = os.path.join(os.path.dirname(run_rs_path), doc_name, run_id)
            else:
                doc_rs_path = os.path.join(rs_base, platform_id, doc_name, run_id)
            
            os.makedirs(doc_rs_path, exist_ok=True)
            comp_dir = os.path.join(doc_rs_path, "text_processor")
            os.makedirs(comp_dir, exist_ok=True)
            
            # Сохраняем NPZ (используем ту же логику, что и для single document)
            status = result.get("status", "ok")
            payload = result
            
            # Flatten scalars
            base_for_scalars = payload.get("features_flat") if isinstance(payload.get("features_flat"), dict) else payload
            scalars = _flatten_scalars(base_for_scalars)
            feature_names = np.asarray(sorted(scalars.keys()), dtype=object)
            feature_values = np.asarray([float(scalars[k]) for k in feature_names.tolist()], dtype=np.float32)
            
            out_path = os.path.join(comp_dir, "text_features.npz")
            payload_summary = _payload_summary(doc=doc, payload=payload)
            
            # Extract empty_reason and error from result
            empty_reason = result.get("empty_reason")
            err = result.get("error")
            if status == "error" and not err:
                errors_by_extractor = result.get("errors_by_extractor", {})
                if errors_by_extractor:
                    err = "; ".join(f"{k}: {v}" for k, v in list(errors_by_extractor.items())[:3])
                else:
                    err = result.get("error") or "unknown_error"
            
            # Extract models_used from payload
            models_used = payload.get("models_used", []) if isinstance(payload, dict) else []
            
            # Primary embedding (если включено)
            primary_embedding = None
            primary_embedding_present = False
            primary_embedding_source = None
            primary_embedding_model = None
            if include_primary_embedding and enable_embeddings and isinstance(payload, dict):
                try:
                    rbe = payload.get("results_by_extractor") if isinstance(payload.get("results_by_extractor"), dict) else {}
                    src = rbe.get("EmbeddingSourceIdExtractor") if isinstance(rbe.get("EmbeddingSourceIdExtractor"), dict) else None
                    emb_info = None
                    if isinstance(src, dict):
                        emb_info = src.get("embedding_source_id")
                    if isinstance(emb_info, dict):
                        relpath = emb_info.get("embedding_relpath")
                        if relpath:
                            artifacts_dir = base_artifacts_dir
                            full_path = os.path.join(artifacts_dir, relpath)
                            if os.path.exists(full_path):
                                primary_embedding = np.load(full_path)
                                primary_embedding_present = True
                                primary_embedding_source = os.path.basename(relpath)
                                primary_embedding_model = emb_info.get("model_version", "unknown")
                except Exception:
                    pass
            
            # Build NPZ
            npz_data = {
                "feature_names": feature_names,
                "feature_values": feature_values,
                "payload": np.asarray(payload_summary, dtype=object),
                "meta": _meta(
                    producer_version="1.0.0",
                    status=status,
                    schema_version="text_npz_v1",
                    extra={
                        "platform_id": platform_id,
                        "video_id": doc_name,
                        "run_id": run_id,
                        "sampling_policy_version": sampling_policy_version,
                        "config_hash": config_hash or "",
                        "dataprocessor_version": dataprocessor_version,
                        "empty_reason": empty_reason,
                        "error": err,
                        "models_used": models_used,
                    },
                ),
            }
            
            if primary_embedding_present and primary_embedding is not None:
                npz_data["primary_embedding"] = primary_embedding
                npz_data["primary_embedding_present"] = np.asarray(True, dtype=bool)
                npz_data["primary_embedding_source"] = np.asarray(primary_embedding_source or "", dtype=object)
                npz_data["primary_embedding_model"] = np.asarray(primary_embedding_model or "", dtype=object)
            
            # Save to temporary file first, then validate
            tmp_npz_path = out_path + ".tmp"
            try:
                _atomic_save_npz(tmp_npz_path, **npz_data)
                
                # Validate before atomic move
                v_ok, issues, meta = validate_npz(tmp_npz_path)
                if not v_ok:
                    status = "error"
                    notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5])
                    logger.error(f"NPZ validation failed for {doc_name}: {notes}")
                    # Don't write invalid NPZ
                    try:
                        os.remove(tmp_npz_path)
                    except Exception:
                        pass
                else:
                    # Move temp file to final location
                    os.replace(tmp_npz_path, out_path)
                    logger.info(f"  [{i+1}/{len(documents)}] ✓ {doc_name}: NPZ saved to {out_path}")
                    success_count += 1
            except Exception as e:
                logger.error(f"  [{i+1}/{len(documents)}] ✗ {doc_name}: error saving - {e}", exc_info=True)
                # Clean up temp file if exists
                try:
                    if os.path.exists(tmp_npz_path):
                        os.remove(tmp_npz_path)
                except Exception:
                    pass
            
            if status == "error":
                logger.error(f"  [{i+1}/{len(documents)}] ✗ {doc_name}: failed (status=error)")
        except Exception as e:
            logger.error(f"  [{i+1}/{len(documents)}] ✗ {doc_name}: error saving - {e}", exc_info=True)
    
    logger.info(f"Batch processing complete: {success_count}/{len(documents)} documents processed successfully")
    return 0 if success_count == len(documents) else 1


def _payload_summary(*, doc: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a privacy-safe payload summary for NPZ.
    This must not contain raw user text by default.
    """
    results_by_extractor = payload.get("results_by_extractor") if isinstance(payload.get("results_by_extractor"), dict) else {}
    timings_by_extractor = payload.get("timings_by_extractor") if isinstance(payload.get("timings_by_extractor"), dict) else {}
    systems_by_extractor = payload.get("systems_by_extractor") if isinstance(payload.get("systems_by_extractor"), dict) else {}

    # Minimal input stats (no raw).
    try:
        title_len = len(_safe_text(getattr(doc, "title", "")))
        desc_len = len(_safe_text(getattr(doc, "description", "")))
        transcripts = getattr(doc, "transcripts", {}) or {}
        transcript_len = len(_safe_text(" ".join(str(v or "") for v in transcripts.values()))) if isinstance(transcripts, dict) else 0
        comments = getattr(doc, "comments", []) or []
        comments_count = len(comments) if isinstance(comments, list) else 0
    except Exception:
        title_len, desc_len, transcript_len, comments_count = 0, 0, 0, 0

    summary = {
        "schema_version": "text_payload_summary_v1",
        "version": payload.get("version"),
        "device": payload.get("device"),
        "content_hash": _content_hash_for_document(doc),
        "input_stats": {
            "title_len_chars": int(title_len),
            "description_len_chars": int(desc_len),
            "transcript_len_chars": int(transcript_len),
            "comments_count": int(comments_count),
        },
        "extractors": {
            "results_keys": sorted(list(results_by_extractor.keys())),
            "timings_keys": sorted(list(timings_by_extractor.keys())),
            "systems_keys": sorted(list(systems_by_extractor.keys())),
        },
        # keep timings only (safe); omit detailed per-extractor results to avoid leaking text.
        "timings_by_extractor": timings_by_extractor,
    }

    # Optional privacy-safe embedding identifier block (source-of-truth for downstream vector stores).
    try:
        embid = results_by_extractor.get("EmbeddingSourceIdExtractor") if isinstance(results_by_extractor, dict) else None
        emb_info = embid.get("embedding_source_id") if isinstance(embid, dict) and isinstance(embid.get("embedding_source_id"), dict) else None
        if isinstance(emb_info, dict):
            out = {}
            for k in ("vector_id", "vector_store_uri", "embedding_relpath", "model_version", "primary_source"):
                v = emb_info.get(k)
                if isinstance(v, str) and v:
                    out[k] = v
            if out:
                summary["embedding_source_id"] = out
    except Exception:
        pass

    return summary


def main() -> int:
    tp_root = Path(__file__).resolve().parent
    repo_root = tp_root.parent

    # Ensure TextProcessor root is on PYTHONPATH so `import src.*` works (namespace package).
    if str(tp_root) not in sys.path:
        sys.path.insert(0, str(tp_root))
    # Ensure DataProcessor root is on PYTHONPATH so `import dp_models` works.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # VisualProcessor utils imports (manifest + validator)
    vp_root = repo_root / "VisualProcessor"
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))

    from src.core.main_processor import MainProcessor, load_document_from_json  # type: ignore
    from utils.manifest import RunManifest, ManifestComponent  # type: ignore
    from utils.artifact_validator import validate_npz  # type: ignore

    parser = argparse.ArgumentParser(description="TextProcessor CLI (per-run NPZ artifacts)")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-json", type=str, default=None, help="Path to VideoDocument JSON (single document)")
    input_group.add_argument("--input-dir", type=str, default=None, help="Directory with JSON documents (batch mode)")
    input_group.add_argument("--input-json-list", type=str, default=None, help="Comma-separated list of JSON paths (batch mode)")
    parser.add_argument(
        "--rs-base",
        type=str,
        default="./_runs/result_store",
        help="Base result_store path (per-run subdir will be created). Default matches DataProcessor baseline layout.",
    )
    parser.add_argument(
        "--run-rs-path",
        type=str,
        default=None,
        help="Explicit per-run result_store directory (overrides --rs-base/platform/video/run). "
             "Expected: <rs_base>/<platform_id>/<video_id>/<run_id>",
    )
    parser.add_argument("--platform-id", type=str, default="youtube")
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sampling-policy-version", type=str, default="v1")
    parser.add_argument("--config-hash", type=str, default=None)
    parser.add_argument("--dataprocessor-version", type=str, default="unknown")

    parser.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="If set, will also run GPU embedders (slower/heavier). Default is CPU-only extractors.",
    )
    parser.add_argument(
        "--devices-config-json",
        type=str,
        default=None,
        help="JSON dict mapping devices to extractor class names. Example: "
             '\'{"cpu":["LexicalStatsExtractor","TagsExtractor"],"cpu2":["ASRTextProxyExtractor"]}\'',
    )
    parser.add_argument(
        "--extractor-params-json",
        type=str,
        default=None,
        help="JSON dict mapping extractor class name -> params dict. Example: "
             '\'{"ASRTextProxyExtractor":{"enable_rhythm":false}}\'',
    )
    parser.add_argument(
        "--disabled-extractors",
        type=str,
        default="",
        help="Comma-separated list of extractor class names to disable (removed from devices_config).",
    )
    parser.add_argument(
        "--no-strict-extractors",
        action="store_true",
        help="If set, extractor import/creation errors will not fail the whole run (NOT recommended).",
    )
    parser.add_argument(
        "--store-raw-payload",
        action="store_true",
        help="Debug-only: store raw TextProcessor payload under _tmp_text/. NOT for production (privacy).",
    )
    parser.add_argument(
        "--include-primary-embedding",
        action="store_true",
        default=True,
        help="Include primary_embedding in NPZ (default: True). Set --no-include-primary-embedding to disable.",
    )
    parser.add_argument(
        "--no-include-primary-embedding",
        action="store_false",
        dest="include_primary_embedding",
        help="Disable primary_embedding in NPZ.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for structured logs (default: <run_rs_path>/_logs/). If None, logs go to stderr.",
    )
    # Batch processing arguments (Stage 4)
    parser.add_argument(
        "--batch-max-workers",
        type=int,
        default=None,
        help="Number of parallel workers for CPU extractors in batch mode (None = auto, typically os.cpu_count()). "
             "Currently used for internal optimizations even when processing single document.",
    )
    parser.add_argument(
        "--no-batch-gpu",
        action="store_true",
        help="Disable GPU batching optimizations (use sequential extract() for GPU extractors).",
    )
    parser.add_argument(
        "--no-batch-cpu-parallel",
        action="store_true",
        help="Disable CPU parallelism optimizations (use sequential processing for CPU extractors).",
    )
    args = parser.parse_args()

    video_id = args.video_id or os.path.splitext(os.path.basename(args.input_json))[0]
    run_id = args.run_id or uuid.uuid4().hex[:12]
    config_hash = args.config_hash
    if not config_hash:
        disabled = [x.strip() for x in str(args.disabled_extractors or "").split(",") if x.strip()]
        cfg_dump = json.dumps(
            {
                "enable_embeddings": bool(args.enable_embeddings),
                "sampling_policy_version": args.sampling_policy_version,
                "devices_config_json": args.devices_config_json,
                "extractor_params_json": args.extractor_params_json,
                "disabled_extractors": disabled,
                "strict_extractors": (not bool(args.no_strict_extractors)),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        config_hash = _sha256_text(cfg_dump)[:16]

    run_rs_path = (
        os.path.abspath(args.run_rs_path)
        if args.run_rs_path
        else os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id)
    )
    os.makedirs(run_rs_path, exist_ok=True)

    component_name = "text_processor"
    comp_dir = os.path.join(run_rs_path, component_name)
    # Per-run sub-artifacts (allowed): store large matrices/vectors as *.npy inside run_rs_path and list them in manifest.
    sub_artifacts_dir = os.path.join(comp_dir, "_artifacts")
    os.makedirs(sub_artifacts_dir, exist_ok=True)

    # Setup structured logging
    log_dir = args.log_dir or os.path.join(run_rs_path, "_logs")
    if args.log_dir or not sys.stderr.isatty():
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "text_processor.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stderr),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
        )
    logger = logging.getLogger("TextProcessor.run_cli")

    manifest_path = os.path.join(run_rs_path, "manifest.json")
    manifest = RunManifest(
        path=manifest_path,
        run_meta={
            "platform_id": args.platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": args.sampling_policy_version,
            "dataprocessor_version": str(args.dataprocessor_version),
            "created_at": _utc_iso_now(),
        },
    )

    # Default devices config: CPU-only by default (baseline-safe).
    devices_config: Dict[str, List[str]] = {"cpu": ["LexicalStatsExtractor", "TagsExtractor", "ASRTextProxyExtractor"]}
    extractor_params: Dict[str, Dict[str, Any]] = {}

    # Optional JSON override (preferred for website-driven configs).
    if args.devices_config_json:
        try:
            dc = json.loads(str(args.devices_config_json))
            if isinstance(dc, dict):
                devices_config = {str(k): (v if isinstance(v, list) else [v]) for k, v in dc.items()}
        except Exception as e:
            raise RuntimeError(f"Invalid --devices-config-json: {e}") from e

    if args.extractor_params_json:
        try:
            ep = json.loads(str(args.extractor_params_json))
            if isinstance(ep, dict):
                extractor_params = {str(k): (v if isinstance(v, dict) else {}) for k, v in ep.items()}
        except Exception as e:
            raise RuntimeError(f"Invalid --extractor-params-json: {e}") from e

    # Embeddings profile augments defaults only if user didn't override devices_config explicitly.
    if args.enable_embeddings and not args.devices_config_json:
        devices_config["gpu"] = ["TitleEmbedder", "DescriptionEmbedder", "TranscriptChunkEmbedder", "CommentsEmbedder", "HashtagEmbedder"]
        devices_config["cpu2"] = [
            "TranscriptAggregatorExtractor",
            "CommentsAggregationExtractor",
            "EmbeddingStatsExtractor",
            "CosineMetricsExtractor",
            "EmbeddingSourceIdExtractor",
        ]

    # Disable extractors (remove from all device groups).
    disabled = {x.strip() for x in str(args.disabled_extractors or "").split(",") if x.strip()}
    if disabled:
        for dev, names in list(devices_config.items()):
            devices_config[dev] = [n for n in (names or []) if str(n) not in disabled]
        # drop empty groups
        devices_config = {k: v for k, v in devices_config.items() if v}

    # Determine batch mode
    input_paths: List[str] = []
    if args.input_json:
        input_paths = [os.path.abspath(args.input_json)]
    elif args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 1
        input_paths = sorted([str(p) for p in input_dir.glob("*.json")])
        if not input_paths:
            logger.error(f"No JSON files found in {input_dir}")
            return 1
        logger.info(f"Batch mode: found {len(input_paths)} documents in {input_dir}")
    elif args.input_json_list:
        input_paths = [os.path.abspath(p.strip()) for p in args.input_json_list.split(",")]
        logger.info(f"Batch mode: processing {len(input_paths)} documents from list")
    
    if not input_paths:
        logger.error("No input documents specified")
        return 1
    
    # Batch mode: process multiple documents
    if len(input_paths) > 1:
        return _process_batch(
            input_paths=input_paths,
            devices_config=devices_config,
            extractor_params=extractor_params,
            strict=not bool(args.no_strict_extractors),
            base_artifacts_dir=sub_artifacts_dir,
            logger=logger,
            batch_max_workers=getattr(args, "batch_max_workers", None),
            batch_enable_gpu_batching=not getattr(args, "no_batch_gpu", False),
            batch_enable_cpu_parallel=not getattr(args, "no_batch_cpu_parallel", False),
            rs_base=args.rs_base,
            run_rs_path=run_rs_path,
            platform_id=args.platform_id,
            run_id=run_id,
            sampling_policy_version=args.sampling_policy_version,
            config_hash=config_hash,
            dataprocessor_version=args.dataprocessor_version,
            enable_embeddings=bool(args.enable_embeddings),
            include_primary_embedding=bool(args.include_primary_embedding),
            store_raw_payload=bool(args.store_raw_payload),
        )
    
    # Single document mode (original logic)
    started_at = _utc_iso_now()
    t0 = time.time()
    status = "ok"
    empty_reason: Optional[str] = None
    err: Optional[str] = None
    payload: Dict[str, Any] = {}
    doc = None
    try:
        doc = load_document_from_json(input_paths[0])
        logger.info(f"TextProcessor: loaded document from {input_paths[0]}")
        processor = MainProcessor(
            devices_config=devices_config,
            extractor_params=extractor_params,
            strict=(not bool(args.no_strict_extractors)),
            artifacts_dir=sub_artifacts_dir,
            logger=logger,
            # Batch processing parameters (Stage 4)
            batch_max_workers=getattr(args, "batch_max_workers", None),
            batch_enable_gpu_batching=not getattr(args, "no_batch_gpu", False),
            batch_enable_cpu_parallel=not getattr(args, "no_batch_cpu_parallel", False),
        )
        payload = processor.run(doc) or {}
        # Use status from MainProcessor if available
        if "status" in payload:
            status = str(payload["status"])
            empty_reason = payload.get("empty_reason")
            if status == "error":
                errors_by_extractor = payload.get("errors_by_extractor", {})
                if errors_by_extractor:
                    err = "; ".join(f"{k}: {v}" for k, v in list(errors_by_extractor.items())[:3])
                else:
                    err = payload.get("error") or "unknown_error"
        logger.info(f"TextProcessor: MainProcessor.run() completed with status={status}")
    except Exception as e:
        status = "error"
        err = str(e)
        logger.error(f"TextProcessor: exception in MainProcessor.run(): {err}", exc_info=True)
    finished_at = _utc_iso_now()
    duration_ms = int((time.time() - t0) * 1000)

    # Fallback empty detection: if MainProcessor didn't set status=empty, check doc text
    if status == "ok":
        try:
            has_any_text = False
            if doc is not None:
                if _safe_text(getattr(doc, "title", "")) or _safe_text(getattr(doc, "description", "")):
                    has_any_text = True
                transcripts = getattr(doc, "transcripts", {}) or {}
                if isinstance(transcripts, dict) and any(_safe_text(v) for v in transcripts.values()):
                    has_any_text = True
                comments = getattr(doc, "comments", []) or []
                if isinstance(comments, list) and any(_safe_text(getattr(c, "text", "")) for c in comments):
                    has_any_text = True
            if not has_any_text:
                status = "empty"
                empty_reason = "no_text_available"
                logger.info(f"TextProcessor: detected empty (no text in document)")
        except Exception:
            # If empty detection failed, keep ok (do not mask real errors).
            pass

    # Prefer stable flat scalar features if provided by the processor/extractors.
    base_for_scalars = payload.get("features_flat") if isinstance(payload.get("features_flat"), dict) else payload
    scalars = _flatten_scalars(base_for_scalars)
    feature_names = np.asarray(sorted(scalars.keys()), dtype=object)
    feature_values = np.asarray([float(scalars[k]) for k in feature_names.tolist()], dtype=np.float32)

    # Canonical single per-run artifact for TextProcessor (docs/contracts/ARTIFACTS_AND_SCHEMAS.md).
    # Deterministic filename avoids orphaned old artifacts on re-run.
    out_path = os.path.join(comp_dir, "text_features.npz")

    # Privacy: do not store raw payload by default.
    payload_summary = _payload_summary(doc=doc, payload=payload) if doc is not None else {"schema_version": "text_payload_summary_v1", "error": "doc_missing"}

    # Optional: export a privacy-safe "primary embedding" into NPZ for downstream similarity modules.
    # This avoids relying on global TREND_TEXT_ARTIFACTS_DIR scanning (non-deterministic).
    primary_embedding = None
    primary_embedding_present = False
    primary_embedding_source = None
    primary_embedding_model = None
    if bool(args.include_primary_embedding) and bool(args.enable_embeddings) and isinstance(payload, dict):
        try:
            rbe = payload.get("results_by_extractor") if isinstance(payload.get("results_by_extractor"), dict) else {}
            src = rbe.get("EmbeddingSourceIdExtractor") if isinstance(rbe.get("EmbeddingSourceIdExtractor"), dict) else None
            emb_info = None
            if isinstance(src, dict):
                emb_info = src.get("embedding_source_id") if isinstance(src.get("embedding_source_id"), dict) else None
            if isinstance(emb_info, dict):
                rel = emb_info.get("embedding_relpath")
                # embedding_relpath is always relative to `text_processor/_artifacts/`
                emb_path = os.path.join(sub_artifacts_dir, rel) if isinstance(rel, str) and rel else None
                if isinstance(emb_path, str) and emb_path and os.path.isfile(emb_path):
                    vec = np.load(emb_path)
                    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                    if vec.size > 0:
                        primary_embedding = vec.astype(np.float32)
                        primary_embedding_present = True
                        primary_embedding_source = str(rel or os.path.basename(emb_path))
                        primary_embedding_model = emb_info.get("model_version") or None
        except Exception:
            primary_embedding = None
            primary_embedding_present = False
            primary_embedding_source = None
            primary_embedding_model = None

    raw_payload_path = None
    if bool(args.store_raw_payload):
        try:
            tmp_dir = os.path.join(run_rs_path, "_tmp_text")
            os.makedirs(tmp_dir, exist_ok=True)
            raw_payload_path = os.path.join(tmp_dir, "raw_payload.json")
            tmp_path = raw_payload_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, raw_payload_path)
        except Exception:
            raw_payload_path = None

    # Model meta: use models_used from MainProcessor (collected from all extractors).
    models_used: list[dict] = payload.get("models_used", []) if isinstance(payload.get("models_used"), list) else []
    if models_used:
        logger.info(f"TextProcessor: collected {len(models_used)} model(s) from extractors")

    # Validate NPZ before writing (in-memory validation via temporary file).
    # Create temporary NPZ, validate it, then atomically move to final location.
    tmp_npz_path = None
    try:
        fd, tmp_npz_path = tempfile.mkstemp(prefix=Path(out_path).name + ".", suffix=".npz", dir=comp_dir)
        os.close(fd)
        
        np.savez_compressed(
            tmp_npz_path,
            feature_names=feature_names,
            feature_values=feature_values,
            payload=np.asarray(payload_summary, dtype=object),
            primary_embedding=(primary_embedding if primary_embedding_present else np.full((1,), np.nan, dtype=np.float32)),
            primary_embedding_present=np.asarray(bool(primary_embedding_present)),
            primary_embedding_source=np.asarray(primary_embedding_source if primary_embedding_source else ""),
            primary_embedding_model=np.asarray(primary_embedding_model if primary_embedding_model else ""),
            meta=_meta(
                producer_version=str(payload.get("version") or "unknown"),
                schema_version="text_npz_v1",
                status=status,
                extra={
                    "platform_id": args.platform_id,
                    "video_id": video_id,
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "sampling_policy_version": args.sampling_policy_version,
                    "dataprocessor_version": str(args.dataprocessor_version),
                    "empty_reason": empty_reason,
                    "error": err,
                    "input_json_basename": os.path.basename(str(args.input_json or "")),
                    "enable_embeddings": bool(args.enable_embeddings),
                    "models_used": models_used,
                    "raw_payload_path": os.path.abspath(raw_payload_path) if raw_payload_path else None,
                },
            ),
        )
        
        # Validate before atomic move
        v_ok, issues, meta = validate_npz(tmp_npz_path)
        notes = None
        if not v_ok:
            status = "error"
            notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5])
            logger.error(f"TextProcessor: NPZ validation failed: {notes}")
            # Don't write invalid NPZ
            try:
                os.remove(tmp_npz_path)
            except Exception:
                pass
            tmp_npz_path = None
        else:
            # Atomic move: tmp → final
            os.replace(tmp_npz_path, out_path)
            tmp_npz_path = None
            logger.info(f"TextProcessor: wrote NPZ to {out_path}")
    except Exception as e:
        status = "error"
        err = str(e) if not err else f"{err}; npz_write_error: {str(e)}"
        logger.error(f"TextProcessor: failed to write NPZ: {err}", exc_info=True)
        if tmp_npz_path:
            try:
                os.remove(tmp_npz_path)
            except Exception:
                pass

    error_code = None
    if status == "error":
        error_code = "artifact_validation_failed" if not v_ok else "exception"

    device_used = "cpu" if not args.enable_embeddings else "cuda"

    # Generate render-context for TextProcessor (before updating manifest)
    render_path = None
    if out_path and os.path.exists(out_path):
        try:
            from src.core.renderer import render_text_processor  # type: ignore
            
            render = render_text_processor(out_path, comp_dir)
            render_path = os.path.join(comp_dir, "_render", "render_context.json")
        except Exception as e:
            # Best-effort: do not fail run if render fails
            logger.warning(f"Failed to generate render-context for TextProcessor: {e}")

    # Register NPZ + sub-artifacts (*.npy) + render for this run.
    artifacts = [{"path": out_path, "type": "npz"}]
    try:
        if os.path.isdir(sub_artifacts_dir):
            for fn in sorted(os.listdir(sub_artifacts_dir)):
                if fn.endswith(".npy"):
                    artifacts.append({"path": os.path.join(sub_artifacts_dir, fn), "type": "npy"})
    except Exception:
        pass
    
    # Add render artifact if available
    if render_path and os.path.exists(render_path):
        artifacts.append({"path": render_path, "type": "render"})
    
    # Add HTML report artifacts if available
    try:
        render_dir = os.path.join(comp_dir, "_render")
        if os.path.isdir(render_dir):
            for fn in sorted(os.listdir(render_dir)):
                if fn.endswith("_report.html"):
                    html_path = os.path.join(render_dir, fn)
                    artifacts.append({"path": html_path, "type": "html_report"})
    except Exception:
        pass

    manifest.upsert_component(
        ManifestComponent(
            name=component_name,
            kind="text",
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            artifacts=artifacts,
            error=err,
            error_code=error_code,
            notes=notes,
            device_used=device_used,
            producer_version=(meta or {}).get("producer_version") if isinstance(meta, dict) else None,
            schema_version=(meta or {}).get("schema_version") if isinstance(meta, dict) else None,
        )
    )

    return 0 if status != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())


