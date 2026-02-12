"""
Batch processing utilities for content_domain component.

Stage 3: GPU batching для content_domain с гибридным подходом:
- Сбор frame embeddings из всех видео (из core_clip)
- Группировка в батчи для text embeddings
- Распределение результатов обратно по видео
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta
from utils.artifact_validator import validate_npz

# Import content_domain functions
_content_domain_path = _visual_processor_path / "core" / "model_process" / "core_identity" / "content_domain"
sys.path.insert(0, str(_content_domain_path.parent.parent.parent.parent))

logger = get_logger("VisualProcessor.content_domain_batch")

# Import from content_domain/main.py
_content_domain_main = _content_domain_path / "main.py"
if _content_domain_main.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("content_domain_main", str(_content_domain_main))
    if spec and spec.loader:
        content_domain_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(content_domain_module)
        
        # Import functions
        _load_npz = getattr(content_domain_module, "_load_npz", None)
        _read_json = getattr(content_domain_module, "_read_json", None)
        _read_jsonl = getattr(content_domain_module, "_read_jsonl", None)
        _l2norm_rows = getattr(content_domain_module, "_l2norm_rows", None)
        _load_triton_spec_via_model_manager = getattr(content_domain_module, "_load_triton_spec_via_model_manager", None)
        _triton_infer = getattr(content_domain_module, "_triton_infer", None)
        _compute_text_label_embeddings_triton = getattr(content_domain_module, "_compute_text_label_embeddings_triton", None)
        _require_frame_indices = getattr(content_domain_module, "_require_frame_indices", None)
        _load_domain_db = getattr(content_domain_module, "_load_domain_db", None)
        NAME = getattr(content_domain_module, "NAME", "content_domain")
        VERSION = getattr(content_domain_module, "VERSION", "0.1")
        SCHEMA_VERSION = getattr(content_domain_module, "SCHEMA_VERSION", "content_domain_npz_v1")
        ARTIFACT_FILENAME = getattr(content_domain_module, "ARTIFACT_FILENAME", "content_domain.npz")
    else:
        raise ImportError("Failed to load content_domain module")
else:
    raise ImportError(f"content_domain/main.py not found at {_content_domain_main}")


def _atomic_save_npz(out_path: str, **kwargs) -> None:
    """Atomic NPZ save."""
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(out_path) + ".",
        suffix=".npz",
        dir=out_dir,
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **kwargs)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def process_content_domain_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Batch processing для content_domain с гибридным подходом.
    
    Stage 3: GPU batching для content_domain.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация content_domain
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"content_domain | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch})"
    )
    
    start_time = time.perf_counter()
    
    # Параметры конфигурации
    domain_db_dir = config.get("domain_db_dir", "dp_models/bundled_models/semantics/content_domain/v1")
    clip_text_model_spec = config.get("clip_text_model_spec", "clip_text_triton")
    topk = int(config.get("topk", 5))
    threshold_global = float(config.get("threshold_global", 0.23))
    thresholds_json = config.get("thresholds_json") or None
    
    # Resolve domain_db_dir path (try DP_MODELS_ROOT if relative)
    if not os.path.isabs(domain_db_dir):
        mr = os.environ.get("DP_MODELS_ROOT")
        if mr:
            candidate = os.path.join(str(mr), domain_db_dir)
            if os.path.isdir(candidate):
                domain_db_dir = candidate
                logger.info(f"{NAME} | batch | Resolved domain_db_dir via DP_MODELS_ROOT: {domain_db_dir}")
    
    # Загружаем domain db и вычисляем text embeddings (один раз для всех видео)
    try:
        label_ids, label_names, prompts_per_label, db_meta, threshold_per_label = _load_domain_db(
            db_dir=domain_db_dir,
            threshold_global=threshold_global,
            thresholds_json=str(thresholds_json) if thresholds_json else None,
        )
    except RuntimeError as e:
        if "not found" in str(e) or "missing" in str(e):
            # Valid empty: database not available
            logger.warning(f"{NAME} | batch | Domain database not found, writing empty artifacts: {e}")
            results = []
            for video_ctx in video_contexts:
                try:
                    metadata = video_ctx.load_metadata()
                    frame_indices = _require_frame_indices(metadata)
                    
                    uts = metadata.get("union_timestamps_sec")
                    if uts is None:
                        raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
                    uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
                    fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
                    if fi_np.size == 0:
                        raise RuntimeError(f"{NAME} | frame_indices is empty (no-fallback)")
                    times_s = uts_arr[fi_np].astype(np.float32)
                    
                    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
                    missing = [k for k in required_run_keys if not metadata.get(k)]
                    if missing:
                        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
                    
                    meta_out: Dict[str, Any] = {
                        "producer": NAME,
                        "producer_version": VERSION,
                        "schema_version": SCHEMA_VERSION,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "status": "empty",
                        "empty_reason": "dependency_missing",
                    }
                    for k in required_run_keys:
                        meta_out[k] = metadata.get(k)
                    meta_out["dataprocessor_version"] = str(metadata.get("dataprocessor_version") or "unknown")
                    meta_out = apply_models_meta(meta_out, models_used=[])
                    
                    component_dir = video_ctx.get_component_rs_path(NAME)
                    npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
                    os.makedirs(component_dir, exist_ok=True)
                    
                    np.savez_compressed(
                        npz_path,
                        frame_indices=fi_np,
                        times_s=times_s,
                        label_ids=np.array([], dtype=np.int32),
                        label_names=np.array([], dtype="U"),
                        frame_topk_ids=np.full((len(frame_indices), topk), -1, dtype=np.int32),
                        frame_topk_scores=np.full((len(frame_indices), topk), 0.0, dtype=np.float32),
                        frame_is_confident_top1=np.zeros((len(frame_indices),), dtype=np.bool_),
                        meta=np.asarray(meta_out, dtype=object),
                    )
                    
                    results.append({
                        "video_id": video_ctx.video_id,
                        "status": "ok",
                        "saved_path": npz_path,
                    })
                except Exception as e:
                    logger.exception(f"{NAME} | batch | video {video_ctx.video_id} failed: {e}")
                    results.append({
                        "video_id": video_ctx.video_id,
                        "status": "error",
                        "error": str(e),
                    })
            return results
        raise
    
    if topk != 5:
        raise RuntimeError(f"{NAME} | batch | topk must be 5 (contract), got {topk}")
    
    # Get triton_http_url from config or environment
    triton_http_url = config.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL")
    
    # Загружаем Triton spec для text embeddings
    txt_mm = _load_triton_spec_via_model_manager(str(clip_text_model_spec), triton_http_url=triton_http_url)
    
    # Вычисляем text embeddings для всех доменов (один раз для всех видео)
    logger.info(f"{NAME} | batch | computing text embeddings for {len(label_ids)} domains")
    label_emb = _compute_text_label_embeddings_triton(prompts_per_label=prompts_per_label, txt_mm=txt_mm)  # (A,512)
    
    # Этап 1: Сбор всех frame embeddings из core_clip с привязкой к видео
    embeddings_by_video: List[Dict[str, Any]] = []
    all_embeddings: List[Tuple[int, int, np.ndarray]] = []  # (video_idx, frame_idx, embedding)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            
            # Получаем frame_indices
            try:
                frame_indices = _require_frame_indices(metadata)
            except Exception as e:
                logger.error(f"{NAME} | batch | video {video_ctx.video_id} failed to get frame_indices: {e}")
                embeddings_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": str(e),
                })
                continue
            
            if not frame_indices:
                logger.warning(f"{NAME} | batch | video {video_ctx.video_id} has no frame_indices")
                embeddings_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "status": "empty",
                })
                continue
            
            # Загружаем core_clip embeddings
            core_clip_path = os.path.join(str(video_ctx.rs_path), "core_clip", "embeddings.npz")
            if not os.path.isfile(core_clip_path):
                logger.error(f"{NAME} | batch | video {video_ctx.video_id} core_clip embeddings not found: {core_clip_path}")
                embeddings_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": f"core_clip embeddings not found: {core_clip_path}",
                })
                continue
            
            clip_npz = _load_npz(core_clip_path)
            clip_fi = np.asarray(clip_npz.get("frame_indices"), dtype=np.int32).reshape(-1)
            clip_emb = np.asarray(clip_npz.get("frame_embeddings"), dtype=np.float32)
            
            if clip_fi.size == 0 or clip_emb.size == 0:
                logger.error(f"{NAME} | batch | video {video_ctx.video_id} core_clip embeddings empty")
                embeddings_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": "core_clip embeddings empty",
                })
                continue
            
            if clip_emb.ndim != 2 or clip_emb.shape[0] != clip_fi.shape[0]:
                logger.error(f"{NAME} | batch | video {video_ctx.video_id} core_clip embeddings invalid shape: {clip_emb.shape}")
                embeddings_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": f"core_clip embeddings invalid shape: {clip_emb.shape}",
                })
                continue
            
            clip_emb = _l2norm_rows(clip_emb)
            clip_map: Dict[int, int] = {int(u): int(i) for i, u in enumerate(clip_fi.tolist())}
            
            # Получаем timestamps
            uts = metadata.get("union_timestamps_sec")
            if uts is None:
                raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
            uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            if fi_np.size == 0:
                raise RuntimeError(f"{NAME} | frame_indices is empty (no-fallback)")
            if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
                raise RuntimeError(f"{NAME} | frame_indices out of range for union_timestamps_sec")
            times_s = uts_arr[fi_np].astype(np.float32)
            
            # Извлекаем embeddings для требуемых frame_indices
            sel_rows: List[int] = []
            for u in frame_indices:
                if int(u) not in clip_map:
                    raise RuntimeError(f"{NAME} | batch | video {video_ctx.video_id} core_clip embeddings do not cover required frame_index={u} (no-fallback)")
                sel_rows.append(int(clip_map[int(u)]))
            
            frame_emb = clip_emb[np.asarray(sel_rows, dtype=np.int32)]  # (N,D)
            
            # Сохраняем embeddings в общий батч
            video_emb_start_idx = len(all_embeddings)
            for i, frame_idx in enumerate(frame_indices):
                all_embeddings.append((video_idx, frame_idx, frame_emb[i]))
            video_emb_end_idx = len(all_embeddings)
            
            # Сохраняем метаданные из core_clip
            clip_meta = clip_npz.get("meta")
            upstream_models_used: List[Dict[str, Any]] = []
            upstream_model_signature: Any = None
            if isinstance(clip_meta, dict):
                if isinstance(clip_meta.get("models_used"), list):
                    upstream_models_used = clip_meta.get("models_used") or []
                upstream_model_signature = clip_meta.get("model_signature")
            
            embeddings_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": frame_indices,
                "times_s": times_s,
                "frame_emb": frame_emb,
                "upstream_models_used": upstream_models_used,
                "upstream_model_signature": upstream_model_signature,
                "emb_start_idx": video_emb_start_idx,
                "emb_end_idx": video_emb_end_idx,
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"{NAME} | batch | video {video_ctx.video_id} failed to prepare: {e}")
            embeddings_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": [],
                "times_s": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_embeddings:
        logger.error(f"{NAME} | batch | no embeddings collected from any video")
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": "no embeddings collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"{NAME} | batch | collected {len(all_embeddings)} frame embeddings from {len(embeddings_by_video)} videos")
    
    # Проверяем размерность embeddings
    if label_emb.ndim != 2 or label_emb.shape[0] != len(label_ids):
        raise RuntimeError(f"{NAME} | batch | label embeddings shape mismatch: {label_emb.shape} vs {len(label_ids)} labels")
    
    # Проверяем размерность frame embeddings
    first_emb = all_embeddings[0][2]
    embed_dim = int(first_emb.shape[0]) if first_emb.ndim == 1 else int(first_emb.shape[1])
    if label_emb.shape[1] != embed_dim:
        raise RuntimeError(f"{NAME} | batch | label embeddings dim mismatch: {label_emb.shape[1]} vs frame_emb dim={embed_dim}")
    
    # Этап 2: Вычисление similarity для всех кадров
    logger.info(f"{NAME} | batch | computing similarities for {len(all_embeddings)} frames")
    
    # Собираем все embeddings в один массив для векторизованного вычисления
    all_emb_array = np.stack([emb for _, _, emb in all_embeddings], axis=0)  # (N_total, D)
    all_emb_array = _l2norm_rows(all_emb_array)
    
    # Вычисляем cosine similarity: (N_total, A)
    sims = np.matmul(all_emb_array, label_emb.T).astype(np.float32)
    
    # Этап 3: Распределение результатов обратно по видео
    logger.info(f"{NAME} | batch | distributing results back to videos")
    
    results = []
    for video_info in embeddings_by_video:
        video_idx = video_info["video_idx"]
        video_ctx = video_contexts[video_idx]
        
        if video_info["status"] != "ok":
            results.append({
                "video_id": video_ctx.video_id,
                "status": video_info["status"],
                "error": video_info.get("error"),
            })
            continue
        
        # Извлекаем similarities для этого видео
        video_frame_indices = video_info["frame_indices"]
        emb_start_idx = video_info.get("emb_start_idx", 0)
        emb_end_idx = video_info.get("emb_end_idx", len(sims))
        
        if emb_start_idx >= emb_end_idx or emb_start_idx >= len(sims):
            results.append({
                "video_id": video_ctx.video_id,
                "status": "empty",
                "empty_reason": "no frames processed",
            })
            continue
        
        # Извлекаем similarities для этого видео
        video_sims = sims[emb_start_idx:emb_end_idx]  # (N, A)
        video_times_s = video_info["times_s"]
        
        # Проверяем соответствие размеров
        if len(video_sims) != len(video_frame_indices) or len(video_sims) != len(video_times_s):
            logger.warning(
                f"{NAME} | batch | video {video_ctx.video_id} size mismatch: "
                f"sims={len(video_sims)}, indices={len(video_frame_indices)}, times={len(video_times_s)}"
            )
            # Используем минимальный размер
            min_size = min(len(video_sims), len(video_frame_indices), len(video_times_s))
            video_sims = video_sims[:min_size]
            video_frame_indices = video_frame_indices[:min_size]
            video_times_s = video_times_s[:min_size]
        
        # Вычисляем top-K для каждого кадра
        A = int(label_emb.shape[0])
        order = np.argsort(-video_sims, axis=1)[:, :topk]  # (N, K) indices into A
        frame_topk_scores = np.take_along_axis(video_sims, order, axis=1).astype(np.float32)
        label_ids_np = np.asarray(label_ids, dtype=np.int32)
        frame_topk_ids = label_ids_np[order].astype(np.int32)
        
        # Вычисляем is_confident для top-1
        threshold_global_val = float(db_meta.get("threshold_global") or threshold_global)
        frame_is_confident_top1 = np.zeros((len(video_frame_indices),), dtype=np.bool_)
        for i in range(int(frame_topk_ids.shape[0])):
            lid = int(frame_topk_ids[i, 0])
            sc = float(frame_topk_scores[i, 0])
            thr = float(threshold_per_label.get(lid, threshold_global_val))
            frame_is_confident_top1[i] = bool(np.isfinite(sc) and sc >= thr)
        
        # Video aggregate (track=1): max over time
        track_ids = np.asarray([0], dtype=np.int32)
        track_present_mask = np.asarray([True], dtype=np.bool_)
        max_scores = np.max(video_sims, axis=0)  # (A,)
        top_vid = np.argsort(-max_scores)[:topk]
        track_topk_scores = np.asarray(max_scores[top_vid], dtype=np.float32).reshape(1, topk)
        track_topk_ids = np.asarray(label_ids_np[top_vid], dtype=np.int32).reshape(1, topk)
        top1_lid = int(track_topk_ids[0, 0])
        top1_sc = float(track_topk_scores[0, 0])
        track_is_confident_top1 = np.asarray([bool(np.isfinite(top1_sc) and top1_sc >= float(threshold_per_label.get(top1_lid, threshold_global_val)))], dtype=np.bool_)
        
        semantic_label_names = np.asarray([f"{int(i)}:{str(n)}" for i, n in zip(label_ids, label_names)], dtype="U")
        
        # Thresholds array
        threshold_per_label_arr = np.full((A,), np.nan, dtype=np.float32)
        for i, lid in enumerate(label_ids):
            if int(lid) in threshold_per_label:
                threshold_per_label_arr[i] = float(threshold_per_label[int(lid)])
        
        # Сохраняем результаты в per-video rs_path
        component_dir = video_ctx.get_component_rs_path(NAME)
        npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
        os.makedirs(component_dir, exist_ok=True)
        
        # Подготовка метаданных
        metadata = video_ctx.load_metadata()
        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
        
        meta_out: Dict[str, Any] = {
            "producer": NAME,
            "producer_version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "ok",
            "empty_reason": None,
        }
        
        # Required run identity fields
        for k in required_run_keys:
            meta_out[k] = metadata.get(k)
        
        meta_out["dataprocessor_version"] = str(metadata.get("dataprocessor_version") or "unknown")
        
        meta_out.update({
            # thresholds + db
            "threshold_global": threshold_global_val,
            "threshold_per_label": threshold_per_label,
            **db_meta,
            # provenance chaining
            "core_clip_model_signature": video_info.get("upstream_model_signature"),
            # model system
            "models_used": [],
        })
        
        # Attach model provenance (clip_text only) + upstream (core_clip) for reproducibility
        meta_out["models_used"].extend(video_info.get("upstream_models_used", []))
        
        # Add clip_text model
        try:
            txt_entry = txt_mm.get("models_used_entry")
            if isinstance(txt_entry, dict):
                existing_names = {str(m.get("model_name") or "") for m in meta_out.get("models_used") or [] if isinstance(m, dict)}
                if str(txt_entry.get("model_name") or "") not in existing_names:
                    meta_out["models_used"].append(txt_entry)
        except Exception:
            pass
        
        meta_out = apply_models_meta(meta_out, models_used=meta_out.get("models_used"))
        
        # Сохранение NPZ
        npz_dict = {
            "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
            "times_s": video_times_s,
            "semantic_label_names": semantic_label_names,
            "threshold_per_label_arr": threshold_per_label_arr.astype(np.float32),
            "track_ids": track_ids,
            "track_present_mask": track_present_mask,
            "track_topk_ids": track_topk_ids,
            "track_topk_scores": track_topk_scores,
            "track_is_confident_top1": track_is_confident_top1,
            "frame_topk_ids": frame_topk_ids,
            "frame_topk_scores": frame_topk_scores,
            "frame_is_confident_top1": frame_is_confident_top1,
            "meta": np.asarray(meta_out, dtype=object),
        }
        
        _atomic_save_npz(npz_path, **npz_dict)
        
        # Валидация NPZ
        ok, issues, _ = validate_npz(npz_path)
        if not ok:
            try:
                if os.path.exists(npz_path):
                    os.remove(npz_path)
            except Exception:
                pass
            raise RuntimeError(
                f"{NAME} | batch | saved artifact failed validation: "
                + "; ".join([f"{i.level}:{i.message}" for i in issues])
            )
        
        results.append({
            "video_id": video_ctx.video_id,
            "status": "ok",
            "saved_path": npz_path,
        })
    
    duration = time.perf_counter() - start_time
    logger.info(
        f"{NAME} | batch | completed in {duration:.2f}s "
        f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
    )
    
    return results

