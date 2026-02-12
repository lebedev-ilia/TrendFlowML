"""
Batch processing utilities for core_clip component.

Stage 2: GPU batching PoC для core_clip с гибридным подходом:
- Сбор кадров из всех видео
- Группировка в батчи по max_frames_per_batch
- Последовательная обработка батчей
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
import torch
from PIL import Image

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz

# Import core_clip functions
_core_clip_path = _visual_processor_path / "core" / "model_process" / "core_clip"
sys.path.insert(0, str(_core_clip_path.parent.parent.parent))

logger = get_logger("VisualProcessor.core_clip_batch")

# Import from core_clip/main.py
# We need to import functions directly from the module file
_core_clip_main = _core_clip_path / "main.py"
if _core_clip_main.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("core_clip_main", str(_core_clip_main))
    if spec and spec.loader:
        core_clip_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_clip_module)
        
        # Import functions
        _clip_preprocess_batch = getattr(core_clip_module, "_clip_preprocess_batch", None)
        _clip_resize_batch_uint8 = getattr(core_clip_module, "_clip_resize_batch_uint8", None)
        _triton_infer_embeddings = getattr(core_clip_module, "_triton_infer_embeddings", None)
        init_clip = getattr(core_clip_module, "init_clip", None)
        compute_text_embeddings = getattr(core_clip_module, "compute_text_embeddings", None)
        _require_frame_indices = getattr(core_clip_module, "_require_frame_indices", None)
        _require_union_timestamps_sec = getattr(core_clip_module, "_require_union_timestamps_sec", None)
        _load_places365_prompts_from_bundle = getattr(core_clip_module, "_load_places365_prompts_from_bundle", None)
        SHOT_QUALITY_PROMPTS = getattr(core_clip_module, "SHOT_QUALITY_PROMPTS", [])
        SCENE_AESTHETIC_PROMPTS = getattr(core_clip_module, "SCENE_AESTHETIC_PROMPTS", [])
        SCENE_LUXURY_PROMPTS = getattr(core_clip_module, "SCENE_LUXURY_PROMPTS", [])
        SCENE_ATMOSPHERE_PROMPTS = getattr(core_clip_module, "SCENE_ATMOSPHERE_PROMPTS", [])
        CUT_DETECTION_TRANSITION_PROMPTS = getattr(core_clip_module, "CUT_DETECTION_TRANSITION_PROMPTS", [])
        POPULARITY_TOPIC_PROMPTS = getattr(core_clip_module, "POPULARITY_TOPIC_PROMPTS", [])
        NAME = getattr(core_clip_module, "NAME", "core_clip")
        VERSION = getattr(core_clip_module, "VERSION", "2.0")
        SCHEMA_VERSION = getattr(core_clip_module, "SCHEMA_VERSION", "core_clip_npz_v1")
        ARTIFACT_FILENAME = getattr(core_clip_module, "ARTIFACT_FILENAME", "embeddings.npz")
        PROMPTS_VERSION = getattr(core_clip_module, "PROMPTS_VERSION", "v3_2026-01-16")
    else:
        raise ImportError("Failed to load core_clip module")
else:
    raise ImportError(f"core_clip/main.py not found at {_core_clip_main}")


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


def process_core_clip_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    runtime: str = "inprocess",
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Batch processing для core_clip с гибридным подходом.
    
    Stage 2: GPU batching PoC для core_clip.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация core_clip
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        runtime: Runtime для inference (inprocess или triton)
        batch_size: Размер батча для inference
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"core_clip | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch}, runtime={runtime})"
    )
    
    start_time = time.perf_counter()
    
    # Этап 1: Сбор всех кадров с привязкой к видео
    frames_by_video: List[Dict[str, Any]] = []
    all_frames: List[Tuple[int, int, np.ndarray]] = []  # (video_idx, frame_idx, frame)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            total_frames = int(metadata.get("total_frames", 0))
            
            # Получаем frame_indices
            try:
                frame_indices = _require_frame_indices(metadata, NAME)
            except Exception as e:
                logger.error(f"core_clip | batch | video {video_ctx.video_id} failed to get frame_indices: {e}")
                frames_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "frame_manager": None,
                    "times_s": None,
                    "status": "error",
                    "error": str(e),
                })
                continue
            
            if not frame_indices:
                logger.warning(f"core_clip | batch | video {video_ctx.video_id} has no frame_indices")
                frames_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "frame_manager": None,
                    "times_s": None,
                    "status": "empty",
                })
                continue
            
            # Создаем FrameManager
            frame_manager = FrameManager(
                frames_dir=video_ctx.frames_dir,
                chunk_size=metadata.get("chunk_size", 32),
                cache_size=metadata.get("cache_size", 2),
            )
            
            # Получаем timestamps
            union_timestamps_sec = _require_union_timestamps_sec(metadata, total_frames=total_frames)
            times_s = union_timestamps_sec[np.asarray(frame_indices, dtype=np.int32)].astype(np.float32)
            
            # Загружаем кадры и сохраняем маппинг
            frames = []
            video_frame_start_idx = len(all_frames)  # Начальный индекс в общем батче для этого видео
            for frame_idx in frame_indices:
                try:
                    frame = frame_manager.get(frame_idx)
                    frames.append(frame)
                    # Сохраняем кадр в общий батч
                    all_frames.append((video_idx, frame_idx, frame))
                except Exception as e:
                    logger.warning(
                        f"core_clip | batch | video {video_ctx.video_id} failed to load frame {frame_idx}: {e}"
                    )
                    continue
            
            # Сохраняем информацию о диапазоне индексов для этого видео
            video_frame_end_idx = len(all_frames)
            
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": frame_indices,
                "frames": frames,
                "frame_manager": frame_manager,
                "times_s": times_s,
                "frame_start_idx": video_frame_start_idx,
                "frame_end_idx": video_frame_end_idx,
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"core_clip | batch | video {video_ctx.video_id} failed to prepare: {e}")
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": [],
                "frame_manager": None,
                "times_s": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_frames:
        logger.error("core_clip | batch | no frames collected from any video")
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": "no frames collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"core_clip | batch | collected {len(all_frames)} frames from {len(frames_by_video)} videos")
    
    # Этап 2: Группировка в батчи и обработка
    model = None
    preprocess = None
    device = "cpu"
    client = None
    
    try:
        if runtime == "inprocess":
            model_name = config.get("model_name", "ViT-B/32")
            model, preprocess, device = init_clip(model_name, preferred_device="auto")
        elif runtime == "triton":
            # TODO: Initialize Triton client
            triton_http_url = config.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL")
            if not triton_http_url:
                raise RuntimeError("core_clip | batch | runtime=triton requires triton_http_url")
            
            from dp_triton import TritonHttpClient, TritonError
            
            client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=60.0)
            if not client.ready():
                raise TritonError("core_clip | batch | Triton is not ready", error_code="triton_unavailable")
        else:
            raise RuntimeError(f"core_clip | batch | invalid runtime: {runtime}")
        
        # Обработка батчей кадров
        n_frames = len(all_frames)
        embeddings_out = None
        embed_dim = None
        
        # Определяем размер батча
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        logger.info(f"core_clip | batch | processing {n_frames} frames in batches of {effective_batch_size}")
        
        with torch.no_grad() if runtime == "inprocess" else None:
            start = 0
            while start < n_frames:
                batch_end = min(start + effective_batch_size, n_frames)
                batch_frames = all_frames[start:batch_end]
                
                # Извлекаем кадры из батча
                batch_images = [frame for _, _, frame in batch_frames]
                
                # Preprocess
                if runtime == "triton":
                    triton_datatype = config.get("triton_image_datatype", "UINT8")
                    image_size = 224  # Default for CLIP
                    if triton_datatype == "UINT8":
                        inp = _clip_resize_batch_uint8(batch_images, image_size=image_size)
                    else:
                        inp = _clip_preprocess_batch(batch_images, image_size=image_size)
                    
                    # Inference через Triton
                    triton_model_name = config.get("triton_image_model_name", "clip_image_224")
                    triton_model_version = config.get("triton_image_model_version")
                    triton_input_name = config.get("triton_image_input_name", "INPUT__0")
                    triton_output_name = config.get("triton_image_output_name", "OUTPUT__0")
                    
                    emb_np = _triton_infer_embeddings(
                        client=client,
                        model_name=triton_model_name,
                        model_version=triton_model_version,
                        input_name=triton_input_name,
                        input_tensor=inp,
                        output_name=triton_output_name,
                        datatype=triton_datatype,
                    )
                else:
                    # Inprocess inference
                    assert model is not None and preprocess is not None
                    imgs = [preprocess(Image.fromarray(fr)) for fr in batch_images]
                    batch_tensor = torch.stack(imgs).to(device)
                    emb = model.encode_image(batch_tensor)
                    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
                    emb_np = emb.detach().cpu().numpy().astype(np.float32)
                
                # Инициализируем выходной массив
                if embeddings_out is None:
                    embed_dim = int(emb_np.shape[1])
                    embeddings_out = np.zeros((n_frames, embed_dim), dtype=np.float32)
                
                # Сохраняем embeddings
                embeddings_out[start:batch_end] = emb_np
                
                if start % (effective_batch_size * 10) == 0:
                    logger.info(f"core_clip | batch | processed {batch_end}/{n_frames} frames")
                
                start = batch_end
        
        # ============================================================
        # ОПТИМИЗАЦИЯ: Освобождение памяти после обработки image embeddings
        # ============================================================
        if runtime == "inprocess" and model is not None:
            # Освобождаем память GPU после обработки всех кадров
            del model
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # ============================================================
        # ОПТИМИЗАЦИЯ: Кеширование text embeddings
        # Text embeddings одинаковы для всех видео, поэтому используем кеш
        # ============================================================
        # Этап 3: Вычисление text embeddings (один раз для всех видео)
        logger.info("core_clip | batch | computing text embeddings")
        
        places365_prompts = _load_places365_prompts_from_bundle()
        all_prompts: List[str] = []
        all_prompts.extend(SHOT_QUALITY_PROMPTS)
        all_prompts.extend(SCENE_AESTHETIC_PROMPTS)
        all_prompts.extend(SCENE_LUXURY_PROMPTS)
        all_prompts.extend(SCENE_ATMOSPHERE_PROMPTS)
        all_prompts.extend(CUT_DETECTION_TRANSITION_PROMPTS)
        all_prompts.extend(POPULARITY_TOPIC_PROMPTS)
        all_prompts.extend(places365_prompts)
        
        # Предвычисляем размеры групп для оптимизации разделения
        n_shot = len(SHOT_QUALITY_PROMPTS)
        n_aes = len(SCENE_AESTHETIC_PROMPTS)
        n_lux = len(SCENE_LUXURY_PROMPTS)
        n_atm = len(SCENE_ATMOSPHERE_PROMPTS)
        n_cut = len(CUT_DETECTION_TRANSITION_PROMPTS)
        n_pop = len(POPULARITY_TOPIC_PROMPTS)
        n_p365 = len(places365_prompts)
        
        # Предвычисляем индексы для разделения (оптимизация)
        idx_shot_end = n_shot
        idx_aes_end = idx_shot_end + n_aes
        idx_lux_end = idx_aes_end + n_lux
        idx_atm_end = idx_lux_end + n_atm
        idx_cut_end = idx_atm_end + n_cut
        idx_pop_end = idx_cut_end + n_pop
        
        # Пытаемся загрузить из кеша (если доступен)
        text_embeddings = None
        cache_path = None
        if runtime == "inprocess":
            # Импортируем функции кеширования из main.py
            from core.model_process.core_clip.main import (
                _get_text_embeddings_cache_key,
                _get_text_embeddings_cache_path,
                _load_cached_text_embeddings,
                _save_text_embeddings_cache,
                PROMPTS_VERSION,
            )
            
            model_name = config.get("model_name", "ViT-B/32")
            # Определяем model_size для кеша
            model_size = "224"  # Default для CLIP
            if "336" in model_name:
                model_size = "336"
            elif "448" in model_name:
                model_size = "448"
            
            cache_key = _get_text_embeddings_cache_key(
                all_prompts=all_prompts,
                triton_model_name=model_name,
                triton_model_version=None,
                prompts_version=PROMPTS_VERSION,
                model_size=model_size,
            )
            cache_path = _get_text_embeddings_cache_path(cache_key, model_size=model_size)
            
            if cache_path:
                cached_emb = _load_cached_text_embeddings(cache_path)
                if cached_emb is not None and cached_emb.shape[0] == len(all_prompts):
                    text_embeddings = cached_emb
                    logger.info(f"core_clip | batch | loaded {len(all_prompts)} text embeddings from cache")
        
        # Вычисляем text embeddings если не загрузили из кеша
        if text_embeddings is None:
            if runtime == "inprocess":
                # Перезагружаем модель для text embeddings (если освободили память)
                if model is None:
                    model, preprocess, device = init_clip(config.get("model_name", "ViT-B/32"), preferred_device="auto")
                
                text_embeddings = compute_text_embeddings(model, device, all_prompts)
                
                # Сохраняем в кеш
                if cache_path:
                    try:
                        _save_text_embeddings_cache(
                            cache_path=cache_path,
                            embeddings=text_embeddings,
                            prompts_count=len(all_prompts),
                            model_size=model_size,
                            prompts_version=PROMPTS_VERSION,
                        )
                    except Exception as e:
                        logger.warning(f"core_clip | batch | failed to save text embeddings cache: {e}")
                
                # Освобождаем память после text embeddings
                del model
                import gc
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            elif runtime == "triton":
                # TODO: Text embeddings через Triton
                logger.warning("core_clip | batch | text embeddings via Triton not yet implemented")
                text_embeddings = np.zeros((len(all_prompts), embed_dim), dtype=np.float32)
        
        # ============================================================
        # ОПТИМИЗАЦИЯ: Векторизованное разделение text embeddings
        # Используем предвычисленные индексы для более эффективного slicing
        # ============================================================
        if text_embeddings is not None:
            shot_quality_text_embeddings = text_embeddings[0:idx_shot_end]
            scene_aesthetic_text_embeddings = text_embeddings[idx_shot_end:idx_aes_end]
            scene_luxury_text_embeddings = text_embeddings[idx_aes_end:idx_lux_end]
            scene_atmosphere_text_embeddings = text_embeddings[idx_lux_end:idx_atm_end]
            cut_detection_transition_text_embeddings = text_embeddings[idx_atm_end:idx_cut_end]
            popularity_topic_text_embeddings = text_embeddings[idx_cut_end:idx_pop_end]
            places365_text_embeddings = text_embeddings[idx_pop_end:]
        else:
            # Fallback: нулевые embeddings
            shot_quality_text_embeddings = np.zeros((n_shot, embed_dim), dtype=np.float32)
            scene_aesthetic_text_embeddings = np.zeros((n_aes, embed_dim), dtype=np.float32)
            scene_luxury_text_embeddings = np.zeros((n_lux, embed_dim), dtype=np.float32)
            scene_atmosphere_text_embeddings = np.zeros((n_atm, embed_dim), dtype=np.float32)
            cut_detection_transition_text_embeddings = np.zeros((n_cut, embed_dim), dtype=np.float32)
            popularity_topic_text_embeddings = np.zeros((n_pop, embed_dim), dtype=np.float32)
            places365_text_embeddings = np.zeros((n_p365, embed_dim), dtype=np.float32)
        
        # Этап 4: Распределение результатов обратно по видео
        logger.info("core_clip | batch | distributing results back to videos")
        
        results = []
        for video_info in frames_by_video:
            video_idx = video_info["video_idx"]
            video_ctx = video_contexts[video_idx]
            
            if video_info["status"] != "ok":
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": video_info["status"],
                    "error": video_info.get("error"),
                })
                continue
            
            # Извлекаем embeddings для этого видео используя сохраненные индексы
            video_frame_indices = video_info["frame_indices"]
            frame_start_idx = video_info.get("frame_start_idx", 0)
            frame_end_idx = video_info.get("frame_end_idx", len(embeddings_out))
            
            if frame_start_idx >= frame_end_idx or frame_start_idx >= len(embeddings_out):
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "empty",
                    "empty_reason": "no frames processed",
                })
                continue
            
            # Извлекаем embeddings для этого видео
            video_embeddings = embeddings_out[frame_start_idx:frame_end_idx]
            video_times_s = video_info["times_s"]
            
            # Проверяем соответствие размеров
            if len(video_embeddings) != len(video_frame_indices) or len(video_embeddings) != len(video_times_s):
                logger.warning(
                    f"core_clip | batch | video {video_ctx.video_id} size mismatch: "
                    f"embeddings={len(video_embeddings)}, indices={len(video_frame_indices)}, times={len(video_times_s)}"
                )
                # Используем минимальный размер
                min_size = min(len(video_embeddings), len(video_frame_indices), len(video_times_s))
                video_embeddings = video_embeddings[:min_size]
                video_frame_indices = video_frame_indices[:min_size]
                video_times_s = video_times_s[:min_size]
            
            # Сохраняем результаты в per-video rs_path
            component_dir = video_ctx.get_component_rs_path("core_clip")
            npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
            
            # Подготовка метаданных
            metadata = video_ctx.load_metadata()
            save_metadata = {
                "producer": NAME,
                "producer_version": VERSION,
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "platform_id": video_ctx.platform_id or metadata.get("platform_id"),
                "video_id": video_ctx.video_id,
                "run_id": video_ctx.run_id or metadata.get("run_id"),
                "sampling_policy_version": video_ctx.sampling_policy_version or metadata.get("sampling_policy_version"),
                "config_hash": video_ctx.config_hash or metadata.get("config_hash"),
                "dataprocessor_version": video_ctx.dataprocessor_version or metadata.get("dataprocessor_version") or "unknown",
                "status": "ok",
                "empty_reason": None,
                "total_frames": metadata.get("total_frames"),
                "processed_frames": len(video_frame_indices),
            }
            
            # Models used
            models_used = []
            if runtime == "inprocess":
                models_used.append(
                    model_used(
                        model_name=config.get("model_name", "ViT-B/32"),
                        model_version=config.get("model_version", "unknown"),
                        weights_digest=config.get("weights_digest", "unknown"),
                        runtime="local",
                        engine=config.get("engine", "torch"),
                        precision=config.get("precision", "fp32"),
                        device=device,
                    )
                )
            elif runtime == "triton":
                models_used.append(
                    model_used(
                        model_name=config.get("triton_image_model_name", "clip_image_224"),
                        model_version=config.get("triton_image_model_version", "1"),
                        weights_digest="unknown",
                        runtime="triton-gpu",
                        engine="triton",
                        precision="fp32",
                        device="cuda",
                    )
                )
            save_metadata["models_used"] = models_used
            save_metadata = apply_models_meta(save_metadata, models_used=models_used)
            
            # Сохранение NPZ
            npz_dict = {
                "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
                "times_s": video_times_s,
                "frame_embeddings": video_embeddings,
                "shot_quality_prompts": np.asarray(SHOT_QUALITY_PROMPTS, dtype=object),
                "shot_quality_text_embeddings": shot_quality_text_embeddings,
                "scene_aesthetic_prompts": np.asarray(SCENE_AESTHETIC_PROMPTS, dtype=object),
                "scene_aesthetic_text_embeddings": scene_aesthetic_text_embeddings,
                "scene_luxury_prompts": np.asarray(SCENE_LUXURY_PROMPTS, dtype=object),
                "scene_luxury_text_embeddings": scene_luxury_text_embeddings,
                "scene_atmosphere_prompts": np.asarray(SCENE_ATMOSPHERE_PROMPTS, dtype=object),
                "scene_atmosphere_text_embeddings": scene_atmosphere_text_embeddings,
                "cut_detection_transition_prompts": np.asarray(CUT_DETECTION_TRANSITION_PROMPTS, dtype=object),
                "cut_detection_transition_text_embeddings": cut_detection_transition_text_embeddings,
                "popularity_topic_prompts": np.asarray(POPULARITY_TOPIC_PROMPTS, dtype=object),
                "popularity_topic_text_embeddings": popularity_topic_text_embeddings,
                "meta": np.asarray(save_metadata, dtype=object),
            }
            
            # Добавляем places365 если есть
            if places365_prompts:
                npz_dict["places365_prompts"] = np.asarray(places365_prompts, dtype=object)
                npz_dict["places365_text_embeddings"] = places365_text_embeddings
            
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
                    f"core_clip | batch | saved artifact failed validation: "
                    + "; ".join([f"{i.level}:{i.message}" for i in issues])
                )
            
            results.append({
                "video_id": video_ctx.video_id,
                "status": "ok",
                "saved_path": npz_path,
            })
        
        # Закрываем FrameManager для всех видео
        for video_info in frames_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        
        duration = time.perf_counter() - start_time
        logger.info(
            f"core_clip | batch | completed in {duration:.2f}s "
            f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
        )
        
        return results
        
    except Exception as e:
        logger.exception(f"core_clip | batch | error: {e}")
        # Закрываем FrameManager в случае ошибки
        for video_info in frames_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        
        # Возвращаем ошибки для всех видео
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": str(e),
            }
            for ctx in video_contexts
        ]
    finally:
        # Очистка ресурсов
        if model is not None:
            del model
            torch.cuda.empty_cache()
        if client is not None:
            try:
                # Triton client cleanup if needed
                pass
            except Exception:
                pass

