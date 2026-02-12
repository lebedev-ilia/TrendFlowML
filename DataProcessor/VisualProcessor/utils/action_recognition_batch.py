"""
Batch processing утилита для action_recognition модуля.

Реализует гибридный подход для GPU batching:
- Сбор клипов из всех видео
- Батчинг с лимитом размера
- Распределение результатов обратно по видео
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from utils.video_context import VideoContext
from utils.frame_manager import FrameManager

logger = logging.getLogger(__name__)


def _load_slowfast_model(
    model_name: str,
    device: str,
    rs_path: Optional[str] = None
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Загружает модель SlowFast через ModelManager.
    
    Returns:
        (model, embedding_proj) - модель и проекция эмбеддингов
    """
    from dp_models.manager import get_global_model_manager
    from dp_models.errors import ModelManagerError
    
    _mm = get_global_model_manager()
    try:
        spec = _mm.get_spec(model_name=model_name)
        device_resolved, precision, runtime, engine, weights_digest, resolved_artifacts = _mm.resolve(spec)
    except ModelManagerError as e:
        raise RuntimeError(f"action_recognition | batch | ModelManager resolve failed: {e}") from e
    
    if runtime != "inprocess":
        raise RuntimeError(f"action_recognition | batch | Unsupported runtime: {runtime}")
    if str(engine).lower() not in ("torch", "pytorch"):
        raise RuntimeError(f"action_recognition | batch | Unsupported engine: {engine}")
    
    # Используем device из параметров, если задан явно
    if device:
        device_resolved = device
    
    # Находим checkpoint
    ckpt_rel = None
    for a in spec.local_artifacts:
        if str(a.kind) == "file":
            ckpt_rel = str(a.path)
            break
    if not ckpt_rel:
        raise RuntimeError(f"action_recognition | batch | No checkpoint file declared in model spec")
    ckpt_path = resolved_artifacts.get(ckpt_rel) or ckpt_rel
    
    # Загружаем модель из pytorchvideo (как в ex.py)
    try:
        from pytorchvideo.models.hub import slowfast_r50
    except ImportError as e:
        raise ImportError(
            f"Cannot import slowfast_r50 from pytorchvideo.models.hub: {e}. "
            "Please install pytorchvideo: pip install pytorchvideo"
        ) from e
    
    # Загружаем модель без pretrained (pytorchvideo использует pretrained=False)
    model = slowfast_r50(pretrained=False)
    
    # Загружаем state_dict
    state: Any = None
    if ckpt_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as _load_safetensors
        except Exception as e:
            raise RuntimeError(f"action_recognition | batch | safetensors required: {e}") from e
        state = _load_safetensors(ckpt_path, device="cpu")
    else:
        try:
            # ВАЖНО: pytorchvideo использует weights_only=False для загрузки полных моделей
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"action_recognition | batch | Failed to load checkpoint: {e}") from e
    
        # В pytorchvideo веса лежат в "model_state" (как в ex.py)
        if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
    
    if not isinstance(state, dict):
        raise RuntimeError(f"action_recognition | batch | Checkpoint does not contain a state_dict dict")
    
    # Cleanup prefix
    if any(str(k).startswith("module.") for k in state.keys()):
        cleaned = {}
        for k, v in state.items():
            ks = str(k)
            if ks.startswith("module."):
                ks = ks[len("module."):]
            cleaned[ks] = v
        state = cleaned
    
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"action_recognition | batch | SlowFast state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )
    
    model = model.to(device_resolved).eval()
    
    # Проекция эмбеддингов
    raw_embedding_dim = 2048
    embedding_dim = 256
    embedding_proj = torch.nn.Linear(raw_embedding_dim, embedding_dim).to(device_resolved)
    torch.nn.init.xavier_uniform_(embedding_proj.weight)
    torch.nn.init.zeros_(embedding_proj.bias)
    embedding_proj.eval()
    
    return model, embedding_proj


def _prepare_slow_fast(batch: torch.Tensor, alpha: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Подготовка slow и fast путей для SlowFast (как в ex.py pack_pathways).
    
    SlowFast использует alpha=4 (T_fast / T_slow).
    Slow путь: каждый alpha-й кадр (frames[:, ::alpha, :, :])
    Fast путь: все кадры (frames)
    
    Args:
        batch: [B, C, T, H, W] - батч клипов
        alpha: коэффициент между fast и slow путями (по умолчанию 4)
    
    Returns:
        (slow, fast) - тензоры для slow и fast путей
    """
    if batch.dim() != 5:
        raise ValueError(f"Expected batch.dim()==5, got {batch.dim()}")
    B, C, T, H, W = batch.shape
    
    # Slow путь: каждый alpha-й кадр (как в ex.py)
    slow = batch[:, :, ::alpha, :, :]
    
    # Fast путь: все кадры
    fast = batch
    
    return slow, fast


def _preprocess_clip(clip: List[np.ndarray], mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Преобразует clip в Tensor [C, T, H, W] float32."""
    processed = []
    for frame in clip:
        frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame_float = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_float - mean) / std
        frame_chw = np.transpose(frame_norm, (2, 0, 1))
        processed.append(frame_chw)
    clip_arr = np.stack(processed, axis=1)  # C,T,H,W
    return torch.from_numpy(clip_arr).float()


def _extract_features(
    model: torch.nn.Module,
    slow: torch.Tensor,
    fast: torch.Tensor,
    raw_embedding_dim: int = 2048
) -> torch.Tensor:
    """Извлекает признаки через SlowFast."""
    with torch.no_grad():
        # pytorchvideo ожидает список [slow, fast] (как в ex.py)
        out = model([slow, fast])
    
    if torch.is_tensor(out):
        feat = out
    elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
        feat = out[0]
    else:
        raise RuntimeError(f"Unexpected model output type: {type(out)}")
    
    # Уменьшаем пространственно-временные размерности
    if feat.dim() == 5:
        feat = feat.mean(dim=[2, 3, 4])
    elif feat.dim() == 4:
        feat = feat.mean(dim=[2, 3])
    elif feat.dim() == 3:
        feat = feat.mean(dim=2)
    
    feat = feat.view(feat.size(0), -1)  # B, D
    
    # Выравнивание по raw_embedding_dim
    if feat.shape[1] > raw_embedding_dim:
        feat = feat[:, :raw_embedding_dim]
    elif feat.shape[1] < raw_embedding_dim:
        pad = torch.zeros((feat.shape[0], raw_embedding_dim - feat.shape[1]), device=feat.device)
        feat = torch.cat([feat, pad], dim=1)
    
    return feat


def process_action_recognition_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 8,
    clip_len: int = 16,
    stride: int = 8,
    alpha: int = 4,
    embedding_dim: int = 256,
    model_name: str = "slowfast_r50_action_recognition",
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Batch processing для action_recognition с гибридным подходом.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация action_recognition
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для inference
        clip_len: Длина клипа в кадрах
        stride: Шаг для создания клипов
        embedding_dim: Размерность эмбеддингов
        model_name: ModelManager spec name
        device: Устройство для обработки
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"action_recognition | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch}, batch_size={batch_size}, clip_len={clip_len}, alpha={alpha})"
    )
    
    start_time = time.perf_counter()
    
    # Нормализация (ImageNet-like)
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    
    # Загружаем модель один раз для всех видео
    model, embedding_proj = _load_slowfast_model(model_name, device)
    raw_embedding_dim = 2048
    
    # Этап 1: Сбор всех клипов из всех видео
    clips_by_video: List[Dict[str, Any]] = []
    all_clips: List[Tuple[int, int, List[np.ndarray], List[int], int]] = []  # (video_idx, track_id, clip_frames, clip_indices, center_idx)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            
            # Загружаем detections.npz
            try:
                from utils.results_store import ResultsStore
                rs = ResultsStore(video_ctx.rs_path)
                detections_path = rs.get_component_path("core_object_detections", "detections.npz")
                if not detections_path or not os.path.exists(detections_path):
                    raise RuntimeError("core_object_detections/detections.npz not found")
                
                detections_data = np.load(detections_path, allow_pickle=True)
                tracks_list = detections_data.get("tracks_list")
                tracks_list_ids = detections_data.get("tracks_list_ids")
                
                if tracks_list is None or tracks_list_ids is None:
                    raise RuntimeError("detections.npz missing tracks_list or tracks_list_ids")
                
                if len(tracks_list) != len(tracks_list_ids):
                    raise RuntimeError(f"tracks_list and tracks_list_ids length mismatch: {len(tracks_list)} != {len(tracks_list_ids)}")
                
            except Exception as e:
                logger.error(f"action_recognition | batch | video {video_ctx.video_id} failed to load detections: {e}")
                clips_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "tracks": {},
                    "frame_manager": None,
                    "status": "error",
                    "error": str(e),
                })
                continue
            
            if len(tracks_list) == 0:
                logger.warning(f"action_recognition | batch | video {video_ctx.video_id} has no tracks")
                clips_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "tracks": {},
                    "frame_manager": None,
                    "status": "empty",
                    "empty_reason": "no_tracks",
                })
                continue
            
            # Создаем FrameManager
            frame_manager = FrameManager(
                frames_dir=video_ctx.frames_dir,
                chunk_size=metadata.get("chunk_size", 32),
                cache_size=metadata.get("cache_size", 2),
            )
            
            # Получаем frame_indices из metadata
            frame_indices_set = set()
            try:
                frame_indices_raw = metadata.get("action_recognition", {}).get("frame_indices")
                if frame_indices_raw:
                    frame_indices_set = set(int(x) for x in frame_indices_raw)
            except Exception:
                pass
            
            # Обрабатываем каждый трек
            video_tracks: Dict[int, Dict[str, Any]] = {}
            video_clip_start_idx = len(all_clips)
            
            for i in range(len(tracks_list_ids)):
                track_id = int(tracks_list_ids[i])
                track_frames = tracks_list[i]
                
                if isinstance(track_frames, np.ndarray):
                    frame_indices_for_track = track_frames.tolist()
                elif isinstance(track_frames, (list, tuple)):
                    frame_indices_for_track = list(track_frames)
                else:
                    continue
                
                # Фильтруем только те индексы, которые есть в frame_indices_set (если задан)
                if frame_indices_set:
                    filtered_indices = [int(idx) for idx in frame_indices_for_track if int(idx) in frame_indices_set]
                else:
                    filtered_indices = [int(idx) for idx in frame_indices_for_track]
                
                if not filtered_indices:
                    continue
                
                filtered_indices = sorted(filtered_indices)
                
                # Загружаем кадры
                frames = []
                for idx in filtered_indices:
                    try:
                        frame = frame_manager.get(idx)
                        if frame is None:
                            continue
                        if frame.ndim == 2:
                            frame = np.stack([frame] * 3, axis=-1)
                        if frame.shape[-1] == 4:
                            frame = frame[..., :3]
                        frames.append(frame.astype(np.uint8))
                    except Exception as e:
                        logger.warning(f"action_recognition | batch | video {video_ctx.video_id} track {track_id} failed to load frame {idx}: {e}")
                        continue
                
                if len(frames) < clip_len:
                    # Padding
                    pad = clip_len - len(frames)
                    frames = frames + [frames[-1]] * pad if frames else []
                    filtered_indices = filtered_indices + [filtered_indices[-1]] * pad if filtered_indices else []
                
                # Создаем клипы
                clips = []
                for start in range(0, len(frames) - clip_len + 1, stride):
                    clip_frames = frames[start:start + clip_len]
                    clip_indices = filtered_indices[start:start + clip_len]
                    center_idx = clip_indices[len(clip_indices) // 2]
                    clips.append((clip_frames, clip_indices, center_idx))
                
                if not clips:
                    # Создаем один клип из последних clip_len кадров
                    clip_frames = frames[-clip_len:]
                    clip_indices = filtered_indices[-clip_len:]
                    center_idx = clip_indices[len(clip_indices) // 2]
                    clips.append((clip_frames, clip_indices, center_idx))
                
                # Сохраняем клипы в общий батч
                for clip_frames, clip_indices, center_idx in clips:
                    all_clips.append((video_idx, track_id, clip_frames, clip_indices, center_idx))
                
                video_tracks[track_id] = {
                    "frame_indices": filtered_indices,
                    "num_clips": len(clips),
                }
            
            video_clip_end_idx = len(all_clips)
            
            clips_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "tracks": video_tracks,
                "frame_manager": frame_manager,
                "clip_start_idx": video_clip_start_idx,
                "clip_end_idx": video_clip_end_idx,
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"action_recognition | batch | video {video_ctx.video_id} failed to prepare: {e}")
            clips_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "tracks": {},
                "frame_manager": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_clips:
        logger.error("action_recognition | batch | no clips collected from any video")
        # Закрываем FrameManager
        for video_info in clips_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": "no clips collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"action_recognition | batch | collected {len(all_clips)} clips from {len(clips_by_video)} videos")
    
    # Этап 2: Батчинг и inference
    try:
        n_clips = len(all_clips)
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        logger.info(f"action_recognition | batch | processing {n_clips} clips in batches of {effective_batch_size}")
        
        # Извлекаем клипы
        clip_data = [(clip_frames, clip_indices, center_idx) for _, _, clip_frames, clip_indices, center_idx in all_clips]
        
        # Inference батчами
        all_normed_embeddings = []
        all_raw_embeddings = []
        
        start = 0
        while start < n_clips:
            batch_end = min(start + effective_batch_size, n_clips)
            batch_clips = clip_data[start:batch_end]
            
            # Препроцессинг
            tensors_cpu = [_preprocess_clip(c[0], mean, std) for c in batch_clips]
            batch = torch.stack(tensors_cpu, dim=0).to(device)  # [B,C,T,H,W]
            
            # SlowFast inference
            slow, fast = _prepare_slow_fast(batch, alpha=alpha)
            feat = _extract_features(model, slow, fast, raw_embedding_dim)  # [B, raw_dim]
            
            # Проекция + нормализация
            with torch.no_grad():
                proj = embedding_proj(feat)  # [B, embedding_dim]
                proj = F.normalize(proj, p=2, dim=1)
            
            all_normed_embeddings.append(proj.cpu().numpy().astype(np.float32))
            all_raw_embeddings.append(feat.cpu().numpy().astype(np.float32))
            
            # Освобождение памяти
            del batch, slow, fast, feat, proj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if start % (effective_batch_size * 10) == 0:
                logger.info(f"action_recognition | batch | processed {batch_end}/{n_clips} clips")
            
            start = batch_end
        
        # Конкатенация
        normed_embeddings = np.concatenate(all_normed_embeddings, axis=0) if all_normed_embeddings else np.zeros((0, embedding_dim), dtype=np.float32)
        raw_embeddings = np.concatenate(all_raw_embeddings, axis=0) if all_raw_embeddings else np.zeros((0, raw_embedding_dim), dtype=np.float32)
        
        # Этап 3: Распределение результатов обратно по видео
        logger.info("action_recognition | batch | distributing results back to videos")
        
        results = []
        for video_info in clips_by_video:
            video_idx = video_info["video_idx"]
            video_ctx = video_contexts[video_idx]
            
            if video_info["status"] != "ok":
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": video_info["status"],
                    "error": video_info.get("error"),
                })
                continue
            
            # Извлекаем результаты для этого видео
            clip_start_idx = video_info.get("clip_start_idx", 0)
            clip_end_idx = video_info.get("clip_end_idx", len(all_clips))
            video_clips = all_clips[clip_start_idx:clip_end_idx]
            video_normed = normed_embeddings[clip_start_idx:clip_end_idx]
            video_raw = raw_embeddings[clip_start_idx:clip_end_idx]
            
            # Группируем по трекам
            per_track_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
            per_track_raw: Dict[int, List[np.ndarray]] = defaultdict(list)
            per_track_centers: Dict[int, List[int]] = defaultdict(list)
            per_track_clip_indices: Dict[int, List[List[int]]] = defaultdict(list)
            
            for clip_idx, (v_idx, track_id, clip_frames, clip_indices, center_idx) in enumerate(video_clips):
                if v_idx != video_idx:
                    continue
                local_clip_idx = clip_idx - clip_start_idx
                if local_clip_idx < 0 or local_clip_idx >= len(video_normed):
                    continue
                per_track_embeddings[track_id].append(video_normed[local_clip_idx])
                per_track_raw[track_id].append(video_raw[local_clip_idx])
                per_track_centers[track_id].append(center_idx)
                per_track_clip_indices[track_id].append(clip_indices)
            
            # Агрегируем результаты по трекам
            track_results: Dict[int, Dict[str, Any]] = {}
            for track_id in per_track_embeddings:
                embs_arr = np.stack(per_track_embeddings[track_id], axis=0) if per_track_embeddings[track_id] else np.zeros((0, embedding_dim), dtype=np.float32)
                raw_arr = np.stack(per_track_raw[track_id], axis=0) if per_track_raw[track_id] else np.zeros((0, raw_embedding_dim), dtype=np.float32)
                
                # Вычисляем метрики (упрощенная версия)
                n = len(embs_arr)
                if n > 0:
                    raw_norms = np.linalg.norm(raw_arr, axis=1)
                    mean_norm_raw = float(np.mean(raw_norms))
                    std_norm_raw = float(np.std(raw_norms))
                    
                    if n > 1:
                        diffs = [np.linalg.norm(embs_arr[i] - embs_arr[i - 1]) for i in range(1, n)]
                        max_jump = float(np.max(diffs))
                        mean_jump = float(np.mean(diffs))
                    else:
                        max_jump = float("nan")
                        mean_jump = float("nan")
                    
                    # Упрощенная стабильность (без PCA/KMeans для batch)
                    stability = float("nan")
                    switches = 0
                else:
                    mean_norm_raw = float("nan")
                    std_norm_raw = float("nan")
                    max_jump = float("nan")
                    mean_jump = float("nan")
                    stability = float("nan")
                    switches = 0
                
                track_results[track_id] = {
                    "embedding_normed_256d": embs_arr,
                    "mean_embedding_norm_raw": mean_norm_raw,
                    "std_embedding_norm_raw": std_norm_raw,
                    "max_temporal_jump": max_jump,
                    "mean_temporal_jump": mean_jump,
                    "stability": stability,
                    "num_switches": switches,
                    "num_clips": n,
                    "track_frame_count": len(video_info["tracks"].get(track_id, {}).get("frame_indices", [])),
                    "clip_center_frame_indices": per_track_centers.get(track_id, []),
                    "clip_frame_indices": per_track_clip_indices.get(track_id, []),
                    "embedding_dim": embedding_dim,
                }
            
            results.append({
                "video_id": video_ctx.video_id,
                "status": "ok" if track_results else "empty",
                "empty_reason": "no_tracks" if not track_results else None,
                "results": track_results,
            })
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"action_recognition | batch | completed in {elapsed:.2f}s")
        
        return results
        
    finally:
        # Закрываем FrameManager
        for video_info in clips_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        
        # Освобождаем модель
        if model is not None:
            del model, embedding_proj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

