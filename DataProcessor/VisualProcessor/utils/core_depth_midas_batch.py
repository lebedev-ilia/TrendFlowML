"""
Batch processing utilities for core_depth_midas component.

Stage 3: GPU batching для core_depth_midas с гибридным подходом:
- Сбор кадров из всех видео
- Группировка в батчи по max_frames_per_batch
- Последовательная обработка батчей через Triton
- Распределение результатов обратно по видео
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import cv2
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz

# Import core_depth_midas functions
_core_depth_midas_path = _visual_processor_path / "core" / "model_process" / "core_depth_midas"
sys.path.insert(0, str(_core_depth_midas_path.parent.parent.parent))

logger = get_logger("VisualProcessor.core_depth_midas_batch")

# Import from core_depth_midas/main.py
_core_depth_midas_main = _core_depth_midas_path / "main.py"
if _core_depth_midas_main.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("core_depth_midas_main", str(_core_depth_midas_main))
    if spec and spec.loader:
        core_depth_midas_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_depth_midas_module)
        
        # Import functions
        _preset_to_input_size = getattr(core_depth_midas_module, "_preset_to_input_size", None)
        _prep_batch_rgb_uint8 = getattr(core_depth_midas_module, "_prep_batch_rgb_uint8", None)
        _require_union_times_s = getattr(core_depth_midas_module, "_require_union_times_s", None)
        NAME = getattr(core_depth_midas_module, "NAME", "core_depth_midas")
        VERSION = getattr(core_depth_midas_module, "VERSION", "2.0")
        SCHEMA_VERSION = getattr(core_depth_midas_module, "SCHEMA_VERSION", "core_depth_midas_npz_v1")
        ARTIFACT_FILENAME = getattr(core_depth_midas_module, "ARTIFACT_FILENAME", "depth.npz")
    else:
        raise ImportError("Failed to load core_depth_midas module")
else:
    raise ImportError(f"core_depth_midas/main.py not found at {_core_depth_midas_main}")


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


def _get_frame_indices(metadata: Dict[str, Any], component_name: str) -> List[int]:
    """Получить frame_indices из метаданных."""
    block = metadata.get(component_name)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{component_name} | metadata missing '{component_name}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices_raw = block.get("frame_indices")
    if not isinstance(frame_indices_raw, list) or not frame_indices_raw:
        raise RuntimeError(f"{component_name} | metadata '{component_name}.frame_indices' is empty/invalid.")
    return [int(x) for x in frame_indices_raw]


def process_core_depth_midas_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Batch processing для core_depth_midas с гибридным подходом.
    
    Stage 3: GPU batching для core_depth_midas.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация core_depth_midas
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для inference
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"core_depth_midas | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch}, batch_size={batch_size})"
    )
    
    start_time = time.perf_counter()
    
    # Инициализация Triton клиента
    # Audit v3: prefer ModelManager spec (no-network, reproducible) over explicit triton_* args.
    triton_http_url = ""
    triton_model_name = config.get("triton_model_name", "midas_256")
    triton_model_version = config.get("triton_model_version", "1")
    triton_input_name = config.get("triton_input_name", "INPUT__0")
    triton_output_name = config.get("triton_output_name", "OUTPUT__0")
    triton_datatype = config.get("triton_datatype", "UINT8")
    triton_preprocess_preset = config.get("triton_preprocess_preset", "midas_256")

    spec_name = config.get("triton_model_spec")
    if spec_name:
        # core_depth_midas main module is already loaded above; reuse its resolver to avoid duplication.
        _load_spec = getattr(core_depth_midas_module, "_load_triton_spec_via_model_manager", None)
        if not callable(_load_spec):
            raise RuntimeError("core_depth_midas | batch | triton_model_spec provided but ModelManager resolver is unavailable")
        mm_entry = _load_spec(str(spec_name))
        rp = mm_entry.get("rp") if isinstance(mm_entry, dict) else None
        if isinstance(rp, dict):
            triton_http_url = str(rp.get("triton_http_url") or "").strip()
            triton_model_name = str(rp.get("triton_model_name") or triton_model_name)
            triton_model_version = str(rp.get("triton_model_version") or triton_model_version)
            triton_input_name = str(rp.get("triton_input_name") or triton_input_name)
            triton_output_name = str(rp.get("triton_output_name") or triton_output_name)
            triton_datatype = str(rp.get("triton_input_datatype") or rp.get("triton_datatype") or triton_datatype)
            triton_preprocess_preset = str(config.get("triton_preprocess_preset") or triton_preprocess_preset)
    if not triton_http_url:
        triton_http_url = str(config.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL") or "").strip()
    if not triton_http_url:
        raise RuntimeError("core_depth_midas | batch | runtime=triton requires TRITON_HTTP_URL (env) or config.triton_http_url or config.triton_model_spec")
    
    from dp_triton import TritonHttpClient, TritonError
    
    client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=240.0)
    if not client.ready():
        raise TritonError("core_depth_midas | batch | Triton is not ready", error_code="triton_unavailable")
    
    # Параметры Triton
    triton_model_name = config.get("triton_model_name")
    if not triton_model_name:
        raise RuntimeError("core_depth_midas | batch | triton_model_name is required")
    
    triton_model_version = config.get("triton_model_version")
    triton_input_name = config.get("triton_input_name", "INPUT__0")
    triton_output_name = config.get("triton_output_name", "OUTPUT__0")
    triton_datatype = config.get("triton_datatype", "UINT8")
    triton_preprocess_preset = config.get("triton_preprocess_preset", "midas_384")
    input_size = _preset_to_input_size(triton_preprocess_preset)
    
    # Параметры выхода
    out_width = int(config.get("out_width", 384))
    out_height = int(config.get("out_height", 384))
    out_size = (out_height, out_width)  # (H, W)
    frames_are_bgr = bool(config.get("frames_bgr", False))
    
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
                frame_indices = _get_frame_indices(metadata, NAME)
            except Exception as e:
                logger.error(f"core_depth_midas | batch | video {video_ctx.video_id} failed to get frame_indices: {e}")
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
                logger.warning(f"core_depth_midas | batch | video {video_ctx.video_id} has no frame_indices")
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
            times_s = _require_union_times_s(metadata, frame_indices)
            
            # Загружаем кадры и сохраняем маппинг
            video_frame_start_idx = len(all_frames)  # Начальный индекс в общем батче для этого видео
            for frame_idx in frame_indices:
                try:
                    frame = frame_manager.get(frame_idx)
                    # Сохраняем кадр в общий батч
                    all_frames.append((video_idx, frame_idx, frame))
                except Exception as e:
                    logger.warning(
                        f"core_depth_midas | batch | video {video_ctx.video_id} failed to load frame {frame_idx}: {e}"
                    )
                    continue
            
            # Сохраняем информацию о диапазоне индексов для этого видео
            video_frame_end_idx = len(all_frames)
            
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": frame_indices,
                "frame_manager": frame_manager,
                "times_s": times_s,
                "frame_start_idx": video_frame_start_idx,
                "frame_end_idx": video_frame_end_idx,
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"core_depth_midas | batch | video {video_ctx.video_id} failed to prepare: {e}")
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
        logger.error("core_depth_midas | batch | no frames collected from any video")
        # Закрываем FrameManager
        for video_info in frames_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": "no frames collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"core_depth_midas | batch | collected {len(all_frames)} frames from {len(frames_by_video)} videos")
    
    # Этап 2: Группировка в батчи и обработка
    try:
        n_frames = len(all_frames)
        depth_maps_out = None
        depth_mean_out = None
        depth_std_out = None
        depth_p05_out = None
        depth_p95_out = None
        
        # Определяем размер батча
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        logger.info(f"core_depth_midas | batch | processing {n_frames} frames in batches of {effective_batch_size}")
        
        start = 0
        while start < n_frames:
            batch_end = min(start + effective_batch_size, n_frames)
            batch_frames = all_frames[start:batch_end]
            
            # Извлекаем кадры из батча
            batch_images = [frame for _, _, frame in batch_frames]
            
            # Preprocess для Triton
            inp = _prep_batch_rgb_uint8(batch_images, input_size=input_size, frames_are_bgr=frames_are_bgr)
            
            # Inference через Triton
            try:
                res = client.infer(
                    model_name=str(triton_model_name),
                    model_version=str(triton_model_version) if triton_model_version else None,
                    input_name=str(triton_input_name),
                    input_tensor=inp,
                    output_name=str(triton_output_name),
                    datatype=str(triton_datatype),
                )
            except Exception as e:
                raise RuntimeError(f"core_depth_midas | batch | Triton infer failed: {e}") from e
            
            out = np.asarray(res.output, dtype=np.float32)
            # Expect (B,1,h,w) or (B,h,w)
            if out.ndim == 4 and out.shape[1] == 1:
                out = out[:, 0, :, :]
            if out.ndim != 3:
                raise RuntimeError(f"core_depth_midas | batch | Triton output has invalid shape: {out.shape}")
            if out.shape[0] != len(batch_images):
                raise RuntimeError(f"core_depth_midas | batch | Triton output batch mismatch: got {out.shape[0]} expected {len(batch_images)}")
            
            # Инициализируем выходные массивы
            if depth_maps_out is None:
                n_total = n_frames
                out_h, out_w = out_size
                depth_maps_out = np.full((n_total, out_h, out_w), np.nan, dtype=np.float32)
                depth_mean_out = np.full((n_total,), np.nan, dtype=np.float32)
                depth_std_out = np.full((n_total,), np.nan, dtype=np.float32)
                depth_p05_out = np.full((n_total,), np.nan, dtype=np.float32)
                depth_p95_out = np.full((n_total,), np.nan, dtype=np.float32)
            
            # ============================================================
            # ОПТИМИЗАЦИЯ: Векторизованная обработка depth maps в батче
            # Вместо цикла по каждому кадру, обрабатываем весь батч сразу
            # ============================================================
            batch_size_actual = out.shape[0]
            
            # Resize для каждого кадра (cv2 не поддерживает batch resize напрямую)
            # Но собираем все resized maps в один массив для векторизованных операций
            resized_maps = []
            for i_local in range(batch_size_actual):
                m = out[i_local]
                # Resize depth map до выходного размера
                dm = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                if not np.isfinite(dm).any():
                    frame_idx = batch_frames[i_local][1]
                    raise RuntimeError(f"core_depth_midas | batch | invalid depth map produced (NaN/empty) at frame_idx={frame_idx}")
                resized_maps.append(dm)
            
            # Конвертируем в numpy array для векторизованных операций
            resized_batch = np.array(resized_maps, dtype=np.float32)  # (B, out_h, out_w)
            
            # ============================================================
            # ОПТИМИЗАЦИЯ 1: Векторизованное вычисление mean и std для всего батча
            # ============================================================
            # Используем reshape для эффективного вычисления по последним двум осям
            depth_maps_flat = resized_batch.reshape(batch_size_actual, -1)  # (B, out_h * out_w)
            
            # Векторизованное вычисление mean и std для каждого кадра одновременно
            means = np.mean(depth_maps_flat, axis=1, dtype=np.float32)  # (B,)
            stds = np.std(depth_maps_flat, axis=1, dtype=np.float32)  # (B,)
            
            # ============================================================
            # ОПТИМИЗАЦИЯ 2: Более эффективное вычисление percentiles
            # Используем маскирование и сортировку только для валидных значений
            # ============================================================
            p05_values = np.full((batch_size_actual,), np.nan, dtype=np.float32)
            p95_values = np.full((batch_size_actual,), np.nan, dtype=np.float32)
            
            # Векторизованная проверка isfinite для всего батча
            finite_mask = np.isfinite(depth_maps_flat)  # (B, out_h * out_w)
            
            for i_local in range(batch_size_actual):
                # Используем предвычисленную маску для фильтрации
                vv = depth_maps_flat[i_local][finite_mask[i_local]]
                if vv.size:
                    # Используем более эффективный метод для percentiles
                    # np.percentile использует сортировку, но для больших массивов это все равно быстрее чем цикл
                    p05_values[i_local] = float(np.percentile(vv, 5))
                    p95_values[i_local] = float(np.percentile(vv, 95))
            
            # Сохраняем результаты в выходные массивы (векторизованно где возможно)
            global_indices = np.arange(start, start + batch_size_actual, dtype=np.int32)
            depth_maps_out[global_indices] = resized_batch
            depth_mean_out[global_indices] = means
            depth_std_out[global_indices] = stds
            depth_p05_out[global_indices] = p05_values
            depth_p95_out[global_indices] = p95_values
            
            if start % (effective_batch_size * 10) == 0:
                logger.info(f"core_depth_midas | batch | processed {batch_end}/{n_frames} frames")
            
            start = batch_end
        
        # Этап 3: Распределение результатов обратно по видео
        logger.info("core_depth_midas | batch | distributing results back to videos")
        
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
            
            # Извлекаем depth maps для этого видео используя сохраненные индексы
            video_frame_indices = video_info["frame_indices"]
            frame_start_idx = video_info.get("frame_start_idx", 0)
            frame_end_idx = video_info.get("frame_end_idx", len(depth_maps_out))
            
            if frame_start_idx >= frame_end_idx or frame_start_idx >= len(depth_maps_out):
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "empty",
                    "empty_reason": "no frames processed",
                })
                continue
            
            # Извлекаем depth maps для этого видео
            video_depth_maps = depth_maps_out[frame_start_idx:frame_end_idx]
            video_depth_mean = depth_mean_out[frame_start_idx:frame_end_idx]
            video_depth_std = depth_std_out[frame_start_idx:frame_end_idx]
            video_depth_p05 = depth_p05_out[frame_start_idx:frame_end_idx]
            video_depth_p95 = depth_p95_out[frame_start_idx:frame_end_idx]
            video_times_s = video_info["times_s"]
            
            # Проверяем соответствие размеров
            n_video_frames = len(video_frame_indices)
            if len(video_depth_maps) != n_video_frames or len(video_times_s) != n_video_frames:
                logger.warning(
                    f"core_depth_midas | batch | video {video_ctx.video_id} size mismatch: "
                    f"depth_maps={len(video_depth_maps)}, indices={len(video_frame_indices)}, times={len(video_times_s)}"
                )
                # Используем минимальный размер
                min_size = min(len(video_depth_maps), len(video_frame_indices), len(video_times_s))
                video_depth_maps = video_depth_maps[:min_size]
                video_depth_mean = video_depth_mean[:min_size]
                video_depth_std = video_depth_std[:min_size]
                video_depth_p05 = video_depth_p05[:min_size]
                video_depth_p95 = video_depth_p95[:min_size]
                video_frame_indices = video_frame_indices[:min_size]
                video_times_s = video_times_s[:min_size]
            
            # Сохраняем результаты в per-video rs_path
            component_dir = video_ctx.get_component_rs_path(NAME)
            npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
            
            # Подготовка метаданных
            metadata = video_ctx.load_metadata()
            total_frames = int(metadata.get("total_frames", 0))

            # ------------------------------------------------------------
            # Backend-friendly proxies (Audit v3)
            # ------------------------------------------------------------
            eps = np.float32(1e-6)
            denom = (video_depth_p95 - video_depth_p05).astype(np.float32)
            denom = np.where(np.isfinite(denom) & (denom > 0), denom, eps).astype(np.float32)
            video_depth_maps_norm = ((video_depth_maps - video_depth_p05[:, None, None]) / denom[:, None, None]).astype(np.float32)
            video_depth_maps_norm = np.clip(video_depth_maps_norm, 0.0, 1.0).astype(np.float32)

            video_depth_range_robust = (video_depth_p95 - video_depth_p05).astype(np.float32)
            video_fg_bg_sep = (video_depth_range_robust / (video_depth_std.astype(np.float32) + eps)).astype(np.float32)

            gx = np.abs(np.diff(video_depth_maps_norm, axis=2))
            gy = np.abs(np.diff(video_depth_maps_norm, axis=1))
            video_depth_complexity = (0.5 * (gx.mean(axis=(1, 2)) + gy.mean(axis=(1, 2)))).astype(np.float32)

            preview_k = 10
            n_prev = int(min(preview_k, video_depth_maps.shape[0]))
            if n_prev <= 0:
                raise RuntimeError(f"core_depth_midas | batch | internal error: n_prev<=0 with N={video_depth_maps.shape[0]}")
            if n_prev == video_depth_maps.shape[0]:
                sel = np.arange(video_depth_maps.shape[0], dtype=np.int64)
            else:
                sel = np.unique(np.round(np.linspace(0, video_depth_maps.shape[0] - 1, n_prev)).astype(np.int64))
                if sel.size < n_prev:
                    missing = n_prev - int(sel.size)
                    tail = np.arange(video_depth_maps.shape[0] - 1, -1, -1, dtype=np.int64)
                    seen = set(map(int, sel.tolist()))
                    for t in tail:
                        if int(t) not in seen:
                            sel = np.append(sel, t)
                            seen.add(int(t))
                            missing -= 1
                            if missing <= 0:
                                break
                    sel = np.sort(sel.astype(np.int64))

            preview_frame_indices = np.asarray(video_frame_indices, dtype=np.int32)[sel]
            preview_times_s = video_times_s.astype(np.float32)[sel]
            preview_depth_maps = video_depth_maps[sel].astype(np.float32, copy=False)
            preview_depth_maps_norm = video_depth_maps_norm[sel].astype(np.float32, copy=False)
            
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
                "model_name": str(triton_model_name),
                "total_frames": total_frames,
                "out_width": out_width,
                "out_height": out_height,
                "batch_size": batch_size,
                "runtime": "triton-gpu",
                "device": "cuda",
                "triton_preprocess_preset": str(triton_preprocess_preset),
                "backend_proxy_version": "core_depth_midas_backend_proxy_v1",
                "preview_k": int(n_prev),
            }
            
            # Models used
            # Models used (prefer ModelManager identity when spec is used)
            if isinstance(mm_entry, dict) and isinstance(mm_entry.get("models_used_entry"), dict):
                models_used = [mm_entry["models_used_entry"]]
                save_metadata["triton_model_spec"] = str(spec_name)
                save_metadata["triton_model_name"] = str(triton_model_name)
            else:
                models_used = [
                    model_used(
                        model_name=str(triton_model_name),
                        model_version=config.get("model_version", "unknown"),
                        weights_digest=config.get("weights_digest", "unknown"),
                        runtime="triton-gpu",
                        engine="onnx",
                        precision=config.get("precision", "fp32"),
                        device="cuda",
                    )
                ]
            save_metadata = apply_models_meta(save_metadata, models_used=models_used)
            
            # Сохранение NPZ
            npz_dict = {
                "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
                "times_s": video_times_s.astype(np.float32),
                "depth_maps": video_depth_maps,
                "depth_maps_norm": video_depth_maps_norm,
                "depth_mean": video_depth_mean.astype(np.float32),
                "depth_std": video_depth_std.astype(np.float32),
                "depth_p05": video_depth_p05.astype(np.float32),
                "depth_p95": video_depth_p95.astype(np.float32),
                "depth_range_robust": video_depth_range_robust.astype(np.float32),
                "depth_complexity_score": video_depth_complexity.astype(np.float32),
                "foreground_background_separation_proxy": video_fg_bg_sep.astype(np.float32),
                "preview_frame_indices": preview_frame_indices.astype(np.int32),
                "preview_times_s": preview_times_s.astype(np.float32),
                "preview_depth_maps": preview_depth_maps.astype(np.float32, copy=False),
                "preview_depth_maps_norm": preview_depth_maps_norm.astype(np.float32, copy=False),
                "meta": np.asarray(save_metadata, dtype=object),
            }
            
            _atomic_save_npz(npz_path, **npz_dict)
            
            # Валидация NPZ
            ok, issues, _ = validate_npz(
                npz_path,
                required_meta_keys=[
                    "producer",
                    "producer_version",
                    "schema_version",
                    "created_at",
                    "platform_id",
                    "video_id",
                    "run_id",
                    "config_hash",
                    "sampling_policy_version",
                    "dataprocessor_version",
                    "status",
                    "empty_reason",
                    "models_used",
                    "model_signature",
                ],
            )
            if not ok:
                try:
                    if os.path.exists(npz_path):
                        os.remove(npz_path)
                except Exception:
                    pass
                raise RuntimeError(
                    f"core_depth_midas | batch | saved artifact failed validation: "
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
            f"core_depth_midas | batch | completed in {duration:.2f}s "
            f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
        )
        
        return results
        
    except Exception as e:
        logger.exception(f"core_depth_midas | batch | error: {e}")
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
        if client is not None:
            try:
                # Triton client cleanup if needed
                pass
            except Exception:
                pass

