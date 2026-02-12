"""
Batch processing utilities for core_optical_flow component.

Stage 3: GPU batching для core_optical_flow с гибридным подходом:
- Сбор кадров из всех видео
- Группировка в батчи по max_frames_per_batch
- Последовательная обработка батчей через Triton (пары кадров)
- Распределение результатов обратно по видео
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import json
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

# Import core_optical_flow functions
_core_optical_flow_path = _visual_processor_path / "core" / "model_process" / "core_optical_flow"
sys.path.insert(0, str(_core_optical_flow_path.parent.parent.parent))

logger = get_logger("VisualProcessor.core_optical_flow_batch")

# Import from core_optical_flow/main.py
_core_optical_flow_main = _core_optical_flow_path / "main.py"
if _core_optical_flow_main.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("core_optical_flow_main", str(_core_optical_flow_main))
    if spec and spec.loader:
        core_optical_flow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_optical_flow_module)
        
        # Import functions
        _preset_to_input_size = getattr(core_optical_flow_module, "_preset_to_input_size", None)
        _prep_batch_rgb_uint8 = getattr(core_optical_flow_module, "_prep_batch_rgb_uint8", None)
        _require_union_times_s = getattr(core_optical_flow_module, "_require_union_times_s", None)
        NAME = getattr(core_optical_flow_module, "NAME", "core_optical_flow")
        VERSION = getattr(core_optical_flow_module, "VERSION", "2.0")
        SCHEMA_VERSION = getattr(core_optical_flow_module, "SCHEMA_VERSION", "core_optical_flow_npz_v1")
    else:
        raise ImportError("Failed to load core_optical_flow module")
else:
    raise ImportError(f"core_optical_flow/main.py not found at {_core_optical_flow_main}")


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


def process_core_optical_flow_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Batch processing для core_optical_flow с гибридным подходом.
    
    Stage 3: GPU batching для core_optical_flow.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация core_optical_flow
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для inference (количество пар кадров)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"core_optical_flow | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch}, batch_size={batch_size})"
    )
    
    start_time = time.perf_counter()
    
    # Инициализация Triton клиента
    triton_http_url = config.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL")
    if not triton_http_url:
        raise RuntimeError("core_optical_flow | batch | runtime=triton requires triton_http_url")
    
    from dp_triton import TritonHttpClient, TritonError
    
    client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=240.0)
    if not client.ready():
        raise TritonError("core_optical_flow | batch | Triton is not ready", error_code="triton_unavailable")
    
    # Параметры Triton
    triton_model_name = config.get("triton_model_name")
    if not triton_model_name:
        raise RuntimeError("core_optical_flow | batch | triton_model_name is required")
    
    triton_model_version = config.get("triton_model_version")
    triton_input0_name = config.get("triton_input0_name", "INPUT0__0")
    triton_input1_name = config.get("triton_input1_name", "INPUT1__0")
    triton_output_name = config.get("triton_output_name", "OUTPUT__0")
    triton_datatype = config.get("triton_datatype", "UINT8")
    triton_preprocess_preset = config.get("triton_preprocess_preset", "raft_256")
    input_size = _preset_to_input_size(triton_preprocess_preset)
    
    # Этап 1: Сбор всех кадров с привязкой к видео
    frames_by_video: List[Dict[str, Any]] = []
    all_frame_pairs: List[Tuple[int, int, int, np.ndarray, np.ndarray]] = []  # (video_idx, prev_idx, cur_idx, prev_frame, cur_frame)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            total_frames = int(metadata.get("total_frames", 0))
            
            # Получаем frame_indices
            try:
                frame_indices = _get_frame_indices(metadata, NAME)
            except Exception as e:
                logger.error(f"core_optical_flow | batch | video {video_ctx.video_id} failed to get frame_indices: {e}")
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
            
            if not frame_indices or len(frame_indices) < 2:
                logger.warning(f"core_optical_flow | batch | video {video_ctx.video_id} has insufficient frame_indices (need at least 2)")
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
            times_s = _require_union_times_s(frame_manager, frame_indices)
            
            # Оптимизация: создаем маппинг frame_idx -> position для O(1) поиска
            frame_idx_to_pos = {idx: pos for pos, idx in enumerate(frame_indices)}
            
            # Предвычисляем dt для всех пар заранее
            dt_pairs = []
            for i in range(1, len(frame_indices)):
                prev_pos = i - 1
                cur_pos = i
                dt = float(times_s[cur_pos] - times_s[prev_pos])
                dt = max(dt, 1e-6)  # Минимальный dt
                dt_pairs.append(dt)
            
            # Загружаем пары кадров и сохраняем маппинг
            video_pair_start_idx = len(all_frame_pairs)  # Начальный индекс в общем батче для этого видео
            pair_dt_list = []  # Сохраняем dt для каждой пары
            for i in range(1, len(frame_indices)):
                prev_idx = frame_indices[i - 1]
                cur_idx = frame_indices[i]
                try:
                    prev_frame = frame_manager.get(prev_idx)
                    cur_frame = frame_manager.get(cur_idx)
                    # Сохраняем пару кадров в общий батч вместе с dt
                    all_frame_pairs.append((video_idx, prev_idx, cur_idx, prev_frame, cur_frame))
                    pair_dt_list.append(dt_pairs[i - 1])  # dt для этой пары
                except Exception as e:
                    logger.warning(
                        f"core_optical_flow | batch | video {video_ctx.video_id} failed to load frame pair ({prev_idx}, {cur_idx}): {e}"
                    )
                    continue
            
            # Сохраняем информацию о диапазоне индексов для этого видео
            video_pair_end_idx = len(all_frame_pairs)
            
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": frame_indices,
                "frame_manager": frame_manager,
                "times_s": times_s,
                "frame_idx_to_pos": frame_idx_to_pos,  # O(1) lookup маппинг
                "pair_start_idx": video_pair_start_idx,
                "pair_end_idx": video_pair_end_idx,
                "pair_dt_list": pair_dt_list,  # Предвычисленные dt для пар
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"core_optical_flow | batch | video {video_ctx.video_id} failed to prepare: {e}")
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": [],
                "frame_manager": None,
                "times_s": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_frame_pairs:
        logger.error("core_optical_flow | batch | no frame pairs collected from any video")
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
                "error": "no frame pairs collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"core_optical_flow | batch | collected {len(all_frame_pairs)} frame pairs from {len(frames_by_video)} videos")
    
    # Этап 2: Группировка в батчи и обработка
    try:
        n_pairs = len(all_frame_pairs)
        motion_norm_per_sec_out = None
        dt_seconds_out = None
        
        # Определяем размер батча
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        logger.info(f"core_optical_flow | batch | processing {n_pairs} frame pairs in batches of {effective_batch_size}")
        
        start = 0
        while start < n_pairs:
            t_batch_start = time.perf_counter()
            batch_end = min(start + effective_batch_size, n_pairs)
            batch_pairs = all_frame_pairs[start:batch_end]

            # Извлекаем кадры из батча (кадры уже загружены на этапе сбора)
            t_extract_start = time.perf_counter()
            prev_frames = [prev_frame for _, _, _, prev_frame, _ in batch_pairs]
            cur_frames = [cur_frame for _, _, _, _, cur_frame in batch_pairs]
            t_extract = time.perf_counter() - t_extract_start
            
            # Preprocess для Triton
            t_prep_start = time.perf_counter()
            inp0 = _prep_batch_rgb_uint8(prev_frames, input_size=input_size)
            inp1 = _prep_batch_rgb_uint8(cur_frames, input_size=input_size)
            t_prep = time.perf_counter() - t_prep_start
            
            # Inference через Triton (два входа)
            t_infer_start = time.perf_counter()
            try:
                if not hasattr(client, "infer_two_inputs"):
                    raise RuntimeError(
                        "core_optical_flow | batch | dp_triton client missing infer_two_inputs(). "
                        "Please update dp_triton to support 2-input models."
                    )
                out0 = client.infer_two_inputs(
                    model_name=str(triton_model_name),
                    model_version=str(triton_model_version) if triton_model_version else None,
                    input0_name=str(triton_input0_name),
                    input0_tensor=inp0,
                    input1_name=str(triton_input1_name),
                    input1_tensor=inp1,
                    output_name=str(triton_output_name),
                    datatype=str(triton_datatype),
                )
            except Exception as e:
                raise RuntimeError(f"core_optical_flow | batch | Triton infer failed: {e}") from e
            t_infer = time.perf_counter() - t_infer_start
            
            flow = np.asarray(out0.output, dtype=np.float32)
            # Expect (B,2,H,W)
            if flow.ndim != 4 or flow.shape[1] != 2:
                raise RuntimeError(f"core_optical_flow | batch | Triton output has invalid shape: {flow.shape}")
            if flow.shape[0] != inp0.shape[0]:
                raise RuntimeError(f"core_optical_flow | batch | Triton output batch mismatch: outB={flow.shape[0]} inB={inp0.shape[0]}")
            
            # Инициализируем выходные массивы
            if motion_norm_per_sec_out is None:
                n_total = n_pairs
                motion_norm_per_sec_out = np.full((n_total,), np.nan, dtype=np.float32)
                dt_seconds_out = np.full((n_total,), np.nan, dtype=np.float32)
            
            # Оптимизация: векторизованное вычисление magnitude для всего батча
            # flow shape: (B, 2, H, W)
            # mag shape: (B, H, W)
            # Используем np.hypot для численной стабильности и согласованности с main.py
            mag = np.hypot(flow[:, 0], flow[:, 1])  # (B, H, W) - более эффективно и численно стабильно
            # Вычисляем mean для каждого элемента батча
            mag_mean = mag.reshape(flow.shape[0], -1).mean(axis=1).astype(np.float32)  # (B,)
            norm = float(max(flow.shape[2], flow.shape[3], 1))
            
            # Собираем dt для всех пар в батче
            batch_dts = []
            for i_local in range(flow.shape[0]):
                video_idx, prev_idx, cur_idx, _, _ = batch_pairs[i_local]
                video_info = frames_by_video[video_idx]
                
                # Оптимизация: используем O(1) lookup вместо O(n) .index()
                frame_idx_to_pos = video_info["frame_idx_to_pos"]
                pair_start_idx = video_info["pair_start_idx"]
                pair_dt_list = video_info["pair_dt_list"]
                
                # Находим индекс пары в списке пар этого видео
                # Пара (prev_idx, cur_idx) соответствует позиции в frame_indices
                prev_pos = frame_idx_to_pos[prev_idx]
                cur_pos = frame_idx_to_pos[cur_idx]
                
                # Пара находится на позиции (cur_pos - 1) в списке пар (т.к. пары начинаются с индекса 1)
                pair_local_idx = cur_pos - 1
                if 0 <= pair_local_idx < len(pair_dt_list):
                    dt = pair_dt_list[pair_local_idx]
                else:
                    # Fallback: вычисляем напрямую (должно быть редко)
                    times_s_video = video_info["times_s"]
                    dt = float(times_s_video[cur_pos] - times_s_video[prev_pos])
                    dt = max(dt, 1e-6)
                
                batch_dts.append(dt)
            
            batch_dts_np = np.asarray(batch_dts, dtype=np.float32)
            
            # Векторизованное вычисление motion norm per second для всего батча
            vals = (mag_mean / np.maximum(batch_dts_np, 1e-6)) / float(max(norm, 1.0))
            
            if not np.all(np.isfinite(vals)):
                bad = np.where(~np.isfinite(vals))[0]
                bad_pairs = [batch_pairs[i] for i in bad]
                raise RuntimeError(
                    f"core_optical_flow | batch | invalid motion value(s) at batch offsets: {bad.tolist()}, "
                    f"pairs: {[(p[1], p[2]) for p in bad_pairs]}"
                )
            
            # Post-process: записываем результаты
            t_post_start = time.perf_counter()
            global_start = start
            global_end = start + flow.shape[0]
            motion_norm_per_sec_out[global_start:global_end] = vals.astype(np.float32)
            dt_seconds_out[global_start:global_end] = batch_dts_np
            t_post = time.perf_counter() - t_post_start
            
            t_batch_total = time.perf_counter() - t_batch_start
            if len(batch_pairs) > 0:
                ms_per_pair = (t_batch_total * 1000.0) / len(batch_pairs)
                if start % (effective_batch_size * 10) == 0:
                    logger.info(
                        f"core_optical_flow | batch | processed {batch_end}/{n_pairs} frame pairs | "
                        f"extract={t_extract*1000:.1f}ms prep={t_prep*1000:.1f}ms "
                        f"infer={t_infer*1000:.1f}ms post={t_post*1000:.1f}ms "
                        f"total={t_batch_total*1000:.1f}ms ({ms_per_pair:.1f}ms/pair)"
                    )
            
            start = batch_end
        
        # Этап 3: Распределение результатов обратно по видео
        logger.info("core_optical_flow | batch | distributing results back to videos")
        
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
            
            # Извлекаем motion для этого видео используя сохраненные индексы
            video_frame_indices = video_info["frame_indices"]
            pair_start_idx = video_info.get("pair_start_idx", 0)
            pair_end_idx = video_info.get("pair_end_idx", len(motion_norm_per_sec_out))
            
            if pair_start_idx >= pair_end_idx or pair_start_idx >= len(motion_norm_per_sec_out):
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "empty",
                    "empty_reason": "no frame pairs processed",
                })
                continue
            
            # Извлекаем motion для этого видео
            video_motion = motion_norm_per_sec_out[pair_start_idx:pair_end_idx]
            video_dt = dt_seconds_out[pair_start_idx:pair_end_idx]
            video_times_s = video_info["times_s"]
            
            # Создаем полные массивы для всех кадров (первый кадр имеет motion=0, dt=NaN)
            n_frames = len(video_frame_indices)
            full_motion = np.full((n_frames,), np.nan, dtype=np.float32)
            full_dt = np.full((n_frames,), np.nan, dtype=np.float32)
            
            full_motion[0] = 0.0  # Первый кадр
            full_motion[1:] = video_motion[:n_frames - 1]  # Остальные кадры
            full_dt[1:] = video_dt[:n_frames - 1]  # dt для остальных кадров
            
            # Проверяем соответствие размеров
            if len(full_motion) != n_frames or len(full_dt) != n_frames or len(video_times_s) != n_frames:
                logger.warning(
                    f"core_optical_flow | batch | video {video_ctx.video_id} size mismatch: "
                    f"motion={len(full_motion)}, dt={len(full_dt)}, indices={len(video_frame_indices)}, times={len(video_times_s)}"
                )
                # Используем минимальный размер
                min_size = min(len(full_motion), len(full_dt), len(video_frame_indices), len(video_times_s))
                full_motion = full_motion[:min_size]
                full_dt = full_dt[:min_size]
                video_frame_indices = video_frame_indices[:min_size]
                video_times_s = video_times_s[:min_size]
            
            # Сохраняем результаты в per-video rs_path
            component_dir = video_ctx.get_component_rs_path(NAME)
            npz_path = os.path.join(component_dir, "flow.npz")
            
            # Подготовка метаданных
            metadata = video_ctx.load_metadata()
            total_frames = int(metadata.get("total_frames", 0))
            
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
                "batch_size": batch_size,
                "runtime": "triton-gpu",
                "device": "cuda",
            }
            
            # Models used
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
            save_metadata["models_used"] = models_used
            save_metadata = apply_models_meta(save_metadata, models_used=models_used)
            
            # Сохранение NPZ
            npz_dict = {
                "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
                "times_s": video_times_s.astype(np.float32),
                "motion_norm_per_sec_mean": full_motion.astype(np.float32),
                "dt_seconds": full_dt.astype(np.float32),
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
                    f"core_optical_flow | batch | saved artifact failed validation: "
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
            f"core_optical_flow | batch | completed in {duration:.2f}s "
            f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
        )
        
        return results
        
    except Exception as e:
        logger.exception(f"core_optical_flow | batch | error: {e}")
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

