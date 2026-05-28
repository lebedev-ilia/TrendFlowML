# action_recognition_slowfast.py
"""
Production-ready action recognition and analytics based on SlowFast (ResNet-50).

Основные отличия от "простого" варианта:
- интеграция с BaseModule для единообразия
- более безопасная обработка ошибок и памяти
- нормализация эмбеддингов выполняется на устройстве (GPU если доступен)
- результаты сохраняются в детерминированном формате, сопровождаемые metadata
- комментарии на русском
"""

import os
import sys

# VisualProcessor root (…/modules/action_recognition/utils/this.py → 3× parent)
_vp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _vp not in sys.path:
    sys.path.insert(0, _vp)
_dp = os.path.abspath(os.path.join(_vp, ".."))
if _dp not in sys.path:
    sys.path.insert(1 if len(sys.path) > 1 else 0, _dp)

from typing import Dict, List, Any, Optional, Tuple, Callable, TYPE_CHECKING
import traceback

if TYPE_CHECKING:
    from utils.video_context import VideoContext
from collections import defaultdict
import random
import time
import json

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

# Импорт SlowFast из pytorchvideo (как в ex.py)
try:
    from pytorchvideo.models.hub import slowfast_r50
except ImportError as e:
    raise ImportError(
        f"Cannot import slowfast_r50 from pytorchvideo.models.hub: {e}. "
        "Please install pytorchvideo: pip install pytorchvideo"
    ) from e
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

from dp_models.manager import get_global_model_manager
from dp_models.errors import ModelManagerError
from dp_models.signatures import model_used


def longest_run_fraction(labels: List[int]) -> float:
    """Доля самой длинной непрерывной серии одинаковых кластеров"""
    if not labels:
        return 0.0
    max_run = cur = 1
    for a, b in zip(labels, labels[1:]):
        cur = cur + 1 if a == b else 1
        max_run = max(max_run, cur)
    return max_run / len(labels)


class SlowFastActionRecognizer(BaseModule):
    """
    Production-реализация анализатора действий на основе SlowFast.
    
    Наследуется от BaseModule для единообразия с другими модулями.
    """

    MODULE_NAME = "action_recognition"
    VERSION = "2.0"
    SCHEMA_VERSION = "action_recognition_npz_v2"
    ARTIFACT_FILENAME = "action_recognition_features.npz"
    DEFAULT_MODEL_NAME = "slowfast_r50_action_recognition"

    def __init__(
        self,
        rs_path: Optional[str] = None,
        clip_len: int = 32,  # Минимум для SlowFast (T_slow=8)
        stride: Optional[int] = None,
        batch_size: int = 8,
        embedding_dim: int = 256,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        seed: Optional[int] = 42,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs: Any
    ):
        """
        Инициализация SlowFastActionRecognizer.
        
        Args:
            rs_path: Путь к хранилищу результатов
            clip_len: Длина клипа в кадрах
            stride: Шаг для создания клипов (по умолчанию clip_len // 2)
            batch_size: Размер батча для inference
            embedding_dim: Размерность эмбеддингов
            device: Устройство для обработки (cuda/cpu)
            seed: Seed для детерминированности
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, logger_name=self.MODULE_NAME, **kwargs)
        
        # параметры
        self.clip_len = int(clip_len)
        self.stride = stride if stride is not None else max(1, self.clip_len // 2)
        self.batch_size = int(batch_size)
        self.embedding_dim = int(embedding_dim)
        self.alpha = int(kwargs.get("alpha", 4))  # SlowFast alpha (T_fast / T_slow, по умолчанию 4)
        # Параметры сегментации
        self.segment_gap_sec = float(kwargs.get("segment_gap_sec", 0.5))  # Временной порог для разрыва сегмента (сек)
        self.min_person_confidence = float(kwargs.get("min_person_confidence", 0.3))  # Минимальный confidence для person детекций
        self._explicit_device = device is not None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.model_name = str(model_name or self.DEFAULT_MODEL_NAME)
        self.progress_callback = progress_callback

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Модель будет загружена в _do_initialize()
        self.model: Optional[torch.nn.Module] = None
        self.embedding_proj: Optional[torch.nn.Module] = None
        self.raw_embedding_dim = 2048
        self._mm = None
        self._models_used_entry: Optional[Dict[str, Any]] = None
        self._last_metadata: Optional[Dict[str, Any]] = None
        self._last_empty_reason: Optional[str] = None

        # нормализация входа (ImageNet-like)
        self.mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        self.std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    
    def _do_initialize(self) -> None:
        """Инициализация модели SlowFast."""
        self.logger.info("Инициализация SlowFast R50 через ModelManager (local-only)")

        self._mm = get_global_model_manager()
        try:
            spec = self._mm.get_spec(model_name=self.model_name)
            device, precision, runtime, engine, weights_digest, resolved_artifacts = self._mm.resolve(spec)
        except ModelManagerError as e:
            raise RuntimeError(f"{self.module_name} | ModelManager resolve failed: {e}") from e

        if runtime != "inprocess":
            raise RuntimeError(f"{self.module_name} | Unsupported runtime for SlowFast: {runtime}")
        if str(engine).lower() not in ("torch", "pytorch"):
            raise RuntimeError(f"{self.module_name} | Unsupported engine for SlowFast: {engine}")

        # prefer explicit device from init, otherwise use ModelManager device policy
        if not self._explicit_device:
            self.device = device

        ckpt_rel = None
        for a in spec.local_artifacts:
            if str(a.kind) == "file":
                ckpt_rel = str(a.path)
                break
        if not ckpt_rel:
            raise RuntimeError(f"{self.module_name} | No checkpoint file declared in model spec")
        ckpt_path = resolved_artifacts.get(ckpt_rel) or ckpt_rel

        # загружаем модель без pretrained (no-network policy)
        # pytorchvideo использует pretrained=False
        model = slowfast_r50(pretrained=False)

        # В pytorchvideo веса лежат в "model_state" (как в ex.py)
        state_dict = self._load_state_dict(ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"{self.module_name} | SlowFast state_dict mismatch: missing={missing}, unexpected={unexpected}"
            )

        self.model = model.to(self.device).eval()

        _mdt = torch.float32
        for _p in self.model.parameters():
            if _p.is_floating_point():
                _mdt = _p.dtype
                break
        # проекция в компактное пространство (та же dtype, что у backbone — иначе matmul падает на mixed precision)
        self.embedding_proj = nn.Linear(self.raw_embedding_dim, self.embedding_dim).to(
            device=self.device, dtype=_mdt
        )
        nn.init.xavier_uniform_(self.embedding_proj.weight)
        nn.init.zeros_(self.embedding_proj.bias)
        self.embedding_proj.eval()

        self._models_used_entry = model_used(
            model_name=spec.model_name,
            model_version=spec.model_version,
            weights_digest=weights_digest,
            runtime=runtime,
            engine=engine,
            precision=precision,
            device=self.device,
        )

        self.logger.info(
            f"{self.module_name} готов | clip_len={self.clip_len} stride={self.stride} "
            f"batch_size={self.batch_size} embedding_dim={self.embedding_dim} device={self.device}"
        )

    def _load_state_dict(self, ckpt_path: str) -> Dict[str, Any]:
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise RuntimeError(f"{self.module_name} | Checkpoint not found: {ckpt_path}")

        state: Any = None
        if ckpt_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as _load_safetensors  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"{self.module_name} | safetensors is required to load {ckpt_path}: {e}"
                ) from e
            state = _load_safetensors(ckpt_path, device="cpu")
        else:
            try:
                # ВАЖНО: pytorchvideo использует weights_only=False для загрузки полных моделей
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as e:
                raise RuntimeError(f"{self.module_name} | Failed to load checkpoint: {e}") from e

        # В pytorchvideo веса лежат в "model_state" (как в ex.py)
        if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
        # Если state уже является словарем с ключами модели, используем его напрямую

        if not isinstance(state, dict):
            raise RuntimeError(f"{self.module_name} | Checkpoint does not contain a state_dict dict")

        # common prefix cleanup
        if any(str(k).startswith("module.") for k in state.keys()):
            cleaned = {}
            for k, v in state.items():
                ks = str(k)
                if ks.startswith("module."):
                    ks = ks[len("module.") :]
                cleaned[ks] = v
            state = cleaned

        return state

    def _load_frames(self, frame_manager: FrameManager, indices: List[int]) -> List[np.ndarray]:
        """Загружает и нормализует кадры как RGB uint8 HxWx3"""
        frames: List[np.ndarray] = []
        for idx in indices:
            im = frame_manager.get(idx)
            if im is None:
                raise ValueError(f"FrameManager.get({idx}) вернул None")
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            if im.shape[-1] == 4:
                im = im[..., :3]
            frames.append(im.astype(np.uint8))
        return frames

    def _make_clips(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
    ) -> List[Tuple[List[np.ndarray], List[int], int]]:
        """Разбивает последовательность кадров на перекрывающиеся клипы и сохраняет индексы."""
        if len(frames) == 0 or len(frame_indices) == 0:
            return []
        if len(frames) != len(frame_indices):
            raise ValueError(f"{self.module_name} | frames и frame_indices длины не совпадают")

        if len(frames) < self.clip_len:
            pad = self.clip_len - len(frames)
            frames = frames + [frames[-1]] * pad
            frame_indices = frame_indices + [frame_indices[-1]] * pad

        clips: List[Tuple[List[np.ndarray], List[int], int]] = []
        for start in range(0, len(frames) - self.clip_len + 1, self.stride):
            clip_frames = frames[start:start + self.clip_len]
            clip_indices = frame_indices[start:start + self.clip_len]
            center_idx = clip_indices[len(clip_indices) // 2]
            clips.append((clip_frames, clip_indices, center_idx))

        if not clips:
            clip_frames = frames[-self.clip_len:]
            clip_indices = frame_indices[-self.clip_len:]
            center_idx = clip_indices[len(clip_indices) // 2]
            clips.append((clip_frames, clip_indices, center_idx))

        return clips

    def _preprocess_clip(self, clip: List[np.ndarray]) -> torch.Tensor:
        """
        Преобразует clip (List[HxWx3]) в Tensor [C, T, H, W] float32 на CPU.
        Приводит кадры к 224x224, выполняет нормализацию.
        
        Гарантирует, что T == clip_len и T кратно alpha (как в ex.py).
        """
        processed = []
        for frame in clip:
            # resize -> float32 -> normalize -> C,H,W
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame_float = frame_resized.astype(np.float32) / 255.0
            frame_norm = (frame_float - self.mean) / self.std
            frame_chw = np.transpose(frame_norm, (2, 0, 1))
            processed.append(frame_chw)
        clip_arr = np.stack(processed, axis=1)  # C,T,H,W
        tensor = torch.from_numpy(clip_arr).float()
        
        # Гарантируем точную длину clip_len (как temporal_subsample_to в ex.py)
        C, T, H, W = tensor.shape
        if T != self.clip_len:
            if T > self.clip_len:
                # Uniform subsample to exact length
                indices = np.linspace(0, T - 1, num=self.clip_len, dtype=int)
                tensor = tensor[:, indices, :, :]
            else:
                # Pad by repeating last frame
                repeats = self.clip_len - T
                last = tensor[:, -1:, :, :].repeat(1, repeats, 1, 1)
                tensor = torch.cat([tensor, last], dim=1)
                self.logger.debug(
                    f"{self.module_name} | Padded clip from T={T} to T={self.clip_len}"
                )
        
        # Гарантируем кратность alpha (как в ex.py)
        C, T, H, W = tensor.shape
        if T % self.alpha != 0:
            # Pad to next multiple of alpha
            target = int((T + self.alpha - 1) // self.alpha * self.alpha)
            repeats = target - T
            last = tensor[:, -1:, :, :].repeat(1, repeats, 1, 1)
            tensor = torch.cat([tensor, last], dim=1)
            self.logger.debug(
                f"{self.module_name} | Padded clip from T={T} to T={target} to be multiple of alpha={self.alpha}"
            )
        
        # Финальная проверка минимальной длины (T_slow должен быть >= 8)
        C, T, H, W = tensor.shape
        T_slow = T // self.alpha
        if T_slow < 8:
            raise ValueError(
                f"{self.module_name} | Clip too short: T_fast={T}, T_slow={T_slow} < 8. "
                f"Minimum clip_len should be {8 * self.alpha} (got {self.clip_len})"
            )
        
        return tensor

    @staticmethod
    def _prepare_slow_fast(batch: torch.Tensor, alpha: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        batch: [B, C, T, H, W] -> возвращает slow и fast пути (как в ex.py)
        
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
            raise ValueError(f"Ожидался batch.dim()==5, получено {batch.dim()}")
        B, C, T, H, W = batch.shape
        
        # Slow путь: каждый alpha-й кадр (как в ex.py pack_pathways)
        # Для batch: [B, C, T, H, W] -> [B, C, T//alpha, H, W]
        slow = batch[:, :, ::alpha, :, :]
        
        # Fast путь: все кадры
        fast = batch
        
        return slow, fast

    def _extract_features(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        """
        Прогоняем через модель и приводим к [B, raw_embedding_dim].
        В случае ошибки выбрасываем исключение (no-fallback).
        """
        if self.model is None:
            raise RuntimeError(f"{self.module_name} | Model is not initialized")

        # Берём dtype с первого вещественного параметра (не все буферы/параметры одного типа у некоторых чекпоинтов)
        model_dtype = torch.float32
        model_device = slow.device
        for p in self.model.parameters():
            if p.is_floating_point():
                model_dtype = p.dtype
                model_device = p.device
                break
        slow = slow.to(device=model_device, dtype=model_dtype)
        fast = fast.to(device=model_device, dtype=model_dtype)

        with torch.inference_mode():
            # Явно без autocast: смешение FP16 backbone и FP32 входов даёт типичные ошибки в Linear/BN.
            if model_device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=False):
                    out = self.model([slow, fast])
            else:
                out = self.model([slow, fast])

        # модель может вернуть tensor или tuple/list; пытаемся извлечь tensor
        if torch.is_tensor(out):
            feat = out
        elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
            feat = out[0]
        else:
            raise RuntimeError("Неожиданный тип выхода модели: %s" % type(out))

        # уменьшаем пространственно-временные размерности
        if feat.dim() == 5:
            # B, C, T, H, W -> усредняем по T,H,W -> B,C
            feat = feat.mean(dim=[2, 3, 4])
        elif feat.dim() == 4:
            # B,C,H,W -> усредняем по H,W -> B,C
            feat = feat.mean(dim=[2, 3])
        elif feat.dim() == 3:
            # B,C,T -> усредняем по T -> B,C
            feat = feat.mean(dim=2)

        feat = feat.view(feat.size(0), -1)  # B, D

        # выравнивание по целевому raw_embedding_dim (dtype должен совпадать — иначе cat/Linear падает на FP16-моделях)
        if feat.shape[1] > self.raw_embedding_dim:
            feat = feat[:, : self.raw_embedding_dim]
        elif feat.shape[1] < self.raw_embedding_dim:
            n_pad = int(self.raw_embedding_dim - feat.shape[1])
            pad = torch.zeros((feat.shape[0], n_pad), device=feat.device, dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=1)

        return feat

    def _extract_embeddings(self, clips: List[List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогоняет все клипы батчами и возвращает:
        - normed_embeddings: [N_clips, embedding_dim]
        - raw_embeddings: [N_clips, raw_embedding_dim]
        """
        if not clips:
            return (
                np.zeros((0, self.embedding_dim), dtype=np.float32),
                np.zeros((0, self.raw_embedding_dim), dtype=np.float32),
            )

        projected_all = []
        raw_all = []
        total_clips = len(clips)
        self.logger.info("Начинаем извлечение эмбеддингов: clips=%d batch_size=%d", total_clips, self.batch_size)

        for start in range(0, total_clips, self.batch_size):
            batch_clips = clips[start: start + self.batch_size]
            tensors_cpu = [self._preprocess_clip(c) for c in batch_clips]
            batch = torch.stack(tensors_cpu, dim=0).to(self.device)

            try:
                proj_np, raw_np = self._infer_batch_tensors(batch)
            except Exception as e:
                use_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
                # Любой сбой батча на GPU при B>1: один проход по клипам (OOM, cudnn, редкие баги батча).
                retry_per_clip = (
                    use_cuda
                    and self.batch_size > 1
                    and not isinstance(e, (KeyboardInterrupt, SystemExit))
                )
                if retry_per_clip:
                    torch.cuda.empty_cache()
                    self.logger.warning(
                        "%s | GPU batch inference failed (%s); retrying per-clip (same model, slower)",
                        self.module_name,
                        e,
                    )
                    p_parts: List[np.ndarray] = []
                    r_parts: List[np.ndarray] = []
                    for c in batch_clips:
                        one = self._preprocess_clip(c).unsqueeze(0).to(self.device)
                        pi, ri = self._infer_batch_tensors(one)
                        p_parts.append(pi)
                        r_parts.append(ri)
                        del one
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    proj_np = np.concatenate(p_parts, axis=0)
                    raw_np = np.concatenate(r_parts, axis=0)
                else:
                    raise

            projected_all.append(proj_np)
            raw_all.append(raw_np)

            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            processed = min(start + self.batch_size, total_clips)
            self._emit_progress(processed, total_clips)

            if (start // self.batch_size) % 10 == 0:
                self.logger.debug("Processed %d/%d clips", processed, total_clips)

        if not projected_all:
            return (
                np.zeros((0, self.embedding_dim), dtype=np.float32),
                np.zeros((0, self.raw_embedding_dim), dtype=np.float32),
            )

        normed = np.concatenate(projected_all, axis=0)
        raw = np.concatenate(raw_all, axis=0)
        # safety: если размер не совпадает — обрезаем/паддим
        if normed.shape[1] != self.embedding_dim:
            if normed.shape[1] > self.embedding_dim:
                normed = normed[:, : self.embedding_dim]
            else:
                pad = np.zeros((normed.shape[0], self.embedding_dim - normed.shape[1]), dtype=np.float32)
                normed = np.concatenate([normed, pad], axis=1)

        if raw.shape[1] != self.raw_embedding_dim:
            if raw.shape[1] > self.raw_embedding_dim:
                raw = raw[:, : self.raw_embedding_dim]
            else:
                pad = np.zeros((raw.shape[0], self.raw_embedding_dim - raw.shape[1]), dtype=np.float32)
                raw = np.concatenate([raw, pad], axis=1)

        return normed, raw

    def _infer_batch_tensors(self, batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        batch: [B, C, T, H, W] float tensor (any device); returns (normed [B,D], raw [B,raw_D]) on CPU.
        """
        slow, fast = self._prepare_slow_fast(batch, alpha=self.alpha)
        feat = self._extract_features(slow, fast)
        if self.embedding_proj is not None:
            w = self.embedding_proj.weight
            feat = feat.to(device=w.device, dtype=w.dtype)
        with torch.inference_mode():
            proj = self.embedding_proj(feat)
            proj = F.normalize(proj, p=2, dim=1)
        return (
            proj.detach().cpu().numpy().astype(np.float32),
            feat.detach().cpu().numpy().astype(np.float32),
        )

    def _aggregate(self, normed_embeddings: np.ndarray, raw_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет метрики для последовательности эмбеддингов трека.
        Возвращает словарь метрик (числа и статистика).
        """
        n = len(normed_embeddings)
        if n == 0:
            # пустой трек — возвращаем нейтральные значения
            return {
                "max_temporal_jump": float("nan"),
                "mean_temporal_jump": float("nan"),
                "stability": float("nan"),
                "stability_centroid_dist": float("nan"),
                "num_switches": 0,
                "num_clips": 0,
                "embedding_dim": self.embedding_dim,
            }

        # Базовые temporal jumps.
        # Для n == 1 трактуем трек как «тривиально стабильный»:
        #   max/mean_jump = 0.0, stability = 1.0, stability_centroid_dist = 0.0, switches = 0.
        if n > 1:
            diffs = [np.linalg.norm(normed_embeddings[i] - normed_embeddings[i - 1]) for i in range(1, n)]
            max_jump = float(np.max(diffs))
            mean_jump = float(np.mean(diffs))
        else:
            max_jump = 0.0
            mean_jump = 0.0

        # Улучшенная кластеризация: используем PCA + KMeans (основной метод) и DBSCAN (альтернативный)
        # Для n == 1 кластеризацию не считаем, но задаём стабильность по умолчанию.
        if n == 1:
            stability = 1.0
            stability_centroid_dist = 0.0
            switches = 0
        else:
            stability = float("nan")
            stability_centroid_dist = float("nan")
            switches = 0
        
        # Для кластеризационных метрик достаточно n >= 2 (PCA с n_components <= n-1 безопасен).
        if n >= 2:
            pca_dim = min(32, normed_embeddings.shape[1], n - 1)
            try:
                emb_pca = PCA(n_components=pca_dim).fit_transform(normed_embeddings)
                
                # Метод 1: KMeans (основной, для stability)
                k = min(5, max(1, n // 2))
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels_kmeans = kmeans.fit_predict(emb_pca)
                stability = longest_run_fraction(labels_kmeans.tolist())
                switches = int(np.sum(labels_kmeans[1:] != labels_kmeans[:-1]))
                
                # Метод 2: Альтернативная метрика - среднее расстояние до центроида кластера
                # (меньше значение = более стабильные действия)
                centroid_dists = []
                for i, label in enumerate(labels_kmeans):
                    centroid = kmeans.cluster_centers_[label]
                    dist = np.linalg.norm(emb_pca[i] - centroid)
                    centroid_dists.append(dist)
                stability_centroid_dist = float(np.mean(centroid_dists)) if centroid_dists else float("nan")
                
                # Метод 3: DBSCAN (опционально, для автоматического определения числа кластеров)
                # Используем только если KMeans дал много кластеров (k >= 3)
                if k >= 3:
                    try:
                        # eps выбираем как медиану расстояний между соседними точками
                        if n > 1:
                            pairwise_dists = [np.linalg.norm(emb_pca[i] - emb_pca[i-1]) for i in range(1, n)]
                            eps = float(np.median(pairwise_dists)) * 1.5  # 1.5x медиана
                        else:
                            eps = 0.5
                        
                        dbscan = DBSCAN(eps=eps, min_samples=2)
                        labels_dbscan = dbscan.fit_predict(emb_pca)
                        # Если DBSCAN нашел кластеры (не все -1), можно использовать для сравнения
                        n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
                        if n_clusters_dbscan > 0:
                            # Логируем для диагностики, но не используем в основной метрике
                            self.logger.debug(
                                f"{self.module_name} | DBSCAN found {n_clusters_dbscan} clusters "
                                f"(KMeans: {k})"
                            )
                    except Exception:
                        # DBSCAN может не сработать - это нормально, игнорируем
                        pass
                        
            except Exception:
                self.logger.exception("Ошибка в PCA/KMeans на треке — возвращаем безопасные метрики.")
                stability = float("nan")
                stability_centroid_dist = float("nan")
                switches = 0

        return {
            "max_temporal_jump": max_jump,
            "mean_temporal_jump": mean_jump,
            "stability": stability,
            "stability_centroid_dist": stability_centroid_dist,  # Новая метрика: среднее расстояние до центроида
            "num_switches": switches,
            "num_clips": n,
            "embedding_dim": self.embedding_dim,
        }

    def _emit_progress(self, processed: int, total: int) -> None:
        if total <= 0:
            return
        pct = float(processed) / float(total) * 100.0
        payload = {
            "component": self.module_name,
            "processed_clips": int(processed),
            "total_clips": int(total),
            "progress_pct": round(pct, 2),
        }
        if callable(self.progress_callback):
            try:
                self.progress_callback(payload)
            except Exception:
                self.logger.debug("progress_callback failed", exc_info=True)
        
        # Baseline contract: state_events.jsonl progress (best-effort)
        self._append_state_event_if_possible(
            rs_path=self.rs_path or "",
            event={
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scope": "progress",
                "processor": "visual",
                "component": self.module_name,
                "status": "running",
                "progress": float(processed) / float(total) if total > 0 else 0.0,
                "done": int(processed),
                "total": int(total),
                "stage": "extract_embeddings",
            },
        )
    
    def _append_state_event_if_possible(self, *, rs_path: str, event: Dict[str, Any]) -> None:
        """Best-effort writer for state_events.jsonl (backend tails this file)."""
        try:
            from pathlib import Path as _Path
            run_rs = _Path(rs_path).resolve()
            rs_base = run_rs.parents[2]  # <rs_base>/<platform>/<video>/<run>
            runs_root = rs_base.parent
            platform_id = str(event.get("platform_id") or "")
            video_id = str(event.get("video_id") or "")
            run_id = str(event.get("run_id") or "")
            if not (platform_id and video_id and run_id):
                # Try to extract from metadata
                if self._last_metadata:
                    platform_id = str(self._last_metadata.get("platform_id") or "")
                    video_id = str(self._last_metadata.get("video_id") or "")
                    run_id = str(self._last_metadata.get("run_id") or "")
            if not (platform_id and video_id and run_id):
                return
            p = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            event["platform_id"] = platform_id
            event["video_id"] = video_id
            event["run_id"] = run_id
            line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            with open(p, "ab") as f:
                f.write(line)
        except Exception:
            return

    @property
    def supports_batch(self) -> bool:
        """Поддержка batch processing для GPU batching."""
        return True

    def required_dependencies(self) -> List[str]:
        return ["core_object_detections"]

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        _ = config
        _ = metadata
        if self._models_used_entry:
            return [self._models_used_entry]
        return []

    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Основной метод обработки видео (интерфейс BaseModule).

        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля

        Returns:
            Dict[track_id, Dict] - результаты по трекам с эмбеддингами
        """
        _ = config
        # Model initialization is handled in run(), not here
        # (to avoid double initialization and ensure proper stage timing)
        if not self._initialized:
            self.initialize()
        self._last_empty_reason = None

        frame_indices_per_person = self._prepare_tracks(frame_indices=frame_indices)
        if not frame_indices_per_person:
            self.logger.warning("Нет кадров с person детекциями для обработки")
            self._last_empty_reason = "no_person_detections"
            return {}

        all_clips: List[List[np.ndarray]] = []
        clip_owner: List[int] = []
        clip_center_indices: List[int] = []
        clip_frame_indices: List[List[int]] = []

        # сбор всех клипов
        for tid, indices in frame_indices_per_person.items():
            if not indices:
                self.logger.debug("Пропускаем трек %s (пустой список индексов)", tid)
                continue
            frames = self._load_frames(frame_manager, indices)
            clips = self._make_clips(frames, indices)
            if not clips:
                self.logger.debug("Трек %s не дал клипов после _make_clips()", tid)
                continue
            for clip_frames, clip_indices, center_idx in clips:
                all_clips.append(clip_frames)
                clip_owner.append(tid)
                clip_center_indices.append(center_idx)
                clip_frame_indices.append(clip_indices)

        if not all_clips:
            self.logger.warning("Нет клипов для обработки")
            self._last_empty_reason = "no_person_detections"
            return {}

        normed, raw = self._extract_embeddings(all_clips)  # [N_clips, D] and [N_clips, raw_D]
        if len(normed) != len(clip_owner):
            raise RuntimeError(
                f"{self.module_name} | embeddings/owners size mismatch: "
                f"{len(normed)} != {len(clip_owner)}"
            )

        metadata = self._last_metadata or {}
        union_ts = metadata.get("union_timestamps_sec")
        if isinstance(union_ts, list):
            union_ts = np.array(union_ts, dtype=np.float32)
        if not isinstance(union_ts, np.ndarray):
            union_ts = None

        per_track_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        per_track_raw: Dict[int, List[np.ndarray]] = defaultdict(list)
        per_track_centers: Dict[int, List[int]] = defaultdict(list)
        per_track_clip_indices: Dict[int, List[List[int]]] = defaultdict(list)
        for owner_tid, emb, raw_emb, center_idx, clip_idx in zip(
            clip_owner, normed, raw, clip_center_indices, clip_frame_indices
        ):
            per_track_embeddings[owner_tid].append(emb)
            per_track_raw[owner_tid].append(raw_emb)
            per_track_centers[owner_tid].append(center_idx)
            per_track_clip_indices[owner_tid].append(clip_idx)

        results: Dict[int, Dict[str, Any]] = {}
        for tid, embs in per_track_embeddings.items():
            embs_arr = np.stack(embs, axis=0) if len(embs) else np.zeros((0, self.embedding_dim), dtype=np.float32)
            raw_arr = (
                np.stack(per_track_raw[tid], axis=0)
                if len(per_track_raw[tid])
                else np.zeros((0, self.raw_embedding_dim), dtype=np.float32)
            )
            metrics = self._aggregate(embs_arr, raw_arr)
            metrics["embedding_normed_256d"] = embs_arr
            metrics["track_frame_count"] = int(len(frame_indices_per_person.get(tid) or []))

            # per-clip diagnostics for UI (stored in results_json)
            centers = per_track_centers.get(tid) or []
            metrics["clip_center_frame_indices"] = centers
            if union_ts is not None and centers:
                metrics["clip_center_times_s"] = [float(union_ts[int(idx)]) for idx in centers if int(idx) < len(union_ts)]
            else:
                metrics["clip_center_times_s"] = []

            if len(embs_arr) > 1:
                jumps = [float(np.linalg.norm(embs_arr[i] - embs_arr[i - 1])) for i in range(1, len(embs_arr))]
            else:
                jumps = []
            metrics["temporal_jumps"] = jumps
            metrics["clip_frame_indices"] = per_track_clip_indices.get(tid) or []

            results[tid] = metrics

        self.logger.info("Обработано треков: %d", len(results))
        return results
    
    def _prepare_tracks(
        self,
        frame_indices: List[int]
    ) -> Dict[int, List[int]]:
        """
        Подготавливает кадры с детекциями person для обработки из detections.npz.
        
        Трекинг удален из core_object_detections, поэтому работаем без трекинга:
        - Загружаем детекции из detections.npz
        - Фильтруем по class_id=0 (person)
        - Группируем кадры с person детекциями в последовательные сегменты
        
        Args:
            frame_indices: Список индексов кадров для обработки
            
        Returns:
            Dict[segment_id, List[frame_indices]] - словарь сегментов с person детекциями
        """
        self.logger.info(f"{self.module_name} | _prepare_tracks: начало загрузки detections.npz")
        
        try:
            detections_data = self.load_core_provider("core_object_detections", "detections.npz")
        except Exception as e:
            self.logger.exception(
                f"{self.module_name} | _prepare_tracks: ошибка при загрузке detections.npz: {e}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise
        
        if detections_data is None:
            raise RuntimeError(
                "Не найдены результаты core_object_detections. "
                "Убедитесь, что модуль object_detections был запущен."
            )
        
        self.logger.info(
            f"{self.module_name} | _prepare_tracks: detections_data загружен, ключи: {list(detections_data.keys())}"
        )
        
        # Извлекаем данные детекций
        try:
            det_frame_indices = detections_data.get("frame_indices")
            class_ids = detections_data.get("class_ids")
            valid_mask = detections_data.get("valid_mask")
            
            if det_frame_indices is None:
                raise RuntimeError("В detections.npz отсутствует frame_indices")
            if class_ids is None:
                raise RuntimeError("В detections.npz отсутствует class_ids")
            if valid_mask is None:
                raise RuntimeError("В detections.npz отсутствует valid_mask")
            
            # Преобразуем в numpy массивы
            det_frame_indices = np.asarray(det_frame_indices, dtype=np.int32).reshape(-1)
            class_ids = np.asarray(class_ids, dtype=np.int32)
            valid_mask = np.asarray(valid_mask, dtype=bool)
            
            self.logger.info(
                f"{self.module_name} | _prepare_tracks: загружено {len(det_frame_indices)} кадров детекций, "
                f"class_ids shape={class_ids.shape}, valid_mask shape={valid_mask.shape}"
            )
        except Exception as e:
            self.logger.exception(
                f"{self.module_name} | _prepare_tracks: ошибка при извлечении данных детекций: {e}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise
        
        # Фильтруем кадры с person детекциями (class_id=0) с учетом confidence
        PERSON_CLASS_ID = 0
        frame_indices_set = set(frame_indices)
        frames_with_person: List[Tuple[int, float]] = []  # (frame_idx, max_confidence)
        
        # Загружаем scores если доступны
        scores = detections_data.get("scores")
        if scores is not None:
            scores = np.asarray(scores, dtype=np.float32)
        else:
            scores = None
        
        for i, frame_idx_global in enumerate(det_frame_indices):
            if int(frame_idx_global) not in frame_indices_set:
                continue
            
            # Проверяем наличие person детекций на этом кадре
            if i >= valid_mask.shape[0]:
                continue
            
            has_person = False
            max_confidence = 0.0
            try:
                frame_valid = valid_mask[i]
                frame_class_ids = class_ids[i] if i < class_ids.shape[0] else None
                frame_scores = scores[i] if scores is not None and i < scores.shape[0] else None
                
                if frame_class_ids is not None:
                    for j in range(min(frame_valid.size, frame_class_ids.size)):
                        if frame_valid[j] and int(frame_class_ids[j]) == PERSON_CLASS_ID:
                            has_person = True
                            # Берем максимальный confidence среди person детекций на кадре
                            if frame_scores is not None and j < frame_scores.size:
                                max_confidence = max(max_confidence, float(frame_scores[j]))
                            else:
                                max_confidence = 1.0  # Если scores нет, считаем confidence=1.0
                            break
            except Exception as e:
                self.logger.debug(
                    f"{self.module_name} | _prepare_tracks: ошибка при проверке кадра {frame_idx_global}: {e}"
                )
                continue
            
            # Фильтруем по минимальному confidence
            if has_person and max_confidence >= self.min_person_confidence:
                frames_with_person.append((int(frame_idx_global), max_confidence))
        
        if not frames_with_person:
            self.logger.info("Нет кадров с person детекциями")
            return {}
        
        self.logger.info(
            f"{self.module_name} | _prepare_tracks: найдено {len(frames_with_person)} кадров с person детекциями"
        )
        
        # Группируем кадры в последовательные сегменты (без трекинга)
        # Улучшенная группировка: используем временной порог + учитываем confidence
        frames_with_person_sorted = sorted(frames_with_person, key=lambda x: x[0])  # Сортируем по frame_idx
        
        # Получаем timestamps если доступны
        metadata = self._last_metadata or {}
        union_ts = metadata.get("union_timestamps_sec")
        if isinstance(union_ts, list):
            union_ts = np.array(union_ts, dtype=np.float32)
        if not isinstance(union_ts, np.ndarray):
            union_ts = None
        
        segments: Dict[int, List[int]] = {}
        current_segment_id = 0
        current_segment: List[int] = []
        prev_frame_idx = None
        prev_time = None
        
        for frame_idx, confidence in frames_with_person_sorted:
            # Вычисляем временной разрыв
            gap_ok = True
            if prev_frame_idx is not None:
                if union_ts is not None and frame_idx < len(union_ts) and prev_frame_idx < len(union_ts):
                    time_gap = abs(union_ts[frame_idx] - union_ts[prev_frame_idx])
                    gap_ok = time_gap <= self.segment_gap_sec
                else:
                    # Fallback: используем frame gap (приблизительно, если fps известен)
                    frame_gap = frame_idx - prev_frame_idx
                    # Предполагаем ~4 fps из sampling policy, тогда 0.5 сек = ~2 кадра
                    estimated_fps = 4.0
                    gap_ok = frame_gap <= int(self.segment_gap_sec * estimated_fps)
            
            if prev_frame_idx is None:
                current_segment.append(frame_idx)
            elif gap_ok:
                current_segment.append(frame_idx)
            else:  # Большой разрыв - начинаем новый сегмент
                if current_segment:
                    segments[current_segment_id] = current_segment
                    current_segment_id += 1
                current_segment = [frame_idx]
            
            prev_frame_idx = frame_idx
            if union_ts is not None and frame_idx < len(union_ts):
                prev_time = union_ts[frame_idx]
        
        # Добавляем последний сегмент
        if current_segment:
            segments[current_segment_id] = current_segment
        
        if not segments:
            self.logger.info("Нет сегментов с person детекциями")
            return {}
        
        self.logger.info(
            f"{self.module_name} | _prepare_tracks: создано {len(segments)} сегментов из кадров с person детекциями"
        )
        
        return segments

    def build_ui_payload(self, results: Dict[int, Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        def _safe_float(x: Any) -> Optional[float]:
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                return float(x)
            except Exception:
                return None

        def _downsample(arr: List[Any], max_points: int = 200) -> List[Any]:
            if not isinstance(arr, list) or len(arr) <= max_points:
                return arr
            step = max(1, int(len(arr) / max_points))
            return arr[::step]

        tracks_payload = []
        stability_vals = []
        stability_centroid_dist_vals = []
        max_jump_vals = []
        mean_jump_vals = []
        total_clips = 0
        for tid, r in results.items():
            stability = _safe_float(r.get("stability"))
            stability_centroid_dist = _safe_float(r.get("stability_centroid_dist"))
            max_jump = _safe_float(r.get("max_temporal_jump"))
            mean_jump = _safe_float(r.get("mean_temporal_jump"))
            num_clips = int(r.get("num_clips") or 0)
            total_clips += num_clips
            if stability is not None:
                stability_vals.append(stability)
            if stability_centroid_dist is not None:
                stability_centroid_dist_vals.append(stability_centroid_dist)
            if max_jump is not None:
                max_jump_vals.append(max_jump)
            if mean_jump is not None:
                mean_jump_vals.append(mean_jump)

            tracks_payload.append(
                {
                    "track_id": int(tid),
                    "num_clips": num_clips,
                    "track_frame_count": int(r.get("track_frame_count") or 0),
                    "stability": stability,
                    "stability_centroid_dist": stability_centroid_dist,
                    "max_temporal_jump": max_jump,
                    "mean_temporal_jump": mean_jump,
                    "clip_center_times_s": _downsample(r.get("clip_center_times_s", [])),
                    "temporal_jumps": _downsample(r.get("temporal_jumps", [])),
                }
            )

        def _avg(xs: List[float]) -> Optional[float]:
            return float(np.mean(xs)) if xs else None

        stability_hist = None
        if stability_vals:
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            hist = [0] * (len(bins) - 1)
            for v in stability_vals:
                for i in range(len(bins) - 1):
                    if bins[i] <= v <= bins[i + 1]:
                        hist[i] += 1
                        break
            stability_hist = {"bins": bins, "counts": hist}

        union_ts = metadata.get("union_timestamps_sec")
        duration_sec = None
        try:
            if isinstance(union_ts, list) and union_ts:
                duration_sec = float(union_ts[-1])
            elif isinstance(union_ts, np.ndarray) and union_ts.size:
                duration_sec = float(union_ts[-1])
        except Exception:
            duration_sec = None

        return {
            "component": self.module_name,
            "schema_version": self.SCHEMA_VERSION,
            "summary": {
                "num_tracks": int(len(results)),
                "num_clips_total": int(total_clips),
                "avg_stability": _avg(stability_vals),
                "avg_stability_centroid_dist": _avg(stability_centroid_dist_vals),
                "avg_max_temporal_jump": _avg(max_jump_vals),
                "avg_mean_temporal_jump": _avg(mean_jump_vals),
                "video_duration_sec": duration_sec,
                "stability_hist": stability_hist,
            },
            "tracks": tracks_payload,
        }

    def run(
        self,
        frames_dir: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if metadata is None:
            metadata = self.load_metadata(frames_dir)
        self._last_metadata = metadata

        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")

        frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
        if not frame_indices:
            raise ValueError(f"{self.module_name} | Нет кадров для обработки")

        # Baseline contract: stage timings
        stage_timings_ms: Dict[str, float] = {}
        t0 = time.time()

        def _resource_profile_snapshot() -> Dict[str, Any]:
            """
            Best-effort, env-gated resource snapshot for Audit 4.2.
            Does not affect output schema keys and is safe to keep optional.
            """
            if str(os.environ.get("VP_RESOURCE_PROFILE", "")).strip().lower() not in ("1", "true", "yes", "on"):
                return {}
            snap: Dict[str, Any] = {}
            try:
                import psutil  # type: ignore

                snap["rss_mb"] = float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                pass
            try:
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    snap["cuda_max_allocated_mb"] = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                    snap["cuda_max_reserved_mb"] = float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
                    snap["cuda_device"] = int(torch.cuda.current_device())
            except Exception:
                pass
            return snap

        resource_profile_before = _resource_profile_snapshot()

        frame_manager = None
        try:
            # Stage: initialization (model loading if needed)
            t_stage = time.time()
            self.initialize()  # Ensure model is loaded
            stage_timings_ms["initialization"] = (time.time() - t_stage) * 1000.0
            
            # Stage: load_deps
            t_stage = time.time()
            frame_manager = self.create_frame_manager(frames_dir, metadata)
            stage_timings_ms["load_deps"] = (time.time() - t_stage) * 1000.0
            
            self.logger.info(
                f"{self.module_name} | Начало обработки {len(frame_indices)} кадров"
            )
            
            # Emit start stage
            self._append_state_event_if_possible(
                rs_path=self.rs_path or "",
                event={
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "scope": "progress",
                    "processor": "visual",
                    "component": self.module_name,
                    "status": "running",
                    "stage": "start",
                    "platform_id": metadata.get("platform_id"),
                    "video_id": metadata.get("video_id"),
                    "run_id": metadata.get("run_id"),
                },
            )

            # Stage: process
            t_stage = time.time()
            results = self.process(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                config=config
            )
            stage_timings_ms["process"] = (time.time() - t_stage) * 1000.0
            
            # Stage: post_process (UI payload building)
            t_stage = time.time()
            ui_payload = self.build_ui_payload(results, metadata)
            stage_timings_ms["post_process"] = (time.time() - t_stage) * 1000.0

            status = "ok" if results else "empty"
            empty_reason = None if results else (self._last_empty_reason or "no_person_detections")

            # Stage: save (preparation)
            t_stage = time.time()
            save_metadata = {
                "total_frames": metadata.get("total_frames"),
                "processed_frames": len(frame_indices),
                "frames_dir": frames_dir,
                "platform_id": metadata.get("platform_id"),
                "video_id": metadata.get("video_id"),
                "run_id": metadata.get("run_id"),
                "sampling_policy_version": metadata.get("sampling_policy_version"),
                "config_hash": metadata.get("config_hash"),
                "dataprocessor_version": metadata.get("dataprocessor_version") or "unknown",
                "analysis_fps": metadata.get("analysis_fps"),
                "analysis_width": metadata.get("analysis_width"),
                "analysis_height": metadata.get("analysis_height"),
                "clip_len": self.clip_len,
                "stride": self.stride,
                "batch_size": self.batch_size,
                "embedding_dim": self.embedding_dim,
                "model_name": self.model_name,
                "processed_tracks": len(results),
                "status": status,
                "empty_reason": empty_reason,
            }

            try:
                save_metadata["models_used"] = self.get_models_used(config=config or {}, metadata=metadata or {})
            except Exception:
                save_metadata["models_used"] = []

            save_metadata["ui_payload"] = ui_payload
            save_metadata["stage_timings_ms"] = stage_timings_ms.copy()  # Copy before save timing
            if resource_profile_before:
                save_metadata["resource_profile_before"] = resource_profile_before

            if results:
                saved_path = self.save_results(
                    results=results,
                    metadata=save_metadata,
                    use_compressed=True,
                    embeddings_key="embedding_normed_256d"
                )
            else:
                empty_results = {
                    "tracks": np.asarray([], dtype=np.int32),
                    "embeddings": np.empty((0,), dtype=object),
                    "results_json": np.empty((0,), dtype=object),
                }
                saved_path = self.save_results(
                    results=empty_results,
                    metadata=save_metadata,
                    use_compressed=False
                )
            
            stage_timings_ms["save"] = (time.time() - t_stage) * 1000.0
            stage_timings_ms["total"] = (time.time() - t0) * 1000.0
            # Update meta with final timings (will be re-saved if needed, but typically meta is already saved)
            
            # Emit done stage
            self._append_state_event_if_possible(
                rs_path=self.rs_path or "",
                event={
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "scope": "progress",
                    "processor": "visual",
                    "component": self.module_name,
                    "status": status,
                    "stage": "done",
                    "platform_id": metadata.get("platform_id"),
                    "video_id": metadata.get("video_id"),
                    "run_id": metadata.get("run_id"),
                },
            )

            self.logger.info(
                f"{self.module_name} | Обработка завершена. Результаты сохранены: {saved_path}"
            )
            return saved_path
        finally:
            if frame_manager is not None:
                try:
                    frame_manager.close()
                except Exception as e:
                    self.logger.exception(
                        f"{self.module_name} | Ошибка при закрытии FrameManager: {e}"
                    )
    
    def process_batch_frames(
        self,
        video_contexts: List["VideoContext"],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Batch processing для action_recognition с гибридным подходом.
        
        Собирает клипы из всех видео, обрабатывает батчем, распределяет результаты обратно.
        
        Args:
            video_contexts: Список VideoContext для каждого видео
            config: Конфигурация модуля
            
        Returns:
            Список результатов для каждого видео
        """
        # Импортируем утилиту для batch processing
        try:
            from utils.action_recognition_batch import process_action_recognition_batch
        except ImportError:
            # Fallback: используем дефолтный process_batch
            self.logger.warning(
                f"{self.module_name} | action_recognition_batch not found, using default process_batch"
            )
            return self.process_batch(video_contexts, config)
        
        # Параметры из конфига
        max_frames_per_batch = config.get("max_frames_per_gpu_batch")
        batch_size = int(config.get("batch_size", self.batch_size))
        clip_len = int(config.get("clip_len", self.clip_len))
        stride = config.get("stride")
        if stride is None:
            stride = max(1, clip_len // 2)
        else:
            stride = int(stride)
        alpha = int(config.get("alpha", self.alpha))
        
        return process_action_recognition_batch(
            video_contexts=video_contexts,
            config=config,
            max_frames_per_batch=max_frames_per_batch,
            batch_size=batch_size,
            clip_len=clip_len,
            stride=stride,
            alpha=alpha,
            embedding_dim=self.embedding_dim,
            model_name=self.model_name,
            device=self.device,
        )