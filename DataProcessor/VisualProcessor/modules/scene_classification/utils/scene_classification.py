"""
Scene classification extractor powered by Places365 models.

Обновления:
    - Модуль приведён к интерфейсу `BaseModule` (есть `process()`, поддержка `run()`/`save_results()`).
    - Интеграция с `core_clip` оптимизирована: `embeddings.npz` загружается один раз (mmap) вместо чтения на каждый кадр.
    - Выход приведён к npz-дружелюбному формату: числовые признаки → numpy массивы, переменной длины поля → object arrays.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
from collections import defaultdict

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dp_models.manager import get_global_model_manager
from dp_models.errors import ModelManagerError

from modules.base_module import BaseModule

from utils.logger import get_logger
logger = get_logger("Places365SceneClassifier")

MODULE_NAME = "scene_classification"


def _resource_profile_snapshot() -> Dict[str, Any]:
    """
    Best-effort resource snapshot for audit/profiling.
    Enabled only when VP_RESOURCE_PROFILE=1|true|yes.
    """
    v = str(os.environ.get("VP_RESOURCE_PROFILE") or "").strip().lower()
    if v not in ("1", "true", "yes", "y", "on"):
        return {}

    out: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        rss = int(getattr(p.memory_info(), "rss", 0) or 0)
        out["rss_bytes"] = rss
        out["rss_mib"] = float(rss) / (1024.0 * 1024.0)
    except Exception:
        pass

    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                out["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                out["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            except Exception:
                pass
    except Exception:
        pass

    return out


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl` (PR-5). Backend tails this file.
    """
    try:
        from pathlib import Path as _Path

        run_rs = _Path(rs_path).resolve()
        rs_base = run_rs.parents[2]  # <rs_base>/<platform>/<video>/<run>
        runs_root = rs_base.parent
        platform_id = str(event.get("platform_id") or "")
        video_id = str(event.get("video_id") or "")
        run_id = str(event.get("run_id") or "")
        if not (platform_id and video_id and run_id):
            return
        p = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_progress(
    *,
    rs_path: str,
    platform_id: str,
    video_id: str,
    run_id: str,
    done: int,
    total: int,
    stage: str,
) -> None:
    if total <= 0:
        return
    progress = float(done) / float(total)
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": _utc_iso_now(),
            "scope": "progress",
            "processor": "visual",
            "component": MODULE_NAME,
            "status": "running",
            "progress": progress,
            "done": int(done),
            "total": int(total),
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


class Places365SceneClassifier(BaseModule):
    """
    Scene classifier built on top of Places365 checkpoints.

    The extractor accepts a list of OpenCV frames (BGR numpy arrays) and returns
    the top-K scene predictions per frame.
    """

    # Supported architectures (must be backed by dp_models specs; no downloads allowed).
    TIMM_MODELS = {
        # EfficientNet - эффективные и точные
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b1": "efficientnet_b1",
        "efficientnet_b2": "efficientnet_b2",
        "efficientnet_b3": "efficientnet_b3",
        # ConvNeXt - современная архитектура, превосходит ResNet
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext_base": "convnext_base",
        # Vision Transformers
        "vit_base_patch16_224": "vit_base_patch16_224",
        "vit_large_patch16_224": "vit_large_patch16_224",
        # RegNet - эффективные модели от Facebook
        "regnetx_002": "regnetx_002",
        "regnetx_004": "regnetx_004",
        "regnetx_006": "regnetx_006",
        # ResNet улучшенные версии
        "resnet50": "resnet50",  # через timm для лучшей оптимизации
        "resnet101": "resnet101",
    }
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]
    VERSION = "2.0.1"
    SCHEMA_VERSION = "scene_classification_npz_v2"
    # Baseline: fixed artifact filename (run_id already provides uniqueness in path).
    ARTIFACT_FILENAME = "scene_classification_features.npz"

    def __init__(
        self,
        *,
        runtime: str = "inprocess",
        triton_model_spec: str = "places365_resnet50_224_triton",
        model_arch: str = "resnet50",
        use_timm: bool = False,
        min_scene_length: int = 30,
        min_scene_seconds: Optional[float] = None,
        batch_size: int = 1,
        device: Optional[str] = None,
        gpu_memory_threshold: float = 0.9,
        log_metrics_every_n_frames: int = 10,
        # Quality improvement options
        input_size: int = 224,
        use_tta: bool = False,
        use_multi_crop: bool = False,
        temporal_smoothing: bool = False,
        smoothing_window: int = 5,
        # Advanced features
        enable_advanced_features: bool = True,
        use_clip_for_semantics: bool = True,
        label_fusion: str = "places",
        # Segmentation policy
        prefer_cut_detection_boundaries: bool = True,
        progress_every_n_frames: int = 25,
        # core-данные
        rs_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param model_arch: model architecture name
            - For Places365: 'resnet18', 'resnet50'
            - For timm (use_timm=True): 'efficientnet_b0', 'convnext_tiny', 'vit_base_patch16_224', etc.
        :param use_timm: use timm library for modern architectures (EfficientNet, ConvNeXt, ViT, etc.)
            If True, model will be pretrained on ImageNet (can be fine-tuned on Places365)
        :param top_k: number of predictions to return per frame
        :param batch_size: number of frames to process simultaneously
        :param device: torch device ('cuda', 'cpu', etc.), autodetected when None
        :param categories_path: (removed) categories are resolved via ModelManager (DP_MODELS_ROOT)
        :param cache_dir: (removed) no implicit downloads/caching; local artifacts must exist
        :param gpu_memory_threshold: BaseExtractor GPU memory threshold
        :param log_metrics_every_n_frames: resource logging cadence
        :param input_size: input image size (224, 256, 320, etc.). Larger = better accuracy, slower
        :param use_tta: enable Test-Time Augmentation (multiple augmentations + averaging)
        :param use_multi_crop: enable multi-crop inference (5 crops: center + 4 corners)
        :param temporal_smoothing: enable temporal smoothing for video sequences
        :param smoothing_window: window size for temporal smoothing (number of frames)
        :param min_scene_seconds: minimal scene length in seconds (fps‑aware). If None,
            value will be derived from ``min_scene_length`` and runtime FPS.
        :param enable_advanced_features: enable advanced features (ontology + core_clip semantics)
        :param use_clip_for_semantics: kept for compatibility; semantics is always core_clip-only (no local CLIP, no heuristics)
        """
        # BaseModule init (results store, logging, metadata helpers)
        super().__init__(rs_path=rs_path, logger_name="scene_classification", **kwargs)

        # Store both frame‑based and time‑based scene length thresholds.
        # Frame threshold is kept for backwards compatibility, but aggregation
        # logic is fps‑aware and primarily uses seconds.
        self.min_scene_length_frames = max(1, int(min_scene_length))
        self.min_scene_seconds = float(min_scene_seconds) if min_scene_seconds is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # cuDNN portability guard: on some GPU/driver combos (observed RTX 2000 Ada + torch
        # 2.4.1+cu121) cuDNN fails to initialize (CUDNN_STATUS_NOT_INITIALIZED) even in a single
        # isolated process, killing scene entirely (L13). Probe cuDNN once; on failure keep the GPU
        # but disable cuDNN (native convs still work); if GPU is unusable at all, fall back to CPU.
        if str(self.device).startswith("cuda"):
            def _probe_conv():
                _p = torch.nn.Conv2d(3, 8, 3).to(self.device)
                with torch.no_grad():
                    _p(torch.zeros(1, 3, 8, 8, device=self.device))
                torch.cuda.synchronize()
                del _p
            try:
                _probe_conv()
            except Exception as _cudnn_e:
                try:
                    torch.backends.cudnn.enabled = False
                    _probe_conv()
                    logger.warning(
                        "scene_classification | cuDNN init failed (%s) → disabled cuDNN, using GPU "
                        "native convs", str(_cudnn_e)[:80])
                except Exception as _gpu_e:
                    self.device = "cpu"
                    logger.warning(
                        "scene_classification | GPU unusable (%s) → falling back to CPU",
                        str(_gpu_e)[:80])
        self.batch_size = max(1, batch_size)
        self.input_size = input_size
        self.use_tta = use_tta
        self.use_multi_crop = use_multi_crop
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = max(1, smoothing_window)
        self.use_timm = bool(use_timm)
        
        # Advanced features policy:
        # - heuristics are forbidden (audit rule)
        # - semantics is computed strictly from core_clip embeddings + core_clip-provided prompt embeddings
        self.enable_advanced_features = bool(enable_advanced_features)
        self.use_clip_for_semantics = bool(use_clip_for_semantics)
        self.label_fusion = str(label_fusion or "places").strip().lower()
        if self.label_fusion not in ("places", "clip"):
            raise ValueError("scene_classification | label_fusion must be one of: places|clip")
        # Policy: use cut_detection hard boundaries for higher precision.
        self.prefer_cut_detection_boundaries = bool(prefer_cut_detection_boundaries)
        self.progress_every_n_frames = max(1, int(progress_every_n_frames))

        # core_clip integration (cache provider output once, not per-frame)
        self._core_clip_path: Optional[str] = None
        self._core_clip_frame_embeddings: Optional[np.ndarray] = None  # may be memmap
        self._core_clip_frame_indices: Optional[np.ndarray] = None
        self._core_clip_index_map: Optional[Dict[int, int]] = None
        self._use_core_clip = False
        if rs_path:
            core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
            if os.path.isfile(core_path):
                self._use_core_clip = True
                self._core_clip_path = core_path
                try:
                    # Load and build strict index map (union-domain frame_indices → row)
                    data = np.load(core_path, allow_pickle=True)
                    idx = data.get("frame_indices")
                    emb = data.get("frame_embeddings")
                    if idx is not None and emb is not None:
                        idx = np.asarray(idx, dtype=np.int32)
                        emb = np.asarray(emb, dtype=np.float32)
                        self._core_clip_frame_indices = idx
                        self._core_clip_frame_embeddings = emb
                        self._core_clip_index_map = {int(fi): i for i, fi in enumerate(idx.tolist())}
                except Exception as e:
                    logger.warning(
                        f"Places365SceneClassifier | core_clip preload failed: {e}. "
                        "Will fallback to per-frame loader."
                    )

        # core_clip text embeddings for scene semantics (provided by core_clip NPZ)
        self._scene_aesthetic_text_embeddings: Optional[np.ndarray] = None
        self._scene_luxury_text_embeddings: Optional[np.ndarray] = None
        self._scene_atmosphere_text_embeddings: Optional[np.ndarray] = None
        self._places365_text_embeddings: Optional[np.ndarray] = None
        self._last_core_clip_models_used: List[Dict[str, Any]] = []
        self._last_places_models_used: List[Dict[str, Any]] = []
        
        # Initialize indoor/outdoor and nature/urban mappings
        # Heuristics policy (owner decision): keyword ontologies are forbidden.
        # (indoor/outdoor, nature/urban) removed.

        self.runtime = str(runtime or "inprocess").strip().lower()
        self.triton_model_spec = str(triton_model_spec or "").strip()
        self._triton_http_url = str(kwargs.get("triton_http_url", "")).strip() if kwargs.get("triton_http_url") else None

        # --- Load Places365 via ModelManager (strict local-only) ---
        self._mm = get_global_model_manager()
        model_arch = str(model_arch or "").strip().lower()
        if self.runtime == "triton":
            if self.input_size not in (224, 336, 448):
                raise ValueError("scene_classification(triton) supports only input_size in {224,336,448} (fixed-shape Triton branches).")
            if self.use_timm:
                raise ValueError("scene_classification(triton): use_timm is not supported (baseline=Places365 ResNet50).")
            if self.use_tta or self.use_multi_crop:
                raise ValueError("scene_classification(triton): TTA/multi-crop are not supported (keep defaults).")
            if not self.triton_model_spec:
                raise ValueError("scene_classification(triton): triton_model_spec is empty.")
            
            # Try to get triton_http_url from parameter or environment
            triton_http_url = self._triton_http_url
            if not triton_http_url:
                triton_http_url = os.environ.get("TRITON_HTTP_URL")
            
            try:
                resolved = self._mm.get(model_name=self.triton_model_spec)
                rp = dict(resolved.spec.runtime_params or {})
                handle = resolved.handle or {}
                client = None
                if isinstance(handle, dict):
                    client = handle.get("client")
                
                # If client is None or rp doesn't have triton_http_url, try to create from env
                if client is None or not rp.get("triton_http_url"):
                    if triton_http_url:
                        from dp_triton import TritonHttpClient, TritonError
                        client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=60.0)
                        if not client.ready():
                            raise TritonError(
                                f"scene_classification | Triton is not ready at {triton_http_url}",
                                error_code="triton_unavailable",
                            )
                        # Update runtime_params with triton_http_url and ensure default params are set
                        if not isinstance(rp, dict):
                            rp = {}
                        rp["triton_http_url"] = str(triton_http_url)
                        # Ensure default parameters are set if missing (from places365_resnet50_224_triton.yaml spec)
                        if not rp.get("triton_model_name"):
                            rp["triton_model_name"] = "places365_resnet50_224"
                        if not rp.get("triton_model_version"):
                            rp["triton_model_version"] = "1"
                        if not rp.get("triton_input_name"):
                            rp["triton_input_name"] = "INPUT__0"
                        if not rp.get("triton_output_name"):
                            rp["triton_output_name"] = "OUTPUT__0"
                        if not rp.get("triton_input_datatype"):
                            rp["triton_input_datatype"] = "UINT8"
                        # Update handle with client
                        if isinstance(handle, dict):
                            handle["client"] = client
                        else:
                            handle = {"client": client}
                    else:
                        raise RuntimeError(
                            f"scene_classification | ModelManager returned empty Triton client handle for: {self.triton_model_spec} "
                            f"and triton_http_url not provided (set TRITON_HTTP_URL env var)"
                        )
                else:
                    # Use triton_http_url from runtime_params if available
                    if not triton_http_url and rp.get("triton_http_url"):
                        triton_http_url = str(rp.get("triton_http_url"))
                
                if client is None:
                    raise RuntimeError(f"scene_classification | Failed to get Triton client for: {self.triton_model_spec}")
                if not isinstance(rp, dict) or not rp:
                    raise RuntimeError(f"scene_classification | Places365 Triton spec has empty runtime_params")
                
                self._triton_handle = handle
                self._triton_rp = rp
            except ModelManagerError as e:
                # If ModelManager fails but we have triton_http_url, create client directly with default params
                if triton_http_url:
                    # This is expected when spec uses ${TRITON_HTTP_URL} - ModelManager doesn't expand env vars during validation
                    # We handle it gracefully with fallback
                    logger.debug(f"scene_classification | ModelManager spec validation failed for {self.triton_model_spec}: {e}, using provided triton_http_url with default places365_resnet50_224 parameters")
                    from dp_triton import TritonHttpClient, TritonError
                    client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=60.0)
                    if not client.ready():
                        raise TritonError(
                            f"scene_classification | Triton is not ready at {triton_http_url}",
                            error_code="triton_unavailable",
                        )
                    # Use default parameters for places365_resnet50_224 model (from spec_catalog/vision/places365_resnet50_224_triton.yaml)
                    rp = {
                        "triton_http_url": str(triton_http_url),
                        "triton_model_name": "places365_resnet50_224",
                        "triton_model_version": "1",
                        "triton_input_name": "INPUT__0",
                        "triton_output_name": "OUTPUT__0",
                        "triton_input_datatype": "UINT8",
                        "categories_relpath": "visual/places365/categories_places365.txt",  # Default path relative to DP_MODELS_ROOT
                    }
                    self._triton_handle = {"client": client}
                    self._triton_rp = rp
                    # Mark that we're in fallback mode (no resolved object)
                    resolved = None
                else:
                    raise RuntimeError(f"scene_classification | ModelManager failed for {self.triton_model_spec}: {e} and triton_http_url not provided (set TRITON_HTTP_URL env var)")
            self.model = None
        elif self.use_timm:
            if model_arch not in self.TIMM_MODELS:
                available = ", ".join(sorted(self.TIMM_MODELS.keys()))
                raise ValueError(f"Unsupported timm model_arch '{model_arch}'. Available: {available}")
            spec_name = f"places365_timm_{model_arch}"
        else:
            if model_arch not in ("resnet18", "resnet50"):
                raise ValueError("Unsupported Places365 model_arch (no-network). Use resnet18/resnet50, or set use_timm=true.")
            spec_name = f"places365_{model_arch}"

        if self.runtime != "triton":
            try:
                resolved = self._mm.get(model_name=spec_name)
            except ModelManagerError as e:
                raise RuntimeError(f"scene_classification | failed to load Places365 via ModelManager: {e}") from e
            self.model = resolved.handle
            self._triton_handle = None
            self._triton_rp = {}
        # Resolve categories file from spec runtime_params.categories_relpath
        rp = (resolved.spec.runtime_params or {}) if "resolved" in locals() and resolved is not None else (self._triton_rp or {})
        cat_rel = rp.get("categories_relpath")
        if not isinstance(cat_rel, str) or not cat_rel.strip():
            raise RuntimeError("scene_classification | ModelSpec missing runtime_params.categories_relpath")
        cat_abs = (resolved.resolved_artifacts.get(cat_rel) if "resolved" in locals() and resolved is not None else None)
        if not cat_abs and isinstance(self._mm, object):
            # For Triton spec, categories are a local_artifact; ModelManager should resolve it too.
            try:
                cat_abs = getattr(resolved, "resolved_artifacts", {}).get(cat_rel) if "resolved" in locals() and resolved is not None else None
            except Exception:
                cat_abs = None
        
        # Fallback: try to resolve categories file directly from DP_MODELS_ROOT
        if not cat_abs:
            dp_models_root = os.environ.get("DP_MODELS_ROOT")
            if dp_models_root and cat_rel:
                # Try to resolve relative to DP_MODELS_ROOT
                cat_abs_candidate = os.path.join(dp_models_root, "bundled_models", cat_rel)
                if os.path.isfile(cat_abs_candidate):
                    cat_abs = cat_abs_candidate
                    logger.debug(f"scene_classification | Resolved categories file from DP_MODELS_ROOT: {cat_abs}")
                else:
                    # Try alternative path (without bundled_models)
                    cat_abs_candidate = os.path.join(dp_models_root, cat_rel)
                    if os.path.isfile(cat_abs_candidate):
                        cat_abs = cat_abs_candidate
                        logger.debug(f"scene_classification | Resolved categories file from DP_MODELS_ROOT (alt): {cat_abs}")
        
        if not cat_abs:
            raise RuntimeError(f"scene_classification | categories file is not resolved: {cat_rel} (tried DP_MODELS_ROOT={os.environ.get('DP_MODELS_ROOT', 'not set')})")
        self.categories = self._parse_categories(Path(cat_abs).read_text(encoding="utf-8"))
        self._last_places_models_used = [resolved.models_used_entry] if "resolved" in locals() and resolved is not None and hasattr(resolved, "models_used_entry") else []
        
        # Base preprocessing (used for single inference)
        resize_size = int(self.input_size * 1.143)  # ~256 for 224, ~366 for 320
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD),
            ]
        )
        
        # TTA augmentations (if enabled)
        if self.use_tta:
            self.tta_transforms = [
                transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD),
                ]),
                transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD),
                ]),
            ]
        
        if self.model is not None:
            self.model.eval()
        
        # NOTE: semantics is strictly core_clip-only. If core_clip is missing, module will fail-fast in process().

    @property
    def module_name(self) -> str:
        # Keep stable module id for metadata section and results folder.
        return "scene_classification"

    def required_dependencies(self) -> List[str]:
        # Strict policy: semantics must be computed from core_clip (no local CLIP, no heuristics).
        # Additionally, segmentation uses hard cut boundaries from cut_detection (precision policy).
        return ["core_clip", "cut_detection"]

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Deterministic: include both the Places365 classifier and the upstream core_clip model mapping.
        out: List[Dict[str, Any]] = []
        if self._last_places_models_used:
            out.extend(self._last_places_models_used)
        if self._last_core_clip_models_used:
            out.extend(self._last_core_clip_models_used)
        return out

    def process(self, frame_manager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        BaseModule entrypoint.
        Returns a npz-friendly dict (numeric arrays where possible).
        """
        if len(frame_indices) < 2:
            raise RuntimeError("scene_classification | frame_indices пустой/меньше 2 (no-fallback)")

        # Enforce time-axis source-of-truth
        union_ts = frame_manager.meta.get("union_timestamps_sec")
        if not isinstance(union_ts, list) or len(union_ts) <= max(frame_indices):
            raise RuntimeError("scene_classification | missing/invalid union_timestamps_sec in frames metadata (no-fallback)")

        # Strict dependency: core_clip and cut_detection must exist.
        self._load_core_clip_dependency()
        shot_boundaries = self._load_cut_detection_boundaries()

        # Apply lightweight runtime overrides (do not rebuild model by default)
        if config:
            self._apply_runtime_config(config)

        # Apply policy defaults and overrides
        min_scene_seconds = float(self.min_scene_seconds) if self.min_scene_seconds is not None else 2.0

        # Stage timings
        t0 = time.perf_counter()

        # Run inference per frame (places or clip distribution), strict no-partial.
        per_frame = self._infer_per_frame(frame_manager=frame_manager, frame_indices=frame_indices)
        t_infer = time.perf_counter()

        # Segment into scenes using cut_detection hard shot boundaries (precision).
        agg, frame_scene_id = self._aggregate_by_shots_min_seconds(
            frame_indices=frame_indices,
            per_frame=per_frame,
            shot_boundaries_frame_indices=shot_boundaries,
            union_timestamps_sec=[float(x) for x in union_ts],
            min_scene_seconds=min_scene_seconds,
        )
        t_agg = time.perf_counter()

        # Provide stable time-axis
        times_s = np.asarray([float(union_ts[int(i)]) for i in frame_indices], dtype=np.float32)

        packed = self._pack_npz_result(
            agg,
            frame_indices=frame_indices,
            times_s=times_s,
            per_frame=per_frame,
            frame_scene_id=frame_scene_id,
        )

        # UI payload for backend (NPZ-derived, no separate JSON artifacts).
        packed["__ui_payload__"] = self._build_ui_payload(
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            times_s=times_s,
            per_frame=per_frame,
            agg=agg,
            frame_scene_id=frame_scene_id,
        )

        # Profiling summary (stored in NPZ, schema requires it).
        packed["summary"] = {
            "success": True,
            "min_scene_seconds": float(min_scene_seconds),
            "stage_timings_ms": {
                "infer_ms": float((t_infer - t0) * 1000.0),
                "aggregate_ms": float((t_agg - t_infer) * 1000.0),
            },
        }
        return packed

    def _load_core_clip_dependency(self) -> None:
        """
        Load core_clip artifacts required for semantics:
        - frame_embeddings aligned by frame_indices (union domain)
        - scene_*_text_embeddings exported by core_clip
        """
        core = self.load_core_provider("core_clip")
        if core is None:
            raise RuntimeError("scene_classification | core_clip is required but not found (no-fallback)")

        core_idx = core.get("frame_indices")
        core_emb = core.get("frame_embeddings")
        if core_idx is None or core_emb is None:
            raise RuntimeError("scene_classification | core_clip missing frame_indices/frame_embeddings (no-fallback)")

        core_idx = np.asarray(core_idx, dtype=np.int32)
        core_emb = np.asarray(core_emb, dtype=np.float32)
        self._core_clip_frame_indices = core_idx
        self._core_clip_frame_embeddings = core_emb
        self._core_clip_index_map = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
        self._use_core_clip = True

        # Required scene semantics embeddings (exported by core_clip)
        aes = core.get("scene_aesthetic_text_embeddings")
        lux = core.get("scene_luxury_text_embeddings")
        atm = core.get("scene_atmosphere_text_embeddings")
        if aes is None or lux is None or atm is None:
            raise RuntimeError("scene_classification | core_clip missing scene_*_text_embeddings (upgrade core_clip) (no-fallback)")
        self._scene_aesthetic_text_embeddings = np.asarray(aes, dtype=np.float32)
        self._scene_luxury_text_embeddings = np.asarray(lux, dtype=np.float32)
        self._scene_atmosphere_text_embeddings = np.asarray(atm, dtype=np.float32)

        # Prompt sets (for reproducibility; owner requested to expose them in this component too).
        # These arrays are small.
        self._scene_aesthetic_prompts = np.asarray(core.get("scene_aesthetic_prompts"), dtype=object) if core.get("scene_aesthetic_prompts") is not None else None
        self._scene_luxury_prompts = np.asarray(core.get("scene_luxury_prompts"), dtype=object) if core.get("scene_luxury_prompts") is not None else None
        self._scene_atmosphere_prompts = np.asarray(core.get("scene_atmosphere_prompts"), dtype=object) if core.get("scene_atmosphere_prompts") is not None else None
        self._places365_prompts = np.asarray(core.get("places365_prompts"), dtype=object) if core.get("places365_prompts") is not None else None

        # Optional (but required for label_fusion=clip): Places365 label embeddings from core_clip.
        p365 = core.get("places365_text_embeddings")
        if p365 is not None:
            self._places365_text_embeddings = np.asarray(p365, dtype=np.float32)
        elif self.label_fusion in ("clip",):
            raise RuntimeError(
                "scene_classification | label_fusion=clip requires core_clip.places365_text_embeddings "
                "(upgrade core_clip and re-run it for this rs_path) (no-fallback)"
            )

        # Capture models_used from core_clip meta for reproducibility chaining.
        meta = core.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("models_used"), list):
            self._last_core_clip_models_used = meta.get("models_used") or []

    def _apply_runtime_config(self, config: Dict[str, Any]) -> None:
        """Apply safe runtime overrides that don't require model rebuild."""
        try:
            if "min_scene_seconds" in config and config["min_scene_seconds"] is not None:
                self.min_scene_seconds = float(config["min_scene_seconds"])
            if "min_scene_length" in config and config["min_scene_length"] is not None:
                self.min_scene_length_frames = max(1, int(config["min_scene_length"]))
            if "enable_advanced_features" in config and config["enable_advanced_features"] is not None:
                self.enable_advanced_features = bool(config["enable_advanced_features"])
            if "label_fusion" in config and config["label_fusion"] is not None:
                lf = str(config["label_fusion"]).strip().lower()
                if lf not in ("places", "clip"):
                    raise ValueError("label_fusion must be one of: places|clip")
                self.label_fusion = lf
        except Exception as e:
            logger.warning(f"Places365SceneClassifier | _apply_runtime_config | Failed to apply overrides: {e}")

    def _select_distribution(
        self,
        *,
        frame_index: int,
        places_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Select final distribution over the SAME 365 labels:
        - places: Places365 supervised probs
        - clip: core_clip zero-shot probs over 365 labels

        Policy: heuristic mixing (fused) is forbidden.
        """
        mode = str(self.label_fusion or "places").strip().lower()
        pp = np.asarray(places_probs, dtype=np.float32).reshape(-1)
        if mode == "places":
            return pp

        te = self._places365_text_embeddings
        if te is None:
            raise RuntimeError("scene_classification | places365_text_embeddings missing for label_fusion (no-fallback)")
        cp = self._core_clip_probs(frame_index=frame_index, text_embeddings=np.asarray(te, dtype=np.float32))
        return cp

    def _get_core_clip_embedding(self, frame_index: int) -> Optional[np.ndarray]:
        """Fast path: use cached core_clip embeddings when available."""
        if self._core_clip_frame_embeddings is not None and self._core_clip_index_map is not None:
            try:
                pos = self._core_clip_index_map.get(int(frame_index), None)
                if pos is None:
                    return None
                emb = np.asarray(self._core_clip_frame_embeddings[int(pos)], dtype=np.float32)
                return emb
            except Exception:
                return None
        return None

    def _core_clip_probs(self, *, frame_index: int, text_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities for a set of CLIP text embeddings using core_clip image embeddings.
        Assumes both image embeddings and text_embeddings are in the same space and L2-normalized (core_clip contract).
        """
        if text_embeddings is None:
            raise RuntimeError("scene_classification | core_clip text embeddings are missing (no-fallback)")
        img = self._get_core_clip_embedding(frame_index)
        if img is None:
            raise RuntimeError(f"scene_classification | core_clip embedding missing for frame_index={frame_index} (no-fallback)")
        img = np.asarray(img, dtype=np.float32)
        img = img / (np.linalg.norm(img) + 1e-9)
        te = np.asarray(text_embeddings, dtype=np.float32)
        # (P,) logits
        logits = img @ te.T
        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        probs = exp / (float(np.sum(exp)) + 1e-9)
        return probs.astype(np.float32)

    def _core_clip_binary_score(
        self,
        *,
        frame_index: int,
        text_embeddings: Optional[np.ndarray],
        pos_indices: Tuple[int, int] = (0, 1),
    ) -> float:
        te = np.asarray(text_embeddings, dtype=np.float32) if text_embeddings is not None else None
        probs = self._core_clip_probs(frame_index=frame_index, text_embeddings=te)
        s = float(probs[int(pos_indices[0])] + probs[int(pos_indices[1])])
        return float(np.clip(s, 0.0, 1.0))

    def _core_clip_atmosphere(self, *, frame_index: int) -> Dict[str, float]:
        te = self._scene_atmosphere_text_embeddings
        probs = self._core_clip_probs(frame_index=frame_index, text_embeddings=np.asarray(te, dtype=np.float32))
        return {
            "cozy": float(probs[0]),
            "scary": float(probs[1]),
            "epic": float(probs[2]),
            "neutral": float(probs[3]),
        }

    def _pack_npz_result(
        self,
        agg: Dict[str, Any],
        *,
        frame_indices: List[int],
        times_s: np.ndarray,
        per_frame: Dict[str, np.ndarray],
        frame_scene_id: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Convert aggregated scene dict to npz-friendly payload:
        - numeric arrays for scalar features
        - object arrays for variable-length lists (indices, dominant_topk_ids, etc.)
        """
        if not agg:
            # Policy: empty is forbidden for this component -> error elsewhere.
            return {
                "frame_indices": np.asarray(frame_indices, dtype=np.int32),
                "times_s": np.asarray(times_s, dtype=np.float32),
                "scene_ids": np.asarray([], dtype=object),
                "scene_label": np.asarray([], dtype=object),
                "start_frame": np.asarray([], dtype=np.int32),
                "end_frame": np.asarray([], dtype=np.int32),
                "start_time_s": np.asarray([], dtype=np.float32),
                "end_time_s": np.asarray([], dtype=np.float32),
                "length_frames": np.asarray([], dtype=np.int32),
                "length_seconds": np.asarray([], dtype=np.float32),
                "scenes_raw": np.asarray({}, dtype=object),
            }

        scene_ids = list(agg.keys())
        scenes = [agg[sid] for sid in scene_ids]

        def f32(name: str) -> np.ndarray:
            return np.asarray([float(s.get(name, 0.0)) for s in scenes], dtype=np.float32)

        def i32(name: str) -> np.ndarray:
            return np.asarray([int(s.get(name, 0)) for s in scenes], dtype=np.int32)

        payload: Dict[str, Any] = {
            "frame_indices": np.asarray(frame_indices, dtype=np.int32),
            "times_s": np.asarray(times_s, dtype=np.float32),
            # runtime configuration snapshot (lightweight, helps audits/debug)
            "label_fusion": np.asarray(str(self.label_fusion or "places"), dtype=object),
            "min_scene_seconds": np.asarray(float(self.min_scene_seconds) if self.min_scene_seconds is not None else 2.0, dtype=np.float32),
            # per-frame compact outputs (for encoder/UI)
            "frame_topk_ids": np.asarray(per_frame["frame_topk_ids"], dtype=np.int32),
            "frame_topk_probs": np.asarray(per_frame["frame_topk_probs"], dtype=np.float32),
            "frame_entropy": np.asarray(per_frame["frame_entropy"], dtype=np.float32),
            "frame_top1_prob": np.asarray(per_frame["frame_top1_prob"], dtype=np.float32),
            "frame_top1_top2_gap": np.asarray(per_frame["frame_top1_top2_gap"], dtype=np.float32),
            "frame_scene_id": np.asarray(frame_scene_id, dtype=np.int32),
            "scene_ids": np.asarray(scene_ids, dtype=object),
            "scene_label": np.asarray([s.get("scene_label", "") for s in scenes], dtype=object),
            "fusion_mode": np.asarray([s.get("fusion_mode", None) for s in scenes], dtype=object),
            "start_frame": i32("start_frame"),
            "end_frame": i32("end_frame"),
            "start_time_s": f32("start_time_s"),
            "end_time_s": f32("end_time_s"),
            "length_frames": i32("length_frames"),
            "length_seconds": f32("length_seconds"),
            # Places metrics
            "mean_score": f32("mean_score"),
            "class_entropy_mean": f32("class_entropy_mean"),
            "top1_prob_mean": f32("top1_prob_mean"),
            "top1_vs_top2_gap_mean": f32("top1_vs_top2_gap_mean"),
            "fraction_high_confidence_frames": f32("fraction_high_confidence_frames"),
            # aesthetics / luxury
            "mean_aesthetic_score": f32("mean_aesthetic_score"),
            "aesthetic_std": f32("aesthetic_std"),
            "aesthetic_frac_high": f32("aesthetic_frac_high"),
            "mean_luxury_score": f32("mean_luxury_score"),
            # atmosphere
            "mean_cozy": f32("mean_cozy"),
            "mean_scary": f32("mean_scary"),
            "mean_epic": f32("mean_epic"),
            "mean_neutral": f32("mean_neutral"),
            "atmosphere_entropy": f32("atmosphere_entropy"),
            # stability
            "scene_change_score": f32("scene_change_score"),
            "label_stability": f32("label_stability"),
            # canonical raw mapping for downstream modules
            "scenes": np.asarray(agg, dtype=object),
            # legacy alias (kept)
            "scenes_raw": np.asarray(agg, dtype=object),
            # prompts (reproducibility)
            "scene_aesthetic_prompts": np.asarray(self._scene_aesthetic_prompts if getattr(self, "_scene_aesthetic_prompts", None) is not None else [], dtype=object),
            "scene_luxury_prompts": np.asarray(self._scene_luxury_prompts if getattr(self, "_scene_luxury_prompts", None) is not None else [], dtype=object),
            "scene_atmosphere_prompts": np.asarray(self._scene_atmosphere_prompts if getattr(self, "_scene_atmosphere_prompts", None) is not None else [], dtype=object),
            "places365_prompts": np.asarray(self._places365_prompts if getattr(self, "_places365_prompts", None) is not None else [], dtype=object),
        }
        
        # variable-length lists - ensure 1D object arrays
        # Explicitly create 1D object arrays to prevent numpy from creating 2D arrays
        # when all lists have the same length
        indices_list = [s.get("indices", []) for s in scenes]
        dominant_ids_list = [s.get("dominant_places_topk_ids", []) for s in scenes]
        dominant_probs_list = [s.get("dominant_places_topk_probs", []) for s in scenes]
        
        indices_arr = np.empty(len(indices_list), dtype=object)
        for i, item in enumerate(indices_list):
            indices_arr[i] = item
        
        dominant_ids_arr = np.empty(len(dominant_ids_list), dtype=object)
        for i, item in enumerate(dominant_ids_list):
            dominant_ids_arr[i] = item
        
        dominant_probs_arr = np.empty(len(dominant_probs_list), dtype=object)
        for i, item in enumerate(dominant_probs_list):
            dominant_probs_arr[i] = item
        
        # Add variable-length lists after payload dict is created
        payload["indices"] = indices_arr
        payload["dominant_places_topk_ids"] = dominant_ids_arr
        payload["dominant_places_topk_probs"] = dominant_probs_arr
        
        return payload

    def classify(self, frame_manager, frame_indices) -> List[Optional[Dict[str, Any]]]:
        """
        Runs scene classification over the provided frames.

        Returns a list of dicts, one per frame:
            { "label": str, "score": float }
        """

        if self.runtime == "triton":
            return self._classify_triton(frame_manager, frame_indices)

        # In-process path: return compact per-frame predictions (top-K + stats), no full 365-class dict list.
        raw_predictions: List[Optional[Dict[str, Any]]] = [None] * len(frame_indices)
        batch_tensors: List[torch.Tensor] = []
        batch_pos: List[int] = []

        def flush_batch() -> None:
            if not batch_pos:
                return
            batch_tensor = torch.cat(batch_tensors, dim=0)
            batch_preds = self._infer_batch_compact(batch_tensor)  # list[dict] aligned to batch_pos
            if len(batch_preds) != len(batch_pos):
                raise RuntimeError("scene_classification(inprocess) | batch output mismatch (no-fallback)")
            for i_local, pos in enumerate(batch_pos):
                raw_predictions[int(pos)] = batch_preds[i_local]
            batch_tensors.clear()
            batch_pos.clear()

        for pos, frame_idx in enumerate(frame_indices):
            frame = frame_manager.get(frame_idx)
            if frame is None:
                raise RuntimeError(f"scene_classification | missing frame {frame_idx} from FrameManager (no-fallback)")
            tensors = self._prepare_frame(frame)
            if tensors is None:
                raise RuntimeError(f"scene_classification | failed to preprocess frame {frame_idx} (no-fallback)")

            # Multi-crop/TTA returns list of tensors. Policy: allowed as option.
            if isinstance(tensors, list):
                # For simplicity we average logits later by running multiple tensors and selecting max top1.
                # This is deterministic but higher cost. Kept only for dev.
                for tensor in tensors:
                    batch_tensors.append(tensor)
                    batch_pos.append(int(pos))
            else:
                batch_tensors.append(tensors)
                batch_pos.append(int(pos))

            if len(batch_pos) >= int(self.batch_size):
                flush_batch()
        flush_batch()

        # If multi-crop/TTA produced duplicates for a frame position, pick the one with max top1_prob.
        if any(p is None for p in raw_predictions):
            # Multi-crop may overwrite, but missing means inference didn't run.
            missing = [i for i, p in enumerate(raw_predictions) if p is None][:5]
            raise RuntimeError(f"scene_classification(inprocess) | missing predictions for positions: {missing} (no-fallback)")

        return raw_predictions

    __call__ = classify

    def _preprocess_places365_u8_nhwc_fixed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Match torchvision pipeline (Resize ~256 on shorter side + CenterCrop 224),
        but output UINT8 NHWC for Triton ensemble input.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return None

        # FrameManager.get() contract: RGB
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            except Exception:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = frame

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        if h <= 0 or w <= 0:
            return None

        # torchvision.transforms.Resize(resize_size) keeps aspect ratio (smaller edge -> resize_size)
        resize_size = int(self.input_size * 1.143)  # same heuristic as inprocess path
        min_edge = min(h, w)
        if min_edge != resize_size:
            scale = float(resize_size) / float(min_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = int(rgb.shape[0]), int(rgb.shape[1])

        crop = int(self.input_size)
        if h < crop or w < crop:
            # Safety: if resize produced smaller due to rounding, pad by resizing to exact crop.
            rgb = cv2.resize(rgb, (crop, crop), interpolation=cv2.INTER_LINEAR)
            h, w = crop, crop

        y0 = max(0, (h - crop) // 2)
        x0 = max(0, (w - crop) // 2)
        rgb = rgb[y0 : y0 + crop, x0 : x0 + crop, :]

        # Triton expects (1,S,S,3) where S is fixed branch size
        return np.expand_dims(rgb, axis=0)

    def _softmax_np(self, logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float32).reshape(-1)
        x = x - float(np.max(x))
        ex = np.exp(x)
        return (ex / (float(np.sum(ex)) + 1e-9)).astype(np.float32)

    def _classify_triton(self, frame_manager, frame_indices) -> List[Optional[Dict[str, Any]]]:
        if not isinstance(self._triton_handle, dict) or "client" not in self._triton_handle:
            raise RuntimeError("scene_classification(triton) | triton handle is missing (no-fallback)")

        rp = self._triton_rp or {}
        model_name = str(rp.get("triton_model_name") or self._triton_handle.get("triton_model_name") or "").strip()
        model_version = str(rp.get("triton_model_version") or "1").strip()
        input_name = str(rp.get("triton_input_name") or "INPUT__0").strip()
        input_dt = str(rp.get("triton_input_datatype") or "UINT8").strip().upper()
        output_name = str(rp.get("triton_output_name") or "OUTPUT__0").strip()
        output_dt = str(rp.get("triton_output_datatype") or "FP32").strip().upper()
        if not model_name:
            raise RuntimeError("scene_classification(triton) | missing triton_model_name (no-fallback)")

        client = self._triton_handle["client"]

        raw_predictions: List[Optional[Dict[str, Any]]] = [None] * len(frame_indices)
        batch_x: List[np.ndarray] = []
        batch_pos: List[int] = []

        def flush_batch() -> None:
            if not batch_pos:
                return
            try:
                inp = np.concatenate(batch_x, axis=0)  # (B,S,S,3) uint8
            except Exception as e:
                raise RuntimeError(f"scene_classification(triton) | failed to stack batch: {e}") from e

            try:
                res = client.infer(
                    model_name=model_name,
                    model_version=model_version,
                    input_name=input_name,
                    input_tensor=inp.astype(np.uint8, copy=False),
                    output_name=output_name,
                    datatype=input_dt,
                )
            except Exception as e:
                raise RuntimeError(f"scene_classification(triton) | infer failed: {e}") from e

            out = np.asarray(res.output, dtype=np.float32)
            if out.ndim == 1:
                out = out.reshape(1, -1)
            elif out.ndim > 2:
                # Best-effort flatten per batch item (some backends may return extra dims)
                out = out.reshape(out.shape[0], -1)
            if out.shape[0] != len(batch_pos):
                raise RuntimeError(
                    f"scene_classification(triton) | output batch mismatch: outB={out.shape[0]} expectedB={len(batch_pos)}"
                )

            for i_local, pos in enumerate(batch_pos):
                logits = np.asarray(out[i_local], dtype=np.float32).reshape(-1)
                probs = self._softmax_np(logits)

                # Optional selection (places | clip) in the SAME 365-label space.
                frame_idx = int(frame_indices[int(pos)])
                final_probs = self._select_distribution(
                    frame_index=frame_idx,
                    places_probs=probs,
                )

                # Final confidence stats (after fusion)
                entropy_val = float(-np.sum(final_probs * np.log(final_probs + 1e-8)))
                top_k = int(min(5, final_probs.shape[0]))
                topk_idx = np.argpartition(-final_probs, top_k - 1)[:top_k]
                topk_idx = topk_idx[np.argsort(-final_probs[topk_idx])]
                topk_probs = final_probs[topk_idx]

                top1_idx = int(topk_idx[0])
                top1_prob_val = float(topk_probs[0])
                top2_prob_val = float(topk_probs[1]) if top_k > 1 else 0.0
                top1_top2_gap = float(max(0.0, top1_prob_val - top2_prob_val))

                label = self.categories[top1_idx] if 0 <= top1_idx < len(self.categories) else f"class_{top1_idx}"
                pred: Dict[str, Any] = {
                    "label": label,
                    "score": top1_prob_val,
                    "entropy": entropy_val,
                    "top1_prob": top1_prob_val,
                    "top2_prob": top2_prob_val,
                    "top1_top2_gap": top1_top2_gap,
                    "class_idx": top1_idx,
                    "topk_class_indices": [int(i) for i in topk_idx.tolist()],
                    "topk_class_probs": [float(v) for v in topk_probs.tolist()],
                    "fusion_mode": str(self.label_fusion or "places"),
                }
                raw_predictions[pos] = pred

            batch_x.clear()
            batch_pos.clear()

        for pos, frame_idx in enumerate(frame_indices):
            frame = frame_manager.get(frame_idx)
            x = self._preprocess_places365_u8_nhwc_fixed(frame)
            if x is None:
                raise RuntimeError(f"scene_classification(triton) | failed to preprocess frame {frame_idx} (no-fallback)")
            batch_x.append(x)
            batch_pos.append(int(pos))
            if len(batch_pos) >= int(max(1, self.batch_size)):
                flush_batch()

        flush_batch()

        if any(p is None for p in raw_predictions):
            missing = [i for i, p in enumerate(raw_predictions) if p is None][:5]
            raise RuntimeError(f"scene_classification(triton) | missing predictions at positions={missing} (no-fallback)")
        return raw_predictions


    def _prepare_frame(self, frame: np.ndarray) -> Optional[torch.Tensor | List[torch.Tensor]]:
        if frame is None or not isinstance(frame, np.ndarray):
            logger.warning("Frame is not a numpy array – skipping")
            return None

        # FrameManager.get() contract: RGB
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            # Most likely RGBA; keep robust fallback for BGRA just in case.
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            except Exception:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = frame  # assume already RGB

        image = Image.fromarray(rgb.astype(np.uint8))

        # Multi-crop: 5 crops (center + 4 corners)
        if self.use_multi_crop:
            crops = self._get_multi_crops(image)
            tensors = []
            for crop in crops:
                tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
                tensors.append(tensor)
            return tensors

        # TTA: multiple augmentations
        if self.use_tta:
            tensors = []
            for transform in self.tta_transforms:
                tensor = transform(image).unsqueeze(0).to(self.device)
                tensors.append(tensor)
            return tensors

        # Standard single inference
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return tensor

    def _infer_batch_compact(self, tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Inprocess inference: returns compact per-sample predictions:
        top-K + entropy + gaps. No per-class dict list (keeps CPU/memory sane).
        """
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        out: List[Dict[str, Any]] = []
        for sample_probs in probs:
            entropy_val = float(-torch.sum(sample_probs * torch.log(sample_probs + 1e-8)).item())
            top_k = int(min(5, int(sample_probs.shape[0])))
            topk_vals, topk_indices = torch.topk(sample_probs, k=top_k)
            topk_indices_list = [int(i.item()) for i in topk_indices]
            topk_probs_list = [float(v.item()) for v in topk_vals]
            top1_idx = int(topk_indices_list[0])
            top1_prob_val = float(topk_probs_list[0])
            top2_prob_val = float(topk_probs_list[1]) if top_k > 1 else 0.0
            top1_top2_gap = float(max(0.0, top1_prob_val - top2_prob_val))
            label = self.categories[top1_idx] if 0 <= top1_idx < len(self.categories) else f"class_{top1_idx}"
            out.append(
                {
                    "label": label,
                    "score": top1_prob_val,
                    "entropy": entropy_val,
                    "top1_prob": top1_prob_val,
                    "top2_prob": top2_prob_val,
                    "top1_top2_gap": top1_top2_gap,
                    "class_idx": top1_idx,
                    "topk_class_indices": topk_indices_list,
                    "topk_class_probs": topk_probs_list,
                    "fusion_mode": str(self.label_fusion or "places"),
                }
            )
        return out


    def _get_multi_crops(self, image: Image.Image) -> List[Image.Image]:
        """Generate 5 crops: center + 4 corners."""
        width, height = image.size
        # Use input_size as crop size, but ensure it fits
        crop_size = min(self.input_size, width, height)
        crops = []
        
        # Center crop
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        crops.append(image.crop((left, top, left + crop_size, top + crop_size)))
        
        # 4 corner crops
        corners = [
            (0, 0),  # top-left
            (width - crop_size, 0),  # top-right
            (0, height - crop_size),  # bottom-left
            (width - crop_size, height - crop_size),  # bottom-right
        ]
        for x, y in corners:
            if x >= 0 and y >= 0 and x + crop_size <= width and y + crop_size <= height:
                crops.append(image.crop((x, y, x + crop_size, y + crop_size)))
        
        return crops[:5]  # Ensure exactly 5 crops

    def _average_predictions(
        self, predictions_list: List[List[Dict[str, Any]]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Average multiple predictions (from TTA or multi-crop)."""
        # Aggregate scores by label
        label_scores: Dict[str, List[float]] = {}
        
        for preds in predictions_list:
            for pred in preds:
                label = pred["label"]
                score = pred["score"]
                if label not in label_scores:
                    label_scores[label] = []
                label_scores[label].append(score)
        
        # Average scores and sort
        averaged = [
            {"label": label, "score": sum(scores) / len(scores)}
            for label, scores in label_scores.items()
        ]
        averaged.sort(key=lambda x: x["score"], reverse=True)
        
        return averaged[:top_k]

    def _apply_temporal_smoothing(
        self, predictions: List[List[Dict[str, Any]]], top_k: int
    ) -> List[List[Dict[str, Any]]]:
        """Apply temporal smoothing using moving average."""
        if len(predictions) == 0:
            return predictions
        
        smoothed: List[List[Dict[str, Any]]] = []
        window = self.smoothing_window
        
        for i in range(len(predictions)):
            # Get window of frames
            start = max(0, i - window // 2)
            end = min(len(predictions), i + window // 2 + 1)
            window_predictions = predictions[start:end]
            
            # Aggregate scores across window
            label_scores: Dict[str, List[float]] = {}
            for frame_preds in window_predictions:
                for pred in frame_preds:
                    label = pred["label"]
                    score = pred["score"]
                    if label not in label_scores:
                        label_scores[label] = []
                    label_scores[label].append(score)
            
            # Average and sort
            averaged = [
                {"label": label, "score": sum(scores) / len(scores)}
                for label, scores in label_scores.items()
            ]
            averaged.sort(key=lambda x: x["score"], reverse=True)
            smoothed.append(averaged[:top_k])
        
        return smoothed

    def _smooth_topk_over_time(self, preds: List[Optional[Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        """
        Temporal smoothing for Places365 predictions using windowed top-K voting.

        This is designed for the baseline path where `preds[i]` is already "best-only" (top1)
        but carries `topk_class_indices/topk_class_probs`.

        Output:
        - preserves list alignment (keeps None)
        - overwrites `label/class_idx/score/top1_prob/top2_prob/top1_top2_gap/topk_*` with smoothed values
        - keeps raw label as `label_raw` for debugging (best-effort)
        """
        if not preds:
            return preds
        w = int(getattr(self, "smoothing_window", 1) or 1)
        w = max(1, w)
        if w <= 1:
            return preds

        out: List[Optional[Dict[str, Any]]] = [None] * len(preds)
        for i in range(len(preds)):
            if preds[i] is None:
                out[i] = None
                continue

            start = max(0, i - w // 2)
            end = min(len(preds), i + w // 2 + 1)
            weight_by_class: Dict[int, float] = {}
            for j in range(start, end):
                pj = preds[j]
                if pj is None:
                    continue
                # Reliability weight: downweight uncertain frames so they don't dominate smoothing.
                try:
                    top1p = float(pj.get("top1_prob", pj.get("score", 0.0)) or 0.0)
                except Exception:
                    top1p = 0.0
                try:
                    gap = float(pj.get("top1_top2_gap", 0.0) or 0.0)
                except Exception:
                    gap = 0.0
                # Heuristic: strong if top1 high and/or gap high. Keep a small floor to avoid zeroing.
                reliability = max(0.05, min(1.0, top1p + 0.5 * gap))
                tk_idx = pj.get("topk_class_indices")
                tk_prob = pj.get("topk_class_probs")
                if isinstance(tk_idx, list) and isinstance(tk_prob, list) and len(tk_idx) == len(tk_prob) and len(tk_idx) > 0:
                    for cid, p in zip(tk_idx, tk_prob):
                        try:
                            c = int(cid)
                            pv = float(p) * reliability
                        except Exception:
                            continue
                        weight_by_class[c] = weight_by_class.get(c, 0.0) + max(0.0, pv)
                else:
                    try:
                        c = int(pj.get("class_idx"))
                        pv = float(pj.get("top1_prob", pj.get("score", 0.0))) * reliability
                        weight_by_class[c] = weight_by_class.get(c, 0.0) + max(0.0, pv)
                    except Exception:
                        continue

            if not weight_by_class:
                out[i] = preds[i]
                continue

            sorted_items = sorted(weight_by_class.items(), key=lambda x: x[1], reverse=True)
            total_w = float(sum(w for _, w in sorted_items)) or 1.0
            topk_ids = [int(cid) for cid, _ in sorted_items[:5]]
            topk_probs = [float(w / total_w) for _, w in sorted_items[:5]]

            top1_id = int(topk_ids[0])
            top1_prob = float(topk_probs[0])
            top2_prob = float(topk_probs[1]) if len(topk_probs) > 1 else 0.0
            top1_top2_gap = float(max(0.0, top1_prob - top2_prob))
            label = self.categories[top1_id] if 0 <= top1_id < len(self.categories) else f"class_{top1_id}"

            cur = dict(preds[i])
            cur.setdefault("label_raw", cur.get("label"))
            cur.setdefault("class_idx_raw", cur.get("class_idx"))
            cur["label"] = label
            cur["class_idx"] = top1_id
            cur["score"] = top1_prob
            cur["top1_prob"] = top1_prob
            cur["top2_prob"] = top2_prob
            cur["top1_top2_gap"] = top1_top2_gap
            cur["topk_class_indices"] = topk_ids
            cur["topk_class_probs"] = topk_probs
            out[i] = cur

        return out

    @staticmethod
    def _parse_categories(raw_text: str) -> List[str]:
        categories: List[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # Places365 categories file often uses hierarchical names like:
            #   /a/apartment_building/outdoor 8
            # We must keep subcategories (apartment_building/outdoor), not truncate to the last token ("outdoor").
            raw_label = " ".join(parts[:-1]).strip()
            segs = raw_label.split("/")
            # Typical: ["", "a", "apartment_building", "outdoor"] -> join from index 2.
            if len(segs) >= 3 and segs[0] == "":
                label = "/".join(segs[2:])
            else:
                label = raw_label.lstrip("/").strip()
            categories.append(label)
        return categories
    
    def _load_cut_detection_boundaries(self) -> List[int]:
        """
        Load hard shot boundaries from cut_detection artifact.
        Policy: scene segmentation uses hard boundaries for precision.
        """
        dep = self.load_dependency_results("cut_detection")
        if dep is None:
            raise RuntimeError("scene_classification | cut_detection is required but not found (no-fallback)")
        det = dep.get("detections")
        if isinstance(det, np.ndarray) and det.dtype == object and det.shape == ():
            det = det.item()
        if not isinstance(det, dict):
            raise RuntimeError("scene_classification | cut_detection.detections missing/invalid (no-fallback)")
        b = det.get("shot_boundaries_frame_indices")
        if not isinstance(b, list) or not b:
            raise RuntimeError("scene_classification | cut_detection missing shot_boundaries_frame_indices (no-fallback)")
        out = [int(x) for x in b]
        out = sorted(set(out))
        if len(out) < 2:
            raise RuntimeError("scene_classification | cut_detection shot boundaries too short (no-fallback)")
        return out

    def _infer_per_frame(self, *, frame_manager, frame_indices: List[int]) -> Dict[str, np.ndarray]:
        """
        Runs Places365 inference for each frame index and returns compact per-frame arrays.
        Policy: no partial results -> any failure is error.
        """
        if self.rs_path is None:
            raise RuntimeError("scene_classification | rs_path is required (no-fallback)")
        meta = frame_manager.meta or {}
        platform_id = str(meta.get("platform_id") or "")
        video_id = str(meta.get("video_id") or "")
        run_id = str(meta.get("run_id") or "")

        preds = self.classify(frame_manager, frame_indices)  # list aligned with frame_indices
        # Optional temporal smoothing (explicit knob). This is an algorithmic option, not a silent heuristic.
        if bool(getattr(self, "temporal_smoothing", False)):
            try:
                preds = self._smooth_topk_over_time(preds)  # type: ignore[assignment]
            except Exception as e:
                raise RuntimeError(f"scene_classification | temporal smoothing failed (no-fallback): {e}") from e
        if len(preds) != len(frame_indices):
            raise RuntimeError("scene_classification | classify() output mismatch (no-fallback)")

        K = 5
        frame_topk_ids = np.full((len(frame_indices), K), -1, dtype=np.int32)
        frame_topk_probs = np.full((len(frame_indices), K), np.nan, dtype=np.float32)
        frame_entropy = np.full((len(frame_indices),), np.nan, dtype=np.float32)
        frame_top1_prob = np.full((len(frame_indices),), np.nan, dtype=np.float32)
        frame_top1_top2_gap = np.full((len(frame_indices),), np.nan, dtype=np.float32)
        frame_top1_id = np.full((len(frame_indices),), -1, dtype=np.int32)

        for i, (fi, p) in enumerate(zip(frame_indices, preds)):
            if p is None:
                raise RuntimeError("scene_classification | got None prediction (no-fallback)")
            try:
                tk_idx = p.get("topk_class_indices")
                tk_prob = p.get("topk_class_probs")
                ent = float(p.get("entropy"))
                top1 = float(p.get("top1_prob", p.get("score")))
                gap = float(p.get("top1_top2_gap", 0.0))
            except Exception as e:
                raise RuntimeError(f"scene_classification | invalid prediction dict at i={i}: {e}") from e

            if not (isinstance(tk_idx, list) and isinstance(tk_prob, list) and len(tk_idx) == len(tk_prob) and len(tk_idx) > 0):
                raise RuntimeError("scene_classification | missing topk fields (no-fallback)")

            kk = min(K, len(tk_idx))
            frame_topk_ids[i, :kk] = np.asarray([int(x) for x in tk_idx[:kk]], dtype=np.int32)
            frame_topk_probs[i, :kk] = np.asarray([float(x) for x in tk_prob[:kk]], dtype=np.float32)
            frame_entropy[i] = float(ent)
            frame_top1_prob[i] = float(top1)
            frame_top1_top2_gap[i] = float(gap)
            frame_top1_id[i] = int(frame_topk_ids[i, 0])

            if (i + 1) % int(self.progress_every_n_frames) == 0:
                _emit_progress(
                    rs_path=str(self.rs_path),
                    platform_id=platform_id,
                    video_id=video_id,
                    run_id=run_id,
                    done=int(i + 1),
                    total=int(len(frame_indices)),
                    stage="infer",
                )

        return {
            "frame_topk_ids": frame_topk_ids,
            "frame_topk_probs": frame_topk_probs,
            "frame_entropy": frame_entropy,
            "frame_top1_prob": frame_top1_prob,
            "frame_top1_top2_gap": frame_top1_top2_gap,
            "frame_top1_id": frame_top1_id,
        }

    def _aggregate_by_shots_min_seconds(
        self,
        *,
        frame_indices: List[int],
        per_frame: Dict[str, np.ndarray],
        shot_boundaries_frame_indices: List[int],
        union_timestamps_sec: List[float],
        min_scene_seconds: float,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Build scenes by accumulating consecutive shots until duration >= min_scene_seconds.
        This is deterministic and uses hard cut boundaries for precision.
        """
        fi = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        if fi.size == 0:
            raise RuntimeError("scene_classification | empty frame_indices (no-fallback)")
        bounds = sorted(set(int(x) for x in shot_boundaries_frame_indices))
        if len(bounds) < 2:
            raise RuntimeError("scene_classification | invalid shot boundaries (no-fallback)")
        ts = np.asarray(union_timestamps_sec, dtype=np.float32).reshape(-1)
        if int(np.max(fi)) >= int(ts.size):
            raise RuntimeError("scene_classification | frame_indices out of bounds for union_timestamps_sec")

        # Assign shot_id per frame index by rightmost boundary <= fi
        b_arr = np.asarray(bounds, dtype=np.int32)
        shot_id = np.searchsorted(b_arr, fi, side="right") - 1
        if np.any(shot_id < 0):
            raise RuntimeError("scene_classification | some frames are before first shot boundary (no-fallback)")

        # Build list of shot windows in time
        shot_start_frames = b_arr
        shot_start_times = ts[shot_start_frames.astype(np.int64)]
        # End time is next shot start time (exclusive), last uses the end of union timeline.
        # We use union_timestamps_sec[-1] as the best available end marker.
        shot_end_times = np.concatenate(
            [
                shot_start_times[1:],
                np.asarray([float(ts[-1])], dtype=np.float32),
            ]
        )
        # Ensure monotonic
        for i in range(len(shot_end_times) - 1):
            if float(shot_end_times[i]) > float(shot_end_times[i + 1]) + 1e-6:
                raise RuntimeError("scene_classification | non-monotonic shot boundaries time axis (no-fallback)")

        # For each shot, collect positions in fi belonging to it
        pos_by_shot: Dict[int, List[int]] = {}
        for pos, sid in enumerate(shot_id.tolist()):
            pos_by_shot.setdefault(int(sid), []).append(int(pos))

        # Accumulate shots into scenes
        scenes: Dict[str, Any] = {}
        frame_scene_id = np.full((int(fi.size),), -1, dtype=np.int32)

        cur_scene_shots: List[int] = []
        cur_start_shot = int(shot_id[0])
        cur_start_time = float(shot_start_times[cur_start_shot])
        cur_end_time = cur_start_time
        scene_idx = 0

        def finalize(shots_list: List[int], start_time: float, end_time: float) -> None:
            nonlocal scene_idx
            if not shots_list:
                return
            # Collect frame positions in these shots (may be empty -> error)
            frame_pos: List[int] = []
            for sid in shots_list:
                frame_pos.extend(pos_by_shot.get(int(sid), []))
            if not frame_pos:
                raise RuntimeError("scene_classification | scene window has zero sampled frames (no-fallback)")
            frame_pos = sorted(frame_pos)
            indices = [int(fi[p]) for p in frame_pos]

            # Dominant label via accumulated top1 probs (more stable than hard majority)
            top1_ids = per_frame["frame_top1_id"][frame_pos].astype(np.int32)
            top1_probs = per_frame["frame_top1_prob"][frame_pos].astype(np.float32)
            weight_by_class: Dict[int, float] = {}
            for cid, w in zip(top1_ids.tolist(), top1_probs.tolist()):
                if int(cid) < 0 or not np.isfinite(float(w)):
                    continue
                weight_by_class[int(cid)] = weight_by_class.get(int(cid), 0.0) + float(w)
            if not weight_by_class:
                raise RuntimeError("scene_classification | cannot determine scene label (no-fallback)")
            top1_id = int(max(weight_by_class.items(), key=lambda x: x[1])[0])
            label = self.categories[top1_id] if 0 <= top1_id < len(self.categories) else f"class_{top1_id}"

            # Fill frame_scene_id
            for p in frame_pos:
                frame_scene_id[int(p)] = int(scene_idx)

            # Aggregates
            ent = per_frame["frame_entropy"][frame_pos].astype(np.float32)
            gap = per_frame["frame_top1_top2_gap"][frame_pos].astype(np.float32)
            top1p = per_frame["frame_top1_prob"][frame_pos].astype(np.float32)
            mean_score = float(np.mean(top1p))
            class_entropy_mean = float(np.mean(ent))
            top1_prob_mean = float(np.mean(top1p))
            top1_vs_top2_gap_mean = float(np.mean(gap))
            fraction_high_confidence_frames = float(np.mean(top1p > 0.7))

            # Label stability
            label_stability = float(np.mean(top1_ids == top1_id))

            # Dominant topK ids aggregated over scene
            tk_ids = per_frame["frame_topk_ids"][frame_pos, :]
            tk_probs = per_frame["frame_topk_probs"][frame_pos, :]
            dom_w: Dict[int, float] = {}
            for row_ids, row_p in zip(tk_ids, tk_probs):
                for cid, pv in zip(row_ids.tolist(), row_p.tolist()):
                    if int(cid) < 0 or not np.isfinite(float(pv)):
                        continue
                    dom_w[int(cid)] = dom_w.get(int(cid), 0.0) + float(pv)
            sorted_items = sorted(dom_w.items(), key=lambda x: x[1], reverse=True)[:5]
            dominant_ids = [int(cid) for cid, _ in sorted_items]
            dominant_probs = [float(w) for _, w in sorted_items]

            start_frame = int(indices[0])
            end_frame = int(indices[-1])
            length_seconds = float(max(0.0, end_time - start_time))
            length_frames = int(len(indices))

            scene_id = f"s{scene_idx:04d}"
            scenes[scene_id] = {
                "scene_label": str(label),
                "indices": indices,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": float(start_time),
                "end_time_s": float(end_time),
                "length_frames": length_frames,
                "length_seconds": length_seconds,
                "fusion_mode": str(self.label_fusion or "places"),
                "mean_score": mean_score,
                "class_entropy_mean": class_entropy_mean,
                "top1_prob_mean": top1_prob_mean,
                "top1_vs_top2_gap_mean": top1_vs_top2_gap_mean,
                "fraction_high_confidence_frames": fraction_high_confidence_frames,
                "label_stability": label_stability,
                "scene_change_score": float(np.std(top1p)) if top1p.size else 0.0,
                "dominant_places_topk_ids": dominant_ids,
                "dominant_places_topk_probs": dominant_probs,
                # Semantics are optional and computed elsewhere (enable_advanced_features)
                "mean_aesthetic_score": 0.0,
                "aesthetic_std": 0.0,
                "aesthetic_frac_high": 0.0,
                "mean_luxury_score": 0.0,
                "mean_cozy": 0.0,
                "mean_scary": 0.0,
                "mean_epic": 0.0,
                "mean_neutral": 0.0,
                "atmosphere_entropy": 0.0,
            }
            scene_idx += 1

        # Iterate shots in order
        max_shot = int(np.max(shot_id))
        cur_shot = int(cur_start_shot)
        while cur_shot <= max_shot:
            cur_scene_shots.append(int(cur_shot))
            # end_time = next shot start time if exists, else last time
            if cur_shot < len(shot_end_times):
                cur_end_time = float(shot_end_times[cur_shot])
            else:
                cur_end_time = float(ts[int(fi[-1])])
            if (cur_end_time - cur_start_time) >= float(min_scene_seconds) and cur_scene_shots:
                finalize(cur_scene_shots, cur_start_time, cur_end_time)
                # start new
                cur_scene_shots = []
                cur_shot += 1
                if cur_shot <= max_shot:
                    cur_start_time = float(shot_start_times[cur_shot])
                continue
            cur_shot += 1

        # tail: if leftovers, merge into last scene (precision-oriented; avoids tiny tail)
        if cur_scene_shots:
            if scenes:
                # merge leftover frames into last scene by extending its indices and end_time.
                last_id = f"s{(scene_idx - 1):04d}"
                last = scenes[last_id]
                frame_pos: List[int] = []
                for sid in cur_scene_shots:
                    frame_pos.extend(pos_by_shot.get(int(sid), []))
                if frame_pos:
                    frame_pos = sorted(frame_pos)
                    last["indices"].extend([int(fi[p]) for p in frame_pos])
                    last["indices"] = sorted(set(last["indices"]))
                    last["end_frame"] = int(last["indices"][-1])
                    # update end_time to end of last shot
                    endt = float(shot_end_times[int(cur_scene_shots[-1])]) if int(cur_scene_shots[-1]) < len(shot_end_times) else float(ts[int(fi[-1])])
                    last["end_time_s"] = endt
                    last["length_frames"] = int(len(last["indices"]))
                    last["length_seconds"] = float(max(0.0, float(last["end_time_s"]) - float(last["start_time_s"])))
                    for p in frame_pos:
                        frame_scene_id[int(p)] = int(scene_idx - 1)
                else:
                    # no sampled frames in leftover -> ok to ignore, but time would be unrepresented
                    pass
            else:
                raise RuntimeError("scene_classification | video too short to form a scene (no-fallback)")

        if np.any(frame_scene_id < 0):
            # Some frames were not assigned to any scene -> error
            bad = int(np.sum(frame_scene_id < 0))
            raise RuntimeError(f"scene_classification | {bad} frames not assigned to any scene (no-fallback)")

        # Optional semantics per scene (model-based, not heuristics)
        if bool(self.enable_advanced_features) and bool(self.use_clip_for_semantics):
            try:
                # Compute per-frame semantics
                aes = [self._core_clip_binary_score(frame_index=int(x), text_embeddings=self._scene_aesthetic_text_embeddings, pos_indices=(0, 1)) for x in fi.tolist()]
                lux = [self._core_clip_binary_score(frame_index=int(x), text_embeddings=self._scene_luxury_text_embeddings, pos_indices=(0, 1)) for x in fi.tolist()]
                atm = [self._core_clip_atmosphere(frame_index=int(x)) for x in fi.tolist()]
                aes = np.asarray(aes, dtype=np.float32)
                lux = np.asarray(lux, dtype=np.float32)
                cozy = np.asarray([a.get("cozy", 0.0) for a in atm], dtype=np.float32)
                scary = np.asarray([a.get("scary", 0.0) for a in atm], dtype=np.float32)
                epic = np.asarray([a.get("epic", 0.0) for a in atm], dtype=np.float32)
                neutral = np.asarray([a.get("neutral", 0.0) for a in atm], dtype=np.float32)
                for sid, s in scenes.items():
                    idxs = [int(x) for x in (s.get("indices") or [])]
                    # Map union frame index -> position in fi
                    pos_map = {int(v): i for i, v in enumerate(fi.tolist())}
                    pos_list = [pos_map.get(int(v), None) for v in idxs]
                    pos_list = [p for p in pos_list if p is not None]
                    if not pos_list:
                        continue
                    a = aes[pos_list]
                    l = lux[pos_list]
                    s["mean_aesthetic_score"] = float(np.mean(a))
                    s["aesthetic_std"] = float(np.std(a))
                    s["aesthetic_frac_high"] = float(np.mean(a > 0.8))
                    s["mean_luxury_score"] = float(np.mean(l))
                    # atmosphere mean + entropy
                    atm_mean = np.asarray(
                        [float(np.mean(cozy[pos_list])), float(np.mean(scary[pos_list])), float(np.mean(epic[pos_list])), float(np.mean(neutral[pos_list]))],
                        dtype=np.float32,
                    )
                    denom = float(np.sum(atm_mean)) or 1.0
                    probs = atm_mean / denom
                    s["mean_cozy"] = float(probs[0])
                    s["mean_scary"] = float(probs[1])
                    s["mean_epic"] = float(probs[2])
                    s["mean_neutral"] = float(probs[3])
                    s["atmosphere_entropy"] = float(-np.sum(probs * np.log(probs + 1e-8)))
            except Exception as e:
                raise RuntimeError(f"scene_classification | semantics computation failed (no-fallback): {e}") from e

        return scenes, frame_scene_id

    def _build_ui_payload(
        self,
        *,
        frame_indices: np.ndarray,
        times_s: np.ndarray,
        per_frame: Dict[str, np.ndarray],
        agg: Dict[str, Any],
        frame_scene_id: np.ndarray,
    ) -> Dict[str, Any]:
        """
        JSON for backend/UI only (stored under meta.ui_payload).
        No images, only indices (backend can render thumbs from frames_dir).
        """
        scenes_list = []
        for sid in sorted(agg.keys()):
            s = agg.get(sid) or {}
            idxs = [int(x) for x in (s.get("indices") or [])]
            if not idxs:
                thumbs = []
            else:
                thumbs = [idxs[0], idxs[len(idxs) // 2], idxs[-1]] if len(idxs) >= 3 else idxs
            scenes_list.append(
                {
                    "scene_id": sid,
                    "scene_label": s.get("scene_label"),
                    "start_time_s": s.get("start_time_s"),
                    "end_time_s": s.get("end_time_s"),
                    "length_seconds": s.get("length_seconds"),
                    "length_frames": s.get("length_frames"),
                    "mean_score": s.get("mean_score"),
                    "class_entropy_mean": s.get("class_entropy_mean"),
                    "top1_prob_mean": s.get("top1_prob_mean"),
                    "top1_vs_top2_gap_mean": s.get("top1_vs_top2_gap_mean"),
                    "label_stability": s.get("label_stability"),
                    "dominant_places_topk_ids": s.get("dominant_places_topk_ids"),
                    "dominant_places_topk_probs": s.get("dominant_places_topk_probs"),
                    "semantics": {
                        "mean_aesthetic_score": s.get("mean_aesthetic_score"),
                        "mean_luxury_score": s.get("mean_luxury_score"),
                        "mean_cozy": s.get("mean_cozy"),
                        "mean_scary": s.get("mean_scary"),
                        "mean_epic": s.get("mean_epic"),
                        "mean_neutral": s.get("mean_neutral"),
                    },
                    "thumb_frame_indices": thumbs,
                }
            )
        return {
            "schema_version": "scene_classification_ui_v1",
            "label_fusion": str(self.label_fusion or "places"),
            "min_scene_seconds": float(self.min_scene_seconds) if self.min_scene_seconds is not None else 2.0,
            "frame_indices": frame_indices.tolist(),
            "times_s": times_s.tolist(),
            "frame_scene_id": frame_scene_id.astype(int).tolist(),
            "frame_entropy": per_frame["frame_entropy"].astype(float).tolist(),
            "frame_top1_prob": per_frame["frame_top1_prob"].astype(float).tolist(),
            "frame_top1_top2_gap": per_frame["frame_top1_top2_gap"].astype(float).tolist(),
            "scenes": scenes_list,
        }

    def run(self, frames_dir: str, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Override BaseModule.run to:
        - emit progress stages
        - embed ui_payload into meta (NPZ source-of-truth remains the artifact)
        """
        if metadata is None:
            metadata = self.load_metadata(frames_dir)
        # Enforce run identity keys (baseline reproducibility).
        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")
        platform_id = str(metadata.get("platform_id") or "")
        video_id = str(metadata.get("video_id") or "")
        run_id = str(metadata.get("run_id") or "")

        frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
        if not frame_indices:
            raise RuntimeError("scene_classification | frame_indices missing/empty (no-fallback)")
        _emit_progress(
            rs_path=str(self.rs_path or ""),
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            done=0,
            total=int(len(frame_indices)),
            stage="start",
        )
        t0 = time.perf_counter()
        resource_profile_before = _resource_profile_snapshot()
        saved_path = ""
        fm = None
        try:
            fm = self.create_frame_manager(frames_dir, metadata)
            _emit_progress(
                rs_path=str(self.rs_path or ""),
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=0,
                total=int(len(frame_indices)),
                stage="load_deps",
            )
            results = self.process(frame_manager=fm, frame_indices=frame_indices, config=config or {})
            t_proc = time.perf_counter()

            # Pull ui_payload from results
            ui_payload = None
            if isinstance(results, dict) and "__ui_payload__" in results:
                ui_payload = results.pop("__ui_payload__")

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
                "ui_payload": ui_payload,
                "models_used": self.get_models_used(config=config or {}, metadata=metadata or {}),
            }
            if isinstance(resource_profile_before, dict) and resource_profile_before:
                save_metadata["resource_profile_before"] = dict(resource_profile_before)
            # Audit v3: stage timings in NPZ meta (required by schema).
            try:
                summ = results.get("summary") if isinstance(results, dict) else None
                st = summ.get("stage_timings_ms") if isinstance(summ, dict) and isinstance(summ.get("stage_timings_ms"), dict) else {}
                st["total_ms"] = float((t_proc - t0) * 1000.0)
                if isinstance(summ, dict):
                    summ["stage_timings_ms"] = st
                    results["summary"] = summ
                save_metadata["stage_timings_ms"] = st
            except Exception:
                save_metadata["stage_timings_ms"] = {"total_ms": float((t_proc - t0) * 1000.0)}

            # Sampling policy / config highlights for reproducibility (best-effort).
            save_metadata["module_sampling_policy_version"] = "segmenter_axis_v1"
            save_metadata["prefer_cut_detection_boundaries"] = bool(getattr(self, "prefer_cut_detection_boundaries", True))
            save_metadata["label_fusion"] = str(getattr(self, "label_fusion", "places"))
            save_metadata["min_scene_length_frames"] = int(getattr(self, "min_scene_length_frames", 1))
            save_metadata["min_scene_seconds"] = float(getattr(self, "min_scene_seconds", 2.0) or 2.0)
            save_metadata["runtime"] = str(getattr(self, "runtime", "inprocess"))
            save_metadata["triton_model_spec"] = str(getattr(self, "triton_model_spec", "") or "")
            save_metadata["triton_http_url"] = str(getattr(self, "_triton_http_url", "") or "")
            save_metadata["model_arch"] = str(getattr(self, "model_arch", "") or "")
            save_metadata["use_timm"] = bool(getattr(self, "use_timm", False))
            save_metadata["input_size"] = int(getattr(self, "input_size", 224))
            save_metadata["batch_size"] = int(getattr(self, "batch_size", 1))
            save_metadata["temporal_smoothing"] = bool(getattr(self, "temporal_smoothing", False))
            save_metadata["smoothing_window"] = int(getattr(self, "smoothing_window", 1))
            save_metadata["use_tta"] = bool(getattr(self, "use_tta", False))
            save_metadata["use_multi_crop"] = bool(getattr(self, "use_multi_crop", False))
            save_metadata["enable_advanced_features"] = bool(getattr(self, "enable_advanced_features", True))

            saved_path = self.save_results(results=results, metadata=save_metadata)
            _emit_progress(
                rs_path=str(self.rs_path or ""),
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=int(len(frame_indices)),
                total=int(len(frame_indices)),
                stage="saved",
            )
            _emit_progress(
                rs_path=str(self.rs_path or ""),
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                done=int(len(frame_indices)),
                total=int(len(frame_indices)),
                stage="done",
            )
            return saved_path
        finally:
            if fm is not None:
                try:
                    fm.close()
                except Exception:
                    pass
    
    def aggregate_scenes(self, res, *, fps: float, union_timestamps_sec: List[float]) -> Dict[str, Any]:
        """
        Aggregate consecutive frames with the same scene label.

        Args:
            res: dict[frame_idx] = {
                "predictions": {
                    "label": str,
                    "score": float,
                    "entropy": float,
                    "top1_prob": float,
                    "top2_prob": float,
                    "top1_top2_gap": float,
                    "class_idx": int,
                    "topk_class_indices": Optional[List[int]],
                    "topk_class_probs": Optional[List[float]],
                },
                "advanced_features": {
                    "indoor_outdoor": {"indoor": float, "outdoor": float},
                    "nature_urban": {"nature": float, "urban": float},
                    "aesthetic_score": float,
                    "luxury_score": float,
                    "atmosphere_sentiment": {"cozy": float, "scary": float, "epic": float, "neutral": float},
                }
            }
            fps: frames per second for current video (used for time‑based stats)

        Returns:
            dict: aggregated segments with means and indices
        """
        import numpy as np

        if not res:
            return {}

        aggregated: Dict[str, Any] = {}
        current_label = None
        current_indices = []
        current_values = None
        current_topk_indices: List[Sequence[int]] = []
        current_topk_probs: List[Sequence[float]] = []

        def reset_values():
            return {
                "score": [], "entropy": [],
                "top1_prob": [], "top2_prob": [], "top1_top2_gap": [],
                "indoor": [], "outdoor": [],
                "nature": [], "urban": [],
                "aesthetic_score": [], "luxury_score": [],
                "cozy": [], "scary": [], "epic": [], "neutral": [],
                "labels": [],
                "fusion_mode": [],
            }

        def finalize_segment(label, indices, values, topk_idx_seq, topk_prob_seq):
            """Return aggregated segment dict."""
            if not indices:
                return None
            length_frames = len(indices)
            fps_safe = float(fps) if fps and fps > 0 else 30.0

            # Determine minimal duration in seconds
            if self.min_scene_seconds is not None:
                min_len_s = self.min_scene_seconds
            else:
                # Backwards‑compatible: interpret frame threshold at runtime FPS
                min_len_s = float(self.min_scene_length_frames) / fps_safe

            start_frame = int(indices[0])
            end_frame = int(indices[-1])
            # Time-axis source-of-truth: union timestamps.
            try:
                start_ts = float(union_timestamps_sec[start_frame])
                end_ts = float(union_timestamps_sec[end_frame])
                length_seconds = float(max(0.0, end_ts - start_ts))
            except Exception:
                length_seconds = float(length_frames) / fps_safe
                start_ts = float(start_frame) / fps_safe
                end_ts = float(end_frame) / fps_safe

            if length_seconds < min_len_s:
                return None

            # Aesthetic / luxury aggregates
            aesthetic_arr = np.asarray(values["aesthetic_score"], dtype=np.float32)
            luxury_arr = np.asarray(values["luxury_score"], dtype=np.float32)
            aesthetic_mean = float(np.mean(aesthetic_arr)) if aesthetic_arr.size else 0.0
            aesthetic_std = float(np.std(aesthetic_arr)) if aesthetic_arr.size else 0.0
            aesthetic_frac_high = float(np.mean(aesthetic_arr > 0.8)) if aesthetic_arr.size else 0.0
            luxury_mean = float(np.mean(luxury_arr)) if luxury_arr.size else 0.0

            # Atmosphere entropy
            atm_mat = np.stack(
                [
                    np.asarray(values["cozy"], dtype=np.float32),
                    np.asarray(values["scary"], dtype=np.float32),
                    np.asarray(values["epic"], dtype=np.float32),
                    np.asarray(values["neutral"], dtype=np.float32),
                ],
                axis=0,
            )
            atm_mean = atm_mat.mean(axis=1)
            atm_sum = float(atm_mean.sum()) or 1.0
            atm_probs = atm_mean / atm_sum
            atm_entropy = float(-np.sum(atm_probs * np.log(atm_probs + 1e-8)))

            # Label stability
            labels = values["labels"]
            if labels:
                from collections import Counter
                cnt = Counter(labels)
                scene_label = cnt.most_common(1)[0][0]
                label_stability = float(cnt[scene_label] / len(labels))
            else:
                scene_label = label
                label_stability = 0.0

            # Fusion mode (if enabled): majority vote across frames in the segment
            fm_list = [str(x) for x in (values.get("fusion_mode") or []) if x]
            if fm_list:
                from collections import Counter
                fm_cnt = Counter(fm_list)
                fusion_mode = fm_cnt.most_common(1)[0][0]
            else:
                fusion_mode = None

            # Scene change score: within‑scene variance of confidence
            score_arr = np.asarray(values["score"], dtype=np.float32)
            scene_change_score = float(np.std(score_arr)) if score_arr.size else 0.0

            # Places confidence aggregates
            top1_arr = np.asarray(values["top1_prob"], dtype=np.float32)
            entropy_arr = np.asarray(values["entropy"], dtype=np.float32)
            gap_arr = np.asarray(values["top1_top2_gap"], dtype=np.float32)
            places_top1_prob_mean = float(np.mean(top1_arr)) if top1_arr.size else 0.0
            places_entropy_mean = float(np.mean(entropy_arr)) if entropy_arr.size else 0.0
            places_top1_vs_top2_gap_mean = float(np.mean(gap_arr)) if gap_arr.size else 0.0
            fraction_high_confidence_frames = (
                float(np.mean(top1_arr > 0.7)) if top1_arr.size else 0.0
            )

            # Dominant Places top‑K ids aggregated over scene
            dominant_topk_ids: List[int] = []
            dominant_topk_probs: List[float] = []
            if topk_idx_seq:
                from collections import Counter

                # Flatten indices and accumulate weights using probs
                weight_by_class: Dict[int, float] = {}
                for idx_list, prob_list in zip(topk_idx_seq, topk_prob_seq):
                    for cid, p in zip(idx_list, prob_list):
                        weight_by_class[int(cid)] = weight_by_class.get(int(cid), 0.0) + float(p)
                if weight_by_class:
                    # Take top‑5 by accumulated weight
                    sorted_items = sorted(
                        weight_by_class.items(), key=lambda x: x[1], reverse=True
                    )
                    for cid, w in sorted_items[:5]:
                        dominant_topk_ids.append(int(cid))
                        dominant_topk_probs.append(float(w))

            return {
                "scene_label": scene_label,
                "indices": indices,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": float(start_ts),
                "end_time_s": float(end_ts),
                "length_frames": length_frames,
                "length_seconds": length_seconds,
                "fusion_mode": fusion_mode,
                "mean_score": float(np.mean(values["score"])),
                "class_entropy_mean": places_entropy_mean,
                "top1_prob_mean": places_top1_prob_mean,
                "top1_vs_top2_gap_mean": places_top1_vs_top2_gap_mean,
                "fraction_high_confidence_frames": fraction_high_confidence_frames,
                "mean_indoor": float(np.mean(values["indoor"])),
                "mean_outdoor": float(np.mean(values["outdoor"])),
                "mean_nature": float(np.mean(values["nature"])),
                "mean_urban": float(np.mean(values["urban"])),
                "mean_aesthetic_score": aesthetic_mean,
                "aesthetic_std": aesthetic_std,
                "aesthetic_frac_high": aesthetic_frac_high,
                "mean_luxury_score": luxury_mean,
                "mean_cozy": float(np.mean(values["cozy"])),
                "mean_scary": float(np.mean(values["scary"])),
                "mean_epic": float(np.mean(values["epic"])),
                "mean_neutral": float(np.mean(values["neutral"])),
                "atmosphere_entropy": atm_entropy,
                "scene_change_score": scene_change_score,
                "label_stability": label_stability,
                "dominant_places_topk_ids": dominant_topk_ids,
                "dominant_places_topk_probs": dominant_topk_probs,
            }

        for idx in sorted(res.keys()):
            d = res[idx]
            label = d["predictions"]["label"]

            if current_label != label:
                if current_label is not None:
                    segment = finalize_segment(
                        current_label, current_indices, current_values,
                        current_topk_indices, current_topk_probs,
                    )
                    if segment:
                        scene_id = f"s{len(aggregated):04d}"
                        aggregated[scene_id] = segment
                # Start new segment
                current_label = label
                current_indices = [idx]
                current_values = reset_values()
                current_topk_indices = []
                current_topk_probs = []
            else:
                current_indices.append(idx)

            # Fill values
            # Advanced features are optional:
            # - enable_advanced_features may be False -> advanced_features=None
            # - ontology helpers may return None (unknown mapping)
            adv = d.get("advanced_features") if isinstance(d, dict) else None
            current_values["score"].append(d["predictions"]["score"])
            current_values["entropy"].append(d["predictions"].get("entropy", 0.0))
            current_values["top1_prob"].append(d["predictions"].get("top1_prob", d["predictions"]["score"]))
            current_values["top2_prob"].append(d["predictions"].get("top2_prob", 0.0))
            current_values["top1_top2_gap"].append(d["predictions"].get("top1_top2_gap", 0.0))
            current_values["labels"].append(label)
            current_values["fusion_mode"].append(d["predictions"].get("fusion_mode"))

            tk_idx = d["predictions"].get("topk_class_indices")
            tk_prob = d["predictions"].get("topk_class_probs")
            if tk_idx is not None and tk_prob is not None:
                current_topk_indices.append(tk_idx)
                current_topk_probs.append(tk_prob)

            indoor_outdoor = adv.get("indoor_outdoor") if isinstance(adv, dict) else None
            nature_urban = adv.get("nature_urban") if isinstance(adv, dict) else None

            if isinstance(indoor_outdoor, dict):
                current_values["indoor"].append(float(indoor_outdoor.get("indoor", 0.0)))
                current_values["outdoor"].append(float(indoor_outdoor.get("outdoor", 0.0)))
            else:
                current_values["indoor"].append(0.0)
                current_values["outdoor"].append(0.0)

            if isinstance(nature_urban, dict):
                current_values["nature"].append(float(nature_urban.get("nature", 0.0)))
                current_values["urban"].append(float(nature_urban.get("urban", 0.0)))
            else:
                current_values["nature"].append(0.0)
                current_values["urban"].append(0.0)

            if isinstance(adv, dict):
                current_values["aesthetic_score"].append(float(adv.get("aesthetic_score", 0.0) or 0.0))
                current_values["luxury_score"].append(float(adv.get("luxury_score", 0.0) or 0.0))
                atm = adv.get("atmosphere_sentiment")
            else:
                current_values["aesthetic_score"].append(0.0)
                current_values["luxury_score"].append(0.0)
                atm = None

            if isinstance(atm, dict):
                current_values["cozy"].append(float(atm.get("cozy", 0.0)))
                current_values["scary"].append(float(atm.get("scary", 0.0)))
                current_values["epic"].append(float(atm.get("epic", 0.0)))
                current_values["neutral"].append(float(atm.get("neutral", 0.0)))
            else:
                current_values["cozy"].append(0.0)
                current_values["scary"].append(0.0)
                current_values["epic"].append(0.0)
                current_values["neutral"].append(0.0)

        # Final segment
        if current_label is not None:
            segment = finalize_segment(
                current_label, current_indices, current_values,
                current_topk_indices, current_topk_probs,
            )
            if segment:
                scene_id = f"s{len(aggregated):04d}"
                aggregated[scene_id] = segment

        return aggregated

    def classify_with_advanced_features(
        self, frame_manager, frame_indices
    ) -> List[Dict[str, Any]]:
        """
        Classify scenes and compute advanced features (if enabled).

        Returns list aligned with frame_indices:
            {
                "predictions": { "label": str, "score": float } or None,
                "advanced_features": dict or None
            }
        """
        
        # Base predictions (already best-only, but enriched with confidence stats)
        # Ensure module is ready even if user calls this directly.
        try:
            self.initialize()
        except Exception:
            # If BaseModule init path isn't used in some legacy contexts, ignore.
            pass

        base_predictions = self.classify(frame_manager, frame_indices)
        if bool(self.temporal_smoothing):
            try:
                base_predictions = self._smooth_topk_over_time(base_predictions)  # type: ignore[assignment]
            except Exception as e:
                logger.warning("Places365SceneClassifier | temporal smoothing failed, using raw predictions: %s", e)

        # If advanced features disabled — simple scene aggregation with numeric stats
        if not self.enable_advanced_features:
            results = {
                frame_idx: {
                    "predictions": pred,
                    "advanced_features": None
                }
                for frame_idx, pred in zip(frame_indices, base_predictions)
                if pred is not None
            }
            fps = getattr(frame_manager, "fps", 30.0)
            union_ts = [float(x) for x in (frame_manager.meta.get("union_timestamps_sec") or [])]
            return self.aggregate_scenes(results, fps=fps, union_timestamps_sec=union_ts)

        # Allocate output dict indexed by frame ID
        results: Dict[int, Dict[str, Any]] = {}

        for frame_idx, pred in zip(frame_indices, base_predictions):
            if pred is None:
                continue

            # Best scene prediction and top‑K info
            top_scene = pred["label"]
            topk_indices = pred.get("topk_class_indices") or []
            topk_probs = pred.get("topk_class_probs") or []
            topk_labels = [
                self.categories[i] if 0 <= i < len(self.categories) else f"class_{i}"
                for i in topk_indices
            ]

            # Compute advanced features
            advanced: Dict[str, Any] = {}

            advanced["indoor_outdoor"] = self._ontology_indoor_outdoor(topk_labels, topk_probs)
            advanced["nature_urban"] = self._ontology_nature_urban(topk_labels, topk_probs)
            # semantics strictly from core_clip (no heuristics, no local CLIP)
            advanced["aesthetic_score"] = self._core_clip_binary_score(
                frame_index=frame_idx, text_embeddings=self._scene_aesthetic_text_embeddings, pos_indices=(0, 1)
            )
            advanced["luxury_score"] = self._core_clip_binary_score(
                frame_index=frame_idx, text_embeddings=self._scene_luxury_text_embeddings, pos_indices=(0, 1)
            )
            advanced["atmosphere_sentiment"] = self._core_clip_atmosphere(frame_index=frame_idx)

            results[frame_idx] = {
                "predictions": pred,
                "advanced_features": advanced
            }

        fps = getattr(frame_manager, "fps", 30.0)
        union_ts = [float(x) for x in (frame_manager.meta.get("union_timestamps_sec") or [])]
        agg_result = self.aggregate_scenes(results, fps=fps, union_timestamps_sec=union_ts)

        return agg_result

