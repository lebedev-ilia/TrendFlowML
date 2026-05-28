"""
Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (core_face_landmarks)
    2. ✅ Использование результатов core провайдеров вместо прямых вызовов моделей
    3. ✅ Интеграция с BaseModule через класс DetalizeFaceModule
    4. ✅ Единый формат вывода для сохранения в npz
"""
from __future__ import annotations

import os
import sys
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque

import cv2 # type: ignore
import numpy as np # type: ignore

# Добавляем путь для импорта BaseModule
_MODULE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

from _modules import MODULE_REGISTRY
from _modules.base_module import FaceModule
from _utils import (
    validate_face_landmarks,
    compute_bbox,
    extract_roi,
)
from _utils.landmarks_utils import LANDMARKS
from _utils.compact_features import extract_per_face_aggregates, extract_compact_features
from utils.logger import get_logger

NAME = "DetalizeFaceExtractorRefactored"
VERSION = "2.0.2"
logger = get_logger(NAME)

# Module-level baseline constants (BaseModule contract)
MODULE_NAME = "detalize_face"
SCHEMA_VERSION = "detalize_face_npz_v3"
ARTIFACT_FILENAME = "detalize_face.npz"


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl` (backend tails this file).
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


def _emit_stage(*, rs_path: str, platform_id: str, video_id: str, run_id: str, stage: str) -> None:
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": _utc_iso_now(),
            "scope": "progress",
            "processor": "visual",
            "component": MODULE_NAME,
            "status": "running",
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


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


def _require_union_times_s(frame_manager: FrameManager, frame_indices: List[int]) -> np.ndarray:
    """
    Segmenter contract: union_timestamps_sec is source-of-truth for time axis.
    No-fallback: if missing/invalid -> error.
    """
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError(f"{MODULE_NAME} | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError(f"{MODULE_NAME} | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32).reshape(-1)
    fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
    if fi.size == 0:
        raise RuntimeError(f"{MODULE_NAME} | frame_indices is empty (no-fallback)")
    if int(np.max(fi)) >= int(uts.shape[0]) or int(np.min(fi)) < 0:
        raise RuntimeError(f"{MODULE_NAME} | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[fi.astype(np.int64)]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError(f"{MODULE_NAME} | union_timestamps_sec is not monotonic for frame_indices (no-fallback)")
    return times_s.astype(np.float32)

def _load_core_face_landmarks(rs_path: Optional[str]):
    """
    Пытается загрузить предрасчитанные Mediapipe‑landmarks из core_face_landmarks.
    Формат: result_store/core_face_landmarks/landmarks.json
    
    Используется для обратной совместимости. В DetalizeFaceModule используется
    load_core_provider() из BaseModule.
    """
    if not rs_path:
        return None

    core_path = os.path.join(rs_path, "core_face_landmarks", "landmarks.json")
    if not os.path.isfile(core_path):
        return None

    try:
        with open(core_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames") or []
        # Преобразуем в dict[frame_index] -> frame_payload
        return {int(f["frame_index"]): f for f in frames if "frame_index" in f}
    except Exception as e:
        logger.warning(f"DetalizeFaceExtractorRefactored | _load_core_face_landmarks | error: {e}")
        return None


def _core_npz_to_frames(
    data: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], List[int], Dict[int, float]]:
    """
    Конвертирует npz-формат core_face_landmarks в dict[frame_index] -> payload.
    Возвращает:
      - frames_by_index
      - frames_with_face (sorted)
      - times_by_index (frame_index -> time_s)
    """
    frame_indices = data.get("frame_indices")
    face_landmarks = data.get("face_landmarks")
    face_present = data.get("face_present")
    times_s = data.get("times_s")

    if frame_indices is None or face_landmarks is None:
        raise RuntimeError("core_face_landmarks npz missing frame_indices/face_landmarks")

    frame_indices = np.asarray(frame_indices).astype(int)
    face_landmarks = np.asarray(face_landmarks)
    face_present = np.asarray(face_present) if face_present is not None else None
    times_s = np.asarray(times_s) if times_s is not None else None

    frames_by_index: Dict[int, Dict[str, Any]] = {}
    frames_with_face: List[int] = []
    times_by_index: Dict[int, float] = {}

    for i, frame_idx in enumerate(frame_indices):
        if i >= face_landmarks.shape[0]:
            continue

        present_mask = None
        if face_present is not None:
            present_mask = face_present[i]
            if not np.any(present_mask):
                continue

        faces_list = []
        for f in range(face_landmarks.shape[1]):
            if present_mask is not None and not bool(present_mask[f]):
                continue
            face_points = face_landmarks[i, f]
            if np.isnan(face_points).all():
                continue
            # Keep ndarray (468,3) float32 to avoid huge Python dicts.
            faces_list.append(np.asarray(face_points, dtype=np.float32))

        if not faces_list:
            continue

        frame_idx_int = int(frame_idx)
        frames_by_index[frame_idx_int] = {
            "frame_index": frame_idx_int,
            "face_landmarks": faces_list,
        }
        frames_with_face.append(frame_idx_int)
        if times_s is not None and i < len(times_s):
            times_by_index[frame_idx_int] = float(times_s[i])

    frames_with_face.sort()
    return frames_by_index, frames_with_face, times_by_index

class DetalizeFaceExtractorRefactored():
    """
    Рефакторинг DetalizeFaceExtractor с использованием модульной архитектуры.
    
    Использует модули для извлечения различных типов фич лица.
    """

    def __init__(
        self,
        *,
        modules: Optional[List[str]] = None,
        module_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_faces: int = 10,
        refine_landmarks: bool = True,
        visualize: bool = False,
        visualize_dir: Optional[str] = None,
        show_landmarks: bool = False,
        # Quality filtering parameters
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        min_face_size: int = 30,
        max_face_size_ratio: float = 0.8,
        min_aspect_ratio: float = 0.6,
        max_aspect_ratio: float = 1.4,
        validate_landmarks: bool = True,
        # core‑данные
        rs_path: Optional[str] = None,
        strict_core: bool = False,
    ) -> None:
        """
        :param modules: список имен модулей для загрузки (если None - загружаются все)
        :param module_configs: конфигурации для конкретных модулей
        :param max_faces: maximum number of faces to detect per frame
        :param refine_landmarks: use refined landmarks (468 points)
        :param kwargs: дополнительные параметры для BaseExtractor
        """

        self.rs_path = rs_path

        # Baseline: DetalizeFaceModule owns sampling and loads core_face_landmarks via BaseModule.
        # This extractor is a pure implementation detail and must not load artifacts itself.
        self.frames_with_face: List[int] = []
        self.core_landmarks: Optional[Dict[int, Dict[str, Any]]] = None
        if strict_core:
            raise RuntimeError("DetalizeFaceExtractorRefactored | init | strict_core=True is not supported in baseline (use DetalizeFaceModule)")

        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks

        logger.info(f"DetalizeFaceExtractorRefactored | init | Используем core_face_landmarks (max_faces = {max_faces})")
        
        # Mediapipe face_mesh удалён - используем только core_face_landmarks

        # Quality filtering parameters
        self.min_face_size = max(10, min_face_size)
        self.max_face_size_ratio = np.clip(max_face_size_ratio, 0.1, 1.0)
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.validate_landmarks = validate_landmarks

        # Visualization settings
        self.visualize = visualize
        self.show_landmarks = show_landmarks and visualize
        if visualize:
            self.visualize_dir = Path(visualize_dir) if visualize_dir else Path("./face_visualizations")
            self.visualize_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"DetalizeFaceExtractorRefactored | init | Visualization enabled. Saving frames to: {self.visualize_dir}")

        logger.info(f"DetalizeFaceExtractorRefactored | init | load modules...")

        # Загружаем модули
        modules_to_load = modules or list(MODULE_REGISTRY.keys())
        self.modules: List[FaceModule] = []

        for module_name in modules_to_load:
            if module_name not in MODULE_REGISTRY:
                logger.info(f"DetalizeFaceExtractorRefactored | init | Модуль '{module_name}' не найден в registry, пропускаем")
                continue

            module_class = MODULE_REGISTRY[module_name]
            module_config = (module_configs or {}).get(module_name, {})

            try:
                module = module_class(config=module_config)
                self.modules.append(module)
                logger.info(f"DetalizeFaceExtractorRefactored | init | Загружен модуль: {module_name}")
            except Exception as e:
                logger.error(f"DetalizeFaceExtractorRefactored | init | Ошибка при загрузке модуля '{module_name}': {e}")

        if not self.modules:
            raise ValueError("DetalizeFaceExtractorRefactored | init | Не удалось загрузить ни одного модуля")

        # Инициализируем модули
        for module in self.modules:
            try:
                module.initialize()
            except Exception as e:
                logger.error(f"DetalizeFaceExtractorRefactored | init | Ошибка при инициализации модуля '{module.module_name}': {e}")

        # Tracking для мульти-лица (tracking_id для каждого лица)
        self._face_tracking: Dict[int, Dict[str, Any]] = defaultdict(dict)  # frame_idx -> {face_idx -> tracking_id}
        self._tracking_counter = 0
        self._track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))  # tracking_id -> history

    def frames_with_face_load(self, filename, rs_path: Optional[str] = None):
        """
        Возвращает список кадров с лицами на основе `core_face_landmarks`.
        
        Args:
            filename: Имя файла или "auto" для автоматического поиска
            rs_path: Путь к хранилищу результатов (если None, использует self.rs_path)
            
        Returns:
            Список индексов кадров с лицами
        """
        rs_path = rs_path or self.rs_path
        if not rs_path:
            logger.warning("DetalizeFaceExtractorRefactored | frames_with_face_load | rs_path не указан, возвращаем пустой список")
            return []
        
        # face_detection module was removed; we rely on core_face_landmarks only.
        try:
            if not isinstance(self.core_landmarks, dict) or not self.core_landmarks:
                return []
            frames_with_face = sorted(int(k) for k in self.core_landmarks.keys())
            logger.info(
                f"DetalizeFaceExtractorRefactored | frames_with_face_load | using core_face_landmarks frames: {len(frames_with_face)}"
            )
            return frames_with_face
        except Exception as e:
            logger.error(f"DetalizeFaceExtractorRefactored | frames_with_face_load | Error: {e}")
            return []

    def extract(self, frame_manager) -> List[List[Dict[str, Any]]]:
        """
        Processes a sequence of OpenCV BGR frames and returns a list where each
        element corresponds to the facial feature set per frame.
        """
        outputs: Dict[int, List[Dict[str, Any]]] = {}
        
        t_extract_start = time.perf_counter()
        total_frames = len(self.frames_with_face)
        logger.info(f"DetalizeFaceExtractorRefactored | extract | processing {total_frames} frames with {len(self.modules)} modules")

        for frame_idx in self.frames_with_face:
            t_frame_start = time.perf_counter()
            frame = frame_manager.get(frame_idx)
            # FrameManager хранит color_space в metadata; конвертируем в BGR при необходимости.
            color_space = getattr(frame_manager, "color_space", None)
            if isinstance(color_space, str) and color_space.upper() == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Используем только core_face_landmarks
            if not isinstance(self.core_landmarks, dict) or int(frame_idx) not in self.core_landmarks:
                logger.warning(f"DetalizeFaceExtractorRefactored | extract | Frame {frame_idx} отсутствует в core_face_landmarks, пропускаем")
                continue

            core_frame = self.core_landmarks[int(frame_idx)]
            frame_results = self._process_with_core(frame, frame_idx, core_frame)
            outputs[frame_idx] = frame_results
            
            t_frame_end = time.perf_counter()
            frame_time_ms = (t_frame_end - t_frame_start) * 1000.0
            
            # Логируем время обработки кадра (только для первых нескольких кадров)
            if frame_idx < 3 or (frame_idx % 20 == 0):
                logger.debug(
                    f"DetalizeFaceExtractorRefactored | extract | frame {frame_idx}: "
                    f"{frame_time_ms:.2f}ms, faces={len(frame_results)}"
                )

            # Visualize frame if enabled and faces detected
            if self.visualize and frame_results:
                self._visualize_frame(frame, frame_idx, frame_results)

        t_extract_end = time.perf_counter()
        total_time_ms = (t_extract_end - t_extract_start) * 1000.0
        avg_time_per_frame = total_time_ms / max(total_frames, 1)
        logger.info(
            f"DetalizeFaceExtractorRefactored | extract | completed: "
            f"{total_frames} frames in {total_time_ms:.1f}ms "
            f"(avg {avg_time_per_frame:.2f}ms/frame, {len(self.modules)} modules)"
        )
        
        return outputs

    __call__ = extract

    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Вычисляет IoU между двумя bbox."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / max(union_area, 1e-6)

    def _assign_tracking_id(self, frame_idx: int, face_idx: int, bbox: np.ndarray, 
                           detection_confidence: float) -> int:
        """
        Назначает tracking_id для лица на основе IoU с предыдущими кадрами.
        """
        # Ищем совпадения в предыдущих кадрах (последние 3 кадра)
        best_match_id = None
        best_iou = 0.3  # Порог IoU для совпадения
        
        for prev_frame_idx in range(max(0, frame_idx - 3), frame_idx):
            if prev_frame_idx in self._face_tracking:
                for prev_face_idx, prev_tracking_id in self._face_tracking[prev_frame_idx].items():
                    # Получаем bbox из истории
                    if prev_tracking_id in self._track_history:
                        hist = list(self._track_history[prev_tracking_id])
                        if len(hist) > 0:
                            prev_bbox = hist[-1].get("bbox")
                            if prev_bbox is not None:
                                iou = self._compute_iou(bbox, np.array(prev_bbox))
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match_id = prev_tracking_id
        
        if best_match_id is not None:
            return best_match_id
        else:
            # Новый track
            self._tracking_counter += 1
            return self._tracking_counter

    def _process_with_core(
        self,
        frame: np.ndarray,
        frame_idx: int,
        core_frame: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Обработка кадра на основе уже готовых core_face_landmarks.
        По сути повторяет _process_frame, но вместо вызова Mediapipe
        использует координаты из core‑слоя.
        """
        height, width = frame.shape[:2]

        faces_with_bbox: List[Tuple[int, Any, np.ndarray, np.ndarray, float]] = []

        # core_frame["face_landmarks"] — list of faces.
        # Preferred baseline representation: ndarray (468,3) with normalized x/y in [0..1].
        # Legacy: list of dict points [{x,y,z}, ...] (should not be present in baseline).
        for face_idx, face_points in enumerate(core_frame.get("face_landmarks", []) or []):
            if face_points is None:
                continue

            if isinstance(face_points, np.ndarray):
                coords = np.asarray(face_points, dtype=np.float32).copy()
                if coords.ndim != 2 or coords.shape[1] < 3:
                    continue
                coords[:, 0] *= float(width)
                coords[:, 1] *= float(height)
            else:
                if not face_points:
                    continue
                # Восстанавливаем coords в пиксельных координатах, совместимых с MediaPipe‑пайплайном
                coords = np.zeros((len(face_points), 3), dtype=np.float32)
                for i, p in enumerate(face_points):
                    coords[i, 0] = float(p.get("x", 0.0)) * float(width)
                    coords[i, 1] = float(p.get("y", 0.0)) * float(height)
                    coords[i, 2] = float(p.get("z", 0.0))

            bbox = compute_bbox(coords, width, height)

            if not validate_face_landmarks(
                bbox,
                coords,
                width,
                height,
                min_face_size=self.min_face_size,
                max_face_size_ratio=self.max_face_size_ratio,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
                validate_landmarks=self.validate_landmarks,
            ):
                continue

            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            faces_with_bbox.append((face_idx, None, coords, bbox, bbox_area))

        # Если ничего не осталось после валидации
        if not faces_with_bbox:
            return []

        # Сортировка по площади для определения primary face
        faces_with_bbox.sort(key=lambda x: x[4], reverse=True)

        frame_features: List[Dict[str, Any]] = []
        self._face_tracking[frame_idx] = {}

        for idx, (face_idx, _unused_landmarks, coords, bbox, bbox_area) in enumerate(faces_with_bbox):
            detection_confidence = 0.9  # эвристика: core‑данные считаем надёжными

            tracking_id = self._assign_tracking_id(frame_idx, face_idx, bbox, detection_confidence)
            self._face_tracking[frame_idx][face_idx] = tracking_id

            if tracking_id not in self._track_history:
                self._track_history[tracking_id] = deque(maxlen=30)

            self._track_history[tracking_id].append(
                {
                    "bbox": bbox.tolist(),
                    "frame_idx": frame_idx,
                    "detection_confidence": detection_confidence,
                }
            )

            roi = extract_roi(frame, bbox)

            bbox_x_min, bbox_y_min = bbox[0], bbox[1]
            coords_roi = coords.copy()
            coords_roi[:, 0] -= bbox_x_min
            coords_roi[:, 1] -= bbox_y_min

            shared_data = {
                "coords": coords,
                "coords_roi": coords_roi,
                "bbox": bbox,
                "roi": roi,
                "frame_shape": frame.shape,
                "face_idx": tracking_id,
                "face_index": face_idx,
                "tracking_id": tracking_id,
                "detection_confidence": detection_confidence,
                "is_primary_face": (idx == 0),
            }

            face_feature: Dict[str, Any] = {
                "frame_index": frame_idx,
                "face_index": face_idx,
                "tracking_id": tracking_id,
                "bbox": bbox.tolist(),
                "detection_confidence": detection_confidence,
                "is_primary_face": (idx == 0),
            }

            for module in self.modules:
                missing = [k for k in module.required_inputs() if k not in shared_data]
                if missing:
                    raise RuntimeError(
                        f"DetalizeFaceExtractorRefactored | missing inputs for module "
                        f"{module.module_name}: {missing}"
                    )

                t_module_start = time.perf_counter()
                module_result = module.process(shared_data)
                t_module_end = time.perf_counter()
                module_time_ms = (t_module_end - t_module_start) * 1000.0
                
                if not isinstance(module_result, dict):
                    raise RuntimeError(
                        f"DetalizeFaceExtractorRefactored | module '{module.module_name}' returned non-dict result"
                    )

                # Логируем время выполнения модуля (только для первых нескольких кадров, чтобы не засорять логи)
                if frame_idx < 3 or (frame_idx % 20 == 0):
                    logger.debug(
                        f"DetalizeFaceExtractorRefactored | frame {frame_idx} | "
                        f"module {module.module_name}: {module_time_ms:.2f}ms"
                    )

                face_feature.update(module_result)

                for key, value in module_result.items():
                    shared_data[key] = value

            if self.visualize:
                face_feature["_landmarks_coords"] = coords.tolist()

            frame_features.append(face_feature)

        return frame_features

    # Метод _process_frame удалён - используем только core_face_landmarks через _process_with_core

    def _visualize_frame(
        self, frame: np.ndarray, frame_idx: int, frame_results: List[Dict[str, Any]]
    ) -> None:
        """Visualize faces with bounding boxes and optionally landmarks."""
        vis_frame = frame.copy()
        
        for face_result in frame_results:
            bbox = face_result["bbox"]
            face_idx = face_result.get("face_index", 0)
            
            # Draw bounding box
            x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw face index label
            label = f"Face {face_idx}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                vis_frame,
                (x_min, y_min - label_size[1] - 5),
                (x_min + label_size[0], y_min),
                color,
                -1,
            )
            cv2.putText(
                vis_frame,
                label,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            
            # Draw landmarks if enabled
            if self.show_landmarks and "_landmarks_coords" in face_result:
                coords = np.array(face_result["_landmarks_coords"], dtype=np.float32)
                for point in coords:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(vis_frame, (x, y), 1, (0, 0, 255), -1)
                
                # Draw key landmarks with labels
                for name, idx in LANDMARKS.items():
                    if idx < len(coords):
                        x, y = int(coords[idx][0]), int(coords[idx][1])
                        cv2.circle(vis_frame, (x, y), 3, (255, 0, 0), -1)
        
        # Save frame
        output_path = self.visualize_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(output_path), vis_frame)
        
        if frame_idx == 0 or (frame_idx + 1) % 10 == 0:
            logger.info(f"Saved visualization: {output_path}")


def _load_core_face_landmarks_from_data(
    data: Dict[str, Any]
) -> Optional[Tuple[Dict[int, Dict[str, Any]], List[int], Dict[int, float]]]:
    """
    Загружает core_face_landmarks из данных, загруженных через BaseModule.
    
    Args:
        data: Данные из load_core_provider("core_face_landmarks")
        
    Returns:
        (frames_by_index, frames_with_face, times_by_index) или None
    """
    if not data:
        return None
    
    try:
        # Baseline (strict): data is core_face_landmarks NPZ payload.
        if "face_landmarks" in data and "frame_indices" in data:
            return _core_npz_to_frames(data)
        
        return None
    except Exception as e:
        logger.warning(f"DetalizeFaceModule | _load_core_face_landmarks_from_data | error: {e}")
        return None


class DetalizeFaceModule(BaseModule):
    """
    Модуль для детального извлечения фичей лица.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует DetalizeFaceExtractorRefactored для обработки кадров.
    
    Зависимости:
    - core_face_landmarks (обязательная) - landmarks лиц (и face presence)
    """

    MODULE_NAME = MODULE_NAME
    ARTIFACT_FILENAME = ARTIFACT_FILENAME
    SCHEMA_VERSION = SCHEMA_VERSION
    VERSION = VERSION
    
    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли модуль батчинг.
        
        DetalizeFaceModule поддерживает batch processing через дефолтный process_batch
        из BaseModule (последовательная обработка каждого видео).
        """
        return True
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        modules: Optional[List[str]] = None,
        module_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_faces: int = 10,
        refine_landmarks: bool = True,
        visualize: bool = False,
        visualize_dir: Optional[str] = None,
        show_landmarks: bool = False,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        min_face_size: int = 30,
        max_face_size_ratio: float = 0.8,
        min_aspect_ratio: float = 0.6,
        max_aspect_ratio: float = 1.4,
        validate_landmarks: bool = True,
        # Audit v3: optional heuristic curves (primary_*). Off by default.
        write_primary_curves: bool = False,
        # Audit v3: model-facing compact primary-face embedding (recommended for transformer encoders).
        # This is cheap once per processed frame (pure numpy) and is safe to keep enabled.
        write_primary_compact_features: bool = True,
        # Audit v3: internal sampling among frames with faces (for very long videos / dense face coverage).
        max_face_frames: Optional[int] = None,
        face_frames_sampling: str = "uniform",
        use_face_detection: bool = True,
        **kwargs: Any
    ):
        """
        Инициализация DetalizeFaceModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            modules: Список имен модулей для загрузки (если None - загружаются все)
            module_configs: Конфигурации для конкретных модулей
            max_faces: Максимальное количество лиц на кадр
            refine_landmarks: Использовать уточненные landmarks (468 точек)
            visualize: Включить визуализацию
            visualize_dir: Директория для визуализаций
            show_landmarks: Показывать landmarks на визуализации
            min_detection_confidence: Минимальная уверенность детекции
            min_tracking_confidence: Минимальная уверенность трекинга
            min_face_size: Минимальный размер лица в пикселях
            max_face_size_ratio: Максимальное отношение размера лица к размеру кадра
            min_aspect_ratio: Минимальное соотношение сторон лица
            max_aspect_ratio: Максимальное соотношение сторон лица
            validate_landmarks: Валидировать landmarks
            use_face_detection: Устаревший флаг. Теперь фильтрация делается по core_face_landmarks, face_detection удалён.
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        # Backward-compat flag: keep it, but it no longer loads face_detection.
        self.use_face_detection = bool(use_face_detection)

        # Optional heuristic outputs (primary_* curves)
        self.write_primary_curves = bool(write_primary_curves)
        self.write_primary_compact_features = bool(write_primary_compact_features)

        # Internal sampling among frames-with-faces
        self.max_face_frames = int(max_face_frames) if max_face_frames is not None else None
        self.face_frames_sampling = str(face_frames_sampling or "uniform")

        # Core alignment helpers (populated in _load_core_landmarks)
        self._core_frame_indices_all: List[int] = []
        
        # Инициализируем extractor
        self.extractor = DetalizeFaceExtractorRefactored(
            modules=modules,
            module_configs=module_configs,
            max_faces=max_faces,
            refine_landmarks=refine_landmarks,
            visualize=visualize,
            visualize_dir=visualize_dir,
            show_landmarks=show_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_face_size=min_face_size,
            max_face_size_ratio=max_face_size_ratio,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            validate_landmarks=validate_landmarks,
            rs_path=rs_path,
            strict_core=False,
        )
        
        # Load core_face_landmarks via BaseModule (strict, no-fallback).
        self._load_core_landmarks()
        
        # Always derive frames_with_face from core_face_landmarks (face_detection удалён).
        if self.extractor.core_landmarks:
            self.extractor.frames_with_face = sorted(self.extractor.core_landmarks.keys())
        else:
            self.extractor.frames_with_face = []
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        Обязательные:
        - core_face_landmarks: landmarks лиц
        
        Опциональные:
        - (none) — face_detection удалён
        """
        return ["core_face_landmarks"]
    
    def _load_core_landmarks(self) -> None:
        """Загружает core_face_landmarks через BaseModule."""
        try:
            landmarks_data = self.load_core_provider("core_face_landmarks")
            if not isinstance(landmarks_data, dict):
                raise RuntimeError("DetalizeFaceModule | core_face_landmarks artifact missing/invalid (no-fallback)")
            try:
                fi_all = landmarks_data.get("frame_indices")
                if fi_all is not None:
                    self._core_frame_indices_all = [int(x) for x in np.asarray(fi_all).reshape(-1).tolist()]
            except Exception:
                self._core_frame_indices_all = []
            parsed = _load_core_face_landmarks_from_data(landmarks_data)
            if not parsed:
                raise RuntimeError("DetalizeFaceModule | failed to parse core_face_landmarks NPZ (no-fallback)")
            core_landmarks, frames_with_face, _times_by_index = parsed
            if not core_landmarks:
                # core_face_landmarks may be status=empty; orchestrator should skip this module in that case.
                self.extractor.core_landmarks = {}
                self.extractor.frames_with_face = []
                return
            self.extractor.core_landmarks = core_landmarks
            self.extractor.frames_with_face = frames_with_face
            self.logger.info(f"DetalizeFaceModule | core_face_landmarks loaded: frames_with_faces={len(frames_with_face)}")
            return
        except Exception as e:
            self.logger.exception(f"DetalizeFaceModule | Ошибка загрузки core_face_landmarks: {e}")
            raise

    def _sample_face_frames(self, face_frames: List[int]) -> List[int]:
        """
        Internal sampling among frames with faces.
        Goal: reduce work while keeping alignment to the Segmenter axis (we will still output full axis arrays).
        """
        if not face_frames:
            return []
        if self.max_face_frames is None:
            return sorted([int(x) for x in face_frames])
        k = int(self.max_face_frames)
        if k <= 0:
            return []
        face_frames = sorted([int(x) for x in face_frames])
        if len(face_frames) <= k:
            return face_frames
        if self.face_frames_sampling not in ("uniform", "even", "linspace"):
            # fail-fast: unknown policy (to avoid silent contract drift)
            raise RuntimeError(f"{self.module_name} | unknown face_frames_sampling='{self.face_frames_sampling}' (expected: uniform)")
        # Uniform over the ordered face-frames list (best-effort time-uniform)
        idxs = np.linspace(0, len(face_frames) - 1, num=k, dtype=int).tolist()
        sampled = [face_frames[i] for i in idxs]
        # Ensure uniqueness + sorted
        return sorted(list({int(x) for x in sampled}))
    
    def _load_frames_with_face(self) -> None:
        """
        Deprecated: face_detection removed. Kept for compatibility; derives frames from core_face_landmarks.
        """
        if self.extractor.core_landmarks:
            self.extractor.frames_with_face = sorted(self.extractor.core_landmarks.keys())
        else:
            self.extractor.frames_with_face = []
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Axis кадры для вывода (Segmenter per-component sampling; union-domain). Модуль считает фичи
                только на кадрах, где `core_face_landmarks` нашёл лица; на остальных пишет NaN/0 + маски.
            config: Конфигурация модуля (не используется, параметры заданы в __init__)
            
        Returns:
            NPZ payload aligned to the Segmenter axis:
            - frame_indices, times_s
            - face_present, processed_mask, face_count
            - optional primary_* time-series (heuristic; gated by flags)
            - faces_agg (per-track aggregates; only for processed frames)
            - summary
        """
        if not frame_indices:
            # No axis to align to (Segmenter contract violation for this module).
            raise RuntimeError(f"{self.module_name} | frame_indices is empty (no-fallback)")

        axis_frame_indices = [int(x) for x in frame_indices]

        # Determine frames with faces on this axis (based on parsed core_face_landmarks).
        core = self.extractor.core_landmarks or {}
        face_frames = [fi for fi in axis_frame_indices if int(fi) in core]
        frames_to_process = self._sample_face_frames(face_frames)

        # Update extractor.frames_with_face only for compute (keep axis separate).
        original_frames_with_face = self.extractor.frames_with_face
        self.extractor.frames_with_face = sorted([int(x) for x in frames_to_process])
        
        try:
            # Strict time-axis from Segmenter (no fps fallback)
            times_s = _require_union_times_s(frame_manager, axis_frame_indices)
            # Derive a best-effort fps estimate for legacy submodules that expect fps.
            dt = np.diff(times_s).astype(np.float32)
            dt = dt[np.isfinite(dt)]
            fps = float(1.0 / float(np.median(dt))) if dt.size else 30.0
            for module in self.extractor.modules:
                if hasattr(module, "fps"):
                    setattr(module, "fps", fps)

            # Compute only on frames_to_process (frames with faces, possibly sampled).
            outputs = self.extractor.extract(frame_manager=frame_manager) if frames_to_process else {}
            
            # Преобразуем результаты в единый формат для npz
            result = self._format_results_for_npz(
                outputs=outputs,
                axis_frame_indices=axis_frame_indices,
                times_s=times_s,
                frames_to_process=frames_to_process,
                write_primary_curves=self.write_primary_curves,
                write_primary_compact_features=self.write_primary_compact_features,
            )
            
            self.logger.info(
                f"DetalizeFaceModule | Обработка завершена: "
                f"axis_frames={len(axis_frame_indices)}, "
                f"face_frames={len(face_frames)}, "
                f"processed_face_frames={len(frames_to_process)}, "
                f"faces_processed_total={sum(len(faces) for faces in outputs.values())}"
            )
            
            return result
            
        finally:
            # Восстанавливаем оригинальный список
            self.extractor.frames_with_face = original_frames_with_face
    
    def _format_results_for_npz(
        self,
        outputs: Dict[int, List[Dict[str, Any]]],
        axis_frame_indices: List[int],
        times_s: np.ndarray,
        frames_to_process: List[int],
        write_primary_curves: bool,
        write_primary_compact_features: bool,
    ) -> Dict[str, Any]:
        """
        Преобразует результаты extractor в формат для сохранения в npz.
        
        Args:
            outputs: Результаты из extractor.extract()
            total_frames: Общее количество кадров
            processed_frames: Количество обработанных кадров
            
        Returns:
            Словарь в формате для npz
        """
        n = int(len(axis_frame_indices))

        # Masks (always present; model-facing alignment helpers)
        core = self.extractor.core_landmarks or {}
        face_present = np.asarray([int(fi) in core for fi in axis_frame_indices], dtype=bool).reshape(-1)
        processed_mask = np.asarray([int(fi) in set(int(x) for x in frames_to_process) for fi in axis_frame_indices], dtype=bool).reshape(-1)
        primary_valid = np.zeros((n,), dtype=bool)

        # Face count from core_face_landmarks payload (cheap, deterministic, available for full axis).
        face_count = np.zeros((n,), dtype=np.float32)
        for i, fi in enumerate(axis_frame_indices):
            payload = core.get(int(fi))
            if isinstance(payload, dict):
                lm = payload.get("face_landmarks")
                if isinstance(lm, list):
                    face_count[i] = float(len(lm))

        # Model-facing compact features for the primary face (recommended for transformer encoders).
        # Filled with zeros when not available; masks indicate validity.
        compact_dim = 40
        primary_compact_features = np.zeros((n, compact_dim), dtype=np.float32)
        primary_tracking_id = np.full((n,), -1, dtype=np.int32)

        # Optional heuristic per-frame curves (primary_*). Only meaningful when processed_mask=True.
        primary_gaze = primary_blink_rate = primary_attention = None
        primary_quality = primary_sharpness = primary_occlusion = primary_speech = None
        if write_primary_curves:
            primary_gaze = np.full((n,), np.nan, dtype=np.float32)
            primary_blink_rate = np.full((n,), np.nan, dtype=np.float32)
            primary_attention = np.full((n,), np.nan, dtype=np.float32)
            primary_quality = np.full((n,), np.nan, dtype=np.float32)
            primary_sharpness = np.full((n,), np.nan, dtype=np.float32)
            primary_occlusion = np.full((n,), np.nan, dtype=np.float32)
            primary_speech = np.full((n,), np.nan, dtype=np.float32)

        def _select_primary(faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            for face in faces:
                if bool(face.get("is_primary_face")):
                    return face
            return faces[0] if faces else None

        for i, frame_idx in enumerate(axis_frame_indices):
            if not bool(processed_mask[i]):
                continue
            faces = outputs.get(int(frame_idx)) or []
            p = _select_primary(faces)
            if not isinstance(p, dict):
                continue

            primary_valid[i] = True
            try:
                tid = p.get("tracking_id")
                if isinstance(tid, int):
                    primary_tracking_id[i] = int(tid)
            except Exception:
                pass

            if write_primary_compact_features:
                try:
                    vec = extract_compact_features(p)
                    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                    if vec.size >= compact_dim:
                        primary_compact_features[i, :] = vec[:compact_dim]
                    elif vec.size > 0:
                        primary_compact_features[i, : vec.size] = vec
                except Exception:
                    pass

            if not write_primary_curves:
                continue
            eyes = p.get("eyes") if isinstance(p.get("eyes"), dict) else {}
            quality = p.get("quality") if isinstance(p.get("quality"), dict) else {}
            lip = p.get("lip_reading") if isinstance(p.get("lip_reading"), dict) else {}
            assert primary_gaze is not None
            assert primary_blink_rate is not None
            assert primary_attention is not None
            assert primary_quality is not None
            assert primary_sharpness is not None
            assert primary_occlusion is not None
            assert primary_speech is not None
            primary_gaze[i] = float(eyes.get("gaze_at_camera_prob")) if isinstance(eyes.get("gaze_at_camera_prob"), (int, float)) else np.nan
            primary_blink_rate[i] = float(eyes.get("blink_rate")) if isinstance(eyes.get("blink_rate"), (int, float)) else np.nan
            primary_attention[i] = float(eyes.get("attention_score")) if isinstance(eyes.get("attention_score"), (int, float)) else np.nan
            primary_quality[i] = float(quality.get("quality_proxy_score")) if isinstance(quality.get("quality_proxy_score"), (int, float)) else np.nan
            primary_sharpness[i] = float(quality.get("face_sharpness")) if isinstance(quality.get("face_sharpness"), (int, float)) else np.nan
            primary_occlusion[i] = float(quality.get("occlusion_proxy")) if isinstance(quality.get("occlusion_proxy"), (int, float)) else np.nan
            primary_speech[i] = float(lip.get("speech_activity_prob")) if isinstance(lip.get("speech_activity_prob"), (int, float)) else np.nan
        
        # Формируем summary
        total_faces = sum(len(faces) for faces in outputs.values())
        primary_faces = sum(
            1 for faces in outputs.values()
            for face in faces
            if face.get("is_primary_face", False)
        )
        
        summary = {
            "axis_frames": int(n),
            "frames_with_faces_total": int(np.sum(face_present)),
            "frames_with_faces_processed": int(np.sum(face_present & processed_mask)),
            "processed_frames": int(np.sum(processed_mask)),
            "total_faces": total_faces,
            "primary_faces": primary_faces,
            "avg_faces_per_processed_face_frame": float(total_faces / max(int(np.sum(processed_mask)), 1)) if outputs else 0.0,
            "stage_timings_ms": {},
        }
        
        # Формируем агрегаты по tracking_id
        per_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for frame_faces in outputs.values():
            for face in frame_faces:
                tracking_id = face.get("tracking_id")
                if isinstance(tracking_id, int):
                    per_track[tracking_id].append(face)

        faces_agg = {
            int(track_id): extract_per_face_aggregates(faces)
            for track_id, faces in per_track.items()
        }

        # Model-facing aggregated stats (tabular head friendly).
        # Use only frames where we have a primary face AND we actually processed the frame.
        valid_mask = (processed_mask.astype(bool) & primary_valid.astype(bool))
        valid_idx = np.where(valid_mask)[0]
        if valid_idx.size > 0:
            cf = primary_compact_features[valid_idx, :]  # (M, 40)
            cf = np.asarray(cf, dtype=np.float32)
            cf_mean = np.mean(cf, axis=0).astype(np.float32)
            cf_std = np.std(cf, axis=0).astype(np.float32)
            cf_p10 = np.percentile(cf, 10, axis=0).astype(np.float32)
            cf_p90 = np.percentile(cf, 90, axis=0).astype(np.float32)

            l2 = np.linalg.norm(cf, axis=1).astype(np.float32)
            l2_mean = float(np.mean(l2))
            l2_std = float(np.std(l2))
            l2_p10 = float(np.percentile(l2, 10))
            l2_p90 = float(np.percentile(l2, 90))
        else:
            cf_mean = np.zeros((40,), dtype=np.float32)
            cf_std = np.zeros((40,), dtype=np.float32)
            cf_p10 = np.zeros((40,), dtype=np.float32)
            cf_p90 = np.zeros((40,), dtype=np.float32)
            l2_mean = 0.0
            l2_std = 0.0
            l2_p10 = 0.0
            l2_p90 = 0.0

        aggregated = {
            "schema_version": "detalize_face_aggregated_v1",
            "valid_frames": int(valid_idx.size),
            "axis_frames": int(n),
            "face_present_ratio": float(np.mean(face_present.astype(np.float32))) if n > 0 else 0.0,
            "processed_ratio": float(np.mean(processed_mask.astype(np.float32))) if n > 0 else 0.0,
            "primary_valid_ratio": float(np.mean(primary_valid.astype(np.float32))) if n > 0 else 0.0,
            "compact_dim": 40,
            "compact_mean": cf_mean,
            "compact_std": cf_std,
            "compact_p10": cf_p10,
            "compact_p90": cf_p90,
            "compact_l2_mean": float(l2_mean),
            "compact_l2_std": float(l2_std),
            "compact_l2_p10": float(l2_p10),
            "compact_l2_p90": float(l2_p90),
        }

        result: Dict[str, Any] = {
            "summary": summary,
            "frame_indices": np.asarray(axis_frame_indices, dtype=np.int32),
            "times_s": np.asarray(times_s, dtype=np.float32).reshape(-1),
            "face_present": np.asarray(face_present, dtype=bool).reshape(-1),
            "processed_mask": np.asarray(processed_mask, dtype=bool).reshape(-1),
            "primary_valid": np.asarray(primary_valid, dtype=bool).reshape(-1),
            "face_count": face_count.astype(np.float32),
            "primary_tracking_id": primary_tracking_id.astype(np.int32),
            "primary_compact_features": primary_compact_features.astype(np.float32),
            "aggregated": aggregated,
            "faces_agg": faces_agg,
        }

        if write_primary_curves:
            assert primary_gaze is not None
            assert primary_blink_rate is not None
            assert primary_attention is not None
            assert primary_quality is not None
            assert primary_sharpness is not None
            assert primary_occlusion is not None
            assert primary_speech is not None
            result.update(
                {
                    "primary_gaze_at_camera_prob": primary_gaze.astype(np.float32),
                    "primary_blink_rate": primary_blink_rate.astype(np.float32),
                    "primary_attention_score": primary_attention.astype(np.float32),
                    "primary_quality_proxy_score": primary_quality.astype(np.float32),
                    "primary_face_sharpness": primary_sharpness.astype(np.float32),
                    "primary_occlusion_proxy": primary_occlusion.astype(np.float32),
                    "primary_speech_activity_prob": primary_speech.astype(np.float32),
                }
            )
        
        return result
    
    def _empty_result(self, total_frames: int) -> Dict[str, Any]:
        """Возвращает пустой результат в правильном формате."""
        return {
            "summary": {
                "total_frames": total_frames,
                "processed_frames": 0,
                "frames_with_faces": 0,
                "total_faces": 0,
                "primary_faces": 0,
                "avg_faces_per_frame": 0.0,
                "stage_timings_ms": {},
            },
            "frame_indices": np.asarray([], dtype=np.int32),
            "times_s": np.asarray([], dtype=np.float32),
            "face_present": np.asarray([], dtype=bool),
            "processed_mask": np.asarray([], dtype=bool),
            "primary_valid": np.asarray([], dtype=bool),
            "face_count": np.asarray([], dtype=np.float32),
            "primary_tracking_id": np.asarray([], dtype=np.int32),
            "primary_compact_features": np.zeros((0, 40), dtype=np.float32),
            "aggregated": {
                "schema_version": "detalize_face_aggregated_v1",
                "valid_frames": 0,
                "axis_frames": 0,
                "face_present_ratio": 0.0,
                "processed_ratio": 0.0,
                "primary_valid_ratio": 0.0,
                "compact_dim": 40,
                "compact_mean": np.zeros((40,), dtype=np.float32),
                "compact_std": np.zeros((40,), dtype=np.float32),
                "compact_p10": np.zeros((40,), dtype=np.float32),
                "compact_p90": np.zeros((40,), dtype=np.float32),
                "compact_l2_mean": 0.0,
                "compact_l2_std": 0.0,
                "compact_l2_p10": 0.0,
                "compact_l2_p90": 0.0,
            },
            "faces_agg": {},
        }

    def _build_ui_payload(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Готовит JSON-представление для backend/UI."""
        frame_indices = results.get("frame_indices")
        times_s = results.get("times_s")

        if isinstance(frame_indices, np.ndarray):
            frame_indices_list = frame_indices.astype(int).tolist()
        else:
            frame_indices_list = [int(x) for x in (frame_indices or [])]

        if isinstance(times_s, np.ndarray):
            times_s_list = times_s.tolist()
        else:
            times_s_list = list(times_s or [])

        payload = {
            "component": self.module_name,
            "schema_version": "detalize_face_ui_v2",
            "frame_indices": frame_indices_list,
            "times_s": times_s_list,
            "curves": {},
            "faces_agg": results.get("faces_agg", {}),
            "summary": results.get("summary", {}),
        }
        # Always expose non-heuristic curves
        payload["curves"]["face_count"] = {"npz_key": "face_count", "label": "faces per frame"}
        payload["curves"]["face_present"] = {"npz_key": "face_present", "label": "face present (mask)"}
        payload["curves"]["processed_mask"] = {"npz_key": "processed_mask", "label": "processed (mask)"}
        payload["curves"]["primary_valid"] = {"npz_key": "primary_valid", "label": "primary face valid (mask)"}
        payload["curves"]["primary_tracking_id"] = {"npz_key": "primary_tracking_id", "label": "primary tracking_id"}
        payload["curves"]["primary_compact_features"] = {"npz_key": "primary_compact_features", "label": "primary compact features (40d, model-facing)"}
        # Optional heuristic curves (only if present in results)
        optional = [
            ("primary_gaze_at_camera_prob", "gaze at camera (heuristic)"),
            ("primary_blink_rate", "blink rate (heuristic)"),
            ("primary_attention_score", "attention score (heuristic)"),
            ("primary_quality_proxy_score", "quality proxy (heuristic)"),
            ("primary_face_sharpness", "sharpness (heuristic)"),
            ("primary_occlusion_proxy", "occlusion proxy (heuristic)"),
            ("primary_speech_activity_prob", "speech activity (heuristic)"),
        ]
        for k, label in optional:
            if k in results:
                payload["curves"][k] = {"npz_key": k, "label": label}
        return payload

    def run(
        self,
        frames_dir: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Полный цикл обработки (NPZ-only, baseline).
        
        IMPORTANT (Audit v3): output is aligned to Segmenter's per-component axis (`metadata[detalize_face].frame_indices`).
        The module computes expensive features only for frames where `core_face_landmarks` found faces (and optionally
        subsamples among them), but always writes full-axis arrays with `face_present`/`processed_mask`.
        """
        if metadata is None:
            metadata = self.load_metadata(frames_dir)

        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")

        platform_id = str(metadata.get("platform_id") or "")
        video_id = str(metadata.get("video_id") or "")
        run_id = str(metadata.get("run_id") or "")

        def _resource_profile_snapshot() -> Dict[str, Any]:
            """
            Best-effort, env-gated resource snapshot for Audit 4.2.
            """
            if str(os.environ.get("VP_RESOURCE_PROFILE", "")).strip().lower() not in ("1", "true", "yes", "on"):
                return {}
            snap: Dict[str, Any] = {}
            try:
                import psutil  # type: ignore
                snap["rss_mb"] = float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                pass
            return snap

        resource_profile_before = _resource_profile_snapshot()

        frame_manager = None
        try:
            if self.rs_path is not None:
                _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="start")
            t0 = time.perf_counter()
            frame_manager = self.create_frame_manager(frames_dir, metadata)
            t_fm = time.perf_counter()

            # Axis for this module (Segmenter contract).
            # Preferred: dedicated `metadata[detalize_face].frame_indices`
            # Fallback (for legacy Segmenter metadata): `metadata[core_face_landmarks].frame_indices`
            axis_frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
            axis_source = "detalize_face"
            if not axis_frame_indices:
                core_section = metadata.get("core_face_landmarks", {})
                core_fi = core_section.get("frame_indices") if isinstance(core_section, dict) else None
                if isinstance(core_fi, list) and core_fi:
                    axis_frame_indices = [int(x) for x in core_fi]
                    axis_source = "core_face_landmarks"
                    self.logger.warning(
                        f"{self.module_name} | missing metadata[detalize_face].frame_indices; "
                        f"falling back to metadata[core_face_landmarks].frame_indices for axis alignment"
                    )
                else:
                    raise RuntimeError(
                        f"{self.module_name} | missing/empty frame_indices in frames metadata "
                        f"(expected metadata[detalize_face].frame_indices or metadata[core_face_landmarks].frame_indices)"
                    )

            # Alignment check: warn if axis has frames not in core_face_landmarks.
            # According to schema, module should align output to Segmenter axis but only compute
            # features for frames with faces. Missing frames will have NaN/0 + masks.
            if self._core_frame_indices_all:
                core_set = set(int(x) for x in self._core_frame_indices_all)
                missing = [int(x) for x in axis_frame_indices if int(x) not in core_set]
                if missing:
                    logger.warning(
                        f"{self.module_name} | axis frame_indices partially covered by core_face_landmarks.frame_indices. "
                        f"Missing {len(missing)} indices (example: {missing[:5]}). "
                        f"These frames will have face_present=False and processed_mask=False."
                    )

            # Determine face frames and internal sampling for progress reporting (compute happens inside process()).
            core = self.extractor.core_landmarks or {}
            face_frames = [int(fi) for fi in axis_frame_indices if int(fi) in core]
            frames_to_process = self._sample_face_frames(face_frames)

            if self.rs_path is not None:
                _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="process")
                _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=int(len(frames_to_process)), stage="process")

            results = self.process(
                frame_manager=frame_manager,
                frame_indices=[int(x) for x in axis_frame_indices],
                config=config,
            )
            t_proc = time.perf_counter()

            # Add video context (total_frames) + timings into summary
            if isinstance(results, dict) and isinstance(results.get("summary"), dict):
                results["summary"]["total_frames_video"] = int(metadata.get("total_frames") or 0)
                st = results["summary"].get("stage_timings_ms") if isinstance(results["summary"].get("stage_timings_ms"), dict) else {}
                st["frame_manager_ms"] = float((t_fm - t0) * 1000.0)
                st["process_ms"] = float((t_proc - t_fm) * 1000.0)
                st["total_ms"] = float((t_proc - t0) * 1000.0)
                results["summary"]["stage_timings_ms"] = st

            # Метаданные для сохранения
            save_metadata = {
                "total_frames": metadata.get("total_frames"),
                "processed_frames": int(results.get("summary", {}).get("processed_frames", 0)),
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
            }
            if resource_profile_before:
                save_metadata["resource_profile_before"] = resource_profile_before

            # Sampling / axis policy
            save_metadata["module_sampling_policy_version"] = (
                "segmenter_axis_v1" if axis_source == "detalize_face" else "core_face_landmarks_axis_fallback_v1"
            )
            save_metadata["face_frames_sampling_policy_version"] = (
                "faces_all_v1"
                if (self.max_face_frames is None)
                else f"faces_uniform_v1_max_{int(self.max_face_frames)}"
            )
            save_metadata["write_primary_curves"] = bool(self.write_primary_curves)
            save_metadata["write_primary_compact_features"] = bool(self.write_primary_compact_features)

            # Stage timings are required by Audit v3 meta contract (put in meta, not only in `summary`).
            try:
                st = results.get("summary", {}).get("stage_timings_ms", {})
                save_metadata["stage_timings_ms"] = st if isinstance(st, dict) else {}
            except Exception:
                save_metadata["stage_timings_ms"] = {}

            is_empty = results.get("summary", {}).get("frames_with_faces_total", 0) == 0
            if is_empty:
                save_metadata["status"] = "empty"
                save_metadata["empty_reason"] = "no_faces_in_video"

            try:
                save_metadata["models_used"] = self.get_models_used(config=config or {}, metadata=metadata or {})
            except Exception:
                save_metadata["models_used"] = []

            # UI payload must live in NPZ meta (no extra JSON artifacts in result_store).
            try:
                save_metadata["ui_payload"] = self._build_ui_payload(results)
            except Exception:
                save_metadata["ui_payload"] = {}

            if self.rs_path is not None:
                _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="save")
            saved_path = self.save_results(results=results, metadata=save_metadata)
            if self.rs_path is not None:
                _emit_stage(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, stage="done")
            return saved_path
        finally:
            if frame_manager is not None:
                try:
                    frame_manager.close()
                except Exception as e:
                    self.logger.exception(
                        f"{self.module_name} | Ошибка при закрытии FrameManager: {e}"
                    )