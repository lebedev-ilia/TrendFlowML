"""
Модуль для комплексного анализа поведения людей в видео.
Реализует все недостающие фичи из FEATURES.MD:
- Детальная классификация жестов рук
- Body language анализ
- Speech-driven behavior
- Engagement Index
- Confidence/Dominance Index
- Signs of stress/anxiety

Модуль для комплексного анализа поведения людей в видео.
Реализует все недостающие фичи из FEATURES.MD:
- Детальная классификация жестов рук
- Body language анализ
- Speech-driven behavior
- Engagement Index
- Confidence/Dominance Index
- Signs of stress/anxiety

Все TODO выполнены:
✓ Переделана логика использования landmarks под работу с массивами numpy
✓ Модуль оптимизирован под работу с BaseModule
✓ Выход приведен к единому формату для сохранения в npz
"""

import os
from modules.base_module import BaseModule

import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque

from utils.frame_manager import FrameManager
from utils.logger import get_logger


def _require_union_times_s(frame_manager: FrameManager, frame_indices: List[int]) -> np.ndarray:
    meta = getattr(frame_manager, "meta", {}) if frame_manager is not None else {}
    union_ts = meta.get("union_timestamps_sec")
    if union_ts is None:
        raise RuntimeError("behavioral | FrameManager.meta missing union_timestamps_sec (strict time axis)")
    uts = np.asarray(union_ts, dtype=np.float32).reshape(-1)
    fi = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if np.any(fi < 0) or np.any(fi >= int(uts.shape[0])):
        raise RuntimeError("behavioral | frame_indices out of range for union_timestamps_sec")
    times_s = uts[fi]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        logger.warning("behavioral | times_s is not monotonic; check Segmenter sampling")
    return times_s.astype(np.float32)

MODULE_NAME = "behavioral"
SCHEMA_VERSION = "behavioral_npz_v1"
ARTIFACT_FILENAME = "behavioral_features.npz"
logger = get_logger(MODULE_NAME)


class HandGestureClassifier:
    """
    Детальная классификация жестов рук.

    ВАЖНО: Классификатор теперь используется только как источник
    вероятностного распределения по жестам (soft representation),
    без `unknown` класса. Для задач sequence-моделирования мы хотим
    гладкий вектор вероятностей, а не жёсткий one-hot/label.
    """
    
    def __init__(self):
        # Базовое множество "осмысленных" жестов (без unknown)
        self.gesture_types = {
            'pointing': self._is_pointing,
            'open_palm': self._is_open_palm,
            'hands_on_hips': self._is_hands_on_hips,
            'self_touch': self._is_self_touch,
            'fist': self._is_fist,
            'thumbs_up': self._is_thumbs_up,
            'thumbs_down': self._is_thumbs_down,
            'victory': self._is_victory,
            'ok': self._is_ok,
            'rock': self._is_rock,
            'call_me': self._is_call_me,
            'love': self._is_love
        }
    
    def _get_finger_states(self, hand_landmarks):
        """
        Определяет состояние пальцев.
        
        Args:
            hand_landmarks: numpy массив формы (21, 3) с координатами [x, y, z]
        """
        if isinstance(hand_landmarks, np.ndarray):
            # Работа с numpy массивом
            if hand_landmarks.shape[0] < 21:
                return {}
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [2, 6, 10, 14, 18]
            
            states = {}
            for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips)):
                tip = hand_landmarks[tip_idx]  # [x, y, z]
                pip = hand_landmarks[pip_idx]  # [x, y, z]
                
                if i == 0:  # thumb
                    states['thumb'] = tip[0] < pip[0] if tip[0] < 0.5 else tip[0] > pip[0]
                else:
                    states[['index', 'middle', 'ring', 'pinky'][i-1]] = tip[1] < pip[1]
            
            return states
        else:
            # Обратная совместимость с MediaPipe объектами
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [2, 6, 10, 14, 18]
            
            states = {}
            for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips)):
                tip = hand_landmarks.landmark[tip_idx]
                pip = hand_landmarks.landmark[pip_idx]
                
                if i == 0:  # thumb
                    states['thumb'] = tip.x < pip.x if tip.x < 0.5 else tip.x > pip.x
                else:
                    states[['index', 'middle', 'ring', 'pinky'][i-1]] = tip.y < pip.y
            
            return states
    
    def _is_pointing(self, hand_landmarks):
        """Указание рукой"""
        states = self._get_finger_states(hand_landmarks)
        return states.get('index', False) and not any([
            states.get('middle', False),
            states.get('ring', False),
            states.get('pinky', False)
        ])
    
    def _is_open_palm(self, hand_landmarks, pose_landmarks=None):
        """Раскрытые ладони"""
        states = self._get_finger_states(hand_landmarks)
        return all(states.values())
    
    def _is_hands_on_hips(self, hand_landmarks, pose_landmarks):
        """Руки в боки"""
        if pose_landmarks is None:
            return False
        
        # Проверяем, что запястья находятся рядом с талией
        if isinstance(hand_landmarks, np.ndarray):
            wrist = hand_landmarks[0]  # [x, y, z]
            wrist_x = wrist[0]
            wrist_y = wrist[1]
        else:
            wrist = hand_landmarks.landmark[0]
            wrist_x = wrist.x
            wrist_y = wrist.y
        
        hip_idx = 23 if wrist_x < 0.5 else 24  # левое или правое бедро
        
        if isinstance(pose_landmarks, np.ndarray):
            if hip_idx < pose_landmarks.shape[0]:
                hip = pose_landmarks[hip_idx]  # [x, y, z, visibility]
                distance = abs(wrist_y - hip[1])
                return distance < 0.1
        else:
            if hip_idx < len(pose_landmarks.landmark):
                hip = pose_landmarks.landmark[hip_idx]
                distance = abs(wrist_y - hip.y)
                return distance < 0.1
        
        return False
    
    def _is_self_touch(self, hand_landmarks, pose_landmarks):
        """Self-touch жесты (поглаживание, почёсывание)"""
        if pose_landmarks is None:
            return False
        
        # Проверяем близость руки к лицу/голове
        if isinstance(hand_landmarks, np.ndarray):
            wrist = hand_landmarks[0]  # [x, y, z]
            wrist_y = wrist[1]
        else:
            wrist = hand_landmarks.landmark[0]
            wrist_y = wrist.y
        
        # Упрощенная проверка: если рука близко к верхней части кадра
        return wrist_y < 0.3
    
    def _is_fist(self, hand_landmarks, pose_landmarks=None):
        """Кулак"""
        states = self._get_finger_states(hand_landmarks)
        return not any(states.values())
    
    def _is_thumbs_up(self, hand_landmarks, pose_landmarks=None):
        """Большой палец вверх"""
        states = self._get_finger_states(hand_landmarks)
        if isinstance(hand_landmarks, np.ndarray):
            thumb_tip = hand_landmarks[4]
            thumb_mcp = hand_landmarks[2]
            return states.get('thumb', False) and thumb_tip[1] < thumb_mcp[1]
        else:
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            return states.get('thumb', False) and thumb_tip.y < thumb_mcp.y
    
    def _is_thumbs_down(self, hand_landmarks, pose_landmarks=None):
        """Большой палец вниз"""
        states = self._get_finger_states(hand_landmarks)
        if isinstance(hand_landmarks, np.ndarray):
            thumb_tip = hand_landmarks[4]
            thumb_mcp = hand_landmarks[2]
            return states.get('thumb', False) and thumb_tip[1] > thumb_mcp[1]
        else:
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            return states.get('thumb', False) and thumb_tip.y > thumb_mcp.y
    
    def _is_victory(self, hand_landmarks, pose_landmarks=None):
        """Победа (V)"""
        states = self._get_finger_states(hand_landmarks)
        return states.get('index', False) and states.get('middle', False) and not any([
            states.get('ring', False),
            states.get('pinky', False)
        ])
    
    def _is_ok(self, hand_landmarks, pose_landmarks=None):
        """OK знак"""
        states = self._get_finger_states(hand_landmarks)
        if isinstance(hand_landmarks, np.ndarray):
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]
            distance = np.sqrt((index_tip[0] - thumb_tip[0])**2 + (index_tip[1] - thumb_tip[1])**2)
        else:
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
        return states.get('thumb', False) and distance < 0.05 and not states.get('index', False)
    
    def _is_rock(self, hand_landmarks, pose_landmarks=None):
        """Рок (рога)"""
        states = self._get_finger_states(hand_landmarks)
        return states.get('index', False) and states.get('pinky', False) and not any([
            states.get('middle', False),
            states.get('ring', False)
        ])
    
    def _is_call_me(self, hand_landmarks, pose_landmarks=None):
        """Позвони мне"""
        states = self._get_finger_states(hand_landmarks)
        if isinstance(hand_landmarks, np.ndarray):
            pinky_tip = hand_landmarks[20]
            thumb_tip = hand_landmarks[4]
            distance = np.sqrt((pinky_tip[0] - thumb_tip[0])**2 + (pinky_tip[1] - thumb_tip[1])**2)
        else:
            pinky_tip = hand_landmarks.landmark[20]
            thumb_tip = hand_landmarks.landmark[4]
            distance = np.sqrt((pinky_tip.x - thumb_tip.x)**2 + (pinky_tip.y - thumb_tip.y)**2)
        return states.get('pinky', False) and distance < 0.05
    
    def _is_love(self, hand_landmarks, pose_landmarks=None):
        """Любовь (сердце)"""
        states = self._get_finger_states(hand_landmarks)
        if isinstance(hand_landmarks, np.ndarray):
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]
            distance = np.sqrt((index_tip[0] - thumb_tip[0])**2 + (index_tip[1] - thumb_tip[1])**2)
        else:
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
        return (
            states.get('thumb', False)
            and states.get('index', False)
            and states.get('middle', False)
            and not states.get('ring', False)
            and not states.get('pinky', False)
            and distance < 0.06
        )
    
    def classify_gesture_hard(self, hand_landmarks, pose_landmarks=None) -> str:
        """Жёсткая классификация жеста (для обратной совместимости/отладки)."""
        for gesture_name, check_func in self.gesture_types.items():
            try:
                if check_func(hand_landmarks, pose_landmarks):
                    return gesture_name
            except Exception:
                continue
        # В новой схеме стараемся не использовать unknown дальше по пайплайну,
        # но для дебага всё ещё возвращаем его здесь.
        return 'unknown'

    def classify_gesture_soft(self, hand_landmarks, pose_landmarks=None) -> Dict[str, float]:
        """
        Мягкое представление жеста: распределение вероятностей по предопределённым типам.

        Т.к. базовые правила детектора дискретные, мы эмулируем "мягкость":
        - все жесты получают небольшой базовый вес (epsilon),
        - найденный по правилам жест получает повышенный вес,
        - затем нормируем вектор до суммы 1.0.
        """
        epsilon = 1e-3
        scores = {g: epsilon for g in self.gesture_types.keys()}

        detected = None
        for gesture_name, check_func in self.gesture_types.items():
            try:
                if check_func(hand_landmarks, pose_landmarks):
                    detected = gesture_name
                    break
            except Exception:
                continue

        if detected is not None:
            # усиливаем найденный класс
            scores[detected] = 1.0

        total = float(sum(scores.values())) or 1.0
        probs = {k: float(v / total) for k, v in scores.items()}
        return probs


class BodyLanguageAnalyzer:
    """Анализ языка тела"""
    
    def __init__(self):
        pass
    
    def analyze_posture(self, pose_landmarks, image_shape):
        """
        Анализирует язык тела.

        В НОВОЙ СХЕМЕ:
        - вместо дискретных поз/ярлыков возвращаем непрерывные физические сигналы:
          * arm_openness
          * pose_expansion
          * body_lean_angle
          * balance_offset
          * shoulder_angle
        Старые флаги (`open_posture`, `closed_posture`, `power_pose`, `rigidity`, ...),
        а также posture='standing/sitting' используются только для обратной
        совместимости и могут быть убраны на следующих шагах.
        
        Args:
            pose_landmarks: numpy массив формы (33, 4) где 4 = [x, y, z, visibility]
                          или объект MediaPipe Pose
            image_shape: форма изображения (h, w, ...)
        """
        if pose_landmarks is None:
            return {}
        
        h, w = image_shape[:2]
        
        def get_coord(idx):
            if isinstance(pose_landmarks, np.ndarray):
                # numpy массив: (33, 4) где 4 = [x, y, z, visibility]
                if idx >= pose_landmarks.shape[0]:
                    return None
                lm = pose_landmarks[idx]
                # Проверяем visibility (если < 0.5, считаем точку невидимой)
                if len(lm) > 3 and float(lm[3]) < 0.5:
                    return None
                return np.array([float(lm[0]) * w, float(lm[1]) * h])
            else:
                # MediaPipe объект
                if idx >= len(pose_landmarks.landmark):
                    return None
                lm = pose_landmarks.landmark[idx]
                return np.array([lm.x * w, lm.y * h])
        
        # Ключевые точки
        left_shoulder = get_coord(11)
        right_shoulder = get_coord(12)
        left_hip = get_coord(23)
        right_hip = get_coord(24)
        left_wrist = get_coord(15)
        right_wrist = get_coord(16)
        nose = get_coord(0)
        
        if any(x is None for x in [left_shoulder, right_shoulder, left_hip, right_hip, nose]):
            return {}
        
        results = {}
        
        # Базовые геометрические величины
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        pelvis_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # ---------
        # Старые флаги (сохраняем временно для обратной совместимости)
        # ---------

        # Поза (стоя/сидя) – будет удалена из FEATURES_DESCRIPTION
        shoulder_hip_distance = np.mean([
            np.linalg.norm(left_shoulder - left_hip),
            np.linalg.norm(right_shoulder - right_hip)
        ])
        results['posture'] = 'standing' if shoulder_hip_distance > h * 0.2 else 'sitting'
        
        # Открытая/закрытая поза (будет удалено из внешнего API)
        if left_wrist is not None and right_wrist is not None:
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            results['open_posture'] = wrist_distance > shoulder_width * 1.2
            results['closed_posture'] = wrist_distance < shoulder_width * 0.8
        else:
            results['open_posture'] = False
            results['closed_posture'] = False
        
        # Power pose (будет удалено из FEATURES_DESCRIPTION)
        if left_wrist is not None and right_wrist is not None:
            hip_center = pelvis_center
            wrist_center = (left_wrist + right_wrist) / 2
            vertical_distance = float(abs(wrist_center[1] - hip_center[1]))
            horizontal_spread = float(np.linalg.norm(left_wrist - right_wrist))
            
            results['power_pose'] = (
                vertical_distance < float(h) * 0.1 and
                horizontal_spread > float(shoulder_width) * 1.5
            )
        else:
            results['power_pose'] = False
        
        # Напряженность (rigidity)
        shoulder_angle_deg = float(np.degrees(np.arctan2(
            float(right_shoulder[1] - left_shoulder[1]),
            float(right_shoulder[0] - left_shoulder[0])
        )))
        results['rigidity'] = abs(shoulder_angle_deg) < 5.0
        
        # Расслабленность
        results['relaxed'] = not results.get('rigidity', False) and not results.get('closed_posture', False)
        
        # ---------
        # Новые непрерывные признаки
        # ---------

        # 1) Arm openness: wrist_distance / shoulder_width
        if left_wrist is not None and right_wrist is not None and shoulder_width > 1e-6:
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            arm_openness = float(wrist_distance / shoulder_width)
        else:
            arm_openness = 0.0
        results['arm_openness'] = arm_openness

        # 2) Pose expansion: отношение площади bbox человека к площади кадра
        keypoints = [left_shoulder, right_shoulder, left_hip, right_hip]
        if left_wrist is not None:
            keypoints.append(left_wrist)
        if right_wrist is not None:
            keypoints.append(right_wrist)

        xs = [p[0] for p in keypoints]
        ys = [p[1] for p in keypoints]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        person_area = max(0.0, (max_x - min_x) * (max_y - min_y))
        frame_area = float(w * h) if w > 0 and h > 0 else 1.0
        pose_expansion = float(person_area / frame_area)
        results['pose_expansion'] = pose_expansion

        # 3) Body lean angle (backward → forward, нормировано в [-1, 1])
        if nose is not None:
            # Вектор от центра таза к носу
            body_vec = nose - pelvis_center
            # Рассматриваем наклон вперёд/назад вдоль оси Y; нормируем на высоту кадра
            lean_raw = -(body_vec[1]) / max(float(h), 1.0)
            body_lean_angle = float(np.clip(lean_raw * 5.0, -1.0, 1.0))
        else:
            body_lean_angle = 0.0
        results['body_lean_angle'] = body_lean_angle

        # 4) Balance offset (как и раньше, [-1,1] влево/вправо)
        center_top = shoulder_center
        center_bottom = pelvis_center
        center_of_mass = (center_top + center_bottom) / 2
        frame_center_x = w / 2
        results['balance_offset'] = float((center_of_mass[0] - frame_center_x) / max(float(w), 1.0))

        # 5) Shoulder angle (абсолютный угол в градусах, и служебно храним исходное значение)
        results['shoulder_angle'] = float(shoulder_angle_deg)
        
        return results


class SpeechBehaviorAnalyzer:
    """
    Анализ динамики рта/речи.

    В новой схеме храним только "сырые" непрерывные признаки и простую
    прокси-метрику речи, пригодные для подачи в VisualTransformer:
      - mouth_width_norm
      - mouth_height_norm
      - mouth_area_norm
      - mouth_velocity
      - mouth_open_ratio
      - speech_activity_proxy
    """
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.mouth_history = deque(maxlen=window_size)
    
    def analyze_mouth_dynamics(self, face_landmarks, image_shape):
        """
        Анализирует динамику рта и прокси-активность речи.
        
        Args:
            face_landmarks: numpy массив формы (max_num_faces, 468, 3) где 3 = [x, y, z]
                          или объект MediaPipe Face Mesh
            image_shape: форма изображения (h, w, ...)
        """
        if face_landmarks is None:
            return {}
        
        h, w = image_shape[:2]
        
        # Индексы точек губ (MediaPipe Face Mesh)
        upper_lip_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        lower_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        
        def get_coord(idx):
            if isinstance(face_landmarks, np.ndarray):
                # numpy массив: (max_num_faces, 468, 3) - берем первый face (индекс 0)
                if face_landmarks.shape[0] == 0 or idx >= face_landmarks.shape[1]:
                    return None
                # Проверяем, что точка не NaN
                lm = face_landmarks[0, idx]  # [x, y, z]
                if np.any(np.isnan(lm)):
                    return None
                return np.array([float(lm[0]) * float(w), float(lm[1]) * float(h)])
            else:
                # MediaPipe объект
                if idx >= len(face_landmarks.landmark):
                    return None
                lm = face_landmarks.landmark[idx]
                return np.array([lm.x * w, lm.y * h])
        
        # Вычисляем параметры рта
        upper_lip_points = [get_coord(i) for i in upper_lip_indices if get_coord(i) is not None]
        lower_lip_points = [get_coord(i) for i in lower_lip_indices if get_coord(i) is not None]
        
        if not upper_lip_points or not lower_lip_points:
            return {}
        
        upper_center = np.mean(upper_lip_points, axis=0)
        lower_center = np.mean(lower_lip_points, axis=0)
        
        # Ширина рта
        mouth_width = np.max([p[0] for p in upper_lip_points]) - np.min([p[0] for p in upper_lip_points])
        
        # Высота рта
        mouth_height = np.linalg.norm(upper_center - lower_center)
        
        # Площадь рта (приблизительно)
        mouth_area = mouth_width * mouth_height
        
        # Сохраняем в историю
        last_area = self.mouth_history[-1]['area'] if len(self.mouth_history) > 0 else None
        self.mouth_history.append({
            'width': mouth_width,
            'height': mouth_height,
            'area': mouth_area
        })
        
        # Мгновенная скорость изменения площади рта (proxy mouth_velocity)
        if last_area is not None:
            mouth_velocity = abs(mouth_area - last_area)
        else:
            mouth_velocity = 0.0

        # Нормировки
        frame_diag = float(np.sqrt(w ** 2 + h ** 2)) or 1.0
        mouth_width_norm = float(mouth_width / frame_diag)
        mouth_height_norm = float(mouth_height / frame_diag)
        mouth_area_norm = float(mouth_area / (w * h + 1e-6))

        # Отношение открытия
        mouth_open_ratio = float(mouth_height / max(mouth_width, 1.0))

        # Прокси активности речи: sigmoid(z(mouth_velocity))
        # Масштабируем скорость и прогоняем через сигмоиду
        scaled = mouth_velocity / (w * 0.01 + 1e-6)
        speech_activity_proxy = float(1.0 / (1.0 + np.exp(-scaled)))
        
        return {
            'mouth_width_norm': mouth_width_norm,
            'mouth_height_norm': mouth_height_norm,
            'mouth_area_norm': mouth_area_norm,
            'mouth_velocity': float(mouth_velocity),
            'mouth_open_ratio': mouth_open_ratio,
            'speech_activity_proxy': speech_activity_proxy,
        }


class EngagementAnalyzer:
    """
    Ранее: hand-crafted индекс вовлеченности на кадр.

    Теперь: интерфейс-заглушка для обратной совместимости.
    Высокоуровневые метрики вовлеченности должны вычисляться уже
    на уровне финальной головы (MLP), а не внутри behavioral.

    Этот класс оставлен, чтобы не ломать импорт, но не используется
    в новой схеме sequence features.
    """
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.engagement_history = deque(maxlen=window_size)
    
    def calculate_engagement(self, *args, **kwargs):
        """
        Возвращает пустую структуру. Логика engagement перенесена
        на уровень агрегированных фичей (post-hoc).
        """
        return {}


class ConfidenceAnalyzer:
    """
    Ранее: кадровый индекс уверенности/доминантности.

    В новой схеме confidence/dominance считаются уже из латентных
    представлений модели (MLP head), а не внутри MediaPipe-пайплайна.
    """
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.confidence_history = deque(maxlen=window_size)
    
    def calculate_confidence(self, *args, **kwargs):
        """Возвращает пустую структуру, логика вынесена наружу."""
        return {}


class StressAnalyzer:
    """Детекция признаков стресса и тревожности"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.blink_history = deque(maxlen=window_size)
        self.movement_history = deque(maxlen=window_size)
    
    def analyze_stress(self, face_landmarks, pose_landmarks, hand_landmarks_list, image_shape):
        """
        Анализирует "сырые" признаки стресса без интерпретаций:
          - blink_flag / blink_rate_short
          - self_touch_flag
          - fidgeting_energy
        """
        h, w = image_shape[:2]

        blink_flag = 0
        blink_rate_short = 0.0
        self_touch_flag = 0
        fidgeting_energy = 0.0
        
        # 1. Моргание (EAR)
        if face_landmarks is not None:
            left_ear = self._calculate_ear(face_landmarks, image_shape, 'left')
            right_ear = self._calculate_ear(face_landmarks, image_shape, 'right')
            avg_ear = (left_ear + right_ear) / 2
            
            is_blinking = avg_ear < 0.2
            blink_flag = int(is_blinking)
            self.blink_history.append(is_blinking)
            
            if len(self.blink_history) > 0:
                blink_rate_short = float(sum(self.blink_history) / len(self.blink_history))

        # 2. Self-touch (через классификацию жестов)
        if hand_landmarks_list:
            gesture_classifier = HandGestureClassifier()
            for hand_landmarks in hand_landmarks_list:
                gesture = gesture_classifier.classify_gesture_hard(hand_landmarks, pose_landmarks)
                if gesture == 'self_touch':
                    self_touch_flag = 1
                    break
        
        # 3. Fidgeting (вариативность позиции носа за последнее окно)
        if pose_landmarks is not None:
            if isinstance(pose_landmarks, np.ndarray):
                if pose_landmarks.shape[0] > 0:
                    nose = pose_landmarks[0]  # [x, y, z, visibility]
                    current_pos = np.array([nose[0], nose[1]])
                    self.movement_history.append(current_pos)
            else:
                if len(pose_landmarks.landmark) > 0:
                    nose = pose_landmarks.landmark[0]
                    current_pos = np.array([nose.x, nose.y])
                    self.movement_history.append(current_pos)
            
            if len(self.movement_history) >= 2:
                positions = np.stack(self.movement_history, axis=0)
                var_x = float(np.var(positions[:, 0]))
                var_y = float(np.var(positions[:, 1]))
                fidgeting_energy = var_x + var_y

        return {
            'blink_flag': int(blink_flag),
            'blink_rate_short': float(blink_rate_short),
            'self_touch_flag': int(self_touch_flag),
            'fidgeting_energy': float(fidgeting_energy),
        }
    
    def _calculate_ear(self, face_landmarks, image_shape, eye_type='left'):
        """
        Вычисляет Eye Aspect Ratio
        
        Args:
            face_landmarks: numpy массив формы (max_num_faces, 468, 3) или объект MediaPipe
            image_shape: форма изображения (h, w, ...)
            eye_type: 'left' или 'right'
        """
        h, w = image_shape[:2]
        
        if eye_type == 'left':
            indices = [33, 160, 158, 133, 153, 144]
        else:
            indices = [362, 385, 387, 263, 373, 380]
        
        def get_coord(idx):
            if isinstance(face_landmarks, np.ndarray):
                # numpy массив: (max_num_faces, 468, 3) - берем первый face
                if face_landmarks.shape[0] == 0 or idx >= face_landmarks.shape[1]:
                    return None
                lm = face_landmarks[0, idx]  # [x, y, z]
                if np.any(np.isnan(lm)):
                    return None
                return np.array([float(lm[0]) * float(w), float(lm[1]) * float(h)])
            else:
                # MediaPipe объект
                if idx >= len(face_landmarks.landmark):
                    return None
                lm = face_landmarks.landmark[idx]
                return np.array([float(lm.x) * float(w), float(lm.y) * float(h)])
        
        try:
            p1, p2, p3, p4, p5, p6 = [get_coord(i) for i in indices]
            if any(p is None for p in [p1, p2, p3, p4, p5, p6]):
                return 0.3  # по умолчанию открыт
            
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            h_dist = np.linalg.norm(p1 - p4)
            
            if h_dist > 0:
                ear = (v1 + v2) / (2.0 * h_dist)
            else:
                ear = 0.3
            return ear
        except:
            return 0.3


class BehaviorAnalyzer(BaseModule):
    """Главный класс для анализа поведения"""
    MODULE_NAME = MODULE_NAME
    SCHEMA_VERSION = SCHEMA_VERSION
    ARTIFACT_FILENAME = ARTIFACT_FILENAME
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(rs_path=rs_path, logger_name=MODULE_NAME, **kwargs)

        self.gesture_classifier = HandGestureClassifier()
        self.body_analyzer = BodyLanguageAnalyzer()
        self.speech_analyzer = SpeechBehaviorAnalyzer()
        self.engagement_analyzer = EngagementAnalyzer()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.stress_analyzer = StressAnalyzer()
        # для динамики головы/плеч/рук
        self._prev_head_center = None
        self._prev_shoulder_angle = None
        self._prev_hands_center = None
        self._last_times_s: Optional[np.ndarray] = None
        self._last_frame_indices: Optional[List[int]] = None
        self._last_landmarks_present: Optional[np.ndarray] = None
        self._last_empty_reason: Optional[str] = None
        self._last_core_meta: Optional[Dict[str, Any]] = None
    
    def required_dependencies(self) -> List[str]:
        """Возвращает список зависимостей модуля."""
        return ["core_face_landmarks"]
    
    @property
    def supports_batch(self) -> bool:
        """
        Behavioral модуль поддерживает batch processing.
        
        Для CPU модулей используется дефолтный process_batch() из BaseModule,
        который последовательно обрабатывает каждое видео.
        """
        return True
    
    # Метод process_frame удалён - используем только core_face_landmarks через _process_with_results
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Основной метод обработки видео (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля (не используется, но требуется BaseModule)
                
        Returns:
            Dict[frame_idx, Dict] - результаты по кадрам
        """
        self.initialize()  # Гарантируем инициализацию
        
        import time

        fps = frame_manager.fps if hasattr(frame_manager, 'fps') else 30.0

        landmarks_data = self.load_core_provider("core_face_landmarks", "landmarks.npz")
        
        if landmarks_data is None:
            raise RuntimeError(
                f"{self.module_name} | process | core_face_landmarks не найдены. "
                f"Убедитесь, что core провайдер core_face_landmarks запущен перед этим модулем. "
                f"rs_path: {self.rs_path}"
            )

        # Загружаем данные landmarks
        landmark_frame_indices = landmarks_data.get("frame_indices")
        pose = landmarks_data.get("pose_landmarks")  # (n_frames, 33, 4)
        hands = landmarks_data.get("hands_landmarks")  # (n_frames, max_num_hands, 21, 3)
        face = landmarks_data.get("face_landmarks")  # (n_frames, max_num_faces, 468, 3)
        self._last_core_meta = landmarks_data.get("meta") if isinstance(landmarks_data.get("meta"), dict) else None
        
        if landmark_frame_indices is None or pose is None or hands is None or face is None:
            raise ValueError(
                f"{self.module_name} | process | Неполные данные landmarks. "
                f"Требуются: frame_indices, pose_landmarks, hands_landmarks, face_landmarks"
            )
        
        # Преобразуем в numpy массивы если нужно
        if not isinstance(landmark_frame_indices, np.ndarray):
            landmark_frame_indices = np.array(landmark_frame_indices, dtype=np.int32)
        
        # Создаем маппинг: frame_idx -> index_in_landmarks_array
        frame_to_landmark_idx = {}
        for idx, frame_idx in enumerate(landmark_frame_indices):
            frame_to_landmark_idx[int(frame_idx)] = idx
        
        times_s = _require_union_times_s(frame_manager, frame_indices)
        self._last_times_s = times_s
        self._last_frame_indices = list(frame_indices)
        landmarks_present = np.zeros((len(frame_indices),), dtype=bool)
        self._last_landmarks_present = landmarks_present

        all_results: Dict[int, Dict[str, Any]] = {}
        c = 0
        t = time.time()
        missing_landmarks = 0

        for pos, frame_idx in enumerate(frame_indices):
            if int(frame_idx) not in frame_to_landmark_idx:
                missing_landmarks += 1
                all_results[frame_idx] = self._build_empty_frame_result(times_s[pos])
                continue

            # Проверяем наличие кадра в landmarks
            # Получаем индекс в массивах landmarks
            landmark_idx = frame_to_landmark_idx[int(frame_idx)]
            
            # Извлекаем данные для кадра
            pose_frame = pose[landmark_idx]  # (33, 4)
            hands_frame = hands[landmark_idx]  # (max_num_hands, 21, 3)
            face_frame = face[landmark_idx]  # (max_num_faces, 468, 3)

            frame = frame_manager.get(frame_idx)
            result = self._process_with_results(frame, pose_frame, hands_frame, face_frame)

            result['timestamp'] = float(times_s[pos])
            result['landmarks_present'] = True
            all_results[frame_idx] = result
            landmarks_present[pos] = True

            c += 1

            if c % 20 == 0:
                l = time.time()
                d = round(l - t, 2)
                t = l
                self.logger.info(f"{self.module_name} | Обработано кадров: {c}/{len(frame_indices)} | Time: {d}")
        
        if missing_landmarks:
            self.logger.warning(
                f"{self.module_name} | process | {missing_landmarks} кадров отсутствуют в core_face_landmarks. "
                "Заполнены NaN и отмечены masks."
            )

        # Нормализованный timestamp (t / video_duration)
        if len(times_s) > 0:
            t0 = float(times_s[0])
            t1 = float(times_s[-1]) if len(times_s) > 1 else float(times_s[0] + 1e-6)
            denom = max(t1 - t0, 1e-6)
            for pos, frame_idx in enumerate(frame_indices):
                res = all_results.get(frame_idx, {})
                t_abs = float(times_s[pos])
                t_rel = float(np.clip((t_abs - t0) / denom, 0.0, 1.0))
                seq = res.setdefault('sequence_features', {})
                seq['timestamp_norm'] = t_rel
                all_results[frame_idx] = res
        
        # Возвращаем per-track формат (в данном случае per-frame)
        # Для совместимости с BaseModule сохраняем как per-track результаты
        return all_results

    def _process_with_results(
        self,
        frame: np.ndarray,
        pose_frame: np.ndarray,
        hands_frame: np.ndarray,
        face_frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Обработка кадра на основе уже готовых результатов из core_face_landmarks.
        
        Args:
            frame: numpy массив кадра (H, W, 3)
            pose_frame: numpy массив формы (33, 4) где 4 = [x, y, z, visibility]
            hands_frame: numpy массив формы (max_num_hands, 21, 3) где 3 = [x, y, z]
            face_frame: numpy массив формы (max_num_faces, 468, 3) где 3 = [x, y, z]
        """
        h, w = frame.shape[:2]

        results: Dict[str, Any] = {}
        sequence_features: Dict[str, Any] = {}

        # 1. Руки / жесты
        hand_gestures = []
        hand_landmarks_list = []
        
        # Фильтруем руки с валидными данными (не все NaN)
        for i in range(hands_frame.shape[0]):
            hand_landmarks = hands_frame[i]  # (21, 3)
            # Проверяем, что рука валидна (не все NaN)
            if not np.all(np.isnan(hand_landmarks)):
                hand_landmarks_list.append(hand_landmarks)
                gesture = self.gesture_classifier.classify_gesture_hard(
                    hand_landmarks,
                    pose_frame
                )
                hand_gestures.append(gesture)

        results['hand_gestures'] = hand_gestures
        num_hands = len(hand_landmarks_list)
        results['num_hands'] = num_hands
        sequence_features['num_hands'] = int(num_hands)
        sequence_features['hands_visibility'] = 1 if num_hands > 0 else 0

        gesture_probs_accum = {g: 0.0 for g in self.gesture_classifier.gesture_types.keys()}
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                probs = self.gesture_classifier.classify_gesture_soft(
                    hand_landmarks,
                    pose_frame
                )
                for g, p in probs.items():
                    gesture_probs_accum[g] += float(p)
            for g in gesture_probs_accum.keys():
                gesture_probs_accum[g] /= float(len(hand_landmarks_list))
        sequence_features['gesture_probs'] = gesture_probs_accum

        current_hands_center = None
        if hand_landmarks_list:
            wrist_points = []
            for hand_landmarks in hand_landmarks_list:
                wrist = hand_landmarks[0]  # [x, y, z]
                if not np.any(np.isnan(wrist)):
                    wrist_points.append(np.array([wrist[0] * w, wrist[1] * h]))
            if wrist_points:
                current_hands_center = np.mean(wrist_points, axis=0)

        if current_hands_center is not None and self._prev_hands_center is not None:
            hand_motion_energy = float(np.linalg.norm(current_hands_center - self._prev_hands_center))
        else:
            hand_motion_energy = 0.0
        sequence_features['hand_motion_energy'] = hand_motion_energy
        self._prev_hands_center = current_hands_center

        # 2. Тело / поза
        # Проверяем, что pose_frame валиден (не все NaN)
        pose_valid = pose_frame is not None and not np.all(np.isnan(pose_frame))
        if pose_valid:
            body_language = self.body_analyzer.analyze_posture(
                pose_frame,
                frame.shape
            )
            if body_language and isinstance(body_language, dict) and len(body_language) > 0:  # Если есть результаты
                results['body_language'] = body_language

                sequence_features['arm_openness'] = float(body_language.get('arm_openness', 0.0))
                sequence_features['pose_expansion'] = float(body_language.get('pose_expansion', 0.0))
                sequence_features['body_lean_angle'] = float(body_language.get('body_lean_angle', 0.0))
                sequence_features['balance_offset'] = float(body_language.get('balance_offset', 0.0))

                shoulder_angle = float(body_language.get('shoulder_angle', 0.0))
                sequence_features['shoulder_angle'] = shoulder_angle

                if self._prev_shoulder_angle is not None:
                    shoulder_angle_velocity = abs(shoulder_angle - self._prev_shoulder_angle)
                else:
                    shoulder_angle_velocity = 0.0
                sequence_features['shoulder_angle_velocity'] = float(shoulder_angle_velocity)
                self._prev_shoulder_angle = shoulder_angle

        # 3. Голова / взгляд
        head_position_x_norm = 0.0
        head_position_y_norm = 0.0
        head_motion_energy = 0.0
        frame_diag = float(np.sqrt(w ** 2 + h ** 2)) or 1.0

        # Берем первый валидный face (не все NaN)
        face_landmarks_for_head = None
        if face_frame is not None and face_frame.shape[0] > 0:
            for face_idx in range(face_frame.shape[0]):
                face_landmarks = face_frame[face_idx]  # (468, 3)
                if not np.all(np.isnan(face_landmarks)):
                    face_landmarks_for_head = face_landmarks
                    break
        
        if face_landmarks_for_head is not None:
            # Вычисляем центр головы по всем landmarks
            valid_points = face_landmarks_for_head[~np.any(np.isnan(face_landmarks_for_head), axis=1)]
            if len(valid_points) > 0:
                w_float = float(w)
                h_float = float(h)
                xs = valid_points[:, 0] * w_float
                ys = valid_points[:, 1] * h_float
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                head_position_x_norm = cx / max(w_float, 1.0)
                head_position_y_norm = cy / max(h_float, 1.0)

                current_head_center = np.array([cx, cy])
                if self._prev_head_center is not None:
                    head_motion_energy = float(np.linalg.norm(current_head_center - self._prev_head_center) / frame_diag)
                self._prev_head_center = current_head_center

        sequence_features['head_position_x_norm'] = float(head_position_x_norm)
        sequence_features['head_position_y_norm'] = float(head_position_y_norm)
        sequence_features['head_motion_energy'] = float(head_motion_energy)
        sequence_features['head_stability'] = float(1.0 / (1.0 + head_motion_energy))

        # 4. Рот / речь
        if face_landmarks_for_head is not None:
            mouth_dynamics = self.speech_analyzer.analyze_mouth_dynamics(
                face_frame,  # Передаем весь face_frame для анализа
                frame.shape
            )
            if mouth_dynamics and isinstance(mouth_dynamics, dict) and len(mouth_dynamics) > 0:
                results['speech_behavior'] = mouth_dynamics
                sequence_features.update(mouth_dynamics)

        # 5. Стресс
        stress = self.stress_analyzer.analyze_stress(
            face_frame,  # Передаем весь face_frame
            pose_frame,
            hand_landmarks_list,
            frame.shape
        )
        if stress and isinstance(stress, dict) and len(stress) > 0:
            results['stress'] = stress
            sequence_features.update(stress)

        results['sequence_features'] = sequence_features
        return results

    def _build_empty_frame_result(self, timestamp_s: float) -> Dict[str, Any]:
        """Плейсхолдер для кадров без landmarks."""
        return {
            "hand_gestures": [],
            "num_hands": None,
            "sequence_features": {
                "num_hands": np.nan,
                "hands_visibility": np.nan,
                "gesture_probs": {g: np.nan for g in self.gesture_classifier.gesture_types.keys()},
                "hand_motion_energy": np.nan,
                "arm_openness": np.nan,
                "pose_expansion": np.nan,
                "body_lean_angle": np.nan,
                "balance_offset": np.nan,
                "shoulder_angle": np.nan,
                "shoulder_angle_velocity": np.nan,
                "head_position_x_norm": np.nan,
                "head_position_y_norm": np.nan,
                "head_motion_energy": np.nan,
                "head_stability": np.nan,
                "mouth_width_norm": np.nan,
                "mouth_height_norm": np.nan,
                "mouth_area_norm": np.nan,
                "mouth_velocity": np.nan,
                "mouth_open_ratio": np.nan,
                "speech_activity_proxy": np.nan,
                "blink_flag": np.nan,
                "blink_rate_short": np.nan,
                "self_touch_flag": np.nan,
                "fidgeting_energy": np.nan,
                "timestamp_norm": np.nan,
            },
            "timestamp": float(timestamp_s),
            "landmarks_present": False,
        }

    def _pack_npz_results(
        self,
        results: Dict[int, Dict[str, Any]],
        frame_indices: List[int],
        times_s: np.ndarray,
        landmarks_present: np.ndarray,
    ) -> Dict[str, Any]:
        n = len(frame_indices)
        frame_results = []
        hand_gestures = []
        seq_keys = [
            "num_hands",
            "hands_visibility",
            "hand_motion_energy",
            "arm_openness",
            "pose_expansion",
            "body_lean_angle",
            "balance_offset",
            "shoulder_angle",
            "shoulder_angle_velocity",
            "head_position_x_norm",
            "head_position_y_norm",
            "head_motion_energy",
            "head_stability",
            "mouth_width_norm",
            "mouth_height_norm",
            "mouth_area_norm",
            "mouth_velocity",
            "mouth_open_ratio",
            "speech_activity_proxy",
            "blink_flag",
            "blink_rate_short",
            "self_touch_flag",
            "fidgeting_energy",
            "timestamp_norm",
        ]
        seq_arrays = {k: np.full((n,), np.nan, dtype=np.float32) for k in seq_keys}
        gesture_prob_arrays = {
            g: np.full((n,), np.nan, dtype=np.float32) for g in self.gesture_classifier.gesture_types.keys()
        }

        for i, frame_idx in enumerate(frame_indices):
            res = results.get(frame_idx) or self._build_empty_frame_result(times_s[i])
            frame_results.append(self.make_serializable(res))
            hand_gestures.append(res.get("hand_gestures", []))
            seq = res.get("sequence_features") or {}
            for k in seq_keys:
                v = seq.get(k, np.nan)
                try:
                    seq_arrays[k][i] = float(v) if v is not None else np.nan
                except Exception:
                    seq_arrays[k][i] = np.nan
            probs = seq.get("gesture_probs") or {}
            for g in gesture_prob_arrays.keys():
                v = probs.get(g, np.nan)
                try:
                    gesture_prob_arrays[g][i] = float(v) if v is not None else np.nan
                except Exception:
                    gesture_prob_arrays[g][i] = np.nan

        packed: Dict[str, Any] = {
            "frame_indices": np.asarray(frame_indices, dtype=np.int32),
            "times_s": np.asarray(times_s, dtype=np.float32),
            "landmarks_present": np.asarray(landmarks_present, dtype=bool),
            "hand_gestures": np.asarray(hand_gestures, dtype=object),
            "frame_results": np.asarray(frame_results, dtype=object),
            "aggregated": np.asarray(self._aggregate_results(results), dtype=object),
        }

        for k, v in seq_arrays.items():
            packed[f"seq_{k}"] = np.asarray(v, dtype=np.float32)
        for g, v in gesture_prob_arrays.items():
            packed[f"seq_gesture_prob_{g}"] = np.asarray(v, dtype=np.float32)

        return packed

    def build_ui_payload(self, packed: Dict[str, Any]) -> Dict[str, Any]:
        def _safe_float(x: Any) -> Optional[float]:
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                return float(x)
            except Exception:
                return None

        def _downsample(times: np.ndarray, values: np.ndarray, max_points: int = 500) -> Dict[str, List[Any]]:
            if times.size == 0 or values.size == 0:
                return {"times_s": [], "values": []}
            step = max(1, int(np.ceil(float(times.size) / float(max_points))))
            t = times[::step]
            v = values[::step]
            out_vals = []
            for x in v:
                if isinstance(x, (float, np.floating)) and np.isnan(x):
                    out_vals.append(None)
                else:
                    try:
                        out_vals.append(float(x))
                    except Exception:
                        out_vals.append(None)
            return {
                "times_s": [float(x) for x in t.tolist()],
                "values": out_vals,
            }

        times_s_raw = packed.get("times_s")
        times_s = np.asarray(times_s_raw if times_s_raw is not None else [], dtype=np.float32).reshape(-1)
        agg = packed.get("aggregated")
        if isinstance(agg, np.ndarray) and agg.dtype == object:
            try:
                agg = agg.item()
            except Exception:
                agg = {}
        if not isinstance(agg, dict):
            agg = {}

        series = []
        def _series(name: str, key: str):
            arr_raw = packed.get(key)
            arr = np.asarray(arr_raw if arr_raw is not None else [], dtype=np.float32).reshape(-1)
            series.append({"name": name, **_downsample(times_s, arr)})

        _series("speech_activity_proxy", "seq_speech_activity_proxy")
        _series("arm_openness", "seq_arm_openness")
        _series("body_lean_angle", "seq_body_lean_angle")
        _series("hand_motion_energy", "seq_hand_motion_energy")
        _series("blink_rate_short", "seq_blink_rate_short")

        # Stress proxy for UI (same weights as aggregation)
        blink_raw = packed.get("seq_blink_rate_short")
        blink = np.asarray(blink_raw if blink_raw is not None else [], dtype=np.float32).reshape(-1)
        self_touch_raw = packed.get("seq_self_touch_flag")
        self_touch = np.asarray(self_touch_raw if self_touch_raw is not None else [], dtype=np.float32).reshape(-1)
        fidget_raw = packed.get("seq_fidgeting_energy")
        fidget = np.asarray(fidget_raw if fidget_raw is not None else [], dtype=np.float32).reshape(-1)
        stress_proxy = 0.4 * blink + 0.3 * self_touch + 0.3 * (1.0 / (1.0 + np.exp(-fidget * 10.0)))
        series.append({"name": "stress_proxy", **_downsample(times_s, stress_proxy)})

        return {
            "component": self.module_name,
            "schema_version": "behavioral_ui_v1",
            "summary": {
                "avg_engagement": _safe_float(agg.get("avg_engagement")),
                "avg_confidence": _safe_float(agg.get("avg_confidence")),
                "avg_stress": _safe_float(agg.get("avg_stress")),
                "gesture_rate_per_sec": _safe_float(agg.get("gesture_rate_per_sec")),
                "hands_visibility_ratio": _safe_float(agg.get("hands_visibility_ratio")),
                "face_visibility_ratio": _safe_float(agg.get("face_visibility_ratio")),
            },
            "distributions": {
                "gesture_counts": agg.get("gesture_counts") or {},
                "gesture_entropy_mean": _safe_float(agg.get("gesture_entropy_mean")),
            },
            "series": series,
        }

    def export_ui_json(self, npz_path: str, out_path: str) -> None:
        data = np.load(npz_path, allow_pickle=True)
        packed = {k: data[k] for k in data.files}
        payload = self.build_ui_payload(packed)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import json as _json
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False, indent=2)

    def make_serializable(self, obj):
        import numpy as np
        # numpy bool
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # python bool
        if isinstance(obj, bool):
            return obj
        # numpy int/float
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        # numpy array
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # tuple -> list
        if isinstance(obj, tuple):
            return [self.make_serializable(x) for x in obj]
        # list
        if isinstance(obj, list):
            return [self.make_serializable(x) for x in obj]
        # dict
        if isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        # objects with __dict__
        if hasattr(obj, "__dict__"):
            return self.make_serializable(obj.__dict__)
        return obj
    
    def _aggregate_results(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Агрегирует результаты по всему видео.

        Важно: здесь допустима интерпретация и high-level метрики,
        но они считаются из сырых sequence_features.
        """
        if not results:
            return {}
        
        aggregated: Dict[str, Any] = {}

        # Собираем последовательности по кадрам
        seq_list = [r.get('sequence_features', {}) for r in results.values()]

        def get_series(key: str):
            if not seq_list:
                return np.zeros(0, dtype=float)
            vals = []
            for s in seq_list:
                v = s.get(key, np.nan)
                try:
                    vals.append(float(v) if v is not None else np.nan)
                except Exception:
                    vals.append(np.nan)
            return np.asarray(vals, dtype=float)


        speech_proxy = get_series('speech_activity_proxy')
        arm_open = get_series('arm_openness')
        body_lean = get_series('body_lean_angle')

        def norm_sig(x):
            return 1.0 / (1.0 + np.exp(-x)) if np.isscalar(x) else 1.0 / (1.0 + np.exp(-x))

        engagement_signal = 0.5 * speech_proxy + 0.3 * norm_sig(arm_open) + 0.2 * norm_sig(body_lean)
        if engagement_signal.size == 0:
            engagement_signal = np.zeros(1)

        aggregated['avg_engagement'] = float(np.nanmean(engagement_signal))
        aggregated['max_engagement'] = float(np.nanmax(engagement_signal))
        aggregated['engagement_variance'] = float(np.nanvar(engagement_signal))

        engagement_peaks = 0
        if engagement_signal.size >= 3:
            for i in range(1, engagement_signal.size - 1):
                if engagement_signal[i] > engagement_signal[i - 1] and engagement_signal[i] > engagement_signal[i + 1]:
                    engagement_peaks += 1
        aggregated['engagement_peaks'] = int(engagement_peaks)

        n = engagement_signal.size
        if n >= 5:
            split = max(int(0.2 * n), 1)
            early = engagement_signal[:split]
            late = engagement_signal[-split:]
            aggregated['early_engagement_mean'] = float(np.nanmean(early))
            aggregated['late_engagement_mean'] = float(np.nanmean(late))
        else:
            aggregated['early_engagement_mean'] = float(np.nanmean(engagement_signal))
            aggregated['late_engagement_mean'] = float(np.nanmean(engagement_signal))


        confidence_signal = 0.6 * norm_sig(arm_open) + 0.4 * norm_sig(body_lean)
        if confidence_signal.size == 0:
            confidence_signal = np.zeros(1)

        aggregated['avg_confidence'] = float(np.nanmean(confidence_signal))
        aggregated['max_confidence'] = float(np.nanmax(confidence_signal))
        aggregated['confidence_variance'] = float(np.nanvar(confidence_signal))

        confidence_peaks = 0
        if confidence_signal.size >= 3:
            for i in range(1, confidence_signal.size - 1):
                if confidence_signal[i] > confidence_signal[i - 1] and confidence_signal[i] > confidence_signal[i + 1]:
                    confidence_peaks += 1
        aggregated['confidence_peak_count'] = int(confidence_peaks)


        blink_rate_short = get_series('blink_rate_short')
        self_touch_flag = get_series('self_touch_flag')
        fidgeting_energy = get_series('fidgeting_energy')

        stress_proxy = 0.4 * blink_rate_short + 0.3 * self_touch_flag + 0.3 * norm_sig(fidgeting_energy * 10.0)
        if stress_proxy.size == 0:
            stress_proxy = np.zeros(1)

        aggregated['avg_stress'] = float(np.nanmean(stress_proxy))
        aggregated['max_stress'] = float(np.nanmax(stress_proxy))
        mean_sp = float(np.nanmean(stress_proxy))
        std_sp = float(np.nanstd(stress_proxy))
        aggregated['stress_spike_count'] = int(np.nansum(stress_proxy > (mean_sp + std_sp)))
        aggregated['stress_duration_ratio'] = float(np.nanmean(stress_proxy > 0.5))


        all_gestures = []
        for r in results.values():
            all_gestures.extend(r.get('hand_gestures', []))
        gesture_counts: Dict[str, int] = {}
        for g in all_gestures:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        aggregated['gesture_counts'] = gesture_counts

        # gesture rate per second (requires real time axis)
        total_frames = max(len(seq_list), 1)
        if self._last_times_s is not None and self._last_times_s.size >= 2:
            duration_sec = max(float(self._last_times_s[-1] - self._last_times_s[0]), 1e-6)
        else:
            duration_sec = max(float(total_frames) / 30.0, 1e-6)
        aggregated['gesture_rate_per_sec'] = float(len(all_gestures) / duration_sec)

        # gesture entropy по soft распределениям
        entropies = []
        for s in seq_list:
            probs = s.get('gesture_probs', {})
            if not probs:
                continue
            p = np.array(list(probs.values()), dtype=float)
            p = p / (p.sum() + 1e-8)
            ent = float(-np.sum(p * np.log2(p + 1e-8)))
            entropies.append(ent)
        aggregated['gesture_entropy_mean'] = float(np.nanmean(entropies)) if entropies else 0.0

        if gesture_counts:
            dominant = max(gesture_counts.values())
            aggregated['dominant_gesture_ratio'] = float(dominant / max(sum(gesture_counts.values()), 1))
        else:
            aggregated['dominant_gesture_ratio'] = 0.0

        # Простая оценка скорости смены жестов
        gesture_switches = 0
        if all_gestures:
            for i in range(1, len(all_gestures)):
                if all_gestures[i] != all_gestures[i - 1]:
                    gesture_switches += 1
        aggregated['gesture_switching_rate'] = float(gesture_switches / max(total_frames - 1, 1))


        pose_expansion = get_series('pose_expansion')
        balance_offset = get_series('balance_offset')

        aggregated['avg_arm_openness'] = float(np.nanmean(arm_open)) if arm_open.size > 0 else 0.0
        aggregated['avg_pose_expansion'] = float(np.nanmean(pose_expansion)) if pose_expansion.size > 0 else 0.0

        # Энергия движения тела: используем head_motion_energy как прокси
        body_motion_energy = get_series('head_motion_energy')
        aggregated['body_motion_energy_mean'] = float(np.nanmean(body_motion_energy)) if body_motion_energy.size > 0 else 0.0
        aggregated['body_motion_energy_var'] = float(np.nanvar(body_motion_energy)) if body_motion_energy.size > 0 else 0.0


        speech_proxy = get_series('speech_activity_proxy')
        aggregated['speech_activity_ratio'] = float(np.nanmean(speech_proxy > 0.5)) if speech_proxy.size > 0 else 0.0

        # burstiness: насколько активность речи сконцентрирована
        if speech_proxy.size > 0:
            mean_s = float(np.nanmean(speech_proxy))
            if mean_s > 0:
                aggregated['speech_burstiness'] = float(np.nanvar(speech_proxy) / (mean_s ** 2 + 1e-8))
            else:
                aggregated['speech_burstiness'] = 0.0
        else:
            aggregated['speech_burstiness'] = 0.0

        aggregated['mouth_rhythm_score'] = float(np.nanstd(speech_proxy)) if speech_proxy.size > 0 else 0.0


        def temporal_contrast(sig):
            if sig.size == 0:
                return 0.0, 0.0, 0.0
            mean_val = float(np.nanmean(sig))
            max_val = float(np.nanmax(sig))
            contrast = max_val - mean_val
            n_local = sig.size
            if n_local >= 5:
                split = max(int(0.2 * n_local), 1)
                early = sig[:split]
                late = sig[-split:]
                early_mean = float(np.nanmean(early))
                late_mean = float(np.nanmean(late))
            else:
                early_mean = late_mean = mean_val
            return contrast, early_mean, late_mean

        engagement_contrast, early_e, late_e = temporal_contrast(engagement_signal)
        confidence_contrast, early_c, late_c = temporal_contrast(confidence_signal)
        stress_contrast, early_s, late_s = temporal_contrast(stress_proxy)

        aggregated['engagement_contrast'] = float(engagement_contrast)
        aggregated['confidence_contrast'] = float(confidence_contrast)
        aggregated['stress_contrast'] = float(stress_contrast)

        aggregated['early_late_ratios'] = {
            'engagement': float((late_e + 1e-6) / (early_e + 1e-6)),
            'speech_activity': float((late_s + 1e-6) / (early_s + 1e-6)),
            'gesture_rate': float(aggregated['gesture_rate_per_sec'])
        }


        num_hands_series = get_series('num_hands')
        hands_visibility_ratio = float(np.nanmean(num_hands_series > 0)) if num_hands_series.size > 0 else 0.0
        aggregated['hands_visibility_ratio'] = hands_visibility_ratio

        # face_visibility_ratio – по наличию head_position (ненулевой x_norm)
        head_x = get_series('head_position_x_norm')
        aggregated['face_visibility_ratio'] = float(np.nanmean(head_x > 0)) if head_x.size > 0 else 0.0

        aggregated['center_bias_mean'] = float(np.nanmean(np.abs(balance_offset))) if balance_offset.size > 0 else 0.0

        return aggregated

    def run(
        self,
        frames_dir: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if metadata is None:
            metadata = self.load_metadata(frames_dir)

        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")

        frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
        if not frame_indices:
            raise ValueError(f"{self.module_name} | Нет кадров для обработки")

        frame_manager = None
        try:
            frame_manager = self.create_frame_manager(frames_dir, metadata)
            self.logger.info(f"{self.module_name} | Начало обработки {len(frame_indices)} кадров")

            results = self.process(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                config=config or {}
            )

            times_s = self._last_times_s if self._last_times_s is not None else _require_union_times_s(frame_manager, frame_indices)
            landmarks_present = self._last_landmarks_present if self._last_landmarks_present is not None else np.zeros((len(frame_indices),), dtype=bool)

            packed = self._pack_npz_results(
                results=results,
                frame_indices=frame_indices,
                times_s=times_s,
                landmarks_present=landmarks_present,
            )

            status = "ok"
            empty_reason = None
            has_any_landmarks = bool(np.any(landmarks_present))
            core_meta = self._last_core_meta if isinstance(self._last_core_meta, dict) else {}
            core_status = str(core_meta.get("status") or "").lower()
            core_empty_reason = core_meta.get("empty_reason")
            if not has_any_landmarks or core_status == "empty":
                status = "empty"
                empty_reason = core_empty_reason or "no_faces_in_video"
            self._last_empty_reason = empty_reason

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
                "status": status,
                "empty_reason": empty_reason,
            }

            try:
                save_metadata["models_used"] = self.get_models_used(config=config or {}, metadata=metadata or {})
            except Exception:
                save_metadata["models_used"] = []

            save_metadata["ui_payload"] = self.build_ui_payload(packed)

            saved_path = self.save_results(
                results=packed,
                metadata=save_metadata,
                use_compressed=False
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
    
    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

