# Micro Emotion Module

Модуль извлечения микроэмоций и Action Units (AU) из видео с использованием **OpenFace** через Docker. Компонент извлекает оптимизированные признаки для анализа микроэмоций, детекции micro-expressions и генерации per-frame векторов для VisualTransformer.

## 📋 Содержание

- [Обзор](#обзор)
- [Архитектура](#архитектура)
- [Зависимости](#зависимости)
- [Установка](#установка)
- [Использование](#использование)
- [Структура выходных данных](#структура-выходных-данных)
- [Параметры конфигурации](#параметры-конфигурации)
- [Технические детали](#технические-детали)
- [Производительность](#производительность)
- [Оптимизация признаков](#оптимизация-признаков)

## Обзор

Модуль `micro_emotion` использует **OpenFace** (CMU) для анализа мимики лица и извлечения Action Units (AU). OpenFace запускается через Docker контейнер и обрабатывает кадры видео, извлекая:

- **Action Units (AU)**: 45 единиц действия лица (интенсивность и presence)
- **Facial Landmarks**: 68 ключевых точек лица (2D и 3D)
- **Head Pose**: Поза головы (поворот, наклон, приближение)
- **Gaze Direction**: Направление взгляда
- **Micro-expressions**: Быстрые эмоциональные вспышки (0.03-0.5 секунды)

Модуль извлекает два типа признаков:
1. **Aggregate Features** — агрегированные статистики по всему видео (для MLP/Tabular Head)
2. **Per-Frame Vectors** — компактные векторы для каждого кадра (для VisualTransformer)

**Версия (producer_version)**: 2.0.2  
**schema_version**: `micro_emotion_npz_v3`  
**Schemas**:
- Human: `DataProcessor/VisualProcessor/modules/micro_emotion/SCHEMA.md`
- Machine: `DataProcessor/VisualProcessor/schemas/micro_emotion_npz_v3.json`

### Audit v3 — Decisions (FINAL)

- **OpenFace runtime**: Docker+image = **hard dependency** (no-fallback). Нет docker/image/FeatureExtraction ⇒ `error`.
- **compact22**: **строго 22 dims**, стабильные имена (контракт для моделей).
- **Outputs**: сохраняем и **wide** `frame_features (N,F)` и **compact22 (N,22)`.
- **Events stream**: `event_*` остаётся **analytics** (QA/debug, не обязателен модели).

### Основные возможности

- ✅ Извлечение 45 Action Units с интенсивностью и presence
- ✅ Оптимизированная обработка: ключевые AU (10-14) + PCA для остальных
- ✅ Baseline subtraction для уменьшения межсубъектного сдвига
- ✅ Детекция micro-expressions (4 типа: smile, surprise, frown, disgust)
- ✅ Компактные геометрические признаки из landmarks (mouth_opening, smile_width, face_asymmetry)
- ✅ Per-frame векторы (~22 числа) для VisualTransformer
- ✅ Интеграция с BaseModule для единообразия с другими модулями
- ✅ Опциональная фильтрация кадров по `core_face_landmarks.face_present`
- ✅ Обработка через Docker (изоляция и воспроизводимость)

## Архитектура

### Используемые технологии

Модуль использует **OpenFace** (CMU) через Docker контейнер:
- **Docker Image**: `openface/openface:latest`
- **OpenFace**: Система анализа лица с открытым исходным кодом
- **Feature Extraction**: Извлечение AU, landmarks, pose, gaze из кадров

### Обработка данных

1. **Загрузка кадров**: Модуль загружает кадры через `FrameManager`
2. **OpenFace анализ**: Кадры обрабатываются через Docker контейнер OpenFace
3. **CSV результаты**: OpenFace сохраняет результаты в CSV формат
4. **Оптимизированная обработка**: `MicroEmotionProcessor` обрабатывает CSV/DataFrame:
   - Вычисление baseline для AU (из нейтральных кадров)
   - Извлечение ключевых AU с baseline subtraction
   - PCA для остальных AU (3-5 компонент)
   - Детекция micro-expressions
   - Вычисление компактных геометрических признаков
   - Генерация per-frame векторов
5. **Агрегация**: Вычисление метрик по всему видео
6. **Сохранение**: Результаты сохраняются через `ResultsStore` в формате `.npz`

### Компоненты модуля

- **`main.py`**: CLI интерфейс и основная логика обработки
- **`micro_emotion_processor.py`**: Оптимизированный процессор для обработки данных OpenFace
  - `MicroEmotionProcessor`: Класс для обработки DataFrame OpenFace
  - `MicroEmotionModule`: Интеграция с BaseModule
- **`OpenFaceAnalyzer`**: Класс для работы с Docker контейнером OpenFace (загружается динамически)

## Зависимости

### Внешние зависимости

Модуль требует:
- **Docker**: Установленный и запущенный Docker daemon
- **OpenFace Image**: Загруженный образ `openface/openface:latest`
  ```bash
  docker pull openface/openface:latest
  ```

Опциональные зависимости:
- **`core_face_landmarks`**: Для фильтрации кадров по `face_present` (если включён `--use-face-detection`)

### Python зависимости

```python
pandas>=1.3.0
numpy>=1.19.0
scipy>=1.7.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
```

### Системные требования

- **Docker**: Установленный Docker (версия 20.10+)
- **RAM**: Минимум 4GB, рекомендуется 8GB+ (для Docker контейнера)
- **CPU**: Рекомендуется многоядерный CPU для ускорения обработки
- **Диск**: ~2GB для Docker образа OpenFace

## Установка

### 1. Установка Docker

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Проверка
docker --version
```

### 2. Загрузка OpenFace образа

```bash
docker pull openface/openface:latest
```

### 3. Установка Python зависимостей

```bash
pip install pandas numpy scipy scikit-learn opencv-python
```

## Использование

### CLI интерфейс

```bash
python3 DataProcessor/VisualProcessor/modules/micro_emotion/main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/results \
    --feature-groups default \
    --openface-batch-size 64 \
    --docker-image openface/openface:latest \
    --fps 30 \
    --microexpr-smoothing-sigma 0.05 \
    --microexpr-delta-threshold 0.4 \
    --microexpr-max-duration-frames 15 \
    --microexpr-min-peak-distance-frames 6 \
    --gaze-centered-threshold 10.0 \
    --pca-components 3 \
    --au-confidence-threshold 0.5 \
    --device cuda \
    --progress-every-frames 50 \
    --log-level INFO
```

#### Параметры CLI

- `--frames-dir` (обязательный): Директория с кадрами (должна содержать `metadata.json`)
- `--rs-path` (обязательный): Путь к хранилищу результатов (ResultStore per-run)
- `--feature-groups` (по умолчанию: `default`): Набор фич (preset или CSV)
- `--openface-batch-size` (по умолчанию: `64`): Максимум кадров на один запуск OpenFace
- `--docker-image` (по умолчанию: `openface/openface:latest`): Docker образ для OpenFace
- `--fps` (по умолчанию: `30`): Кадров в секунду (для нормализации времени и окон micro-expressions)
- `--microexpr-smoothing-sigma` (по умолчанию: `0.05`): Сглаживание micro-expressions (в секундах)
- `--microexpr-delta-threshold` (по умолчанию: `0.4`): Порог изменения интенсивности для micro-expression
- `--microexpr-max-duration-frames` (по умолчанию: `15`): Максимальная длительность micro-expression в кадрах
- `--microexpr-min-peak-distance-frames` (по умолчанию: `6`): Минимальное расстояние между пиками micro-expression
- `--gaze-centered-threshold` (по умолчанию: `10.0`): Порог для определения взгляда в камеру (градусы)
- `--pca-components` (по умолчанию: `3`): Количество PCA компонент для AU
- `--au-confidence-threshold` (по умолчанию: `0.5`): Порог уверенности AU для флагов надёжности
- `--device` (по умолчанию: `cuda`): Устройство для обработки (cuda/cpu/auto). **Note**: Модуль требует `cuda` для OpenFace (GPU-only).
- `--progress-every-frames` (по умолчанию: `50`): Как часто писать прогресс в `state_events.jsonl`
- `--log-level` (по умолчанию: `INFO`): Уровень логирования (DEBUG/INFO/WARN/ERROR)

**Примечание**: CLI по умолчанию использует `use_face_detection=True`, что означает фильтрацию кадров по `core_face_landmarks.face_present` перед запуском OpenFace. Это оптимизирует обработку, запуская OpenFace только на кадрах с лицами. При программном использовании `MicroEmotionModule` значение по умолчанию — `False`.

### Программный интерфейс

#### Использование через main.py

```python
from modules.micro_emotion.main import run_pipeline

saved_path = run_pipeline(
    frames_dir="/path/to/frames",
    rs_path="/path/to/results",
    feature_groups="default",
    openface_batch_size=64,
    docker_image="openface/openface:latest",
    fps=30,
    microexpr_smoothing_sigma=0.05,
    microexpr_delta_threshold=0.4,
    microexpr_max_duration_frames=15,
    microexpr_min_peak_distance_frames=6,
    gaze_centered_threshold=10.0,
    pca_components=3,
    au_confidence_threshold=0.5,
    device="cuda",
    progress_every_frames=50,
)
```

#### Использование через MicroEmotionModule (BaseModule)

```python
from modules.micro_emotion.micro_emotion_processor import MicroEmotionModule
from utils.frame_manager import FrameManager

# Инициализация
module = MicroEmotionModule(rs_path="/path/to/results", device="cuda")

# Загрузка метаданных
metadata = module.load_metadata("/path/to/frames")
frame_indices = module.get_frame_indices(metadata)  # строго Segmenter-owned

# Создание FrameManager
frame_manager = module.create_frame_manager("/path/to/frames", metadata)

try:
    # Обработка
    saved_path = module.run(frames_dir="/path/to/frames", config={})
finally:
    frame_manager.close()
```

#### Использование MicroEmotionProcessor напрямую

```python
from modules.micro_emotion.micro_emotion_processor import MicroEmotionProcessor
import pandas as pd

# Загрузка DataFrame OpenFace
df = pd.read_csv("/path/to/openface.csv")

# Инициализация процессора
processor = MicroEmotionProcessor(fps=30)

# Обработка
processed = processor.process_openface_dataframe(df, fit_models=True)

# Результаты
features = processed['features']
per_frame_vectors = processed['per_frame_vectors']
reliability_flags = processed['reliability_flags']
microexpr_features = processed['microexpr_features']
```

## Структура выходных данных (NPZ source-of-truth)

Результаты сохраняются в формате `.npz` (compressed numpy) и содержат:

### Метаданные

NPZ meta содержит baseline identity keys +:
- `ui_payload`: JSON для фронта (внутри NPZ meta, не отдельный JSON-артефакт)

### Aggregate Features (для MLP/Tabular Head)

В `micro_emotion_npz_v3` агрегаты по видео хранятся таблично:

- `feature_names[V]` + `feature_values[V]` (float32, NaN если недоступно)

#### Ключевые Action Units (10-14 AU)

Для каждого ключевого AU (AU06, AU12, AU04, AU01, AU02, AU25, AU26, AU07, AU23, AU45, AU43, AU15, AU20, AU10):

- **`{au}_intensity_mean`** (`float`): Средняя интенсивность AU (0.0-5.0)
- **`{au}_intensity_std`** (`float`): Стандартное отклонение интенсивности
- **`{au}_intensity_delta_mean`** (`float`): Средняя интенсивность относительно baseline
- **`{au}_presence_rate`** (`float`): Доля кадров с presence==1 (0.0-1.0)
- **`{au}_peak_count`** (`int`): Количество пиков интенсивности

#### PCA для остальных AU

- **`au_pca_1`, `au_pca_2`, `au_pca_3`** (`float`): Первые 3 PCA компоненты
- **`au_pca_var_explained_1..k`** (`float`): Доля объяснённой дисперсии для каждой компоненты

#### Head Pose

- **`pose_Rx_mean`, `pose_Ry_mean`, `pose_Rz_mean`** (`float`): Средние значения поворотов (градусы)
- **`pose_Rx_std`, `pose_Ry_std`, `pose_Rz_std`** (`float`): Стандартные отклонения поворотов
- **`pose_Rx_min`, `pose_Rx_max`, `pose_Ry_min`, `pose_Ry_max`** (`float`): Экстремальные значения
- **`pose_Tz_mean`, `pose_Tz_std`** (`float`): Приближение/удаление от камеры
- **`pose_stability_score`** (`float`, 0.0-1.0): Оценка стабильности позы

#### Gaze Direction

- **`gaze_x_mean`, `gaze_y_mean`** (`float`): Средние углы взгляда (градусы)
- **`gaze_x_std`, `gaze_y_std`** (`float`): Стандартные отклонения углов
- **`gaze_centered_ratio`** (`float`, 0.0-1.0): Доля кадров с взглядом в камеру
- **`blink_rate_per_min`** (`float`): Частота миганий в минуту
- **`eye_contact_score`** (`float`, 0.0-1.0): Комбинированная оценка зрительного контакта

#### Facial Landmarks

- **`mouth_opening_mean`, `mouth_opening_std`** (`float`): Открытие рта (нормализованное)
- **`smile_width_mean`, `smile_width_std`** (`float`): Ширина улыбки
- **`face_asymmetry_score`** (`float`, 0.0-1.0): Оценка асимметрии лица
- **`landmarks_pca_1..5`** (`float`): Первые 5 PCA компонент для landmarks
- **`head_depth_variation`** (`float`): Вариация глубины головы

#### Micro-expressions

- **`microexpr_count`** (`int`): Количество обнаруженных micro-expressions
- **`microexpr_rate_per_min`** (`float`): Частота micro-expressions в минуту
- **`microexpr_max_intensity`** (`float`): Максимальная интенсивность
- **`microexpr_types_distribution`** (`Dict[str, int]`): Распределение по типам (smile, surprise, frown, disgust)
- **`microexpr_timestamps`** (`List[float]`): Временные метки micro-expressions (секунды)
- **`microexpr_types`** (`List[str]`): Типы для каждого micro-expression

#### Видео-уровневые агрегаты

- **`smile_ratio`** (`float`): Доля кадров с улыбкой
- **`eye_contact_ratio`** (`float`): Доля кадров с взглядом в камеру
- **`face_presence_ratio`** (`float`): Доля кадров с обнаруженным лицом
- **`avg_mouth_opening`** (`float`): Среднее открытие рта

#### Reliability Flags

- **`au_quality_overall`** (`float`, 0.0-1.0): Средняя уверенность AU
- **`au_quality_reliable`** (`bool`): Флаг надёжности AU данных
- **`landmark_visibility_mean`** (`float`, 0.0-1.0): Средняя доля видимых landmarks
- **`landmark_visibility_reliable`** (`bool`): Флаг надёжности landmarks
- **`occlusion_flag`** (`bool`): Флаг окклюзии лица
- **`lighting_flag`** (`bool`): Флаг качества освещения

### Per-Frame Features

Модуль сохраняет два типа per-frame признаков:

#### Wide Frame Features (`frame_features`, `frame_feature_names`)

Массив `frame_features` имеет форму `[N_frames, F]` (где F ~40-80) и содержит для каждого кадра:
- `time_norm`: Нормализованное время кадра (0.0-1.0)
- `face_present_any`: Флаг наличия лица (0.0/1.0)
- `{AU}_delta`: Интенсивность ключевых AU относительно baseline (AU12, AU06, AU04, AU25, и др.)
- `pose_Rx`, `pose_Ry`, `pose_Rz`, `pose_Tz`: Поза головы
- `gaze_angle_x`, `gaze_angle_y`: Углы взгляда

Значения `NaN` используются для кадров без лиц.

#### Compact22 Features (`compact22`, `compact22_feature_names`)

Массив `compact22` имеет форму `[N_frames, 22]` и содержит оптимизированные признаки для VisualTransformer:

1. **time_norm** (1): Нормализованное время кадра (0.0-1.0)
2. **face_presence_flag** (1): Флаг наличия лица (0.0/1.0)
3. **au12_intensity_delta** (1): AU12 интенсивность относительно baseline
4. **au6_intensity_delta** (1): AU06 интенсивность относительно baseline
5. **au4_intensity_delta** (1): AU04 интенсивность относительно baseline
6. **au25_intensity_delta** (1): AU25 интенсивность относительно baseline
7. **au25_presence_rate_short** (1): AU25 presence в коротком окне (±0.5s)
8. **blink_flag** (1): Флаг мигания (AU45 presence)
9. **pose_Ry_norm** (1): Нормализованный горизонтальный поворот
10. **pose_Rx_norm** (1): Нормализованный наклон/кивок
11. **gaze_centered_flag** (1): Флаг взгляда в камеру
12. **gaze_x** (1): Нормализованный горизонтальный угол взгляда
13. **gaze_y** (1): Нормализованный вертикальный угол взгляда
14. **mouth_opening_norm** (1): Нормализованное открытие рта
15. **face_asymmetry_score** (1): Оценка асимметрии лица
16. **microexpr_recent_count** (1): Число микровспышек в последние 1-2s
17. **au_pca_1** (1): Первая PCA компонента для остальных AU
18. **au_pca_2** (1): Вторая PCA компонента
19. **au_pca_3** (1): Третья PCA компонента
20. **au_quality_flag** (1): Флаг качества AU

### Пример структуры

```python
result = {
    "frame_indices": np.array([0, 5, 10, 15, ...], dtype=np.int32),  # Union-domain frame indices
    "times_s": np.array([0.0, 0.167, 0.333, ...], dtype=np.float32),  # Time axis
    "face_present_any": np.array([True, True, False, ...], dtype=bool),  # Face presence per frame
    "frame_feature_names": np.array(["time_norm", "face_present_any", "AU12_delta", ...], dtype=object),
    "frame_features": np.array([
        [0.0, 1.0, 0.2, 0.1, 0.0, 0.3, 15.5, -2.3, 0.05, 0.02, ...],  # Wide features [N, F]
        [0.01, 1.0, 0.25, 0.12, 0.0, 0.32, 16.2, -2.1, 0.06, 0.03, ...],
        [0.02, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ...],  # No face
        ...
    ], dtype=np.float32),  # [N_frames, F] where F ~40-80
    "compact22": np.array([
        [0.0, 1.0, 0.2, 0.1, 0.0, 0.3, 0.5, 0.0, 0.1, 0.0, 1.0, 0.05, 0.02, 0.1, 0.0, 0.0, 0.1, 0.2, 0.0, 1.0, 0.0, 1.0],
        [0.01, 1.0, 0.25, 0.12, 0.0, 0.32, 0.52, 0.0, 0.11, 0.0, 1.0, 0.06, 0.03, 0.11, 0.0, 0.0, 0.12, 0.21, 0.01, 1.0, 0.0, 1.0],
        [0.02, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ...
    ], dtype=np.float32),  # [N_frames, 22] - compact features for VisualTransformer
    "compact22_feature_names": np.array(["c22_0", "c22_1", ...], dtype=object),
    "event_times_s": np.array([1.2, 2.5, 3.8, ...], dtype=np.float32),  # Micro-expression timestamps
    "event_type_id": np.array([1, 1, 2, ...], dtype=np.int16),  # 1=smile, 2=surprise, 3=frown, 4=disgust
    "event_strength": np.array([0.85, 0.72, 0.91, ...], dtype=np.float32),
    "feature_names": np.array(["pose_stability_score", "gaze_centered_ratio", "blink_rate_per_min", ...], dtype=object),
    "feature_values": np.array([0.85, 0.72, 18.5, ...], dtype=np.float32),
    "microexpr_features": {
        "microexpr_count": 12,
        "microexpr_rate_per_min": 2.4,
        "microexpr_max_intensity": 0.85,
        "microexpr_types_distribution": {"smile": 8, "surprise": 4},
        "microexpr_timestamps": [1.2, 2.5, 3.8, ...],
        "microexpr_types": ["smile", "surprise", "smile", ...]
    },
    "summary": {
        "total_frames": 900,
        "frames_with_face": 820,
        "frames_processed_openface": 820,
        "success": True,
        "fps": 30,
        "stage_timings_ms": {
            "deps_load_ms": 45.2,
            "openface_run_ms": 1234.5,
            "micro_emotion_features_ms": 12.3,
            "process_ms": 1291.8
        }
    }
}
```

Подробное описание всех фичей см. в [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md).

## Параметры конфигурации

### Основные параметры MicroEmotionProcessor

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `fps` | `int` | `30` | Кадров в секунду |
| `microexpr_smoothing_sigma` | `float` | `0.05` | Сглаживание для micro-expressions (в секундах) |
| `microexpr_delta_threshold` | `float` | `0.4` | Порог изменения интенсивности для micro-expression |
| `microexpr_max_duration_frames` | `int` | `15` | Максимальная длительность micro-expression в кадрах (0.5s при 30fps) |
| `microexpr_min_peak_distance_frames` | `int` | `6` | Минимальное расстояние между пиками (0.2s при 30fps) |
| `gaze_centered_threshold` | `float` | `10.0` | Порог для определения взгляда в камеру (градусы) |
| `pca_components` | `int` | `3` | Количество PCA компонент для AU |
| `au_confidence_threshold` | `float` | `0.5` | Порог уверенности AU |

### Параметры CLI

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--openface-batch-size` | `int` | `64` | Размер батча для обработки кадров OpenFace (вызов Docker) |
| `--feature-groups` | `str` | `default` | Группы фич (`default`, `all`, и т.п.) |
| `--docker-image` | `str` | `openface/openface:latest` | Docker образ для OpenFace |
| `--fps` | `int` | `30` | Кадров в секунду |
| `--microexpr-smoothing-sigma` | `float` | `0.05` | Сглаживание для micro-expressions (в секундах) |
| `--microexpr-delta-threshold` | `float` | `0.4` | Порог изменения интенсивности для micro-expression |
| `--microexpr-max-duration-frames` | `int` | `15` | Максимальная длительность micro-expression в кадрах |
| `--microexpr-min-peak-distance-frames` | `int` | `6` | Минимальное расстояние между пиками (в кадрах) |
| `--gaze-centered-threshold` | `float` | `10.0` | Порог для определения взгляда в камеру (градусы) |
| `--pca-components` | `int` | `3` | Количество PCA компонент для AU |
| `--au-confidence-threshold` | `float` | `0.5` | Порог уверенности AU для флагов надёжности |
| `--device` | `str` | `cuda` | Устройство обработки (cuda/cpu/auto) |
| `--progress-every-frames` | `int` | `50` | Шаг прогресса по кадрам для state_events |

### Рекомендации по настройке

- **`batch_size`**:
  - Меньшие батчи (20-30): для ограниченной памяти
  - Стандартные батчи (50): баланс скорости и памяти
  - Большие батчи (100+): для быстрой обработки (требует больше памяти)

- **`microexpr_smoothing_sigma`**:
  - 0.03-0.05s: для быстрых micro-expressions
  - 0.05-0.1s: стандартное значение
  - >0.1s: для более плавных сигналов

- **`pca_components`**:
  - 3 компоненты: компактное представление (рекомендуется)
  - 5 компонент: более детальное представление (больше размерность)

## Технические детали

### OpenFace обработка

1. **Запуск Docker контейнера**: OpenFace запускается в изолированном контейнере
2. **Обработка кадров**: Кадры передаются в контейнер батчами
3. **CSV результаты**: OpenFace сохраняет результаты в CSV формат с колонками:
   - `frame`: индекс кадра
   - `success`: флаг успешного обнаружения лица
   - `AU{01..45}_r`: интенсивность AU (0.0-5.0)
   - `AU{01..45}_c`: presence/confidence AU (0.0-1.0)
   - `x_{0..67}`, `y_{0..67}`: 2D координаты landmarks
   - `X_{0..67}`, `Y_{0..67}`, `Z_{0..67}`: 3D координаты landmarks
   - `pose_Rx`, `pose_Ry`, `pose_Rz`: повороты головы (градусы)
   - `pose_Tx`, `pose_Ty`, `pose_Tz`: трансляции головы
   - `gaze_angle_x`, `gaze_angle_y`: углы взгляда (градусы)

### Оптимизированная обработка

1. **Baseline Subtraction**:
   - Вычисляется средняя интенсивность AU для нижних 20% кадров по общей активности (нейтральные кадры)
   - Хранится `intensity_delta = intensity - baseline` для уменьшения межсубъектного сдвига

2. **PCA для AU**:
   - Применяется PCA к интенсивностям всех AU, кроме ключевых
   - Сохраняются первые 3-5 компонент, объясняющих 90%+ дисперсии

3. **Micro-expressions Detection**:
   - Сглаживание AU интенсивности гауссом (σ = 0.03-0.1s)
   - Поиск пиков с порогом > baseline + 1.5*std
   - Фильтрация по длительности (≤ 0.5s) и расстоянию между пиками (≥ 0.2s)
   - Комбинации AU для определения типа выражения

4. **Gaze Centered Detection**:
   - Взгляд считается направленным в камеру, если `|gaze_x| < 10°` и `|gaze_y| < 10°`

5. **Blink Detection**:
   - Мигание детектируется как AU45 presence с длительностью < 0.25s

6. **Landmark PCA**:
   - Применяется PCA к координатам всех 68 landmarks (2D), сохраняются первые 5 компонент

7. **Geometric Features**:
   - Вычисляются компактные геометрические признаки (mouth_opening, smile_width, face_asymmetry) вместо хранения всех координат

### Обработка ошибок

- **Отсутствие Docker**: Выбрасывает `RuntimeError` с описанием проблемы
- **Отсутствие OpenFace образа**: Выбрасывает `RuntimeError` с инструкцией по загрузке
- **Ошибки OpenFace**: Логируются с предупреждением, обработка продолжается
- **Отсутствие DataFrame**: Возвращает пустой результат с предупреждением
- **Кадры без лиц**: Пропускаются с логированием

## Производительность

### Бенчмарки (приблизительные)

На CPU (Intel i7-9700K, Docker):
- **Скорость**: ~10-30 FPS (зависит от разрешения кадров и сложности сцены)
- **Память**: ~2-4GB (включая Docker контейнер)
- **Время обработки**: ~30-60 секунд для 1 минуты видео (30fps)

На GPU (через Docker с GPU support):
- **Скорость**: ~30-50 FPS
- **Память**: ~3-5GB
- **Время обработки**: ~20-40 секунд для 1 минуты видео

### Оптимизация

1. **Используйте батчи**: Увеличьте `batch_size` до предела памяти
2. **Фильтрация кадров**: Используйте `--use-face-detection` для пропуска кадров без лиц
3. **Оптимизированная обработка**: Используйте `MicroEmotionProcessor` вместо прямого использования CSV
4. **Кэширование**: Результаты OpenFace можно кэшировать для повторного использования

### Ограничения

- Модуль требует Docker и загруженный образ OpenFace
- Обработка через Docker может быть медленнее, чем нативная установка
- Требует значительных вычислительных ресурсов
- Обработка очень длинных видео может быть медленной (рассмотрите обработку по частям)

## Оптимизация признаков

Модуль использует несколько техник оптимизации для уменьшения размерности признаков:

### 1. Ключевые AU вместо всех 45

Вместо хранения всех 45 AU в непроцессированном виде, выделен ключевой поднабор (10-14 AU), которые наиболее информативны для UGC/вовлечённости:
- **AU06, AU12**: Улыбка/счастье
- **AU04, AU01, AU02**: Удивление/грусть/фокус
- **AU25, AU26**: Говорение/удивление
- **AU07, AU23**: Напряжение/негатив
- **AU45, AU43**: Мигание/сонливость
- **AU15**: Печаль/негатив
- **AU20, AU10**: Другие эмоции

### 2. Baseline Subtraction

Вычисляется средняя интенсивность AU для данного лица в «нейтральных» кадрах (нижние 20% по общей активности) и хранится `au_intensity_delta = intensity - baseline`. Это существенно уменьшает межсубъектный сдвиг.

### 3. PCA для остальных AU

AU, не входящие в ключевой набор, проецируются через PCA (3-5 компонент), что даёт компактное представление без потери важной информации.

### 4. Компактные геометрические признаки

Вместо хранения всех 68 landmarks в явном виде, вычисляются компактные геометрические признаки:
- `mouth_opening`: Расстояние между верхней и нижней губой
- `smile_width`: Расстояние между уголками губ
- `face_asymmetry_score`: Оценка асимметрии лица
- `landmarks_pca_1..5`: PCA проекции для всех landmarks

### 5. Per-Frame Vectors

Компактный per-frame вектор (~22 числа) для VisualTransformer вместо хранения всех признаков OpenFace (~1000+ чисел на кадр).

## Примеры использования

### Базовый пример

```bash
python main.py \
    --frames-dir ./frames \
    --rs-path ./results \
    --batch-size 50
```

### С фильтрацией по face presence

```bash
python main.py \
    --frames-dir ./frames \
    --rs-path ./results \
    --use-face-detection \
    --batch-size 50
```

### Программный интерфейс

```python
from modules.micro_emotion.micro_emotion_processor import MicroEmotionProcessor
import pandas as pd

# Загрузка CSV OpenFace
df = pd.read_csv("openface_results.csv")

# Инициализация процессора
processor = MicroEmotionProcessor(
    fps=30,
    microexpr_smoothing_sigma=0.05,
    pca_components=3
)

# Обработка
processed = processor.process_openface_dataframe(df, fit_models=True)

# Использование результатов
features = processed['features']
per_frame_vectors = processed['per_frame_vectors']  # Внутренний формат процессора
microexpr_features = processed['microexpr_features']

print(f"Micro-expressions: {microexpr_features['microexpr_count']}")
print(f"Per-frame vectors shape: {per_frame_vectors.shape}")

# Примечание: В NPZ сохраняются frame_features (wide) и compact22 (для VisualTransformer)
# вместо per_frame_vectors. См. раздел "Структура выходных данных" для деталей.
```

## Логирование

Модуль использует стандартный logger из `utils.logger`:

```python
from utils.logger import get_logger

logger = get_logger("micro_emotion")
```

Уровни логирования:
- **DEBUG**: Детальная информация о каждом батче и обработке
- **INFO**: Общая информация о процессе (каждые N кадров)
- **WARN**: Предупреждения (пропущенные кадры, ошибки OpenFace)
- **ERROR**: Критические ошибки

## Интеграция с пайплайном

Модуль интегрирован с VisualProcessor через `BaseModule`:

- Автоматическая загрузка зависимостей через `load_core_provider()` (`core_face_landmarks`)
- Стандартизированное сохранение результатов через `save_results()` (NPZ source-of-truth)
- Единый интерфейс `process()` для всех модулей
- Запуск OpenFace (Docker, GPU-only) только на кадрах с лицом (`core_face_landmarks.face_present`)

### Порядок выполнения в пайплайне

1. **core_face_landmarks** (обязательная зависимость) → определение кадров с лицами + face landmarks (для UI отдельно)
2. **micro_emotion** → анализ мимики через OpenFace
3. **Другие модули** → могут использовать результаты micro_emotion

## Дополнительная документация

- [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md) — подробное описание всех извлекаемых фичей
- [BaseModule documentation](../../base_module.py) — базовый класс для модулей
- [OpenFace documentation](https://github.com/TadasBaltrusaitis/OpenFace/wiki) — официальная документация OpenFace

## Примечания

### Docker изоляция

Модуль использует Docker для изоляции OpenFace, что обеспечивает:
- Воспроизводимость результатов
- Изоляцию зависимостей
- Простоту развёртывания

### Оптимизированная обработка

Модуль использует оптимизированный `MicroEmotionProcessor` для обработки данных OpenFace:
- Baseline subtraction для уменьшения межсубъектного сдвига
- PCA для компактного представления AU
- Детекция micro-expressions для анализа искренности
- Компактные геометрические признаки вместо полных landmarks

### Per-Frame Vectors

Per-frame векторы оптимизированы для VisualTransformer:
- Компактное представление (~22 числа вместо 1000+)
- Все значения нормализованы
- Включают ключевые признаки для временного моделирования

## Лицензия

Модуль использует OpenFace (CMU), который распространяется под лицензией MIT. См. [OpenFace-license.txt](./OpenFace/OpenFace-license.txt) для деталей.

## Авторы

Компонент разработан как часть проекта TrendFlowML для анализа микроэмоций и мимики в видео контенте.
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
