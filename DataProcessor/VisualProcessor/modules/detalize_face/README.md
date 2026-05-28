# Detalize Face

Модульная система для детального извлечения признаков лица из видео. Компонент использует результаты `core_face_landmarks` (MediaPipe FaceMesh) и вычисляет производные фичи лица (геометрия, поза, качество, глаза, движение, структура, чтение по губам).

## 📋 Содержание

- [Обзор](#обзор)
- [Архитектура](#архитектура)
- [Зависимости](#зависимости)
- [Установка](#установка)
- [Использование](#использование)
- [Модули](#модули)
- [Структура выходных данных](#структура-выходных-данных)
- [Sampling requirements](#sampling-requirements)
- [Models](#models)
- [Parallelization](#parallelization)
- [Performance characteristics](#performance-characteristics)
- [Quality validation & human-friendly inspection](#quality-validation--human-friendly-inspection)
- [Параметры конфигурации](#параметры-конфигурации)
- [Технические детали](#технические-детали)
- [Производительность](#производительность)
- [Этика и Privacy](#этика-и-privacy)

## Обзор

Модуль `detalize_face` предоставляет комплексную систему для анализа лиц в видео. В отличие от базового `core_face_landmarks`, который только извлекает координаты ключевых точек, этот модуль вычисляет производные метрики и признаки для различных аспектов лица.

**Версия (producer_version)**: 2.0.2  
**schema_version**: `detalize_face_npz_v3`  
**Schemas**:
- Human: `DataProcessor/VisualProcessor/modules/detalize_face/SCHEMA.md`
- Machine: `DataProcessor/VisualProcessor/schemas/detalize_face_npz_v3.json`

### Основные возможности

- ✅ **Модульная архитектура**: 7 специализированных модулей для разных типов фичей
- ✅ **Интеграция с core провайдерами**: Использует результаты `core_face_landmarks` (обязательная зависимость)
- ✅ **Мульти-лицо**: Поддержка до 10 лиц на кадр с трекингом через IoU
- ✅ **Временной анализ**: История кадров для вычисления динамических метрик (моргание, движение, стабильность)
- ✅ **Компактные фичи**: Функция извлечения компактного набора фичей (~40 dims) для VisualTransformer
- ✅ **Визуализация**: Опциональная визуализация результатов с landmarks и bbox
- ✅ **Интеграция с BaseModule**: Единый интерфейс для пайплайна VisualProcessor
- ✅ **Фильтрация качества**: Валидация лиц по размеру, соотношению сторон, уверенности детекции
- ✅ **Batch processing**: Поддержка обработки нескольких видео одновременно
- ✅ **Render system**: Генерация human-friendly визуализаций (JSON и HTML) для debugging и UI интеграции

### Типы извлекаемых фичей

1. **Геометрия** - размеры, позиция, форма лица
2. **Поза** - ориентация головы (yaw, pitch, roll)
3. **Качество** - резкость, шум, экспозиция, окклюзия
4. **Глаза** - открытие, моргание, направление взгляда
5. **Движение** - скорость, ускорение, микро-выражения
6. **Структура** - mesh векторы, идентичность (hash), выражение
7. **Чтение по губам** - параметры рта, речевая активность

## Архитектура

### Компоненты

```
detalize_face/
├── main.py                          # CLI интерфейс
├── detalize_face_refactored.py     # Основной класс и DetalizeFaceModule
├── _modules/                        # Модули извлечения фичей
│   ├── base_module.py              # Базовый интерфейс FaceModule
│   ├── geometry_module.py          # Геометрические фичи
│   ├── pose_module.py              # Поза головы
│   ├── quality_module.py           # Качество изображения
│   ├── eyes_module.py              # Глаза
│   ├── motion_module.py            # Движение
│   ├── structure_module.py         # Структура
│   ├── lip_reading_module.py       # Чтение по губам
│   └── ...                         # (удалены чувствительные/шумные модули)
└── _utils/                         # Утилиты
    ├── landmarks_utils.py          # Работа с landmarks
    ├── bbox_utils.py                 # Работа с bbox
    ├── compression_utils.py           # Сжатие векторов
    ├── compact_features.py            # Компактные фичи
    └── face_helpers.py                # Вспомогательные функции
```

### Поток обработки

1. **Загрузка зависимостей**: Загружаются результаты `core_face_landmarks` (`landmarks.npz`)
2. **Инициализация модулей**: Загружаются и инициализируются выбранные модули из `MODULE_REGISTRY`
3. **Обработка кадров**: Для каждого кадра с лицом:
   - Восстановление координат landmarks из core данных
   - Валидация лиц (размер, соотношение сторон, уверенность)
   - Назначение tracking_id через IoU
   - Извлечение ROI (области лица)
   - Обработка через каждый модуль
4. **Агрегация**: Результаты собираются в единый формат для сохранения
5. **Сохранение**: Результаты сохраняются в фиксированный NPZ (`detalize_face.npz`), а UI-данные — в `meta.ui_payload` (без отдельных JSON артефактов в result_store)

### Модульная система

Каждый модуль наследуется от `FaceModule` и реализует:

- `required_inputs()` - список необходимых входных данных
- `process(data)` - обработка данных и возврат результатов
- `initialize()` - инициализация (загрузка моделей, если нужно)

Модули могут зависеть друг от друга: результаты одного модуля доступны другим через `shared_data`.

## Зависимости

### Обязательные зависимости

Модуль **требует** результаты модуля `core_face_landmarks` (hard dependency, no-fallback):

- **Каноничный файл**: `result_store/.../core_face_landmarks/landmarks.npz`
- **Поведение**: Если `core_face_landmarks` отсутствует или не содержит кадров с лицами, модуль вернёт пустой результат (`status="empty"`, `empty_reason="no_faces_in_video"`).
- Примечание: baseline не требует отдельных JSON артефактов. UI рендер должен опираться на `meta.ui_payload` в NPZ.

NPZ содержит:
- `frame_indices (N,) int32` — индексы кадров с лицами
- `times_s (N,) float32` — временные метки кадров
- `face_landmarks (N, FACES, 468, 3) float32` (нормализованные координаты [0..1])
- `face_present (N, FACES) bool` — флаги наличия лиц

**Важно**: Компонент обрабатывает **только** кадры, где `core_face_landmarks` обнаружил лица. Все остальные кадры игнорируются.

### Python зависимости

```python
numpy>=1.19.0
opencv-python>=4.5.0
scikit-learn>=0.24.0  # Для PCA и других методов
```

### Системные требования

- **RAM**: Минимум 2GB, рекомендуется 4GB+ (зависит от количества модулей)
- **CPU**: Достаточно для обработки landmarks (не требует GPU)
- **Диск**: Минимальные требования (только для сохранения результатов)

## Установка

Модуль является частью VisualProcessor и не требует отдельной установки. Убедитесь, что все зависимости установлены:

```bash
pip install numpy opencv-python scikit-learn
```

## Использование

### CLI интерфейс

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/results \
    --modules geometry,pose,quality,eyes \
    --max-faces 4 \
    --min-detection-confidence 0.7 \
    --visualize \
    --visualize-dir ./face_visualizations
```

#### Основные параметры CLI

- `--frames-dir` (обязательный): Директория с кадрами (должна содержать `metadata.json`)
- `--rs-path` (обязательный): Путь к хранилищу результатов (ResultsStore)
- `--modules` (опционально): Список модулей через запятую (по умолчанию - все модули)
- `--max-faces` (по умолчанию: 4 в CLI, 10 в классе): Максимальное количество лиц на кадр
- `--refine-landmarks` (по умолчанию: включено): Использовать уточненные landmarks (468 точек), если доступны
- `--min-detection-confidence` (по умолчанию: 0.7): Минимальная уверенность детекции лица
- `--min-tracking-confidence` (по умолчанию: 0.7): Минимальная уверенность трекинга лица
- `--visualize`: Включить визуализацию результатов
- `--visualize-dir` (по умолчанию: `./face_visualizations`): Директория для сохранения визуализаций
- `--show-landmarks`: Показывать landmarks на визуализации
- `--log-level` (по умолчанию: INFO): Уровень логирования

#### Параметры фильтрации качества

- `--min-face-size` (по умолчанию: 30): Минимальный размер лица в пикселях
- `--max-face-size-ratio` (по умолчанию: 0.8): Максимальное отношение размера лица к кадру
- `--min-aspect-ratio` (по умолчанию: 0.6): Минимальное соотношение сторон лица
- `--max-aspect-ratio` (по умолчанию: 1.4): Максимальное соотношение сторон лица
- `--no-validate-landmarks`: Отключить валидацию landmarks

### Программный интерфейс

#### Через DetalizeFaceModule (BaseModule)

```python
from modules.detalize_face.detalize_face_refactored import (
    DetalizeFaceModule
)
# Инициализация
module = DetalizeFaceModule(
    rs_path="/path/to/results",
    modules=["geometry", "pose", "quality"],
    max_faces=4
)

# Полный запуск
module.run(frames_dir="/path/to/frames", config={})

# NPZ сохраняется в result_store (UI — через meta.ui_payload)
```

## Модули

### Доступные модули

Модуль можно указать через параметр `--modules` или в коде:

1. **`geometry`** - Геометрические фичи (размер, позиция, форма)
2. **`pose`** - Поза головы (yaw, pitch, roll)
3. **`quality`** - Качество изображения (резкость, шум, экспозиция)
4. **`eyes`** - Глаза (открытие, моргание, взгляд)
5. **`motion`** - Движение (скорость, ускорение, микро-выражения)
6. **`structure`** - Структура (mesh векторы, идентичность hash, выражение)
7. **`lip_reading`** - Чтение по губам (параметры рта, речевая активность)

### Выбор модулей

По умолчанию загружаются **все модули**. Для оптимизации производительности можно выбрать только нужные:

```bash
# Только базовые фичи
--modules geometry,pose,quality

# Для анализа внимания/речи
--modules geometry,pose,eyes,lip_reading
```

## Структура выходных данных

Результаты сохраняются в фиксированный `.npz` (per-run):

- `result_store/.../detalize_face/detalize_face.npz`

### Формат результатов

```python
{
    "summary": {
        "axis_frames": int,
        "frames_with_faces_total": int,
        "frames_with_faces_processed": int,
        "processed_frames": int,
        "total_faces": int,
        "primary_faces": int,
        "avg_faces_per_processed_face_frame": float,
        "stage_timings_ms": dict,
    },
    "frame_indices": np.ndarray,
    "times_s": np.ndarray,
    # model-facing time-series (aligned to frame_indices)
    "face_present": np.ndarray,   # bool (N,)
    "processed_mask": np.ndarray, # bool (N,)
    "primary_valid": np.ndarray,  # bool (N,)
    "face_count": np.ndarray,
    "primary_tracking_id": np.ndarray,      # int32 (N,), -1 if missing
    "primary_compact_features": np.ndarray, # float32 (N,40), 0 if missing
    # optional heuristic curves (only if write_primary_curves=True)
    "primary_gaze_at_camera_prob": np.ndarray,
    "primary_blink_rate": np.ndarray,
    "primary_attention_score": np.ndarray,
    "primary_quality_proxy_score": np.ndarray,
    "primary_face_sharpness": np.ndarray,
    "primary_occlusion_proxy": np.ndarray,
    "primary_speech_activity_prob": np.ndarray,
    # per-track aggregates (dict)
    "faces_agg": dict,
}
```

Примечание: baseline не хранит “сырые per-frame dict’ы” в NPZ; для UI используется `meta.ui_payload` (pointers на NPZ keys).

### Структура face_feature

Каждое лицо представлено словарем с ключами:

#### Базовые метаданные

- `frame_index` (`int`): Индекс кадра
- `face_index` (`int`): Индекс лица в кадре
- `bbox` (`list[float]`): [x_min, y_min, x_max, y_max]
- `detection_confidence` (`float`): Уверенность детекции (0.0-1.0)
- `tracking_id` (`int`): ID трека для временного анализа
- `is_primary_face` (`bool`): Является ли лицо primary (наибольшее в кадре)

#### Фичи от модулей

Каждый модуль добавляет свои ключи в `face_feature`. Например:

**Geometry Module:**
- `face_bbox_area`, `face_relative_size`, `face_box_ratio`
- `face_center_x_norm`, `face_center_y_norm`
- `face_rotation_in_frame`, `aspect_ratio_stability`
- `jaw_width`, `cheekbone_width`, `forehead_height`
- `face_shape_vector` (16 dims)

**Pose Module:**
- `yaw`, `pitch`, `roll` (в градусах)
- `yaw_norm`, `pitch_norm`, `roll_norm` (нормализованные)
- `head_pose_variability`, `pose_stability_score`
- `head_turn_frequency`, `attention_to_camera_ratio`
- `looking_direction_vector` (3D unit vector)

**Quality Module:**
- `face_sharpness`, `face_noise_level`, `face_exposure_score`
- `occlusion_proxy`, `quality_proxy_score`

**Eyes Module:**
- `eye_opening_ratio`, `eye_opening_left`, `eye_opening_right`
- `blink_rate`, `blink_intensity`, `blink_flag`
- `gaze_vector`, `gaze_at_camera_prob`
- `attention_score`, `iris_position`

И многие другие...

### Компактные фичи

Для использования в VisualTransformer доступна функция извлечения компактного набора фичей (~40 dims):

```python
from _utils.compact_features import extract_compact_features

compact = extract_compact_features(face_feature)
# Возвращает numpy array shape (~40,)
```

Подробное описание всех фичей см. в [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md).

## Sampling requirements

- **Axis**: выход выровнен по `metadata[detalize_face].frame_indices` (Segmenter contract; union-domain).
  - Fallback (legacy): если ключа нет, используем `metadata[core_face_landmarks].frame_indices` как axis (с warning), чтобы сохранить выравнивание с sampling group лица.
- **Compute gating**: фичи реально считаются только для кадров, где `core_face_landmarks` обнаружил лица (`face_present=true`).
- **Internal sampling (опционально)**: если face-кадров слишком много, можно ограничить вычисления uniform‑выборкой среди face-кадров:
  - `max_face_frames=<K>` (в конструкторе/конфиге)
  - `face_frames_sampling="uniform"`
  - тогда `processed_mask` будет `false` на части face-кадров, а `primary_*` (если включены) будут `NaN`.
- Источник истины по времени: `union_timestamps_sec` из `metadata.json` (no-fallback).
- Если лиц нет для всех axis кадров → `status="empty"`, `empty_reason="no_faces_in_video"`.
- В `meta.module_sampling_policy_version` фиксируется: `segmenter_axis_v1`, а политика внутренней выборки — в `meta.face_frames_sampling_policy_version`.

## Models

Компонент **не запускает собственных ML‑моделей**. Он использует результаты `core_face_landmarks` (MediaPipe FaceMesh), которые уже содержат вычисленные landmarks.

- **`models_used`**: пустой список (компонент не запускает модели напрямую)
- **Зависимость от моделей**: косвенная через `core_face_landmarks` (MediaPipe FaceMesh)
- **Вычисления**: все модули выполняют геометрические вычисления на landmarks без использования нейросетей

## Quality Status & Refinements (Audit v3)

Ключевые изменения/нововведения, зафиксированные в ходе аудита:

- **v2 → v3 schema bump** (`detalize_face_npz_v3`, `producer_version=2.0.2`):
  - output строго выровнен по Segmenter axis (`metadata[detalize_face].frame_indices`), с fallback на `metadata[core_face_landmarks].frame_indices` для legacy метаданных;
  - добавлены model-facing маски `face_present`, `processed_mask`, `primary_valid` для корректного обучения/инференса;
  - добавлены model-facing `primary_compact_features (N,40)` и `primary_tracking_id`;
  - добавлен model-facing `aggregated` (tabular/baseline head friendly статистики по compact-векторам).
- **Heuristic policy**:
  - `primary_*` кривые (gaze/blink/attention/quality/…) считаются **эвристиками** и выключены по умолчанию (`write_primary_curves=false`).
- **Internal sampling среди face-кадров**:
  - допускается cap `max_face_frames` с uniform выборкой среди face-кадров; `processed_mask` отражает, где реально были вычисления.
- **Offline render**:
  - HTML рендер работает offline (без CDN) и показывает маски + sanity-check по compact-вектору.

## Parallelization

- **Внутренний параллелизм**: нет. Обработка идёт последовательно по кадрам.
- **Внешний параллелизм**: допускается запуск нескольких экземпляров на разных `run_id`.
- **Thread‑safety**: компонент безопасен при параллельном запуске на разных видео (разные `result_store` пути).

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео:

- **Batch-safe**: использует per-video rs_path (нет shared mutable state между видео).
- **Дефолтный process_batch()**: последовательная обработка каждого видео через BaseModule.
- **GPU batching**: не требуется (CPU-only модуль, работает только с landmarks).

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): без изменений (компонент работает через subprocess)
- Для single video: без изменений

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными.

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/detalize_face_costs_v1.json` (TBD)

**Единица обработки**: `frame_with_face`

**Типичные значения** (preset="default", все 7 модулей, CPU):
- **Latency per frame**: ~2-3 ms (зависит от количества модулей и лиц)
- **CPU RAM peak**: ~200-500 MB (зависит от количества обрабатываемых кадров)
- **GPU VRAM**: не требуется (CPU-only модуль)

**Почему так быстро?**
- Модули работают только с landmarks (координатами точек), которые уже вычислены в `core_face_landmarks`
- Нет inference нейросетей — все вычисления на NumPy (геометрические операции)
- Обработка только кадров с лицами (не всех кадров видео)

**Пример**: Для видео с 72 кадрами с лицами и 7 модулями:
- Общее время: ~150-200 ms
- Среднее время на кадр: ~2-3 ms
- Все модули выполняются для каждого кадра

## Quality validation & human-friendly inspection

### Как проверить качество выхода компонента

1. **Human-friendly визуализация**
   - Рекомендуется визуализировать bbox + landmarks + ключевые таймлайны (gaze/quality/attention).
   - Компонент пишет UI данные в `meta.ui_payload` внутри `detalize_face.npz` (без отдельных JSON артефактов).
   - Demo-скрипт: `modules/detalize_face/quality_report/demo_detalize_face_quality.py`
   - Пример запуска:
     ```bash
     python3 modules/detalize_face/quality_report/demo_detalize_face_quality.py \
       --rs-path /path/to/result_store \
       --out-dir /path/to/output
     ```

2. **Статистическая валидация**
   - Проверить диапазоны фичей: `quality_proxy_score` ∈ [0,1], `gaze_at_camera_prob` ∈ [0,1], `blink_rate` разумен.
   - Отсутствие NaN/Inf в итоговых сериях.

3. **Интеграция с downstream**
   - Проверить, что модель и UI корректно читают `detalize_face.npz` и `meta.ui_payload`.

### Human-friendly визуализация (Render System)

`detalize_face` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/detalize_face/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по обработке лиц (total_frames, processed_frames, frames_with_faces, total_faces, primary_faces, avg_faces_per_frame)
- **Timeline**: данные по каждому кадру (frame_index, time_s, face_count, primary_gaze_at_camera_prob, primary_blink_rate, primary_attention_score, primary_quality_proxy_score, primary_face_sharpness, primary_occlusion_proxy, primary_speech_activity_prob)
- **Distributions**: распределения метрик (face_count, primary_gaze_at_camera_prob, primary_blink_rate, primary_attention_score, primary_quality_proxy_score, primary_face_sharpness, primary_occlusion_proxy, primary_speech_activity_prob) с min, max, mean, std, median, percentiles
- **Faces Aggregates**: агрегированные фичи по каждому отслеженному лицу (tracking_id)

Render-context может быть использован:
- **LLM** для генерации текстовых описаний анализа лиц в видео
- **Frontend** для построения графиков и визуализаций (timeline charts с метриками лица, distributions метрик, face aggregates)
- **Debugging**: быстрая проверка качества обработки лиц без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../detalize_face/_render/render.html`
- **Offline** mini-dashboard (без CDN): SVG‑таймлайны по ключевым метрикам + таблицы распределений + список треков.
- Содержит блоки:
  - **Key facts**: `schema_version`, `producer_version`, `status`, `empty_reason`, `meta.stage_timings_ms`
  - **Timeline charts**: `face_count`, `primary_*` метрики (SVG)
  - **Top / Anti-top**: примеры кадров с max `primary_attention_score` и min `primary_quality_proxy_score`
  - **Distributions** и **tracked faces**

**Конфигурация** (в `global_config.yaml`):
```yaml
detalize_face:
  modules: null  # или список модулей
  max_faces: 4
  # ... другие параметры ...
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

## Параметры конфигурации

### Основные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `modules` | `List[str]` | `None` (все) | Список модулей для загрузки |
| `max_faces` | `int` | `10` (CLI: `4`) | Максимальное количество лиц на кадр |
| `refine_landmarks` | `bool` | `True` | Использовать уточненные landmarks (468 точек) |
| `min_detection_confidence` | `float` | `0.7` | Минимальная уверенность детекции |
| `min_tracking_confidence` | `float` | `0.7` | Минимальная уверенность трекинга |
| `min_face_size` | `int` | `30` | Минимальный размер лица в пикселях |
| `max_face_size_ratio` | `float` | `0.8` | Максимальное отношение размера лица к кадру |
| `min_aspect_ratio` | `float` | `0.6` | Минимальное соотношение сторон лица |
| `max_aspect_ratio` | `float` | `1.4` | Максимальное соотношение сторон лица |
| `validate_landmarks` | `bool` | `True` | Валидировать landmarks |
| `visualize` | `bool` | `False` | Включить визуализацию |
| `visualize_dir` | `str` | `"./face_visualizations"` | Директория для визуализаций |
| `show_landmarks` | `bool` | `False` | Показывать landmarks на визуализации |

### Рекомендации по настройке

- **`max_faces`**: 
  - 1-2 для портретных видео
  - 4-6 для групповых сцен
  - 10 для очень плотных сцен (может замедлить обработку)

- **`min_detection_confidence`**:
  - 0.7-0.8 для высокого качества (меньше ложных срабатываний)
  - 0.5-0.6 для максимального покрытия (больше лиц, но больше шума)

- **`modules`**:
  - Выбирайте только нужные модули для оптимизации производительности
- Некоторые модули требуют больше ресурсов (например, `lip_reading`)

## Технические детали

### Обработка landmarks

1. **Загрузка**: Landmarks загружаются из `core_face_landmarks/landmarks.npz`
2. **Восстановление координат**: Нормализованные координаты (0-1) преобразуются в пиксельные
3. **Валидация**: Проверка размера, соотношения сторон, уверенности
4. **Трекинг**: Назначение `tracking_id` через IoU между кадрами

### Временной анализ

Многие метрики требуют истории кадров:

- **Окно истории**: 30 кадров (≈1-1.5 сек при 25-30 fps)
- **Экспоненциальное скользящее среднее**: Для быстрого обновления
- **Трекинг**: Отслеживание лиц через `tracking_id` для агрегации по времени

### Сжатие векторов

Для экономии памяти и ускорения обработки:

- **PCA/Projection**: Landmarks сжимаются до 8-16 dims
- **Нормализация**: Все метрики нормализуются в [0, 1] или [-1, 1]
- **Компактные фичи**: Функция `extract_compact_features()` для VisualTransformer

### Обработка ошибок

- **Отсутствие core_face_landmarks**: Выбрасывается `RuntimeError` с описанием
- **Ошибки в модулях**: Логируются, но не прерывают обработку других модулей
- **Пропущенные кадры**: Кадры без лиц пропускаются с предупреждением
- **Некорректные данные**: Валидация на входе предотвращает ошибки

## Производительность

### Бенчмарки (измеренные)

На CPU (Intel i7-9700K), все 7 модулей, max_faces=4:
- **Скорость**: ~2-3 ms на кадр с лицом (≈300-500 кадров/сек)
- **Память**: ~200-500 MB (зависит от количества обрабатываемых кадров)
- **GPU**: не требуется (CPU-only модуль)

**Пример реальных измерений**:
- 72 кадра с лицами, 7 модулей: ~150 ms общее время
- Среднее время на кадр: ~2.1 ms
- Все модули выполняются для каждого кадра

### Почему так быстро?

1. **Только геометрические вычисления**: Модули работают с landmarks (координатами точек), которые уже вычислены в `core_face_landmarks`
2. **Нет inference нейросетей**: Все вычисления на NumPy (быстрые векторные операции)
3. **Обработка только кадров с лицами**: Не обрабатываются кадры без лиц (определяется `core_face_landmarks`)
4. **Оптимизированные NumPy операции**: Векторизованные вычисления для всех модулей

### Оптимизация

1. **Выбирайте нужные модули**: Отключите ненужные для ускорения (каждый модуль добавляет ~0.2-0.5 ms на кадр)
2. **Уменьшите max_faces**: Меньше лиц = быстрее обработка (линейная зависимость)
3. **Используйте фильтрацию**: `min_detection_confidence` и валидация уменьшают количество обрабатываемых лиц
4. **Сэмплирование**: Кадры определяются `core_face_landmarks` (модуль сам sampling не делает)

### Ограничения

- Модуль требует результаты `core_face_landmarks` (обязательная зависимость)
- Обработка очень длинных видео может быть медленной при большом количестве кадров с лицами (рассмотрите сэмплирование на уровне `core_face_landmarks`)
- Некоторые модули требуют больше ресурсов (`lip_reading` — обработка ROI изображения)

## Этика и Privacy

### Удаленные или помеченные как audit-only фичи

- Удалены чувствительные фичи (skin/accessories/professional/3d). Для identity используется hash.

### Рекомендации

- Любые метрики, связанные с "полом", "раской", "привлекательностью" — должны быть документированы, иметь justification и bias-audit
- Для identity-ориентированных векторов (`identity_shape_vector`) применяется privacy-preserving hashing или минимальная размерность
- Исключено хранение raw face embeddings, если есть GDPR/CCPA риски

### Privacy-preserving опции

- **`use_privacy_preserving`**: В `StructureModule` для `identity_shape_vector` применяется SHA-256 hashing вместо raw вектора
- **Минимальная размерность**: Векторы сжимаются до минимально необходимой размерности

## Примеры использования

### Базовый пример

```python
from modules.detalize_face.detalize_face_refactored import DetalizeFaceModule

module = DetalizeFaceModule(
    rs_path="./results",
    modules=["geometry", "pose", "quality"],
    max_faces=4
)

module.run(frames_dir="./frames", config={})
```

### Анализ вовлеченности

```python
module = DetalizeFaceModule(
    rs_path="./results",
    modules=["geometry", "pose", "eyes", "lip_reading"],
    max_faces=1,  # Только primary face
    min_detection_confidence=0.8  # Высокое качество
)

# Извлечение метрик вовлеченности
for frame_idx, faces in results.items():
    for face in faces:
        engagement = face.get("engagement_level", 0.0)
        gaze_at_camera = face.get("gaze_at_camera_prob", 0.0)
        attention = face.get("attention_score", 0.0)
        # ...
```

### Визуализация результатов

```python
module = DetalizeFaceModule(
    rs_path="./results",
    modules=["geometry", "pose", "quality"],
    visualize=True,
    visualize_dir="./face_visualizations",
    show_landmarks=True
)

module.run(frames_dir="./frames", config={})
```

## Логирование

Модуль использует стандартный logger из `utils.logger`:

```python
from utils.logger import get_logger

logger = get_logger("detalize_face")
```

Уровни логирования:
- **DEBUG**: Детальная информация о каждом кадре и модуле
- **INFO**: Общая информация о процессе (загрузка модулей, обработка кадров)
- **WARN**: Предупреждения (пропущенные кадры, ошибки в модулях)
- **ERROR**: Критические ошибки (отсутствие зависимостей, ошибки инициализации)

## Интеграция с пайплайном

Модуль интегрирован с VisualProcessor через `BaseModule`:

- Автоматическая загрузка зависимостей через `load_core_provider()`
- Стандартизированное сохранение результатов через `save_results()`
- Единый интерфейс `process()` для всех модулей
- Управление зависимостями через `required_dependencies()`

## Дополнительная документация

- [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md) — подробное описание всех извлекаемых фичей (464 строки)
- [BaseModule documentation](../../base_module.py) — базовый класс для модулей
- [core_face_landmarks README](../../core/model_process/core_face_landmarks/README.md) — описание провайдера landmarks

## Лицензия

Модуль является частью проекта TrendFlowML и использует результаты MediaPipe FaceMesh (лицензия Apache 2.0).

## Авторы

Компонент разработан как часть проекта TrendFlowML для анализа видео контента.

