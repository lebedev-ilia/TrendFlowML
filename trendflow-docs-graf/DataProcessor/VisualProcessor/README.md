# VisualProcessor

VisualProcessor — процессор визуальной модальности. Он извлекает **визуальные признаки** из кадров видео и сохраняет результат в **per‑run `result_store`**.

**Нормализация (portfolio + prod):** [docs/NORMALIZATION_WAVE4.md](docs/NORMALIZATION_WAVE4.md) · [docs/PORTFOLIO_PROGRESS_LOG.md](../docs/PORTFOLIO_PROGRESS_LOG.md)

## Контракт входа

- **Единица обработки**: `frames_dir` с кадрами видео + `metadata.json` (contract `video_metadata_v1`).
- **Источник**: Segmenter генерирует кадры и `metadata.json` в `frames_dir`, VisualProcessor читает их:
  - **Single-file mode**: через флаг `--frames-dir` (обязателен)
  - **Batch mode**: через флаг `--video-input-dir` или `--video-input-list` (обязателен для batch mode)
- **Требования (no‑fallback)**:
  - Если VisualProcessor включён в профиль как required, отсутствие `metadata.json` → run должен падать на уровне CLI (`raise RuntimeError`).
  - Отсутствие обязательных компонентов-зависимостей → fail-fast (`raise RuntimeError`).
  - Модели должны грузиться **только локально** через `dp_models` (no‑network policy).
- **Batch mode**: Каждый frames_dir должен содержать `metadata.json`. Директории без этого файла пропускаются с предупреждением.

### Формат `metadata.json`

Ключевые поля:
- `schema_version="video_metadata_v1"`
- `total_frames`, `analysis_fps`, `analysis_width`, `analysis_height`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `frame_indices` (опционально, для каждого компонента)
- `chunk_size`, `cache_size` (параметры FrameManager)

## Контракт выхода (result_store)

VisualProcessor пишет **отдельные NPZ артефакты для каждого компонента**:

- `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/<component_name>_features.npz`

и обновляет:

- `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`

### NPZ schema

Схема: `schema_version="visual_npz_v1"` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Минимальные ключи:
- Компонент-специфичные ключи (например, `frame_embeddings`, `detections`, `landmarks`, `depth_maps`)
- `meta: object(dict)` — run identity + версии + статус + метаданные

Обязательные поля `meta`:
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status` ∈ {`ok`, `empty`, `error`}
- `empty_reason` (только если `status="empty"`, иначе null)
- `models_used[]`, `model_signature` (если используются модели)
- `device_used` (если применимо)
- `total_frames`, `processed_frames`
- `stage_timings_ms` — словарь с таймингами ключевых стадий выполнения (в миллисекундах)

## Архитектура компонентов

VisualProcessor состоит из двух типов компонентов:

### Core Components (`core/model_process/`)

Базовые провайдеры, которые извлекают низкоуровневые признаки:

- **`core_clip`**: CLIP embeddings для кадров (GPU, Triton)
- **`core_object_detections`**: Детекция объектов (YOLO, GPU/CPU)
- **`core_optical_flow`**: Оптический поток (RAFT, GPU, Triton)
- **`core_depth_midas`**: Оценка глубины (MiDaS, GPU, Triton)
- **`core_face_landmarks`**: Landmarks лиц (MediaPipe, CPU)
- **`ocr_extractor`**: OCR текст (Tesseract, CPU)
- **`core_identity/*`**: Семантические головы (brand, car, face, franchise, place, content_domain)

### Modules (`modules/`)

Высокоуровневые модули, которые анализируют признаки:

- **`cut_detection`**: Детекция склеек
- **`shot_quality`**: Качество кадров
- **`video_pacing`**: Темп видео
- **`scene_classification`**: Классификация сцен
- **`story_structure`**: Структура истории
- **`emotion_face`**: Эмоции на лицах
- **`detalize_face`**: Детальный анализ лиц
- **`behavioral`**: Поведенческий анализ
- **`action_recognition`**: Распознавание действий
- **`color_light`**: Анализ цвета и света
- **`frames_composition`**: Композиция кадров
- **`similarity_metrics`**: Метрики схожести
- **`uniqueness`**: Уникальность контента
- **`text_scoring`**: Оценка текста
- **`high_level_semantic`**: Высокоуровневая семантика
- **`micro_emotion`**: Микро-эмоции
- **`optical_flow`**: Анализ оптического потока

### Зависимости компонентов

Компоненты имеют зависимости друг от друга:

- **Core dependencies**: `core_brand_semantics` → `core_object_detections`, `core_face_identity` → `core_object_detections`, `core_face_landmarks`, etc.
- **Module dependencies**: `shot_quality` → `cut_detection`, `video_pacing` → `cut_detection`, `high_level_semantic` → `cut_detection`, `emotion_face`, etc.
- **Module → Core dependencies**: `cut_detection` → `core_optical_flow`, `scene_classification` → `core_clip`, `shot_quality` → `core_clip`, `core_depth_midas`, etc.

VisualProcessor автоматически разрешает зависимости и выполняет компоненты в правильном порядке.

## Запуск (CLI)

### Single-file mode (одиночная обработка)

Standalone (с Segmenter contract):

```bash
python3 VisualProcessor/main.py \
  --cfg-path /path/to/config.yaml
```

Конфигурация через YAML:

```yaml
global:
  frames_dir: /path/to/frames_dir
  rs_path: /path/to/result_store
  platform_id: youtube
  video_id: <video_id>
  run_id: <run_id>
  sampling_policy_version: v1
  dataprocessor_version: unknown

core_providers:
  core_clip: true
  core_object_detections: true
  core_optical_flow: true
  core_depth_midas: true
  core_face_landmarks: true

modules:
  cut_detection: true
  shot_quality: true
  video_pacing: true
```

### Batch mode (батчевая обработка)

**Статус**: Stage 0-2 завершены, Stage 3-5 в разработке.

Batch mode позволяет обрабатывать несколько видео одновременно с оптимизацией:

- **Двухуровневая параллельность**: параллельная обработка нескольких видео + параллельная обработка кадров (Stage 4+)
- **GPU batching**: батчинг кадров из всех видео для ML-моделей (Stage 2+ реализован для `core_clip`)
- **CPU parallelism**: параллельная обработка CPU-компонентов (Stage 4+)
- **Изоляция артефактов**: каждый видео имеет свой `rs_path` (Stage 1+)

**Реализованные возможности**:
- ✅ Базовый каркас batch processing (`run_batch()`, `VideoContext`)
- ✅ Изоляция артефактов между видео (per-video `rs_path`)
- ✅ GPU batching для `core_clip` с гибридным подходом (сбор кадров из всех видео → батчинг → распределение результатов)

#### Использование batch mode

```bash
python3 VisualProcessor/main.py \
  --cfg-path /path/to/config.yaml \
  --video-input-dir /path/to/videos_dir
```

Или через JSON список:

```bash
python3 VisualProcessor/main.py \
  --cfg-path /path/to/config.yaml \
  --video-input-list /path/to/videos_list.json
```

Формат `videos_list.json`:

```json
[
  "/path/to/video1/frames_dir",
  "/path/to/video2/frames_dir",
  "/path/to/video3/frames_dir"
]
```

Или:

```json
{
  "frames_dirs": [
    "/path/to/video1/frames_dir",
    "/path/to/video2/frames_dir"
  ]
}
```

#### Конфигурация batch processing

Через `global_config.yaml`:

```yaml
visual:
  batch_processing:
    enabled: true  # Enable batch processing optimizations
    max_video_workers: 4  # Количество параллельных воркеров для видео (null = auto, обычно os.cpu_count())
    enable_video_parallel: false  # Включить параллельную обработку видео (Stage 4+, пока последовательно)
    max_frames_per_gpu_batch: 32  # Лимит размера батча для GPU компонентов (null = без лимита)
    enable_gpu_batching: true  # Включить GPU batching для кадров из нескольких видео (Stage 2+)
    enable_cpu_parallel: false  # Включить CPU параллелизм для независимых компонентов (Stage 4+)
```

**Примечания**:
- `enable_gpu_batching=true` активирует batch processing для `core_clip` (если включен в `core_providers`)
- `max_frames_per_gpu_batch` контролирует размер батча для GPU inference (по умолчанию используется `batch_size` из конфигурации компонента)
- Batch processing автоматически используется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`

#### Программный интерфейс batch mode

```python
from VisualProcessor.main import run_batch
from VisualProcessor.utils.batch_utils import collect_video_contexts
from VisualProcessor.utils.video_context import VideoContext

# Сбор контекстов видео
video_contexts = collect_video_contexts(
    video_input_dir="/path/to/videos_dir",
    rs_base="/path/to/result_store",
    platform_id="youtube",
    run_id="run_123"
)

# Запуск batch processing
results = run_batch(
    config=config,
    video_contexts=video_contexts,
    max_video_workers=4,
    enable_video_parallel=True,
    enable_gpu_batching=True,
    enable_cpu_parallel=True
)
```

## Программный интерфейс

### Использование BaseModule

Все модули наследуются от `BaseModule`:

```python
from modules.color_light.processor import ColorLightProcessor

# Инициализация
module = ColorLightProcessor(rs_path="/path/to/results")

# Конфигурация
config = {
    "max_frames_per_scene": 350,
    "stride": 5
}

# Запуск
saved_path = module.run(
    frames_dir="/path/to/frames",
    config=config
)
```

### Batch processing для модулей

Модули могут поддерживать batch processing:

```python
from modules.color_light.processor import ColorLightProcessor
from utils.video_context import VideoContext

# Инициализация
module = ColorLightProcessor(rs_path="/path/to/results")

# Проверка поддержки batch
if module.supports_batch:
    # Batch processing
    video_contexts = [
        VideoContext(video_id="video1", frames_dir="/path/to/video1", rs_path="/path/to/rs1"),
        VideoContext(video_id="video2", frames_dir="/path/to/video2", rs_path="/path/to/rs2"),
    ]
    results = module.process_batch(video_contexts, config={})
else:
    # Последовательная обработка
    for video_ctx in video_contexts:
        result = module.process(...)
```

## Структура выходных данных

Результаты сохраняются в формате NPZ через `BaseModule.save_results()`. Структура данных зависит от компонента:

### Core Components

- **`core_clip`**: `frame_embeddings`, `frame_indices`, `text_embeddings` (если есть)
- **`core_object_detections`**: `detections`, `frame_indices`, `summary`
- **`core_optical_flow`**: `flow`, `motion_norm_per_sec_mean`, `frame_indices`
- **`core_depth_midas`**: `depth_maps`, `frame_indices`
- **`core_face_landmarks`**: `landmarks`, `frame_indices`, `has_any_face`

### Modules

- **`cut_detection`**: `cuts`, `cut_scores`, `frame_indices`
- **`shot_quality`**: `quality_scores`, `frame_indices`
- **`video_pacing`**: `pacing_metrics`, `frame_indices`
- **`emotion_face`**: `emotions`, `emotion_probs`, `frame_indices`

## Human-friendly визуализация (Render System)

VisualProcessor генерирует **render-context JSON** для каждого компонента:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по извлечённым признакам (counts, dimensions, norms, means, stds)
- **Timeline**: данные по каждому кадру/сегменту с временными метками (frame_index, time_sec, values)
- **Distributions**: распределения значений (min, max, mean, std, median, percentiles)
- **Quality flags**: предупреждения, confidence scores, валидационные флаги

Render-context может быть использован:
- **LLM** для генерации текстовых описаний видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions, heatmaps)
- **Debugging**: быстрая проверка качества извлечённых признаков без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../<component_name>/_render/render.html`
- Содержит offline SVG графики (без CDN):
  - Timeline: визуализация признаков по времени
  - Distributions: статистики и распределения значений
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
visual:
  core_providers:
    core_clip:
      render:
        enable_render: true  # Генерировать render-context JSON
        enable_html_render: true  # Генерировать HTML debug страницу
  
  modules:
    cut_detection:
      render:
        enable_render: true
        enable_html_render: true
```

**Примечания**:
- Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс)
- Каждый компонент может иметь свой собственный `render.py` с функциями `render_<component_name>()` и `render_<component_name>_html()`
- Render система динамически загружает renderer'ы из `core/model_process/<component>/render.py` или `modules/<component>/render.py`
- Если renderer не найден, генерируется базовый render-context с минимальной информацией

**Примеры визуализации**:
- **`core_clip`**: Timeline с embedding norms и cosine similarity, распределения norms, информация о text embeddings
- **`cut_detection`**: Timeline с cut scores и detected cuts, распределения scores
- **`shot_quality`**: Timeline с quality scores по времени, статистики по качеству кадров

## Зависимости

Основные зависимости:

- `numpy`, `opencv-python`, `PIL`
- `torch`, `torchvision` (для ML-моделей)
- `mediapipe` (для face landmarks)
- `tesseract` (для OCR)
- `transformers` (для некоторых моделей)

См. `core/model_process/REQUIREMENTS.md` для детальных требований по компонентам.

## Документация

- [Batch Processing Plan](docs/BATCH_PROCESSING_PLAN.md) — план адаптации для batch processing
- [Core Components README](core/model_process/README_RUN_ALL_CORE.md) — документация по core components
- [Module READMEs](modules/*/README.md) — документация по отдельным модулям

## Примеры использования

### Обработка одного видео

```bash
python3 VisualProcessor/main.py \
  --cfg-path configs/visual_baseline.yaml
```

### Обработка нескольких видео (batch mode)

```bash
python3 VisualProcessor/main.py \
  --cfg-path configs/visual_baseline.yaml \
  --video-input-dir /path/to/videos
```

### Запуск всех core components

```bash
python3 VisualProcessor/core/model_process/run_all_core_components.py \
  --frames-dir /path/to/frames \
  --rs-path /path/to/result_store \
  --triton-http-url http://localhost:8000 \
  --out-dir /path/to/output \
  --batch-size 16
```

## Профилирование и мониторинг

Все компоненты VisualProcessor поддерживают **профилирование времени выполнения**:

- **Логирование таймингов**: после завершения обработки каждый компонент логирует тайминги всех стадий в консоль:
  ```
  core_clip | stage timings (ms): image_embeddings_total=4488.0, image_inference=4101.7, ...
  core_object_detections | stage timings (ms): initialization=0.5, process_frames=7899.0, ...
  ```
- **Сохранение в артефактах**: тайминги сохраняются в `meta.stage_timings_ms` в NPZ артефакте для последующего анализа
- **Стадии профилирования**: каждый компонент измеряет время ключевых стадий (initialization, load_deps, process_frames, saving, total)

Профилирование помогает:
- Выявить узкие места в производительности
- Оптимизировать компоненты
- Планировать ресурсы для обработки видео

## Примечания

- VisualProcessor требует корректный `metadata.json` в `frames_dir`
- Все компоненты сохраняют результаты в NPZ формате с валидацией
- **Batch processing** (Stage 0-3 завершены):
  - Изоляция артефактов между видео (per-video `rs_path`)
  - GPU batching для `core_clip`, `core_object_detections`, `core_depth_midas` с гибридным подходом
  - Автоматическое использование при обработке нескольких видео
- **Планируется** (Stage 4-5):
  - Двухуровневая параллельность (параллельная обработка видео + кадров)
  - CPU parallelism для независимых компонентов
  - CLI интеграция для batch mode
---

## Навигация

[VisualProcessor](docs/MAIN_INDEX.md) · [DataProcessor](../docs/MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
