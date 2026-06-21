# План адаптации VisualProcessor для батчевой обработки

## Обзор задачи

Адаптировать VisualProcessor и все его компоненты для одновременной обработки нескольких видео с:
- **Двухуровневой параллельностью**:
  - **Уровень 1**: Параллельная обработка нескольких видео через `ThreadPoolExecutor` (max_video_workers)
  - **Уровень 2**: Параллельная обработка кадров внутри одного видео (существующая frame-level parallelism)
- **Батчингом на GPU** для ML-моделей (CLIP, object detection, optical flow, depth, face landmarks, identity) с гибридным подходом:
  - Сбор кадров из всех видео
  - Батчинг с лимитом размера (max_frames_per_gpu_batch)
  - Распределение результатов обратно по видео
- **Сохранением изоляции** данных между файлами
- **Корректной обработкой метаданных** для каждого видео (metadata.json)
- **Валидацией NPZ файлов** для каждого компонента

## Статус реализации

🚧 **В разработке**. План создан на основе опыта разработки AudioProcessor и TextProcessor batch processing (Stage 0-5).

**Последнее обновление**: 
- ✅ Stage 0 завершена: базовый каркас `run_batch()`, `process_batch()`, `VideoContext` реализованы.
- ✅ Stage 1 завершена: изоляция артефактов реализована - каждый видео имеет свой `rs_path`, компоненты пишут в per-video директории, метаданные изолированы.
- ✅ Stage 2 завершена: GPU batching для core_clip реализован с гибридным подходом (батчинг кадров из всех видео).

---

## 0. Acceptance Criteria (критерии готовности / DoD)

Эти пункты — **критерии**, по которым можно идти "по компонентам" и рефакторить.  
Формат: сначала делаем **batch-safe** (безопасно для многовидео обработки), затем **batch-optimized** (ускорение).

### 0.1 Корректность (обязательное)

- **Эквивалентность результатов**: для каждого видео результаты `run_batch([video])` совпадают с `run(video)` (допустимы только тривиальные float-расхождения).
- **Изоляция**:
  - Артефакты (`*.npz`, временные файлы) пишутся **внутрь per-run ResultStore** и **не конфликтуют** между видео;
  - Нет shared mutable state между видео внутри компонентов (кроме read-only моделей/корпусов);
  - Каждое видео имеет свой `rs_path` для временных артефактов.
- **Метаданные**:
  - Корректная обработка `metadata.json` для каждого видео;
  - Изоляция метаданных между видео (нет пересечений в `frame_indices`, `video_id`, `run_id`);
  - Корректная работа с `frames_dir` для каждого видео.
- **Детерминизм**: запрещены `glob + mtime`, "последний файл", зависимости от абсолютных путей как source-of-truth.
- **Политика ошибок**:
  - падение одного видео **не валит** весь батч, если компонент не marked required;
  - required компонент → падение видео помечает **это видео** как error (и/или валит batch — выбрать и зафиксировать контрактом).
- **Наблюдаемость**: логирование/прогресс должны быть **с привязкой к video_id**.

### 0.2 Производительность (измеримое, но не блокирующее для MVP)

- **GPU batching** даёт ускорение на ML-моделях (CLIP, object detection, optical flow, depth, face landmarks, identity) относительно поштучного прогона.
- **CPU parallelism** даёт ускорение на "чисто CPU" этапах (color_light, frames_composition, similarity_metrics, etc.) без неограниченного роста RAM.
- **Frame batching**: оптимизация обработки кадров внутри одного видео (микробатчинг).
- Добавлены метрики: wall-time по стадиям/компонентам, утилизация GPU (best-effort), peak RAM (best-effort).

### 0.3 NPZ файлы и валидация (критично для VisualProcessor)

- **Корректность NPZ файлов**: каждый компонент должен генерировать корректные NPZ файлы с правильной структурой.
- **Валидация NPZ**: NPZ файлы должны проходить валидацию через `validate_npz()`.
- **Изоляция NPZ**: NPZ файлы для разных видео не должны пересекаться.
- **Тестирование NPZ**: автоматическая проверка корректности NPZ файлов для всех компонентов.

---

## 1. Чеклист внедрения (итеративно, стадиями)

### Стадия 0 — "каркас" без оптимизаций (MVP API) ✅

- [x] `BaseModule`: добавить `process_batch(video_contexts)` (дефолт — цикл `process`) и `supports_batch` (дефолт `False`).
- [x] `BaseModule`: добавить `process_batch_frames()` для батчинга кадров из нескольких видео (для GPU modules).
- [x] `MainProcessor` (main.py): добавить `run_batch(video_contexts)` (дефолт — последовательный вызов `_run_one_component()` на каждый видео).
- [x] `MainProcessor`: добавить `VideoContext` для изоляции контекста каждого видео (video_id, frames_dir, rs_path, metadata.json).
- [ ] Smoke: `run_batch([video])` == `run(video)` по базовым полям (`status/error/empty_reason`) + запуск через `DP_MODELS_ROOT`/`PYTHONPATH`.
- [ ] **Специфика VisualProcessor**: поддержка двухуровневой параллельности (видео + кадры).

### Стадия 1 — изоляция артефактов и видео-контекст (batch-safe foundation) ✅

- [x] Ввести **VideoContext**: `video_id`, `frames_dir`, `rs_path`, ссылки на result_store paths, `metadata.json` path.
- [x] Все компоненты, которые пишут файлы, должны писать **в свой per-video rs_path** (не общий).
  - Реализовано: `run_batch()` устанавливает `rs_path` для каждого компонента для каждого видео отдельно через `_process_single_video()`.
- [x] Везде, где имена артефактов фиксированные (`embeddings.npz`, `detections.npz`), обеспечить, что они фиксированные **внутри per-video rs_path** (иначе конфликт при батче).
  - Реализовано: каждое видео имеет свой `rs_path = <base_rs_path>/<platform_id>/<video_id>/<run_id>`.
- [x] Инвариант: артефакты содержат только relpath'и внутри **своего** `rs_path/`.
  - Реализовано: компоненты используют `self.rs_path` для сохранения .npz файлов через subprocess с переопределенным `rs_path`.
- [x] **Метаданные изоляция**: корректная обработка `metadata.json` для каждого видео.
  - Реализовано: `_process_single_video()` загружает метаданные для каждого видео отдельно через `video_ctx.load_metadata()`.
- [x] **NPZ изоляция**: NPZ файлы для разных видео не пересекаются.
  - Реализовано: каждый компонент запускается в subprocess с per-video `rs_path`, что обеспечивает полную изоляцию артефактов.

### Стадия 2 — первый GPU batching PoC (минимум: `core_clip`) ✅

- [x] Реализовать batch processing для `core_clip` с гибридным батчингом:
  - Сбор кадров из всех видео через `process_core_clip_batch()`
  - Группировка в батчи по `max_frames_per_batch` (если задан)
  - Последовательная обработка батчей через Triton или inprocess
  - Распределение результатов обратно по видео
- [x] Интеграция в `run_batch()`: batch processing для core_clip вызывается перед обработкой остальных компонентов.
- [ ] Добавить micro-bench: `scripts/bench_clip_batch.py` (loop `process()` vs `process_batch_frames()`) - отложено до тестирования.
- [x] **Специфика**: поддержка батчинга кадров внутри одного видео и между видео (гибридный подход).

### Стадия 3 — batching переменной длины (hard cases)

- [ ] `CoreObjectDetectionsModule.process_batch()`: батчирование кадров object detection из всех видео → batch inference → распределение обратно → сохранение per-video artifacts.
- [ ] `CoreOpticalFlowModule.process_batch()`: аналогично для optical flow.
- [x] `CoreDepthMidasModule.process_batch()`: аналогично для depth estimation.
  - Реализовано: `process_core_depth_midas_batch()` в `utils/core_depth_midas_batch.py`
  - Интегрировано в `run_batch()` в `main.py`
  - Гибридный батчинг: сбор кадров из всех видео → batch inference через Triton → распределение результатов обратно
- [ ] `CoreFaceLandmarksModule.process_batch()`: аналогично для face landmarks.
- [ ] `CoreIdentityModules.process_batch()`: аналогично для identity modules (brand, car, face, franchise, place, content_domain).
- [ ] **Специфика**: обработка кадров разного размера, padding для батчинга.

### Стадия 4 — Двухуровневая параллельность и CPU parallelism (по уровням зависимостей)

- [ ] Добавлены параметры двухуровневой параллельности в `MainProcessor.run_batch()`:
  - `max_video_workers`: количество параллельных воркеров для обработки видео (уровень 1)
  - `enable_video_parallel`: включение параллельной обработки нескольких видео
  - `max_frame_workers`: количество параллельных воркеров для кадров (уровень 2, для CPU modules)
  - `enable_frame_parallel`: включение параллельной обработки кадров
  - `enable_gpu_batching`: включение GPU batching для кадров
  - `max_frames_per_gpu_batch`: лимит размера батча для GPU modules (null = без лимита)
  - `enable_cpu_parallel`: включение CPU параллелизма для независимых modules
- [ ] Реализован граф зависимостей (`_build_dependency_levels()`) с топологической сортировкой для группировки компонентов по уровням.
- [ ] Использовать существующий граф зависимостей (`MODULE_DEPS`, `MODULE_CORE_DEPS`, `CORE_DEPS`) для определения порядка компонентов.
- [ ] **Уровень 1 (видео)**: ThreadPoolExecutor для параллельной обработки нескольких видео (если `enable_video_parallel=True`).
- [ ] **Уровень 2 (кадры)**:
  - CPU modules: существующая frame-level parallelism через ThreadPoolExecutor внутри `process()`
  - GPU modules: гибридный батчинг кадров из всех видео с лимитом размера батча
- [ ] Обработка по уровням зависимостей: компоненты одного уровня могут выполняться параллельно/в батче для всех видео одновременно.
- [ ] GPU batch modules обрабатываются батчем кадров из всех видео (если `supports_batch=True` и `enable_gpu_batching=True`).
- [ ] CPU modules обрабатываются параллельно через ThreadPoolExecutor на уровне видео и кадров (если `enable_cpu_parallel=True`).
- [ ] GPU legacy modules обрабатываются последовательно для каждого видео.
- [ ] Лимиты: `max_video_workers` контролирует параллельность на уровне видео, `max_frame_workers` - на уровне кадров.

**Реализованные зависимости** (из `main.py`):
- Core dependencies: `core_brand_semantics` → `core_object_detections`, `core_car_semantics` → `core_object_detections`, `core_place_semantics` → `core_object_detections`, `core_clip`, `core_face_identity` → `core_object_detections`, `core_face_landmarks`
- Module dependencies: `shot_quality` → `cut_detection`, `video_pacing` → `cut_detection`, `high_level_semantic` → `cut_detection`, `emotion_face`, `scene_classification` → `cut_detection`
- Module → Core dependencies: `cut_detection` → `core_optical_flow`, `optical_flow` → `core_optical_flow`, `scene_classification` → `core_clip`, `video_pacing` → `core_optical_flow`, `core_clip`, `shot_quality` → `core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks`, etc.

### Стадия 5 — CLI интеграция и production-ready batch processing

- [ ] Добавлены CLI аргументы `--video-input-dir` и `--video-input-list` для batch режима.
- [ ] Интеграция в верхний оркестратор (`DataProcessor/main.py`) с поддержкой batch флагов (через `global_config_parser.get_visual_cli_args()`).
- [ ] Конфигурация через `global_config.yaml`:
  - `visual.batch_processing.enabled`: включение batch режима
  - `visual.batch_processing.max_video_workers`: количество параллельных воркеров для видео (null = auto, обычно os.cpu_count())
  - `visual.batch_processing.enable_video_parallel`: включение параллельной обработки нескольких видео
  - `visual.batch_processing.max_frame_workers`: количество параллельных воркеров для кадров (null = auto, для CPU modules)
  - `visual.batch_processing.enable_frame_parallel`: включение параллельной обработки кадров
  - `visual.batch_processing.enable_gpu_batching`: включение GPU batching для кадров
  - `visual.batch_processing.max_frames_per_gpu_batch`: лимит размера батча для GPU modules (null = без лимита)
  - `visual.batch_processing.enable_cpu_parallel`: включение CPU параллелизма для независимых modules
- [ ] CLI флаги для тонкой настройки:
  - `--batch-max-workers`: переопределение max_workers
  - `--no-batch-gpu`: отключение GPU batching
  - `--no-batch-cpu-parallel`: отключение CPU параллелизма
  - `--batch-max-frames-per-gpu-batch`: лимит размера батча для GPU modules
- [ ] Изоляция результатов: каждое видео сохраняется в отдельную директорию внутри ResultStore.
- [ ] Валидация NPZ файлов для каждого видео в batch режиме.
- [ ] **NPZ валидация**: автоматическая проверка корректности NPZ файлов для всех компонентов (опционально, можно отложить).

---

## 2. Специфичные моменты VisualProcessor (уроки из AudioProcessor/TextProcessor)

### 2.1 Файлы NPZ и их корректность

**Проблема**: Каждый компонент должен генерировать корректные NPZ файлы с правильной структурой и метаданными.

**Решение**:
- Автоматическая валидация NPZ файлов при сохранении через `validate_npz()`.
- Тестирование NPZ файлов на реальных данных.
- Документирование формата NPZ для каждого компонента.

**Чеклист**:
- [ ] Все компоненты генерируют корректные NPZ файлы.
- [ ] NPZ файлы проходят валидацию через `validate_npz()`.
- [ ] NPZ файлы изолированы между видео.
- [ ] Автоматические тесты для NPZ файлов.

### 2.2 Время и производительность

**Проблема**: VisualProcessor обрабатывает большие видео файлы и кадры, что требует оптимизации.

**Решение**:
- GPU batching для ML-моделей (CLIP, object detection, optical flow, depth, face landmarks, identity).
- CPU parallelism для CPU modules (color_light, frames_composition, similarity_metrics, etc.).
- Frame batching внутри одного видео (микробатчинг).

**Метрики**:
- Wall-time по стадиям/компонентам.
- Утилизация GPU (best-effort).
- Peak RAM (best-effort).
- Время обработки на видео.

### 2.3 Ошибки и error handling

**Проблема**: Ошибки в одном компоненте не должны валить весь batch.

**Решение**:
- Каждый компонент обёрнут в try/except, ошибки собираются в `errors_by_component`.
- Required компоненты (через `required_components` параметр) fail-fast при ошибках.
- Optional компоненты логируют warning и продолжают run.

**Чеклист**:
- [ ] Error handling для всех компонентов.
- [ ] Логирование ошибок с привязкой к video_id.
- [ ] Fail-fast для required компонентов.
- [ ] Graceful degradation для optional компонентов.

### 2.4 Взаимодействия между компонентами

**Проблема**: VisualProcessor имеет сложный граф зависимостей между core components и modules.

**Решение**:
- Корректная обработка графа зависимостей для каждого видео.
- Изоляция результатов между видео.
- Корректная работа с зависимостями (`MODULE_DEPS`, `MODULE_CORE_DEPS`, `CORE_DEPS`).

**Чеклист**:
- [ ] Корректная обработка графа зависимостей для каждого видео.
- [ ] Изоляция результатов между видео.
- [ ] Корректная работа с зависимостями (core → core, module → module, module → core).

### 2.5 Зависимости

**Проблема**: Компоненты имеют зависимости друг от друга (например, `shot_quality` → `cut_detection`, `core_brand_semantics` → `core_object_detections`).

**Решение**:
- Использовать существующий граф зависимостей (`MODULE_DEPS`, `MODULE_CORE_DEPS`, `CORE_DEPS`) для определения порядка компонентов.
- Топологическая сортировка для группировки компонентов по уровням.
- Валидация зависимостей перед запуском batch.

**Чеклист**:
- [ ] Граф зависимостей компонентов корректно определен.
- [ ] Топологическая сортировка работает корректно.
- [ ] Зависимости валидируются перед запуском batch.
- [ ] Автоматическое добавление недостающих зависимостей (опционально).

### 2.6 Батчинг и параллелизм

**Проблема**: Нужно оптимизировать обработку для ускорения.

**Решение**:
- GPU batching для ML-моделей (CLIP, object detection, optical flow, depth, face landmarks, identity).
- CPU parallelism для CPU modules.
- Frame batching внутри одного видео.

**Чеклист**:
- [ ] GPU batching реализован для ML-моделей.
- [ ] CPU parallelism реализован для CPU modules.
- [ ] Frame batching реализован внутри одного видео.
- [ ] Метрики производительности собираются.

### 2.7 Render система (Human-friendly визуализация)

**Проблема**: Каждый компонент должен иметь корректный `render.py` для генерации render-context JSON и HTML debug страниц.

**Решение**:
- Автоматическая генерация render-context JSON для каждого компонента после успешной обработки.
- Динамическая загрузка renderer'ов из `core/model_process/<component>/render.py` или `modules/<component>/render.py`.
- Конфигурация через `global_config.yaml` с флагами `enable_render` и `enable_html_render`.
- Best-effort генерация: ошибки render не валят основной процесс.

**Чеклист**:
- [x] Render система реализована в `utils/renderer.py` с динамической загрузкой renderer'ов.
- [x] `core_clip` имеет `render.py` с функциями `render_core_clip()` и `render_core_clip_html()`.
- [x] `core_depth_midas` имеет `render.py` с функциями `render_core_depth_midas()` и `render_core_depth_midas_html()`.
- [ ] Остальные core components имеют `render.py` (по мере необходимости).
- [ ] Modules имеют `render.py` (по мере необходимости).
- [x] Render система интегрирована в `main.py` через `_run_component_subprocess()`.
- [x] Конфигурация render системы добавлена в `global_config.yaml` для `core_clip` и `core_depth_midas`.

### 2.8 Документация

**Проблема**: Нужна полная документация для batch processing.

**Решение**:
- Обновить `README.md` с информацией о batch режиме.
- Создать примеры использования.
- Документировать конфигурацию через `global_config.yaml`.

**Чеклист**:
- [x] `README.md` обновлен с информацией о batch режиме и render системе.
- [ ] Примеры использования созданы.
- [x] Конфигурация документирована.
- [ ] API документирован.

### 2.9 Глобальный конфиг и флаги

**Проблема**: Нужна конфигурация batch processing через `global_config.yaml`.

**Решение**:
- Добавить секцию `visual.batch_processing` в `global_config.yaml`.
- Парсинг конфигурации в `config_parser.py`.
- Передача параметров в `MainProcessor` и CLI.

**Чеклист**:
- [ ] Секция `visual.batch_processing` добавлена в `global_config.yaml`.
- [ ] Парсинг конфигурации реализован в `config_parser.py`.
- [ ] Параметры передаются в `MainProcessor` и CLI.
- [ ] CLI флаги для тонкой настройки реализованы.

### 2.9 Модели и ModelManager

**Проблема**: Модели должны загружаться только через `dp_models` (ModelManager), без сетевых загрузок.

**Решение**:
- Использовать `get_global_model_manager()` для загрузки моделей.
- Enforce offline/no-network policy.
- Ленивая загрузка моделей (lazy loading).

**Чеклист**:
- [ ] Все модели загружаются через ModelManager.
- [ ] Offline/no-network policy enforced.
- [ ] Ленивая загрузка моделей реализована.
- [ ] Модели изолированы между видео (read-only shared state).

### 2.10 FrameManager и кадры

**Проблема**: FrameManager должен корректно работать с несколькими видео одновременно.

**Решение**:
- Изоляция FrameManager для каждого видео.
- Корректная работа с `frames_dir` для каждого видео.
- Оптимизация загрузки кадров для batch режима.

**Чеклист**:
- [ ] FrameManager изолирован для каждого видео.
- [ ] Корректная работа с `frames_dir` для каждого видео.
- [ ] Оптимизация загрузки кадров для batch режима.
- [ ] Кеширование кадров для batch режима (опционально).

---

## 3. Матрица готовности по компонентам (чеклист)

Легенда:
- **batch-safe**: корректно работает при обработке нескольких видео (без утечек/конфликтов), допускается внутренний цикл по видео.
- **batch-optimized**: реализован `process_batch()` и есть ожидаемое ускорение.
- **artifacts**: пишет ли `*.npz` и требуется ли раздельный rs_path.
- **npz**: есть ли корректный NPZ файл с валидацией.

### Core Components

| Component (class) | Device | Зависимости | batch-safe | batch-optimized | artifacts | npz | Критичные риски/заметки |
|---|---|---|---|---|---:|---:|---|
| `core_clip` | GPU | - | ⬜ | ⬜ | ✅ | ⬜ | ML-модель, GPU batching возможен (Stage 2) |
| `core_object_detections` | GPU | - | ⬜ | ⬜ | ✅ | ⬜ | ML-модель (YOLO), GPU batching возможен (Stage 3) |
| `core_optical_flow` | GPU | - | ⬜ | ⬜ | ✅ | ⬜ | ML-модель (RAFT), GPU batching возможен (Stage 3) |
| `core_depth_midas` | GPU | - | ✅ | ✅ | ✅ | ⬜ | ML-модель (MiDaS), GPU batching реализован (Stage 3) |
| `core_face_landmarks` | CPU/GPU | - | ⬜ | ⬜ | ✅ | ⬜ | MediaPipe, может быть параллелизован (Stage 3) |
| `ocr_extractor` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | Tesseract, может быть параллелизован |
| `core_brand_semantics` | GPU | core_object_detections | ⬜ | ⬜ | ✅ | ⬜ | Embedding Service, зависит от object detections |
| `core_car_semantics` | GPU | core_object_detections | ⬜ | ⬜ | ✅ | ⬜ | Embedding Service, зависит от object detections |
| `core_face_identity` | GPU | core_object_detections, core_face_landmarks | ⬜ | ⬜ | ✅ | ⬜ | Embedding Service, зависит от detections и landmarks |
| `core_franchise_recognition` | GPU | - | ⬜ | ⬜ | ✅ | ⬜ | CLIP text encoder, GPU batching возможен |
| `core_place_semantics` | GPU | core_object_detections, core_clip | ⬜ | ⬜ | ✅ | ⬜ | Зависит от detections и CLIP |
| `core_content_domain` | GPU | - | ⬜ | ⬜ | ✅ | ⬜ | CLIP text encoder, GPU batching возможен |

### Modules

| Module (class) | Device | Зависимости | batch-safe | batch-optimized | artifacts | npz | Критичные риски/заметки |
|---|---|---|---|---|---:|---:|---|
| `cut_detection` | CPU | core_optical_flow | ⬜ | ⬜ | ✅ | ⬜ | Зависит от optical flow, может быть параллелизован |
| `shot_quality` | CPU/GPU | cut_detection, core_clip, core_depth_midas, core_object_detections, core_face_landmarks | ⬜ | ⬜ | ✅ | ⬜ | Агрегатор, зависит от многих core components |
| `video_pacing` | CPU | cut_detection, core_optical_flow, core_clip | ⬜ | ⬜ | ✅ | ⬜ | Зависит от cut detection и core components |
| `scene_classification` | CPU/GPU | cut_detection, core_clip | ⬜ | ⬜ | ✅ | ⬜ | Зависит от cut detection и CLIP |
| `story_structure` | CPU/GPU | core_clip, core_optical_flow, core_face_landmarks | ⬜ | ⬜ | ✅ | ⬜ | Агрегатор, зависит от core components |
| `uniqueness` | CPU/GPU | core_clip | ⬜ | ⬜ | ✅ | ⬜ | Зависит от CLIP |
| `behavioral` | CPU | core_face_landmarks | ⬜ | ⬜ | ✅ | ⬜ | Зависит от face landmarks |
| `high_level_semantic` | CPU/GPU | cut_detection, emotion_face, core_clip | ⬜ | ⬜ | ✅ | ⬜ | Агрегатор, зависит от cut detection, emotion и CLIP |
| `micro_emotion` | CPU | core_face_landmarks | ⬜ | ⬜ | ✅ | ⬜ | OpenFace, зависит от face landmarks |
| `detalize_face` | CPU/GPU | core_face_landmarks | ⬜ | ⬜ | ✅ | ⬜ | Зависит от face landmarks |
| `emotion_face` | CPU/GPU | core_face_landmarks | ⬜ | ⬜ | ✅ | ⬜ | EmoNet, зависит от face landmarks |
| `action_recognition` | GPU | core_object_detections | ⬜ | ⬜ | ✅ | ⬜ | SlowFast, зависит от object detections |
| `text_scoring` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | OCR consumer, может быть опциональным |
| `color_light` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | CPU extractor, может быть параллелизован |
| `frames_composition` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | CPU extractor, может быть параллелизован |
| `similarity_metrics` | CPU | - | ⬜ | ⬜ | ✅ | ⬜ | CPU extractor, может быть параллелизован |
| `optical_flow` | CPU/GPU | core_optical_flow | ⬜ | ⬜ | ✅ | ⬜ | Зависит от core_optical_flow |

---

## 4. Риски и митигация

### 4.1 Риск: Сложность изоляции артефактов
**Митигация**: Использовать per-video rs_path с четкой изоляцией директорий

### 4.2 Риск: Зависимости между компонентами
**Митигация**: Использовать существующий граф зависимостей (`MODULE_DEPS`, `MODULE_CORE_DEPS`, `CORE_DEPS`) для определения порядка компонентов

### 4.3 Риск: Разные размеры кадров
**Митигация**: Padding для батчинга, группировка по размерам

### 4.4 Риск: Память GPU
**Митигация**: Динамический размер батча, мониторинг памяти

### 4.5 Риск: Метаданные
**Митигация**: Строгая валидация `metadata.json` для каждого видео

### 4.6 Риск: NPZ файлы
**Митигация**: Автоматическая валидация NPZ файлов, тестирование на реальных данных

### 4.7 Риск: FrameManager
**Митигация**: Изоляция FrameManager для каждого видео, оптимизация загрузки кадров

### 4.8 Риск: GPU gating
**Митигация**: Сохранение существующего GPU gating через semaphore, адаптация для batch режима

---

## 5. Дополнительные улучшения (опционально)

### 5.1 Кеширование
- Общий кеш кадров для всех видео в батче
- Кеширование результатов CPU modules

### 5.2 Асинхронная обработка
- Асинхронная загрузка кадров для следующего батча
- Перекрытие GPU и CPU обработки

### 5.3 Мониторинг
- Метрики производительности в реальном времени
- Алерты при превышении лимитов памяти/времени

---

## 6. Следующие шаги

1. **Начать с Stage 0**: создать базовый каркас `run_batch()` API
2. **Stage 1**: реализовать изоляцию артефактов и видео-контекст
3. **Stage 2**: реализовать GPU batching для CLIP module'а
4. **Stage 3**: расширить GPU batching на другие ML-модели
5. **Stage 4**: реализовать CPU parallelism по уровням зависимостей
6. **Stage 5**: интегрировать в CLI и production

---

## 7. Ссылки

- [AudioProcessor Batch Processing Plan](../../AudioProcessor/docs/BATCH_PROCESSING_PLAN.md) — исходный план для AudioProcessor
- [TextProcessor Batch Processing Plan](../../TextProcessor/docs/BATCH_PROCESSING_PLAN.md) — исходный план для TextProcessor
- [VisualProcessor README](../README.md) — основная документация VisualProcessor
- [Dependency Graph](../../docs/reference/component_graph.yaml) — граф зависимостей компонентов
- [BaseModule](../modules/base_module.py) — базовый класс для всех modules
---

## Навигация

[Module README](../README.md) · [VisualProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
