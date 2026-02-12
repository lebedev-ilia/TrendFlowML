# Аудит соответствия franchise_recognition требованиям baseline

**Дата проверки**: 2026-01-XX  
**Компонент**: `franchise_recognition` (Semantic head, v1)  
**Расположение**: `VisualProcessor/core/model_process/core_identity/franchise_recognition/`  
**Статус аудита**: ✅ **CLOSED** (2026-01-XX)

## Резюме

`franchise_recognition` — semantic head компонент для распознавания конкретных франшиз/тайтлов в видео (игры, аниме, мультфильмы). Компонент использует frame embeddings из `core_clip`, выполняет поиск через Embedding Service и возвращает per-frame и video-level top‑K идентификаций франшиз.

Текущее состояние: компонент приведён к baseline‑контракту (no-fallback sampling, progress reporting, stage timings, обязательный `dataprocessor_version`, сохранение `times_s`, использование Embedding Service, атомарное сохранение, валидация артефакта).

## Оценки (1–10)

- **Качество кода и алгоритмов**: **8/10**
- **Логика алгоритмов**: **8/10**
- **Логика глобального взаимодействия**: **9/10**
- **Оптимизации (параллелизм, батчинг)**: **7/10**

## ✅ Соответствие требованиям

### 1. Baseline contract (sampling, no-fallback, union-domain)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент **строго** читает `frame_indices` из `metadata.json[core_clip.frame_indices]`
- При отсутствии/пустоте `frame_indices` → **fail-fast** (no‑fallback)
- `frame_indices` — shared sampling group (контракт Segmenter)

### 2. Time axis → `times_s`

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент требует `union_timestamps_sec` в `frames_dir/metadata.json`
- Сохраняет `times_s = union_timestamps_sec[frame_indices]` в артефакт

### 3. Artifact write: atomic save + runtime validation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Запись `franchise_recognition.npz` выполняется атомарно (tmp → `os.replace()`)
- После сохранения выполняется `artifact_validator.validate_npz()`
- При провале валидации — файл удаляется и компонент падает (fail‑fast)

### 4. `dataprocessor_version` всегда присутствует

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если входная meta не содержит `dataprocessor_version`, компонент пишет `"unknown"` (dev/default)
- В production должен передаваться реальный dp version (зафиксировать на уровне пайплайна)

### 5. Artifact filename

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Используется фиксированное имя артефакта: `ARTIFACT_FILENAME = "franchise_recognition.npz"`
- Сохраняется в `result_store/<platform_id>/<video_id>/<run_id>/franchise_recognition/franchise_recognition.npz`

### 6. Stage timings + progress (state_events)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент измеряет время выполнения стадий: `initialization`, `load_deps`, `process_frames`, `saving`, `total`
- Тайминги сохраняются в `meta.stage_timings_ms` (в миллисекундах)
- Компонент публикует прогресс в `state_events.jsonl`:
  - Стадии: `start → load_deps → process_frames → save → done`
  - Гранулярный прогресс во время `process_frames` (≥10 обновлений по кадрам)

### 7. NPZ output schema

**Статус**: ✅ **СООТВЕТСТВУЕТ**

Артефакт: `franchise_recognition.npz` содержит:
- `frame_indices (N,) int32` — shared sampling group
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `semantic_label_names (A,) str` — массив строк "id:name" для маппинга label_id → franchise_name
- `threshold_per_label_arr (A,) float32` — пороги per-label (NaN если нет)
- `track_ids (1,) int32` — video-level aggregate (=0)
- `track_present_mask (1,) bool`
- `track_topk_ids (1, K) int32` — Top‑K франшиз на видео (max over time)
- `track_topk_scores (1, K) float32` — Similarity scores для видео
- `track_is_confident_top1 (1,) bool`
- `track_topk_evidence_frame_indices (1, K) int32` — union frame index, где similarity максимальна
- `frame_topk_ids (N, K) int32` — Top‑K франшиз на кадр
- `frame_topk_scores (N, K) float32` — Similarity scores для кадров
- `frame_is_confident_top1 (N,) bool`
- `meta` (dict, object-array) — canonical meta с `stage_timings_ms`

### 8. Dependencies

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- `core_clip` **обязателен** и должен быть выполнен перед `franchise_recognition` (no-fallback)
- `core_clip` является источником истины для `frame_indices` и frame embeddings
- Embedding Service должен быть доступен (fail-fast при недоступности)
- OCR опционален (не является hard-dependency)

### 9. Embedding Service integration

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент использует Embedding Service для поиска франшиз (категория `franchise`)
- При недоступности Embedding Service → **fail-fast** (error, не empty)
- Используется `EmbeddingServiceClient` для взаимодействия с API
- Поддерживается retry logic с exponential backoff

### 10. Empty semantics

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если Embedding Service недоступен → **error** (fail-fast, не empty)
- Если `core_clip` отсутствует → **error** (fail-fast)
- OCR отсутствует → **не error** (компонент работает через полный поиск)

### 11. Features contract

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент имеет управляемые параметры для тонкой настройки:
  - `--topk` (фиксировано на 5 по контракту)
  - `--similarity-threshold` (не гейтит top-K, только для фильтрации)
  - `--threshold-global` (для `is_confident` флагов)
  - `--use-ocr-filtering` (опционально, для cost control)
  - `--batch-size` (scheduler-controlled)
- Все параметры задокументированы в README

### 12. README документация

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- README содержит раздел **"Sampling / units-of-processing requirements"** с описанием требований к выборке
- README содержит раздел **"Models"** с описанием GPU моделей и внешних сервисов
- README содержит раздел **"Parallelization"** с описанием внутреннего/внешнего параллелизма
- README содержит раздел **"Performance characteristics"** (заполнен предварительными оценками)
- README содержит раздел **"Quality validation & human-friendly inspection"** с описанием методов проверки качества
- README содержит раздел **"Параметры конфигурации компонента"** с таблицей параметров

## 📊 Performance / resource costs (baseline unit-cost)

**Примечание**: Измерения производительности планируются. Компонент использует Embedding Service (HTTP), поэтому latency зависит от сетевых условий и нагрузки на сервис. Также зависит от количества кадров.

**Единица обработки**: `frame` (один кадр)

**Предварительные оценки** (требуют подтверждения измерениями):

| Resolution | Latency per frame | CPU RAM peak | Notes |
|------------|-------------------|--------------|-------|
| 1920x1080 | ~100-200 ms | ~200 MB | зависит от Embedding Service latency |
| 1280x720 | ~100-200 ms | ~200 MB | зависит от Embedding Service latency |

**Для видео с N кадрами**: Total latency ≈ N × latency_per_frame + Embedding Service overhead

**Полные данные**: будут добавлены после проведения измерений в `docs/models_docs/resource_costs/franchise_recognition_costs_v1.json`

## 🔍 Quality validation (минимальный набор)

- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- Проверка валидности NPZ через `artifact_validator.validate_npz()`
- Human-friendly demo: `quality_report/demo_franchise_recognition_quality.py` (обновлен под новые контракты)

**Рекомендуемые проверки**:
- Similarity sanity: similarity scores в диапазоне [0, 1], top-1 обычно > 0.3 для корректных распознаваний
- Стабильность: одинаковые франшизы должны иметь похожие similarity scores на соседних кадрах
- Coverage: проверка, что распознавание покрывает разные части видео (начало/середина/конец)

## Вопросы / открытые решения

1. **Измерения производительности**: Требуется провести измерения latency/RAM для типичных сценариев (зависит от Embedding Service и количества кадров)
2. **Batch API**: Рассмотреть использование batch API Embedding Service для оптимизации (если доступен) — текущая реализация использует последовательный поиск
3. **Оптимизация поиска**: Текущая реализация загружает кадры через FrameManager для поиска в Embedding Service. Возможна оптимизация через прямой поиск по embeddings (если Embedding Service поддерживает embedding-to-embedding search)

## Update: baseline contracts compliance (2026-01-XX)

Изменения:
- Переписан для использования Embedding Service вместо локальной базы данных
- Добавлен `ARTIFACT_FILENAME = "franchise_recognition.npz"`
- Добавлен progress reporting в `state_events.jsonl` (stage-based + granular)
- Добавлен `stage_timings_ms` в `meta`
- Обновлен формат `meta` для соответствия baseline контрактам (producer, producer_version, created_at, run identity keys, dataprocessor_version)
- Использован `apply_models_meta` для правильного формата `models_used`
- Добавлено атомарное сохранение (tmp → `os.replace()`)
- Добавлена валидация артефакта через `artifact_validator.validate_npz()`
- Изменена empty semantics: Embedding Service недоступен → error (не empty)
- Добавлены параметры для тонкой настройки (`--batch-size`, `--use-ocr-filtering`, и др.)
- Обновлен README с полным описанием (sampling, models, parallelization, features, performance)
- Обновлен demo script под новые контракты

---

## Ссылки

- **README компонента**: `VisualProcessor/core/model_process/core_identity/franchise_recognition/README.md`
- **Критерии аудита baseline**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Контракты**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- **Semantic heads schema**: `docs/models_docs/SCHEMA_SEMANTIC_HEADS_NPZ.md`

