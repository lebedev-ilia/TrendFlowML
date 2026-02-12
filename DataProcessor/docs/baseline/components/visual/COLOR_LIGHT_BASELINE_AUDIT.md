# ✅ Baseline Audit — `color_light`

Компонент: `DataProcessor/VisualProcessor/modules/color_light/`  
Тип: Visual module (Tier‑0 baseline)  
Статус: **✅ CLOSED (baseline)** (2026-01-30)

---

## Резюме

`color_light` — модуль для комплексного анализа цвета и освещения видео. Извлекает покадровые (frame-level), сценовые (scene-level) и видеоуровневые (video-level) признаки для анализа визуального стиля, цветокоррекции и качества освещения.

**Текущее состояние**: компонент приведён к baseline‑контракту (no-fallback sampling, atomic save, progress reporting, stage timings, обязательный `dataprocessor_version`, сохранение `times_s`, UI payload, все обязательные meta поля).

## Оценки (1–10)

- **Качество кода и алгоритмов**: **8/10**
- **Логика алгоритмов**: **8/10**
- **Логика глобального взаимодействия**: **9/10**
- **Оптимизации (параллелизм, батчинг)**: **7/10**

## ✅ Соответствие требованиям

### 1. Baseline contract (sampling, no-fallback, union-domain)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент **строго** читает `frame_indices` из `metadata.json[color_light.frame_indices]`
- При отсутствии/пустоте `frame_indices` → **fail-fast** (no‑fallback)
- `frame_indices` — union-domain (контракт Segmenter)
- Использует `scene_classification` для группировки кадров по сценам, но не пересэмплирует кадры

### 2. Time axis → `times_s` (strict, no-fallback)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент требует `union_timestamps_sec` в `frames_dir/metadata.json`
- Сохраняет `times_s = union_timestamps_sec[frame_indices]` в артефакт
- Также сохраняет `sequence_times_s` для sequence inputs

### 3. Artifact write: atomic save + runtime validation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Запись `color_light_features.npz` выполняется атомарно (tmp → `os.replace`)
- После сохранения выполняется `artifact_validator.validate_npz()`
- При провале валидации — файл удаляется и компонент падает (fail‑fast)

### 4. `dataprocessor_version` всегда присутствует

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- `dataprocessor_version` всегда присутствует в meta (default: `"unknown"` если не указан в metadata)
- Значение берётся из `metadata.dataprocessor_version` или устанавливается в `"unknown"`

### 5. Обязательные meta поля

**Статус**: ✅ **СООТВЕТСТВУЕТ**

Все обязательные поля присутствуют:
- **Базовые**: `producer`, `producer_version`, `schema_version`, `created_at`
- **Run identity**: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- **Версия пайплайна**: `dataprocessor_version` (обязательно)
- **Статус**: `status` (`ok`/`empty`/`error`), `empty_reason` (если `status="empty"`)
- **Модели**: `models_used[]` (пустой список, т.к. модели не используются), `model_signature`

### 6. Progress reporting & stage timings

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент публикует прогресс в `state_events.jsonl` (best-effort)
- Прогресс обновляется **не реже 15 раз за run** (каждые ~6.7% обработанных кадров)
- Стадии явно выделены: `start` → `initialization` → `load_deps` → `process_frames` → `post_process` → `save` → `done`
- `stage_timings_ms` сохраняется в `meta.summary.stage_timings_ms`:
  - `initialization` — инициализация модуля
  - `load_deps` — загрузка зависимостей (`scene_classification`)
  - `process_frames` — извлечение признаков по кадрам
  - `post_process` — агрегация video-level фич и формирование sequences
  - `save` — сохранение NPZ
  - `total` — общее время работы компонента

### 7. UI payload

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент строит `meta.ui_payload` через `_build_ui_payload()` → `build_presentation()`
- UI payload содержит:
  - `schema_version`: `color_light_presentation_v1`
  - `run_identity`: platform_id, video_id, run_id, config_hash, sampling_policy_version
  - `summary`: ключевые метрики (color_distribution_entropy, cinematic_lighting_score, etc.)
  - `style_probs`: вероятности стилей цветокоррекции
  - `timeline`: временные ряды признаков с гистограммами и sparklines

### 8. Valid empty outputs

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если после фильтрации по сценам нет ни одного кадра → `status="empty"`, `empty_reason="after_filt_empty"`
- Численные массивы содержат `NaN` (не нули, не пустые массивы)
- `empty_reason` обязателен если `status="empty"`, иначе `null`

### 9. Models

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент **не использует модели** (только CPU вычисления на основе OpenCV/scikit-learn)
- Метод `get_models_used()` реализован и возвращает пустой список
- Эстетические модели (NIMA/LAION) планируются к интеграции, но пока не подключены (значения будут NaN)

### 10. README документация

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- ✅ Раздел "Sampling / units-of-processing requirements" — описан контракт Segmenter
- ✅ Раздел "Models" — указано что модели не используются (CPU-only)
- ✅ Раздел "Parallelization" — описаны внутренний и внешний параллелизм
- ✅ Раздел "Performance characteristics" — указан источник данных (resource_costs)
- ✅ Раздел "Quality validation & human-friendly inspection" — описан скрипт демо

## 📊 Performance / resource costs (baseline unit-cost)

**Источник данных**: `docs/models_docs/resource_costs/color_light_costs_v1.json`  
**Единица обработки**: `frame`  
**Статус**: ⚠️ **Требуется измерение** latency/RAM/VRAM на типичных разрешениях

**TODO**: провести измерения и заполнить `resource_costs/color_light_costs_v1.json`

## 🔍 Quality validation (минимальный набор)

### Human-friendly визуализация

**Скрипт**: `VisualProcessor/modules/color_light/quality_report/demo_color_light_quality.py`

```bash
python VisualProcessor/modules/color_light/quality_report/demo_color_light_quality.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store/<platform>/<video>/<run> \
  --out-html /path/to/color_light_quality.html
```

**Что визуализируется**:
- Timeline с признаками цвета и освещения
- Распределения значений признаков
- Графики временных рядов (hue, colorfulness, brightness, contrast)
- Стили цветокоррекции (Teal & Orange, Film, Vintage, TikTok)

### Статистическая валидация

**Диапазоны значений** (см. `FEATURES_DESCRIPTION.md`):
- `hue_mean_norm`: [0, 1]
- `colorfulness_norm`: [0, 1]
- `global_contrast_norm`: [0, 1]
- `style_*_prob`: [0, 1]
- `cinematic_lighting_score`: [0, 1] или NaN (если модели не подключены)

## Известные ограничения / next steps

1. **Эстетические модели**: NIMA/LAION не подключены (значения NaN). Планируется интеграция через ModelManager/Triton.
2. **Производительность**: требуется измерение latency/RAM/VRAM и заполнение `resource_costs/color_light_costs_v1.json`.
3. **Батчинг**: внутренний батчинг вычислений по кадрам не реализован (TODO: оптимизация для больших видео).

## Ссылки

- **README**: `VisualProcessor/modules/color_light/README.md`
- **Feature description**: `VisualProcessor/modules/color_light/FEATURES_DESCRIPTION.md`
- **Контракты**: 
  - `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
  - `docs/contracts/SEGMENTER_CONTRACT.md`
  - `docs/contracts/PER_COMPONENT.md`
- **Baseline criteria**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Quality demo**: `VisualProcessor/modules/color_light/quality_report/demo_color_light_quality.py`
