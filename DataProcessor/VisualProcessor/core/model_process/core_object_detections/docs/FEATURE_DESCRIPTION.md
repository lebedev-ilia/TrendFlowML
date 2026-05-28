# core_object_detections — описание фич (Audit v2/v3)

**Компонент:** `core_object_detections` (VisualProcessor core)  
**producer** в `meta` NPZ: `core_object_detections`  
**producer_version (код `main.py`):** `2.2`  
**schema_version NPZ:** `core_object_detections_npz_v2`  
**Артефакт:** `<result_store>/<platform_id>/<video_id>/<run_id>/core_object_detections/detections.npz` (см. [README.md](../README.md))

## Назначение

Детекция объектов (YOLO / ultralyics или Triton) по семплированным кадрам: боксы, скоры, классы по таксономии v1 (41 класс, см. `TAXONOMY_V1.yaml` / `DETECTOR_TAXONOMY_V1_40_NAMES.txt`), фиксированный слот **M = 100** (`MAX_DETECTIONS`) на кадр. **Persistent track id нет** — адресация `(кадр N, слот m)`.

## Ключи NPZ (сводка)

| Группа | Ключи |
|--------|--------|
| Ось кадра **N** | `frame_indices`, `times_s` (float32) |
| Детекции **(N, M, …)** | `boxes`, `boxes_norm`, `centers_norm`, `areas_frac`, `scores`, `class_ids`, `valid_mask` |
| Таксономия | `class_names` — `(41,)` строки вида `id:name` |
| Аналитика по кадру **(N,)** | `det_count`, `person_count`, `text_region_count`, `logo_region_count`, `sum_*_area_frac`, `max_*_area_frac` (person / text / logo) |
| Служебное | `meta` (dict), `meta_json` (JSON-строка) |

## Meta (сводка)

- Run: `platform_id`, `video_id`, `run_id`, `sampling_policy_version`, `config_hash`, `dataprocessor_version`  
- Модель: `impl` (`yolo` или `triton:...`), `model` (путь/спек), `box_threshold`, `batch_size`, `device`  
- Счётчики: `total_frames` (в источнике), `total_detections` (суммарно валидных детекций)  
- `stage_timings_ms`: `initialization`, `load_deps`, `process_frames`, `saving`, `total` (мс) → в плоском CSV `meta_timing_*` без суффикса `_ms` в имени ключа внутри dict, в колонках — `meta_timing_initialization` и т.д. (как в `component_feature_qa.flatten_meta`).

**Пусто:** `status=empty`, `empty_reason=no_detections_above_threshold` (все кадры ниже порога).

## Нормальные диапазоны (для QA / `--ranges` в валидаторе)

| Поле / группа | Ожидание |
|---------------|-----------|
| `valid_mask` | по оси M ровно **100** слотов; на валидных слотах `scores` **∈ [0, 1]**, `class_ids` **∈ [0, 40]** |
| `det_count` | для каждого кадра **=** `sum(valid_mask, axis=1)` |
| `boxes_norm` | компоненты **∈ [0, 1]** на валидных слотах (норм. xyxy) |
| `centers_norm` | **∈ [0, 1]** |
| `areas_frac` | **∈ [0, 1]** |
| `person_count`, `text_region_count`, `logo_region_count` | **≤** `det_count` (по кадру) |
| `sum_*_area_frac`, `max_*_area_frac` | **∈ [0, 1]** (доля площади кадра) |
| `meta.box_threshold` | **∈ [0, 1]** (дублируется в wide CSV как `meta_box_threshold`) |
| `meta.total_detections` | **=** `sum(valid_mask)` (и **=** `sum(det_count)`), при наличии в meta |
| `times_s` | Неубывающий ряд (ось кадра по union) |
| `len(frame_indices)` vs `meta.total_frames` | **N ≤ total_frames** (семпл — подмножество union) |
| `meta.stage_timings_ms` | ключи: `initialization`, `load_deps`, `process_frames`, `saving`, `total` (мс) → в CSV `meta_timing_<ключ>` (без суффикса `_ms` в имени ключа) |

Синхрон: **`view_csv_feature_qa.json`** (правила meta / QA) + **`view_csv_melt_interesting.json`** (колонки melt) + **`view_csv_feature_descriptions_ru.json`** (подписи в wide/melt-отчётах, в т.ч. HTML-таблицы) → компонент `core_object_detections`.

## Валидатор

Из корня репозитория (нужен `numpy`; в проекте удобно venv `DataProcessor/VisualProcessor/.vp_venv`):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_object_detections/utils/validate_core_object_detections_npz.py \
  <path/to/detections.npz> [--struct] [--qa] [--ranges]
```

- `--struct` — ключи, N, M=100, 41 class name, `meta_json`  
- `--qa` — плоский `meta` + `view_csv_feature_qa.json`  
- `--ranges` — согласованность счётчиков, `meta.total_detections` vs `sum(valid_mask)`, `N` vs `meta.total_frames`, `times_s`, диапазоны массивов (см. таблицу выше)  

Пакетный обход (только struct-эквивалент, без `--qa`):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_object_detections/utils/validate_core_object_detections_npz.py \
  --results-base /path/to/storage/result_store --platform-id youtube
```

## Сверка с реальным прогоном (2026)

Проверено на артефакте:

`storage/result_store/youtube/-5EYUqIlyJU/0dfdf2fd-360f-49c1-8f75-7845a0189461/core_object_detections/detections.npz`

- **Ключи** в файле совпадают с контрактом (`areas_frac` … `valid_mask`, плюс `meta` / `meta_json`).  
- **`python3 … --struct --qa --ranges`**: схема OK, структура OK, диапазоны OK, QA OK (`view_csv_feature_qa.json` → компонент `core_object_detections`).

Wide/melt: колонки `meta_*` соответствуют `view_csv_melt_interesting.json` → `core_object_detections` и подсказкам в `view_csv_feature_descriptions_ru.json` (в т.ч. per-array поля при melt-развёртке, если включена в пайплайне отчёта).

## CSV / wide-отчёт

Колонки: `meta_producer`, `meta_batch_size`, `meta_box_threshold`, `meta_device`, `meta_impl`, `meta_model`, `meta_model_signature`, `meta_total_detections`, `meta_total_frames`, run identity, тайминги. Пример строки: batch-отчёты `csv/core_object_detections.csv` при наличии.

## Схема

Полная таблица полей: [SCHEMA.md](SCHEMA.md), JSON: `DataProcessor/VisualProcessor/schemas/core_object_detections_npz_v2.json` (в репозитории). Поведение и ключи: [README.md](../README.md).
