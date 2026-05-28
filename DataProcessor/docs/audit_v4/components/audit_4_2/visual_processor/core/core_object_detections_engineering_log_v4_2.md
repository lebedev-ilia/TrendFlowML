# Audit v4.2 — engineering log: `core_object_detections`

**Дата:** 2026-04-13  
**Компонент:** `core_object_detections` (VisualProcessor core)  
**Цель:** довести отчёт Audit v4 по компоненту до **L2 (A+B)** и добавить наблюдаемость ресурсов без изменения контракта.

## Изменения кода (после L1)

### 1) Env-gated resource profiling (RSS + CUDA)

Добавлено best-effort поле `meta.resource_profile_before`, которое записывается **только** при включении переменной окружения:

- `VP_RESOURCE_PROFILE=1|true|yes|y|on` → в meta появится `resource_profile_before`
- иначе поле отсутствует

Содержимое (best-effort):

- `rss_bytes`, `rss_mib` (через `psutil`)
- `cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes` (через `torch.cuda`, если доступно)

Файл:

- `DataProcessor/VisualProcessor/core/model_process/core_object_detections/main.py`

## L2 статистика (A+B, 5 прогонов)

JSON:

- `storage/audit_v4/core_object_detections_l2/core_object_detections_audit_v4_stats.json`

Ключевые итоги (по агрегатам JSON):

- `N_total=543`, `M_set=[100]`, `class_names_len_set=[41]`
- `det_count_sum_total=2299`, `det_count_max_max=19`, `det_count_matches_mask_all=true`
- `valid_mask_true_ratio` в диапазоне **~0.02375 … 0.05184**
- Разделение по порогу `box_threshold≈0.6` подтверждено эмпирически:
  - `score_valid_min_min≈0.60001`, `score_valid_max_max≈0.97930`
  - `score_invalid_max_max≈0.59886`

## Наблюдения/риски для downstream

- `boxes`/`scores`/`class_ids` в слотах с `valid_mask=False` могут содержать «мусорные» значения (не нули). **Единственный источник истины — `valid_mask`**.

## Что осталось (DoD)

- Набор **C** (edge) + **§4.8 golden**: TODO (зафиксировать «golden signature» по A и минимальный набор инвариантов).

