# Аудит соответствия core_optical_flow требованиям baseline

**Дата проверки**: 2026-01-14 (обновлено: 2026-01-XX)  
**Компонент**: `core_optical_flow` (Core provider, Tier‑0 baseline)  
**Расположение**: `VisualProcessor/core/model_process/core_optical_flow/`  
**Runtime (prod)**: `triton` (GPU-only)  
**Статус аудита**: ✅ **CLOSED** (исправления внесены: прогресс, stage timings)

## Резюме

`core_optical_flow` вычисляет **кривую движения** (mean optical flow magnitude, нормированную на \(dt\) и размер кадра) на sampled кадрах (union-domain) и пишет артефакт `flow.npz`, который используется downstream модулями:
- `video_pacing` (hard dependency)
- `story_structure` (hard dependency)
- `cut_detection` (optional reuse: `prefer_core_optical_flow` / `require_core_optical_flow`, строго при совпадении `frame_indices`)

Критический для системы момент: **alignment по `frame_indices`**. Downstream компоненты валидируют совпадение и при mismatch падают или делают fallback (в зависимости от модуля/флага).

## Оценки (1–10)

- **Качество кода и алгоритмов**: **7/10**
- **Логика алгоритмов**: **8/10**
- **Логика глобального взаимодействия**: **8/10**
- **Оптимизации (параллелизм, батчинг)**: **8/10**

## ✅ Соответствие требованиям

### 1) Baseline contract (sampling, no-fallback, union-domain)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Строго читает `frame_indices` из `frames_dir/metadata.json[core_optical_flow.frame_indices]`
- `frame_indices` отсутствует/пустой → **error**
- `len(frame_indices) < 2` → **error**

### 2) Time axis → `times_s` (strict, no-fallback)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Требует `union_timestamps_sec` в `frames_dir/metadata.json`
- Сохраняет `times_s = union_timestamps_sec[frame_indices]` в артефакт
- `dt_seconds[i] = max(times_s[i] - times_s[i-1], 1e-6)` (а для первого кадра `NaN`)

### 3) Artifact write: atomic save + runtime validation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Запись `flow.npz` атомарная (tmp → `os.replace`)
- После записи выполняется `artifact_validator.validate_npz()` (fail-fast)
- Валидация строго требует `dataprocessor_version` в meta

### 4) `dataprocessor_version` всегда присутствует

**Статус**: ✅ **СООТВЕТСТВУЕТ (baseline)**  

- Если входная meta не содержит `dataprocessor_version`, компонент пишет `"unknown"`
- В production должен передаваться реальный dp version (зафиксировать на уровне пайплайна/оркестратора)

### 5) Batch size (scheduler-controlled)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Внутренний батчинг по **парам кадров** через `--batch-size` (в одном Triton запросе обрабатывается B пар)
- Для unit-cost тестов выставляем `batch_size=1`

### 6) Triton runtime

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- В режиме `runtime=triton` модель вызывается через Triton HTTP v2
- Baseline контракт на вход (ensemble): `UINT8 NHWC` для `input0/input1`

### 7) NPZ output schema

**Статус**: ✅ **СООТВЕТСТВУЕТ**

Артефакт: `flow.npz` содержит:
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `motion_norm_per_sec_mean (N,) float32`
- `dt_seconds (N,) float32`
- `meta` (dict, object-array)

Schema: `core_optical_flow_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`)

### 8) Параметры конфигурации компонента (1.11)

**Статус**: ✅ **СООТВЕТСТВУЕТ** (исправлено)

**Реализовано**:
- ✅ Добавлен раздел **"Параметры конфигурации компонента"** в README с таблицей всех параметров
- ✅ Описано влияние параметров на скорость и стоимость:
  - Таблица влияния `--triton-preprocess-preset` (raft_256/384/512) с Δ latency и Δ cost относительно baseline
  - Таблица влияния `--batch-size` на latency и VRAM
- ✅ Добавлены примеры конфигурации (минимальная и расширенная) в формате YAML
- ✅ Указаны рекомендации по выбору параметров для разных сценариев использования

**Evidence**: Раздел добавлен в README (после раздела "Models", перед "Parallelization").

### 9) Features contract (1.12)

**Статус**: ✅ **СООТВЕТСТВУЕТ (упрощённый случай)**

Компонент выдаёт **одну основную фичу**: `motion_norm_per_sec_mean` (кривая движения).

**Обоснование**:
- Компонент является простым core provider с единственной целью — вычисление optical flow кривой
- Все выходные данные (frame_indices, times_s, dt_seconds) являются служебными для выравнивания и временной оси
- Явный механизм выбора фич не требуется, так как компонент всегда выдаёт одну фичу

**Вывод**: Для простых core providers с единственной фичей явный features contract не требуется (но должен быть задокументирован в README).

### 10) Промежуточный прогресс (1.13)

**Статус**: ✅ **СООТВЕТСТВУЕТ** (исправлено)

**Реализовано**:
- Добавлены функции `_append_state_event_if_possible()`, `_emit_stage()`, `_emit_progress()` (аналогично `core_clip`, `core_depth_midas`)
- Компонент репортит стадии: `start → load_deps → process_frames → post_process → save → done`
- Для `process_frames` отправляется гранулярный прогресс (не менее ~15 обновлений на видео): `progress ∈ [0,1]`, `done`, `total`
- Прогресс записывается в `state_events.jsonl` по пути `runs/state/<platform>/<video>/<run>/state_events.jsonl`

**Evidence**: Функции добавлены в `main.py` (строки 73-135), интегрированы в основной цикл обработки (строки 264-384, 450-456).

### 11) Профилирование по стадиям (stage timings) (1.14)

**Статус**: ✅ **СООТВЕТСТВУЕТ** (исправлено)

**Реализовано**:
- Компонент измеряет время ключевых стадий через `time.perf_counter()`
- Сохраняет тайминги в `meta.stage_timings_ms` (dict с миллисекундами)
- Измеряются стадии: `initialization`, `flow_inference_total`, `saving`, `total`

**Evidence**: Измерение времени добавлено в `main.py` (строки 220-222, 260-261, 338-339, 365-366, 420, 440-442, 470-472), сохранение в meta (строка 458).

## 🔗 Взаимосвязи / downstream contract

### Alignment requirement

- `video_pacing` требует, чтобы `core_optical_flow.frame_indices` покрывали `video_pacing.frame_indices` (иначе error).
- `story_structure` выравнивает по `frame_indices` (и падает, если не может).
- `cut_detection` может reuse motion curve **только если** `core_optical_flow.frame_indices` ровно совпали с `cut_detection.frame_indices`.

Вывод: Segmenter должен включать `core_optical_flow` в **shared primary sampling group** с компонентами, где нужен строгий alignment (уже реализовано в Segmenter policy A).

## 🔍 Quality validation (минимальный набор)

Минимальные sanity checks (можно автоматизировать в demo/скрипте):
- `times_s` монотонен, `dt_seconds[1:] > 0`, `dt_seconds[0]=NaN`
- `motion_norm_per_sec_mean[0]=0` и конечен для остальных
- “разумность” кривой: на статичных участках близко к 0, на динамичных пик больше (на глаз)

Human-friendly demo (evidence):
- Script: `scripts/baseline/demo_core_optical_flow_quality.py`
- Example HTML: `storage/reports/out/core_optical_flow_quality_demo_NSumhkOwSg.html`

## 📊 Performance / resource costs (baseline unit-cost)

**Источник данных**: `docs/models_docs/resource_costs/core_optical_flow_costs_v1.json`  
**Единица обработки**: `frame_pair` (одна пара соседних sampled кадров)

**Типичные значения (preset="raft_256", batch_size=1)**:

| Resolution preset | Latency per unit | CPU RAM peak | GPU VRAM peak (Triton) | Notes |
|-------------------|------------------|--------------|------------------------|-------|
| raft_256 | ~213 ms | ~75 MB | ~1012 MB | стабильно, рекомендуется для baseline |
| raft_384 | ~440 ms | ~101 MB | ~1168 MB | баланс качества/скорости |
| raft_512 | ~743 ms | ~135 MB | ~3642 MB | spikes=true, VRAM drift на 6GB GPU |

**Для видео с N кадрами**: Total latency ≈ (N-1) × latency_per_unit (так как первый кадр имеет motion=0)

**Полные данные**: см. `docs/models_docs/resource_costs/core_optical_flow_costs_v1.json` (B=1) и `core_optical_flow_costs_b8_v1.json` (B=8)

---

**Источник**: `docs/models_docs/resource_costs/core_optical_flow_costs_v1.json` (unit-cost, `model-batch-size=1`)  
**Evidence**: `storage/reports/out/checklist-raft-b1/`

Ключевые наблюдения (local RTX 2060 6GB, Triton):
- `raft_256`: ~213ms per frame_pair (стабильно)
- `raft_384`: ~440ms per frame_pair
- `raft_512`: ~743ms per frame_pair, **spikes=true** и отмечен VRAM drift → может требовать restart Triton между группами / аккуратного использования на 6GB

### Таблица unit-cost (B=1)

VRAM в baseline отчётах фиксируется **по процессу `tritonserver`**:
- `vram_triton_peak_mb`: пик VRAM у `tritonserver` во время замера (MB)
- `vram_triton_delta_run_mb`: \(\Delta\) = `peak - before` для конкретного прогона (MB)

| Branch | unit | mean_stable_ms | p95_ms | cpu_rss_peak_mb | vram_triton_peak_mb | vram_triton_delta_run_mb | spikes |
|--------|------|----------------|--------|------------------|---------------------|--------------------------|--------|
| 256 | frame_pair | ~213.2 | ~231.5 | ~75.0 | ~1012 | ~794 | false |
| 384 | frame_pair | ~439.8 | ~458.4 | ~101.1 | ~1168 | ~882 | false |
| 512 | frame_pair | ~742.9 | ~762.4 | ~135.1 | ~3642 | ~3232 | true |

Где смотреть “сырые” значения:
- `storage/reports/out/checklist-raft-b1/SUMMARY.md` (строки `vram_delta_run`, `drift`, `restart`)
- `storage/reports/out/checklist-raft-b1/checklist_micro_results.json` → `models_triton.raft_*/vram_*`

### Таблица unit-cost (B=8) + заметки по запуску Triton

Источник: `docs/models_docs/resource_costs/core_optical_flow_costs_b8_v1.json` (derived unit-cost, `model-batch-size=8`)  
Evidence:
- `storage/reports/out/checklist-raft-b8/`
- `storage/reports/out/checklist-raft-512-b8/`

Важно:
- Для `raft_512` при `B=8` Triton внутри Docker требует увеличенного shared memory, иначе возможна ошибка вида `Failed to increase the shared memory pool size ... No space left on device`.
- Рабочий запуск (пример): `docker run ... --shm-size=1g ... tritonserver --model-repository=/models`

| Branch | unit | mean_stable_ms_per_unit | p95_ms_per_unit | cpu_rss_peak_mb | vram_triton_peak_mb | vram_triton_delta_run_mb | vram_triton_drift_mb | spikes | restart_recommended |
|--------|------|--------------------------|-----------------|------------------|---------------------|--------------------------|----------------------|--------|---------------------|
| 256 | frame_pair | ~181.6 | ~190.6 | ~197.1 | ~1274 | ~1072 | ~260 | true | true |
| 384 | frame_pair | ~424.2 | ~436.0 | ~378.6 | ~2372 | ~1910 | ~1098 | false | true |
| 512 | frame_pair | ~760.7 | ~784.0 | ~548.2 | ~4310 | ~4108 | ~4108 | false | true |

Где смотреть “сырые” значения:
- `storage/reports/out/checklist-raft-b8/SUMMARY.md`, `.../checklist_micro_results.json`
- `storage/reports/out/checklist-raft-512-b8/SUMMARY.md`, `.../checklist_micro_results.json`

## 📋 Итоговая оценка соответствия

### Чек-лист архитектурных требований

| Критерий | Статус | Примечание |
|----------|--------|------------|
| 1.1 Наследование и интерфейсы | ✅ | CLI интерфейс через argparse |
| 1.2 Контракты входа/выхода | ✅ | Читает frame_indices из metadata, использует union_timestamps_sec |
| 1.3 No-fallback policy | ✅ | Строгие проверки с raise RuntimeError |
| 1.4 Per-run storage | ✅ | Сохраняет в result_store/<platform>/<video>/<run>/core_optical_flow/ |
| 1.5 Валидация артефактов | ✅ | Проходит artifact_validator.validate_npz() |
| 1.6 Valid empty outputs | ✅ | Empty недопустим (компонент всегда должен вычислить кривую) |
| 1.7 Документация sampling requirements | ✅ | Есть в README, раздел "Sampling / units-of-processing requirements" |
| 1.8 Документация моделей | ✅ | Есть в README, раздел "Models" |
| 1.9 Документация параллелизма | ✅ | Есть в README, раздел "Parallelization" |
| 1.10 Batching / scheduler contract | ✅ | Batch size контролируется верхним scheduler |
| 1.11 Параметры конфигурации | ✅ | Раздел добавлен в README с таблицей и влиянием на стоимость (исправлено) |
| 1.12 Features contract | ✅ | Упрощённый случай (одна фича) |
| 1.13 Промежуточный прогресс | ✅ | Репортит прогресс в state_events.jsonl (исправлено) |
| 1.14 Профилирование по стадиям | ✅ | Измеряет stage_timings_ms (исправлено) |

### Процент соответствия

**Соответствие архитектурным требованиям**: **100%** (14/14 критериев)

**Критические проблемы**: ✅ **Исправлены**
- ✅ Репортинг прогресса реализован (1.13)
- ✅ Stage timings реализованы (1.14)

**Важные улучшения** (желательно для полного baseline):
- ⚠️ Добавить раздел "Параметры конфигурации" в README с таблицей и влиянием на стоимость

## 🔧 План действий

### ✅ Критично (исправлено)

1. ✅ **Добавлен репортинг прогресса**:
   - Реализованы функции `_append_state_event_if_possible()`, `_emit_stage()`, `_emit_progress()` (аналогично `core_clip`, `core_depth_midas`)
   - Репортируются стадии: `start → load_deps → process_frames → post_process → save → done`
   - Для `process_frames` отправляется гранулярный прогресс (не менее ~15 обновлений)

2. ✅ **Добавлено измерение stage timings**:
   - Измеряется время стадий: `initialization`, `flow_inference_total`, `saving`, `total`
   - Сохраняется в `meta.stage_timings_ms` (dict с миллисекундами)

### ✅ Важно (исправлено)

3. ✅ **Добавлен раздел "Параметры конфигурации" в README**:
   - Таблица всех параметров с описанием, значениями по умолчанию
   - Влияние на скорость (Δ latency ms/frame_pair) и стоимость (Δ cost относительно baseline)
   - Примеры блоков конфигурации (минимальная и расширенная)

## Вопросы / открытые решения

Решения зафиксированы (по ответам владельца):

1) **Норма кривой**: оставляем текущую (mean magnitude / dt / max(H,W)) как baseline-универсальную.
2) **Нулевая точка**: сохраняем baseline семантику `motion_norm_per_sec_mean[0]=0`, `dt_seconds[0]=NaN`.
3) **Shared sampling group**: обеспечиваем строгий alignment через Segmenter policy (см. ниже).
4) **cut_detection**: `core_optical_flow` считается **обязательным**; fallback запрещён в baseline.
5) **Triton presets**: baseline-обязательные пресеты `raft_256`, `raft_384`, `raft_512`.
6) **Batch sizes**: целевые значения зависят от unit-cost/VRAM, требуется тестирование.
7) **Triton batching**: модели должны быть batch-enabled (`batch_size>=1`) аналогично CLIP.
8) **Meta**: текущего meta достаточно (доп. поля не требуются, кроме `stage_timings_ms`).

### Update: cut_detection требует core_optical_flow (2026-01-14)

Изменение:
- `cut_detection` переведён на baseline-политику **require_core_optical_flow=true by default**.
- Segmenter гарантирует **строгое равенство** `cut_detection.frame_indices == core_optical_flow.frame_indices` (иначе reuse невозможен).

## 📚 Ссылки

- **README компонента**: `VisualProcessor/core/model_process/core_optical_flow/README.md`
- **Критерии аудита**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Контракты**: 
  - `docs/contracts/ARTIFACTS_AND_SCHEMAS.md` (NPZ meta, schema_version)
  - `docs/contracts/SEGMENTER_CONTRACT.md` (sampling, union_timestamps_sec)
- **Resource costs**: 
  - `docs/models_docs/resource_costs/core_optical_flow_costs_v1.json` (B=1)
  - `docs/models_docs/resource_costs/core_optical_flow_costs_b8_v1.json` (B=8)
- **Human-friendly demo**: `VisualProcessor/core/model_process/core_optical_flow/quality_report/demo_core_optical_flow_quality.py`


