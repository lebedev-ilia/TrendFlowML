# `video_pacing` (Visual module, Tier‑0 baseline)

- **Контракт NPZ, melt/QA, валидатор:** [docs/FEATURE_DESCRIPTION.md](docs/FEATURE_DESCRIPTION.md)
- **Валидатор:** `utils/validate_video_pacing.py` (`--struct`, `--qa`, `--ranges` или батч `--results-base`)

Модуль считает **признаки темпа/монтажа** (shot pacing) и связанные метрики движения/семантических/цветовых изменений **строго на sampled кадрах** от Segmenter.

## Входы

### Основной вход
- **`frames_dir`**: директория Segmenter с `metadata.json` и батчами кадров.
- **`metadata["video_pacing"]["frame_indices"]`**: список индексов в **union-domain** (0..N-1) — кадры, которые обрабатывает модуль.

### Time-axis (обязательно)
- **`metadata["union_timestamps_sec"]`**: список timestamp’ов (сек) для каждого union-кадра. Это **source-of-truth** для времени (монтаж/окна/скорости).

### Зависимости (hard deps, no-fallback)
Модуль не имеет права “молча деградировать”: если зависимости отсутствуют/не покрывают `frame_indices` → **error**.

- **`cut_detection`**: результаты модуля `cut_detection` (shot boundaries как source-of-truth).
- **`core_optical_flow`**: `result_store/.../core_optical_flow/flow.npz`
  - используется `motion_norm_per_sec_mean` (кривая движения).
- **`core_clip`**: `result_store/.../core_clip/embeddings.npz`
  - используется `frame_embeddings` (семантические эмбеддинги).

## Выход (артефакт)

Пишется через `BaseModule.save_results()` в:
- `result_store/<platform_id>/<video_id>/<run_id>/video_pacing/video_pacing_features.npz` (**фиксированное имя**)

- **Сводка полей, meta → CSV, melt/QA:** `docs/FEATURE_DESCRIPTION.md`

### Ключи NPZ
- **`frame_indices`**: `(N,) int32` — union-domain индексы кадров модуля.
- **`times_s`**: `(N,) float32` — `union_timestamps_sec[frame_indices]` (source-of-truth).
- **`shot_boundary_frame_indices`**: `(S,) int32` — union-domain индексы кадров, являющихся **началом** нового шота (первый элемент обычно равен `frame_indices[0]`).
- **`motion_norm_per_sec_mean`**: `(N,) float32` — motion curve aligned to `frame_indices` (from `core_optical_flow`).
- **`semantic_change_rate_per_sec`**: `(N,) float32` — semantic change rate (/s) aligned to `frame_indices` (from `core_clip`).
- **`color_change_rate_per_sec`**: `(N,) float32` — color change rate (/s) aligned to `frame_indices` (cheap LAB proxy on downscaled frames).
- **`feature_names`**: `(F,) object` — имена агрегированных model-facing scalar фич (фиксированный порядок).
- **`feature_values`**: `(F,) float32` — значения scalar фич (bool как 0/1).
- **`meta`**: `object(dict)` — canonical meta от `BaseModule` (run identity keys, schema/producer versions, models_used/model_signature, status/empty_reason и т.д.).
  - `meta.schema_version = "video_pacing_npz_v3"`

## No-fallback / empty semantics

- **No-fallback**:
  - нет `frame_indices` / нет `union_timestamps_sec` / time-axis не монотонна;
  - нет `cut_detection` / нет `cut_detection.detections.shot_boundaries_frame_indices`;
  - нет `core_clip` / нет `core_optical_flow`;
  - `core_*` не покрывают `frame_indices` (несогласованный sampling между компонентами).
- **Empty outputs**: не предусмотрены для baseline; “пусто” считается ошибкой входа (например, `frame_indices` пустой).

## Параметры (config)

Передаются через `config` (CLI: `VisualProcessor/modules/video_pacing/main.py`):
- **`downscale_factor`** (`float`, default `0.25`): downscale для дешёвых визуальных метрик (shot detection / color / lighting).
- **`min_shot_length_seconds`** (`float`, default `0.15`): минимальная длительность шота (в секундах) для merge слишком коротких шотов.
- **`shot_detect_k`** (`float`, default `6.0`): множитель для робастных порогов (MAD) в shot boundary detection.
- **`min_frames`** (`int`, default `30`): fail-fast минимум кадров (no-fallback).
- **Feature flags** (в `global_config.yaml` под `video_pacing.feature_flags`, по умолчанию `false`):
  - `enable_entropy_features`: включить энтропии/Gini (может быть шумно при малом числе шотов)
  - `enable_histograms`: включить histogram-based pacing features (может быть шумно)
  - `enable_pace_curve_peaks`: включить peak features по длительностям шотов (может быть шумно)
  - `enable_periodicity`: включить autocorr periodicity features (может быть шумно)
  - `enable_bursts`: включить burst features (quick cuts / semantic / color bursts)

## Фичи (model-facing scalars, `feature_names/feature_values`)

Примечание: часть фич считается “шумной” и по умолчанию выключена (см. флаги `enable_*` выше).

### A) Shot statistics (в секундах)
- **`shots_count`**: число шотов.
- **`shot_duration_mean`**, **`shot_duration_median`**, **`shot_duration_min`**, **`shot_duration_max`**, **`shot_duration_std`**
- **`shot_duration_entropy`**: энтропия распределения длительностей (20 бинов).
- **`shot_duration_mean_normalized`**: `mean / video_length_seconds`.
- **`shot_length_gini`**: Джини по длительностям шотов.
- **`short_shot_fraction`**: доля шотов короче 0.5s.
- **`quick_cut_burst_count`**: число “бурстов” ≥3 cut’ов в окне 1s.
- **`shot_length_histogram_5bins`**: 5‑мерный вектор долей шотов по бинам длительности `[0–0.3, 0.3–0.7, 0.7–1.5, 1.5–3.0, >3.0]`.
- **`tempo_entropy`**: энтропия распределения длительностей по 5 бинам.
- **`cuts_variance`**: дисперсия длительностей шотов (sec²).
- **`cuts_per_10s`**, **`cuts_per_10s_max`**, **`cuts_per_10s_median`**: частота cut’ов (в окнах 10s; значения в 1/sec).
- **`cut_density_map_8bins`**: 8‑мерный вектор плотности cut’ов по 8 равным временным сегментам (в 1/sec).

### B) Pace curve (pattern)
- **`pace_curve_slope`**: тренд по последовательности `log1p(shot_duration_sec)` (линейная регрессия по номеру шота).
- **`pace_curve_slope_normalized`**: `pace_curve_slope * mean(shot_duration_sec)` (масштабирование).
- **`pace_curve_peaks`**, **`pace_curve_peaks_mean_prominence`**, **`pace_curve_peak_positions`**
- **`pace_curve_dominant_period_sec`**, **`pace_curve_power_at_period`**: периодичность по автокорреляции.

### C) Motion (from `core_optical_flow`)
Все значения считаются по кривой `motion_norm_per_sec_mean` (уже нормализована на dt/max(H,W) в core provider).
- **`mean_motion_speed_per_shot`**, **`motion_speed_median`**, **`motion_speed_variance`**, **`motion_speed_90perc`**
- **`share_of_high_motion_frames`**: доля кадров выше 75‑го перцентиля.
- **`share_of_high_motion_shots`**: доля шотов с высокой средней скоростью.
- **`motion_shot_corr`**: корреляция (Пирсон) между длительностью шота и его средней “motion speed”.

### D) Content change rate (from `core_clip`, per-second)
CLIP cosine distance между соседними кадрами, нормализованная на \(dt\) (`union_timestamps_sec`).
- **`frame_embedding_diff_mean`**, **`frame_embedding_diff_std`**
- **`high_change_frames_ratio`**: доля переходов выше 75‑го перцентиля.
- **`scene_embedding_jumps`**: число переходов выше `mean + 2σ`.
- **`semantic_change_burst_count`**: число “бурстов” ≥3 high-change переходов в окне 5s.

### E) Color pacing (per-second)
DeltaE(LAB) между соседними кадрами, нормализованная на \(dt\).
- **`color_change_rate_mean`**, **`color_change_rate_std`**
- **`color_change_bursts`**: пики detrended‑скорости DeltaE.
- **`saturation_change_rate`**, **`brightness_change_rate`**: std от \(\Delta S / dt\), \(\Delta V / dt\) (HSV).

### F) Lighting pacing
- **`luminance_spikes_per_minute`**: количество резких изменений яркости в минуту (по робастному порогу на \(\Delta L / dt\)).

### G) Structural pacing
Медианная длительность шота (sec) в четвертях видео (по последовательности шотов).
- **`intro_speed`**, **`main_speed`**, **`climax_speed`**
- **`pacing_symmetry`**: `(climax - intro) / median_overall`.

## Производительность

- CPU‑heavy: требует чтения sampled кадров и вычисления простых метрик; сложность \(O(N)\) по числу sampled кадров.
- Память: хранит только небольшие массивы переходов/агрегатов, без сохранения пер‑кадровых эмбеддингов (они живут в `core_clip`).

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/video_pacing_costs_v1.json`  
**Единица обработки**: `frame` (per sampled frame)

Типичные значения (preset="default", 720p):

| Resolution | Latency per frame | CPU RAM peak | Notes |
|------------|-------------------|--------------|-------|
| 1280x720 | ~48 ms | ~565 MB | measured by component micro-bench |

Полные данные: см. `docs/models_docs/resource_costs/video_pacing_costs_v1.json`.

## Sampling / units-of-processing requirements

- **Sampling owner**: Segmenter (модуль не генерирует sampling сам).
- **Единица обработки**: один sampled кадр (но часть метрик строится по переходам между соседними sampled кадрами).
- **Alignment policy**:
  - `video_pacing.frame_indices` должны быть **подмножеством** `core_clip.frame_indices` и `core_optical_flow.frame_indices`.
  - Если зависимости не покрывают `frame_indices` → **error** (no-fallback).
- **Рекомендовано для baseline Tier‑0**:
  - target_frames ~ 200 (на коротких видео — плотнее, см. Segmenter primary group policy)
  - равномерное покрытие по времени + учет резких переходов (если policy расширяется)

## Models

`video_pacing` **не загружает модели напрямую**. Модельная часть приходит через hard deps:

### GPU models (через core providers)
- **CLIP image encoder** (через `core_clip`, runtime=triton)
- **RAFT optical flow** (через `core_optical_flow`, runtime=triton)

### CPU models
- Нет (только OpenCV/NumPy/scipy/skimage эвристики).

## Parallelization

- **Внутренний параллелизм**: нет (один проход по sampled кадрам и переходам).
- **Внешний параллелизм**:
  - можно запускать параллельно на разных видео / разных `run_id` (разные директории `result_store`).
  - если параллельно идёт Triton‑нагрузка по зависимостям (`core_clip`, `core_optical_flow`), лимитируйте concurrency на уровне scheduler (GPU shared).

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: обработка каждого видео через subprocess с оптимизацией конфигурации
- **Оптимизации производительности**:
  - Переиспользование конфигурации для всех видео в батче
  - Параллельная обработка видео (если включено в конфигурации)
  - Оптимизация передачи параметров через CLI

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_cpu_parallel` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_video_workers`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **1.5-3x** (за счет оптимизации конфигурации и параллельной обработки)
- Для single video: без изменений (компонент работает через subprocess)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

## Human-friendly визуализация (Render System)

`video_pacing` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/video_pacing/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по pacing метрикам (frames_count, shots_count, avg_shot_duration_seconds, cuts_per_10s, motion_mean, semantic_change_mean, color_change_mean)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, is_shot_boundary, motion, semantic_change, color_change)
- **Distributions**: распределения метрик (motion, semantic_change, color_change) с min, max, mean, std, median, percentiles
- **Features**: все извлечённые признаки (shot statistics, pace curve, motion metrics, content change rate, color pacing, lighting pacing, structural pacing)

Render-context может быть использован:
- **LLM** для генерации текстовых описаний темпа и монтажа видео
- **Frontend** для построения графиков и визуализаций (timeline charts с motion/semantic/color curves, shot boundaries, distributions метрик)
- **Debugging**: быстрая проверка качества pacing анализа без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../video_pacing/_render/render.html`
- Содержит offline SVG графики (без CDN):
  - Timeline: кривые motion, semantic change rate, color change rate по времени с отмеченными shot boundaries
  - Key Features: таблица с ключевыми признаками (shots_count, shot_duration_mean, cuts_per_10s, motion metrics, semantic/color change rates)
  - Distributions: статистики по метрикам (motion, semantic_change, color_change)
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
video_pacing:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

## Quality validation & human-friendly inspection

### Human-friendly demo (HTML)

Скрипт:
- `scripts/baseline/demo_video_pacing_quality.py`

Он:
- валидирует NPZ (`validate_npz`)
- проверяет sanity (`times_s` монотонен, границы шотов в диапазоне)
- генерирует HTML с thumbnails по границам шотов + subset ключевых метрик.

Пример запуска (после того как прогнаны `core_clip`, `core_optical_flow`, `video_pacing`):

```bash
PY="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/.data_venv/bin/python"
$PY scripts/baseline/demo_video_pacing_quality.py --frames-dir "<frames_dir>" --rs-path "<result_store_run>" --out-dir "<out_dir>"
```

### Known limitations

- Метрики чувствительны к sampling density (слишком редкий sampling сгладит темп и уменьшит количество detected boundaries).
- На очень коротких видео / при малом числе шотов некоторые “структурные” метрики могут быть менее стабильны (но не должны ломать пайплайн).


