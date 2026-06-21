# FINAL TEST REPORT — similarity_metrics (Audit v3)

## 1. Setup

- **Module**: `similarity_metrics` (SimilarityBaselineModule)
- **Version / Schema**: `2.0.2` / `similarity_metrics_npz_v3`
- **Artifact**: `similarity_metrics/results.npz`
- **Profile**: `DataProcessor/configs/audit_v3/visual/profile_similarity_metrics.yaml`
- **Visual config**: `DataProcessor/configs/audit_v3/visual/visual_similarity_metrics_only.yaml`
- **Core providers enabled**:
  - `core_clip` (runtime: inprocess, ViT-B/32)
- **Other visual modules enabled**:
  - `similarity_metrics` only (no other visual modules)
- **Axis / Segmenter policy**:
  - `frame_indices` are owned by Segmenter (union domain)
  - For Audit v3 we enforced strict equality `similarity_metrics.frame_indices == core_clip.frame_indices` via Segmenter primary group.

## 2. Test run

- **Total videos**: 21
  - 1 legacy run: `archive/old_videos/test_video_1/test_run_1_no_optimizations`
  - 1 smoke test: `test_similarity_metrics_single`
  - 1 shortest video: `test_similarity_metrics_shortest`
  - 18 main tests: `test_similarity_metrics_2..20`
- **Runner scripts**:
  - `run_tests.sh` — последовательный прогон 20 видео
- **Result location**:
  - `DataProcessor/dp_results/youtube/<video_id>/<run_id>/similarity_metrics/results.npz`

## 3. Validation

Validator: `validate_similarity_metrics.py`.

Summary (последний прогон):

- **Total videos checked**: 21
- **Total issues**: 0
- **Schema checks**:
  - Все NPZ содержат обязательные ключи: `frame_indices`, `times_s`, `centroid_sims`, `temporal_sim_next`, `reference_present`, `feature_names`, `feature_values`, `meta`.
  - `frame_indices`: 1D, `int32`, отсортированы и уникальны, без отрицательных значений.
  - `times_s`: 1D, `float32`, длина `N`, монотонно неубывающая.
  - `centroid_sims`: 1D, `float32`, длина `N`, значения в допустимом диапазоне (cosine, примерно [-1, 1]).
  - `temporal_sim_next`: 1D, `float32`, длина `N-1`, значения в допустимом диапазоне (cosine).
  - `feature_names` / `feature_values`: согласованные 1D массивы (`F` элементов, `float32`), содержат как минимум intra‑video агрегаты (`n_frames`, `centroid_sim_mean`, `centroid_sim_std`, `temporal_sim_mean`, `temporal_sim_std`).
- **Meta contract**:
  - `meta.status` == `ok` для всех тестовых видео.
  - `meta.schema_version` соответствует `similarity_metrics_npz_v3`.
  - Базовые meta‑поля (`producer`, `producer_version`, `schema_version`, run‑идентификаторы) присутствуют.

## 4. Analysis

Analyzer: `analyze_all_results.py`.

- **Total videos in analysis**: 21
- **Reference usage**:
  - `reference_present = False` для всех видео (в тестах не задавали `reference_set_id`).
- **Frame axis**:
  - Количество кадров на видео: от ~200 до ~250 (по выборочной проверке).
  - Поле `n_frames` в `feature_values` консистентно с длиной `frame_indices`.
- **Intra‑video coherence** (по выборочной выборке трёх видео):
  - Пример `test_similarity_metrics_10`:
    - `N = 228`
    - `centroid_sim_mean ≈ 0.80`, `centroid_sim_std ≈ 0.05`
    - `temporal_sim_mean ≈ 0.85`, `temporal_sim_std ≈ 0.11`
  - Пример `test_similarity_metrics_11`:
    - `N = 239`
    - `centroid_sim_mean ≈ 0.82`, `centroid_sim_std ≈ 0.05`
    - `temporal_sim_mean ≈ 0.93`, `temporal_sim_std ≈ 0.05`
  - Пример `test_similarity_metrics_12`:
    - `N = 240`
    - `centroid_sim_mean ≈ 0.86`, `centroid_sim_std ≈ 0.10`
    - `temporal_sim_mean ≈ 0.95`, `temporal_sim_std ≈ 0.08`
- **Аномалии**:
  - Z‑score анализ по mean‑значениям `centroid_sim_mean` и `temporal_sim_mean` не выявил явных выбросов (анализатор не сообщил anomaly‑записей).
  - Распределения `centroid_sims` и `temporal_sim_next` лежат в разумном диапазоне для косинусной схожести, без `NaN`/`inf`.

## 5. Issues & Fixes

1. **frame_indices mismatch vs core_clip (strict)**
   - Ошибка: `RuntimeError: similarity_metrics | frame_indices mismatch vs core_clip (strict). Segmenter must provide consistent indices across core_clip and this module.`
   - Причина: Segmenter не обеспечивал жёсткое выравнивание `similarity_metrics.frame_indices` с `core_clip.frame_indices` в первичной sampling‑группе.
   - Фикс: В `_apply_primary_visual_sampling_group` (`Segmenter/segmenter.py`) добавлено правило:
     - `similarity_metrics.frame_indices_source = core_clip.frame_indices_source` (strict equality, no‑fallback), аналогично уже существующей политике для `high_level_semantic`.
   - Результат: после фикса все запуски similarity_metrics проходят без ошибок оси, валидатор не находит несоответствий.

## 6. Conclusions

- Модуль `similarity_metrics` успешно прошёл **smoke‑тест** и прогон на 20 тестовых видео (плюс 1 legacy‑run).
- Все NPZ‑артефакты соответствуют схеме `similarity_metrics_npz_v3` и базовому meta‑контракту; валидация не выявила ошибок.
- Структура осей (`frame_indices`, `times_s`) и согласованность с `core_clip` теперь жёстко обеспечиваются Segmenter‑ом.
- Intra‑video coherence метрики (centroid/temporal similarity) выглядят стабильными и находятся в ожидаемом диапазоне без аномалий.

Модуль `similarity_metrics` можно считать **протестированным и готовым к использованию в Audit v3 пайплайне** (в режиме без reference‑set). При добавлении reference‑наборов рекомендуется расширить анализатор для проверки распределения `reference_similarity_*` фич.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
