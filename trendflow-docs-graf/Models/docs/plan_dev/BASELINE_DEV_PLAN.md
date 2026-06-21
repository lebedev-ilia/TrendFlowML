## Baseline (Boosting) — полный план разработки

### 0) Цель baseline

Baseline — это:
- **контрольная точка качества** (sanity/smoke для данных и таргетов),
- production fallback в degraded-mode,
- быстрый цикл обучения/итераций,
- “табличный view” для объяснимости.

Контракты baseline: `Models/docs/contracts/BASELINE_MODEL.md`.

---

### 1) Входы/выходы, которые считаем «зафиксированными»

**Targets**:
- `views` и `likes`
- горизонты: 7d(masked), 14d, 21d
- таргет: `log1p(delta)` относительно `snapshot_0`

**Baseline inputs** (schema v1.0):
- 7 visual modules: `cut_detection`, `optical_flow`, `scene_classification`, `shot_quality`, `story_structure`, `uniqueness`, `video_pacing`
- 3 audio extractors: `clap_extractor`, `loudness_extractor`, `tempo_extractor`
- `snapshot_0` meta (см. `TARGETS_SPLITS_METRICS.md`)
- required core providers:
  - `core_brand_semantics`, `core_car_semantics`, `core_clip`, `core_face_identity`, `core_face_landmarks`,
    `core_optical_flow`, `core_place_semantics`, `core_depth_midas`, `core_object_detections`

**Baseline outputs**:
- 2 модели: views и likes
- каждая multi-output на 7/14/21 (7d masked)

---

### 2) Milestones (M0..M9)

#### M0 — Freeze “baseline feature set v0” (движущийся)
**Цель**: договориться, что такое baseline table и где она “собирается”.

**Deliverables**:
- `feature_schema_version=v0` (движущийся) + changelog изменений
- черновой `feature_spec.yaml` (single source of truth) для baseline features

**DoD**:
- можно собрать таблицу фич на ≥100 run’ов (даже если не идеально), повторяемо.

---

#### M1 — DatasetBuilder: build features + build targets
**Цель**: стабильный dataset pipeline, который одинаково работает для train/eval.

**Работы**:
- feature extraction из NPZ (через manifest), строго по `feature_spec.yaml`
- targets builder: snapshots → deltas → `log1p(delta)` + masks (7d)
- enrichment: `video_id → channel_id` (YouTube API) для channel-group split

**Deliverables**:
- `dataset.parquet` (features + targets + masks + ids + timestamps)
- `dataset_metadata.json`:
  - fingerprint/hash
  - выборка/фильтры
  - `feature_schema_version`
  - кодовые версии/seed

**DoD**:
- повторная сборка из тех же run’ов даёт идентичный output (по hash/fingerprint)
- leakage проверка (future fields не попали во features)

---

#### M2 — Сплиты и метрики как “quality gate”
**Цель**: формализовать offline оценку, чтобы сравнивать эксперименты.

**Работы**:
- split: hybrid time-split по `publishedAt` + channel-group split по `channel_id`
- metric suite:
  - north star: Spearman по `log1p(delta)`
  - secondary: MAE по `log1p(delta)` + Spearman по age buckets
- golden sets:
  - holdout=2000
  - regression mini=200

**Deliverables**:
- `evaluate_baseline.py` (или аналог) + отчёт (json + markdown)

**DoD**:
- один “кнопочный” прогон оценки baseline на holdout

---

#### M3 — Первая baseline модель: CatBoost/LightGBM
**Цель**: получить первичный рабочий baseline (quality baseline).

**Работы**:
- выбор библиотеки (CatBoost или LightGBM) по простоте multi-output/NaN support
- 2 модели: views и likes
- обработка missing:
  - если деревья/библиотека умеют NaN → оставляем NaN
  - иначе → иммутация + mask-features (по spec)
- регуляризация/гиперпараметры: базовый grid/random search
- masked 7d: loss/metric считает только по доступным таргетам

**Deliverables**:
- `baseline_views` + `baseline_likes` (модельные артефакты)
- `training_run_manifest.json`:
  - dataset fingerprint
  - feature list
  - hyperparams
  - seed
  - versions (`feature_schema_version`, `feature_extractor_version`)
- отчёт метрик + error analysis (top ошибок, buckets)

**DoD**:
- baseline обучается воспроизводимо и даёт стабильные метрики на holdout

---

#### M4 — Uncertainty для baseline (p10/p50/p90)
**Цель**: baseline выдаёт интервалы, совместимые с UI.

**Варианты реализации**:
- A) quantile regression (если поддерживается библиотекой и multi-output)
- B) conformal prediction поверх point model (по bucket’ам age + horizon)

**Deliverables**:
- baseline output включает p10/p50/p90 (или p10/p90 + p50)
- calibration report (coverage vs target)

**DoD**:
- интервалы не “ломаются” на разных age buckets (coverage адекватен)

---

#### M5 — Feature schema v1 freeze
**Цель**: заморозить baseline feature set, чтобы можно было собирать датасет и переиспользовать результаты.

**Правило** (контракт):
- после начала baseline dataset collection нельзя менять алгоритмы/выходы компонент baseline feature set
- улучшения только аддитивно через новые компоненты/ветки + bump `feature_schema_version`

**Deliverables**:
- `feature_schema_version=v1` объявлен frozen
- CI/проверка: если изменились фичи v1 без bump → fail

**DoD**:
- можно собрать 15k baseline runs + добирать до 100k без “двойной обработки”

---

#### M6 — Packaging и хранение артефактов baseline
**Цель**: стандартизировать baseline как “prediction model artifact”.

**Работы**:
- формат артефакта:
  - weights/model file
  - `feature_spec.yaml` snapshot
  - `training_run_manifest.json`
  - `metrics.json`
- хранение:
  - HuggingFace repo (pinned revision/tag)
  - (опционально) mirror в MinIO/S3
- no-network policy:
  - inference не должен “качнуть что-то” в runtime

**DoD**:
- можно воспроизвести inference по артефактам без доступа к внешней сети

---

#### M7 — Inference интеграция (baseline)
**Цель**: baseline prediction работает end-to-end.

**Работы**:
- `extract_features()` использует тот же `feature_spec.yaml`
- strict validation:
  - missing required artifacts/fields → error
  - optional → warning + заполнение по правилам spec
- output contract:
  - 6 значений (или p10/p50/p90 по 6 значениям)
  - `prediction_status` (ok/degraded)
  - `model_version`, `feature_schema_version`

**DoD**:
- e2e прогон на regression mini (200) как smoke test

---

#### M8 — Monitoring/observability baseline
**Цель**: видеть качество и деградации в проде.

**Работы**:
- логирование per prediction:
  - latency, feature missing rate, drift proxies
  - распределение outputs по buckets
- алерты:
  - error rate/latency p95/p99
  - “необычные” распределения outputs

**DoD**:
- есть dashboard/таблица метрик baseline по версиям

---

#### M9 — Retrain cadence
**Цель**: формализовать обновления baseline.

**Политика**:
- retrain при bump `feature_schema_version`
- плюс периодически (например раз в N недель), если есть новый датасет/дрейф

**DoD**:
- “одна команда” для retrain + публикации новой версии baseline

---

### 3) Риски и как их закрываем

- **Feature drift / schema churn**: freeze v1 + строгая версия `feature_schema_version`.
- **Leakage**: авто-проверки (snapshot_0 only), unit tests.
- **Missing data**: NaN+masks + validation rules.
- **Сложность multi-output**: допускаем временно 6 моделей как fallback план, но цель — 2 модели.

---

### 4) Что нужно от тебя (точки синхронизации)

- доступ к HF dataset со snapshots, когда дойдём до массового построения targets
- подтверждение момента “feature_schema_version=v1 frozen”
---

## Навигация

[README](README.md) · [Models](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
