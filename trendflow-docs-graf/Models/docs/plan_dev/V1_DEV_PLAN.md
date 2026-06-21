## v1 (Transformers + trainable Encoder) — полный план разработки

Контракт v1: `Models/docs/contracts/V1_TRANSFORMER_MODEL.md`  
Финальные решения: `Models/docs/contracts/MODEL_CONTRACTS_V1.md`

---

### 0) Цель v1

v1 — основная multimodal модель, которая:
- использует **trainable encoder** (VisualEncoder/AudioEncoder) end-to-end,
- учитывает text/comments через **несколько text tokens** (Kc=4..8),
- выдаёт multi-horizon/multi-target прогноз (6 значений),
- выдаёт uncertainty через **quantile heads** (p10/p50/p90).

---

### 1) Scope и ключевые инварианты (что нельзя нарушать)

- Вход = **только snapshot_0 + артефакты DataProcessor** (никаких future-derived фич).
- Targets = `log1p(delta)` на 7/14/21, 7d masked.
- Encoder:
  - O(N) по длине исходной последовательности
  - time-axis per modality (union_timestamps_sec для visual; seconds для audio)
  - `summary_times_s` = центры uniform bins
  - adaptive K: 64/96/128 по duration_sec
- No-network / fail-fast: inference не должен требовать runtime downloads.
- Reproducibility: все версии/подписи (`model_signature`, `feature_schema_version`, т.п.) фиксируются по правилам `MODEL_SYSTEM_RULES.md`.

---

### 2) Milestones (V0..V12)

#### V0 — Подготовка “model-ready” датасета для v1
**Цель**: один воспроизводимый датасет, из которого можно обучить v1.

**Работы**:
- использовать те же targets/splits/golden (как baseline)
- собрать v1 features sources:
  - encoder inputs (seq fields)
  - snapshot_0 numeric + channel stats
  - comments_text_list_0 (для извлечения embeddings; raw не сохраняем)
- определить хранение intermediate (embeddings) без raw:
  - per-comment embeddings + агрегаты (counts/len/lang)

**Deliverables**:
- `v1_dataset_index.parquet` (ссылки на run_id/artifacts + snapshot_0 + targets/masks)
- `v1_text_embeddings.parquet|npz` (embeddings + meta, без raw)
- `dataset_metadata.json` (fingerprint, версии, правила фильтрации)

**DoD**:
- можно воспроизвести batch sampling и получить идентичные train/val/test splits.

---

#### V1 — Encoder v0 (deterministic) как baseline для v1
**Цель**: быстрый запуск “псевдо-v1” без обучения encoder.

**Работы**:
- реализовать Encoder v0 по контракту:
  - uniform binning → stats → projection → tokens
  - `summary_times_s` = bin centers
  - adaptive K rule (64/96/128)
- валидатор shapes/masks

**Deliverables**:
- `encoder_v0` код + unit tests
- e2e прогон на небольшом наборе (golden mini)

**DoD**:
- encoder outputs стабильны и повторяемы; O(N) соблюдается.

---

#### V2 — v1 модель “skeleton” (без сложности, но с правильными интерфейсами)
**Цель**: end-to-end пайплайн обучения работает, даже если качество пока среднее.

**Работы**:
- модель:
  - Visual tokens + Audio tokens + Text tokens + Meta token(s)
  - FusionTransformer (cross-attention)
  - 6 heads на p50 (пока без квантилей)
- loss:
  - masked loss для 7d
  - horizon weights (7d=0.5, 14/21=1.0) + обучаемые weights (uncertainty weighting) с cap
- time encoding:
  - `time_pos_emb = MLP(t_center/duration)` добавляется к tokens

**Deliverables**:
- training script (минимальный)
- checkpoint + отчёт метрик (Spearman/MAE)

**DoD**:
- стабильное обучение без NaN/взрывов градиента, воспроизводимость run’а.

---

#### V3 — Text/comments pipeline (Kc=4..8 tokens)
**Цель**: качественная интеграция текста без хранения raw.

**Работы**:
- text encoder выбор/фиксация (multilingual sentence embedding; pinned revision)
- per-comment embeddings (≤100)
- агрегация в tokens:
  - обязательный `comments_global_token`
  - + `comments_topk_tokens` (Kc=4..8)
- сохранить embeddings как artifacts (для train), raw не хранить

**Deliverables**:
- `TextEncoder` (train-time) + экспортируемые embedding artifacts
- тесты: determinism + schema/versioning

**DoD**:
- pipeline делает embeddings детерминированно и хранит только embeddings+meta.

---

#### V4 — Encoder v1 (trainable) end-to-end
**Цель**: включить trainable encoder и обучать end-to-end вместе с v1.

**Работы**:
- модель encoder (per modality):
  - вход: variable-length seq fields + masks
  - выход: `global_embedding`, `summary_tokens`, `summary_times_s`, `summary_mask`
- обеспечить adaptive K внутри forward (64/96/128) с правильной mask логикой
- стабилизация обучения:
  - norm layers
  - gradient clipping
  - mixed precision policy (fp16) по правилам system rules

**Deliverables**:
- trainable encoder weights входят в v1 checkpoint
- сравнение encoder_v0 vs encoder_v1 (абляция)

**DoD**:
- encoder_v1 улучшает метрики vs encoder_v0 (или даёт явный выигрыш на buckets/edge cases).

---

#### V5 — Quantile heads (p10/p50/p90) + калибровка
**Цель**: uncertainty в формате, пригодном для UI и мониторинга.

**Работы**:
- заменить/добавить головы на quantiles:
  - минимум p10/p50/p90 для каждого из 6 выходов
- loss:
  - pinball loss по каждому quantile
  - совместить с horizon weighting (обучаемые weights)
- calibration report:
  - coverage на holdout overall + по age buckets + по horizon
  - sanity: p10 ≤ p50 ≤ p90 почти всегда

**Deliverables**:
- inference output: p10/p50/p90
- `calibration_report.json`

**DoD**:
- интервалы адекватны (coverage не проваливается) и не деградируют ранжирование.

---

#### V6 — Оптимизация архитектуры (quality-first, но устойчиво)
**Цель**: добиться лучшего качества без хрупкости.

**Работы (итерациями)**:
- ablation matrix:
  - с/без text tokens
  - разные Kc (4/8)
  - разные размеры fusion transformer (layers/heads)
  - encoder depth/width
- регуляризация/stability:
  - dropout
  - EMA (опционально)
  - weight decay
- data баланс:
  - sampling по age buckets
  - handling extreme deltas (robust losses)

**Deliverables**:
- таблица экспериментов (config → метрики)
- выбранный “default v1 preset”

**DoD**:
- v1 стабильно лучше baseline по north star метрике (и не хуже по secondary).

---

#### V7 — Packaging v1 артефакта (HF + pinned)
**Цель**: v1 — воспроизводимый модельный артефакт для inference.

**Состав артефакта**:
- weights (encoder+transformer)
- `model_config.json` (архитектура, D, K rules, Kc, quantiles, preprocessing flags)
- `training_run_manifest.json` (dataset fingerprint, metrics, seeds)
- schema/version stamps (`feature_schema_version`, если применимо)

**DoD**:
- можно загрузить v1 артефакт в офлайн окружении и сделать inference.

---

#### V8 — Inference интеграция (v1)
**Цель**: v1 prediction в системе работает end-to-end.

**Работы**:
- предобработка:
  - собрать encoder inputs (seq) из NPZ
  - собрать meta (snapshot_0 numeric)
  - text embeddings/tokens (без raw)
- strict validation:
  - missing required → error
  - optional → warning + masks
- output contract:
  - p10/p50/p90 для 6 heads
  - `prediction_status`, `model_version`, `feature_schema_version`, `context_schema_version` (если будет)

**DoD**:
- regression mini (200) прогон на каждом релизе модели.

---

#### V9 — Производительность и latency (2–5s после encoder tokens)
**Цель**: уложиться в compute budget 30–50M params и latency 2–5s.

**Работы**:
- профилирование:
  - encoder forward
  - fusion transformer
  - text embedding (должен быть заранее/закэширован)
- оптимизации:
  - AMP fp16
  - batch sizing по чеклисту
  - (позже) ONNX/TensorRT если нужно

**DoD**:
- p95 latency соблюдается на target hardware.

---

#### V10 — Monitoring/observability v1
**Цель**: видеть качество/дрейф/ошибки v1.

**Работы**:
- логирование:
  - latency, OOM, missing features rate
  - распределение p50 и ширины интервалов (p90-p10) по buckets
- алерты:
  - рост ошибок/latency
  - сдвиги распределений выходов (drift proxy)

**DoD**:
- dashboard по версиям v1 с метриками и health.

---

#### V11 — Retrain cadence и release process
**Цель**: контролируемые обновления v1.

**Политика**:
- retrain при существенных изменениях в feature_schema или encoder inputs
- периодически (N недель) при накоплении новых данных/дрейфе
- релиз: pinned HF revision + changelog + regression mini pass

**DoD**:
- “одна команда” для retrain + publish новой stable версии.

---

#### V12 — Edge cases и robustness
**Цель**: устойчивость на неполных данных.

**Проверки**:
- видео без лиц (core_face_* empty)
- видео без/с плохим аудио (audio empty)
- короткие/длинные видео (adaptive K границы)
- missing 7d targets (mask)

**DoD**:
- модель не падает; корректно выставляет masks; quality не “обваливается” в отдельных buckets.

---

### 3) Риски и mitigations

- **Сложность end-to-end encoder**: начать с encoder v0, затем включать v1.
- **Uncertainty калибровка**: обязательный calibration report + sanity constraints.
- **Text pipeline**: строго “no raw”; embeddings фиксируются и версионируются.
- **Data leakage**: автоматические проверки на уровне dataset builder.

---

### 4) Точки синхронизации (что потребуется от тебя)

- подтвердить, когда baseline feature schema станет frozen (это влияет на v1 reuse)
- доступ к HF dataset со snapshots для массового обучения/оценки
- согласовать “default preset” v1 после абляций (архитектурные числа внутри 30–50M params)
---

## Навигация

[README](README.md) · [Models](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
