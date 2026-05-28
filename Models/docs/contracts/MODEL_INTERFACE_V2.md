# TrendFlow — Model ⇄ DataProcessor Interface (v2) — FINAL

Дата: 2026-02-19  
Статус: **FINAL (v2)**  
Scope: интерфейс между **DataProcessor (Segmenter/Visual/Audio/Text)** и **моделями (Baseline/vNext)**.

Этот документ фиксирует **финальные правила**, которые далее считаются стабильными.  
Изменения возможны только через bump версий (`model_interface_version`, `token_stream_schema_version`, `feature_spec_version`, `token_spec_version`, `sampling_plan_version`).

---

## 0) Цель

Свести все процессоры к единому, воспроизводимому интерфейсу для моделей:

- **Tabular path (baseline/fallback)**: фиксированные скаляры по `FeatureSpec`.
- **Token path (vNext)**: типизированные последовательности `TokenStreams` + learned pooling/tokenizer на стороне моделей.

Ключевой принцип: **процессоры не “подгоняются под конкретную модель”**. Они публикуют согласованные артефакты, а модели выбирают нужные подмножества через spec’ы.

---

## 1) Канонические артефакты DataProcessor (что считается source-of-truth)

### 1.1 Segmenter: time-axis + sampling product

- **Visual time axis (source-of-truth)**: `frames_dir/metadata.json.union_timestamps_sec`
- **Union-domain кадры**: `frames_dir/video/batch_*.npy` содержит **только union-sampled кадры**.
- **Per-component frame_indices**: в `metadata.json` для каждого компонента — индексы **в union domain**.

### 1.2 AudioProcessor: сегменты + последовательности

- Source-of-truth для временных окон: `frames_dir/audio/segments.json` (`schema_version="audio_segments_v1"`)
- Артефакты: `.../<audio_extractor>/<name>_features.npz`
  - скаляры: `feature_names[]` + `feature_values[]`
  - sequences/embeddings: разрешены (например `segment_centers_sec`, `embedding_sequence`)

### 1.3 TextProcessor: документная структура + privacy

- Вход: `VideoDocument` (title/description/comments + ссылки на ASR outputs)  
- Артефакты: `.../<text_extractor>/*.npz` + sub-artifacts в `_artifacts/`
- **Raw текст не должен считаться source-of-truth для моделей** (см. privacy).

### 1.4 VisualProcessor: визуальные сигналы

- Артефакты: `.../<visual_component>/*.npz` (обязательна маркировка tiers и схемы для audited компонентов)

---

## 2) Два режима потребления моделей

### 2.1 Tabular path (baseline, production fallback, QA)

**Определение**: baseline читает только те табличные признаки, которые перечислены в `FeatureSpec`.

- **FeatureSpec**: YAML-документ, versioned.
  - `feature_spec_version`
  - список фичей (имя, источник: component + key + трансформация)
  - политика missing (`nan_allowed`, `impute_strategy`)
  - политика required/optional по источникам (для degraded-mode)

**Правило**: baseline больше не “зависит от списка компонент”, он зависит от `FeatureSpec`.

### 2.2 Token path (vNext, production mainline)

**Определение**: модели потребляют типизированные streams, а “приведение к fixed-budget” делается на стороне моделей.

#### TokenStreams (артефакт)

`TokenStreams` — набор именованных stream’ов (по модальностям/типам), где каждый stream имеет:

- `tokens: float32[N, D]`
- `times_s: float32[N]` (центр/якорь)
- `token_type: int32[N]` (enum)
- `mask: bool[N]`
- optional:
  - `spans_s: float32[N,2]` (start/end) для сцен/событий
  - `importance: float32[N]` (top-K / selection hints)
  - `meta_json: object(dict)` (digest’ы баз, версии, источники)

Версионирование:

- `model_interface_version="v2"`
- `token_stream_schema_version="token_streams_v1"`

#### TokenSpec (модельный запрос)

`TokenSpec` — декларативное “что нужно модели”:

- какие stream’ы нужны (visual/audio/text)
- какие `token_type` допустимы
- бюджеты (`K_visual/K_audio/K_text`) + приоритеты (events > scenes > frames)
- политика downsampling/selection

---

## 3) Encoder/Tokenizer в v2

В v2 “Encoder” трактуется как **Tokenizer + Learned Pooling**:

- процессоры публикуют TokenStreams (или сырые seq для сборки TokenStreams),
- модельный слой строит fixed-budget токены через:
  - **Perceiver-style cross-attention pooling**, или
  - **hierarchical tokenization** (events/scenes → keyframes), или
  - комбинированно.

**Важно**: uniform time-binning допускается только как legacy fallback (v1), но не как каноничный способ.

---

## 4) Sampling: от “segmenter-only” к model-driven плану

### 4.1 TokenSpec/FeatureSpec → SamplingPlan

В v2 вводится контракт планирования:

- вход: `TokenSpec` + `FeatureSpec` + ресурсные бюджеты
- выход: `SamplingPlan`:
  - `sampling_plan_version`
  - список sampling families/групп (visual/audio/text)
  - multi-pass политика (coarse → refine around events)
  - лимиты на кадры/сегменты/комментарии

### 4.2 Acceptance для SamplingPlan

- воспроизводимость: одинаковый plan при одинаковых входах (config_hash + spec versions)
- отчётность: “plan vs fact” в per-run reports

---

## 5) Privacy & data retention (обязательные правила)

- **Raw OCR/ASR/Comments текст**:
  - по умолчанию **не хранить** как source-of-truth;
  - разрешено только под явными флагами retain_* (dev-only/controlled).
- Модели обучаются на **embeddings/tokens**, а не на raw текстах.
- Для дедупликации/идемпотентности разрешены **privacy-safe hashes** (как в TextProcessor).

---

## 6) Версионирование и “что считается замороженным”

Финальные версии, которые обязаны присутствовать в meta артефактов:

- `model_interface_version` (v2)
- `token_stream_schema_version` (если есть token streams)
- `feature_spec_version` (если есть tabular)
- `token_spec_version` (если есть token-driven)
- `sampling_plan_version` (если используется model-driven sampling)

Любое изменение смысла/формата → bump соответствующей версии.


