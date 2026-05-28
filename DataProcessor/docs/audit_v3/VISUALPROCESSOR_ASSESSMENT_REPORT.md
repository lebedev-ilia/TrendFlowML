# VisualProcessor — полный отчёт (функциональность, качество фич, риски, пробелы, рекомендации)
Дата: 2026-02-19  
Scope: Audit v3 (feature & logic audit)  
Owner: DataProcessor / VisualProcessor

---

## 0) TL;DR (самое важное)

- **VisualProcessor действительно сильный источник сигналов**, но текущая связка “Visual → Encoder(v0/v1) → fixed uniform bins” навязывает процессору неудобный формат (равномерные бины), хотя часть визуальных сигналов по природе событийная/иерархическая (cuts, shots, scenes, tracks).
- **Главная рекомендация в этой версии отчёта**: менять **не только Visual**, а совместно перепроектировать **интерфейс модели** (Encoder/tokenizer) так, чтобы:
  - сохранять “структуру” (events/scenes/tracks) без принудительного равномерного биннинга,
  - позволить Visual отдавать богатые, но типизированные streams,
  - а модели сами выбирали, как сжать это в фиксированный бюджет (learned pooling / Perceiver / hierarchical tokenization).
- **Операционный разрыв всё ещё реален**: дефолтный `DataProcessor/configs/global_config.yaml` включает только часть Visual (`core_clip/core_depth_midas/core_optical_flow`), а baseline‑7 modules и большая часть core providers выключены. Это не “ошибка Visual”, а отсутствие **model-driven preset’ов** (профили под baseline/v1/vNext).
- **Сильная ось сигнала** сохраняется: `core_clip` (семантика) + `core_optical_flow` (движение/монтаж) + `cut_detection` (структура) + quality (`shot_quality`).  
  Но в vNext важно вводить **политику дедупликации на уровне tokenizer/модели**, а не пытаться вручную “вычистить корреляции” только в процессоре.
- **Главный функциональный пробел для “максимального качества popularity”** остаётся: отсутствие **tracking‑слоя** как отдельного audited компонента (screen-time, устойчивость объектов/брендов/персон, траектории, product-demo паттерны). Это полезно и для процессора, и для модели (как отдельный тип токенов/агрегатов).

---

## 0.1) Важная рамка (что меняется в этом документе)

Исторически Audit v3 рассматривал `Models/docs/contracts/*` как “жёсткие требования”, а VisualProcessor — как слой, который должен под них подстроиться.

В текущей постановке задачи это ограничение снято: **мы можем менять архитектуру и контракты моделей, Encoder, sampling policy и логику компонентов**. Поэтому ниже:

- “Требования моделей” описаны в двух режимах:
  - **as-is (v1.0)** — что зафиксировано сейчас,
  - **to-be (vNext)** — как рациональнее перепроектировать интерфейс моделей ↔ процессоров для качества/устойчивости/развития.
- “Пробелы” формулируются как **проблемы стыка** (model↔processor), а не как “вина Visual”.

## 1) Канонические требования (source-of-truth)

### 1.1 Audit v3 (что обязаны соблюдать компоненты)
Источник: `DataProcessor/docs/audit_v3/DECISIONS_AND_RULES.md`

- **NPZ = source-of-truth**, рендеры (HTML/JSON) — dev/debug/presentation.
- **Sampling ownership (as-is)**: Segmenter владеет sampling: Visual не генерирует выборку кадров, а **требует** её и fail-fast при нарушении.
- **Sampling ownership (vNext)**: допускаем “model-driven sampling” как отдельный слой планирования:
  - модель (baseline/v1/vNext) публикует `TokenSpec/FeatureSpec` + budgets,
  - планировщик (Segmenter/Orchestrator) строит multi-pass sampling (coarse → refine around events),
  - компоненты Visual становятся потребителями декларативных budgets, а не источником sampling логики.
- **No-fallback** для hard deps.
- **Empty outputs валидны** (через маски + NaN + `empty_reason`), но должны быть строго описаны.
- Для audited компонентов: **общая система схем** (human `SCHEMA.md` + machine schema JSON + runtime validation).
- Явная граница **`model_facing | analytics | debug-only`**.

### 1.2 Требования моделей (что именно нужно downstream)
Источник: `Models/docs/contracts/*`

#### Baseline (boosting)
Источник: `Models/docs/contracts/BASELINE_MODEL.md` и `Models/docs/contracts/MODEL_CONTRACTS_V1.md`

- **as-is (v1.0)**: baseline читает табличные фичи из NPZ (предпочтительно `feature_names/feature_values`, иначе “summarize numeric arrays”), и ожидает покрытие:
  - **Visual modules (7)**:  
    `cut_detection`, `optical_flow`, `scene_classification`, `shot_quality`, `story_structure`, `uniqueness`, `video_pacing`
  - **Required core providers**:  
    `core_brand_semantics`, `core_car_semantics`, `core_clip`, `core_face_identity`, `core_face_landmarks`, `core_optical_flow`, `core_place_semantics`, `core_depth_midas`, `core_object_detections`
  - Freeze policy: после старта сборки baseline-датасета нельзя менять алгоритмы/выходы задействованных компонент без bump `feature_schema_version`.

- **to-be (vNext)**: baseline лучше сделать “моделью поверх общего представления”, а не “моделью поверх списка компонентов”.
  - **Решение**: ввести единый слой “FeatureSpec + FeatureExtractor” (табличный) как *явный контракт модели*, который выбирает подмножество сигналов из result_store.  
    Тогда:
    - VisualProcessor может эволюционировать (добавлять ключи/streams), не ломая baseline,
    - baseline становится стабильным за счёт versioned `feature_spec.yaml`, а не “списка включенных компонентов”.
  - **Минимальный baseline preset (рекомендация)**: `core_clip` + `core_optical_flow` + `cut_detection` + `shot_quality` + `snapshot_0 meta`. Остальные компоненты — как “tiers” для качества, но не как жёсткий hard requirement.  
    Обоснование: это максимальная “сила сигнала на GPU-стоимость” и минимизация operational risk.

#### Encoder → v1 Transformer
Источник: `Models/docs/contracts/ENCODER_CONTRACT.md`

- **as-is (v1.0)**:
  - Visual time axis: `frames_dir/metadata.json.union_timestamps_sec` (union-domain).
  - Encoder читает **последовательности**, а табличные агрегаты остаются для baseline/QA.
  - Missing values: **NaN+masks**, не “нули-заглушки”.
  - Encoder v0/v1 сжимает seq в фиксированное $K$ через **uniform time-binning**.

- **to-be (vNext): заменить “Encoder=uniform bins” на “Tokenizer + Learned Pooling”**
  - **Проблема uniform bins**: равномерные бины плохо сохраняют:
    - события (cuts, jump-cuts, peaks эмоций, CTA моменты),
    - иерархию (shots → scenes → video),
    - объектную структуру (tracks) и “screen-time”.
  - **Новый интерфейс (предложение)**: модели принимают **TokenStreams**, а “сжатие до fixed-budget” делается *контентно* (learned) и/или *иерархически*.
    - Вход модели (per modality):
      - `tokens (N, D)` — эмбеддинги/векторы
      - `times_s (N,)` и опционально `spans_s (N,2)` (start/end) для событий/сцен
      - `token_type (N,)` (enum: frame/scene/event/track/…)
      - `mask (N,)`
      - `importance (N,)` (опционально: score для top-K selection)
    - Tokenizer’ы:
      - **Event-aware**: сохраняет события `cut_detection` и другие event streams без усреднения.
      - **Scene-aware**: строит scene tokens (например, mean/attention over frames внутри сцен).
      - **Track-aware (если добавим tracking)**: строит track tokens + track events.
    - Сжатие:
      - **Perceiver-style cross-attention pooling** (фиксированный $K$, устойчив к длинным N),
      - или **hierarchical top-K** (сначала события/сцены, затем ключевые кадры).
  - **Практический эффект**: VisualProcessor перестает быть “подчинённым uniform bins”, а модель получает более естественное представление сигналов.

---

## 2) Что такое “качество VisualProcessor” в терминах моделей

Ниже — критерии, которыми удобно измерять, “насколько Visual даст качественный выход”:

- **Coverage сигналов** (закрываем ли ключевые факторы): монтаж/темп, motion, качество, сцены/жанр, новизна/повторы, лица/эмоции/вовлечённость, текст на экране, high-level semantics (бренд/домены/франшизы), композиция/свет.
- **Стабильность контрактов**: schema+tiers, фиксированный порядок tabular features, deterministic label-space для retrieval heads.
- **Согласованность по времени**: строгая alignment политика (frame_indices/times_s), корректные маски и present_ratio.
- **Контроль шума**: эвристики (особенно “high-level” и “style” сигналы) должны быть либо подтверждены статистикой, либо переведены в `analytics/debug` или gated флагами.
- **Отсутствие leakage**: всё, что сравнивает с внешним корпусом (reference similarity), должно быть time-frozen и воспроизводимо.

---

## 3) Инвентаризация компонентов VisualProcessor (core + modules)

### 3.1 Core (Tier‑0 baseline providers)
Док-индекс: `DataProcessor/VisualProcessor/docs/MAIN_INDEX.md`

- `core_clip` — CLIP frame embeddings + text embeddings prompts + backend proxy scores (`core_clip_npz_v2`).
- `core_object_detections` — YOLO detections v2 (tracking removed) + нормализованная геометрия + frame aggregates.
- `core_optical_flow` — RAFT (Triton-only), motion curve + camera motion proxies (`core_optical_flow_npz_v3`).
- `core_depth_midas` — MiDaS (Triton-only), depth maps + proxies (`core_depth_midas_npz_v3`).
- `core_face_landmarks` — MediaPipe FaceMesh (+ pose/hands optional), person-mask gating (`core_face_landmarks_npz_v2`).
- `ocr_extractor` — OCR по `text_region` из detections (ppocr_rec_onnx recommended) (`ocr_extractor_npz_v2`).

### 3.2 Core (Tier‑1 semantic heads)
- `brand_semantics`, `car_semantics`, `face_identity`, `place_semantics` — retrieval через Embedding Service + deterministic label-space + `db_digest`.
- `content_domain`, `franchise_recognition` — CLIP text-retrieval поверх `core_clip` embeddings + offline DB provenance.

### 3.3 Modules

#### Tier‑0 baseline modules (критичны для baseline)
- `cut_detection` (имеет отдельный model-facing NPZ).
- `shot_quality`
- `video_pacing`
- `scene_classification`

#### Tier‑1 advanced modules (опционально / расширение сигнала)
- `story_structure`, `uniqueness`, `optical_flow` (consumer-only) — формально baseline‑входы, но по реализации могут быть “легковесными” слоями.
- `high_level_semantic` — агрегатор событий/сцен/мультимодальных кривых.
- `color_light`, `frames_composition` — эстетика/композиция.
- face-line: `emotion_face`, `detalize_face`, `micro_emotion`, `behavioral`
- `text_scoring` — потребитель OCR.
- `similarity_metrics` — intra + optional reference similarity.
- `action_recognition` — SlowFast, помечен как “требует доработки качества”.

---

## 4) Карта сигналов для моделей (TokenStreams: dense time-series / events / embeddings / tracks)

Эта секция фиксирует “какие типы входов мы реально даём моделям” и как их разумнее токенизировать в vNext.

### 4.1 Embeddings (per-frame / per-scene)
- **`core_clip.frame_embeddings (N,D)`** — основной visual embedding stream.  
  - Сильный универсальный сигнал.
  - Риск: далее много модулей извлекают производные от этого же потока → нужна политика дедупликации.
- **`high_level_semantic.scene_embeddings (S,D)`** — scene-level embeddings (mean over frames in scene, L2 norm).

### 4.2 Dense time-series (aligned to `frame_indices`)
Примеры “качественных” model-facing кривых:
- **Motion**: `core_optical_flow.motion_norm_per_sec_mean`, `video_pacing.motion_norm_per_sec_mean`, `optical_flow` compact features.
- **Semantic change**: `video_pacing.semantic_change_rate_per_sec`, `high_level_semantic.clip_novelty_prev/clip_sim_prev`.
- **Quality**: `shot_quality` frame-level metrics (резкость/шум/экспозиция/…).
- **Faces**: `emotion_face` (valence/arousal/probs) + masks; `detalize_face.primary_compact_features`.
- **Text**: (через `text_scoring` больше таблично; как seq — опционально, если захотим event/timeline).

### 4.3 Sparse events stream (aligned to time)
- `cut_detection` — события склеек/переходов (идеально для event tokens).
- `high_level_semantic` — unified events: hard_cut + semantic_jump + emotion_keyframe.
- `emotion_face` — keyframes peaks/transitions (можно маппить в общий events stream).
- `micro_emotion` — micro-expressions events (если включено).
- `text_scoring` — CTA peaks / emphasis peaks (хороший кандидат для events stream, но важно избегать raw текста).

**Рекомендация (vNext)**: ввести **1 каноничный “EventStream v1”** как “язык событий”, и сделать его поддерживаемым и со стороны VisualProcessor (генерация), и со стороны моделей (встроенный tokenizer). `high_level_semantic` может быть реализацией/агрегатором, но контракт должен быть общий.

---

## 5) Оценка полезности фич по группам (и где ожидаем корреляции)

### 5.1 “Главные оси” сигнала (что почти всегда нужно)

#### A) Монтаж / темп / движение
Компоненты: `core_optical_flow`, `cut_detection`, `video_pacing`, `optical_flow(module)`

- **Плюсы**: обычно один из топ‑предикторов удержания/досмотров/engagement; помогает baseline и transformer.
- **Риск**: дублирование (motion curve присутствует в нескольких местах).
- **Рекомендация**:
  - оставить один canonical motion curve (например из `core_optical_flow`),
  - в других компонентах держать либо derived/агрегаты, либо ссылку/копию только для удобства UI.

#### B) Визуальная семантика / “о чём видео”
Компоненты: `core_clip`, semantic heads (`content_domain`, `franchise_recognition`, `brand_semantics`, `car_semantics`, `place_semantics`, `face_identity`)

- **Плюсы**: сильный контентный сигнал (нишевость, узнаваемость, домен).
- **Риск**: dependency на качество и versioning баз (Embedding Service / offline DB).
- **Рекомендация**: для audited состояния фиксировать:
  - deterministic label-space,
  - `db_digest`,
  - строгую schema validation (allow_extra_keys=false).

#### C) Техническое качество картинки
Компоненты: `shot_quality`, + частично `color_light`, `frames_composition`

- **Плюсы**: качество картинки коррелирует с профессиональностью, рекламностью и доверем.
- **Риски**:
  - потенциально шумные/нестабильные метрики (lens-dirt, экзотика) → лучше gated presets,
  - сильная корреляция между модулями эстетики.

### 5.2 Высокие корреляции/дубли (требуют политики)

#### CLIP-derived “новизна/схожесть”
Компоненты/ключи:
- `core_clip.consecutive_cosine_prev`
- `high_level_semantic.clip_sim_prev / clip_novelty_prev`
- `video_pacing.semantic_change_rate_per_sec`
- `uniqueness.cos_dist_next`, `similarity_metrics.temporal_sim_next`

**Риск**: один и тот же сигнал может зайти в baseline несколько раз (feature leakage внутри модели).  
**Рекомендация**:
- для baseline выбрать 1–2 представителя (например `uniqueness` + `video_pacing`),
- остальное оставить для transformer seq/QA, либо перевести в `analytics`.

#### Cut/pacing duplication
`cut_detection` и `video_pacing` оба содержат много про ритм/шоты.  
**Рекомендация**: `cut_detection` = source-of-truth структуры и событий; `video_pacing` = агрегаты/гистограммы/robust scalars.

### 5.3 Явные источники шума / эвристики

Это не “плохо”, но должно быть строго промаркировано и gated:

- `story_structure`: документ прямо говорит “heuristics/proxies”.  
  - **Рекомендация**: держать как baseline‑кандидат, но иметь quality flags (`frame_feature_present_ratio`, `topic_shift_curve_present`) и минимизировать “магические” правила в model_facing.
- `behavioral`: много rule-based эвристик (жесты/поза/стресс).  
  - **Рекомендация**: до статистического подтверждения держать `analytics` или optional.
- `scene_classification.luxury/atmosphere`: потенциально bias‑чувствительно.  
  - **Рекомендация**: пока считать `analytics` или использовать только после bias review + feature importance на датасете.
- `similarity_metrics.reference` часть: риск leakage и нестабильности reference корпуса.  
  - **Рекомендация**: “reference similarity” не включать в model_facing до формального anti‑leakage контракта.
- `action_recognition`: сам документ помечает “нужна валидация качества”.  
  - **Рекомендация**: experimental/optional.

---

## 6) Пробелы текущей логической реализации (что стоит улучшить)

### 6.1 Tracking как отдельный audited компонент (главный пробел)
Контекст: tracking удалён из `core_object_detections` (решение baseline).  
Последствия:
- сложнее посчитать screen-time и устойчивость объектов/брендов,
- семантические головы используют surrogate track_ids per-detection,
- часть downstream паттернов (product demo, “объект в руках”, “бренд на экране”) теряет структуру.

**Рекомендация**: добавить новый компонент (примерно):
- `core_object_tracks` (или `object_tracking_npz_v1`)
- вход: `core_object_detections` (boxes/scores/class_ids/valid_mask)
- выход:
  - track_ids per detection slot,
  - track-level aggregates (duration, avg area, stability),
  - events (track_start/track_end)
- tier: часть может быть `model_facing` (таблично), часть `analytics`.

### 6.2 Конфиг несоответствует baseline contract (операционный разрыв)
В `DataProcessor/configs/global_config.yaml` (секция `processors.visual.inline_config`) многие required для baseline компоненты выключены.

**Рекомендация**:
- завести **model-driven профили** (baseline/v1/vNext) и перестать полагаться на “случайное” состояние inline_config;
- в vNext дополнительно рассмотреть генерацию inline_config из `FeatureSpec/TokenSpec` (модельный запрос → минимальный набор компонентов + budgets).

### 6.3 Anti-leakage контракт для reference similarity
Если `similarity_metrics` будет частью model_facing:
- reference pack должен быть **time-frozen** относительно prediction time,
- должен иметь `reference_pack_digest` и версию,
- должен быть одинаков на train/infer.

### 6.4 Политика feature_schema_version vs schema_version
Нужно явно фиксировать:
- `schema_version` (контракт NPZ компонента),
- `feature_schema_version` (контракт набора model_facing фич для training/inference).

**Рекомендация**: baseline dataset builder должен читать только “frozen subset” ключей, даже если NPZ содержит больше debug/analytics.

### 6.5 Privacy/bias контуры для face/text
Риски:
- “identity-like” вектора (face shape) и любые поля, которые могут повысить re-identification риск.
- text features: raw OCR текст нельзя хранить в проде (у вас уже есть retain flag).

**Рекомендация**:
- жёстко держать raw текст/чувствительные поля за флагами dev-only,
- документировать, какие face‑фичи допускаются как model_facing, а какие только analytics.

### 6.6 (vNext) Sampling как узкое место качества (нужен model-driven планировщик)
Симптомы текущего подхода:
- uniform sampling “по времени” хорошо для усреднений, но плохо для событий/пиков/редких моментов;
- разные компоненты имеют разные “полезные частоты” (motion/scene/face/text), а один union-domain часто приводит либо к перерасходу, либо к пропускам.

**Рекомендация**:
- ввести **TokenSpec/FeatureSpec → SamplingPlan** (declarative):
  - какие streams нужны модели,
  - какие event-driven зоны важнее (например, ±1с вокруг cut),
  - какие максимальные бюджеты по модальностям.
- реализовать **multi-pass sampling**:
  - pass-0: быстрый coarse (низкий fps, дешёвые признаки) → находим события/сцены/пики
  - pass-1: локально повышаем плотность around events для дорогих компонентов (CLIP/RAFT/face)

Это снимает давление с VisualProcessor (ему не нужно “угадывать sampling”), и повышает качество токенов для vNext.

---

## 7) Рекомендованные пресеты конфигурации (оптимальная обработка)

Цель: сделать 3–4 пресета, которые можно включать профилями. В vNext пресеты должны соответствовать не только “списку компонентов”, но и **типу потребителя** (baseline табличный vs transformer tokens).

### 7.1 Preset A — `visual_baseline_v1` (для обучения baseline)
**Включить обязательно**:
- Core providers (required baseline):
  - `core_clip`, `core_object_detections`, `core_optical_flow`, `core_depth_midas`, `core_face_landmarks`,
  - semantic heads: `brand_semantics`, `car_semantics`, `face_identity`, `place_semantics` (и при необходимости `content_domain`)
- Modules (baseline‑7):
  - `cut_detection`, `optical_flow`, `scene_classification`, `shot_quality`, `story_structure`, `uniqueness`, `video_pacing`

**Параметры качества (стартовые)**:
- sampling: `core_clip` cap ~3000 @ 20min (как в Audit v3), `cut_detection` не слишком редкий (max gap ≤ 6s).
- `core_optical_flow`: `raft_256` как baseline.
- `shot_quality`: preset default, без редких/нестабильных метрик в model_facing (gated).

### 7.2 Preset B — `visual_transformer_quality_v1` (для Encoder/v1)
Всё из Preset A +:
- `high_level_semantic` (unified events stream + scene embeddings + multimodal curves)
- `color_light`, `frames_composition`
- face-line (по продуктовой необходимости): `emotion_face`, `detalize_face`

### 7.2b Preset B2 — `visual_transformer_vNext` (tokenizer + learned pooling)
Если идём по пути vNext (см. 1.2), то “обязательность” части baseline‑7 модулей уменьшается, потому что модель получает более естественные токены:
- Обязательно:
  - `core_clip`, `core_optical_flow`, `cut_detection`, `shot_quality`
  - + `high_level_semantic` **или** отдельный **новый** компонент `event_stream` (как контрактный источник событий)
- Опционально (tiers):
  - `scene_classification`, `uniqueness`, `video_pacing`, face-line, semantic heads
- Если добавим tracking:
  - `core_object_detections` + новый `core_object_tracks`

### 7.3 Preset C — `visual_research_full` (QA/исследования)
Всё из Preset B + optional:
- `micro_emotion` (тяжёлый, требует QA)
- `text_scoring` (OCR consumer)
- `similarity_metrics` (intra — ok; reference — только после anti-leakage)
- `behavioral`, `action_recognition` (experimental)

---

## 8) Конкретные рекомендации “что менять” (action items)

### 8.1 Документировать и заморозить model_facing subset
- Для каждого baseline‑7 модуля: выделить **фиксированный набор** `feature_names/feature_values` и гарантировать стабильный порядок (у многих уже сделано).
- Для seq/tokenizer: определить “канонические” TokenStreams (и их types), а всё остальное увести в analytics.

### 8.2 Политика дедупликации CLIP-derived сигналов
- baseline: 1–2 представителя novelty/coherence (например uniqueness + pacing).
- transformer: оставить richness (seq), но иметь masks/present_ratio.

### 8.3 Tracking компонент (как отдельный проектный пункт)
- добавить `core_object_tracks` (см. 6.1) как optional core provider,
- downstream heads/modules смогут использовать tracks для screen-time и устойчивости.

### 8.4 Reference similarity anti-leakage контракт
- если остаётся: ввести `reference_pack_manifest.json` с digest/created_at/source_window и хранить это в meta.

### 8.5 QA-пакет видео и “quality gates”
- Минимум 10–20 видео (разные типы) как в Audit v3.
- Для каждого ключевого компонента: expected distributions + анти‑паттерны в render.

### 8.6 (vNext) Ввести TokenStreams contract как отдельный документ Models↔DataProcessor
- Новая точка правды: “что такое token stream”, какие обязательные поля (`tokens/times_s/token_type/mask`) и как версионировать.
- TokenStreams должны жить как артефакт (NPZ/JSON+NPZ) и быть воспроизводимы.
- Encoder/Tokenizer становится “model-side” компонентом, но его контракт должен быть согласован с процессорами.

### 8.7 (vNext) Ввести TokenSpec/FeatureSpec → SamplingPlan как контракт планирования
- Model-side публикует spec (что нужно) и бюджеты (сколько можно).
- DataProcessor публикует “actual plan vs fact” отчёт (для воспроизводимости и дебага).
- Segmenter становится исполнителем plan’а (multi-pass) вместо “единственного автора” sampling политики.

---

## 9) Приложение: ссылки на ключевые документы/файлы

- Audit v3 rules: `DataProcessor/docs/audit_v3/DECISIONS_AND_RULES.md`
- Audit v3 templates: `DataProcessor/docs/audit_v3/TEMPLATES.md`
- Models contracts:
  - `Models/docs/contracts/BASELINE_MODEL.md`
  - `Models/docs/contracts/ENCODER_CONTRACT.md`
  - `Models/docs/contracts/MODEL_CONTRACTS_V1.md`
  - `Models/docs/contracts/MODEL_INTERFACE_V2.md`
- VisualProcessor docs index: `DataProcessor/VisualProcessor/docs/MAIN_INDEX.md`
- VisualProcessor global config (inline): `DataProcessor/configs/global_config.yaml` → `processors.visual.inline_config`

### 9.1 Мини-спецификация (vNext): TokenStreams + TokenSpec/FeatureSpec

Чтобы избежать “неявных требований” между процессорами и моделями, в vNext предлагается вынести в отдельный документ/контракт два слоя:

- **TokenStreams** (артефакт, который модели потребляют):
  - обязательные поля: `tokens`, `times_s`, `token_type`, `mask`
  - опционально: `spans_s`, `importance`, `meta_json`
  - версионирование: `token_stream_schema_version` + `producer_versions[]`

- **TokenSpec/FeatureSpec** (декларативное “что нужно модели”):
  - `feature_spec.yaml`: список табличных фичей и источников (baseline)
  - `token_spec.yaml`: какие streams нужны, какие типы токенов допускаем, бюджеты/приоритеты
  - из этих spec’ов строится `SamplingPlan` (план выборки), который исполняет Segmenter/Orchestrator

---

## 10) VisualProcessor prod-ready (DoD) — закрывающий блок

Этот раздел — финальный чеклист, после выполнения которого VisualProcessor считаем **prod-ready по логике/фичам/контрактам** (останутся оптимизации/бенчи/масштабные прогоны).

### 10.1 DoD по контрактам и артефактам

- **Схемы**: для всех baseline-required core + baseline‑7 modules есть:
  - human `SCHEMA.md`
  - machine schema JSON
  - runtime validation (allow_extra_keys=false для audited)
- **Tiers**: все ключи промаркированы `model_facing | analytics | debug-only`.
- **Empty semantics**: определены `status/empty_reason`, маски и NaN-политика для всех model_facing массивов.
- **Versioning**: `producer_version`/`schema_version` bump при любом изменении смысла/формата.

### 10.2 DoD по “модельному интерфейсу v2” (готовность к TokenStreams)

Связанный контракт: `Models/docs/contracts/MODEL_INTERFACE_V2.md`.

- Есть каноничный **EventStream v1** (либо внутри `high_level_semantic`, либо отдельный компонент `event_stream`):
  - события: минимум `hard_cut`, опционально `semantic_jump`, `emotion_keyframe`
  - поля: `event_type`, `time_s` и/или `span_s`, `confidence`, `meta_json`
- Для vNext возможна сборка `TokenStreams` без “uniform bins” как обязательного шага:
  - frame embeddings (`core_clip`)
  - event tokens (`cut_detection`/EventStream)
  - scene tokens (если используем `scene_classification`/scene grouping)
- Дедупликация сигналов CLIP-derived переносится в **tokenizer/model-side** (а не ручной “вычищалкой” по компонентам).

### 10.3 DoD по baseline path (табличные фичи)

- У baseline‑7 модулей стабильный `feature_names/feature_values`:
  - стабильный порядок
  - задокументированный frozen subset (что считается model_facing для baseline)
- Введены **model-driven presets** (baseline/v1/vNext) и прекращена зависимость от “случайного” inline_config.

### 10.4 Прод-риски, которые должны быть закрыты до включения в model_facing

- **Reference similarity** (`similarity_metrics.reference`): запрещено в model_facing до anti-leakage контракта (time-frozen pack + digest + одинаково на train/infer).
- **Privacy**: raw OCR/face-sensitive/debug поля — только под dev flags, не model_facing по умолчанию.


