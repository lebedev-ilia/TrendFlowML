## Semantic Heads & Object Detections — Contracts + Q&A (working doc)

Этот файл — **единственная точка правды** для нового функционала вокруг:
- `core_object_detections` (proposal generator; Audit v3 baseline: tracking removed, используем surrogate `track_ids` downstream)
- semantic heads (brands / car make+segment / face identity / scene+place)
- переиспользование `core_clip`
- правила качества, тесты, валидации, idempotency, dynamic batching constraints.

Формат работы: я задаю вопросы **по раундам**, ты отвечаешь **прямо под каждым вопросом**.
После каждого раунда я обновляю правила/контракты и добавляю следующий раунд.

---

### 0) Канонические инварианты проекта (согласованы с текущими docs)

- **NPZ = source-of-truth**, JSON только presentation (кроме `manifest.json`).
- **No-network**: никаких runtime downloads (веса/токенайзеры/категории).
- **No-fallback** на альтернативные модели при проблемах (если модель/артефакт нужен → fail-fast).
- **Idempotency key** компонента:
  \[
  (platform\_id, video\_id, component, config\_hash, sampling\_policy\_version,
   producer\_version, schema\_version, model\_signature)
  \]
- **Model system**: любой компонент, вызывающий ML, пишет `models_used[]` + вычисляет `model_signature`
  (runtime/engine/precision/device входят).
- **Segmenter = единственный источник `frame_indices`**. Компоненты не выбирают кадры сами.
- **Time axis source-of-truth**: `frames_dir/metadata.json.union_timestamps_sec`.
- **Shared primary sampling group** (для наших core‑слоёв): все зависящие head’ы выравниваются по `frame_indices`
  от `core_object_detections` (или общей группы).
- **Tracking (baseline)**: в Audit v3 tracking **удалён** из `core_object_detections`; если нужен “настоящий” track_id, вернём как future-improvement без изменения core NPZ контракта.
- **Triton batching пока не меняем**: cost считаем на unit=1; throughput — cross-video micro-batching scheduler’ом.

---

## 1) Визия архитектуры (кто за что отвечает)

### 1.1 Proposal vs semantics (принцип)

- `core_object_detections` даёт **геометрию (bbox) + базовые object/region классы** (baseline: без tracking; downstream используют surrogate `track_ids` per-detection при необходимости).
- Fine-grained semantics НЕ запихиваем в YOLO как 500 классов брендов/людей/зданий:
  - brands/logos → `core_brand_semantics`
  - car make/model/segment → `core_car_semantics`
  - popular people → `core_face_landmarks` + `core_face_identity`
  - scene/place → `scene_classification` + future `core_place_semantics` / `core_landmark_semantics`

---

## 2) Контрактные артефакты (черновик v1)

### 2.1 `core_object_detections` (существует)

Artifact: `result_store/<platform>/<video>/<run>/core_object_detections/detections.npz`

Обязательные ключи (Audit v3 baseline, schema v2):
- `frame_indices (N,)`
- `times_s (N,)`
- `boxes (N, MAX, 4)` xyxy
- `boxes_norm (N, MAX, 4)`
- `centers_norm (N, MAX, 2)`
- `areas_frac (N, MAX)`
- `scores (N, MAX)`
- `class_ids (N, MAX)`
- `valid_mask (N, MAX)`
- `class_names (C,)` `"id:name"`
- `meta` (dict, object-array; общие поля + models_used[])
- `meta_json` (str; JSON-строка meta для совместимости между окружениями)

### 2.2 `core_brand_semantics` (существует, MVP)

Artifact: `.../core_brand_semantics/brand_semantics.npz`

Ключи (v1):
- `frame_indices (N,)` (строго = core_object_detections)
- `semantic_label_names (A,)` `"id:name"`
- `track_ids (T,)`, `track_topk_ids (T,K)`, `track_topk_scores (T,K)`
- `det_topk_ids (N,MAX,K)`, `det_topk_scores (N,MAX,K)`
- `meta` (dict; gating + models_used)

### 2.3 Новые head’ы (будут)

Семантика машин/лиц/мест должна следовать тому же принципу:
- **top‑K ids+scores** (sparse)
- **track‑level** по умолчанию (дёшево) + optional det‑level
- strict alignment по `frame_indices`
- reproducibility (labels db/version + thresholds + models_used)

---

## 3) Quality: минимальные требования (policy)

Запрещаем слабые “эвристики ради эвристик”. Разрешены только:
- дешёвые **гейты** для стоимости (top‑K boxes, min score, track sampling cadence),
- и простые агрегаты/метрики, которые легко валидируются.

Для каждого нового компонента фиксируем:
- **Acceptance criteria** (что значит “закрыт”)
- **Quality validation** (набор видео, метрики, golden/regression)
- **Failure/empty semantics** (что error, что empty, какие `empty_reason`)

---

## 4) Q&A — Round 1 (ответь под каждым пунктом)

### 4.1 Общие вопросы (влияют на все head’ы)

**Q1.1** Какие head’ы считаем required для prediction v1 (fail-fast), а какие optional (best-effort)?

**A1.1**: давай считать все (8 компонентов из них 3 уже существуют, но нуждаются в доработке)

**Q1.2** Где хранится “база” для сравнения (brands / car makes / celebrities / places)?
Варианты: локальные файлы в repo, `dp_models` bundle, отдельный “artifact package” под ModelManager, DB (но no-network).

**A1.2**: Да, пока это локальные базы. Нужно кстати отдельный план по тому как это вообще реализовать, как собрать такие базы и тд.

**Q1.3** Как версионируем эти базы?
Нужно: `labels_version`/`db_version` + `weights_digest`/hash списка → в `meta` и в idempotency key.

**A1.3**: реши сам

**Q1.4** Какой top‑K нужен по умолчанию для всех head’ов? (K=3? 5? 10?)

**A1.4**: 5 достаточно

**Q1.5** Политика “не уверен”:
если similarity низкая — пишем `-1`/NaN или всё равно top‑1?

**A1.5**: реши сам

**Q1.6** Нужны ли пороги качества per label (например “nike_logo threshold=0.28”) или единый threshold?

**A1.6**: думаю да, реши сам

**Q1.7** Требование к детерминизму: нужно ли фиксировать seed и добиваться повторяемости “бит‑в‑бит” на одной машине?

**A1.7**: думаю что нет, не сильно в этом разбираюсь

### 4.2 `core_object_detections` (что фиксируем до новой модели)

**Q1.8** Подтверди: таксономия детектора v1 = 40 классов (как в `DETECTOR_TAXONOMY_V1_40_NAMES.txt`) — финал?
Если нет, какие изменения и почему (только additive, ids стабильны).

**A1.8**: а у нас в финальной модели будут же дефолтные классы COCO80 (и + наши 40) или будут только наши?

**Q1.9** Какие из 40 классов “ядро” качества (must-have) для Train0?
Список 10–15 классов.

**A1.9**: выбери сам, по твоему мнению какие больше всего влияют на популярность видео контента

**Q1.10** Для `logo_region` / `text_region` мы хотим high recall или high precision на старте?

**A1.10**: я думаю recall, но решение за тобой.

### 4.3 Brands: `core_brand_semantics` (MVP→v1)

**Q1.11** Brand taxonomy: сколько “брендовых сущностей” в первой базе?
Порядок: 50? 200? 2k? И какие домены (одежда/косметика/еда/техника/авто)?

**A1.11**: в базе брэндов будут все области брэндов (одежда/косметика/еда/техника/авто и тд.). Для начала я думаю 500 брэндов будет достаточно

**Q1.12** Делаем ли отдельные “типы” брендов в ids (например `nike_logo` vs `nike_text`) или один id на бренд?

**A1.12**: один

**Q1.13** Какие источники изображений для базы (для валидации)?
Например: логотипы PNG + синтетика + реальные кадры.

**A1.13**: реши сам, а я уже буду отталкиваться от твоего решения

### 4.4 Cars: `core_car_semantics` (новый)

**Q1.14** Контракт вывода: что хотим предсказывать?
Варианты:
- make (BMW, Toyota)
- model (Camry)
- segment (luxury/economy)
- body type (SUV/sedan)
- price bucket

**A1.14**: все перечисленное + top-3 по каждому + conf по каждому

**Q1.15** Источник proposals: только `car` bbox из YOLO или также `truck/bus`?

**A1.15**: реши сам

**Q1.16** База машин: откуда берём список makes/models и как версионируем?

**A1.16**: не знаю откуда брать. подсказывай

### 4.5 People: `core_face_identity` (новый, поверх `core_face_landmarks`)

**Q1.17** Что считаем “популярным человеком” в MVP:
- closed-set (список N знаменитостей) или open-set (retrieval по большой базе)?

**A1.17**: должна быть база, но в каком виде и кол-ве я не знаю, подсказывай

**Q1.18** Какие output ids:
- `celebrity_id` (int) + optional `name` в отдельном mapping,
- или строковые ids?

**A1.18**: реши сам

**Q1.19** Требования к приватности/ретеншну:
храним ли face crops/embeddings в result_store? если да — на сколько и в каком виде?

**A1.19**: реши сам

### 4.6 Scene + place: `scene_classification` и future place/landmark head

**Q1.20** Что именно хотим улучшить в `scene_classification`:
- точность Places365?
- стабильность по времени?
- добавление “place retrieval” по базе известных мест?

**A1.20**: все перечисленное

**Q1.21** Place/landmark база: это топ‑N известных мест (закрытый список) или retrieval по “галерее”?

**A1.21**: реши сам, не знаю точно

---

## 4) Q&A — Round 2 (фиксируем полуфинальные контракты и дефолты)

Ниже — вопросы “про решения”, чтобы мы могли начать разработку **до** готовой модели и базы, не рискуя качеством.

### 4.7 Базы (brands/cars/celebs/places): формат, версионирование, воспроизводимость

**Q2.1** Где физически хранить базы (no-network) в MVP?
Вариант, который я предлагаю: `dp_models/bundled_models/semantics/<domain>/<version>/...` (как часть offline bundle), а в NPZ `meta` писать:
- `db_name`, `db_version`
- `db_digest` (sha256 от canonical списка + ассетов)
- `db_path` (внутри bundle)
Ок?

**A2.1**: ок

**Q2.2** Multi-language: базы будут RU/EN или только EN внутри базы, а UI/aliases отдельно?

**A2.2**: а как лучше. можно и так и так сделать. как ты скажешь

### 4.8 `core_object_detections`: COCO80 + наши 40?

**Q2.3** По твоему вопросу A1.8: предлагаю **не смешивать COCO80 и наши 40 в одном `class_ids`**.
Режимы:
- `detector_taxonomy_v1_40` (finetuned) → только 40 классов, stable ids 0..39
- `coco80` (baseline) → только COCO80
Если нужно “оба мира” одновременно — делаем это отдельной semantic head (или отдельным полем), а не мешаем ids.
Подтверди: **Ок / Не ок (почему)**.

**A2.3**: ок

### 4.9 Confidence, top‑K, thresholds (без эвристик, но с чёткой политикой)

**Q2.4** Top‑K=5 (принято). Что делать если “не уверен”?
Выбери политику:
- A) всегда пишем top‑5, + `is_confident_mask` (bool) по порогу
- B) если ниже порога → `top1_id=-1` и scores=NaN (но raw top‑5 всё равно сохраняем в отдельный `raw_topk_*`)

**A2.4**: да нет, просто top-5 без порога

**Q2.5** Thresholds (ты сказал “думаю да”):
подтверди гибрид:
- `threshold_global`
- optional `threshold_per_label[id]`

**A2.5**: да

### 4.10 Output contracts per head (shape/dtype/units)

**Q2.6** Выходы head’ов: фиксируем общий шаблон (чтобы encoder был простым)?
Предлагаю для всех semantic heads:
- `frame_indices (N,) int32` (строго aligned с core_object_detections)
- `semantic_label_names (A,) str` (`"id:name"`)
- `track_ids (T,) int32`
- `track_topk_ids (T,K) int32`, `track_topk_scores (T,K) float32`
- optional det-level: `det_topk_ids (N,MAX,K)`, `det_topk_scores (N,MAX,K)`
Ок?

**A2.6**: ок

### 4.11 Brands (500): качество базы и алгоритм (минимум эвристик)

**Q2.7** Brand base v1: что считаем “брендом”?
Варианты:
- A) 1 id = бренд (nike), независимо от продукта/категории
- B) 1 id = бренд + категория (nike_shoes, nike_clothing)
Сейчас у нас A1.12=“один”. Подтверди что это A.

**A2.7**: давай для начала A

**Q2.8** Brand head: какой метод хотим в MVP (без “хаков”):
- A) CLIP text‑matching по prompts (текущий MVP) + строгая база prompts
- B) CLIP image‑prototype retrieval (галерея логотипов) + cosine к image embeddings
- C) гибрид A+B (рекомендую)
Что выбираем для v1?

**A2.8**: гибрид

### 4.12 Cars: make/model/segment/price (top‑3)

**Q2.9** Сколько bucket’ов хочешь для:
- `segment` (например 8)
- `price_bucket` (например 8–12)
Назови числа.

**A2.9**: реши сам

**Q2.10** Источник proposals для car head:
предлагаю: `car` + optional `truck/bus/motorcycle` (включаем флагами).
Ок?

**A2.10**: ок

### 4.13 People identity: база, приватность, хранение

**Q2.11** Privacy/retention policy (предлагаю):
- в result_store НЕ храним face crops
- face embeddings храним только если включён debug и TTL=7 days, иначе нет
- всегда храним только top‑K ids+scores
Ок?

**A2.11**: ок

**Q2.12** Размер базы “популярных людей” для v1:
1k / 10k / 100k — что реалистично?

**A2.12**: 500 хватит для начала

### 4.14 Scene + place: улучшения и база

**Q2.13** Place/landmark подход:
- A) closed‑set top‑N
- B) retrieval по gallery embeddings
Я предлагаю начать с B (retrieval) и потом добавить A. Ок?

**A2.13**: ок

### 4.15 Quality validation: как именно “не накосячить”

**Q2.14** Golden набор видео для регрессий: сколько видео готов выделить сейчас (10/30/100)?

**A2.14**: сколько нужно, столько и выделю

**Q2.15** Acceptance критерий для каждого head’а (MVP):
подтверди, что для v1 достаточно:
- smoke (артефакт+schema)
- stability по track (не прыгает каждый кадр)
- ручная проверка precision на 50–200 событий (top‑K релевантность)

**A2.15**: да

## 5) Test & validation plan (черновик, будем финализировать по ответам)

Для каждого компонента/головы:
- smoke‑test на 1–2 видео (артефакт пишется, schema валидна)
- quality sanity на 10–30 видео (ручной просмотр + минимальные метрики стабильности)
- golden/regression: фиксированный набор видео (5–10) + tolerances
- resource tests: latency/memory по чеклисту для scheduler’а

---

## 6) Resolved decisions (v0.1) — зафиксировано по Round 1–2

### 6.1 Required set (v1)

- Все целевые компоненты считаем **required** для prediction v1 (fail-fast).
  - Ты указал “8 компонентов (3 уже есть, но нужно доработать)”.
  - **Зафиксированный список (8)**:
    - `core_object_detections` (proposal generator + tracking)
    - `core_brand_semantics` (brands/logos, CLIP hybrid)
    - `core_car_semantics` (car make/model retrieval)
    - `core_face_landmarks` (faces/landmarks presence)
    - `core_face_identity` (celebrity retrieval; требует face-embed модель)
    - `core_clip` (frame embeddings)
    - `scene_classification` (scene context)
    - `core_place_semantics` (place/landmark retrieval)

### 6.2 Базы (no-network)

- **Локальные базы**: да (MVP).
- Хранение: **в offline bundle** (`dp_models/bundled_models/semantics/<domain>/<version>/...`) — **OK** (A2.1).
- Версионирование: будет фиксироваться через `db_name/db_version/db_digest` в `meta` + включаться в idempotency key (согласно A2.1 + Model system rules).

### 6.3 `core_object_detections`: COCO80 vs наши 40

- Не смешиваем COCO80 и taxonomy_v1_40 в одном `class_ids` — **OK** (A2.3).
  - Будет два режима/ветки: `coco80` (baseline) и `taxonomy_v1_40` (finetuned).

### 6.4 Outputs heads: общий шаблон

- Общий шаблон NPZ (frame_indices + semantic_label_names + track_topk + optional det_topk) — **OK** (A2.6).
- Top‑K default: **K=5**.

### 6.5 Thresholds / confidence

- **Thresholds per label**: да (A2.5).
- При этом: **не “режем” вывод по порогу** — всегда сохраняем top‑5 (A2.4).
  - Значит thresholds используются не для “сокрытия” результата, а для:
    - `is_confident_*` флагов,
    - отборов в downstream/encoder,
    - quality dashboards / warnings.

### 6.6 Brands

- Бренды v1: **500**.
- 1 id = 1 бренд (A2.7 = A).
- Метод v1: **гибрид** CLIP prompts + image‑prototype retrieval (A2.8).

### 6.7 Cars

- Car head нужен: make/model/segment/body_type/price_bucket, top‑3 по каждому + conf (A1.14).
- Proposals: `car` + optional `truck/bus/motorcycle` — OK (A2.10).
- Число bucket’ов для segment/price — TBD (A2.9 = “реши сам”).

### 6.8 People identity

- Privacy: не храним face crops; embeddings только debug+TTL=7d; всегда top‑K ids+scores — OK (A2.11).
- Размер базы celebrities v1: **500** (A2.12).

### 6.9 Scene + place

- Улучшать всё перечисленное (Places365 точность + стабильность + place retrieval) — OK (A1.20).
- Place/landmark подход: **retrieval по gallery embeddings** (B) — OK (A2.13).

### 6.10 Golden/regression

- Golden набор: ты готов выделить “сколько нужно” (A2.14).
- Acceptance (MVP): smoke + track stability + ручная precision проверка 50–200 событий — OK (A2.15).

---

## 7) Q&A — Round 3 (финализируем схемы и “как собирать базы”)

### 7.1 Multi-language (из A2.2: “как лучше”)

**Q3.1 (proposal)** Я предлагаю правило:
- Канонические `semantic_label_names` всегда **на английском** (stable ids, reproducibility).
- База хранит `aliases_ru/aliases_en` для поиска/QA и генерации prompts.
- Prompts: и RU и EN (multi-lingual CLIP text) — но на выходе id один и тот же.
Ок?

**A3.1**: ок

### 7.2 NPZ schema details (точно фиксируем поля)

**Q3.2** Для всех semantic heads добавляем ли в NPZ:
- `times_s (N,) float32` (алиас к union_timestamps_sec[frame_indices])?
- `valid_mask` для `track_topk` (например `track_present_mask (T,) bool`)?
Ок/не ок?

**A3.2**: ок

**Q3.3** Для top‑K scores: cosine similarity ([-1..1]) или softmax‑probability (0..1)?
Предлагаю хранить **cosine** (сырое), а “вероятности” — дело encoder’а/калибровки.
Ок?

**A3.3**: ок

**Q3.4** Thresholds per label: где именно храним?
Предлагаю в `meta`:
- `threshold_global`
- `threshold_per_label` (dict id->float)
и опционально массивом `threshold_per_label_arr (A,)` aligned с `semantic_label_names`.
Ок?

**A3.4**: ок

### 7.3 Формат баз (artifact packages) — чтобы их можно было собирать/валидировать

**Q3.5** Brand base (v1=500): что храним обязательно на бренд?
Предлагаю минимум:
- `brands.jsonl`: `{id, name, aliases:[...], category?, prompts_en:[...], prompts_ru:[...] }`
- `prototypes/` (optional): 1–10 изображений логотипа на бренд
Ок?

**A3.5**: ок

**Q3.6** Car base: предлагаю начать с public структурированных датасетов (без сети в runtime):
- makes/models: open datasets (Wikipedia-derived dumps / Kaggle exports) → мы сохраняем в bundle
- segment/price: ручная таблица buckets (мы задаём правила)
Ок?

**A3.6**: ок

**Q3.7** Segment/price buckets (A2.9 ты делегировал мне):
предлагаю:
- `segment_id`: 8 buckets (economy, compact, midsize, fullsize, luxury, sport, suv/crossover, pickup/van)
- `price_bucket_id`: 8 buckets (по лог-шкале; границы в USD экв.)
Ок или хочешь другое число?

**A3.7**: ок

### 7.4 Quality validation без полной разметки

**Q3.8** Для brand/car/celebs/place предлагаю одинаковую “quality panel”:
- track stability (std/flip rate по top1)
- “event sampling” для ручной проверки: top‑N событий по score и случайные N
Ок?

**A3.8**: ок

**Q3.9** Golden set size: предлагаю начать с **30 видео** (покрыть мульт/лица/бренды/авто/текст/ночь),
и потом расширять до 100.
Ок?

**A3.9**: ок
---

## Навигация

[README](README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
