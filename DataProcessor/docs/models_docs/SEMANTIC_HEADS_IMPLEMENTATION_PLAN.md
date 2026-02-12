## Implementation Plan — Object detections + Semantic Heads (v1)

Этот план опирается на зафиксированные решения в:
- `docs/models_docs/SEMANTIC_HEADS_CONTRACTS_QA.md` (Round 1–3, Resolved decisions v0.1)
- `DynamicBatch/docs/DYNAMIC_BATCHING_CHECKLIST.md`
- `Models/docs/contracts/ENCODER_CONTRACT.md` (канонический контракт encoder'а)
- `Models/docs/contracts/MODEL_SYSTEM_RULES.md` (канонические правила моделей)
- `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`

Цель: реализовать “богатую семантику” для popularity‑модели без слабых эвристик, с воспроизводимостью,
контрактами NPZ, валидациями качества и ресурсными таблицами.

---

### 0) Глобальные инварианты (не нарушаем)

- **No-network** (веса/базы/ассеты только локально).
- **No-fallback** на альтернативные модели (если required — fail-fast).
- **NPZ = source-of-truth**, JSON только presentation (кроме `manifest.json`).
- **Time-axis**: `frames_dir/metadata.json.union_timestamps_sec`.
- **Shared sampling group**: head’ы выравниваются по `core_object_detections.frame_indices`.
- **Top‑K=5** всегда сохраняем (scores = cosine similarity).
- **Thresholds per-label** храним, но **не режем** вывод (используем только для `is_confident`/dashboard/encoder).
- **COCO80 vs taxonomy_v1_40**: не смешиваем в одном `class_ids`; отдельные режимы.

---

## 1) Этап 1 — “Бумага → код”: контракты и схемы (без моделей)

### 1.1 Зафиксировать список 8 required компонентов (финально)

Список должен совпадать с реальными именами в коде/папках.
Результат: обновление `SEMANTIC_HEADS_CONTRACTS_QA.md` (раздел 6.1) + ссылки на README каждого компонента.

### 1.2 NPZ контракты для всех head’ов (единый шаблон)

Для каждого head’а:
- `frame_indices (N,) int32`
- `times_s (N,) float32` (алиас к `union_timestamps_sec[frame_indices]`)
- `semantic_label_names (A,) str` (`"id:name"`)
- `threshold_global`, `threshold_per_label` в `meta` + optional `threshold_per_label_arr (A,)`
- `track_ids (T,) int32`
- `track_topk_ids (T,5) int32`, `track_topk_scores (T,5) float32`
- optional det-level: `det_topk_ids (N,MAX,5)`, `det_topk_scores (N,MAX,5)`
- `is_confident_top1 (T,) bool` (и/или per-level masks)
- `meta.models_used[]` + `db_*` поля (см. ниже)

Результат: README/SCHEMA для каждого head’а + (опционально) единый schema registry файл.

### 1.3 “Skeleton mode” для новых head’ов (без базы/модели)

Добавить компоненты с корректным IO/contract, но с `status="empty"` или fail-fast по базе (решим per-head):
- `core_car_semantics`
- `core_face_identity`
- `core_place_semantics` (или `core_landmark_semantics`)

Цель: уже сейчас включать их в DAG, manifest, encoder pipelines и тесты.

---

## 2) Этап 2 — Offline базы (artifact packages) + digest/versioning

### 2.1 Базовая инфраструктура пакетов

Уже начато:
- `dp_models/bundled_models/semantics/README.md`
- `dp_models/bundled_models/semantics/_tools/build_db_manifest.py`

Дальше:
- определить домены и версии:
  - `brands/v1/`, `cars/v1/`, `celebs/v1/`, `places/v1/`
- формат `manifest.json` (db_name/db_version/db_digest/files[])
- CI/скрипт проверки: “manifest соответствует файлам”.

### 2.2 Формат содержимого баз (v1)

- **Brands v1 (500)**:
  - `brands.jsonl`: `{id,name,aliases_en,aliases_ru,prompts_en,prompts_ru}`
  - `prototypes/<id>/*.png|jpg` (optional 1–10 на бренд)
- **Cars v1**:
  - `makes.jsonl`, `models.jsonl`, `taxonomy.json` (segment/body_type/price buckets)
  - (optional) gallery prototypes per make/model
- **Celebs v1 (500)**:
  - `celebs.jsonl`: `{id,name,aliases}`
  - `gallery_embeddings.npy` + mapping (если retrieval)
- **Places v1**:
  - `places.jsonl` + gallery embeddings (CLIP)

### 2.3 Multi-language policy

Канон: id/name на EN, алиасы RU/EN + prompts RU/EN внутри базы.

---

## 3) Этап 3 — `core_brand_semantics` (MVP → v1)

### 3.1 Довести NPZ schema до v1

- добавить `times_s`
- добавить `is_confident` маски
- добавить `db_*` поля в meta (`db_name`, `db_version`, `db_digest`, `db_path`)
- сохранить thresholds (global + per-label)

### 3.2 Алгоритм v1 (без слабых эвристик)

Гибрид (принято):
- A) CLIP text‑matching (prompts RU/EN)
- B) image‑prototype retrieval (галерея логотипов)
- итоговый score: (взвешенная) комбинация или max (решим и зафиксируем)

### 3.3 Cost control / gating (обязательно)

- 1 crop на track (score×area×(optional sharpness))
- только proposals классов `logo_region` (по умолчанию)
- лимиты: max_tracks, max_dets_per_frame

### 3.4 Валидации качества

- track stability метрики
- ручная проверка precision на 50–200 событий
- golden set (30 видео)

---

## 4) Этап 4 — `core_car_semantics` (v1: top‑3 make/model/segment/body_type/price)

### 4.1 Контракт

- top‑3 по каждой “оси” (make/model/segment/body/price) + scores
- единый NPZ (или несколько NPZ per-axis; предпочтительно единый чтобы encoder проще)

### 4.2 Алгоритм v1

Начать с retrieval/CLIP по car crops + структурированная таблица buckets (8 сегментов, 8 price buckets — принято).
Дальше: специализированная модель классификации (позже).

### 4.3 Gating

- track-level на классах `car` (+ optional `truck/bus/motorcycle`)

### 4.4 Валидации

Те же панели качества + отдельный набор примеров “машины/ночь/движение”.

---

## 5) Этап 5 — People: `core_face_identity` (v1: 500 celebs)

### 5.1 Privacy/retention policy (принято)

- не храним face crops
- embeddings только debug+TTL=7d (опционально)
- всегда: top‑K ids+scores

### 5.2 Алгоритм v1

- `core_face_landmarks` даёт face ROIs/landmarks
- дальше: face embedding model (через ModelManager/Triton) + retrieval по gallery embeddings (500)

### 5.3 Валидации

- stability по face-track
- precision panel на golden set

---

## 6) Этап 6 — Scene + place (v1)

### 6.1 `scene_classification` improvements

- улучшить точность Places365 (если нужно — модель/ветки)
- улучшить temporal stability агрегаты

### 6.2 `core_place_semantics` (retrieval)

- CLIP frame embeddings → retrieval по gallery places embeddings
- top‑K места + scores + events (optional)

---

## 7) Этап 7 — Интеграция в DAG / manifest / encoder

### 7.1 DAG

- порядок:
  - `core_object_detections` → brand/car heads
  - `core_face_landmarks` → face_identity
  - `core_clip` → scene/place retrieval
- fail-fast semantics (все required)

### 7.2 Manifest + meta

- каждый компонент пишет `meta` по контракту (`models_used`, `db_*`, thresholds, batch_size, etc.)
- manifest содержит статусы/ошибки/артефакты

### 7.3 FeatureEncoder integration

- encoder читает только NPZ + `union_timestamps_sec`
- строит tokens/aggregates (fixed budgets)

---

## 8) Этап 8 — Тесты и quality gates

### 8.1 Smoke tests

- 1–2 видео: артефакты пишутся, schema валидна, manifest обновлён

### 8.2 Golden/regression

- 30 видео (потом 100)
- tolerances по “стабильности” и top‑K согласованности (best-effort)

### 8.3 Performance/resource

- micro-bench scripts + resource_costs JSON
- обновление `DYNAMIC_BATCHING_CHECKLIST.md`

---

## 9) Этап 9 — Train0 детектора (в параллели с head’ами)

Пока ты размечаешь:
- Train0 (1–2k кадров) → экспорт ONNX → Triton ветки → интеграция `core_object_detections` taxonomy_v1_40.
- Active learning итерации (5–10k) по ошибкам.

---

### Definition of Done (v1)

- Контракты NPZ фиксированы и валидируются.
- Все 8 компонентов работают end-to-end на golden set.
- Нет runtime downloads / нет fallback.
- Есть quality панели и regression.
- Есть resource_costs + чеклист dynamic batching обновлён.

---

## Milestones (пошагово, с “Done” критериями)

### M0 — Contracts locked (v0.1 → v0.2)

- **Done**:
  - `SEMANTIC_HEADS_CONTRACTS_QA.md` содержит:
    - точный список 8 required компонентов (имена совпадают с кодом)
    - финальные NPZ поля (включая `times_s`, thresholds, `is_confident`)
    - COCO80 vs 40 policy зафиксирована

### M1 — Bases infrastructure (offline packages)

- **Done**: ✅
  - есть структура `dp_models/bundled_models/semantics/<domain>/<version>/`
  - для каждого пакета есть `manifest.json` с `db_digest`
  - есть документация "как собирать базы" (`SEMANTIC_BASES_BUILD_GUIDE.md`)

### M2 — Brand head v1

- **Done**: ✅
  - `core_brand_semantics` пишет NPZ по контракту (top‑5 cosine, `times_s`, thresholds, `is_confident`)
  - читает brand base package (db_version/db_digest в meta)
  - реализован в `VisualProcessor/core/model_process/core_brand_semantics/`

### M3 — Car head skeleton → v1 retrieval

- **Done**: ✅
  - `core_car_semantics` существует, пишет NPZ по контракту
  - gating по car tracks работает
  - реализован в `VisualProcessor/core/model_process/core_car_semantics/`
  - база cars v1 подключена (retrieval через CLIP)

### M4 — Face identity skeleton → v1 retrieval

- **Done**: ✅
  - `core_face_identity` пишет NPZ по контракту
  - privacy policy реализована (нет face crops; embeddings только debug+TTL)
  - реализован в `VisualProcessor/core/model_process/core_face_identity/`
  - база celebs v1 подключена

### M5 — Place retrieval v1 + scene_classification improvements

- **Done**: ✅
  - `core_place_semantics` (retrieval) пишет NPZ по контракту
  - реализован в `VisualProcessor/core/model_process/core_place_semantics/`
  - `scene_classification` улучшения интегрированы

### M6 — DAG/manifest/encoder wiring

- **Done**:
  - DAG запускает 8 required компонентов в правильном порядке
  - `manifest.json` содержит статусы/артефакты/версии для всех
  - encoder читает NPZ и формирует токены (fixed budgets) без специальных костылей

### M7 — Quality gates + resource costs

- **Done**:
  - golden/regression на 30 видео с панелями качества (stability + ручная precision проверка)
  - resource_costs json + `DYNAMIC_BATCHING_CHECKLIST.md` обновлены для новых head’ов



