# Аудит качества фич и semantic-хедов core_identity (задача 6.1)

Дата: 2026-06-29. Цель — оценить готовность 6 semantic-хедов `core_identity` и
качество их выходных фич, зафиксировать gap'ы и приоритезированный backlog
доработок перед обучением модели популярности.

Методология — 4 оси качества из `docs/FEATURE_QUALITY_PLAYBOOK.md`:
**корректность**, **стабильность**, **различимость**, **предсказательная ценность**.

## 1. Состояние хедов (по фактам репозитория)

| Хед | Схема | FEATURE_DESCRIPTION | Валидатор | Источник эмбеддингов | Прод-база |
|---|---|---|---|---|---|
| `brand_semantics` | v2 | да | `validate_brand_semantics.py` | Embedding Service (CLIP 336) по кропам | ✅ `known_brands` + `known_brands_auto` (в HF) |
| `car_semantics` | v2 | да | `validate_car_semantics.py` | Embedding Service (CLIP) по кропам | ✅ `known_cars` (в HF) |
| `content_domain` | v2 | да | `validate_content_domain.py` | `core_clip` + CLIP text-retrieval (prompts) | ✅ малый набор доменов (`known_domains`) |
| `face_identity` | v2 | да | `validate_core_face_identity_npz.py` | Embedding Service (по face-кропам) | ✅ `known_people` (в HF) |
| `place_semantics` | v2 | да | `validate_place_semantics_npz.py` | `core_clip/embeddings.npz` напрямую (10–50× быстрее HTTP) | ❌ **нет прод-базы** (только `seed_e2e_place_and_franchise.py`) |
| `franchise_recognition` | v2 | да | `validate_franchise_recognition.py` | Embedding Service labels | ❌ **нет прод-базы** (только e2e-seed) |

Хорошая новость: покрытие контрактом (схема v2, FEATURE_DESCRIPTION, валидатор)
есть у **всех 6** хедов; инварианты заданы в `SEMANTIC_HEADS_CONTRACTS_QA.md`
(NPZ = SoT, no-network, no-fallback, shared `frame_indices` от Segmenter/core_object_detections,
idempotency key, `models_used`/`model_signature`).

## 2. Ключевые gap'ы качества

### G1 — нет прод-баз для `place_semantics` и `franchise_recognition` (приоритет: высокий)
В рантайме эти хеды вернут пусто или упадут по no-fallback. Нужны реальные базы
(landmarks/места; франшизы/IP) — собрать по `SEMANTIC_BASES_BUILD_GUIDE.md` и
залить в Embedding Service категории `place`/`franchise`.

### G2 — базы в старом формате `known_*`, а не версионный artifact-package (средний)
Гайд требует `dp_models/bundled_models/semantics/<domain>/<version>/` с
`manifest.json` + `db_digest` (воспроизводимость, stable ids). Сейчас base-данные —
это просто папки изображений (`known_brands/…`), смигрированные в HF. Нужна
конвертация в версионные пакеты + `build_db_manifest.py` + preflight
(`scripts/preflight/check_semantic_bases.py`).

### G3 — неоднородный путь эмбеддингов: HTTP-per-crop vs embeddings-direct (средний)
`place_semantics` уже перешёл на прямое сравнение `core_clip/embeddings.npz`
(10–50× быстрее, без HTTP на кадр). `brand_semantics`/`car_semantics`/`face_identity`
всё ещё ходят в Embedding Service по кропам — это стоимость/латентность на 200k.
Кандидаты на унификацию там, где допустимо (по крайней мере кэширование/батч).

### G4 — предсказательная ценность (ось 4) не измерена (высокий, но зависит от данных)
Ни по одной фиче нет offline feature-importance/ablation против таргета. Это
делается на стабильном `dataset_v1` (зависит от прогона корпуса и таргетов) —
ставим как gate перед включением фичи в модель, не «тихо».

### G5 — различимость/стабильность не прогнаны на корпусе (средний)
Инструмент есть (`tools/batch_runs_feature_report.py`), но нет зафиксированного
прогона по сотням видео: доля NaN/const, дисперсия, повторяемость. До этого
качество фич — «на бумаге».

### G6 — выравнивание под кастомную YOLO-таксономию v1 (зависит от чекпоинта)
Хеды-консьюмеры ждут конкретные proposal-классы: `logo_region`/`text_region`
(brand), `person`/`face_region` (face), `car` (car_semantics). После дроп-ина
дообученной YOLO — сверить `TAXONOMY_V1.yaml` ↔ ожидания хедов, чтобы gating
кропов работал.

### G7 — render/QA-визуализация неоднородна (низкий)
`render.py` в корне есть только у `content_domain`; у остальных рендер описан в
README, но как отдельный файл в корне не лежит (проверить расположение/единый стиль).

## 3. Backlog (приоритезированный)

| # | Задача | Ось | Приоритет | Зависимость |
|---|---|---|---|---|
| Q1 | Собрать прод-базы `place` и `franchise` (+ залить в ES, db_digest) | корректность | высокий | референс-данные |
| Q2 | Прогнать `batch_runs_feature_report` по 200–500 видео, отчёт NaN/const/дисперсия/повторяемость | стабильность, различимость | высокий | прогон корпуса |
| Q3 | Конвертировать базы в `semantics/<domain>/<version>/` + manifest/db_digest + preflight | корректность | средний | — |
| Q4 | Унифицировать brand/car/face на embeddings-direct/кэш (снизить HTTP-стоимость) | стоимость | средний | — |
| Q5 | Offline feature-importance/ablation против таргета на `dataset_v1` | предсказат. ценность | высокий | таргеты/датасет |
| Q6 | Сверка YOLO taxonomy v1 ↔ ожидания хедов (gating кропов) | корректность | средний | YOLO-чекпоинт |
| Q7 | Единый render/QA-дашборд по хедам | — | низкий | — |

Рекомендуемый старт без зависимостей от чекпоинта и таргетов: **Q3** (версионные
пакеты баз — чисто инженерное) и подготовка **Q2** (скрипт-обёртка отчёта), затем
**Q1** (сбор place/franchise). Q5/Q6 — когда будут таргеты и YOLO.

## Прогресс (2026-06-29)

- **Q3 (тулинг) ✅** Реализован недостающий тулинг версионных баз (гайд на него
  ссылался, но его не было):
  - `dp_models/bundled_models/semantics/_tools/build_db_manifest.py` — `manifest.json` + воспроизводимый `db_digest`;
  - `dp_models/bundled_models/semantics/_tools/convert_known_to_semantics.py` — `known_*` → версионный пакет (контиг. id, jsonl, images, manifest); для cars — makes/models/taxonomy;
  - `dp_models/bundled_models/semantics/README.md` — спецификация формата под preflight.
  Проверено на синтетике: brands и cars, id контиг. 0..N-1, `db_digest` стабилен между прогонами.
  - `_tools/build_gallery.py` — сборка `gallery_embeddings.npy` по `images/<id>/` через
    канонический эмбеддер (`EmbeddingManager._get_manager(category).extract_embedding`,
    CLIP/Triton). Запускается на машине с поднятыми Embedding Service + Triton.
  Остаётся (на твоей машине): прогнать конвертер на реальных `known_*` → build_gallery → build_db_manifest → preflight.
- **Q2 — уточнение:** тулинг УЖЕ полный (`feature_quality_audit.py` = coverage/nan_rate/std/constant_like,
  `golden_batch_compare.py` = повторяемость §5, `feature_batch_drift.py` = дрейф §4,
  оркестратор `feature_qa_pipeline.py`). Дублировать не нужно — остаётся **прогнать**
  `feature_qa_pipeline.py` по пилотной пачке реальных run'ов (зависит от данных, не от кода).

## 4. Что проверено для этого аудита
- Прочитаны README/схемы 6 хедов, `FEATURE_QUALITY_PLAYBOOK.md`,
  `FEATURE_COVERAGE_AUDIT.md`, `SEMANTIC_HEADS_CONTRACTS_QA.md`,
  `OBJECT_DETECTIONS_AND_SEMANTICS_ROADMAP.md`, `SEMANTIC_BASES_BUILD_GUIDE.md`.
- Проверено наличие FEATURE_DESCRIPTION + валидаторов по каждому хеду (есть у всех 6).
- Проверено наличие баз: brand/car/domain/face — есть (в HF `trendflow_models`);
  place/franchise — реальной базы нет (только e2e-seed).
