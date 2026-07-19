# FINAL REPORT — `core_identity/place_semantics`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Часть группы `core_identity` (place/brand/car/content_domain/franchise + face_identity). Общий отчёт валидации —
> [`core_identity/REPORT_2026-07-16.md`](../REPORT_2026-07-16.md). Здесь — place_semantics как отдельный компонент.

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `place_semantics` (VisualProcessor core_identity, semantic head, Tier-2) |
| Версия кода | `0.2` |
| Схема NPZ | `place_semantics_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/place_semantics/place_semantics.npz` |
| Модель | **CLIP image/text** (Triton fp16) + **Embedding Service** (HTTP kNN vs place-DB) |
| Hard deps | `core_object_detections` (frame_indices) + core_clip-эмбеддинги + Embedding Service |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → core_identity ✅ (2026-07-16, авто) |
| Отчёт валидации | [`core_identity/REPORT_2026-07-16.md`](../REPORT_2026-07-16.md), [`CRITERIA.md`](../CRITERIA.md) |
| Код | `core/model_process/core_identity/place_semantics/main.py` + `utils/embedding_service_client.py` |

## 2. Резюме

`place_semantics` — **распознавание конкретных мест/локаций через retrieval**: эмбеддит кадры через CLIP и ищет
похожие места в базе **Embedding Service** (top-5, similarity threshold), группируя совпавшие кадры во временные
**треки** (по одному на распознанное место). Возвращает per-track/per-frame top-K идентификации мест с оценками
и флагами уверенности. Механизм валидирован (схема/ось/empty-путь PASS), **но на 100% реального корпуса компонент
пуст** (`status=empty, no_places_detected`, 0 треков на всех 23 NPZ) — потому что **база мест Embedding Service
фактически стаб** (пространство меток `A=1`), а датасет-видео не содержат мест из базы. golden/длины (U5/U6)
**пропущены** — Embedding Service недоступен локально. Итог: качество распознавания на реальных данных **не
подтверждено ничем**, реального сигнала нет.

## 3. Функционал

Стоит в Tier-2 (semantic head). Пайплайн:

1. Берёт кадры по `core_object_detections.frame_indices`.
2. **CLIP image-эмбеддинг** кадра (Triton fp16).
3. **Embedding Service kNN** — ищет ближайшие места в базе (`place_category="place"`, top-5, `similarity_threshold`).
4. **Temporal tracking** — группирует кадры с одним местом в треки (`min_track_length`, `max_gap_sec`).
5. Возвращает `track_topk_ids/scores`, `frame_topk_*`, флаги `is_confident_top1` (score ≥ threshold_global).

**Зачем продукту:** узнавание **конкретных мест** (не «салон красоты» вообще, а именно узнаваемая локация/
достопримечательность/студия) — нишевый семантический сигнал: тип съёмочной площадки, узнаваемость локаций
(travel/влоги о местах). Model-вход (место как контекст) + аналитика («в видео узнаны места: …»).

## 4. Вход

- **`core_object_detections`** (hard) — `frame_indices` (строгое совпадение оси).
- **CLIP image (Triton)** — эмбеддинги кадров; **core_clip provenance** в meta.
- **Embedding Service** (HTTP) — база place-эмбеддингов (`db_name`, `db_version`, `db_digest`); нет совпадений
  ≥ threshold → `status=empty, no_places_detected`.
- **`union_timestamps_sec`** — ось времени.

## 5. Выход

- **Label-space:** `semantic_label_names (A)` `"id:name"` + `semantic_object_ids (A)` (UUID) + `threshold_per_label_arr (A)`.
- **Per-track (model):** `track_ids (T)`, `track_present_mask`, `track_topk_ids/scores (T,K)`, `track_is_confident_top1`,
  `track_topk_evidence_frame_indices`.
- **Per-frame:** `frame_topk_ids/scores`, `frame_is_confident_top1`.
- **Ось:** `frame_indices`, `times_s`.
- **Empty by design:** нет совпадений → `track_ids=[]`, `num_places=0`, все треки пусты.

## 6. Фичи (важное/неочевидное)

- **Retrieval, а не классификация** — не фикс-набор классов, а kNN против расширяемой базы: новые места
  добавляются в DB без переобучения. `db_digest` в meta фиксирует версию базы (воспроизводимость).
- **Temporal tracking мест** — совпадения группируются в треки (место присутствует в интервале), а не пофреймово.
- **`A=1` — база фактически пуста/стаб** — пространство меток из 1 элемента → сопоставлять не с чем → всегда empty.
  Это корень пустоты (глубже, чем «в видео нет этих мест»).
- **Двойной флаг уверенности** (frame + track `is_confident_top1` по threshold_global) — разделяет «нашлось похожее»
  и «уверенно распознано».
- **Overlap со scene_classification** — тот делает place-**категорию** (Places365: beauty_salon), place_semantics —
  **конкретное место** (retrieval). Категорийный путь работает; named-place — ниша + нет базы.

## 7. Алгоритм / архитектура

- **CLIP image embed (Triton fp16)** → **Embedding Service kNN** (HTTP, cosine top-5) → temporal grouping (numpy).
- Модель не обучается — retrieval против внешней базы; качество ∝ покрытию базы.
- **Детерминизм:** golden **не проверен** (U5 SKIP — сервис недоступен).
- **Версия 0.2** — ранняя, как ocr/text_scoring.

## 8. Оптимизации

- **Retrieval-архитектура** — расширяемость базы без переобучения (добавил место → узнаётся).
- **Reuse CLIP-эмбеддингов** (Triton) — общий с core_clip/scene путь, не отдельная модель.
- **Temporal tracking** — компактный per-track выход вместо пофреймового шума.
- **db_digest provenance** — воспроизводимость привязки к версии базы.

## 9. Слабые места

- **Пуст на 100% реального корпуса (главное).** Все 23 NPZ `no_places_detected`, 0 треков. Корень — **база мест
  Embedding Service = стаб** (`A=1`): нет курируемой базы известных мест → сопоставлять не с чем. Компонент **не
  распознал ни одного места** за весь корпус.
- **Качество не подтверждено ничем** — U5/U6 SKIP (Embedding Service offline), реальных матчей нет → ни точность,
  ни калибровка threshold не проверены. Валидированы только схема/ось/empty-путь.
- **Полная зависимость от внешней базы** — без наполнения place-DB компонент бесполезен; наполнение — зона владельца/инфры.
- **Overlap со scene_classification** — категорийная семантика места уже есть там (Places365); ценность
  place_semantics только в *конкретных* местах, что для большинства YouTube-контента ниша.
- **Инфра-хрупкость** — цепочка Triton CLIP + Embedding Service HTTP; обе должны быть подняты (недоступны при валидации).
- **Версия 0.2** — сырая.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс., блокер] Наполнить place-DB Embedding Service** — сейчас `A=1` (стаб); без курируемой базы известных
   мест компонент пуст навсегда. Определить, какие места вообще нужны продукту (достопримечательности? студии?).
2. **[выс.] Проверить на реальных матчах** (с наполненной базой) — точность retrieval, калибровка threshold, U5/U6.
3. **[сред.] Прояснить границу со scene_classification** — не дублировать категорию; place_semantics = только
   конкретные именованные места, иначе избыточность.
4. **[сред.] Поднять Embedding Service в CI/валидации** — иначе golden/качество непроверяемы.
5. **[низ.] Оценить продуктовую нужность** — стоит ли named-place identity усилий, если ниша узкая.

## 11. Рекомендации по архитектуре / связям

- **Общий Embedding Service для place/brand/car/franchise** — единый retrieval-бэкенд (уже так); наполнять базы
  согласованно.
- **Reuse CLIP-эмбеддингов** с core_clip/scene_classification — не эмбеддить кадры трижды.
- **Связка со scene_classification** — сначала категория (Places365), затем при уверенности — конкретное место (retrieval).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct | 23 NPZ | 23/23 VALID rc=0 | схема корректна |
| U2 ось времени | 23 | fi↑, ts монотонны | ось ок |
| U3 различимость | 23 | empty-паттерн: track_ids=[] | пустой путь корректен |
| U4 expected-empty | 23 | status=empty, track_ids=[] | by design (нет совпадений) |
| U5 golden | — | **SKIP** (сервис offline) | детерминизм не проверен |
| U6 длины | — | **SKIP** | не проверено |
| **Реальный storage (мой прогон)** | **6 видео** | **все empty (no_places_detected), 0 треков, DB A=1** | 0 распознанных мест; база-стаб |

Вывод: **схема/ось/empty валидны, но качество распознавания не подтверждено вообще** — база пуста, реальных
матчей нет, golden не гонялся.

## 13. Интерпретируемость

- **Потенциально отличная** (когда есть база): «в видео узнано место: …» — прямой понятный факт.
- **Сейчас нечего показывать** — все empty. После наполнения базы: список узнанных мест с уверенностью и таймлайном.

## 14. Польза для моделей

**Потенциально нишевая, фактически нулевая.** Конкретное место — контекстный признак (travel/локации), но:
(1) sparse retrieval-выход, (2) на реальных данных пуст на 100% (база-стаб), (3) категорийную семантику места
уже даёт scene_classification. Фактическая польза для моделей = 0; потенциал ограничен нишевостью named-place.

## 15. Польза для аналитиков

**Потенциально нишевая, фактически нулевая.** «Какие узнаваемые места в видео» интересно для travel/обзоров, но
на всём реальном корпусе — пусто, а для большинства контента конкретное место нерелевантно (важнее категория из
scene_classification). До наполнения базы аналитик не получает ничего.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Retrieval-архитектура расширяема, но ниша + пустая база |
| 5. Выход (контракт) | 4 | Чистый per-track/per-frame top-K + provenance базы; sparse |
| 6. Фичи | 2 | Механизм осмыслен, но база A=1 → всё пусто; overlap со scene |
| 8. Оптимизации | 3 | Retrieval-расширяемость, reuse CLIP, db_digest; сервис-зависимость |
| 9. Слабые места (инверсно) | 1 | Пуст на 100%, база-стаб, качество не проверено, overlap, v0.2 |
| 12. Результаты тестов | 2 | Схема/ось/empty ок, но U5/U6 SKIP, 0 матчей |
| 13. Интерпретируемость | 4 | «Узнано место X» понятно (когда есть) |
| 14. Польза для моделей | 2 | Нишевый потенциал, факт=0 |
| 15. Польза для аналитиков | 2 | Нишевый потенциал, факт=0 |

### Итоговые оценки

- **Польза для моделей: 2/5.** Конкретное место — контекстный признак, но выход sparse, база Embedding Service —
  стаб (A=1) → 100% empty, а категорийную семантику места уже покрывает scene_classification. Фактическая польза
  нулевая, потенциал ограничен нишевостью named-place identity.
- **Польза для аналитиков: 2/5.** «Узнаваемые места» ценны для travel/обзоров, но на всём корпусе пусто, а для
  большинства контента конкретное место нерелевантно (важнее place-категория). Балл отражает факт (0 сигнала) и
  узкую нишу, а не абстрактный потенциал.

## 17. Источники

- `core/model_process/core_identity/place_semantics/main.py`, `utils/{embedding_service_client.py, validate_place_semantics_npz.py, render.py}`
- `core/model_process/core_identity/place_semantics/docs/{SCHEMA.md, FEATURE_DESCRIPTION.md}`
- `DataProcessor/docs/component_reports/core_identity/{REPORT_2026-07-16.md, CRITERIA.md}` (общий по группе)
- Cross-ref: `scene_classification` (Places365 категория — overlap), `brand_semantics`/`car_semantics` (тот же retrieval-паттерн), Embedding Service
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/place_semantics/place_semantics.npz`
  (**все status=empty, no_places_detected, 0 треков, place-DB A=1**)

## 18. Визуализации

![place_semantics overview](place_semantics_overview.png)

`place_semantics_overview.png`: слева — все 6 реальных видео `status=empty` (0 распознанных мест; place-DB label
space A=1 = стаб); справа — механизм (CLIP embed → Embedding Service kNN → temporal tracks) и причины пустоты
(пустая база + U5/U6 SKIP) + overlap со scene_classification (категория vs конкретное место). Подтверждает: логика
retrieval валидна, но реальной пользы нет — база не наполнена, качество не проверено.
