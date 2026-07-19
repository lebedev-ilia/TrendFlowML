# FINAL REPORT — `core_identity/car_semantics`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Часть группы `core_identity`. Общий отчёт — [`core_identity/REPORT_2026-07-16.md`](../REPORT_2026-07-16.md).
> Родственник [`brand_semantics`](../brand_semantics/FINAL_REPORT.md)/[`place_semantics`](../place_semantics/FINAL_REPORT.md) (тот же retrieval-паттерн).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `car_semantics` (VisualProcessor core_identity, semantic head, Tier-2) |
| Версия кода | `0.2` |
| Схема NPZ | `car_semantics_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/car_semantics/car_semantics.npz` |
| Модель | **CLIP image** (Triton) + **Embedding Service** (kNN vs car-DB make/model) |
| Hard deps | `core_object_detections` (боксы `car`, COCO class 2) + Embedding Service |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → core_identity ✅ (2026-07-16, авто) |
| Отчёт валидации | [`core_identity/REPORT_2026-07-16.md`](../REPORT_2026-07-16.md), [`CRITERIA.md`](../CRITERIA.md) |
| Код | `core/model_process/core_identity/car_semantics/main.py` |

## 2. Резюме

`car_semantics` — **распознавание марки/модели/сегмента автомобилей через crop-based retrieval**: берёт боксы
`car` из детектора (COCO class 2), кропает, эмбеддит CLIP и ищет похожие авто в базе **Embedding Service** (top-5),
возвращая per-track **make/model/segment** с лучшим bbox. В отличие от brand, **детектор-класс работает** (COCO
`car` есть в таксономии — `proposal_class_ids=[2]` резолвится). Но на 100% реального корпуса компонент **пуст**
(`status=empty, no_car_proposals`, 0 треков на всех 26 NPZ) по двум мягче-связанным причинам: (1) датасет-видео —
talking-head/indoor, **машин в кадре нет**; (2) car-DB Embedding Service — **стаб** (label space `A=1`), make/model
сопоставлять не с чем. Это **самый нишевый** из retrieval-компонентов (только авто-контент). Качество не проверено
(U5/U6 SKIP). v0.2.

## 3. Функционал

Tier-2, semantic head, crop-based:

1. Загружает `car`-боксы из `core_object_detections` (класс есть — COCO car=2).
2. Кропает регион авто, **CLIP embed** (Triton).
3. **Embedding Service kNN** vs car-DB (`category=car`, top-5).
4. Группирует в **треки** с best-bbox/evidence.
5. Возвращает `semantic_label_make`/`semantic_label_model` + per-track/frame top-K + segment.

**Зачем продукту:** марка/модель авто — **нишевый признак** для авто-контента (обзоры машин, гонки, тест-драйвы):
какие авто показаны/обсуждаются. Ценно только для узкой вертикали; для общего YouTube-контента нерелевантно.

## 4. Вход

- **`core_object_detections`** (hard) — боксы `car` (COCO class 2, класс присутствует); нет машин → `no_car_proposals`.
- **CLIP (Triton)** — эмбеддинги кропов.
- **Embedding Service** — car-DB (make/model, `db_digest`); нет совпадений → empty.
- **`union_timestamps_sec`** + `frame_indices` — ось.

## 5. Выход

- **Label-space:** `semantic_label_names (A)`, **`semantic_label_make`**, **`semantic_label_model`** (fine-grained),
  `semantic_object_ids`, `threshold_per_label_arr`.
- **Per-detection/track/frame:** `det_topk_*`, `track_topk_ids/scores`, `track_best_bbox_xyxy`/`class_id`/`det_score`/
  `frame_pos`, `frame_topk_*`, флаги `is_confident_top1`.
- **Ось:** `frame_indices`, `times_s`. Empty by design → `track_ids=[]`.

## 6. Фичи (важное/неочевидное)

- **Детектор-класс работает** (в отличие от brand): `proposal_class_ids=[2]` = COCO car резолвится в таксономии.
  То есть блокер car **мягче** — пустота от отсутствия машин в кадре + стаб-базы, а не от отсутствия детектор-класса.
- **make/model/segment** (`semantic_label_make`/`model`) — тонкая гранулярность (не «машина», а «Toyota Camry,
  седан»). Требует богатой car-DB — которой нет (A=1).
- **Crop-based + best-bbox** — точная локализация авто на кадре (для UI overlay).
- **`car-DB A=1` — стаб** — даже при машинах в кадре make/model не с чем сопоставить.
- **Самый узкий scope** — релевантно только авто-вертикали; для большинства контента машин нет by nature.

## 7. Алгоритм / архитектура

- **car-боксы → crop → CLIP embed (Triton) → Embedding Service kNN → temporal tracking** (numpy).
- Retrieval, не обучается; качество ∝ car-DB. **v0.2.** Детерминизм не проверен (U5 SKIP).

## 8. Оптимизации

- **Reuse `car`-детекций** (COCO класс, не нужен свой детектор — плюс vs brand).
- **Crop-based** (не весь кадр) + **best-bbox/evidence** — точность + UI.
- **Reuse CLIP-эмбеддингов** (Triton).
- **db_digest provenance**.

## 9. Слабые места

- **Пуст на 100% корпуса (главное).** Все 26 NPZ `no_car_proposals`, 0 треков. Причины: (1) в датасет-видео нет
  машин (talking-head/indoor); (2) car-DB `A=1` (стаб). Компонент **не распознал ни одной машины** за корпус.
- **Самый нишевый компонент** — make/model релевантно лишь авто-контенту; для общего YouTube — бесполезно by nature.
- **car-DB стаб** — без базы марок/моделей fine-grained retrieval невозможен.
- **Качество не проверено** — U5/U6 SKIP (сервис offline), 0 матчей.
- **v0.2** — сырая.
- **Плюс относительно brand:** детектор-класс есть (COCO car) — единственный из «no_*_proposals», где блокер не в детекторе.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Наполнить car-DB Embedding Service** (make/model) — сейчас `A=1`; без базы марок retrieval невозможен.
2. **[сред.] Проверить на авто-контенте** — прогнать на видео с машинами (детектор-класс уже работает), измерить
   точность make/model, U5/U6.
3. **[сред.] Оценить продуктовую нужность** — узкая вертикаль; стоит ли усилий вне авто-ниши.
4. **[низ.] Fine-grained car-модель** вместо CLIP-retrieval — если авто-вертикаль приоритетна (специализированный
   make/model классификатор точнее generic CLIP).

## 11. Рекомендации по архитектуре / связям

- **Общий Embedding Service + retrieval** с place/brand/franchise; car — единственный с рабочим детектор-классом.
- **Условная активация** — считать car_semantics только когда `car`-детекции есть (экономия на не-авто-контенте).
- **best-bbox → UI** «Toyota Camry здесь» (когда есть база).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct | 26 NPZ | 26/26 VALID | схема ок |
| U2 ось | 26 | fi↑, ts монотонны | ось ок |
| U3/U4 empty | 26 | status=empty, track_ids=[] | пустой путь by design |
| U5/U6 | — | **SKIP** | не проверено |
| **Реальный storage (мой прогон)** | **6 видео** | **все empty (no_car_proposals), 0 треков, car-DB A=1** | 0 машин; детектор-класс OK, но нет машин + стаб-база |

Вывод: схема/ось/empty валидны; детектор-класс (COCO car) работает, но реальных машин в корпусе нет и car-DB
пуста → 0 сигнала; качество make/model не подтверждено.

## 13. Интерпретируемость

- **Потенциально хорошая** (в авто-нише): «в видео: Toyota Camry, седан» + bbox — понятный факт.
- **Сейчас нечего показывать** — всё empty.

## 14. Польза для моделей

**Нишевая, фактически нулевая.** Марка/модель авто — признак только для авто-вертикали; для общего контента
нерелевантен. Sparse retrieval + пусто на 100% (нет машин + стаб-база). Фактическая польза = 0; потенциал самый
узкий из retrieval-группы.

## 15. Польза для аналитиков

**Нишевая, фактически нулевая.** Для авто-каналов «какие машины показаны» ценно, но на общем корпусе — пусто и
by nature нерелевантно большинству. До наполнения car-DB и авто-контента аналитик не получает ничего.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 2 | Узкая авто-ниша; детектор-класс работает, но база пуста |
| 5. Выход (контракт) | 4 | make/model/segment + best-bbox + per-track/frame; sparse |
| 6. Фичи | 2 | Fine-grained make/model продуман, но car-DB стаб + нет машин |
| 8. Оптимизации | 3 | Reuse COCO-car детекций (плюс), crop-based, best-bbox |
| 9. Слабые места (инверсно) | 2 | Пуст на 100%, стаб-база, самая узкая ниша, v0.2 |
| 12. Результаты тестов | 2 | Схема/ось/empty ок, U5/U6 SKIP, 0 матчей |
| 13. Интерпретируемость | 4 | «Toyota Camry здесь» понятно (в нише) |
| 14. Польза для моделей | 2 | Нишевый потенциал, факт=0 |
| 15. Польза для аналитиков | 2 | Нишевый потенциал, факт=0 |

### Итоговые оценки

- **Польза для моделей: 2/5.** Марка/модель авто — признак только авто-вертикали, нерелевантный большинству
  контента; выход sparse и на реальных данных пуст на 100% (нет машин + car-DB стаб). Плюс относительно brand —
  детектор-класс (COCO car) работает, но это не спасает при пустой базе. Самый узкий потенциал из retrieval-группы.
- **Польза для аналитиков: 2/5.** «Какие машины в видео» ценно для авто-каналов, но на общем корпусе пусто и by
  nature нерелевантно. Балл отражает факт (0 сигнала) и узкую нишу.

## 17. Источники

- `core/model_process/core_identity/car_semantics/main.py`, `docs/SCHEMA.md`, `utils/*`
- `DataProcessor/docs/component_reports/core_identity/{REPORT_2026-07-16.md, CRITERIA.md}`
- Cross-ref: `brand_semantics`/`place_semantics` (retrieval-паттерн), `core_object_detections` (car=COCO class 2), Embedding Service
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/car_semantics/car_semantics.npz`
  (**все status=empty, no_car_proposals, 0 треков, car-DB A=1; proposal_class_ids=[2] резолвится**)

## 18. Визуализации

![car_semantics overview](car_semantics_overview.png)

`car_semantics_overview.png`: слева — все 6 реальных видео `status=empty` (no_car_proposals, 0 машин); справа —
механизм (car box COCO-2 → crop → CLIP → Embedding Service kNN make/model → tracks) и причины пустоты: детектор-
класс **работает** (`proposal_class_ids=[2]`, в отличие от brand), но в talking-head/indoor видео машин нет +
car-DB стаб (A=1) + узкая авто-ниша. Подтверждает: логика валидна, но реальной пользы нет и вертикаль узкая.
