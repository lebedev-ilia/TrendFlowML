# FINAL REPORT — `core_identity/brand_semantics`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Часть группы `core_identity`. Общий отчёт валидации — [`core_identity/REPORT_2026-07-16.md`](../REPORT_2026-07-16.md).
> Родственник [`place_semantics`](../place_semantics/FINAL_REPORT.md) (тот же retrieval-паттерн, но crop-based).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `brand_semantics` (VisualProcessor core_identity, semantic head, Tier-2) |
| Версия кода | `0.2` |
| Схема NPZ | `brand_semantics_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/brand_semantics/brand_semantics.npz` |
| Модель | **CLIP-336** (Triton, высокое разрешение для логотипов) + **Embedding Service** (kNN vs brand-DB) |
| Hard deps | `core_object_detections` (боксы `logo_region`/`text_region`) + Embedding Service |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → core_identity ✅ (2026-07-16, авто) |
| Отчёт валидации | [`core_identity/REPORT_2026-07-16.md`](../REPORT_2026-07-16.md), [`CRITERIA.md`](../CRITERIA.md) |
| Код | `core/model_process/core_identity/brand_semantics/main.py` |

## 2. Резюме

`brand_semantics` — **распознавание брендов/логотипов через crop-based retrieval**: берёт боксы `logo_region`/
`text_region` из детектора, кропает их, эмбеддит **CLIP-336** (высокое разрешение под мелкие логотипы) и ищет
похожие бренды в базе **Embedding Service** (top-5), группируя в треки с лучшим bbox/кадром. Продуктово это
**самый ценный из retrieval-компонентов** (спонсорство, product placement, brand-safety — важно для креаторов и
рекламодателей). **Но на 100% реального корпуса компонент пуст** (`status=empty, no_logo_proposals`, 0 треков на
всех 22 NPZ) из-за **двойного блокера:** (1) детектор в батче на **COCO-весах** не имеет классов `logo_region`/
`text_region` → нет логотип-предложений (тот же корень, что у `ocr_extractor`); (2) brand-DB Embedding Service —
**стаб** (label space `A=1`). Качество не подтверждено (U5/U6 SKIP, сервис offline). v0.2.

## 3. Функционал

Tier-2, semantic head. Отличие от place_semantics — **crop-based на детекциях**, не по всему кадру:

1. Загружает `logo_region`/`text_region` боксы из `core_object_detections`.
2. Кропает регион, **CLIP-336 embed** (высокое разрешение — логотипы мелкие).
3. **Embedding Service kNN** vs brand-DB (top-5).
4. Группирует детекции в **треки** с `track_best_bbox_xyxy`/`track_best_det_score`/`track_best_frame_pos`.
5. Возвращает per-detection (`det_topk_*`) + per-track + per-frame top-K брендов.

**Зачем продукту:** узнавание брендов — **прямой коммерческий сигнал**: спонсорские интеграции, product placement,
brand-safety (какие бренды показаны), конкурентный анализ. Ценно и для модели (бренды ↔ монетизация/формат), и
особенно для аналитика/рекламодателя.

## 4. Вход

- **`core_object_detections`** (hard) — боксы `logo_region`/`text_region` (`--proposal-classes`); нет боксов →
  `status=empty, no_logo_proposals`.
- **CLIP-336 (Triton)** — эмбеддинги кропов.
- **Embedding Service** — brand-DB (`db_name`, `db_digest`, `category=brand`); нет совпадений → empty.
- **`union_timestamps_sec`** + `frame_indices` — ось.

## 5. Выход

- **Label-space:** `semantic_label_names (A)`, `semantic_object_ids`, `threshold_per_label_arr`.
- **Per-detection:** `det_topk_ids/scores`, `det_present_mask`, `det_is_confident_top1`.
- **Per-track:** `track_ids`, `track_topk_ids/scores`, `track_best_bbox_xyxy`, `track_best_class_id`,
  `track_best_det_idx/score/frame_pos`, `track_is_confident_top1`.
- **Per-frame:** `frame_topk_ids/scores`, `frame_is_confident_top1`. Ось: `frame_indices`, `times_s`.
- **Empty by design:** нет логотип-предложений/совпадений → `track_ids=[]`.

## 6. Фичи (важное/неочевидное)

- **CLIP-336 (не 224)** — осознанный выбор под мелкие логотипы: выше разрешение → лучше узнавание мелких брендов.
- **Crop-based (по детекциям), не по кадру** — точнее и дешевле, чем эмбеддить весь кадр; но полностью зависит от
  наличия `logo_region`-детектора (которого в COCO-весах нет).
- **`track_best_bbox` + evidence** — для UI: где именно на кадре бренд (можно подсветить). Хорошо для интерпретации.
- **Двойная зависимость** — И детектор логотипов, И база брендов; отказ любого → пусто. Оба сейчас отсутствуют.
- **`brand-DB A=1`** — база брендов стаб (как place-DB); даже при логотип-боксах сопоставлять не с чем.

## 7. Алгоритм / архитектура

- **Detector proposals → crop → CLIP-336 embed (Triton) → Embedding Service kNN → temporal tracking** (numpy).
- Не обучается — retrieval; качество ∝ (покрытие логотип-детектора) × (покрытие brand-DB).
- **Детерминизм:** golden не проверен (U5 SKIP). **v0.2.**

## 8. Оптимизации

- **CLIP-336 для логотипов** (разрешение), **crop-based** (не весь кадр) — точность + экономия.
- **Reuse детекций** core_object_detections (не детектит заново).
- **Temporal tracking с best-evidence** — компактный per-track выход + bbox для UI.
- **db_digest provenance** — воспроизводимость версии базы.

## 9. Слабые места

- **Двойной блокер → пуст на 100% (главное).** (1) `no_logo_proposals`: детектор на COCO-весах не имеет
  `logo_region`/`text_region` (тот же корень, что `ocr_extractor`); (2) brand-DB `A=1` (стаб). Компонент **не
  распознал ни одного бренда** за весь корпус.
- **Качество не подтверждено** — U5/U6 SKIP (сервис offline), 0 матчей → ни точность, ни threshold не проверены.
- **Полная зависимость от двух отсутствующих ресурсов** — logo-детектор (зона владельца) + наполнение brand-DB (инфра).
- **Инфра-хрупкость** — CLIP-336 Triton + Embedding Service HTTP.
- **v0.2** — сырая.
- **Наибольший разрыв «ценность vs факт»** — самый коммерчески-ценный из retrieval-компонентов, но нулевой сигнал.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс., блокер 1] Обучить/подключить `logo_region`-детектор** — общий с ocr_extractor блокер (custom-классы
   детектора отсутствуют в прод-весах); без него нет кропов для brand-retrieval.
2. **[выс., блокер 2] Наполнить brand-DB Embedding Service** — сейчас `A=1` (стаб); нужна база известных брендов/логотипов.
3. **[выс.] Пере-прогнать детектор на 41-весах** (не COCO) — оживит logo_region/text_region (общий с ocr).
4. **[сред.] Проверить точность retrieval** на реальных логотипах (с базой + детектором), U5/U6.
5. **[сред.] Определить продуктовый scope брендов** — какие бренды/категории нужны (спонсоры/масс-маркет).

## 11. Рекомендации по архитектуре / связям

- **Общий Embedding Service + retrieval-паттерн** с place/car/franchise — наполнять базы согласованно.
- **Общий custom-детектор** (logo_region/text_region) для brand_semantics + ocr_extractor + text_scoring — единый
  блокер всей «текст/бренд»-ветки; решать в первую очередь.
- **track_best_bbox → UI overlay** «здесь бренд X» — понятный аналитику вывод (когда живо).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct | 22 NPZ | 22/22 VALID | схема ок |
| U2 ось | 22 | fi↑, ts монотонны | ось ок |
| U3/U4 empty | 22 | status=empty, track_ids=[] | пустой путь by design |
| U5/U6 | — | **SKIP** (сервис offline) | детерминизм/длины не проверены |
| **Реальный storage (мой прогон)** | **6 видео** | **все empty (no_logo_proposals), 0 треков, brand-DB A=1** | 0 брендов; двойной блокер |

Вывод: **схема/ось/empty валидны, но качество не подтверждено вообще** — нет ни логотип-детектора, ни базы.

## 13. Интерпретируемость

- **Потенциально отличная** (когда живо): «в видео показан бренд X на 0:30» + подсветка bbox — прямой понятный
  коммерческий инсайт. `track_best_bbox` уже есть для overlay.
- **Сейчас нечего показывать** — всё empty.

## 14. Польза для моделей

**Потенциально высокая, фактически нулевая.** Бренды ↔ монетизация/спонсорство/формат — коммерчески значимый
признак. Но sparse retrieval-выход + двойной блокер → 100% empty. Фактическая польза для моделей = 0; потенциал
выше, чем у place (бренды релевантнее конкретных мест), но реализуется только после logo-детектора + базы.

## 15. Польза для аналитиков

**Потенциально очень высокая, фактически нулевая.** Детекция брендов/спонсорства/product-placement — одна из
самых ценных аналитик для креаторов и рекламодателей (brand-safety, конкурентный анализ). Но на всём реальном
корпусе — пусто. До оживления детектора и базы аналитик не получает ничего.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Коммерчески ценная роль (бренды), но двойной блокер |
| 5. Выход (контракт) | 4 | Богатый per-det/track/frame + best-bbox для UI; sparse |
| 6. Фичи | 2 | CLIP-336/crop/best-bbox продуманы, но всё пусто (нет детектора+базы) |
| 8. Оптимизации | 3 | CLIP-336, crop-based, reuse детекций; сервис/детектор-зависимость |
| 9. Слабые места (инверсно) | 1 | Двойной блокер, 100% empty, качество не проверено, v0.2 |
| 12. Результаты тестов | 2 | Схема/ось/empty ок, но U5/U6 SKIP, 0 матчей |
| 13. Интерпретируемость | 4 | «Бренд X здесь» + bbox понятно (когда есть) |
| 14. Польза для моделей | 2 | Высокий потенциал, факт=0 |
| 15. Польза для аналитиков | 2 | Очень высокий потенциал, факт=0 |

### Итоговые оценки

- **Польза для моделей: 2/5.** Бренды/спонсорство — коммерчески значимый признак с потенциалом выше, чем у place,
  но выход sparse и на реальных данных пуст на 100% (нет ни logo-детектора, ни brand-DB). Фактическая польза
  нулевая до устранения двойного блокера.
- **Польза для аналитиков: 2/5.** Детекция брендов/product-placement — потенциально одна из самых ценных для
  креаторов/рекламодателей аналитик, но на всём корпусе пусто. Балл отражает факт (0 сигнала), а не потенциал
  (который здесь высок — 4).

## 17. Источники

- `core/model_process/core_identity/brand_semantics/main.py`, `docs/SCHEMA.md`, `utils/*`
- `DataProcessor/docs/component_reports/core_identity/{REPORT_2026-07-16.md, CRITERIA.md}`
- Cross-ref: `ocr_extractor`/`text_scoring` (общий logo/text-детектор блокер), `place_semantics`/`car_semantics` (retrieval-паттерн), Embedding Service
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/brand_semantics/brand_semantics.npz`
  (**все status=empty, no_logo_proposals, 0 треков, brand-DB A=1**)

## 18. Визуализации

![brand_semantics overview](brand_semantics_overview.png)

`brand_semantics_overview.png`: слева — все 6 реальных видео `status=empty` (no_logo_proposals, 0 брендов);
справа — механизм (logo box → crop → CLIP-336 → Embedding Service kNN → tracks с best-bbox) и **двойной блокер**
(нет logo_region-детектора на COCO-весах — как у ocr; brand-DB A=1 стаб) + высокая продуктовая ценность (спонсорство/
brand-safety) при нулевом факте. Подтверждает: логика retrieval валидна, но реальной пользы нет — нет детектора и базы.
