# FINAL REPORT — `core_object_detections`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `core_object_detections` (VisualProcessor **core** provider, Tier-0) |
| Версия кода (`VERSION`) | `2.2` |
| Схема NPZ (`SCHEMA_VERSION`) | `core_object_detections_npz_v2` (прод-контракт; v3+track_ids — экспериментальная ветка, не в контракте) |
| Артефакт | `result_store/<platform>/<video>/<run>/core_object_detections/detections.npz` |
| Модель | **YOLO11x** (ultralytics), канон-вес `yolo11x_41_best.pt` — **41-класс** проектная таксономия; COCO-80 — legacy fallback |
| MAX_DETECTIONS (M) | 100 · box_threshold дефолт 0.6 · person = class 0 |
| Дата разбора | 2026-07-17 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → core_object_detections ✅ (2026-07-05) |
| Отчёт валидации | [`REPORT_2026-07-05.md`](REPORT_2026-07-05.md); Audit v4 [`core_object_detections_audit_v4.md`](../../audit_v4/components/visual_processor/core/core_object_detections_audit_v4.md) |
| Код | `DataProcessor/VisualProcessor/core/model_process/core_object_detections/main.py` (1250 строк) |

## 2. Резюме

`core_object_detections` — **Tier-0 детектор объектов** визуальной цепочки и один из самых широко
потребляемых core-провайдеров. На primary-выборке кадров (владелец — Segmenter) он прогоняет YOLO11x и
выдаёт до M=100 боксов на кадр: `boxes/boxes_norm/centers_norm/areas_frac (N,M,…)`, `class_ids/scores (N,M)`,
`valid_mask (N,M)` плюс продуктовые per-frame агрегаты (person/text/logo counts и площади). Детектор
обучен на **проектной 41-класс таксономии** (люди/толпа, транспорт, гаджеты, одежда, аксессуары, логотипы,
текст, косметика, экраны, еда) — заточенной под контент-анализ креаторов. Его выход **фундаментален**:
person-боксы кормят `action_recognition`, `shot_quality`, `frames_composition`, `detalize_face`,
`micro_emotion`, `behavioral`, `scene_classification`, `cut_detection`. Компонент прод-готов по контракту:
schema v2 стабильна на 5 прогонах Audit v4, golden детерминирован (boxes идентичны), на 26 реальных
артефактах — 0 NaN в scores. **Важная оговорка по данным:** stored-артефакты выданы на COCO-весах (class_id —
COCO-индексы), каноничная 41-таксономия в проде ещё не прогнана массово (§9).

## 3. Функционал

Стоит в начале визуального пайплайна (Tier-0, после Segmenter). Для каждого sampled-кадра YOLO11x находит
объекты и возвращает боксы, классы и уверенности. Компонент:

1. **Пакует детекции в фиксированный тензор** (N кадров × M=100 слотов): `boxes` (пиксельные x1y1x2y2),
   `boxes_norm` (в [0,1]), `centers_norm` (центры), `areas_frac` (доля площади кадра), `class_ids`,
   `scores`, `valid_mask` (какие слоты реальны — паддинг не обнуляется, см. §6).
2. **Считает продуктовые агрегаты** per-frame: `person_count`, `text_region_count`, `logo_region_count`,
   `sum/max_person_area_frac`, `sum/max_text_area_frac`, `sum/max_logo_area_frac`, `det_count`.

**Зачем продукту:** это **«кто и что в кадре»** — фундамент почти всей объектной семантики. Люди в кадре, их
крупность и позиция определяют, можно ли анализировать лица/позы/эмоции/действия (downstream строит person-
клипы и ROI по этим боксам). Продуктовая таксономия (одежда, гаджеты, логотипы, косметика, еда) прямо
поддерживает анализ монетизируемого контента (фэшн/тех/бренды/фуд) и распознавание брендов. Без детектора не
работают action_recognition, shot_quality, detalize_face, micro_emotion и др.

## 4. Вход

Контракт строгий, **no-fallback**:

- **Кадры** — `FrameManager.get(idx)` из `frames_dir`, RGB uint8.
- **`metadata.json.core_object_detections.frame_indices`** (обяз.) — Segmenter-выборка; пусто → error.
- **`metadata.json.union_timestamps_sec`** — ось времени; `times_s = union_timestamps_sec[frame_indices]`.
- **run identity** (обяз.): platform/video/run_id, config_hash, sampling_policy_version, dataprocessor_version.
- **`--batch-size`** (дефолт 16) — кадров на YOLO-инференс. **Вес детектора** — `yolo11x_41_best.pt`
  (канон) либо COCO-fallback; таксономия читается из `DETECTOR_TAXONOMY_V1_40_NAMES.txt` (41 имя).
- **Триtop-путь** (`run_yolo_triton`) либо inprocess ultralytics — оба дают тот же v2-артефакт.

Работает на том же shared-sampling `frame_indices`, что core_clip/depth/optical_flow/face_landmarks.

## 5. Выход

NPZ `detections.npz`, `allow_extra_keys=false`. Классы ключей:

- **model-facing (для Encoder), seq детекций:** `boxes/boxes_norm (N,M,4)`, `centers_norm (N,M,2)`,
  `areas_frac (N,M)`, `class_ids/scores (N,M)`, `valid_mask (N,M)`, `frame_indices/times_s (N,)`. Это
  переменное число объектов на кадр, упакованное в фиксированный M — Encoder читает через `valid_mask`.
- **analytics (per-frame N,):** `det_count`, `person_count`, `text_region_count`, `logo_region_count`,
  `sum/max_person_area_frac`, `sum/max_text_area_frac`, `sum/max_logo_area_frac` — продуктовые счётчики/площади.
- **reference:** `class_names (41,)` — стабильный маппинг `'id:name'` для 0..40.
- **debug:** `meta`, `meta_json` (дубль meta JSON-строкой для cross-venv безопасности).

**Ключевой инвариант:** источник истины валидности слота — **только `valid_mask`**, НЕ порог по score.
Невалидные (паддинговые) слоты **не обнулены**: их `scores` могут быть до ~0.599, `class_ids` произвольны.
Downstream обязан маскировать. `det_count` синхронизирован с `valid_mask` (проверено Audit v4). M=100 фикс.

## 6. Фичи (важное/неочевидное)

- **`valid_mask` — единственный фильтр правды.** Это самая частая грабля контракта: читать
  `boxes/scores/class_ids` без маски = мусор из паддинга (score до 0.599, случайные классы). Порог 0.6
  применяется внутри при формировании маски, но невалидные слоты сохраняют «сырые» значения.
- **`person_count` / `max_person_area_frac`** — самые надёжные агрегаты: person = class 0 **и в
  41-таксономии, и в COCO-80**, поэтому корректны независимо от того, каким весом прогнан детектор. По 26
  реальным видео: person present в среднем в 17% кадров, `max_person_area_frac` mean 0.091 (человек обычно
  занимает ~9% кадра — средний план).
- **`boxes_norm`/`centers_norm`/`areas_frac`** — нормированы на размер кадра → сравнимы между видео разного
  разрешения; `areas_frac` = доля площади кадра (крупность объекта), важна для shot_quality/composition.
- **`text_region_count`/`logo_region_count`** — считаются по class_id таксономии (33=logo_region,
  34=text_region). Работают **только** если детектор прогнан на 41-таксономии; на COCO-весах этих классов
  нет → счётчики нулевые (на 26 реальных артефактах — 0/26, т.к. они на COCO-весах, см. §9).
- **Разреженность детекций:** на реальных данных mean **1.0 детекция/кадр**, **46% кадров = 0 объектов**,
  max 5. Контент часто «пустой» для детектора (talking-head без гаджетов/брендов) — это реальность, не баг.

## 7. Алгоритм / архитектура

- **Модель:** **YOLO11x** (ultralytics, крупнейший вариант) — one-stage anchor-free детектор. Канон-вес
  `yolo11x_41_best.pt` — дообучен на **41-класс проектную таксономию**. COCO-80 (`yolo11l/x.pt`) — legacy
  fallback. Внешняя сеть, fine-tune — зона владельца.
- **Таксономия v1 (41 класс):** person, crowd, car/motorcycle/bicycle/bus/truck, pet, sports_ball,
  phone/laptop/tablet/smartwatch/watch/headphones/camera/microphone/game_controller/tv_device/monitor_device,
  clothing_top/bottom/outerwear/suit/dress/shoes/bag/hat/glasses/ring/bracelet/earrings/pendant,
  logo_region, text_region, cosmetics_product, screen_phone/laptop/monitor, tv_screen, food_item.
- **Инференс:** батчами (дефолт 16); inprocess ultralytics или Triton-путь. Боксы усекаются до M=100/кадр,
  порог 0.6 → valid_mask.
- **Где идёт:** GPU (A4500/A-серия). Стоимость ~34–59 c/видео (YOLO на разреженной выборке); трекер (в
  экспериментальной ветке) добавляет единицы %.
- **Сложность:** линейна по N кадрам; YOLO11x — самый тяжёлый из семейства (точность↑, скорость↓).

## 8. Оптимизации

- **Фиксированный тензор M=100 + valid_mask** вместо ragged-структур — удобно для батч-обработки и Encoder;
  осознанный контракт (паддинг не обнуляется ради скорости — маска обязательна).
- **Продуктовые агрегаты векторизованы** (`np.sum/np.max` по маске классов) — person/text/logo считаются
  numpy-операциями на весь тензор, без циклов по кадрам.
- **Батчинг YOLO (16)** + fp16-опция; canonical yolo11x_41_best для качества, COCO fallback для отладки.
- **`meta_json` дублирует `meta`** JSON-строкой — cross-venv безопасность (object-array numpy может ломаться
  между версиями); прагматичное решение надёжности.
- **appearance-tracker (ветка):** histogram/OSNet-эмбеддер бокса + ассоциация → когерентные person-треки
  (mean 52–127 кадров, frac_single≈0), но **вне прод-контракта v2** (track_ids не в схеме).
- **Атомарная запись NPZ** + пост-валидация схемы.

## 9. Слабые места

- **[критично для семантики] Stored-артефакты выданы на COCO-весах, не на 41-таксономии.** На 26 реальных
  артефактах `class_id` содержит COCO-индексы (67=cell phone, 45=bowl, 58=potted plant, 39=bottle,
  40=wine glass — значения >40 присутствуют в 5+ артефактах), а `class_names`/meta местами заявляют
  41-таксономию → **рассогласование id↔имя**. Следствие: любой downstream, читающий class_id как
  семантику таксономии (logo/text/clothing/cosmetics), получит неверный результат; `text/logo_count`
  нулевые (0/26). **person (class 0) не затронут** — совпадает в обеих таксономиях. Это data-hygiene /
  версионирование весов, а не баг кода, но блокер для продуктовой семантики. Нужно: пере-прогнать корпус на
  `yolo11x_41_best.pt` и гарантировать, что class_names/meta соответствуют реально применённому весу.
- **Паддинг не обнуляется** — `valid_mask` обязателен; невалидные слоты содержат score до 0.599 и случайные
  классы. Известная грабля (зафиксирована в Audit v4 §4.2 как ◐).
- **Нет persistent track_id в контракте v2** — идентичность объекта между кадрами downstream строит сам
  (action_recognition — свои клипы, brand_semantics синтезирует track id «вне контракта»). Appearance-трекер
  реализован, но не интегрирован в схему.
- **YOLO11x дорог** (~34–59 c/видео) — для 200k критично держать прогретый воркер, батч, fp16.
- **Разреженность / малый N** — 46% кадров без детекций + медиана N=12: объектная статистика на коротких
  роликах бедна.
- Отдельного `LOGIC_ERRORS_FOR_CLAUDE.md` L-номера нет; ключевой техдолг — таксономический микс выше.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Пере-прогнать storage на каноничном `yolo11x_41_best.pt`** и жёстко связать `class_ids` ↔
   `class_names` ↔ meta.model — добавить в валидатор проверку `max(valid class_id) ≤ 40` при
   таксономия-name-map (ловить COCO-микс автоматически). Без этого продуктовая семантика недостоверна.
2. **[выс.] Ввести версионирование таксономии в meta** (`taxonomy_version`) + отказ (error), если id
   выходят за диапазон объявленной таксономии — исключить тихий COCO-fallback в проде.
3. **[сред.] Интегрировать appearance-tracker в контракт (schema v3, track_ids)** — когерентные person-треки
   уже доказаны; это упростит action_recognition/behavioral (не строить треки в каждом модуле заново).
4. **[сред.] Обнулять паддинг-слоты** (или хотя бы score=0/class=-1 для valid_mask=False) — убрать грабли
   downstream, снизить риск чтения без маски. Небольшой breaking-change схемы.
5. **[низ.] Адаптивный порог/soft-NMS** для мелких брендов/логотипов — сейчас 0.6 может пропускать мелкие
   logo/text-регионы, важные для монетизации.

## 11. Рекомендации по архитектуре / связям

- **Единый детектор — единый источень боксов** для action_recognition/shot_quality/frames_composition/
  detalize_face/micro_emotion/behavioral: закрепить, что все они читают этот артефакт, а не детектят заново.
- **Трекинг вынести сюда, а не в каждый модуль** — track_ids в core избавит action_recognition/brand от
  дублирования логики ассоциации (сейчас каждый строит свои треки/синтезирует id вне контракта).
- **Shared sampling group** (core_clip/depth/optical_flow/detections/landmarks) гарантировать на Segmenter.
- **face-цепочка зависит от person-боксов** (core_face_landmarks/detalize_face нужны person-регионы) —
  задокументировать порядок Tier: детекции до лиц.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что реально говорит |
|---|---|---|---|
| Output-валидатор (schema v2/v3) | 4 видео | VALID, все поля/формы верны | контракт соблюдён |
| Golden-детерминизм (2026-07-05, GPU A4500) | 2 прогона | **boxes + track_ids идентичны** | детектор (и трекер) детерминированы |
| Трекинг (валидация) | 4 видео | треки mean 52–127 кадров, frac_single≈0 | appearance-трекер когерентен (ветка) |
| Audit v4 L2 (A+B) | 5 run, N_total=543 | ✓ ~8.8/10 | M=100, class_names (41,), det_count↔valid_mask, 0 NaN scores |
| — паддинг | Audit v4 | score невалидных до 0.599 | valid_mask обязателен (◐) |
| **Реальные артефакты storage (мой прогон)** | **26 видео, 591 кадр** | **score_nan=0, det_count↔mask ок** | контракт здоров на проде |
| — детекции/кадр | 591 кадр | mean 1.0, 46% кадров = 0, max 5 | разреженный, реалистичный сигнал |
| — топ-класс | 590 валид. дет. | **person 52%** | человек доминирует в контенте |
| — таксономия | 26 артефактов | **class_id — COCO-индексы (не 41-таксономия)** | ⚠️ семантика text/logo недостоверна |
| — person агрегаты | 26 видео | present 17% кадров, max_area 0.091 | надёжны (class 0 инвариантен) |

Вывод: **контракт и численные инварианты надёжны** (schema, valid_mask, golden, 0 NaN), но **продуктовая
семантика классов на stored-корпусе недостоверна** из-за прогона на COCO-весах — надо пере-прогнать на
41-таксономии. person-ветка — надёжна уже сейчас.

## 13. Интерпретируемость

**Есть:** dev-рендер (`utils/render.py`) — боксы поверх кадров; `class_names` даёт словесные метки;
per-frame агрегаты (person/text/logo counts) человекочитаемы.

**Добавить (для обычного пользователя):**
- **Кадры с нарисованными боксами** (K превью) — «что модель нашла в вашем видео» самое понятное.
- **Словесная сводка:** «в основном: человек крупным планом; замечены: телефон, ноутбук, логотип бренда X».
- **Timeline присутствия человека** (person_count по времени) — «когда в кадре люди».
- **Топ-объекты видео** (частотный список классов) — понятный монетизационный инсайт (сколько брендов/гаджетов).
- Приложенная визуализация (`core_object_detections_distributions.png`) — топ-классы + плотность детекций.

## 14. Польза для моделей

`boxes/centers/areas/class_ids/valid_mask` — model-facing seq-детекций для Encoder; это «сцена объектов»,
которую модель может учитывать наравне с CLIP-эмбеддингами. Наличие людей, их крупность/позиция, набор
объектов (гаджеты/бренды/одежда) правдоподобно коррелируют с типом и привлекательностью контента. Но формат
(переменное число объектов в M=100 с маской) **сложнее** для трансформера, чем плоские ряды optical_flow, и
требует аккуратного пулинга по valid_mask. Продуктовые агрегаты (person_count, area) — более прямой сигнал.
Гипотеза: полезен как **дополнение** (что в кадре), особенно person-присутствие; ценность классовой семантики
раскроется только после прогона на 41-таксономии. Прямых feature-importance данных нет.

## 15. Польза для аналитиков

- **`person_count` / `max_person_area_frac`** → «сколько людей, крупный ли план» — надёжно и наглядно.
- **Топ-объекты / классы** (после 41-таксономии) → бренды, гаджеты, одежда, косметика, еда в кадре —
  прямой монетизационный/контентный инсайт для креатора.
- **`boxes`/превью** → визуальное «что нашла модель».
- **`text_region`/`logo_region`** → плотность текста/логотипов (важно для рекламного/бренд-контента) —
  **после** пере-прогона на каноничном весе.
- Оговорка: на текущем storage-корпусе классовая семантика недостоверна (COCO-веса); person-метрики валидны.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Фундамент объектной семантики, кормит 8+ модулей |
| 5. Выход (контракт) | 4 | Богатый seq+агрегаты; valid_mask-грабля и паддинг не обнулён |
| 6. Фичи | 4 | person/area надёжны, продуктовая таксономия мощна, но семантика требует правильного веса |
| 8. Оптимизации | 4 | Фикс-тензор, векторизация, meta_json, батч; трекер вне контракта |
| 9. Слабые места (инверсно) | 2 | Таксономический микс (COCO вместо 41) недостоверит семантику; паддинг-грабля |
| 12. Результаты тестов | 4 | Golden diff=0, schema стабильна, 0 NaN; но корпус на неканоничном весе |
| 13. Интерпретируемость | 3 | Боксы/классы рендерятся, но словесная/overlay-подача в TODO |
| 14. Польза для моделей | 4 | Уникальная «сцена объектов» + person; формат сложнее, семантика ждёт 41-веса |
| 15. Польза для аналитиков | 4 | person-метрики + топ-объекты очень понятны; классы ждут пере-прогона |

### Итоговые оценки

- **Польза для моделей: 4/5.** Даёт Encoder'у уникальный слой «кто и что в кадре» (люди + продуктовые
  объекты) — потенциально сильный контентный сигнал, особенно person-присутствие. Снижают оценку сложный для
  трансформера формат (M×маска) и то, что классовая семантика раскроется лишь после прогона на 41-таксономии.
- **Польза для аналитиков: 4/5.** person-метрики и (после пере-прогона) топ-объекты/бренды — один из самых
  понятных и монетизационно-ценных выходов для креатора. Ограничивают текущий COCO-микс в storage и пока
  отсутствующая словесная подача.

## 17. Источники

- `DataProcessor/VisualProcessor/core/model_process/core_object_detections/main.py`
- `.../core_object_detections/README.md`, `.../docs/SCHEMA.md`, `.../docs/FEATURE_DESCRIPTION.md`
- `.../core_object_detections/{TAXONOMY_V1.yaml, DETECTOR_TAXONOMY_V1_40_NAMES.txt}`
- `.../core_object_detections/utils/{appearance_tracker.py, validate_core_object_detections_npz.py, render.py}`
- `DataProcessor/docs/component_reports/core_object_detections/REPORT_2026-07-05.md`
- `DataProcessor/docs/audit_v4/components/visual_processor/core/core_object_detections_audit_v4.md`
- Downstream (grep detections.npz/class_ids/person_count): `modules/{action_recognition, shot_quality,
  frames_composition, scene_classification, cut_detection, behavioral, detalize_face, micro_emotion}`
- `automation/runner/AGENT_CONTEXT.md` (разделы 6/7: тайминги 34–59 c, golden diff=0, треки, yolo11x_41_best)
- Реальные артефакты: 26× `storage/result_store/youtube/*/*/core_object_detections/detections.npz` (591 кадр)

## 18. Визуализации

![Распределения core_object_detections](core_object_detections_distributions.png)

`core_object_detections_distributions.png` (построено на 26 реальных артефактах, 591 кадр): топ классов
(person доминирует — 52% валидных детекций; id67/id45/id58 = COCO-индексы, видно таксономический микс),
плотность детекций (46% кадров = 0 объектов), `max_person_area_frac` (крупность человека, mean 0.09) и N
кадров (медиана 12). Подтверждает: контракт здоров (0 NaN), сигнал разрежен и person-центричен, но class_id
на storage-корпусе — COCO, а не каноничная 41-таксономия (data-hygiene риск для семантики).
