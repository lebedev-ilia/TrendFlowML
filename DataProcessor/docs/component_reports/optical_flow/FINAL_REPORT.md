# FINAL REPORT — `optical_flow` (module)

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».
> **Не путать с `core_optical_flow`** (Tier-0 RAFT-провайдер) — этот компонент его *потребитель*, см.
> [`core_optical_flow/FINAL_REPORT.md`](../core_optical_flow/FINAL_REPORT.md).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `optical_flow` (VisualProcessor `BaseModule`, **не** core, Tier-1 consumer) |
| Версия кода (`VERSION`) | `2.0.2` |
| Схема NPZ (`SCHEMA_VERSION`) | `optical_flow_npz_v3` |
| Артефакт | `result_store/<platform>/<video>/<run>/optical_flow/optical_flow.npz` |
| Модель | **нет собственной** — consumer-only, читает `core_optical_flow/flow.npz` (RAFT). CPU/numpy |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → optical_flow ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Прод-код | `DataProcessor/VisualProcessor/modules/optical_flow/utils/optical_flow.py` (импортируется `main.py`) |
| Схема (машинная) | `DataProcessor/VisualProcessor/schemas/optical_flow_npz_v3.json` |

## 2. Резюме

`optical_flow` — **лёгкий CPU-потребитель** кривой движения `core_optical_flow`. Сам RAFT не считает: он
загружает `core_optical_flow/flow.npz`, **пере-выравнивает** motion-кривую и 15 per-frame статистик потока
на СВОЙ Segmenter-набор `frame_indices` (который может отличаться от core-сэмплинга), собирает плотную
матрицу `frame_feature_values (N,16)` и добавляет **9 video-level табличных агрегатов** (motion mean/median/p90/
variance, missing_frame_ratio, camera shake/rotation/translation, flow_consistency), плюс `ui_payload` для UI.
Это тонкий агрегационный слой поверх core: **никакого нового сигнала он не порождает**, но даёт готовые
video-level фичи (без пулинга) и стабильную привязку к домену модуля. Прод-готов: 100% PASS U1–U6/C1–C4,
golden diff=0 (pure numpy), на 26 реальных артефактах (6 уникальных видео) — валиден, 0 неожиданных NaN.
Наследует все сильные и слабые стороны core (мёртвый `bg_ratio`, NaN на idx0).

## 3. Функционал

Стоит **после** `core_optical_flow` в визуальной цепочке (Tier-1, CPU). Делает три вещи:

1. **Пере-выравнивание (re-alignment).** core считает поток на своём наборе кадров; `optical_flow`
   берёт `frame_indices` СВОЕГО Segmenter-профиля модуля и через `mapping = {core_idx → pos}` переносит
   motion-кривую и 15 статистик на эту ось. Кадры, не покрытые core → `NaN` + учёт в `missing_frame_ratio`.
2. **Video-level агрегация.** Считает 9 табличных фич (mean/median/p90/variance motion-кривой, доля
   пропущенных кадров, среднее дрожание/поворот/панорама камеры, средняя когерентность потока) — готовые
   скаляры на видео, которые аналитик и baseline-модель могут использовать **без** обучаемого пулинга.
3. **UI payload.** Кладёт в `meta.ui_payload` (schema `optical_flow_ui_v1`) кривую движения по времени
   для рендера графика в кабинете.

**Зачем продукту:** core-выход — это «сырьё для Encoder» (dense seq); а `optical_flow` — это **тот же
сигнал в удобной для дашбордов/baseline/UI форме**: одна цифра «насколько динамичное видео», timeline темпа
и профиль камеры. Именно этот артефакт предпочитает читать `text_scoring` как motion-сигнал (core — только
fallback). То есть компонент — «человеко- и таблично-ориентированный фасад» над тяжёлым core-провайдером.

## 4. Вход

Контракт строгий, **no-fallback** по зависимости:

- **`core_optical_flow/flow.npz`** (обяз., hard-dep) — через `load_core_provider("core_optical_flow")`.
  Требуются `frame_indices`, `motion_norm_per_sec_mean` **и все 17 v3-ключей** (15 per-frame статистик +
  `dt_seconds` + `times_s`); отсутствие любого → `RuntimeError` (жёсткая проверка `required_v3`).
- **`metadata.union_timestamps_sec`** (обяз.) — ось времени; `times_s = union_timestamps_sec[frame_indices]`,
  проверка границ и монотонности, no-fallback.
- **`metadata.<module>.frame_indices`** (обяз., ≥1) — Segmenter-выборка ЭТОГО модуля; если пусто → `ValueError`.
  **Важно:** это отдельная секция metadata (`optical_flow: true` в конфиге), не core-выборка — исторический
  баг раннера был именно про перепутанный конфиг (см. §9).
- **run identity** (обяз.): platform/video/run_id, config_hash, sampling_policy_version.
- **`frame_manager`** создаётся, но **игнорируется** (`# consumer-only`) — кадры не читаются.

Пропагация пустоты: если `core.meta.status == "empty"` → модуль возвращает валидный empty-NPZ (все 16
per-frame имён + `(N,16)` NaN, 9 агрегатов = NaN кроме `missing_frame_ratio=1.0`, `status=empty`).

## 5. Выход

NPZ `optical_flow.npz`, все ключи `model_facing`:

- **Ось (N,):** `frame_indices` (int32, union-domain), `times_s` (float32), `motion_norm_per_sec_mean`
  (float32) — тот же сигнал, что в core, только пере-выровненный.
- **Per-frame матрица:** `frame_feature_names (16,)` + `frame_feature_values (N,16)` — плотная копия 16
  per-frame рядов core (motion, mag std/p95, dx/dy, dir sin/cos/dispersion, div, consistency, cam
  scale/rotation/tx/ty/shake, bg_ratio), выровненная на ось модуля. NaN там, где кадр не покрыт core.
- **Video-level агрегаты:** `feature_names (9,)` + `feature_values (9,)` — фиксированный набор:
  `motion_curve_{mean,median,p90,variance}`, `missing_frame_ratio`, `cam_shake_std_mean`,
  `cam_rotation_abs_mean`, `cam_translation_abs_mean` (= √(tx²+ty²)), `flow_consistency_mean`. Считаются с
  `nan*`-функциями и **отбросом первого элемента** (idx0 = нет предыдущего кадра).
- **`meta.ui_payload`** (опц.): кривая движения по времени для UI-графика.

Ключевое отличие от core: core даёт ещё debug-preview теплокарты (K=10, 64×64) — здесь их **нет**
(consumer их не тянет). Зато здесь есть готовые video-level скаляры, которых нет в core.

## 6. Фичи (важное/неочевидное)

- **`motion_curve_mean` — несущий агрегат.** Среднее motion-кривой (без idx0). На 6 уникальных реальных
  видео per-video 0.013…0.307, **CV=0.80** (на 26 прогонах с дублями core-числа совпадают, CV=1.36 —
  завышен повторами одного ролика). Это самый различимый сигнал компонента: динамика vs статика.
- **`motion_curve_p90` / `motion_curve_variance`** — «пики движения» и разброс темпа: у экшн-роликов
  высокие, у talking-head — низкие. Полезнее среднего для детекции «есть ли резкие сцены».
- **`flow_consistency_mean` ∈[0,1]** — средняя когерентность/плавность потока; реально 0.66…0.88.
  Прокси «плавное движение камеры vs хаотичный поток» (качество съёмки/монтажа).
- **`cam_*` агрегаты** — усреднённый профиль движения камеры: shake (дрожание/штатив), rotation (наклон),
  translation √(tx²+ty²) (панорама). Наследуются из affine-RANSAC core.
- **`missing_frame_ratio`** — **уникальная фича этого модуля** (нет в core): доля кадров модуля, не
  покрытых core-сэмплингом. На реальных данных = **0.0 везде** (сэмплинги совпадают) → в норме это
  «здоровье выравнивания», при рассинхроне сэмплеров стало бы >0.
- **`bg_ratio` (col 15) ≡ 0.40 by design** — унаследованная от core мёртвая фича, на всех 26 артефактах
  ровно 0.40 (CV≈0). Занимает столбец матрицы, для модели/аналитика бесполезна (см. §9/§10).

## 7. Алгоритм / архитектура

- **Модель:** нет. Чистый numpy/CPU. Вся «тяжесть» (RAFT) — в `core_optical_flow`.
- **Алгоритм:** (1) `_times_s_from_union` — построение оси времени с проверкой границ/монотонности;
  (2) `_load_core_optical_flow` — загрузка + жёсткая проверка 17 v3-ключей; (3) `mapping`-цикл по кадрам
  модуля — перенос кривой и 15 статистик, NaN на непокрытых; (4) `np.nan*`-агрегация 9 скаляров с
  отбросом idx0; (5) сборка `(N,16)` матрицы + `ui_payload`.
- **Сложность:** O(N) по кадрам модуля, один проход + несколько векторных редукций. `process_ms` ≈ 2–2.4 с
  на видео (доминирует загрузка NPZ и FrameManager, не арифметика). Golden — pure numpy, детерминизм тривиален.
- **Где идёт:** CPU, `supports_batch=True` (последовательный per-video цикл в `process_batch`).

## 8. Оптимизации

- **Consumer-only reuse** — главное архитектурное решение: не считать RAFT второй раз, а переиспользовать
  готовый core-артефакт. Осознанное (в докстринге прямо: «MUST NOT compute RAFT itself»).
- **Пере-выравнивание через dict-mapping** O(N) вместо повторного инференса — дёшево переносит сигнал на
  любую сэмпл-сетку модуля.
- **NaN-tolerant агрегация** (`nanmean/nanmedian/nanpercentile/nanvar`) — устойчива к пропущенным кадрам и
  idx0, не роняет всю статистику из-за одного NaN.
- **`ui_payload` выносится из results в `meta`** в `run()` (`results.pop("ui_payload")`) — не хранится
  тяжёлый JSON как top-level ключ NPZ.
- **`models_used` best-effort пробрасывается из core-meta** для provenance (consumer не знает модель сам).
- Всё это — необходимость (модуль по определению тонкий), а не глубокая оптимизация.

## 9. Слабые места

- **Нулевой value-add по сигналу.** Компонент не добавляет новой информации сверх core: motion-кривая и 16
  рядов — копия. Реальная добавка — только 9 агрегатов + UI payload + пере-выравнивание. Возникает вопрос,
  зачем это отдельный NPZ, а не часть core (см. §11).
- **Мёртвый `bg_ratio` унаследован** — ≡0.40 на всех данных; занимает столбец матрицы, фича-мусор (как в core).
- **Легаси-дубль кода.** В модуле лежат ДВЕ реализации: прод `utils/optical_flow.py` (consumer) и
  `core/optical_flow.py` (старый self-computing RAFT-пайплайн, 520+ строк, `OpticalFlowProcessor`, torch/cv2).
  Легаси не используется (`main.py` импортирует `utils/`), но путает и тянет лишние зависимости — техдолг на удаление.
- **NaN на коротких видео.** idx0-строка (16 NaN) by design; на медианном реальном ролике **N=12** это
  1/12 ≈ **7.8% NaN** в `frame_feature_values` — ниже гейта U3 «≥98% finite» (гейт валидировали на N=34+,
  где 1 строка = 2.9%). Не баг, но на коротком корпусе finite-доля деградирует, и часть агрегатов (var/p90)
  считается по ~11 точкам — статистически шумит.
- **Хрупкость к рассинхрону сэмплеров.** Если Segmenter даст модулю кадры вне core-набора,
  `missing_frame_ratio`>0 и агрегаты поедут на NaN. Сейчас держится тем, что сэмплинги совпадают (ratio=0),
  но это неявный контракт, а не гарантия.
- **История багов (L-уровня нет, но 3 бага при валидации):** конфиг раннера (перепутан
  `visual_core_optical_flow_only`→`visual_optical_flow_only`), empty-путь не возвращал все схемные ключи
  (frame_feature_* и 4/9 feature_names). Все исправлены 2026-07-16 (см. AGENT_CONTEXT §7, REPORT).
- **Корпус беден:** 26 артефактов = **6 уникальных видео** (один ролик `-Q6fnPIybEI` × 21 идентичный прогон);
  реальная различимость меряется на 6 точках.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Удалить легаси `core/optical_flow.py`** (+ `core/camera_motion.py`, `advanced_features.py`,
   `flow_statistics.py`, `config.py`, если они обслуживают только его) — мёртвый self-computing путь путает
   и тянет torch/torchvision в CPU-only модуль. Оставить один consumer-файл.
2. **[выс.] Убрать `bg_ratio` из матрицы** (или починить в core и здесь синхронно) — мёртвый столбец,
   0 информации для 15/16 → 16/16 полезных фич.
3. **[сред.] Добавить агрегаты, которых нет в core-выходе, но которые дают value-add:** например
   `motion_p95_over_median` (импульсивность темпа), `cam_static_ratio` (доля кадров с почти нулевым движением
   камеры = «штатив/статика»), `motion_curve_slope` (нарастает/спадает динамика к концу). Это оправдало бы
   существование отдельного модуля как «аналитического слоя».
4. **[сред.] Явно обрабатывать короткие N.** При N<~5 помечать агрегаты флагом `low_confidence` в meta
   (var/p90 по <4 точкам ненадёжны), чтобы дашборд не показывал шум как факт.
5. **[низ.] Не дублировать все 16 per-frame рядов**, если Encoder всё равно берёт их из core — хранить
   в `optical_flow.npz` только то, что реально нужно UI/аналитике (кривая + агрегаты), уменьшив NPZ.

## 11. Рекомендации по архитектуре / связям

- **Главный вопрос: нужен ли отдельный модуль?** Сейчас `optical_flow` — тонкий фасад над core.
  Два жизнеспособных пути: (а) **слить агрегацию в core_optical_flow** (core и так считает все ряды —
  добавить 9 агрегатов + ui_payload туда, убрать отдельный модуль); либо (б) **осознанно оставить как
  «аналитический слой»**, но тогда наполнить его value-add фичами из §10.3, иначе это дубль NPZ.
- **Закрепить общий Segmenter-домен** core↔optical_flow, чтобы `missing_frame_ratio` гарантированно был 0,
  а не «повезло». Иначе пере-выравнивание — источник тихих NaN.
- **`text_scoring` уже зависит** от `optical_flow.npz` как предпочтительного motion-источника (core — fallback):
  это фиксирует роль модуля как «стабильного публичного motion-артефакта». Если сливать в core — не сломать
  этот контракт (`text_scoring._load_motion_signal_optional`).
- **UI payload** — единый формат timeline движения с preview-теплокартами core → общий backend-рендер «темп видео».

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что реально говорит |
|---|---|---|---|
| U1 validate `--struct --ranges` | 3 видео + empty | **rc=0 на всех** | контракт/оси/диапазоны валидны |
| U2 ось времени | 3 видео | times_s монотонны, из union | ось корректна |
| U3 finite ≥98% | N=112:99.16%, N=250:99.63%, N=34:97.24%* | PASS (by-design) | NaN только idx0; на N=12 деградирует до ~92% |
| U4 expected-empty | core.status=empty | status=empty, (34,16) all-NaN, rc=0 | empty-путь чинён, все ключи есть |
| U5 golden | 3 прогона | **diff=0** (pure numpy) | детерминизм тривиально держится |
| U6 разные длины | N=34/112/250 | PASS | работает на разных длинах |
| C1 CV motion_curve_mean | тест-корпус | **CV=0.40** (порог 0.20) | различает динамику; на 6 реальн. CV=0.80 |
| C2 столбцы не константа | корпус | min std=0.049; bg_ratio исключён | матрица информативна кроме bg_ratio |
| C3 диапазоны агрегатов | 3 видео | missing_ratio=0, motion≥0, consist∈[0,1] | численно здоров |
| C4 выравнивание | 3 видео | **missing_count=0** | сэмплинги совпадают |
| **Реальные артефакты storage (мой прогон)** | **26 NPZ, 6 уникальных видео** | валидны, 0 неожид. NaN | прод-путь пишет корректно |
| — motion per-video | 6 уникальных | mean-разброс 0.013…0.307, CV=0.80 | различает, но корпус беден (6 точек) |
| — bg_ratio | 26 | ровно 0.400 всегда | подтверждает «мёртвая, by design» |
| — NaN share vs N | 26 | 0.8% (N=119) … 7.8% (N=12) | idx0-строка доминирует на коротких |

*U3 N=34: 97.24% < 98% — idx0 by design, задекларировано в CRITERIA.

Вывод: как **тонкий детерминированный агрегатор** компонент надёжен (все гейты PASS, golden=0, реальные
артефакты валидны). Но тесты меряют по сути корректность *переноса* сигнала core, а не собственную ценность —
её у модуля мало.

## 13. Интерпретируемость

**Есть:** `ui_payload` (кривая движения по времени) + `render.py` (dev-рендер); 9 понятных video-level
агрегатов с говорящими именами.

**Добавить (для обычного пользователя):**
- **Одна фраза словами** из `motion_curve_mean`+`p90`: «динамичное видео с резкими сценами» / «спокойный
  статичный кадр» — прямо в кабинет.
- **Timeline темпа** (уже есть в ui_payload) с отметкой пиков = резкие сцены/склейки.
- **Профиль камеры словами** из `cam_shake/rotation/translation`: «штатив / панорама / дрожащая съёмка с рук».
- **Скрыть `bg_ratio` и `missing_frame_ratio=0`** из пользовательского вида (технические/константы).
- Приложенная визуализация `optical_flow_distributions.png` — пример подачи распределений.

## 14. Польза для моделей

Умеренная и **дублирующая core.** Per-frame матрица `(N,16)` — копия core-рядов, которые Encoder и так
получает из `core_optical_flow` (перечислены в FEATURE_ENCODER_CONTRACT). Реальная добавка для модели — **9
готовых video-level агрегатов**: для baseline (XGBoost/LightGBM), которому нужны табличные фичи без пулинга,
это удобная форма — «средний/пиковый темп, дрожание камеры, когерентность» одной строкой на видео. Для
трансформер-Encoder ценности почти нет (он берёт seq из core). `missing_frame_ratio` полезен как флаг
здоровья данных. `bg_ratio` — мёртвый вход. Прямых feature-importance нет (модель в разработке).

## 15. Польза для аналитиков

**Это главный потребитель компонента.** Именно `optical_flow` даёт аналитику готовые скаляры:
- **`motion_curve_mean/p90`** → «насколько динамичное видео / есть ли резкие сцены» — сравнимо между роликами.
- **`cam_shake/rotation/translation_abs_mean`** → профиль движения камеры (штатив/панорама/дрожание) — понятный съёмочный инсайт.
- **`flow_consistency_mean`** → «плавно vs хаотично» — прокси качества съёмки/монтажа.
- **`motion_curve_variance`** → «ровный темп vs рваный монтаж».
- **UI timeline** → визуальная кривая темпа по видео.
Оговорки: `bg_ratio` показывать не стоит (константа); на коротких видео (N≈12) var/p90 шумят; корпус пока 6 видео.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Полезная роль (агрегатор+UI-фасад), но целиком производна от core; собственного сигнала нет |
| 5. Выход (контракт) | 4 | Чистый model_facing NPZ, 9 понятных агрегатов, UI payload; дублирует 16 рядов core |
| 6. Фичи | 3 | Агрегаты motion/cam/consistency полезны; bg_ratio мёртв, per-frame — копия core |
| 8. Оптимизации | 4 | Правильный consumer-reuse, NaN-tolerant, O(N); но легаси-дубль кода рядом |
| 9. Слабые места (инверсно) | 3 | Нулевой value-add сигнала, легаси-дубль, bg_ratio, NaN на коротких, беден корпус |
| 12. Результаты тестов | 4 | Все U/C PASS, golden=0, 26 реальных артефактов валидны; тесты меряют перенос, не ценность |
| 13. Интерпретируемость | 3 | UI payload + понятные агрегаты есть; словесная подача и чистка в TODO |
| 14. Польза для моделей | 3 | Табличные агрегаты удобны baseline'у; для Encoder — дубль core |
| 15. Польза для аналитиков | 4 | Готовые «темп/камера/плавность» скаляры + timeline — наглядно и сравнимо |

### Итоговые оценки

- **Польза для моделей: 3/5.** Даёт baseline-модели удобные табличные агрегаты движения (средний/пиковый
  темп, камера, когерентность) одной строкой на видео. Но для трансформер-Encoder это дубль сигнала, который
  тот и так берёт из `core_optical_flow`; собственной информации модуль не добавляет, `bg_ratio` мёртв.
- **Польза для аналитиков: 4/5.** Это и есть основное назначение компонента: «насколько динамичное видео»,
  профиль движения камеры, плавность потока и timeline темпа — наглядные, сравнимые между роликами метрики,
  готовые к дашборду. Ограничивают только мёртвый bg_ratio, шум на коротких видео и бедный корпус (6 видео).

## 17. Источники

- `DataProcessor/VisualProcessor/modules/optical_flow/utils/optical_flow.py` (прод-consumer, 2.0.2)
- `DataProcessor/VisualProcessor/modules/optical_flow/main.py` (CLI)
- `DataProcessor/VisualProcessor/modules/optical_flow/core/optical_flow.py` (**легаси** self-computing RAFT — dead)
- `DataProcessor/VisualProcessor/modules/optical_flow/docs/{SCHEMA.md,FEATURE_DESCRIPTION.md}`
- `DataProcessor/VisualProcessor/modules/optical_flow/utils/{validate_optical_flow_npz.py,render.py,analyze_all_results.py}`
- `DataProcessor/docs/component_reports/optical_flow/{REPORT_2026-07-16.md,CRITERIA.md}`
- `DataProcessor/docs/component_reports/core_optical_flow/FINAL_REPORT.md` (провайдер, cross-ref)
- Downstream: `DataProcessor/VisualProcessor/modules/text_scoring/utils/text_scoring.py`
  (`_load_motion_signal_optional`: optical_flow.npz предпочтителен, core — fallback)
- `DataProcessor/scripts/run_optical_flow_module_local.py`; `configs/visual_triton_baseline_gpu_local.yaml`
- `automation/runner/AGENT_CONTEXT.md` §7 (урок 2026-07-16: empty-путь должен вернуть все схемные ключи)
- Реальные артефакты: 26× `storage/result_store/youtube/*/*/optical_flow/optical_flow.npz` (6 уникальных видео)

## 18. Визуализации

![Распределения optical_flow](optical_flow_distributions.png)

`optical_flow_distributions.png` (6 уникальных реальных видео, Triton raft_256 → CPU-агрегация):
`motion_curve_mean` per-video (CV=0.80 — различает динамику/статику), NaN-share vs N (idx0-строка by design
доминирует на коротких клипах, на N=12 ≈7.8% > гейта 2%), `flow_consistency_mean` (0.66–0.88) и timeline
темпа примера. Подтверждает: сигнал = переaligned копия core, bg_ratio мёртв, короткие видео повышают долю NaN.
