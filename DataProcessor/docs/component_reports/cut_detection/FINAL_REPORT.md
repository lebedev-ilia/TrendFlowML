# FINAL REPORT — `cut_detection`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `cut_detection` (VisualProcessor **module**, детектор границ шотов / анализ монтажа) |
| Схемы NPZ | `cut_detection_npz_v1` (features/analytics) + `cut_detection_model_facing_npz_v1` (для Encoder) |
| Артефакты | `cut_detection/cut_detection_features_<ts>_<uid>.npz` + `cut_detection_model_facing_<ts>_<uid>.npz` |
| Движок | **numpy/OpenCV CPU** + reuse CLIP (стилизованные) + reuse RAFT-потока из core_optical_flow |
| Hard deps | `core_optical_flow`, `core_face_landmarks`, `core_object_detections` (baseline required); `core_clip` опц. |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → cut_detection ✅ (2026-07-05) |
| Отчёт валидации | [`REPORT_2026-07-05.md`](REPORT_2026-07-05.md); Audit v4 [`cut_detection_audit_v4.md`](../../audit_v4/components/visual_processor/modules/cut_detection_audit_v4.md) |
| Код | `DataProcessor/VisualProcessor/modules/cut_detection/utils/cut_detection.py` (+ flow_features.py, visual_features.py) |

## 2. Резюме

`cut_detection` — **детектор границ шотов и анализатор монтажа**. На выборке кадров (Segmenter) он ищет
переходы четырёх типов: **hard cut** (склейка — мульти-сигнал hist+SSIM+flow+deep, порог ≥2), **soft**
(fade-in/out, dissolve по яркости/гистограммам), **motion** (whip-pan/zoom/speed-ramp по оптическому
потоку с RANSAC-компенсацией камеры) и **jump cut** (смена позы/лица при похожем фоне). Пишет **два
артефакта**: `features` (75 агрегатов темпа/ритма монтажа + `detections`) для аналитики и `model_facing`
(плотные per-pair кривые + sparse-поток событий с таксономией) для Encoder. Его границы шотов — **база
сегментации** для `scene_classification` и `shot_quality`. Компонент прод-готов по контракту: обе схемы
стабильны на 5 прогонах Audit v4 (L2 ~8.5), в 22 реальных прогонах **flow берётся из core_optical_flow
(reuse RAFT работает)**. Ключевой изъян: **deep-embedding канал полностью выключен (100% NaN во всех 22
видео)** → jump_cut и часть робастности hard_cut деградированы.

## 3. Функционал

Стоит в визуальной цепочке после core-провайдеров (нужны поток/лица/объекты). Детектирует и классифицирует
переходы между кадрами:

1. **Hard cuts** — резкие склейки: комбинация 4 сигналов (L1 HSV-гистограмм, падение SSIM, скачок потока,
   разница deep-эмбеддингов), склейка при счёте ≥2, + морфология/медианный фильтр против ложных.
2. **Soft transitions** — fade-in/out (монотонная яркость HSV/LAB + проверка потока), dissolve (плавная
   гистограмма без скачка яркости).
3. **Motion transitions** — whip-pan (высокая когерентность направления потока), zoom (низкая когерентность
   + высокая магнитуда), speed-ramp (высокая дисперсия потока); RANSAC-homography отделяет движение камеры.
4. **Stylized** — glitch/flash/wipe/slide через CLIP zero-shot (candidate-first: CLIP только на кандидатах).
5. **Jump cuts** — смена позы/лица (MediaPipe) при похожем фоне (deep cosine >0.85, face-ID эмбеддинги).

**Зачем продукту:** это **анализ монтажа и темпа** — как часто режут, какой стиль переходов, динамичный ли
монтаж. Темп монтажа — сильный драйвер удержания (быстрый монтаж ↔ retention у молодой аудитории). Плюс
границы шотов — фундамент для посегментной агрегации (scene/shot_quality) и sparse-сигнал «событий» для
трансформера (attention по моментам склеек).

## 4. Вход

Контракт строгий, **no-fallback**:

- **`metadata["cut_detection"].frame_indices`** (обяз., ≥2) + `union_timestamps_sec` → `times_s`.
- **`core_optical_flow`** (baseline required) — reuse `motion_norm_per_sec_mean` как `flow_mag` (per-pair
  shift); в 22/22 реальных `flow_source=core_optical_flow`. Farneback — только Triton-free fallback.
- **`core_face_landmarks`** + **`core_object_detections`** (baseline required) — для jump_cut (поза/лицо/фон).
- **`core_clip`** (опц.) — стилизованные переходы + hard-cut deep-косинус.
- **run identity** + флаги (`--prefer/require-core-optical-flow`, `--use-deep-features`,
  `--use-adaptive-thresholds`, `--write/require-model-facing-npz`).

## 5. Выход

**Два NPZ** (осознанное разделение):

- **`features` (analytics, `cut_detection_npz_v1`):** boxed dict — `features` (75 агрегатов: hard/soft/motion/
  jump counts, strengths, per-minute, cut-интервалы, ритм) + `detections` (18 ключей: позиции, индексы,
  soft_events, shot/scene boundaries) + `frame_indices/times_s`.
- **`model_facing` (`cut_detection_model_facing_npz_v1`):** для Encoder —
  - **dense per-pair кривые (N−1):** `hist_diff_l1`, `ssim_drop`, `flow_mag`, `hard_score` + valid-маски
    (`ssim/flow/deep_valid_mask`); NaN где сигнал не считался.
  - **sparse события (E):** `event_times_s`, `event_type_id` (таксономия 1=hard_cut…9=jump_cut), `event_strength`,
    `event_pair_index`, опц. `event_contrib_mask`.
  - опц. soft/motion ряды, thresholds, `pair_times_s`, `pair_dt_s`.
- Связь: `model_facing_npz_path` в features-NPZ.

**Таксономия событий:** 1 hard_cut, 2 fade_in, 3 fade_out, 4 dissolve, 5 motion_cut, 6 whip_pan, 7 zoom,
8 speed_ramp, 9 jump_cut, 100+ stylized (CLIP). Справочник в `meta.event_type_map`.

## 6. Фичи (важное/неочевидное)

- **`hard_score` (per-pair 0..N сигналов)** — «сколько детекторов сработало» до порога; сырьё для Encoder
   (не бинарное решение). По Audit v4 диапазон 0…3 (без deep был бы 0…4).
- **flow_mag = reuse core_optical_flow** (не пересчёт!) — в 22/22 реальных `flow_source=core_optical_flow`,
   flow_valid 100%, 0 NaN. Это делает поток **почти бесплатным** (главная оптимизация, §8).
- **deep-канал полностью мёртв** — `deep_valid_mask` all-False в 22/22 видео, `deep_cosine_dist` 100% NaN.
   Значит: (а) hard_cut работает на 3 сигналах вместо 4; (б) **jump_cut деградирован** (нужны face-ID/фон-
   эмбеддинги для «тот же человек, похожий фон») — на реальных данных jump_cut практически не детектится.
- **ssim считается на ~27% пар** (`ssim_valid` mean 0.269) — SSIM-ветка выборочна (по кандидатам/бюджету),
   на остальных `ssim_drop`=NaN + mask=false. Encoder обязан уважать маски.
- **События разрежены и camera-motion-центричны:** 22 видео → 47 событий (2.1/видео), доминируют
   **whip_pan (17), hard_cut (10), zoom (9)**, fade/speed-ramp/dissolve редки. Т.е. на коротком реальном
   контенте больше движений камеры, чем классических склеек.
- **`hard_cuts_per_minute` mean 1.4** (count mean 0.45, max 5) — темп монтажа на этой выборке низкий
   (короткие talking-head ролики).

## 7. Алгоритм / архитектура

- **Движок:** numpy/OpenCV на CPU. Сигналы: HSV-гистограммы (L1), SSIM (на downscale `ssim_max_side`),
   оптический поток (reuse RAFT / farneback fallback), RANSAC-homography (камера vs объекты), CLIP zero-shot
   (стилизованные, candidate-first), MediaPipe поза/лицо + deep-эмбеддинги (jump_cut — сейчас без deep).
- **Порог hard_cut:** сумма триггеров ≥2 + морфологическая очистка + медианный фильтр.
- **Где идёт:** CPU; поток и лица/объекты — из уже посчитанных core-артефактов.
- **Сложность:** при reuse core_optical_flow стоимость ≈ пороги+SSIM+CLIP-кандидаты (дёшево). Farneback-
   fallback — дорог (137 c в Triton-free валидации, НЕ прод-стоимость).

## 8. Оптимизации

- **[сильно] Reuse core_optical_flow (RAFT)** — поток считается один раз (нужен и video_pacing), cut_detection
   читает готовый `flow.npz` → его стоимость ≈0. **В 22/22 реальных артефактах это реально работает**
   (`flow_source=core_optical_flow`). Образцовая экономия; RAFT ещё и качественнее farneback.
- **Candidate-first CLIP** для стилизованных переходов — CLIP только на окнах-кандидатах (отфильтрованных
   дешёвыми сигналами) → 5–20× меньше инференса.
- **Downscale для SSIM/flow/motion** (`ssim_max_side`, `flow_max_side`, 256×256 для motion) — грубые сигналы
   не требуют полного разрешения.
- **Два артефакта** — узкий model_facing для модели vs полный features-dict для аналитики (не грузить модель
   75 агрегатами).
- **valid-маски на каждый сигнал** — честно помечают, где сигнал не считался (NaN), а не подставляют 0.
- **Sparse event-stream с cap** (top-k по strength) — компактный вход для attention.

## 9. Слабые места

- **[выс.] Deep-канал полностью выключен (100% NaN, 22/22).** Следствия: hard_cut на 3 сигналах вместо 4
   (менее робастен); **jump_cut фактически не работает** (требует face-ID + фон-эмбеддинги). Документация
   описывает эти фичи как рабочие — реально они деградированы/пусты. Нужно либо включить deep-эмбеддинги
   (переиспользовать CLIP/face-эмбеддинги), либо честно пометить jump_cut как недоступный.
- **[сред.] SSIM разрежен (~27% пар)** — часть hard-cut робастности теряется; Encoder должен уважать маски.
- **[сред.] Эвристики без калибровки** — пороги (percentile-95 для motion, cosine>0.85 для jump, счёт≥2) не
   валидированы на размеченных склейках; точность/полнота детекции не измерены на ground-truth.
- **[низ.] Имена файлов с timestamp+hash** — усложняют автопоиск артефакта без manifest (Audit v4 ◐);
   опираться на `manifest.json`/`model_facing_npz_path`.
- **[низ.] Golden не гонялся** (§4.8 не закрыт) — детерминизм ожидается по построению (нет случайности, кроме
   RANSAC — а он сидируется), но формально не зафиксирован.
- **Короткие видео / малый N** (медиана 12) — событий мало (2.1/видео), ритм-статистики бедны.
- Отдельного `LOGIC_ERRORS_FOR_CLAUDE.md` L-номера нет.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Включить deep-канал через reuse эмбеддингов** — использовать `frame_embeddings` core_clip как
   deep-сигнал для hard_cut (4-й триггер) и face-ID/фон core_face_landmarks для jump_cut. Эмбеддинги уже
   посчитаны — это почти бесплатно и оживит jump_cut + добавит робастности.
2. **[выс.] Либо честно отключить jump_cut** в контракте, пока deep недоступен, чтобы не заявлять
   несуществующую фичу.
3. **[сред.] Калибровать пороги на размеченном наборе склеек** (хотя бы десятки видео с ручной разметкой
   переходов) — измерить precision/recall hard/soft/motion.
4. **[сред.] Стабилизировать SSIM** — считать на всех парах на downscale (дёшево) вместо выборочных 27%, или
   задокументировать, что SSIM — вспомогательный кандидатный сигнал.
5. **[низ.] Фиксировать golden** и перейти на стабильные имена артефактов (без timestamp) либо жёстко через manifest.

## 11. Рекомендации по архитектуре / связям

- **Reuse core_optical_flow закрепить как единственный прод-путь** (уже работает в 22/22) — farneback только
   для Triton-free/debug с downscale.
- **Deep-сигналы шэрить из core_clip/core_face_landmarks** — не считать эмбеддинги заново, оживить deep-ветку.
- **Границы шотов — единый источник сегментации** для scene_classification/shot_quality; закрепить, что оба
   читают `shot_boundaries` отсюда, а не режут заново.
- **event-stream в общий формат** с другими «событийными» сигналами (Encoder attention по моментам).
- **Aligned sampling group** обязателен (cut_detection на тех же `frame_indices`, что flow/face/objects).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что реально говорит |
|---|---|---|---|
| Output-валидатор (features schema) | тест-видео | VALID | контракт features соблюдён |
| Audit v4 L2 (A+B) | 5 run, N_total=543 | ✓ **~8.5/10** | обе схемы совпадают, model_facing_npz_path связывает NPZ |
| — deep-канал | Audit v4 + мой прогон | **all-NaN (мёртв)** | jump_cut/4-й hard-сигнал не активны |
| — flow-канал | Audit v4 + мой прогон | valid 100%, 0 NaN | reuse core_optical_flow надёжен |
| — ssim-канал | Audit v4 + мой прогон | valid ~25–27% | выборочный сигнал |
| **Реальные артефакты storage (мой прогон)** | **22 run × 2 NPZ** | **flow_source=core_optical_flow 22/22** | reuse-архитектура работает в проде |
| — события | 22 видео | 47 всего, 2.1/видео | whip_pan 17, hard_cut 10, zoom 9, fade 7, … |
| — hard_cuts_per_minute | 22 видео | mean 1.4, max ~5 | низкий темп монтажа (короткие ролики) |
| — features dict | 22 видео | 75 ключей, 0 NaN во float | аналитический слой полон |
| Golden §4.8 | — | ✗ не гонялся | детерминизм ожидается, не зафиксирован |

Вывод: **контракт, reuse-поток и аналитика надёжны** (обе схемы, 22 реальных, flow из RAFT), но **deep-
зависимые детекции (jump_cut, 4-й hard-сигнал) де-факто не работают** — главный разрыв между документацией и
реальностью.

## 13. Интерпретируемость

**Есть (сильно):** `features`-dict — прямые понятные метрики (`hard_cuts_per_minute`, `cuts_per_minute`,
`median_cut_interval`, counts по типам); `render.py`; event-stream с таксономией.

**Добавить (для обычного пользователя):**
- **«Темп монтажа» словами:** «динамичный монтаж, ~15 склеек/мин» / «спокойный, длинные планы».
- **Timeline переходов** с типами (склейка/затемнение/панорама/зум) — визуально «как смонтировано».
- **Стиль монтажа:** «много движений камеры (whip-pan, zoom)» vs «классические склейки» — понятный инсайт.
- **Сравнение с успешными видео ниши** по темпу монтажа.
- Приложенная визуализация (`cut_detection_events.png`) — распределение типов событий + темп.

## 14. Польза для моделей

`model_facing` — **идеальный вход для трансформера**: плотные per-pair кривые (`hist_diff_l1`, `ssim_drop`,
`flow_mag`, `hard_score`) для устойчивой работы на любой длине + sparse event-stream для attention по
моментам склеек. Темп/стиль монтажа правдоподобно **сильно** коррелируют с удержанием (динамичный монтаж ↔
retention). Reuse-поток надёжен. Снижают ценность: мёртвый deep-канал (нет jump_cut, слабее hard_cut),
разреженность SSIM и некалиброванность порогов. Гипотеза: сильный ритмический сигнал, но сейчас неполный.

## 15. Польза для аналитиков

- **Темп монтажа** (`hard_cuts_per_minute`, `cuts_per_minute`, интервалы) → «динамичный/спокойный монтаж» —
   понятный и сравнимый показатель стиля.
- **Типы переходов** (склейки/затемнения/панорамы/зумы) → стиль монтажа креатора.
- **Границы шотов** → структура видео (сколько планов, их длительность).
- Оговорка: jump_cut недостоверен (deep выключен); пороги некалиброваны (подавать как относительное); на
   коротких видео событий мало.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Уникальный анализ монтажа/темпа + база сегментации для scene/shot_quality |
| 5. Выход (контракт) | 4 | Чистое разделение model_facing/analytics, event-таксономия; deep-поля пусты |
| 6. Фичи | 3 | Богатая таксономия, но deep-зависимые (jump_cut) мертвы, SSIM разрежен, без калибровки |
| 8. Оптимизации | 5 | Reuse RAFT реально работает, candidate-first CLIP, downscale, dual-artifact |
| 9. Слабые места (инверсно) | 3 | Deep-канал мёртв (jump_cut), SSIM 27%, golden нет, timestamp-имена |
| 12. Результаты тестов | 4 | L2 8.5, обе схемы, 22 реальных, flow reuse; deep/golden открыты |
| 13. Интерпретируемость | 4 | Темп/типы монтажа очень понятны, словесная подача в TODO |
| 14. Польза для моделей | 4 | Плотные кривые+event-stream — идеальны для attention; deep-дыра |
| 15. Польза для аналитиков | 4 | Темп/стиль монтажа — сильный инсайт; jump_cut недостоверен |

### Итоговые оценки

- **Польза для моделей: 4/5.** `model_facing` (dense curves + sparse events) — почти идеальная форма для
  трансформера (устойчивость к длине + attention по склейкам), а темп монтажа правдоподобно сильно влияет на
  удержание. Снижают мёртвый deep-канал (нет jump_cut, слабее hard_cut) и некалиброванные пороги.
- **Польза для аналитиков: 4/5.** Темп и стиль монтажа — понятный, сравнимый и продуктово-ценный инсайт для
  креатора. Ограничивают недостоверный jump_cut, некалиброванность и разреженность событий на коротких видео.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/cut_detection/utils/{cut_detection.py, flow_features.py, visual_features.py}`
- `.../cut_detection/utils/{validate_cut_detection_npz.py, render.py}`
- `.../cut_detection/docs/{SCHEMA.md, SCHEMA_MODEL_FACING.md, FEATURE_DESCRIPTION.md, FEATURES_DESCRIPTION.md}`, `README.md`
- `DataProcessor/docs/component_reports/cut_detection/REPORT_2026-07-05.md`
- `DataProcessor/docs/audit_v4/components/visual_processor/modules/cut_detection_audit_v4.md`
- Upstream: `core_optical_flow` (flow reuse), `core_face_landmarks`, `core_object_detections`, `core_clip`
- Downstream: `modules/{scene_classification, shot_quality}` (shot boundaries)
- `automation/runner/AGENT_CONTEXT.md` (разделы 6/7: farneback 137c=Triton-free, reuse RAFT прод-путь)
- Реальные артефакты: 22 run × 2 NPZ = 44× `storage/result_store/youtube/*/*/cut_detection/*.npz`

## 18. Визуализации

![Типы событий cut_detection](cut_detection_events.png)

`cut_detection_events.png` (построено на 22 реальных видео, 47 событий): распределение типов монтажных
событий (whip_pan 17, hard_cut 10, zoom 9, fade/speed-ramp/dissolve реже), темп `hard_cuts_per_minute`
(mean 1.4), N кадров и `ssim_valid` ratio (~0.27). Подтверждает: reuse RAFT работает (`flow_source=
core_optical_flow` 22/22), deep-канал мёртв (100% NaN), события разрежены и доминируют движения камеры.
