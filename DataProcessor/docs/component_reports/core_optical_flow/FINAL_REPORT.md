# FINAL REPORT — `core_optical_flow`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `core_optical_flow` (VisualProcessor **core** provider, Tier-0) |
| Версия кода (`VERSION`) | `2.2` |
| Схема NPZ (`SCHEMA_VERSION`) | `core_optical_flow_npz_v3` |
| Артефакт | `result_store/<platform>/<video>/<run>/core_optical_flow/flow.npz` |
| Модель | **RAFT** (torchvision `raft_small`/`raft_large`), прод-путь Triton `raft_{256,384,512}` ONNX; дефолт `raft_256_small` |
| Дата разбора | 2026-07-17 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → core_optical_flow ✅ (2026-07-12) |
| Отчёт валидации | [`REPORT_2026-07-12.md`](REPORT_2026-07-12.md), [`CRITERIA.md`](CRITERIA.md); Audit v4 [`core_optical_flow_audit_v4.md`](../../audit_v4/components/visual_processor/core/core_optical_flow_audit_v4.md) |
| Код | `DataProcessor/VisualProcessor/core/model_process/core_optical_flow/main.py` (953 строки) |

## 2. Резюме

`core_optical_flow` — **Tier-0 провайдер кривой движения** визуальной цепочки. На той же primary-выборке
кадров, что и остальные core-провайдеры (владелец выборки — Segmenter), он прогоняет RAFT по **парам
соседних кадров** и извлекает нормированную по времени и размеру кадра метрику движения
`motion_norm_per_sec_mean (N,)` плюс богатый набор per-frame статистик потока: разброс/p95 магнитуды,
направление (взвешенные sin/cos + дисперсия), дивергенция/консистентность, аффинные прокси движения камеры
(scale/rotation/tx/ty/shake) и `bg_ratio`. Его motion-кривую **жёстко потребляют** `video_pacing`
(no-fallback) и `cut_detection` (reuse вместо своего farneback), а per-frame ряды идут в Encoder. Компонент
прод-готов: 100% PASS всех гейтов H1–H6/C1–C4, golden детерминирован (diff=0 после фикса RANSAC-seed), на
26 реальных Triton-артефактах — 0 NaN (кроме idx0 by design), consistency-идентичность держится на 3e-8.

## 3. Функционал

Стоит в начале визуального пайплайна (Tier-0, после Segmenter). Для каждой пары соседних sampled-кадров
(i−1, i) RAFT выдаёт плотное поле смещений (dx, dy на пиксель), из которого компонент считает:

1. **Кривую движения** `motion_norm_per_sec_mean` — среднюю величину потока, **нормированную на dt и на
   max(H,W)**, чтобы сравнивать между видео с разным fps/разрешением.
2. **Компактные статистики потока** — распределение магнитуды (std/p95), доминирующее направление и его
   когерентность, дивергенцию (зум/расширение) и консистентность поля.
3. **Аффинные прокси движения камеры** — оценивает глобальное движение фона (RANSAC-affine): масштаб (зум),
   поворот, панорамирование tx/ty, дрожание камеры (shake).

**Зачем продукту:** это **единственный источник динамики/темпа** в системе — насколько видео «живое»,
статичный ли это talking-head/слайды или экшн с быстрым монтажом и движением камеры. Темп и динамика —
сильные драйверы удержания и вовлечённости (а значит просмотров/лайков), а также прямое сырьё для темпа
(`video_pacing`) и детекции склеек (`cut_detection`, который переиспользует этот RAFT вместо своего потока).

## 4. Вход

Контракт строгий, **no-fallback**:

- **Кадры** — `FrameManager.get(idx)` из `frames_dir`, RGB uint8.
- **`metadata.json.core_optical_flow.frame_indices`** (обяз., **≥2 кадров**) — Segmenter-выборка; при
  `<2` кадрах штатный `RuntimeError` (пар нет — движения нет).
- **`metadata.json.union_timestamps_sec`** (обяз.) — ось времени; `times_s = union_timestamps_sec[frame_indices]`,
  строгий no-fallback + проверка монотонности; `dt_seconds[1:] = max(diff(times_s), 1e-6)`.
- **run identity** (обяз.): platform/video/run_id, config_hash, sampling_policy_version, dataprocessor_version.
- **`--batch-size`** — число **пар кадров** на Triton-запрос (дефолт 16; для unit-cost = 1).
- **Triton** (runtime строго `triton`): `--triton-model-spec` (2-input модель, `infer_two_inputs`) через
  ModelManager либо явные url/name. Preset `raft_{256,384,512}`. Валидация шла inprocess-обходом (Вариант A).

Работает на том же shared-sampling `frame_indices`, что core_clip/depth/detections/landmarks.

## 5. Выход

NPZ `flow.npz`, `allow_extra_keys=false`. Оси N (по кадрам) и K (preview-пары):

- **model-facing (для Encoder), seq (N,):** `motion_norm_per_sec_mean`, `flow_mag_std/p95_per_sec_norm`,
  `flow_dx/dy_mean_per_sec_norm`, `flow_dir_sin/cos_mean`, `flow_dir_dispersion`, `cam_affine_scale/rotation`,
  `cam_tx/ty_per_sec_norm`, `cam_shake_std_norm`, плюс `frame_indices`, `times_s`. Это готовый dense
  time-series — **идеальная форма для трансформер-Encoder** (в отличие от тяжёлых карт depth/CLIP).
- **analytics (N,):** `dt_seconds`, `flow_div_abs_mean`, `flow_consistency`, `bg_ratio`.
- **debug / backend-preview:** `preview_pair_pos`, `preview_prev/cur_frame_indices`, `preview_prev/cur_times_s`,
  `preview_flow_mag_map_norm (K,64,64)` — K=10 теплокарт магнитуды, нормированных [0,1]; `meta`.

**NaN-политика (by design):** idx 0 не имеет предыдущего кадра → `motion[0]=0`, `dt[0]=NaN`, все остальные
flow/cam ряды на idx 0 = NaN. Размерности стабильны: K=10, preview-карты 64×64.

## 6. Фичи (важное/неочевидное)

- **`motion_norm_per_sec_mean`** — несущая фича. **Двойная нормализация** (÷dt, ÷max(H,W)) делает её
  сравнимой между видео с разным fps и разрешением. По 26 реальным видео per-video mean 0.046, диапазон
  0.013…0.307, **CV=1.36** — сильнейшая различимость из всех разобранных core-фич (динамика vs статика).
  Within-video p95/median median 1.59, max 56 — у экшн-видео огромный разброс (пики движения = склейки/резкие
  сцены).
- **`flow_consistency = 1/(1+flow_div_abs_mean)`** ∈(0,1] — «плавность/когерентность» поля: 1 = гладкое
  трансляционное движение (панорама), ниже = турбулентный/расходящийся поток. Реально per-video 0.66…0.88.
  Идентичность consistency↔div держится на **3e-8** (проверено на 26 видео).
- **`flow_dir_dispersion` + sin/cos** — направление движения и его когерентность (0 = все векторы в одну
  сторону, →1 = хаос). Взвешено по магнитуде, считается только на «движущихся» пикселях (>перцентиль50).
- **`cam_*` (affine RANSAC)** — разделяют движение **камеры** (глобальный фон) и движение **объектов**:
  scale=зум, rotation=наклон, tx/ty=панорама, shake=дрожание. Детерминизм обеспечен `cv2.setRNGSeed(0)`.
- **`bg_ratio` ≈ 0.40 by design, НЕ баг.** Определён как `mean(mag ≤ percentile40(mag))` → по построению
  ≈0.40 почти константно. На 26 реальных видео — **ровно 0.400 на всех** (CV≈0). Валиден (∈[0,1]), но как
  фича неинформативен — кандидат на переосмысление порога или исключение (§9/§10).

## 7. Алгоритм / архитектура

- **Модель:** **RAFT** (Recurrent All-Pairs Field Transforms) — SOTA-плотный оптический поток. Прод —
  ONNX через Triton `raft_{256,384,512}` (2-input ensemble, CPU-preprocess → GPU-inference). Валидация —
  inprocess torchvision `raft_small`/`raft_large`. Внешняя предобученная сеть, не обучается.
- **Препроцессинг:** клиент делает только resize до S×S и шлёт **UINT8 NHWC RGB** двумя входами (prev, cur);
  нормализация в Triton ensemble.
- **Пост-обработка (CPU, numpy/cv2):** магнитуда `hypot(dx,dy)`, векторизованные mean/std/p95 по батчу;
  направление/дивергенция/affine — поштучно на downsampled 64×64 сетке (RANSAC-affine на bg-точках).
- **Где идёт:** Triton GPU (прод, no-network), либо inprocess raft (обход).
- **Сложность:** линейна по числу пар N−1. Чистый инференс raft_256 **~0.58 мс/пара** (RTX 2000 Ada),
  ролик N≤300 — 83–160 мс; raft_512 до 307 мс. Пресет/модель **почти не влияют на motion-агрегаты**
  (корреляция кривой 0.995 на статике; агрегаты meanΔ≤0.03) → raft_256_small оптимален.

## 8. Оптимизации

- **Двойная нормализация motion (÷dt, ÷max(H,W))** — делает кривую инвариантной к fps/разрешению; осознанное
  ключевое решение, ради сравнимости между видео.
- **Preprocess в Triton** + лёгкий uint8-транспорт от клиента (baseline-паттерн core-провайдеров).
- **Векторизованная batch-статистика магнитуды** (`np.hypot`, mean/std/quantile по оси) — тяжёлое считается
  на весь батч разом; дорогой affine/direction — на downsampled 64×64, не на полном разрешении (осознанный
  компромисс скорость↔точность).
- **Reuse RAFT для cut_detection** — cut_detection в прод-baseline требует именно этот артефакт вместо
  собственного farneback (архитектурная экономия: один тяжёлый поток на два потребителя).
- **Preview K=10 карт 64×64** вместо всех полей потока наружу — компактный backend-payload.
- **Фикс RANSAC-seed** (`cv2.setRNGSeed(0)`) — сделал недетерминированные cam_* воспроизводимыми (golden diff=0).
- **Атомарная запись NPZ** + пост-валидация схемы с удалением артефакта при провале.

## 9. Слабые места

- **`bg_ratio` неинформативен (≡0.40 by design)** — на 26 реальных видео константа. Занимает место в NPZ,
  фича-мусор для Encoder. Не блокер, но техдолг.
- **ДЕФЕКТ batch-пути:** `VisualProcessor/utils/core_optical_flow_batch.py` пишет только motion_norm/dt/preview,
  **не заполняет audit-v3 per-frame фичи** → batch-NPZ провалит структурный валидатор. Валидация шла через
  per-video main.py; **обязательно синхронизировать batch перед прод-масштабом 200k** (иначе массовый прогон
  даст невалидные артефакты). Это самый серьёзный пункт.
- **Триtop-сервер с RAFT ensemble не был поднят при валидации** (launch-скрипта нет) — валидация inprocess;
  но 26 реальных storage-артефактов **уже с Triton raft_256** (значит прод-путь работает).
- **RAFT дороже depth/farneback** — self-fps низкий на длинных видео; для 200k критично держать один прогретый
  воркер и батчить пары (init raft-весов ~9–11 c/подпроцесс доминирует в per-video режиме).
- **cam_* affine — грубый прокси** (downsampled 64×64, RANSAC на bg-точках) — различает камеру vs объекты
  приблизительно, при сложных сценах может путать. Достаточно для агрегатов, не для точной стабилизации.
- **Короткие видео / малый N** — медиана N на реальных данных = **12**: motion-кривая из ~11 пар статистически
  бедна; within-video p95/median шумит.
- Отдельного `LOGIC_ERRORS_FOR_CLAUDE.md` L-номера нет (кроме описанного batch-дефекта в REPORT).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Починить batch-путь** `core_optical_flow_batch.py` — заполнять все audit-v3 per-frame поля, иначе
   прод-масштаб даст невалидные NPZ. Блокер для 200k.
2. **[выс.] Переосмыслить/убрать `bg_ratio`** — либо сделать порог адаптивным (например доля пикселей ниже
   абсолютного порога движения, а не перцентиля 40 → тогда значение реально варьируется), либо исключить из
   контракта как мёртвую фичу.
3. **[сред.] Закрепить raft_256_small как единственный прод-дефолт** — доказано, что пресет не влияет на
   агрегаты, а 512/large только дороже; не тратить бюджет на большие пресеты.
4. **[сред.] Общий прогретый RAFT-воркер + батчинг пар** для 200k — init весов доминирует, амортизировать
   один раз на воркер; переиспользовать один прогон для cut_detection.
5. **[низ.] Разделить motion объектов и камеры явной фичей** — сейчас `motion_norm` смешивает движение
   объектов и панораму; `motion_minus_camera` (остаток после вычитания affine) мог бы дать чище «динамику сцены».

## 11. Рекомендации по архитектуре / связям

- **Закрепить reuse RAFT для cut_detection** на уровне оркестрации (уже baseline-требование) — не считать
  оптический поток дважды; убедиться, что frame_indices выровнены.
- **video_pacing hard-depends (no-fallback)** на motion-кривой — это правильно; задокументировать, что
  core_optical_flow обязан отработать до video_pacing/cut_detection (порядок Tier).
- **Единый preview-формат** теплокарт потока с preview depth/thumbnails core_clip → общий backend-рендер
  «что двигалось в кадре».
- **Shared sampling group** (core_clip/depth/detections/landmarks/optical_flow) — гарантировать на Segmenter,
  а не ловить mismatch падением downstream (общий пункт со всеми core-провайдерами).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что реально говорит |
|---|---|---|---|
| Хард-гейты H1–H6 (validate `--struct --ranges`) | 13 видео + 12 пресетов | **rc=0 на всех** | контракт/оси/finite/разные длины ок |
| C1 различимость motion | корпус | CV=**0.752**, within p95/median 12.8–16.6 | motion сильно различает видео |
| C2 разделяющая способность | динамика vs статика | медиана **4.9×** (0.21 vs 0.043) | чётко отделяет экшн от talking-head |
| C3 диапазоны/согласованность | 13 видео, 100% finite | все ряды в контракте; consistency err **3e-8** | численно здоров |
| C4 golden (после RANSAC-seed) | 3 прогона | **diff=0** для всех массивов, вкл. cam_* | воспроизводим |
| Сравнение пресетов | 3 видео × 4 конфига | агрегаты стабильны, meanΔ≤0.03 | raft_256_small оптимален |
| Audit v4 L2 (A+B) | 5 run, N_total=543 | ✓ ~8.8/10 | схема стабильна, NaN только idx0, preview [0,1] |
| **Реальные артефакты storage (мой прогон)** | **26 видео, 591 кадр, Triton raft_256** | **0 NaN (idx0 by design), consistency err 3e-8** | прод-путь здоров |
| — motion per-video | 26 видео | mean 0.046, CV **1.36** | ещё сильнее различает на реальном корпусе |
| — bg_ratio per-video | 26 видео | **ровно 0.400 на всех** | подтверждает «by design, неинформативна» |
| — N на видео | 26 видео | min 12 / median 12 / max 119 | короткие ролики доминируют |

Вывод: надёжность **высокая** по качеству сигнала (motion — самый различимый core-выход, детерминизм строгий),
но **прод-готовность неполная** из-за batch-дефекта, который надо закрыть до масштаба.

## 13. Интерпретируемость

**Есть:** dev-рендер (`utils/render.py`) + preview-теплокарты магнитуды в NPZ (K=10, 64×64, [0,1]); README
и FEATURE_DESCRIPTION подробно объясняют motion-кривую и camera-прокси.

**Добавить (для обычного пользователя):**
- **Теплокарты движения поверх кадров** (preview_flow_mag_map_norm уже в NPZ) — «здесь двигалось» магмой.
- **Одна фраза словами:** «динамичное видео / спокойный talking-head» из `motion` + `p95/median`.
- **«Движение камеры vs объектов»** словами из cam_* (панорама/зум/дрожание/статичная камера) — понятный
  профессиональный инсайт для креатора.
- **Timeline темпа** (motion по времени) с отметками пиков = резкие сцены/склейки.
- Приложенная визуализация (`core_optical_flow_distributions.png`) — пример подачи распределений.

## 14. Польза для моделей

Почти все per-frame ряды — **model-facing** во [`FEATURE_ENCODER_CONTRACT`](../../../../Models/docs/source_migrations/FEATURE_ENCODER_CONTRACT.md)
(`motion_norm_per_sec_mean`, `dt_seconds` явно перечислены). В отличие от depth/CLIP, это **готовый лёгкий
dense time-series** (N скаляров/кадр, не карты) — идеальная форма для трансформер-Encoder без пулинга.
Динамика/темп/движение камеры — правдоподобно **сильные** предикторы удержания и вовлечённости (быстрый темп
↔ retention), поэтому motion-ветка потенциально одна из самых информативных немультимодальных фичей. Прямых
feature-importance данных нет (модель в разработке), но гипотеза сильная. `bg_ratio` — мёртвый вход, стоит убрать.

## 15. Польза для аналитиков

- **`motion_norm_per_sec_mean`** → «насколько динамичное видео» (самый различимый показатель, CV=1.36) —
  сравнимо между роликами благодаря нормализации.
- **`cam_*`** → профиль движения камеры: панорама/зум/дрожание/штатив — понятный съёмочный инсайт.
- **`flow_consistency`/`flow_dir_dispersion`** → «плавное движение vs хаотичное» — прокси качества съёмки/монтажа.
- **Теплокарты движения** → визуально «что двигалось в кадре».
- Оговорка: `bg_ratio` показывать не стоит (константа); motion на коротких видео (N≈12) статистически шумит.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Единственный источник динамики/темпа; кормит video_pacing+cut_detection+Encoder |
| 5. Выход (контракт) | 5 | Богатый, лёгкий seq-выход, чистая tier-классификация, стабильные K/preview |
| 6. Фичи | 4 | motion/cam/consistency мощны и нормированы; bg_ratio мёртвый |
| 8. Оптимизации | 4 | Норм., reuse для cut_detection, векторизация, seed-fix; batch-путь сломан |
| 9. Слабые места (инверсно) | 3 | batch-дефект (блокер масштаба), bg_ratio-мусор, RAFT дорог, малый N |
| 12. Результаты тестов | 4 | 100% гейты+golden diff=0, 26 реальных Triton-видео чисты; batch не валидирован |
| 13. Интерпретируемость | 3 | Теплокарты+cam_* есть в данных, но overlay/словесная подача в TODO |
| 14. Польза для моделей | 5 | Лёгкий dense-seq, динамика — сильный предиктор retention |
| 15. Польза для аналитиков | 4 | Динамика+профиль камеры наглядны; часть фич требует пояснений/чистки |

### Итоговые оценки

- **Польза для моделей: 5/5.** Даёт Encoder'у уникальную ось — динамику/темп/движение камеры — в самой
  удобной для трансформера форме (лёгкий dense time-series, не карты). Динамика правдоподобно сильно
  коррелирует с удержанием. Снижает только мёртвый `bg_ratio` (легко убрать).
- **Польза для аналитиков: 4/5.** «Насколько динамичное видео» + профиль движения камеры — наглядный,
  интерпретируемый и сравнимый выход; ограничивают только мёртвый bg_ratio, шум на коротких видео и
  отсутствие пока готовой словесной/overlay-подачи.

## 17. Источники

- `DataProcessor/VisualProcessor/core/model_process/core_optical_flow/main.py`
- `.../core_optical_flow/README.md`, `.../docs/SCHEMA.md`, `.../docs/FEATURE_DESCRIPTION.md`
- `.../core_optical_flow/utils/{validate_core_optical_flow_npz.py,render.py}`, `.../scripts/audit_v4_npz_stats.py`
- `DataProcessor/docs/component_reports/core_optical_flow/{REPORT_2026-07-12.md,CRITERIA.md}`
- `DataProcessor/docs/audit_v4/components/visual_processor/core/core_optical_flow_audit_v4.md`
- `Models/docs/source_migrations/FEATURE_ENCODER_CONTRACT.md` (motion_norm_per_sec_mean/dt_seconds как core-вход)
- Downstream: `modules/video_pacing/utils/video_pacing.py` (hard-dep motion, no-fallback),
  `modules/cut_detection/main.py` (reuse flow.npz вместо farneback), `modules/optical_flow/*`
- `automation/runner/AGENT_CONTEXT.md` (разделы 6/7: тайминги 0.58 мс/пара, init 9–11 c, golden diff=0, bg_ratio≈0.40, batch-дефект)
- Реальные артефакты: 26× `storage/result_store/youtube/*/*/core_optical_flow/flow.npz` (591 кадр, Triton raft_256)

## 18. Визуализации

![Распределения core_optical_flow](core_optical_flow_distributions.png)

`core_optical_flow_distributions.png` (построено на 26 реальных Triton-артефактах, 591 кадр): распределения
`motion_norm_per_sec_mean` (CV=1.36 — самая различимая core-фича), within-video `p95/median` (пики движения),
`flow_consistency` и числа кадров N (медиана 12). Подтверждает: motion чётко отделяет статику от динамики;
0 NaN (кроме idx0 by design); `bg_ratio` ровно 0.40 на всех видео (мёртвая фича).
