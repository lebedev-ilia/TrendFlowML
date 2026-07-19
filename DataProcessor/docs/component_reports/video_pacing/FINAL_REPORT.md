# FINAL REPORT — `video_pacing`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `video_pacing` (VisualProcessor `BaseModule`, Tier-3, CPU-only) |
| Версия кода | `2.0.1` |
| Схема NPZ | `video_pacing_npz_v3` |
| Артефакт | `result_store/<platform>/<video>/<run>/video_pacing/video_pacing_features.npz` |
| Модель | **нет** — CV/статистика (numpy/opencv); потребляет 3 провайдера |
| Hard deps | `cut_detection` + `core_clip` + `core_optical_flow` (no-fallback, union-domain) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → video_pacing ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`REPORT_2026-07-13.md`](REPORT_2026-07-13.md), [`CRITERIA.md`](CRITERIA.md) |
| Баг-реестр | `LOGIC_ERRORS_FOR_CLAUDE.md` L2 (min_frames=30 на коротких) |
| Код | `DataProcessor/VisualProcessor/modules/video_pacing/utils/video_pacing.py` (1366 строк) |

## 2. Резюме

`video_pacing` — **анализатор темпа монтажа**. Из shot-границ (`cut_detection`), кривой движения
(`core_optical_flow`) и CLIP-эмбеддингов (`core_clip`) он строит характеристики ритма видео: длительность
планов, частоту склеек (cuts/10s), скорость движения по планам, темп семантических/цветовых изменений, и
собирает **3 model-facing кривых длины N** (motion / semantic_change / color_change per sec) + `shot_boundary_
frame_indices (S)` + **57 табличных скаляров**. Чистый CPU, golden-детерминизм. Прод-готов: все U1–U6/C1–C4
PASS, на реальном корпусе (6 видео) скаляры **живые и различимые** (cuts/10s 0.93–3.45, motion 0.012–0.30).
Оговорки: (1) `min_frames=30` по умолчанию рушил короткие ролики (L2, чинён патчем до 8, «правильный» fix не
сделан); (2) реальный батч прогнан **вариантом A** (entropy/histograms OFF) → 13 NaN-фич, тогда как
заштампованный дефолт — **вариант B** (5 NaN) → 8 аналитических фич в storage мёртвы (config-drift).

## 3. Функционал

Стоит в Tier-3 — **после** cut_detection/core_clip/core_optical_flow (сам их шэрит, ничего тяжёлого не
считает). Логика:

1. **Планы (shots)** — берёт `shot_boundaries_frame_indices` из `cut_detection` (no-fallback), считает
   длительности планов, cuts/10s, распределение длин (entropy/gini/histogram), short-shot-фракцию, burst склеек.
2. **Motion-темп** — выравнивает motion-кривую `core_optical_flow` на свою ось, считает скорость движения по
   планам (mean/median/var/p90), долю high-motion кадров/планов, корреляцию motion↔shot.
3. **Semantic-темп** — косинусная дистанция между соседними CLIP-эмбеддингами `/dt` → `semantic_change_rate`;
   агрегаты (mean/std, high-change ratio, scene jumps).
4. **Color-темп** — средняя LAB-дельта соседних кадров `/dt` → `color_change_rate`; saturation/brightness change.
5. **Нарратив-профиль** — intro/main/climax speed, pacing_symmetry (по трети видео).

**Зачем продукту:** темп монтажа — **один из сильнейших драйверов удержания**. Быстрые склейки и высокий
темп ↔ динамичный контент (клипы/трейлеры), медленный ↔ спокойный (влог/лекция). Это и model-сигнал (ритм для
предсказания удержания), и понятная креатору аналитика («у вас 2 склейки в 10 с, средний план 4.3 с»).

## 4. Вход

- **`cut_detection`** (hard) — `detections.shot_boundaries_frame_indices` (список); пусто → error (no-fallback).
- **`core_optical_flow`** (hard) — `motion_norm_per_sec_mean` + `frame_indices`; не покрывает ось → error.
- **`core_clip`** (hard) — `frame_embeddings` + `frame_indices` для semantic-темпа; не покрывает → error.
- **`union_timestamps_sec`** + Segmenter `frame_indices` — строгая ось; немонотонность/непокрытие → error.
- **`min_frames`** (config, дефолт 30) — N < min_frames → RuntimeError (no-fallback) — см. §9 (L2).
- **Feature gating** (config): `enable_entropy_features`, `enable_histograms`, `enable_pace_curve_peaks`,
  `enable_periodicity`, `enable_bursts`.

## 5. Выход

- **Model-facing кривые (N,):** `motion_norm_per_sec_mean`, `semantic_change_rate_per_sec`,
  `color_change_rate_per_sec` (первый элемент = 0 by design — нет предыдущего кадра).
- **Shot-границы:** `shot_boundary_frame_indices (S,)` int32 (dedup+sorted из cut_detection).
- **Табличные:** `feature_names (57,)` + `feature_values (57,)` — `_FEATURE_NAMES_V1` (shot-длительности,
  cuts/10s, entropy/gini/histogram, motion-агрегаты, semantic/color-темп, narrative intro/main/climax).
- **Ось:** `frame_indices (N,)`, `times_s (N,)`.
- **NaN-политика:** 5 гейтнутых фич NaN by-design (pace_curve_peaks/period/power, semantic/color bursts) +
  structural (climax_speed/pacing_symmetry при shots<4). При варианте A ещё +8 (entropy/gini/histogram).

## 6. Фичи (важное/неочевидное)

- **`cuts_per_10s` + `shot_duration_mean`** — ядро темпа; на реальных данных 0.93–3.45 и 1.45–5.35 с
  (сильно различают динамику монтажа). Прямой прокси «клиповости».
- **`semantic_change_rate`** = косинус-дистанция соседних CLIP-эмбеддингов /dt — «скорость смены содержания»
  (не путать с motion: медленная камера может быстро менять сцены нарезкой). Уникальный семантический темп.
- **`color_change_rate`** = LAB-дельта /dt — «мелькание/смена цветовой палитры»; коррелирует со стробингом.
- **`mean_motion_speed_per_shot`** — движение усреднённое ПО ПЛАНАМ (а не по кадрам) → устойчивее к длине.
- **narrative intro/main/climax_speed + pacing_symmetry** — профиль «как темп меняется от начала к концу»
  (structural, требует ≥4 планов, иначе NaN — разумно).
- **5 гейтнутых фич (pace_curve peaks/periodicity, bursts)** — намеренно OFF: шумные/ненадёжные на коротких.

## 7. Алгоритм / архитектура

- **Чистый CPU**: opencv (LAB/SSIM/Canny/resize), numpy (агрегаты), потребляет готовые массивы deps.
- **Сложность:** ~12–18 c/видео (GPU-под, 4fps/480w); в полной цепочке доминирует `cut_detection` (farneble
  ~60–70 c), не сам video_pacing. Для 200k дёшев, узкое место — deps.
- **Детерминизм:** golden diff=0 при `OMP_NUM_THREADS=1` (numpy/opencv). Батч-путь = per-video subprocess
  main.py → расхождений с single-путём нет.

## 8. Оптимизации

- **Потребитель 3 провайдеров** — не считает поток/CLIP/склейки заново (reuse тяжёлого GPU).
- **Feature gating (варианты A/B)** — можно отключать шумные фичи (peaks/periodicity/bursts) — осознанный контроль.
- **motion по планам, а не кадрам** — устойчивее к длине видео.
- **downscale_factor=0.25** для color-темпа (LAB на уменьшенных кадрах).
- **shot-границы напрямую из cut_detection** — единый источник склеек (нет второго детектора).

## 9. Слабые места

- **`min_frames=30` no-fallback (L2, главный дизайн-долг).** Короткие ролики (N=12, ~10 с) падали с hard
  RuntimeError. Чинено E2E-патчем до `min_frames=8` (в storage N=12 прошёл ok), но **«правильный» fix**
  (`min_frames` по длительности/policy ИЛИ `status=empty` вместо краха) не сделан. Хрупко на грани.
- **Config-drift батча (важно для данных).** Реальный storage прогнан **вариантом A** (`enable_entropy_features
  =False, enable_histograms=False`) → **13 NaN-фич**, тогда как заштампованный дефолт — **вариант B**
  (entropy+histograms ON, 5 NaN). Итог: 8 аналитических фич (shot_duration_entropy, shot_length_gini,
  tempo_entropy, 5×shot_length_histogram) в реальных артефактах мёртвы, хотя валидировались живыми.
- **Жёсткий no-fallback на 3 зависимости** — если любая из cut_detection/core_clip/core_optical_flow упала,
  весь video_pacing = error. Много точек отказа для Tier-3.
- **color_change re-декодит кадры** — LAB считается повторным `frame_manager.get`+resize (не переиспользует
  кэш пайплайна) → лишний I/O/CPU.
- **Зависимость от качества cut_detection** — если склейки детектятся плохо (см. FINAL_REPORT cut_detection:
  deep-канал мёртв, пороги без калибровки), все shot-фичи наследуют эту неточность.
- **57 фич — умеренная раздутость** (cut_density_map_8bins, histogram); часть коррелирует.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Довести fix L2 до «правильного»** — `min_frames` от длительности/fps или `status=empty` +
   `empty_reason=too_few_frames` вместо hard error/патча-костыля 8.
2. **[выс.] Пере-прогнать storage вариантом B** (или согласовать, что дефолт = A) — сейчас 8 фич в реальных
   данных NaN, хотя заштампован вариант B; устранить config-drift.
3. **[сред.] Смягчить no-fallback** — при падении одной deps считать доступную часть (например без semantic,
   если core_clip нет) + пометить соответствующие фичи NaN, вместо полного error.
4. **[сред.] Переиспользовать кэш кадров** для color_change (не re-декодить LAB).
5. **[низ.] Проредить 57 фич** — убрать коррелирующие histogram/density-bins.

## 11. Рекомендации по архитектуре / связям

- **Reuse motion/CLIP/склеек закреплён** — правильно; убедиться в shared sampling group (общий frame_indices),
  иначе no-fallback error каскадит.
- **Пара cut_detection↔video_pacing** тесно связана (video_pacing целиком строится на shot-границах) —
  качество video_pacing ограничено сверху качеством cut_detection; улучшать их совместно.
- **motion-кривая шэрится** с core_optical_flow/optical_flow — единый источник динамики (не дублировать).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct+ranges+qa | 3 pod + 22 batch + empty | rc=0 на всех | контракт/диапазоны ок |
| U2 ось времени | 3 | frame_indices↑, times_s монотонна | ось корректна |
| U3 finite/curves | 3 | 3 кривые len==N; NaN by-design | health ок |
| U4 expected-empty | после фикса | status=empty, rc=0 | пустой путь чинён (2026-07-16) |
| U5/C4 golden | 33s+150s | identical=True (diff=[]) | детерминизм (при OMP=1) |
| U6 разные длины | N=34/112/120 + batch 12–119 | все ok | масштабируется |
| C1 различимость | 3 pod + 6 batch | 4/4 CV>0.30 | темп сильно различает видео |
| C2 согласованность deps | 3 | shots_count=S; кривые len N; доли∈[0,1] | связь с cut_detection корректна |
| C3 NaN-политика | 3 | 5 by-design + structural | NaN объяснимы |
| **Реальный storage (мой прогон)** | **6 видео, все ok** | скаляры живые/различимы (cuts 0.93–3.45, motion 0.012–0.30); **13 NaN (вариант A)** | model-вход жив; 8 аналитич. фич мёртвы (config-drift) |

Вывод: **редкий случай — на реальных данных компонент по-настоящему жив и различим** (в отличие от соседних,
где core-фичи мертвы). Минус — батч прогнан вариантом A, 8 фич NaN сверх заштампованного.

## 13. Интерпретируемость

- **Сильная сторона:** темп понятен интуитивно. 3 кривые + shot-границы отлично визуализируются (timeline
  монтажа); `render.py` есть.
- **Добавить:** словесная сводка «динамичный монтаж: 2 склейки/10с, планы по 4 с» / «спокойный, длинные планы»;
  overlay кривых темпа на прогресс-бар видео с отметками склеек; сравнение темпа с медианой ниши.

## 14. Польза для моделей

**Высокая.** Темп монтажа — правдоподобно **сильный предиктор удержания** (быстрый монтаж ↔ retention), а
video_pacing даёт его в двух формах: 3 dense-кривые (motion/semantic/color) для Encoder + компактные скаляры
(cuts/10s, shot_duration, motion_speed) для baseline. На реальном корпусе фичи **живые и различимые** (CV
0.3–0.8) — то, чего не хватало нескольким соседним компонентам. Ограничивают: config-drift (8 фич NaN в батче),
зависимость от точности cut_detection, хрупкий min_frames. Крепкое «хорошо».

## 15. Польза для аналитиков

**Высокая.** «Насколько динамичный у вас монтаж» — понятнейшая креатору метрика: частота склеек, средняя длина
плана, скорость смены сцен/цвета, профиль темпа (intro/main/climax). Всё сравнимо между роликами и наглядно
(timeline). Ограничения: 8 аналитических фич (entropy/gini/histogram) в реальных данных NaN (вариант A), 57
полей слегка раздуты, качество shot-фич наследует неточность cut_detection.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Темп монтажа — сильный самостоятельный сигнал; 3 кривые + shot + narrative-профиль |
| 5. Выход (контракт) | 4 | 3 кривые + shot-границы + 57 скаляров; часть гейтнута/раздута |
| 6. Фичи | 4 | cuts/shot/semantic/color-темп информативны и различимы; narrative structural |
| 8. Оптимизации | 4 | Reuse 3 deps, gating, motion-по-планам; color re-декодит кадры |
| 9. Слабые места (инверсно) | 3 | min_frames хрупок, config-drift (8 NaN), no-fallback×3, зависит от cut_detection |
| 12. Результаты тестов | 4 | Все гейты PASS + на реальных данных живо и различимо; вариант A в батче |
| 13. Интерпретируемость | 4 | Темп интуитивен, 3 кривые+границы визуализируемы |
| 14. Польза для моделей | 4 | Темп — сильный предиктор удержания, в удобной форме; config-drift ограничивает |
| 15. Польза для аналитиков | 4 | «Динамичность монтажа» наглядна; часть фич NaN в батче |

### Итоговые оценки

- **Польза для моделей: 4/5.** Темп монтажа — один из самых правдоподобно-сильных предикторов удержания, и
  video_pacing отдаёт его и как dense-кривые (для Encoder), и как компактные скаляры (для baseline), причём на
  реальном корпусе фичи живые и различимые (CV 0.3–0.8) — редкость среди разобранных. Ниже 5 держат config-drift
  (8 фич NaN в батче), хрупкий min_frames и зависимость от точности cut_detection.
- **Польза для аналитиков: 4/5.** «Насколько динамичный монтаж» (склейки, длина планов, скорость смены сцен,
  профиль темпа) — наглядная, сравнимая, интуитивная аналитика. Ограничивают 8 мёртвых в реальном батче
  entropy/histogram-фич и лёгкая раздутость 57 полей.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/video_pacing/utils/video_pacing.py` (1366 строк)
- `DataProcessor/VisualProcessor/modules/video_pacing/main.py`, `utils/{validate_video_pacing.py,render.py}`
- `DataProcessor/VisualProcessor/modules/video_pacing/docs/SCHEMA.md`
- `DataProcessor/docs/component_reports/video_pacing/{REPORT_2026-07-16.md,REPORT_2026-07-13.md,CRITERIA.md}`
- `DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md` (L2 min_frames)
- Cross-ref deps: `cut_detection`, `core_clip`, `core_optical_flow` (FINAL_REPORTs)
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/video_pacing/video_pacing_features.npz`
  (все ok; **вариант A → 13 NaN; скаляры живые/различимые**)

## 18. Визуализации

![video_pacing overview](video_pacing_overview.png)

`video_pacing_overview.png`: слева — 4 ключевых pacing-скаляра на 6 реальных видео (cuts_per_10s, shot_duration_
mean, motion_speed, semantic_diff) реально различаются (CV 0.3–0.8) → на реальных данных компонент жив; справа —
3 model-facing кривые (motion/semantic/color change) + отметки shot-границ на одном видео. Подтверждает: темп
монтажа — живой, различимый, визуализируемый сигнал (при этом 8 entropy/histogram-фич в батче NaN — вариант A).
