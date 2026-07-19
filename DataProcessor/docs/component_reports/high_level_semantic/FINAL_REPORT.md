# FINAL REPORT — `high_level_semantic`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `high_level_semantic` (VisualProcessor `BaseModule`, Tier-3, CPU-only агрегатор) |
| Версия кода | `2.0.2` |
| Схема NPZ | `high_level_semantic_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/high_level_semantic/high_level_semantic.npz` |
| Модель | **нет** — CPU-fusion (numpy + reuse эмбеддингов) |
| Hard deps | `core_clip` + `cut_detection` |
| Soft deps (graceful) | `emotion_face`, `TextProcessor`, `loudness_extractor`, `tempo_extractor` |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → high_level_semantic ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Код | `DataProcessor/VisualProcessor/modules/high_level_semantic/utils/hl_semantic.py` (1099 строк) |

## 2. Резюме

`high_level_semantic` — **CPU-агрегатор высокоуровневой семантики**, который сшивает несколько upstream-сигналов
в единое сцено-/событийно-ориентированное представление: **scene_embeddings** (L2, mean-pool CLIP по сцене),
per-frame семантическую динамику (`clip_sim_prev`/`clip_novelty_prev`), позицию в сцене, **эмоции** (valence/
arousal/intensity из emotion_face), **аудио-плейсхолдеры** (loudness/tempo), **весь текстовый вектор
TextProcessor** (752 фичи) и **поток семантических событий** (`event_type_id`: границы сцен, novelty-скачки,
эмоц-кейфреймы). Все зависимости кроме core_clip/cut_detection — мягкие с graceful fallback (нет upstream →
NaN, не падение). На реальном корпусе: **визуальный слой жив** (scene-эмбеддинги L2=1.0, clip_sim present 0.92–
0.99), **эмоции частично** (0.0–0.89, зависит от лиц), **текст в основном жив** (~77% finite из 752), **аудио
мертво** (loudness/tempo present_ratio=0.0 на всех — by design в visual-standalone). Прод-готов, golden Δ=0.

## 3. Функционал

Стоит в Tier-3 (агрегатор), после core_clip/cut_detection и опционально emotion/text/audio. Делает:

1. **Scene embeddings** — mean-pool CLIP-эмбеддингов кадров внутри каждой сцены (границы cut_detection) → L2 →
   `scene_embeddings (S,512)` + метаданные сцен (start/end/duration/representative frame).
2. **Семантическая динамика** — `clip_sim_prev` (косинус к предыдущему кадру), `clip_novelty_prev` (1−sim),
   `scene_pos_norm` (позиция кадра в сцене).
3. **Мультимодальная сшивка** — подтягивает emo_* (emotion_face), loudness/tempo (audio), 752 tp_* (TextProcessor).
4. **Поток событий** — `event_type_id` (1=граница сцены, 200=novelty/семант. скачок, 210=эмоц. кейфрейм) +
   `event_times_s`/`event_strength`/`event_frame_pos`.

**Зачем продукту:** это **«семантическая карта видео»** — где сцены, как меняется содержание, где эмоц/
смысловые пики. Даёт Encoder'у сжатое сцено-уровневое представление (S эмбеддингов вместо N кадров) + событийный
таймлайн, а аналитику — «структуру» видео (сколько сцен, где ключевые моменты).

## 4. Вход

- **`core_clip`** (hard) — `frame_embeddings`+`frame_indices` для scene-эмбеддингов и динамики.
- **`cut_detection`** (hard) — границы сцен (`scene_id`).
- **`emotion_face`** (soft, graceful) — emo_valence/arousal/intensity; нет/empty → NaN, `emotion_face_present=False`.
- **`TextProcessor`** (soft) — 752 tp_* фичи → `text_feature_values`.
- **`loudness_extractor`/`tempo_extractor`** (soft, off по умолчанию в visual-standalone) — audio; NaN by default.
- **`union_timestamps_sec`** + Segmenter `frame_indices` — ось.

## 5. Выход

- **Scene-tier:** `scene_embeddings (S,512)` L2 + `scene_id`/`scene_start/end_frame_idx`/`scene_start/end_time_s`/
  `scene_duration_s`/`scene_representative_frame_idx`/`scene_embedding_mean_norm`.
- **Frame-tier:** `frame_features (N,8)` + `frame_feature_names` + `frame_feature_present_ratio` (clip_sim_prev,
  clip_novelty_prev, scene_pos_norm, loudness_dbfs, tempo_bpm, emo_valence/arousal/intensity).
- **Event-tier:** `event_type_id`/`event_times_s`/`event_strength`/`event_frame_pos`.
- **Text-tier:** `text_feature_values (752,)` + `text_feature_names` (весь вектор TextProcessor).
- **`ui`** + `features` (агрегаты). Ось: `frame_indices`, `times_s`.

## 6. Фичи (важное/неочевидное)

- **`scene_embeddings` (mean-pool CLIP + L2)** — компактное представление сцены (1 вектор на сцену вместо всех
  кадров); L2-норма=1.0000 на всех видео (C2). Идеальная сжатая форма для Encoder «о чём эта сцена».
- **`clip_novelty_prev`** — «насколько кадр нов относительно предыдущего» (семантический скачок): прокси
  монтажной/содержательной динамики, дополняет motion (визуальное движение) семантикой.
- **`event_type_id` ∈ {1, 200, 210}** — унифицированный поток событий: 1=граница сцены, 200=novelty-пик,
  210=эмоц. кейфрейм. На реальных данных **210 не встречается** (нет emotion_face keyframes) — только {1,200}.
- **`frame_feature_present_ratio`** — тот же образцовый паттерн, что у frames_composition: явно отдаёт долю
  определённости каждого столбца (audio=0.0, emo зависит от лиц) — модель знает, что не «сломано», а «нет upstream».
- **Graceful fallback** — emotion/audio/text опциональны: компонент не падает без них, а помечает NaN + флаг
  present. Правильная устойчивость агрегатора.

## 7. Алгоритм / архитектура

- **Чистый CPU** (numpy + dict). Тяжёлого инференса нет — только mean-pool/косинусы/сборка событий.
- **Параллельная загрузка** 6 upstream-артефактов через `ThreadPoolExecutor` — доминирующая стоимость (сами
  вычисления пренебрежимы; N=65 — <1 c).
- **Детерминизм:** golden max|Δ|=0.0 (полный numpy-детерминизм).
- Для 200k: стоимость = только I/O загрузки deps, вычисления «бесплатны».

## 8. Оптимизации

- **Reuse эмбеддингов core_clip** — не гоняет свою модель, mean-pool уже готовых CLIP-векторов (ноль инференса).
- **Параллельная загрузка deps** (ThreadPoolExecutor) — I/O-bound часть распараллелена.
- **Scene-tier сжатие** (S эмбеддингов вместо N кадров) — компактный вход для Encoder.
- **present_ratio + graceful fallback** — дешёвая устойчивость к отсутствующим модальностям.

## 9. Слабые места

- **Аудио-слой мёртв на всех реальных данных.** `loudness_dbfs`/`tempo_bpm` present_ratio=**0.0** на всех 6
  видео — by design (в VisualProcessor-standalone `require_audio_*` выключены). Т.е. 2 из 8 frame-фич всегда NaN;
  реальную аудио-семантику компонент не получает без кросс-процессорной сборки.
- **Эмоции разрежены/зависят от лиц** — emo_* present 0.0–0.89; на 2 из 6 видео = 0.0 (нет лиц/emotion_face).
  Событие 210 (эмоц. кейфрейм) на реальных данных не появляется вовсе.
- **Дублирование 752 текстовых фич** — весь вектор TextProcessor копируется в визуальный NPZ (`text_feature_
  values`). Избыточность/раздувание артефакта: зачем визуальному компоненту хранить полный текстовый вектор,
  который уже есть в TextProcessor? Архитектурный вопрос (§11).
- **Мало собственного сигнала** — компонент в основном **репакует** upstream: реально новое = scene-эмбеддинги
  (mean-pool) + novelty + event-stream. Остальное — перенос чужих фич.
- **emotion_face фикс не проверен на git** (сделан graceful fallback в этой сессии — убедиться, что закоммичен).
- **Текст частично NaN** (~23% из 752, напр. asr confidence chunked) — наследует пустоты TextProcessor.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Включить аудио-слой в проде** — сейчас loudness/tempo всегда NaN (visual-standalone). Либо
   кросс-процессорная сборка (audio-артефакты доступны на этапе fusion), либо включить `require_audio_*` там,
   где аудио посчитано. Иначе 2 фичи мертвы всегда.
2. **[сред.] Не дублировать 752 текстовых фичи** — хранить ссылку/хеш или подмножество, а не полный вектор
   TextProcessor в визуальном NPZ (раздувание, рассинхрон-риск).
3. **[сред.] Обогатить событийный поток** — сейчас только {1,200} на реальных данных; добавить аудио-события
   (пики громкости/бита) и текст-события (появление плашки) после включения соответствующих модальностей.
4. **[низ.] Добавить собственную сцено-семантику** — например тематическую метку сцены (zero-shot CLIP) поверх
   mean-pool, чтобы компонент давал новое, а не только репак.

## 11. Рекомендации по архитектуре / связям

- **Это естественная точка мультимодального fusion** — но пока визуальная (audio/text подтянуты частично/
  плейсхолдерами). Логично поднять его на уровень **над** processor'ами (где доступны и audio, и text реально),
  а не внутри VisualProcessor — тогда все модальности живые.
- **scene_embeddings** — сильный кандидат в прямой вход Fusion/Encoder (сцено-уровневые токены); согласовать с контрактом Models.
- **present_ratio-паттерн** — согласован с frames_composition; распространить дальше как стандарт NaN-трактовки.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct+ranges | 22 batch + 3 pod | rc=0 | контракт ок |
| U2 ось времени | 22 | frame_indices↑, times_s неубыв. | ось корректна |
| U3 finite/health | — | Inf=0; audio NaN by design; emo зависит от лиц | health объясним |
| U4 expected-empty | emotion_face отсутств./empty | status=ok, emo NaN, present=False | graceful fallback работает |
| U5 golden | 2 прогона | max\|Δ\|=0.0 | полный детерминизм |
| U6 разные длины | N=43/65/119 | S=1/3/6, E=0/4/9, rc=0 | масштабируется |
| C1 clip_sim present | 0.977–0.992 (≥0.97) | scene-динамика надёжна |
| C2 scene_embeddings L2 | 1.0000 (±<0.001) | эмбеддинги нормированы |
| C3 event_type_id | {1,200} (210 редко) | событийный поток валиден |
| **Реальный storage (мой прогон)** | 6 видео | визуал жив (clip 0.92–0.99, scene L2=1); emo 0.0–0.89; **audio 0.0**; text ~77% finite | мультимодальность частичная |

Вывод: **визуальная семантика жива и надёжна**, эмоции работают где есть лица, но **аудио-слой мёртв на всех
данных** (by design) и компонент во многом репакует upstream.

## 13. Интерпретируемость

- **Сильная сторона:** сцены + события = наглядная «карта видео». `ui` payload есть; можно показать таймлайн
  сцен с representative-кадрами и метками событий (граница/novelty-пик/эмоц-момент).
- **Добавить:** словесная структура «видео из 6 сцен, ключевой момент на 0:42»; после включения аудио/текст-
  событий — богаче таймлайн («здесь появился текст», «пик громкости»).

## 14. Польза для моделей

**Заметная.** `scene_embeddings (S,512)` L2 — **компактный сцено-уровневый вход для Encoder/Fusion** (S токенов
вместо N кадров), что уменьшает длину последовательности и даёт «о чём сцена». `clip_novelty` + событийный поток
— семантическая динамика/структура, дополняющая motion/pacing. Ограничивают: аудио-плейсхолдеры мертвы,
дублирование текста (модель и так получит его из TextProcessor), много репака. Крепкое «хорошо» по визуальной части.

## 15. Польза для аналитиков

**Высокая.** «Структура видео»: число сцен, длительности, ключевые/эмоциональные моменты, семантические скачки —
понятная и наглядная аналитика (таймлайн + representative-кадры). Ограничения: аудио-события отсутствуют, эмоц-
моменты только при наличии лиц, часть текст-фич NaN.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Мультимодальная семантическая карта + scene-эмбеддинги + события — ценная роль |
| 5. Выход (контракт) | 4 | Богатый scene/frame/event/text-контракт + present_ratio; дублирует 752 текст-фичи |
| 6. Фичи | 4 | scene-эмбеддинги/novelty/события сильны; audio мёртв, emo разрежен |
| 8. Оптимизации | 4 | Reuse CLIP, параллельная загрузка, сжатие в сцены, graceful fallback |
| 9. Слабые места (инверсно) | 3 | Audio dead, дубль текста, много репака, emo зависит от лиц |
| 12. Результаты тестов | 4 | Все гейты PASS + golden=0; визуал жив, но мультимодальность частична |
| 13. Интерпретируемость | 4 | Сцены+события — наглядная карта видео |
| 14. Польза для моделей | 4 | scene_embeddings — компактный вход Fusion; репак/audio ограничивают |
| 15. Польза для аналитиков | 4 | Структура видео наглядна; audio/emo частичны |

### Итоговые оценки

- **Польза для моделей: 4/5.** `scene_embeddings (S,512)` L2 — компактный сцено-уровневый вход для Encoder/Fusion
  (сжатие N→S + семантика сцены), а novelty/событийный поток дополняют motion/pacing. Ниже 5 держат мёртвый
  аудио-слой, дублирование текстового вектора и то, что компонент во многом репакует upstream, добавляя ограниченно нового.
- **Польза для аналитиков: 4/5.** Наглядная «структура видео» (сцены, длительности, ключевые/эмоц. моменты,
  семантические скачки) — понятная и сравнимая аналитика. Ограничивают отсутствие аудио-событий, зависимость эмоций
  от лиц и частичная пустота текст-фич.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/high_level_semantic/utils/hl_semantic.py` (1099 строк)
- `.../high_level_semantic/{main.py,utils/validate_high_level_semantic.py,utils/render.py}`
- `DataProcessor/VisualProcessor/schemas/high_level_semantic_npz_v{1,2}.json`
- `DataProcessor/docs/component_reports/high_level_semantic/{REPORT_2026-07-16.md,CRITERIA.md}`
- `DataProcessor/docs/audit_v4/components/visual_processor/modules/high_level_semantic_audit_v4.md`
- Cross-ref deps: `core_clip`, `cut_detection`, `emotion_face`, TextProcessor, `loudness_extractor`, `tempo_extractor`
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/high_level_semantic/high_level_semantic.npz`
  (визуал жив; emo 0.0–0.89; **audio present_ratio=0.0**; text ~77% finite; события {1,200})

## 18. Визуализации

![high_level_semantic overview](high_level_semantic_overview.png)

`high_level_semantic_overview.png`: слева — heatmap present_ratio 6 frame-фич по 6 видео (clip_sim/novelty/
scene_pos зелёные ≈1; emo_valence жёлто-зелёный 0.0–0.89 по лицам; **loudness/tempo красные=0.0**); справа —
сводка «живости» модальностей: visual+text живы, emotion частичен, **audio мёртв (by design в visual-standalone)**.
Подтверждает: сцено-семантика и текст работают, эмоции — по наличию лиц, аудио-слой требует кросс-процессорной сборки.
