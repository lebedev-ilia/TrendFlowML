# FINAL REPORT — `emotion_face`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `emotion_face` (VisualProcessor `BaseModule`, Tier-2, GPU) |
| Версия кода | `2.0.2` |
| Схема NPZ | `emotion_face_npz_v3` |
| Артефакт | `result_store/<platform>/<video>/<run>/emotion_face/emotion_face.npz` |
| Модель | **EmoNet** (`emonet_8.pth`, n_expression=8) — valence/arousal + 8 эмоций (Ekman+Contempt) |
| Hard dep | `core_face_landmarks` (кадры с лицами) |
| Потребитель | `high_level_semantic` (emo_* + keyframes → событие 210) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → emotion_face ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Баг-реестр | `LOGIC_ERRORS_FOR_CLAUDE.md` L8 (EmoNet source not found на поде) |
| Код | `DataProcessor/VisualProcessor/modules/emotion_face/core/video_processor.py` (2022 строки) + `main.py` |

## 2. Резюме

`emotion_face` — **распознаватель эмоций лица** на базе EmoNet: на кадрах с лицами (гейтинг по
`core_face_landmarks`) выдаёт **valence** (позитив/негатив ∈[-1,1]), **arousal** (возбуждённость ∈[-1,1]),
**intensity** = √(v²+a²) и распределение по **8 эмоциям** (Neutral/Happy/Sad/Surprise/Fear/Disgust/Anger/
Contempt), плюс ключевые эмоциональные кадры (keyframes). На реальном корпусе **ядро живо и правдоподобно**:
на 4 видео с лицами valence −0.48…+0.01, softmax валиден, доминирующие эмоции варьируются (Happy/Anger/Neutral/
Disgust); на 2 видео без лиц — корректный `status=empty`. **Но:** (1) `stride=4` → EmoNet прогоняется лишь по
**~25%** кадров-с-лицами (2–10 кадров/видео — статистически тонко); (2) `keyframes=0` **на всех** storage-видео
(фикс keyframes 2026-07-16 дал kf=1/8/13 на поде, но **батч не пере-прогнан** — именно поэтому событие 210 в
`high_level_semantic` не срабатывает). golden Δ=0 (fp32).

## 3. Функционал

Стоит в Tier-2, после `core_face_landmarks`. Логика:

1. Берёт кадры с лицами (`frames_with_face` из landmarks), применяет `face_frame_stride=4` (субсэмплинг) и
   `max_frames=200`.
2. Кропает лицо, прогоняет **EmoNet** → valence, arousal, 8-мерный softmax эмоций, intensity, dominant_emotion_id.
3. Считает `emotion_confidence`, `advanced_features`/`sequence_features` (динамика эмоций), выделяет keyframes
   (кадры пиковой эмоц. значимости).
4. `processed_mask ⊆ face_present`; на непроцессированных кадрах v/a = NaN by design.

**Зачем продукту:** эмоции лица — **сильный драйвер вовлечённости**: эмоциональный контент (радость, удивление,
драма) удерживает и провоцирует реакции (лайки/комменты). Это model-сигнал (эмоц. дуга ↔ engagement) и понятная
креатору аналитика («ваше видео преимущественно нейтральное / радостное / напряжённое»).

## 4. Вход

- **`core_face_landmarks`** (hard) — кадры с лицами; нет лиц → `status=empty, no_faces_in_video`.
- **Кадры** — `FrameManager.get(idx)` для кропов лиц.
- **EmoNet веса** (`emonet_8.pth` через ModelManager; исходники `emonet/*.py` должны быть на поде — L8).
- **`--face-frame-stride`** (дефолт 4), **`--max-frames`** (200), **`EMOTION_FACE_USE_AMP`** (0=fp32 для golden).
- **`union_timestamps_sec`** + Segmenter `frame_indices` — ось.

## 5. Выход

- **Per-frame:** `valence`, `arousal`, `intensity`, `emotion_probs (N,8)`, `dominant_emotion_id`,
  `emotion_confidence`, `processed_mask`, `face_present`, `face_count` (NaN/0 на непроцессированных).
- **Динамика:** `advanced_features`, `sequence_features`, `keyframes` (эмоц. пики — **=0 в батче**).
- **Агрегаты:** `summary`, `features`. Ось: `frame_indices`, `times_s`, `axis_source`.

## 6. Фичи (важное/неочевидное)

- **valence/arousal (circumplex-модель)** — непрерывное 2D эмоц. пространство (позитив↔негатив, спокойный↔
  возбуждённый), богаче дискретных меток; intensity=√(v²+a²) — «сила» эмоции. На реальных данных valence
  **скошен в минус** (среднее −0.24) — известная OOD-склонность EmoNet на «диких» кадрах, не обязательно
  реальный негатив контента.
- **8-эмоц. softmax** — probs_rowsum≈1.0 (валидно); dominant варьируется по видео (Happy/Anger/Neutral/Disgust)
  — различимо, но на 2–10 кадрах статистически шумно.
- **`keyframes` — эмоц. пики** — задуманы как «где эмоция всплеснула»; **=0 на всём реальном корпусе** (фикс не
  пере-прогнан) → downstream-событие 210 в high_level_semantic мёртво.
- **`stride=4` субсэмплинг** — EmoNet дорог, поэтому обрабатывается ~1/4 кадров-с-лицами (осознанная экономия,
  но на коротких/малолицых видео остаётся 2 кадра → почти нет сигнала).
- **`processed_mask ⊆ face_present`** — эмоции только там, где лицо; NaN на остальных (present_ratio-подход).

## 7. Алгоритм / архитектура

- **Модель:** EmoNet (n_expression=8), предобученная (не дообучается); GPU-инференс. fp16 (AMP) в проде,
  fp32 для golden (`EMOTION_FACE_USE_AMP=0`).
- **Сложность:** process_frames ~1.1–1.8 c/видео (GPU, stride=4); load_deps 18–47 c (OD+face_landmarks доминируют).
- **Детерминизм:** golden max|Δvalence|=max|Δprobs|=0.0 при fp32.
- **Over-engineering:** `core/` содержит `memory_manager`, `cache_with_ttl`, `retry_strategy`, `metrics_exporter`,
  `edge_cases`, `protocols` — тяжёлая обвязка (2022 строки в video_processor) для по сути per-frame инференса.

## 8. Оптимизации

- **stride=4 субсэмплинг** кадров-с-лицами — главная экономия (EmoNet дорог); осознанный компромисс сигнал↔скорость.
- **max_frames=200** — потолок на длинных видео.
- **AMP (fp16)** в проде, fp32 только для golden.
- **Гейтинг по face_present** — не гоняет EmoNet по кадрам без лиц.
- **ModelManager spec** для EmoNet (digest весов в meta).

## 9. Слабые места

- **`keyframes=0` на всём реальном батче (главное).** Фикс keyframes (2026-07-16, kf=1/8/13 на поде) **не
  пере-прогнан** → в storage kf=0 везде → downstream-событие 210 (эмоц. кейфрейм) в `high_level_semantic`
  никогда не срабатывает. Задекларированная фича мертва в данных.
- **Очень мало processed-кадров** — stride=4 + мало лиц → 2–10 EmoNet-инференсов на видео. Эмоц. агрегаты на
  2 кадрах (напр. -3Mbinqz: proc=2) статистически ненадёжны; per-video mean шумит.
- **EmoNet скошен в негатив на OOD** — valence склонен к минусу (−0.24 среднее) на «диких» YouTube-кадрах; без
  калибровки риск систематического смещения «всё негативное».
- **EmoNet source-код не на Network Volume (L8)** — инфра-хрупкость: без `emonet/*.py` на поде → error-каскад в
  high_level_semantic. Требует синка на все поды.
- **AMP batch-путь hardcoded** (`emotion_face_batch.py` use_amp=True) — рассинхрон с env перед прод-200k.
- **Over-engineered core/** (memory_manager/cache/retry/metrics) — 2000+ строк обвязки для прямого инференса;
  техдолг читаемости/поддержки.
- **Только лица** — контент без людей эмоц. сигнала не получает (empty); эмоция голоса/сцены не учитывается.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Пере-прогнать батч с фиксом keyframes** — сейчас kf=0 везде, событие 210 в hls мёртво; после
   пере-прогона эмоц. таймлайн оживёт.
2. **[выс.] Синхронизировать EmoNet source (`emonet/*.py`) на все поды** (L8) — иначе error-каскад.
3. **[сред.] Калибровать valence** (или задокументировать OOD-смещение EmoNet) — иначе «всё негативное» вводит в
   заблуждение аналитика/модель.
4. **[сред.] Пересмотреть stride на коротких/малолицых видео** — адаптивный stride (обрабатывать больше, если
   лиц мало), чтобы не оставалось 2 кадра.
5. **[низ.] Упростить core/-обвязку** — убрать неиспользуемые memory_manager/cache/retry (техдолг).

## 11. Рекомендации по архитектуре / связям

- **Reuse face-кадров из core_face_landmarks закреплён** — правильно; убедиться в shared sampling group.
- **Связка emotion_face → high_level_semantic (event 210)** — восстановить, пере-прогнав keyframes; это прямой
  путь эмоц. таймлайна в семантическую карту.
- **Объединение с micro_emotion** (OpenFace AU) — обе про эмоции лица; возможен общий face-emotion слой (см. L8:
  micro_emotion требует OpenFace docker) вместо двух хрупких инфра-зависимостей.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct+qa | 3 ok + 1 empty | rc=0 | контракт ок |
| U2 ось времени | N=112/246 | fi↑, ts неубыв., probs (N,8) | ось/форма ок |
| U3 health | processed | v/a/probs finite, не константа | сигнал здоров |
| U4 expected-empty | нет лиц | status=empty, no_faces, N=0, rc=0 | пустой путь ок |
| U5 golden | fp32 ×2 | max\|Δvalence\|=max\|Δprobs\|=0.0 | детерминизм |
| U6 разные длины | N=112/246/272 | rc=0 | масштабируется |
| C1 v/a∈[-1,1] | v[-0.73,0.39], a[-0.43,0.87] | диапазон верен |
| C2 probs_rowsum≈1 | [0.9999,1.0000] | softmax валиден |
| C3 различимость | valence CV≈138% | эмоции различают видео |
| C4 proc⊆face_present | True | гейтинг корректен |
| **Реальный storage (мой прогон)** | 6 видео (4 ok, 2 empty) | valence −0.48…+0.01, softmax ок, эмоции варьируются; **keyframes=0 везде, proc=2–10** | ядро живо, keyframes мертвы, сигнал тонкий |

Вывод: **эмоц. ядро (v/a/probs) живо, детерминировано и различимо**, но `keyframes` мертвы в батче, а
subsampling оставляет очень мало кадров → per-video эмоция статистически тонка.

## 13. Интерпретируемость

- **Сильная сторона:** эмоции интуитивны. valence/arousal + доминирующая эмоция → прямой понятный вывод.
- **Добавить:** словесная сводка «преимущественно нейтральное / радостное / напряжённое видео»; эмоц. таймлайн
  (v/a по времени) с keyframes (после пере-прогона); распределение 8 эмоций как диаграмма. Оговорить OOD-смещение
  valence, чтобы не пугать «негативом».

## 14. Польза для моделей

**Средняя.** valence/arousal/8-эмоций — осмысленная эмоц. ось, правдоподобно связанная с вовлечённостью
(эмоц. контент ↔ engagement). Но: (1) только ~2–10 processed-кадров/видео → тонкий, шумный per-video сигнал;
(2) keyframes (эмоц. динамика для событий) мертвы в батче; (3) OOD-смещение valence в минус; (4) сигнал есть
только при лицах. Полезно как дополнительная фича, но не сильная в текущем состоянии данных.

## 15. Польза для аналитиков

**Высокая по понятности, ограничена данными.** «Какие эмоции на лице в видео» (радость/гнев/нейтраль,
позитив/негатив) — наглядная и понятная креатору аналитика. Но per-video на 2–10 кадрах ненадёжна, эмоц.
таймлайн/keyframes пусты, valence смещён. После пере-прогона и калибровки — один из самых relatable компонентов.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Эмоции лица (v/a + 8) — сильная самостоятельная ось |
| 5. Выход (контракт) | 4 | Богатый per-frame v/a/probs + маски; keyframes мертвы в батче |
| 6. Фичи | 3 | circumplex+8 эмоций сильны; keyframes=0, proc тонок, valence смещён |
| 8. Оптимизации | 3 | stride/max_frames/AMP разумны; over-engineered core, AMP-hardcode |
| 9. Слабые места (инверсно) | 2 | keyframes мертвы, 2–10 кадров, OOD-bias, L8 инфра, code-bloat |
| 12. Результаты тестов | 4 | Гейты PASS + golden=0; ядро живо, но keyframes/тонкость |
| 13. Интерпретируемость | 5 | Эмоции — предельно понятны креатору |
| 14. Польза для моделей | 3 | Эмоц. ось ценна, но тонкий/шумный сигнал, keyframes мертвы |
| 15. Польза для аналитиков | 4 | Эмоции наглядны; per-video ненадёжна, timeline пуст |

### Итоговые оценки

- **Польза для моделей: 3/5.** valence/arousal/8-эмоций — осмысленная эмоц. ось, правдоподобно связанная с
  вовлечённостью, детерминированная (golden=0). Но subsampling оставляет 2–10 кадров (шумный per-video сигнал),
  keyframes мертвы в батче, valence OOD-смещён — фактическая польза средняя до пере-прогона/калибровки.
- **Польза для аналитиков: 4/5.** «Эмоции на лице» (радость/гнев/нейтраль, позитив↔негатив) — одна из самых
  relatable креатору аналитик, наглядная и понятная. Балл ниже 5 держат ненадёжность на 2–10 кадрах, пустой
  эмоц. таймлайн (keyframes=0) и смещение valence.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/emotion_face/core/video_processor.py` (2022 строки), `main.py`, `core/advanced_emotion_features.py`
- `.../emotion_face/{utils/validate_emotion_face.py,utils/render.py,docs/SCHEMA.md}`
- `DataProcessor/docs/component_reports/emotion_face/{REPORT_2026-07-16.md,CRITERIA.md}`
- `DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md` (L8 EmoNet source not found; каскад в high_level_semantic)
- Cross-ref: `core_face_landmarks` (dep), `high_level_semantic` (потребитель emo_*/keyframes/event 210), `micro_emotion` (OpenFace AU)
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/emotion_face/emotion_face.npz`
  (4 ok / 2 empty; valence −0.48…+0.01; **keyframes=0 везде; processed=2–10 (stride=4)**)

## 18. Визуализации

![emotion_face overview](emotion_face_overview.png)

`emotion_face_overview.png`: слева — face_present vs processed кадры (stride=4 → EmoNet по ~25%, 2–10 кадров/
видео); центр — valence/arousal ∈[-1,1] по видео (различимы, valence скошен в минус — OOD-склонность EmoNet);
справа — распределение доминирующих эмоций (Happy/Neutral/Anger/Disgust/Fear). Примечание на графике:
**keyframes=0 во всех storage-артефактах** (фикс не пере-прогнан) → эмоц.-событие 210 в high_level_semantic мёртво.
