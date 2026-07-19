# FINAL REPORT — `rhythmic_extractor`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `rhythmic_extractor` (AudioProcessor, CPU) |
| Версия кода | `2.0.1` |
| Схема NPZ | `rhythmic_extractor_npz_v2` |
| Артефакт | `.../rhythmic_extractor/*.npz` |
| Модель | **нет** — librosa (beat-tracking + ритмическая статистика) |
| Hard dep | аудио (Segmenter) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → rhythmic_extractor ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Код | `AudioProcessor/src/extractors/rhythmic_extractor/` |

## 2. Резюме

`rhythmic_extractor` — **самый богатый анализатор ритма** (третий после `tempo` и `onset`): tempo_bpm,
beats_count/density, регулярность, tempo_variation, beat_consistency, а также polyrhythm, syncopation,
metrical_stability, beat_strength, IBI-tempo, статистики периодов (~25 ключей). **Полностью живой** (13 ok/4 empty,
0 NaN, health_score=1.0). **Сильно пересекается** с tempo (`rhythm_tempo_bpm` снова попадает в librosa-дефолт
129.2 на речи) и onset (`rhythm_beats_count` ≈ onset_count). **Но добавляет ценное:** `rhythm_regularity` и
`tempo_variation` **честно разделяют ритмичную музыку от аритмичной речи** — -Q6fnPIy (музыка): regularity=0.56,
variation=0.8 (стабильный бит); речевые: regularity 0.02–0.07, variation 13–57 (нерегулярно). Это то, чего
не смог tempo (его confidence не отражал наличие бита). Исправлены 2 бага (мёртвый код сохранения beat_times,
missing import). Богатый, но избыточный — кандидат на консолидацию ритм-трио.

## 3. Функционал

Работает после Segmenter (аудио). librosa beat-tracking + статистика:

1. **Beat-tracking** → `rhythm_tempo_bpm`, `beats_count`, `beat_density`, `beat_strength_mean/std`.
2. **Регулярность/стабильность:** `rhythm_regularity`, `beat_consistency`, `metrical_stability`, `tempo_variation`,
   `tempo_std/min/max/mean`.
3. **Ритмическая структура:** `polyrhythm_score`, `syncopation_score`, IBI-tempo, периоды (avg/median/min/max/std).

**Зачем продукту:** насколько выражен и регулярен ритм — **сигнал музыкальности/энергетики**: регулярный бит
(танцевальная/фоновая музыка) vs аритмичная речь. `regularity`/`tempo_variation` отличают музыку от речи лучше,
чем сам BPM.

## 4. Вход

- **Аудио** (Segmenter) — нет аудио → `status=empty` (нулевой сегмент от Segmenter).

## 5. Выход

- **9 табличных фич** (`rhythm_tempo_bpm`, `beats_count`, `beat_density`, `regularity`, `tempo_variation`,
  `beat_consistency` + мета) + **~16 доп. ключей** (beat_strength, polyrhythm, syncopation, metrical_stability,
  IBI-tempo, period-статистики, tempo_min/max/mean/std, median_bpm).
- **Beat-times** сохраняются в `_artifacts/*_beat_times_sec.npy` при большом объёме (после фикса).
- `segment_*_sec`, `segment_mask`. NaN-политика: empty → NaN.

## 6. Фичи (важное/неочевидное)

- **`rhythm_regularity` — главный value-add (различает музыку/речь)** — 0.56 у музыки (стабильный бит), 0.02–0.07
  у речи (нерегулярно). Это то, чего не дал tempo. Осмысленный «есть ли ритм».
- **`rhythm_tempo_variation`** — стабильность темпа: 0.8 (музыка, ровно) vs 13–57 (речь, скачет). Тоже music-детектор.
- **`rhythm_beats_count`** (11–767) — очень различима, но ≈ `onset_count` (дубль onset).
- **`rhythm_tempo_bpm` — снова дефолт 129.2** на -3Mbinqzi (тот же librosa-артефакт, что у tempo_extractor).
  BPM избыточен и наследует ту же проблему.
- **polyrhythm/syncopation/metrical_stability** — продвинутые ритм-метрики; вероятно слабы на речевом корпусе
  (мало сложного ритма), но задел для музыкального контента.

## 7. Архитектура / алгоритм

- **librosa beat-tracking** + обширная ритмическая статистика (numpy).
- **Сложность:** дёшев, CPU.
- **Детерминизм:** заявлен PASS (librosa CPU); health_score=1.0.
- **Баги (исправлены):** `_save_beat_times_npy` был мёртвым кодом (данные >10000 битов терялись); missing `Any` в валидаторе.

## 8. Оптимизации

- **Чистый librosa** — без GPU.
- **Богатый ритм-профиль** (~25 метрик) — от базового BPM до polyrhythm/syncopation.
- **Beat-times в .npy** при большом объёме (после фикса) — не раздувать NPZ.

## 9. Слабые места

- **Третий анализатор ритма (главное)** — дублирует `tempo` (BPM, тот же дефолт-129) и `onset` (beats_count ≈
  onset_count). Три компонента считают beat независимо (три librosa-прохода). Явная фрагментация.
- **`rhythm_tempo_bpm` наследует дефолт-129 на речи** — как у tempo; BPM-часть спурьёзна на не-музыке.
- **Продвинутые метрики вероятно слабы на речи** — polyrhythm/syncopation осмысленны на музыке, шум на talking-head-корпусе.
- **~25 фич, сильная взаимокорреляция** — beats_count/density, tempo_bpm/median_bpm/IBI_tempo дублируют друг друга.
- **Опаковость** — polyrhythm/metrical_stability непонятны без интерпретации.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Консолидировать ритм-трио** (tempo/onset/rhythmic) в один beat-модуль — один librosa-проход, единый
   набор ритм-метрик; сейчас тройной дубль.
2. **[выс.] Взять `regularity`/`tempo_variation` как канонический «есть ли ритм»-gate** (лучше tempo-confidence) —
   и не выдавать BPM при низкой регулярности (устранить дефолт-129).
3. **[сред.] Проредить ~25 коррелирующих фич** — оставить ядро (regularity, tempo, beat_density, syncopation).
4. **[низ.] Пометить продвинутые метрики** как «музыкальные» (не показывать на речи).

## 11. Рекомендации по архитектуре / связям

- **Единый ритм-модуль** = onset-envelope → beat-tracking → {tempo, density, regularity, syncopation}.
- **regularity-gate** для всего ритм-семейства — определять музыкальность один раз.
- **Кросс-модально с video_pacing** — аудио-ритм + видео-темп монтажа.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate | 17 NPZ | 17/17 rc=0 (после 2 фиксов) | схема ок |
| U2 ось | 15 ok | монотонны; end==start при mask=False by design | ось ок |
| U3 finite | ok | health_score=1.0, nan=0, tempo∈[40,300], reg∈[0,1] | здоров |
| U4 expected-empty | 4 | пустой путь ок | ок |
| U5 golden | — | PASS (librosa) | детерминизм |
| U6 разные длины | N разные | ок | масштаб ок |
| **Реальный storage (мой прогон)** | 6 видео | reg 0.02↔0.56, var 0.8↔57 (музыка/речь); beats 11–767; **bpm снова 129.2 дефолт** | различимо (regularity!), но дубль tempo/onset |

Вывод: **живой, богатый, с ценной регулярностью** (music/speech-детектор), но избыточен (третий beat-анализатор),
BPM-часть наследует дефолт-129.

## 13. Интерпретируемость

- **Средняя:** «выраженный регулярный ритм / аритмично» (regularity) — понятно; polyrhythm/syncopation — опаковы.
- **Добавить:** «в видео есть ритмичная музыка (regularity высокая)»; не показывать BPM без ритма.

## 14. Польза для моделей

**Умеренная.** `regularity`/`tempo_variation` — реально различают музыку/речь (то, чего не дал tempo), полезный
music-детектор. Но beats_count дублирует onset, tempo_bpm дублирует tempo (с дефолт-129), ~25 фич коррелируют.
Ценность = regularity/variation сверх onset/tempo; остальное избыточно.

## 15. Польза для аналитиков

**Умеренная.** «Есть ли выраженный регулярный ритм/музыка» (regularity) — понятный инсайт для музыкального контента.
Ограничивают дубль с tempo/onset, опаковость продвинутых метрик, спурьёзный BPM на речи.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Богатый ритм-профиль + music-детектор; третий дубль beat |
| 5. Выход (контракт) | 3 | ~25 фич; сильная взаимокорреляция, BPM-дубль |
| 6. Фичи | 3 | regularity/variation ценны; beats/bpm дублируют onset/tempo |
| 8. Оптимизации | 3 | Богато; но третий librosa-проход, 2 бага были |
| 9. Слабые места (инверсно) | 3 | Дубль трио, дефолт-129, коррелирующие фичи |
| 12. Результаты тестов | 4 | Гейты PASS, health=1.0, 2 бага чинены |
| 13. Интерпретируемость | 3 | Regularity понятна; polyrhythm опаков |
| 14. Польза для моделей | 3 | regularity/variation — value-add; остальное дубль |
| 15. Польза для аналитиков | 3 | «Есть ритм» понятно; дубль/опаковость |

### Итоговые оценки

- **Польза для моделей: 3/5.** `rhythm_regularity`/`tempo_variation` реально отделяют ритмичную музыку от
  аритмичной речи (music-детектор, которого не дал `tempo`) — полезный value-add. Ниже 4 держат тройное
  дублирование (beats_count≈onset, tempo_bpm≈tempo с тем же дефолт-129) и ~25 коррелирующих фич; чистая ценность —
  regularity/variation сверх onset/tempo.
- **Польза для аналитиков: 3/5.** «Есть ли выраженный регулярный ритм/музыка» — понятный инсайт (лучше tempo-
  confidence). Ограничивают избыточность с tempo/onset, опаковость polyrhythm/syncopation и спурьёзный BPM на речи.

## 17. Источники

- `AudioProcessor/src/extractors/rhythmic_extractor/` (utils, docs), `utils/validate_rhythmic.py`
- `DataProcessor/docs/component_reports/rhythmic_extractor/{REPORT_2026-07-16.md, CRITERIA.md}`
- Cross-ref: `tempo_extractor` (BPM-дубль, дефолт-129), `onset_extractor` (beats-дубль), `chroma_extractor` (музыкальность)
- Реальные артефакты: 6 уникальных× `.../rhythmic_extractor/*.npz`
  (**все ok; regularity 0.02–0.56, beats 11–767; rhythm_tempo_bpm снова 129.2 дефолт; 0 NaN; health=1.0**)

## 18. Визуализации

![rhythmic_extractor overview](rhythmic_extractor_overview.png)

`rhythmic_extractor_overview.png`: слева — `rhythm_regularity` vs `tempo_variation`: **value-add** — регулярность/
вариация честно отделяют музыку (-Q6fnPIy: reg 0.56/var 0.8 — стабильный бит) от речи (reg 0.02–0.07/var 13–57 —
аритмично), чего не смог tempo; справа — сводка ~25 фич и **тройное дублирование** (rhythm_tempo_bpm≈tempo с тем же
дефолт-129, beats_count≈onset) + 2 исправленных бага. Подтверждает: богатый живой ритм-набор с полезной
регулярностью, но избыточен — кандидат на слияние ритм-трио в один модуль.
