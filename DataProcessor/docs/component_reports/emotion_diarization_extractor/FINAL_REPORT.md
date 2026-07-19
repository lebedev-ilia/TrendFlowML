# FINAL REPORT — `emotion_diarization_extractor`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `emotion_diarization_extractor` (AudioProcessor, CPU/GPU) |
| Версия кода | `3.1.2` |
| Схема NPZ | `emotion_diarization_extractor_npz_v*` |
| Артефакт | `.../emotion_diarization_extractor/*.npz` |
| Модель | **SpeechBrain Speech-Emotion-Diarization (WavLM Large)** — 4-класс (angry/neutral/happy/sad) |
| Hard dep | аудио (Segmenter, family=emotion) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → emotion_diarization_extractor ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Баг-реестр | `LOGIC_ERRORS_FOR_CLAUDE.md` L7 (audio_too_short empty) |
| Код | `AudioProcessor/src/extractors/emotion_diarization_extractor/main.py` |

## 2. Резюме

`emotion_diarization_extractor` — **диаризация эмоций речи** на **SpeechBrain WavLM Large** (Speech Emotion
Diarization): определяет эмоцию речи (**angry/neutral/happy/sad**) во времени и её динамику — доминирующая эмоция,
энтропия, переходы, стабильность, разнообразие (7 табличных + per-segment). **Один из более сильных аудио-
компонентов:** реальная нейросеть, работает на **5/6 видео** (1 empty — `-Q6fnPIy audio_too_short`, музыка без
речи), даёт **осмысленные различимые эмоции**: happy/neutral/sad детектятся, `emotion_transitions` 1–7, `entropy`
0.6–1.19 (-4RHVBIik — самая динамичная: 7 переходов). Это **аудио-аналог `emotion_face`** (визуальной эмоции) —
вместе дают мультимодальную эмоц. картину. Ограничения: 4-класс SED груб, эмоции разрежены на talking-head-корпусе
(стабильно-нейтральные), только речь (empty на музыке, L7).

## 3. Функционал

Работает после Segmenter (family=emotion). SpeechBrain WavLM:

1. **WavLM Speech-Emotion-Diarization** → эмоция каждого речевого сегмента (a/n/h/s) + вероятности.
2. **Агрегаты:** `dominant_emotion_id/prob`, `emotion_entropy` (разнообразие), `emotion_transitions_count`,
   `emotion_stability_score`, `emotion_diversity_score`.
3. **Per-segment/распределения:** `emotion_id/probs/labels`, `emotion_distribution`, `duration/segments_per_emotion`,
   `emotion_confidence`, `quality_metrics`.

**Зачем продукту:** эмоция речи — **сильный сигнал вовлечённости**: эмоциональная подача (радость/драма/гнев)
удерживает и провоцирует реакции; монотонно-нейтральная — усыпляет. Динамика эмоций (переходы) = живость. Аудио-
эмоция дополняет визуальную (emotion_face): что слышно + что видно на лице.

## 4. Вход

- **Аудио** (Segmenter, family=emotion) — эмоц. сегменты <5s → `audio_too_short` empty (L7); нет аудио → audio_missing.
- **SpeechBrain WavLM checkpoint** (local artifact).

## 5. Выход

- **7 табличных фич:** `emotion_entropy`, `dominant_emotion_id/prob`, `emotion_transitions_count`,
  `emotion_stability_score`, `emotion_diversity_score`, `segments_count`.
- **Per-segment/распределения:** `emotion_id/probs/labels (a/n/h/s)`, `emotion_distribution`, `mean_probs`,
  `duration/segments_per_emotion`, `emotion_confidence`, `quality_metrics`.
- **NaN-политика:** empty (audio_too_short) → 6/7 NaN.

## 6. Фичи (важное/неочевидное)

- **`dominant_emotion_id` + labels (a/n/h/s)** — доминирующая эмоция речи: на корпусе happy/neutral/sad детектятся
  (не всё нейтральное) — модель различает. dominant_prob 0.5–0.81 (уверенность).
- **`emotion_transitions_count` / `entropy` — динамика (несущий инсайт)** — -4RHVBIik: 7 переходов, entropy 1.19
  (эмоционально живая речь); остальные: стабильно 1 переход, entropy 0.6 (монотонная подача). **Живость vs монотонность.**
- **`emotion_stability_score`** 0.89–0.97 — насколько постоянна эмоция; высокая = одна эмоция весь ролик.
- **4-класс SED (a/n/h/s)** — грубая гамма (нет surprise/fear/disgust как у emotion_face); возможна sad-склонность
  (3/5 → sad — либо контент такой, либо over-prediction).
- **Empty на музыке (L7)** — -Q6fnPIy (тон, без речи) → audio_too_short; корректно (нет речевой эмоции).
- **Аудио-аналог emotion_face** — audio-эмоция (что слышно) + visual-эмоция (что на лице) = мультимодально.

## 7. Архитектура / алгоритм

- **SpeechBrain Speech-Emotion-Diarization (WavLM Large)** — предобученная нейросеть, эмоция по речевым сегментам.
- **Сложность:** WavLM-инференс (тяжелее DSP-фич); torch 2.12/speechbrain 1.1, CPU/GPU.
- **Детерминизм:** заявлен PASS.

## 8. Оптимизации

- **Готовая SpeechBrain-модель** — не обучается, local artifact.
- **Агрегаты динамики** (transitions/entropy/stability) — эмоц. дуга компактно.
- **Гранулярный empty** (audio_too_short) — честный маркер нет-речи.

## 9. Слабые места

- **4-класс SED груб** — angry/neutral/happy/sad; нет surprise/fear/disgust (у визуального emotion_face 8 классов).
  Возможная sad-склонность (3/5 sad) — либо контент, либо over-prediction (не верифицировано).
- **Эмоции разрежены на корпусе** — 4/5 стабильно-монотонны (1 переход, high stability); реальная эмоц. динамика
  только у 1 видео. Talking-head-контент эмоционально ровный → сигнал беден на этом наборе.
- **Только речь** — empty на музыке/не-речи (L7 audio_too_short); эмоция инструментальной музыки не ловится.
- **L7-хрупкость** — audio_too_short при эмоц. сегментах <5s; может ложно срабатывать на stale NPZ (L10).
- **Дублирование эмоции с emotion_face** — две эмоц.-ветки (аудио/видео); нужна fusion, не два разрозненных сигнала.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[сред.] Fusion аудио+видео эмоции** (emotion_diarization + emotion_face) — единая эмоц. дуга (слышно+видно),
   а не два разрозненных сигнала.
2. **[сред.] Проверить sad-склонность** — 3/5 sad подозрительно; калибровать/верифицировать на разметке.
3. **[сред.] Рассмотреть более богатую гамму** (>4 классов) или valence/arousal (как emotion_face) для согласованности.
4. **[низ.] Ужесточить L7** — различать реальный audio_too_short от stale NPZ (проверка segments_total vs Segmenter).
5. **[низ.] Ярлыки динамики** — «эмоционально живая / монотонная подача» из transitions/entropy.

## 11. Рекомендации по архитектуре / связям

- **Мультимодальная эмоция** — audio (этот) + face (emotion_face) → high_level_semantic/Fusion как один эмоц. слой.
- **Связка с pitch/voice_quality** — интонация (f0_std) + эмоция речи = согласованный вокально-эмоц. профиль.
- **Reuse диаризации** — speaker_diarization + emotion_diarization = «кто с какой эмоцией говорит».

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1–U6 | 14 NPZ (5 ok/9 empty) | все PASS | схема/ось/health/empty ок |
| — эмоции | 5 ok | happy/neutral/sad, prob 0.5–0.81 | реальные различимые эмоции |
| — динамика | 5 ok | transitions 1–7, entropy 0.6–1.19 | живость/монотонность различимы |
| L7 empty | audio_too_short | -Q6fnPIy (музыка) | корректный empty на не-речи |
| **Реальный storage (мой прогон)** | 6 видео (5 ok/1 empty) | dom h/n/s×3, stability 0.89–0.97, 0 NaN на ok | работает, различимо; эмоции разрежены |

Вывод: **реальный нейросетевой аудио-эмоц. компонент, работает на 5/6**, различим по эмоции и динамике; гамма
груба, эмоции на корпусе разрежены (монотонны).

## 13. Интерпретируемость

- **Хорошая:** «речь happy/neutral/sad», «эмоционально живая/монотонная» — понятно и relatable.
- **Добавить:** эмоц. таймлайн речи (transitions по времени); fusion с лицевой эмоцией; оговорить 4-класс/sad-склонность.

## 14. Польза для моделей

**Умеренно-высокая.** Реальная нейросетевая эмоция речи + динамика — правдоподобно сильный сигнал вовлечённости
(эмоц. подача ↔ реакции), работает на большинстве (5/6), различим. Аудио-аналог визуальной эмоции (мультимодальность).
Ограничивают грубая 4-гамма, разрежённость эмоций на talking-head-корпусе, only-speech. Ценная эмоц. ось.

## 15. Польза для аналитиков

**Умеренно-хорошая.** «Эмоциональная окраска речи (happy/neutral/sad) и её динамика (живо/монотонно)» — понятный,
relatable инсайт (совет «эмоциональнее подавать»). Ограничивают грубость гаммы, возможная sad-склонность и
неприменимость к музыке/не-речи.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Реальная нейросетевая эмоция речи + динамика; аудио-аналог emotion_face |
| 5. Выход (контракт) | 4 | 7 фич + per-segment распределения/labels; грубая 4-гамма |
| 6. Фичи | 3 | dominant/transitions/entropy осмысленны; 4-класс, sad-склонность? |
| 8. Оптимизации | 4 | Готовая SpeechBrain, компактные агрегаты динамики |
| 9. Слабые места (инверсно) | 3 | Грубая гамма, разрежённость, only-speech, дубль с emotion_face |
| 12. Результаты тестов | 4 | Гейты PASS, 5/6 работает, различимо |
| 13. Интерпретируемость | 4 | «Эмоция речи / живость» relatable |
| 14. Польза для моделей | 4 | Сильный эмоц. сигнал, мультимодальный; гамма/разрежённость |
| 15. Польза для аналитиков | 3 | Relatable эмоция; грубая гамма, only-speech |

### Итоговые оценки

- **Польза для моделей: 4/5.** Реальная нейросетевая эмоция речи (WavLM) + динамика (переходы/энтропия) —
  правдоподобно сильный предиктор вовлечённости (эмоц. подача ↔ реакции), работает на 5/6, различим, и служит
  аудио-аналогом визуальной эмоции (мультимодальность). Ниже 5 держат грубую 4-классовую гамму (a/n/h/s vs 8 у
  emotion_face), разрежённость эмоций на talking-head-корпусе и неприменимость к музыке.
- **Польза для аналитиков: 3/5.** «Эмоциональная окраска и динамика речи» — понятный relatable инсайт (эмоц.
  подача — actionable). Балл держат грубость гаммы, возможная sad-склонность (3/5 sad не верифицировано) и
  неприменимость к не-речевому контенту.

## 17. Источники

- `AudioProcessor/src/extractors/emotion_diarization_extractor/main.py` (utils, docs), `utils/validate_*`
- `DataProcessor/docs/component_reports/emotion_diarization_extractor/{REPORT_2026-07-16.md, CRITERIA.md}`
- `DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md` (L7 audio_too_short)
- Cross-ref: `emotion_face` (визуальная эмоция — fusion), `pitch`/`voice_quality` (вокально-эмоц.), `speaker_diarization` (кто с эмоцией)
- Реальные артефакты: 6 уникальных× `.../emotion_diarization_extractor/*.npz`
  (**5 ok (dom happy/neutral/sad, prob 0.5–0.81, transitions 1–7) / 1 empty (audio_too_short, музыка); 0 NaN на ok**)

## 18. Визуализации

![emotion_diarization overview](emotion_diarization_overview.png)

`emotion_diarization_overview.png`: слева — `emotion_entropy` + `transitions_count` по видео с доминирующей эмоцией
(a/n/h/s): -4RHVBIik эмоционально живая (entropy 1.19, 7 переходов) vs монотонные стабильные (остальные); -Q6fnPIy
empty (музыка, audio_too_short); справа — сводка: реальная SpeechBrain WavLM 4-эмоц. диаризация, работает на 5/6,
аудио-аналог emotion_face, но грубая гамма и эмоции разрежены на talking-head-корпусе. Подтверждает: сильный
реальный аудио-эмоц. компонент, ограниченный грубостью гаммы и речевой природой.
