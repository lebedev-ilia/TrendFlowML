# FINAL REPORT — `asr_text_proxy_audio_features`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `asr_text_proxy_audio_features` (TextProcessor) |
| Артефакт | tp_asrproxy_* (37 полей) в `text_features.npz` |
| Модель | **нет** — эвристики из ASR-текста + тайминга |
| Hard dep | `asr_extractor` (транскрипт/тайминг) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → asr_text_proxy_audio_features ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Код | `DataProcessor/TextProcessor/src/extractors/asr_text_proxy_audio_features/` |

## 2. Резюме

`asr_text_proxy_audio_features` — **прокси-аудио-фичи из ASR-текста**: TextProcessor не имеет доступа к аудио,
поэтому **оценивает характеристики речи из транскрипта** — темп речи, паузы, слова-паразиты, шум, интонацию (37
`tp_asrproxy_*` фич). На **реальных данных** (5/6 present, -Q6fnPIy present=0 — музыка). **Ключевая находка: здешний
`speech_rate_wpm` — ПРАВИЛЬНЫЙ (word-based)** — 66–217 wpm (реалистично!), в отличие от **сломанного token-based
773 wpm в `asr_extractor`**. То есть в пайплайне ДВА расчёта темпа речи, и этот (по словам) — корректный. Проксирует
delivery-характеристики (пейс/паузы/паразиты) из текста — понятный actionable сигнал подачи. Слабости: часть фич
недоиспользована (filler_ratio=0, noise~0, confidence=0 — наследует Whisper), overlap с asr_extractor, config-балласт.

## 3. Функционал

Работает после `asr_extractor` (транскрипт+тайминг):

1. **Темп:** `speech_rate_wpm` (по **словам**), `words_per_minute_baseline`, `ratio_to_baseline` (быстро/медленно vs норма).
2. **Паузы:** `pause_density` (из временных разрывов сегментов).
3. **Паразиты:** `filler_ratio` (доля «эм/ну»-слов = отшлифованность речи).
4. **Шум/качество текста:** `noise_proxy`, `text_noise_oov_ratio`, `text_noise_rare_ratio` (OOV/редкие слова = ASR-ошибки/шум).
5. **Интонация:** `sentence_intonation` (по пунктуации), `rhythm`.
6. **Confidence:** `confidence_mean/std/present_rate` (из ASR conf — отсутствует у Whisper).

**Зачем продукту:** delivery-характеристики (темп/паузы/паразиты) — **прямой coaching-сигнал**: слишком быстрая/
медленная речь, много пауз, слова-паразиты снижают качество подачи → удержание. Оценка из текста (без аудио) —
умный обход для текст-ветки.

## 4. Вход

- **`asr_extractor`** (hard) — транскрипт (слова) + тайминг; нет ASR → present=0 (-Q6fnPIy музыка).

## 5. Выход

- **37 tp_asrproxy_* фич:** speech_rate_wpm/baseline/ratio, pause_density, filler_ratio, noise_proxy/oov/rare,
  sentence_intonation, confidence_*, word_count, text_chars, char_density, segments_count, audio_duration + enabled/flag-поля.
- **NaN-политика:** нет ASR → present=0.

## 6. Фичи (важное/неочевидное)

- **`speech_rate_wpm` ПРАВИЛЬНЫЙ (word-based) — 66–217 wpm** (главная находка): считает по **словам** (word_count
  18/63/88) → реалистично, в отличие от **сломанного token-based 773 wpm в asr_extractor**. Здешний — корректный;
  использовать его, не asr_extractor-овский.
- **`pause_density`** 0–1.33 — паузы из временных разрывов (реально варьируется); прокси пейса/рваности речи.
- **`filler_ratio`=0** — слова-паразиты не найдены (короткие транскрипты/язык словаря); недоиспользовано.
- **`noise_proxy`~0** (OOV/rare-слова) — прокси ASR-ошибок; на этих данных почти 0.
- **`confidence_*`=0** — Whisper без per-word confidence (наследует, как transcript-компоненты).
- **`ratio_to_baseline`** — темп относительно нормы (быстро/медленно) — хороший нормированный delivery-сигнал.
- **Config-балласт** — много `*_enabled`/`*_flag` полей (intonation/noise/rhythm_enabled и т.п.) — константы, не сигнал.

## 7. Архитектура / алгоритм

- **Эвристики над ASR-текстом** (numpy/regex): подсчёт слов/пауз/паразитов/OOV, интонация по пунктуации.
- **Сложность:** тривиально.
- **Детерминизм:** тривиально детерминирован.

## 8. Оптимизации

- **Проксирование аудио из текста** — умный обход отсутствия аудио в TextProcessor.
- **word-based wpm** (корректный) + baseline-нормировка.
- **Reuse ASR** (не считает заново).

## 9. Слабые места

- **Инконсистентность wpm с asr_extractor (важно)** — здесь word-based (66–217, верно), там token-based (773,
  сломано). Два разных значения темпа речи в пайплайне; надо унифицировать на word-based (этот).
- **Недоиспользованные фичи** — filler_ratio=0, noise_proxy~0, confidence=0 (Whisper) — на данных пусты/тривиальны.
- **Overlap с asr_extractor** — оба дают speech-статистику (токены/слова, длительность); дублирование.
- **Config-балласт** — много enabled/flag-полей-констант в feature_values.
- **Наследует ASR-качество** — Whisper-small ошибки → шум в прокси; present=0 на не-речи.
- **Проксирование ≠ реальное аудио** — паузы/интонация из текста грубее реальных (реальные — в loudness/pitch/onset).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Унифицировать `speech_rate_wpm`** — использовать word-based (этот) везде; починить/убрать token-based
   в asr_extractor (773 wpm неверно).
2. **[сред.] Задействовать filler/noise** — на реальных длинных транскриптах проверить (сейчас 0/тривиально).
3. **[сред.] Убрать config-флаги из фич** (в meta) — как у tags.
4. **[низ.] Свериться с реальным аудио** (loudness/pitch/onset) — прокси-паузы vs реальные; выбрать источник.

## 11. Рекомендации по архитектуре / связям

- **Единый speech_rate** (word-based) для asr/asr_proxy/speech_analysis — устранить 773-баг разом.
- **Delivery-профиль** (темп/паузы/паразиты) → аналитика подачи (coaching).
- **Прокси vs реальное аудио** — определить, где брать паузы/интонацию (текст-прокси vs onset/pitch).

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1–U6 (отчёт) | 28 | авто-штамп | схема/гейты ок |
| **Реальный storage (мой прогон)** | 6 видео (5 present/1 absent) | speech_rate 66–217 (**word-based, верно**); pause 0–1.33; filler/noise~0 | реальный delivery-прокси; wpm корректнее asr_extractor |

Вывод: **реальный, полезный delivery-прокси из текста** с **корректным word-based wpm** (лучше asr_extractor);
часть фич недоиспользована + config-балласт.

## 13. Интерпретируемость

- **Хорошая:** «темп речи N слов/мин (быстро/медленно), паузы, слова-паразиты» — понятный coaching-инсайт.
- **Добавить:** «вы говорите слишком быстро (217 wpm) / много пауз / слова-паразиты» — прямой совет подачи.

## 14. Польза для моделей

**Умеренная.** Delivery-прокси (темп/паузы) — правдоподобный сигнал качества подачи (быстро/монотонно ↔ удержание),
на **реальных данных** с **корректным word-based wpm**. Ограничивают недоиспользованные фичи (filler/noise=0),
overlap с asr_extractor и config-балласт. Полезная delivery-ось; лучше token-based asr_extractor по wpm.

## 15. Польза для аналитиков

**Хорошая.** «Темп речи, паузы, слова-паразиты» — прямой actionable coaching-инсайт для креатора («говорите быстрее/
медленнее, меньше пауз/паразитов»). Реальные данные, понятные метрики (в отличие от опаковых эмбеддингов). Ограничивают
недоиспользованные filler/noise на этом корпусе.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Delivery-прокси из текста — умный обход + реальные данные |
| 5. Выход (контракт) | 3 | 37 фич; корректный wpm, но config-балласт + недоиспользованные |
| 6. Фичи | 3 | word-based wpm верный, pause варьируется; filler/noise=0, overlap |
| 8. Оптимизации | 4 | Проксирование, word-based, reuse ASR |
| 9. Слабые места (инверсно) | 3 | wpm-инконсистентность, config-балласт, overlap, недоиспользование |
| 12. Результаты тестов | 4 | Гейты PASS + реальные данные, корректный wpm |
| 13. Интерпретируемость | 4 | Темп/паузы/паразиты — понятный coaching |
| 14. Польза для моделей | 3 | Delivery-сигнал, реальный; overlap/недоиспользование |
| 15. Польза для аналитиков | 3 | Actionable delivery-coaching; часть фич пуста |

### Итоговые оценки

- **Польза для моделей: 3/5.** Delivery-прокси (темп/паузы) — правдоподобный сигнал качества подачи (быстрая/рваная
  речь ↔ удержание), на реальных данных и с **корректным word-based `speech_rate_wpm`** (66–217, лучше сломанного
  token-based 773 в asr_extractor). Ниже 4 держат недоиспользованные фичи (filler/noise=0), overlap с asr_extractor
  и config-балласт в feature_values.
- **Польза для аналитиков: 3/5.** «Темп речи, паузы, слова-паразиты» — прямой actionable coaching-инсайт («говорите
  быстрее/медленнее, меньше пауз») на реальных данных, понятный (в отличие от опаковых эмбеддингов). Балл держат
  недоиспользованные filler/noise на корпусе и overlap с asr.

## 17. Источники

- `DataProcessor/TextProcessor/src/extractors/asr_text_proxy_audio_features/`, `schemas/`
- `DataProcessor/docs/component_reports/asr_text_proxy_audio_features/{REPORT_2026-07-16.md, CRITERIA.md}`
- Cross-ref: `asr_extractor` (источник; token-based wpm СЛОМАН 773 — здесь word-based верно 66–217), `speech_analysis_extractor` (тоже наследует asr wpm), реальное аудио (loudness/pitch/onset)
- Реальные артефакты: 6 уникальных× tp_asrproxy_* (37) в text_features.npz
  (**5/6 present; speech_rate_wpm 66–217 word-based (верно); pause 0–1.33; filler/noise~0; confidence=0**)

## 18. Визуализации

![asr_text_proxy overview](asr_text_proxy_overview.png)

`asr_text_proxy_overview.png`: слева — `speech_rate_wpm` из **слов** (66–217, реалистично, зелёные) — корректнее
сломанного token-based 773 в asr_extractor; справа — сводка: прокси-аудио из ASR-текста (темп/паузы/паразиты/шум/
интонация), реальные данные (5/6), но filler/noise=0 (недоиспользованы), config-балласт, overlap с asr. Подтверждает:
умный delivery-прокси с корректным word-based темпом; часть фич не задействована.
