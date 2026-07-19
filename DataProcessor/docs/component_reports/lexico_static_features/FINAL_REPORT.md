# FINAL REPORT — `lexico_static_features`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `lexico_static_features` (TextProcessor) |
| Артефакт | tp_lex_* (67 полей) в `text_features.npz` |
| Модель | **нет** — статические лексические эвристики (title/description/transcript) |
| Hard deps | title/description (метаданные) + transcript (ASR) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → lexico_static_features ✅ (2026-07-16) |
| Отчёт валидации | [`REPORT_2026-07-16.md`](REPORT_2026-07-16.md), [`CRITERIA.md`](CRITERIA.md) |
| Код | `DataProcessor/TextProcessor/src/extractors/lexico_static_features/` |

## 2. Резюме

`lexico_static_features` — **статические лексические фичи** текста (заголовок/описание/транскрипт): богатый дизайн
на **67 полей** — clickbait-score, читаемость, лексическое разнообразие, named-entity-density, эмодзи, type-token-
ratio, энтропия пунктуации, stopword-ratio, счётчики вопросов/восклицаний и т.д. **Но на реальных данных ценные
сигналы МЕРТВЫ:** все **9/12 transcript-лексических фич = NaN** (читаемость, лексическое разнообразие, rare-word,
stopword — **несмотря на реальный доступный транскрипт!**), `title_clickbait_score=0` (не детектит), `named_entity_
density=NaN`, `emoji_count=0`. Живут только **14/67 базовых** (счётчики символов/слов/пунктуации title/description),
и те mock-driven (5/6 идентичны — mock-заголовки; варьируются лишь из-за -Q6fnPIy). Сильнейшие текст-предикторы
(clickbait/читаемость/NER) не работают.

## 3. Функционал

Работает в TextProcessor. Считает лексику по группам (title/description/clickbait/emoji/transcript):

- **Title:** len_chars/words, avg_word_len, capital_words_ratio, type_token_ratio, stopword_ratio, question/
  exclamation_count, clickbait_score, number_presence, emoji_count, punctuation_ratio.
- **Description:** len_words, emoji_count, num_urls/mentions, has_timestamps.
- **Transcript:** lexical_diversity, readability_score, avg_sentence_len, rare_word_ratio, stopword_ratio,
  orthographic_error_rate, question_ratio, avg_token_frequency_percentile — **все NaN**.
- **Общее:** named_entity_density, punctuation_entropy, special_character_ratio, emoji_diversity.

**Зачем продукту:** лексика — **сильные текст-предикторы CTR/качества:** clickbait (кликбейт-заголовки → CTR),
читаемость/лексическое разнообразие (качество контента), NER (упоминания брендов/персон), эмодзи (стиль). Прямые
actionable SEO/copywriting-сигналы.

## 4. Вход

- **title/description** (метаданные; mock в E2E) + **transcript** (ASR, реальный).
- Нет текста → present=0.

## 5. Выход

- **67 tp_lex_* фич** (title/description/transcript лексика + config-флаги/present/policy).
- **NaN-политика:** transcript-лексика NaN (не вычислена); нет текста → present=0.

## 6. Фичи (важное/неочевидное)

- **9/12 transcript-лексических фич = NaN (главный дефект)** — читаемость, лексическое разнообразие, rare-word,
  stopword, orthographic-error, sentence-len **не вычислены**, ХОТЯ транскрипт реален (transcript_aggregator работает).
  Wiring-gap: богатейший лексический слой (качество спич-контента) мёртв.
- **`title_clickbait_score=0` на всех** — кликбейт-детекция не срабатывает (mock-заголовки или не считает). Ключевой
  CTR-предиктор мёртв.
- **`named_entity_density=NaN`** — NER не производит (упоминания брендов/персон отсутствуют).
- **`emoji_count=0`** — эмодзи не найдены (mock без эмодзи или lib missing).
- **14/67 варьируются** — только базовые счётчики символов/слов/пунктуации, и те mock-driven (5/6 идентичны).
- **11/67 all-NaN, много config-флагов** — group_enabled/present/policy/flag поля-константы + timings = балласт.

## 7. Архитектура / алгоритм

- **Статические эвристики** (regex/подсчёт/словари) над текстом. Модели нет.
- **Сложность:** тривиально (compute_ms 2–3).
- **Детерминизм:** тривиально детерминирован.
- **Проблема:** transcript-блок не считает (NaN), clickbait/NER не производят.

## 8. Оптимизации

- **Мультиисточник** (title/description/transcript) — комплексная лексика.
- **Config-флаги групп** — включаемость блоков.
- (Но ценные блоки мертвы.)

## 9. Слабые места

- **Transcript-лексика NaN (главное)** — 9/12 фич не вычислены при реальном транскрипте; wiring/логика-gap.
  Читаемость/разнообразие спич-контента — сильные сигналы — мертвы.
- **clickbait_score=0, NER=NaN, emoji=0** — ключевые CTR/семантические предикторы не производят.
- **Title/desc лексика mock** — 5/6 идентичны (mock-заголовки).
- **Config-балласт (11 NaN + флаги)** — раздувание 67 полей.
- **Наследует mock title + ASR-качество** — двойная зависимость.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Починить transcript-лексику** — 9/12 NaN при реальном транскрипте; вычислять читаемость/разнообразие/
   rare-word из транскрипта (сильные сигналы качества контента).
2. **[выс.] Починить clickbait_score** — 0 на всех; ключевой CTR-предиктор (проверить на реальных заголовках).
3. **[выс.] Включить NER** — named_entity_density=NaN; упоминания брендов/персон ценны.
4. **[сред.] Реальные заголовки** — title/desc лексика mock.
5. **[сред.] Убрать config-балласт** из feature_values (в meta).

## 11. Рекомендации по архитектуре / связям

- **Transcript-лексика ← transcript** (реальный) — приоритетно оживить (сильные сигналы качества спич-контента).
- **clickbait/NER** — ключевые предикторы; отладить на реальных заголовках/тексте.
- **Config-флаги в meta**, не в фичах.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1–U6 (отчёт) | 28 | авто-штамп | схема/гейты ок |
| **Реальный storage (мой прогон)** | 6 видео | **9/12 transcript-lex NaN; clickbait=0; NER=NaN; emoji=0; 14/67 vary (mock title)** | ценные лексические сигналы мертвы |

Вывод: **богатый дизайн (67 фич), но ценные сигналы мертвы** — transcript-лексика NaN (при реальном транскрипте),
clickbait/NER не производят; живы лишь базовые счётчики mock-заголовков.

## 13. Интерпретируемость

- **Потенциально отличная** — clickbait/читаемость/эмодзи/NER понятны и actionable.
- **Сейчас нечего показывать** — ключевые мертвы; после починки — сильная SEO/copywriting-аналитика.

## 14. Польза для моделей

**Низкая.** Лексика — сильные текст-предикторы (clickbait/читаемость/разнообразие/NER), но на данных ключевые
мертвы (transcript-лексика NaN, clickbait=0, NER=NaN), а живые — mock-title счётчики. Модель получает почти нулевой
лексический сигнал. После починки (transcript-лексика + clickbait + NER) — потенциально сильный вклад.

## 15. Польза для аналитиков

**Низкая (потенциал высок).** Clickbait-score, читаемость, эмодзи-стиль, упоминания — ценнейшие SEO/copywriting-
инсайты («заголовок кликбейтный/скучный, текст сложный/простой»). Но на данных мертвы; после починки — одна из
сильнейших текст-аналитик. Сейчас — базовые счётчики mock-заголовков.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Богатый лексический дизайн; но ключевое мертво |
| 5. Выход (контракт) | 3 | 67 фич, но 11 all-NaN + config-балласт |
| 6. Фичи | 1 | Transcript-лексика NaN, clickbait/NER мертвы, title mock |
| 8. Оптимизации | 3 | Мультиисточник, группы; ценные блоки не считают |
| 9. Слабые места (инверсно) | 1 | Ценные сигналы мертвы, mock, config-балласт |
| 12. Результаты тестов | 2 | Формально ok, но ключевые фичи мертвы |
| 13. Интерпретируемость | 3 | Clickbait/readability понятны (когда есть) |
| 14. Польза для моделей | 2 | Сильный потенциал, факт≈0 (мертво) |
| 15. Польза для аналитиков | 2 | Высокий потенциал, факт≈0 (мертво) |

### Итоговые оценки

- **Польза для моделей: 2/5.** Лексика — сильные текст-предикторы (clickbait ↔ CTR, читаемость/разнообразие ↔
  качество, NER ↔ темы), но на реальных данных ключевые **мертвы**: transcript-лексика NaN (при реальном транскрипте),
  clickbait_score=0, NER=NaN; живы лишь базовые счётчики mock-заголовков. Фактический лексический вклад почти нулевой;
  потенциал (после починки) сильный.
- **Польза для аналитиков: 2/5.** Clickbait-score, читаемость, эмодзи-стиль, упоминания — ценнейшие SEO/copywriting-
  инсайты, но все мертвы на данных (transcript-лексика NaN, clickbait/NER не производят). После починки — одна из
  сильнейших текст-аналитик; сейчас — только базовые счётчики mock-заголовков.

## 17. Источники

- `DataProcessor/TextProcessor/src/extractors/lexico_static_features/`, `schemas/`
- `DataProcessor/docs/component_reports/lexico_static_features/{REPORT_2026-07-16.md, CRITERIA.md}`
- Cross-ref: `transcript_aggregator`/`transcript_chunk_embedder` (реальный транскрипт — но лексика по нему NaN), title/description (mock), `semantics_topics_keyphrases` (родственная лексико-семантика)
- Реальные артефакты: 6 уникальных× tp_lex_* (67) в text_features.npz
  (**9/12 transcript-лексики NaN; clickbait=0; NER=NaN; emoji=0; 14/67 vary — mock title-счётчики; 11 all-NaN**)

## 18. Визуализации

![lexico_static overview](lexico_static_overview.png)

`lexico_static_overview.png`: слева — раскладка 67 лексических фич: **11 all-NaN, 9/12 transcript-лексики NaN**,
только 14 варьируются (базовые mock-title счётчики); справа — сводка: богатый дизайн (clickbait/readability/NER/
emoji/lexical_diversity), но **ценные сигналы мертвы** — transcript-лексика NaN (при реальном транскрипте!),
clickbait=0, NER=NaN. Подтверждает: сильный лексический потенциал не реализован — ключевые блоки не считают.
