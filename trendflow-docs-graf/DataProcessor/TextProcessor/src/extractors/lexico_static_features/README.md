## `lexico_static_features` (Lexical and Static Features Extractor)

### Назначение

Извлекает набор **детерминированных** лексических/статических признаков из текстовых полей видео: заголовка (title), описания (description) и транскрипта (transcript). Компонент **не использует** тяжёлые NLP‑модели (spaCy/langdetect) и не требует сети; любые модели должны быть отдельными extractor’ами через `dp_models`.

**Версия**: 1.2.0  
**Категория**: lexical features  
**GPU**: не требуется (CPU-only)

**Контракт Audit v3**: [`SCHEMA.md`](SCHEMA.md) · [`../../schemas/lexico_static_features_output_v1.json`](../../schemas/lexico_static_features_output_v1.json)  
**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_lexico_static_features_text_npz.py`](utils/validate_lexico_static_features_text_npz.py)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/lexico_static_features_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/lexico_static_features_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/lexico_static_features_l2/`

### Входы

- **`VideoDocument`** с полями:
  - `title` (optional): заголовок (str). В типичном ранне **после** `TagsExtractor` — без inline `#тегов`, если включена очистка.
  - `description` (optional): описание (str), аналогично title.
  - `asr` (optional, preferred): payload от AudioProcessor, `asr.segments[].text` — **source-of-truth** для transcript при `transcript_source_policy` включающем ASR
  - `transcripts` (legacy, optional): допускается **только** при явной политике источника транскрипта (см. ниже)

### Выходы

Экстрактор возвращает `ExtractorResult`. Для dataset/UI **источник истины** — `result.features_flat` (плоский dict числовых скаляров).

#### Основные результаты

- `lexical_stats.metrics`: числовые скалярные признаки (**stable names**, `tp_lex_*`)
- `features_flat`: тот же словарь (канонический плоский вывод)

#### Метаданные

- `device`: устройство обработки (`"cpu"`)
- `version`: версия экстрактора

#### Системные метрики

- `system.pre_init`: снимок системы до инициализации
- `system.post_init`: снимок системы после инициализации
- `system.post_process`: снимок системы после обработки
- `system.peaks.ram_peak_mb`: пиковое использование RAM (MB)
- `system.peaks.gpu_peak_mb`: пиковое использование GPU памяти (MB, всегда 0)

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `error`: описание ошибки (если произошла) или `None`

### Алгоритм обработки

#### 1. Предобработка текстов

- **Title/Description**: `normalize_whitespace`, затем опциональный truncation по `max_*_chars`
- **Transcript**: выбор источника по `transcript_source_policy` (см. конфиг), затем `normalize_whitespace` и truncation
- **Токенизация**: разбиение на токены через `\w+` (Unicode)

#### 2. Признаки заголовка (Title Features)

- `title_len_words`: количество слов
- `title_len_chars`: количество символов
- `title_avg_word_len`: средняя длина слова
- `title_exclamation_count`: количество восклицательных знаков
- `title_question_count`: количество вопросительных знаков
- `emoji_count_title`: количество эмодзи
- `title_type_token_ratio`: отношение уникальных слов к общему количеству (лексическое разнообразие)
- `title_punctuation_ratio`: доля знаков пунктуации
- `title_capital_words_ratio`: доля слов в верхнем регистре
- `title_question_prefix_flag`: наличие вопросительных слов в начале (кто/что/где/когда/почему/зачем/как)
- `title_number_presence`: наличие чисел
- `title_time_mention_flag`: наличие упоминаний времени/даты
- `title_clickbait_score`: оценка clickbait (0.0-1.0) на основе ключевых слов и пунктуации
- `title_stopword_ratio`: доля стоп-слов

#### 3. Признаки описания (Description Features)

- `description_len_words`: количество слов
- `description_num_urls`: количество URL-адресов
- `description_num_mentions`: количество упоминаний (@username)
- `description_has_timestamps_flag`: наличие временных меток (формат 01:23 или 1:02:03)
- `emoji_count_description`: количество эмодзи

#### 4. Признаки транскрипта (Transcript Features)

- `transcript_len_words`: количество слов
- `transcript_avg_sentence_len`: средняя длина предложения в словах
- `lexical_diversity_transcript`: лексическое разнообразие (отношение уникальных слов к общему количеству)
- `rare_word_ratio_transcript`: доля "редких" слов (длиннее 12 символов, как прокси)
- `stopword_ratio_transcript`: доля стоп-слов
- `question_ratio_transcript`: доля вопросительных предложений
- `readability_score_transcript`: прокси читаемости (avg_sentence_len / avg_word_len)
- POS/NER/язык **не вычисляются** в этом extractor’е (только отдельным компонентом через `dp_models`)
- `tp_lex_named_entity_density` зарезервирован (всегда `NaN`), а `tp_lex_named_entity_density_enabled=0`
- `orthographic_error_rate`: доля "неправильно сформированных" токенов (прокси орфографических ошибок)
- `avg_token_frequency_percentile`: прокси частоты токенов (на основе нормализованной длины слова)

#### 5. Общие признаки (Combined Features)

- `emoji_diversity`: разнообразие эмодзи (отношение уникальных эмодзи к общему количеству) по всем полям
- `punctuation_entropy`: энтропия распределения знаков пунктуации (title + description)
- `special_character_ratio`: доля специальных символов (не буквы/цифры/пробелы)
- `upper_lower_ratio_title`: отношение заглавных букв к строчным в заголовке
- Язык (langdetect) **не определяется** в этом extractor’е (только отдельным компонентом через `dp_models`)

### Конфигурация

Поддерживает feature‑gating через параметры конструктора (передаются через `TextProcessor/run_cli.py --extractor-params-json`):

- `enabled` (bool, default true): выключить extractor целиком (stable schema сохраняется; `tp_lex_disabled_by_policy=1`)
- `enable_title` (bool, default true)
- `enable_description` (bool, default true)
- `enable_transcript` (bool, default true)
- `require_transcript` (bool, default false): если `true` и `enable_transcript`, пустой транскрипт после политики → **RuntimeError** (строгий режим с обязательным ASR)
- `enable_emoji` (bool, default **true**)
- `emoji_policy` (`required|optional`, default **`optional`**):
  - `required`: если `enable_emoji=true`, а пакет `emoji` не установлен → **fail-fast** при **инициализации**
  - `optional`: если пакет отсутствует → эмодзи‑фичи **`NaN`** и `tp_lex_emoji_dependency_missing_flag=1`
- `enable_clickbait_heuristic` (bool, default true)
- `transcript_source_policy` (`asr_only|asr_then_legacy|legacy_only`, default `asr_only`)
- `allow_legacy_transcripts` (bool, default false): **legacy alias** (deprecated). Если `true` и `transcript_source_policy` не задан, будет использовано `asr_then_legacy`.
- `max_title_chars` (int|null, default **null** — без усечения)
- `max_description_chars` (int|null, default **null**)
- `max_transcript_chars` (int|null, default **null**)

**Зависимости**:
- `emoji` — *только если включён `enable_emoji=true`* (иначе фичи по эмодзи = NaN)

**Важно**: `langdetect/spacy` намеренно не используются в этом extractor’е (production policy: no-network + ModelManager packaging).
Если понадобится язык/POS/NER — это отдельный качественный extractor через `dp_models`.

### Особенности

- **Комплексный анализ**: более 30 различных лингвистических признаков
- **Мультиязычность**: поддержка русского и английского языков
- **Valid empty semantics**:
  - отсутствие входа — валидно
  - метрики для отсутствующего/выключенного источника → `NaN`
  - presence флаги: `tp_lex_present_*`
- **Stable schema**: все `tp_lex_*` ключи присутствуют всегда
- **Эффективность**: быстрая обработка без тяжёлых моделей
- **Регулярные выражения**: использование эффективных паттернов для извлечения признаков
- **Прокси-метрики**: использование простых эвристик для сложных метрик (читаемость, частота слов)

### Архитектура

1. **Инициализация**: проверка зависимостей по gating (например, `emoji`)
2. **Извлечение текстов**: получение title, description, transcripts из документа
3. **Токенизация**: разбиение текстов на слова и предложения
4. **Вычисление признаков title**: все метрики для заголовка
5. **Вычисление признаков description**: все метрики для описания
6. **Вычисление признаков transcript**: все метрики для транскрипта
7. **Вычисление общих признаков**: метрики, объединяющие несколько полей
8. **NLP модели**: не используются в этом extractor’е (вынесено в отдельные компоненты через `dp_models`)
10. **Агрегация результатов**: сбор всех метрик в единый словарь
11. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

- **Отсутствие полей**: обрабатывается через `getattr()` с значениями по умолчанию
- **Пустые тексты**: признаются валидной “пустотой”; метрики → `NaN`, presence flags → `0/1`
- **emoji**:
  - `emoji_policy=required`: нет зависимости → **error** (fail-fast)
  - `emoji_policy=optional`: нет зависимости → валидный empty для emoji‑фич (NaN)

### Performance characteristics

**Resource costs**:
- **CPU**: низкие-умеренные (в основном регулярные выражения и простые вычисления)
- **GPU**: не используется
- **Измерения**: см. `DataProcessor/docs/models_docs/resource_costs/text_processor_lexico_static_features_costs_v1.json` (placeholder)

**Параметры производительности**:
- Длина транскрипта: линейная сложность для большинства операций
 - NLP модели отсутствуют (spaCy/langdetect вынесены в отдельные компоненты)

### Зависимости

**Обязательные**:
- `numpy`: численные вычисления
- `re`: регулярные выражения
- `unicodedata`: работа с Unicode символами

**Опциональные**:
- `emoji` (только при `enable_emoji=true`)

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **VideoDocument**: схема документа с текстовыми полями
- **text_utils.normalize_whitespace**: нормализация пробелов

### Примечания

1. **ASR источник истины**: transcript берётся из `VideoDocument.asr` (AudioProcessor). `transcripts` используется только если это разрешено политикой `transcript_source_policy`. Для полного Audit v3 — канон **`asr_only`**.
2. **Прокси-метрики** (см. `SCHEMA.md`, tier **analytics** в machine JSON): читаемость, clickbait score, orthographic proxy, стоп-слова — **эвристики**, не NLP ground truth.
3. **Стоп-слова**: используется простой список для русского и английского языков
4. **Clickbait-слова**: список ключевых слов для определения clickbait-контента
5. **Энтропия**: вычисляется через формулу Шеннона с защитой от log(0)

### Feature groups и зависимости (важно для UI)

**Audit v3 full**: рекомендуется **`transcript_source_policy="asr_only"`**; см. `TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md` и [`SCHEMA.md`](SCHEMA.md).

Группы можно включать/выключать независимо, но есть зависимости:
- `emoji_features` → зависит от `enable_emoji` и наличия пакета `emoji`
- `clickbait_features` → зависит от `title_features` (требует title)
- `transcript_features` → зависит от `enable_transcript` и наличия `doc.asr.segments` (или legacy transcripts при `transcript_source_policy=asr_then_legacy|legacy_only`)

При выключении группы соответствующие фичи становятся `NaN`, а `tp_lex_group_*_enabled` отражает конфиг.

### Примеры использования

**Базовое использование**:
```python
extractor = LexicalStatsExtractor()
result = extractor.extract(video_doc)
ff = result["result"]["features_flat"]
title_len = ff["tp_lex_title_len_words"]
clickbait = ff["tp_lex_title_clickbait_score"]
```

**Анализ конкретных признаков**:
```python
result = extractor.extract(video_doc)
ff = result["result"]["features_flat"]

# Анализ заголовка
print(f"Длина заголовка: {ff['tp_lex_title_len_words']} слов")
print(f"Clickbait score: {ff['tp_lex_title_clickbait_score']:.2f}")
print(f"Эмодзи: {ff['tp_lex_title_emoji_count']}")

# Анализ транскрипта
print(f"Лексическое разнообразие: {ff['tp_lex_transcript_lexical_diversity']:.2f}")
print(f"Читаемость: {ff['tp_lex_transcript_readability_score']:.2f}")
```

### Выходные метрики

Канонический вывод для dataset/UI: `result.features_flat` со **стабильными ключами** `tp_lex_*`.

Рекомендуемый способ получения полного списка ключей: смотреть `features_flat` в `main.py` (это source-of-truth).

Ключевые группы:
- `tp_lex_enabled`, `tp_lex_disabled_by_policy`
- `tp_lex_present_*` (presence), `tp_lex_group_*_enabled` (gating)
- `tp_lex_transcript_source_policy_*` + `tp_lex_transcript_source_used_*`
- `tp_lex_*_chars_*` + `tp_lex_*_truncated_flag` (cost-control)
- метрики `tp_lex_title_*`, `tp_lex_description_*`, `tp_lex_transcript_*`, `tp_lex_*` (combined)
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
