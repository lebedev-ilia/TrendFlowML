## `semantics_topics_keyphrases` (Semantic Topics and Keyphrases Extractor)

### Назначение

Извлекает **глобальные (сопоставимые между видео)** темы из текста через **retrieval по фиксированной taxonomy** (bundled `topics.jsonl` + embeddings через `dp_models`), а также ключевые фразы и дешёвые стилистические proxy-флаги.

Важно: component больше **не обучает** темы per-video (BERTopic/KMeans) — это было несопоставимо между видео и плохо для ML/аналитики.

**Версия**: 2.1.0  
**Категория**: topic modeling, keyphrase extraction, style analysis  
**GPU**: поддерживается (cuda), опционально fp16

**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_semantics_topics_keyphrases_text_npz.py`](utils/validate_semantics_topics_keyphrases_text_npz.py)

**Контракт:** [SCHEMA.md](./SCHEMA.md) · machine [`schemas/semantics_topics_keyphrases_output_v1.json`](../../schemas/semantics_topics_keyphrases_output_v1.json) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/semantics_topics_keyphrases_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/semantics_topics_keyphrases_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/semantics_topics_keyphrases_l2/`

### Входы

- **`VideoDocument`** с полями:
  - `asr.segments[].text` — preferred transcript source-of-truth
  - `transcripts` — legacy fallback (только если `allow_legacy_transcripts=True`)
  - `title`: заголовок видео (str)
  - `description`: описание видео (str)

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Основные результаты

Экстрактор возвращает **privacy-safe** `result.features_flat` (только числовые скаляры), префикс: `tp_topics_*`.

**Основные флаги**:
- `tp_topics_present`: наличие данных (1.0 если есть текст, 0.0 если пусто)
- `tp_topics_disabled_by_policy`: экстрактор отключен через `enabled=False` (1.0 если отключен)

**Присутствие данных**:
- `tp_topics_text_chars`: количество символов в объединенном тексте
- `tp_topics_has_asr`: наличие ASR транскрипта (1.0/0.0)
- `tp_topics_has_title`: наличие заголовка (1.0/0.0)
- `tp_topics_has_description`: наличие описания (1.0/0.0)

**Topics (retrieval)**:
- `tp_topics_topic_top1_id`: ID топ-1 темы
- `tp_topics_topic_top1_score`: сходство с топ-1 темой
- `tp_topics_topic_top1_prob`: вероятность топ-1 темы (softmax)
- `tp_topics_topic_top{i}_id`, `tp_topics_topic_top{i}_score`, `tp_topics_topic_top{i}_prob` (i=1..top_k_slots; стабильная схема)
- `tp_topics_entropy_topk`: энтропия Шеннона распределения вероятностей по топ-K темам
- `tp_topics_entropy_topk_norm`: нормализованная энтропия (энтропия / log(K))
- `tp_topics_perplexity_topk`: perplexity = exp(энтропия)

**Keyphrases**:
- `tp_topics_keyphrases_count`: количество извлеченных ключевых фраз
- `tp_topics_keyphrase_score_top1`: оценка топ-1 ключевой фразы
- `tp_topics_keyphrase_score_mean`: средняя оценка всех ключевых фраз
- `tp_topics_keyphrases_dim`: размерность эмбеддингов ключевых фраз (если включены embeddings)

**Keyphrases (privacy-safe export)**:
- `tp_topics_kp_top{i}_present`: наличие ключевой фразы в слоте i (1.0/0.0)
- `tp_topics_kp_top{i}_hash01`: хеш ключевой фразы (первый байт SHA256)
- `tp_topics_kp_top{i}_len`: длина ключевой фразы в символах
  (i=1..keyphrase_slots) при `export_keyphrases_mode="hashed"`

**Style proxies (heuristics, configurable)**:
- `tp_topics_style_faq_qmarks`: количество предложений, заканчивающихся на "?"
- `tp_topics_style_instructional_flag`: присутствие инструктивных ключевых слов (1.0/0.0)
- `tp_topics_style_audience_flag`: присутствие обращений к аудитории (1.0/0.0)
- `tp_topics_style_cta_flag`: присутствие призывов к действию (1.0/0.0)

**Конфигурационные флаги** (отражают настройки экстрактора):
- `tp_topics_enable_topic_distribution`: включено ли извлечение тем (1.0/0.0)
- `tp_topics_enable_keyphrases`: включено ли извлечение ключевых фраз (1.0/0.0)
- `tp_topics_enable_keyphrase_embeddings`: включены ли эмбеддинги ключевых фраз (1.0/0.0)
- `tp_topics_export_keyphrases_mode_raw`: режим экспорта "raw" (1.0/0.0)
- `tp_topics_export_keyphrases_mode_hashed`: режим экспорта "hashed" (1.0/0.0)
- `tp_topics_export_keyphrases_mode_none`: режим экспорта "none" (1.0/0.0)
- `tp_topics_enable_style_flags`: включены ли стилистические флаги (1.0/0.0)
- `tp_topics_allow_legacy_transcripts`: разрешены ли legacy транскрипты (1.0/0.0)
- `tp_topics_top_k_topics`: количество тем для извлечения
- `tp_topics_top_k_slots`: количество слотов для топ-K тем
- `tp_topics_temperature`: температура для softmax

**Optional raw export (debug / verified-only)**:
- `tp_topics_keyphrases_raw`: список ключевых фраз (только при `export_keyphrases_mode="raw"`)

#### In-memory registry (`doc.tp_artifacts`)

Экстрактор сохраняет данные в in-memory registry для использования downstream экстракторами:

**Topics distribution**:
- `doc.tp_artifacts["topics"]["topk_distribution"]`: словарь с топ-K распределением тем:
  - `k`: количество тем
  - `temperature`: температура для softmax
  - `topic_ids`: список ID тем
  - `topic_scores`: список сходств с темами
  - `topic_probs`: список вероятностей тем
  - `entropy_topk`: энтропия распределения
  - `model_name`, `model_version`, `model_weights_digest`: метаданные модели
  - `topics_db_spec_name`, `topics_db_weights_digest`: метаданные topics DB

**Keyphrase embeddings** (если `enable_keyphrase_embeddings=True`):
- `doc.tp_artifacts["topics"]["keyphrase_embeddings"]`: словарь с метаданными эмбеддингов:
  - `relpath`: относительный путь к `.npy` файлу с эмбеддингами
  - `count`: количество ключевых фраз
  - `dim`: размерность эмбеддингов
  - `model_name`, `model_version`, `weights_digest`: метаданные модели

#### Метаданные

- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `version`: версия экстрактора
- `model_version`: embedding модель для retrieval/keyphrases (например, `"intfloat/multilingual-e5-large"`)

#### Системные метрики

- `system.pre_init`: снимок системы до инициализации
- `system.post_init`: снимок системы после инициализации
- `system.post_process`: снимок системы после обработки
- `system.peaks.ram_peak_mb`: пиковое использование RAM (MB)
- `system.peaks.gpu_peak_mb`: пиковое использование GPU памяти (MB)

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `error`: описание ошибки (если произошла) или `None`

### Алгоритм обработки

#### 1. Подготовка текста

**Процесс**:
1. **Транскрипт**: `doc.asr.segments[].text` (preferred), legacy `doc.transcripts` только при `allow_legacy_transcripts=True`
2. **Добавление метаданных**: добавление `title` и `description` к тексту
3. **Нормализация**: применение `normalize_whitespace()` к полному тексту
4. **Лимит**: `max_text_chars` (cost-control)

#### 2. Topics retrieval (global taxonomy)

Источник тем: резолвится через `dp_models` spec (`topics_db_spec_name`, по умолчанию `"topics_taxonomy_v1"`). Topics DB содержит список тем с промптами на русском и английском языках.

Процесс:
1. Загружаем список prompts из topics DB (через `dp_models.resolve()`).
2. Prompt embeddings строятся один раз и сохраняются в **cache** (`default_cache_dir()/tp_topics_db/*.npy`) — это не `result_store`. Кеш ускоряет повторные запуски и управляется параметрами `cache_enabled`, `cache_ttl_s`, `cache_max_total_mb`.
3. Кодируем `full_text` через embedding модель (через `dp_models`, с L2-нормализацией).
4. Считаем cosine similarity (через dot product на нормализованных векторах) и агрегируем prompt→topic через `max` (для каждого topic_id берется максимальный score среди всех его prompts).
5. Выдаём top‑K тем + softmax‑probabilities (с температурой `temperature`) и энтропию.

#### 3. Keyphrases

Текущая реализация: лёгкий deterministic scorer (без внешних зависимостей).

**Алгоритм**:
1. **Токенизация**: разбиение текста на слова (поддержка русского и английского)
2. **Генерация кандидатов**: создание n-грамм (1-3 слова) с фильтрацией стоп-слов
3. **Оценка**: для каждой фразы вычисляется score = `tf * (1 / (1 + first_position)) * length_bonus`:
   - `tf`: частота появления фразы в тексте
   - `first_position`: позиция первого вхождения фразы
   - `length_bonus`: бонус за длину (1.0 + 0.1 * (количество_слов - 1))
4. **Фильтрация**: удаление фраз длиннее `max_keyphrase_len_chars`
5. **Сортировка**: сортировка по score и выбор топ-K фраз

**Более качественно, но сложнее (зафиксировано как future improvement)**:
- YAKE (рекомендуется) или KeyBERT/MMR (через `dp_models`) — если потребуется higher precision/recall.

#### 4. Стилистические признаки

**FAQ-подобные вопросы**:
- Подсчёт предложений, заканчивающихся на `?`

**Инструктивный язык**:
- Поиск ключевых слов: `["нажмите", "сделайте", "кликните"]`

**Обращение к аудитории**:
- Поиск ключевых слов: `["вы", "ты", "тебя", "вас"]`

**Призыв к действию**:
- Поиск ключевых слов: `["подпишитесь", "лайк", "комментарий"]`

**Важно**: placeholder-метрики удалены из prod-выхода. Более качественные версии (NER/FAQ classifier/coherence) — отдельными компонентами.

### Конфигурация

```python
{
    "device": "cpu",                                          # "cpu" | "cuda"
    "artifacts_dir": None,                                    # per-run `text_processor/_artifacts`
    "enabled": True,
    "enable_topic_distribution": True,
    "topics_db_spec_name": "topics_taxonomy_v1",
    "model_name": "intfloat/multilingual-e5-large",
    "top_k_topics": 5,
    "top_k_slots": 5,
    "temperature": 0.07,
    "enable_keyphrases": True,
    "max_keyphrases": 10,
    "keyphrase_slots": 10,
    "max_keyphrase_len_chars": 64,
    "export_keyphrases_mode": "none",                          # none | raw | hashed
    "enable_keyphrase_embeddings": True,
    "enable_style_flags": True,
    "style_instruction_words_ru": None,                       # list[str] или None (дефолт: ["нажмите", "сделайте", ...])
    "style_audience_words_ru": None,                           # list[str] или None (дефолт: ["вы", "ты", ...])
    "style_cta_words_ru": None,                                # list[str] или None (дефолт: ["подпишитесь", "лайк", ...])
    "allow_legacy_transcripts": False,
    "transcript_source_policy": "asr_only",                    # asr_only | asr_then_legacy | legacy_only
    "max_text_chars": 20000,
    "cache_enabled": True,                                     # включить кеш для prompt embeddings
    "cache_ttl_s": 7 * 24 * 3600.0,                           # TTL кеша в секундах (7 дней)
    "cache_max_total_mb": 512                                  # максимальный размер кеша в MB
}
```

**Параметры**:
- `device`: устройство обработки (cpu или cuda)
- `artifacts_dir`: директория для сохранения артефактов (эмбеддинги ключевых фраз)
- `enabled`: включить/выключить экстрактор
- `enable_topic_distribution`: включить извлечение тем через retrieval
- `topics_db_spec_name`: имя спецификации topics DB в dp_models
- `model_name`: название embedding модели для retrieval/keyphrases
- `top_k_topics`: количество тем для извлечения
- `top_k_slots`: количество слотов для топ-K тем (стабильная схема)
- `temperature`: температура для softmax при вычислении вероятностей тем
- `enable_keyphrases`: включить извлечение ключевых фраз
- `max_keyphrases`: максимальное количество ключевых фраз для извлечения
- `keyphrase_slots`: количество слотов для ключевых фраз (стабильная схема)
- `max_keyphrase_len_chars`: максимальная длина ключевой фразы в символах
- `export_keyphrases_mode`: режим экспорта ключевых фраз (none|raw|hashed)
- `enable_keyphrase_embeddings`: включить вычисление эмбеддингов ключевых фраз
- `enable_style_flags`: включить стилистические флаги
- `style_instruction_words_ru`: список инструктивных ключевых слов (по умолчанию: ["нажмите", "сделайте", "кликните", "откройте", "выберите"])
- `style_audience_words_ru`: список обращений к аудитории (по умолчанию: ["вы", "ты", "тебя", "вас", "твой", "ваш"])
- `style_cta_words_ru`: список призывов к действию (по умолчанию: ["подпишитесь", "лайк", "комментарий", "ставьте лайк", "подписывайтесь", "переходите"])
- `allow_legacy_transcripts`: разрешить использование legacy транскриптов
- `transcript_source_policy`: политика выбора источника транскрипта (asr_only|asr_then_legacy|legacy_only)
- `max_text_chars`: максимальная длина объединенного текста (cost-control)
- `cache_enabled`: включить кеш для prompt embeddings (ускоряет повторные запуски)
- `cache_ttl_s`: время жизни кеша в секундах
- `cache_max_total_mb`: максимальный размер кеша в MB

### Особенности

- **Множественные источники**: объединение транскрипта, заголовка и описания
- **Ключевые фразы**: извлечение и эмбеддинги топ-10 ключевых фраз
- **Стилистический анализ**: эвристические признаки для определения типа контента
- **Энтропия тем**: измерение разнообразия тем
- **GPU поддержка**: опциональное использование CUDA с fp16 для ускорения
- **Атомарная запись**: использование временных файлов для безопасного сохранения

### Архитектура

1. **Инициализация**: установка устройства и директории артефактов
2. **Подготовка текста**: объединение ASR (preferred) + title + description + лимит `max_text_chars`
3. **Topics retrieval**: retrieval по prompts из `topics.jsonl` (без обучения per-video)
4. **Keyphrases**: deterministic extractor (и optional embeddings)
5. **Style proxies**: дешёвые эвристики (configurable)
6. **Сохранение артефактов**: per-run `.npy` для keyphrase embeddings (если включено)
7. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

- **Пустой текст**: валидный empty (features_flat `tp_topics_present=0`), логируется без PII
- **Topics DB отсутствует**: `RuntimeError` (fail-fast), если `enable_topic_distribution=True`
- **Ошибка сохранения keyphrase embeddings**: embeddings пропускаются (present=0), ошибка логируется

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (embedding inference + retrieval; зависит от размера текста и числа prompts)
- **GPU**: опционально (значительное ускорение при использовании CUDA)
- **Estimated duration**: зависит от размера текста и числа prompts; embeddings для prompts кешируются (cache, не result_store)

**Параметры производительности**:
- Размер текста: линейно влияет на время обработки
- `fp16`: уменьшает использование GPU памяти в 2 раза

### Зависимости

- `numpy`: численные операции
- `torch`: для работы с моделями (если используется GPU)
- `sentence-transformers`: библиотека для эмбеддингов
- `hashlib`: генерация хешей для артефактов

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **model_registry**: реестр моделей для разделения между экстракторами
- **VideoDocument**: схема входного документа
- **text_utils.normalize_whitespace**: нормализация текста
- **path_utils.default_artifacts_dir**: путь к директории артефактов по умолчанию

### Примечания

1. **Topics DB**: расширяйте `topics.jsonl` до 200–500 тем; ids должны быть стабильны.
2. **Энтропия**: высокая энтропия = распределение тем “размазано”, низкая = одна доминирующая тема.
3. **Placeholder метрики**: удалены из prod-выхода; реализуются отдельными компонентами при необходимости.
4. **Размерность эмбеддингов**: зависит от embedding модели (например, E5-large).

### Примеры интерпретации результатов

**Тематические метрики**:
- `transcript_topic_id_top1 = 3`: доминирующая тема #3
- `transcript_topic_probs_vector = [0.6, 0.3, 0.1, ...]`: 60% вероятность темы #3, 30% темы #2, и т.д.
- `topic_entropy = 1.2`: умеренная энтропия (разнообразные темы)
- `topic_entropy = 0.1`: низкая энтропия (концентрированная тема)

**Ключевые фразы**:
- `top_keyphrases_list = ["машинное обучение", "нейронные сети", ...]`: топ-10 ключевых фраз
- `top_keyphrases_with_scores = [("машинное обучение", 0.95), ...]`: фразы с оценками

**Стилистические признаки**:
- `faq_like_question_count = 5`: 5 вопросоподобных предложений
- `instructional_language_flag = True`: присутствует инструктивный язык
- `audience_addressing_flag = True`: есть обращения к аудитории
- `call_to_action_flag = True`: есть призывы к действию

### Порядок выполнения экстракторов

`SemanticTopicExtractor` может выполняться независимо, но рекомендуется:

1. `SemanticTopicExtractor` - извлечение тем и ключевых фраз
2. Компоненты для анализа тем (используют сохранённые эмбеддинги ключевых фраз)
3. Компоненты для стилистического анализа

### Требования к зависимостям

**Обязательные**:
- `numpy`
- `torch` (если используется GPU)
- `sentence-transformers`

