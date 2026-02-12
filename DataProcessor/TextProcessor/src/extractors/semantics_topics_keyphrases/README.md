## `semantics_topics_keyphrases` (Semantic Topics and Keyphrases Extractor)

### Назначение

Извлекает **глобальные (сопоставимые между видео)** темы из текста через **retrieval по фиксированной taxonomy** (bundled `topics.jsonl` + embeddings через `dp_models`), а также ключевые фразы и дешёвые стилистические proxy-флаги.

Важно: component больше **не обучает** темы per-video (BERTopic/KMeans) — это было несопоставимо между видео и плохо для ML/аналитики.

**Версия**: 2.0.0  
**Категория**: topic modeling, keyphrase extraction, style analysis  
**GPU**: поддерживается (cuda), опционально fp16

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

**Topics (retrieval)**:
- `tp_topics_topic_top1_id`
- `tp_topics_topic_top1_score`
- `tp_topics_topic_top1_prob`
- `tp_topics_topic_top{i}_score`, `tp_topics_topic_top{i}_prob` (i=1..top_k_slots; стабильная схема)
- `tp_topics_entropy_topk`
- `tp_topics_entropy_topk_norm`
- `tp_topics_perplexity_topk`

**Keyphrases**:
- `tp_topics_keyphrases_count`
- `tp_topics_keyphrase_score_top1`, `tp_topics_keyphrase_score_mean`
- `tp_topics_keyphrases_dim` (если включены embeddings)

**Keyphrases (privacy-safe export)**:
- `tp_topics_kp_top{i}_present/hash01/len` (i=1..keyphrase_slots) при `export_keyphrases_mode="hashed"`

**Style proxies (heuristics, configurable)**:
- `tp_topics_style_faq_qmarks`
- `tp_topics_style_instructional_flag`
- `tp_topics_style_audience_flag`
- `tp_topics_style_cta_flag`

**Optional raw export (debug / verified-only)**:
- `tp_topics_keyphrases_raw` — включается только через `export_keyphrases_mode="raw"`.

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

Источник тем: `DataProcessor/dp_models/bundled_models/text/topics_v1/topics.jsonl`

Процесс:
1. Загружаем список prompts из `topics.jsonl`.
2. Prompt embeddings строятся один раз и сохраняются в **cache** (`default_cache_dir()/tp_topics_db/*.npy`) — это не `result_store`.
3. Кодируем `full_text` через embedding модель (через `dp_models`).
4. Считаем cosine similarity и агрегируем prompt→topic через `max`.
5. Выдаём top‑K тем + softmax‑probabilities и энтропию.

#### 3. Keyphrases

Текущая реализация: лёгкий deterministic scorer (без внешних зависимостей).

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
    "export_keyphrases_mode": "none",                          # none | raw | hashed
    "enable_keyphrase_embeddings": True,
    "enable_style_flags": True,
    "allow_legacy_transcripts": False,
    "transcript_source_policy": "asr_only",                    # asr_only | asr_then_legacy | legacy_only
    "max_text_chars": 20000
}
```

**Параметры**:
- `device`: устройство обработки (cpu или cuda)
- `artifacts_dir`: директория для сохранения артефактов (эмбеддинги ключевых фраз)

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

