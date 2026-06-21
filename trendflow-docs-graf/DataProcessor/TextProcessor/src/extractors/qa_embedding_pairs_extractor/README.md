## `qa_embedding_pairs_extractor` (QAEmbeddingPairsExtractor)

### Назначение

Извлекает **вопросоподобные** фразы из различных источников текста (заголовок, описание, транскрипт, комментарии) и вычисляет их **L2-нормализованные** эмбеддинги (**матрица N×D**). Имя конфига/класса историческое: **пары вопрос–ответ не строятся** — см. [`SCHEMA.md`](SCHEMA.md).

**Версия**: 1.3.0  
**Категория**: question extraction, embeddings  
**GPU**: поддерживается (cuda), опционально fp16

**Контракт (Audit v3)**: [`SCHEMA.md`](SCHEMA.md) · machine: [`schemas/qa_embedding_pairs_extractor_output_v1.json`](../../schemas/qa_embedding_pairs_extractor_output_v1.json)  
**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_qa_embedding_pairs_extractor_text_npz.py`](utils/validate_qa_embedding_pairs_extractor_text_npz.py)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/qa_embedding_pairs_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/qa_embedding_pairs_extractor_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/qa_embedding_pairs_extractor_l2/`

### Входы

- **`VideoDocument`** с полями:
  - `title`: заголовок видео (str)
  - `description`: описание видео (str)
  - `asr`: ASR payload от AudioProcessor (preferred) с `segments[].text`
  - `transcripts`: legacy fallback (только если включён `allow_legacy_transcripts=True`)
  - `comments`: список объектов комментариев с полем `text` (list)

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Основные результаты

Эмбеддинги вопросов сохраняются в per-run sub‑artifact:
- `text_processor/_artifacts/qa_question_embeddings.npy`

Valid empty semantics (A-policy):
- если вопросов не найдено → `tp_qa_present=0`, **артефакты не пишутся**, `doc.tp_artifacts["qa"]` не заполняется.

Возвращаемые скалярные признаки (`result.features_flat`):

**Основные метрики**:
- `tp_qa_present` (0/1): вопросы найдены и эмбеддинги вычислены
- `tp_qa_num_questions`: общее количество найденных вопросов
- `tp_qa_embedding_dim`: размерность эмбеддингов

**Количество вопросов по источникам**:
- `tp_qa_q_title`: количество вопросов из заголовка
- `tp_qa_q_description`: количество вопросов из описания
- `tp_qa_q_transcript`: количество вопросов из транскрипта
- `tp_qa_q_comments`: количество вопросов из комментариев

**Политики и конфигурация**:
- `tp_qa_enabled`: включен ли экстрактор
- `tp_qa_disabled_by_policy`: отключен ли политикой
- `tp_qa_allow_legacy_transcripts`: разрешены ли legacy транскрипты
- `tp_qa_transcript_source_policy_asr_only`: политика asr_only
- `tp_qa_transcript_source_policy_asr_then_legacy`: политика asr_then_legacy
- `tp_qa_transcript_source_policy_legacy_only`: политика legacy_only

**Feature gating**:
- `tp_qa_use_title`: используется ли заголовок
- `tp_qa_use_description`: используется ли описание
- `tp_qa_use_transcript`: используется ли транскрипт
- `tp_qa_use_comments`: используются ли комментарии

**Параметры извлечения**:
- `tp_qa_require_min_questions`: минимальное требуемое количество вопросов
- `tp_qa_max_questions_total`: максимальное общее количество вопросов
- `tp_qa_max_questions_per_source`: максимальное количество вопросов на источник
- `tp_qa_max_comments`: максимальное количество комментариев для обработки
- `tp_qa_max_transcript_chars`: максимальное количество символов транскрипта
- `tp_qa_min_chars_per_question`: минимальная длина вопроса
- `tp_qa_max_question_chars`: максимальная длина вопроса
- `tp_qa_dedup_questions`: включена ли дедупликация вопросов

**Опциональные артефакты**:
- `tp_qa_write_question_hashes_artifact_enabled`: включена ли запись хешей
- `tp_qa_write_question_source_ids_artifact_enabled`: включена ли запись source IDs
- `tp_qa_hashes_written`: были ли записаны хеши
- `tp_qa_source_ids_written`: были ли записаны source IDs

**Порог комментариев (всегда в `features_flat`)**:
- `tp_qa_max_chars_per_comment`: лимит символов на комментарий из конфига

**Дополнительные метрики** (только при `emit_extra_metrics=True`; иначе **NaN** и `tp_qa_mean_cosine_to_centroid_present=0`):
- `tp_qa_questions_per_min`: вопросов в минуту (нужен конечный `audio_duration_sec` > 0; при valid empty и включённом extra → **0.0** или **NaN** по длительности)
- `tp_qa_questions_per_1k_chars`: вопросов на 1000 символов текста вопросов
- `tp_qa_mean_cosine_to_centroid` / `tp_qa_mean_cosine_to_centroid_present`: дисперсия по эмбеддингам (N≥2)

Для детерминированного доступа downstream‑экстракторами в рамках этого же run:
`doc.tp_artifacts["qa"]["question_embeddings"]["relpath"]` (+ `model_version`, `weights_digest`).

Опциональные privacy‑safe sub‑artifacts (включаются флагами):
- `qa_question_hashes.npy` — массив коротких sha256-хэшей (без текста)
- `qa_question_source_ids.npy` — массив int16 source_id, где словарь хранится в `doc.tp_artifacts["qa"]["question_embeddings"]["source_vocab"]`

#### Метаданные

- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `version`: версия экстрактора
- `model_name`: идентификатор модели в `dp_models` (например, `intfloat/multilingual-e5-large`)
- `model_version`: версия/метка артефакта модели из реестра
- `weights_digest`: дайджест весов (как у других embedder’ов Audit v3)

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

#### 1. Извлечение текстов из источников

**Процесс**:
1. **Заголовок**: извлечение и нормализация `doc.title` (если `use_title=True`)
2. **Описание**: извлечение и нормализация `doc.description` (если `use_description=True`)
3. **Транскрипт**: выбор источника по `transcript_source_policy`:
   - `asr_only`: только `doc.asr.segments[].text` (preferred)
   - `legacy_only`: только `doc.transcripts` (whisper + youtube_auto, требует `allow_legacy_transcripts=True`)
   - `asr_then_legacy`: сначала ASR, если пусто — legacy (требует `allow_legacy_transcripts=True`)
   - Применяется truncation до `max_transcript_chars`
4. **Комментарии**: извлечение текста из каждого комментария в `doc.comments` (первые `max_comments`, каждый до `max_chars_per_comment`)

**Нормализация**: применение `normalize_whitespace()` ко всем текстам

#### 2. Сегменты с «?»

- Текст блока нормализуется (`normalize_whitespace`), затем **делится по пробелам после `?` / `？`** (сегменты должны оканчиваться знаком вопроса).
- Для **комментариев**: каждый обрезанный текст комментария — отдельный блок на вход этого шага.

#### 3. Фильтрация вопросов

**Regex паттерн**:
```python
question_words_ru = ["кто","что","где","когда","почему","зачем","как","какой","какая","какие","сколько"]
question_words_en = ["who","what","where","when","why","how","which"]
```

**Процесс**:
1. Из текста извлекаются сегменты, оканчивающиеся `?` или `？` (знак сохраняется).
2. Сегменты обрезаются до `max_question_chars` (с сохранением знака вопроса).
3. Фильтруются по минимальной длине `min_chars_per_question`.
4. Сегменты фильтруются по списку question words (RU/EN) — конфигурируется через `question_langs` и `question_words_ru`/`question_words_en`.
5. Применяются лимиты: `max_questions_per_source` на источник, `max_questions_total` общий.
6. (Опционально) дедупликация по canonical форме вопроса (casefold, нормализация пробелов).
7. Отслеживается источник каждого вопроса (title/description/transcript/comments).

#### 4. Вычисление эмбеддингов

**Процесс**:
1. **Батчинг**: обработка вопросов батчами размером `batch_size`
2. **Кодирование**: использование sentence-transformers модели для кодирования
3. **Нормализация**: L2-нормализация каждого эмбеддинга
4. **Объединение**: объединение всех батчей в единую матрицу (N, D)

**Формула нормализации**:
```
emb_norm = emb / ||emb||
```

#### 5. Сохранение артефактов

**Файлы**:
- `qa_question_embeddings.npy`: матрица эмбеддингов (N, D) в формате float32
- optional:
  - `qa_question_hashes.npy` (privacy-safe)
  - `qa_question_source_ids.npy` (privacy-safe)

**Важно**:
- JSON sidecar в `text_processor/_artifacts/` не создаётся (per-run JSON запрещён).
- Исходные тексты вопросов не сохраняются.

### Конфигурация

```python
{
    "model_name": "intfloat/multilingual-e5-large",          # dp_models / Audit v3 default
    "artifacts_dir": None,                                    # Путь к артефактам (по умолчанию: default_artifacts_dir())
    "device": "cpu",                                          # "cpu" | "cuda"
    "fp16": True,                                             # Использовать float16 на GPU
    "batch_size": 64,                                         # Размер батча для обработки
    "enabled": True,                                          # feature-gating
    "allow_legacy_transcripts": False,                        # Разрешить legacy doc.transcripts
    "transcript_source_policy": "asr_only",                   # asr_only | asr_then_legacy | legacy_only

    # feature gating (UI/config)
    "use_title": True,
    "use_description": True,
    "use_transcript": True,
    "use_comments": True,

    # question policy
    "question_langs": "ru,en",
    "question_words_ru": None,                                # default list inside component
    "question_words_en": None,
    "min_chars_per_question": 8,
    "max_question_chars": 240,
    "dedup_questions": True,

    # cost control
    "max_questions_total": 128,
    "max_questions_per_source": 64,
    "max_comments": 200,
    "max_chars_per_comment": 280,
    "max_transcript_chars": 200000,
    "require_min_questions": 0,

    # privacy-safe sub-artifacts
    "write_question_hashes_artifact": False,
    "write_question_source_ids_artifact": False,

    # extra metrics (gated)
    "emit_extra_metrics": False,  # включает tp_qa_questions_per_min, tp_qa_questions_per_1k_chars, tp_qa_mean_cosine_to_centroid
}
```

**Параметры**:
- `model_name`: название модели sentence-transformers
- `artifacts_dir`: директория для сохранения артефактов
- `device`: устройство обработки (cpu или cuda)
- `fp16`: использование float16 на GPU (уменьшает память, минимальная потеря точности)
- `batch_size`: размер батча для обработки вопросов

### Особенности

- **Множественные источники**: извлечение вопросов из заголовка, описания, транскрипта и комментариев
- **Regex фильтрация**: автоматическое определение вопросов по паттерну
- **Privacy-safe**: исходные тексты вопросов не сохраняются, только хеши
- **Батчинг**: эффективная обработка множества вопросов
- **L2 нормализация**: все эмбеддинги нормализованы для использования в косинусной метрике
- **GPU поддержка**: опциональное использование CUDA с fp16 для ускорения
- **Атомарная запись**: использование временных файлов для безопасного сохранения
- **Метрики**: детальные системные метрики и тайминги

### Архитектура

1. **Инициализация**: загрузка модели через `get_model()` из registry
2. **Извлечение текстов**: сбор текстов из всех источников
3. **Разбиение на предложения**: разбиение текстов на предложения
4. **Фильтрация вопросов**: применение regex для поиска вопросов
5. **Кодирование**: батчевая обработка вопросов через модель
6. **Нормализация**: L2-нормализация каждого эмбеддинга
7. **Сохранение**: сохранение матрицы эмбеддингов и метаданных
8. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

- **Пустые источники**: если источник отсутствует или отключен через feature-gating, он пропускается
- **Ошибка обработки комментария**: ошибка обрабатывается через try-except, комментарий пропускается
- **Пустой список вопросов**: valid empty (`tp_qa_present=0`), артефакты не создаются, `doc.tp_artifacts["qa"]` не заполняется
- **Недостаточно вопросов**: если `require_min_questions > 0` и `num_questions < require_min_questions` → RuntimeError (fail-fast)
- **Ошибка сохранения артефакта**: RuntimeError (если компонент включён — должен отдать результат)

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (sentence-transformers операции)
- **GPU**: опционально (значительное ускорение при использовании CUDA)
- **Estimated duration**: ~0.1-1.0 секунд в зависимости от количества вопросов

**Параметры производительности**:
- `batch_size`: большие значения → быстрее на GPU, но больше памяти
- Количество вопросов: линейно влияет на время обработки
- `fp16`: уменьшает использование GPU памяти в 2 раза, минимальная потеря точности

### Зависимости

- `numpy`: численные операции (нормализация, работа с массивами)
- `torch`: для работы с моделями (если используется GPU)
- `sentence-transformers`: библиотека для эмбеддингов
- `re`: регулярные выражения для фильтрации вопросов
- `hashlib`: генерация хешей для privacy-safe хранения

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **model_registry**: реестр моделей для разделения между экстракторами
- **VideoDocument**: схема входного документа
- **text_utils.normalize_whitespace**: нормализация текста
- **path_utils.default_artifacts_dir**: путь к директории артефактов по умолчанию

### Примечания

1. **Размерность**: зависит от модели (Audit v3 default **`intfloat/multilingual-e5-large`** → **1024**)
2. **Нормализация**: все эмбеддинги L2-нормализованы (норма ≈ 1.0)
3. **Privacy**: исходные тексты вопросов не сохраняются, только хеши для идентификации
4. **Модели**: модели разделяются через registry, не загружаются повторно
5. **Regex**: вопросительные слова по **`question_langs`** (по умолчанию RU + EN списки в коде)
6. **Источники**: вопросы извлекаются из всех доступных источников независимо
7. **Пустые результаты**: если вопросов не найдено, возвращается пустая матрица без ошибки

### Примеры использования

**Вопросы в заголовке**:
- "Как сделать это?" → извлекается
- "Это интересно." → не извлекается (нет вопросительного слова)

**Вопросы в комментариях**:
- "Где можно купить?" → извлекается
- "Отличное видео!" → не извлекается

**Вопросы в транскрипте**:
- "Почему это происходит?" → извлекается
- "Это происходит потому что..." → не извлекается (нет знака вопроса)

### Порядок выполнения экстракторов

`QAEmbeddingPairsExtractor` может выполняться независимо, но для полного анализа рекомендуется:

1. `QAEmbeddingPairsExtractor` - извлечение вопросов и их эмбеддингов
2. Компоненты для поиска похожих вопросов (используют сохранённые эмбеддинги)
3. Компоненты для анализа FAQ-контента
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
