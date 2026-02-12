## `title_to_hashtag_cosine_extractor` (Similarity metric)

### Назначение

Вычисляет **косинусное сходство** между эмбеддингом заголовка и эмбеддингом хэштегов видео. Компонент читает relpath эмбеддингов из `doc.tp_artifacts` (без `glob+mtime`) и вычисляет cosine similarity.

**Версия**: 1.1.0  
**Категория**: similarity metric  
**GPU**: не требуется

### Входы

- **`doc`** (Any): документ (используется только для структуры, не требует специфических полей)

**Зависимости**:
- `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (создаёт `title_embedder`)
- `doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]` (создаёт `hashtag_embedder`)

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Скалярные признаки (`result.features_flat`)

- **Canonical (new)**: `tp_titlehashcos_*` (стабильная схема, ключи всегда присутствуют)
  - `tp_titlehashcos_present` (0/1)
  - `tp_titlehashcos_cosine` (float, \([-1, 1]\) или NaN)
  - `tp_titlehashcos_title_present`, `tp_titlehashcos_hashtag_present`
  - `tp_titlehashcos_dim_mismatch_flag`, `tp_titlehashcos_zero_norm_flag`, `tp_titlehashcos_unsafe_relpath_flag`
  - `tp_titlehashcos_enabled`, `tp_titlehashcos_disabled_by_policy`
  - `tp_titlehashcos_require_title_embedding_enabled`, `tp_titlehashcos_require_hashtag_embedding_enabled`

- **Legacy aliases (back-compat)**:
  - `tp_title_hashtag_cosine_present`
  - `tp_title_hashtag_cosine`

#### Метаданные (общие для всех экстракторов)

- `device`: устройство обработки (всегда `"cpu"`)
- `version`: версия экстрактора (`"1.0.0"`)
- `system`: системные метрики (pre_init, post_init, post_process, peaks)
- `timings_s.total`: общее время обработки (секунды)
`error` не используется для “optional missing input” (valid empty). Fail-fast возможен только при `require_*`.

### Алгоритм

1. **Поиск эмбеддингов**: детерминированный выбор relpath из `doc.tp_artifacts` (без `glob+mtime`)
2. **L2-нормализация**: нормализация обоих векторов к единичной длине
3. **Вычисление cosine similarity**: скалярное произведение нормализованных векторов
4. **Возврат результата**: возврат значения схожести в диапазоне [-1, 1]

Формула:
```
cosine_similarity = dot(normalize(title_vec), normalize(hashtag_vec))
```

### Поиск эмбеддингов

Экстрактор использует:
- `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
- `doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]`
- **Формат**: 1D numpy array, автоматически приводится к `[-1]` (flatten)

Безопасность:
- relpath резолвится через safe-join (защита от path traversal), при подозрительных значениях → `tp_titlehashcos_unsafe_relpath_flag=1` и valid empty (если не require).

### Конфигурация

```python
{
    "artifacts_dir": None,                 # per-run artifacts dir (по умолчанию: default_artifacts_dir())
    "enabled": True,                       # feature-gating
    "require_title_embedding": False,      # fail-fast если нет title embedding
    "require_hashtag_embedding": False     # fail-fast если нет hashtag embedding
}
```

### Особенности

- **L2-нормализация**: автоматическая нормализация векторов для корректного вычисления cosine similarity
- **Простота**: минималистичный компонент для одной метрики
- **Метрики производительности**: отслеживание времени выполнения и использования памяти

### Обработка ошибок

- **Отсутствие входов**: valid empty (NaN + `*_present=0`), fail-fast при `require_*`
- **Несоответствие размерностей**: `tp_titlehashcos_dim_mismatch_flag=1`, valid empty
- **Нулевые нормы**: `tp_titlehashcos_zero_norm_flag=1`, valid empty (no fake metrics)

### Архитектура

1. Читает relpath из `doc.tp_artifacts`
2. Safe-join и `np.load(...)`, reshape(-1)
3. Валидация размерности и zero-norm
4. L2-нормализация и dot-product
5. Возврат только `features_flat`

### Performance characteristics

**Resource costs**:
- **CPU**: минимальные (только загрузка файлов и одно скалярное произведение)
- **RAM**: минимальные (два вектора эмбеддингов в памяти)
- **Estimated duration**: <0.001 секунды для типичных эмбеддингов

**Параметры производительности**:
- Размерность эмбеддинга: не влияет значительно (операция векторизована)
- Размер файлов: минимальное влияние (загрузка из диска)

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **title_embedder**: компонент, создающий эмбеддинг заголовка (зависимость)
- **hashtag_embedder**: компонент, создающий эмбеддинг хэштегов (зависимость)
- **system_snapshot, process_memory_bytes**: утилиты для сбора метрик

### Примечания

1. **Зависимости**: требует предварительного создания эмбеддингов заголовка и хэштегов
2. **Размерность эмбеддингов**: должна совпадать между заголовком и хэштегами (обычно определяется моделью)
3. **Выбор последних эмбеддингов**: если создано несколько эмбеддингов, используется самый свежий для каждого типа
4. **Диапазон значений**: cosine similarity в диапазоне [-1, 1], где:
   - `1.0`: идентичные векторы (максимальное сходство)
   - `0.0`: ортогональные векторы (нет сходства)
   - `-1.0`: противоположные векторы (максимальное различие)
5. **Нормализация**: L2-нормализация гарантирует, что результат является именно cosine similarity
6. **Использование**: метрика может использоваться для анализа согласованности заголовка и хэштегов видео
7. **Производительность**: компонент очень быстрый, так как выполняет только одну операцию скалярного произведения

