## `title_embedder` (Text embeddings)

### Назначение

Извлекает **L2-нормализованные эмбеддинги** для заголовков видео с использованием моделей sentence transformers. Компонент поддерживает батчинг, кеширование на диск, GPU ускорение и возвращает как нормализованные векторы, так и L2-нормы необработанных векторов.

**Версия**: 1.2.0  
**Категория**: text, embeddings  
**GPU**: опционально (поддерживается CUDA с fp16)

**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_title_embedder_text_npz.py`](utils/validate_title_embedder_text_npz.py)

**Контракт Audit v3 (`features_flat`)**: [`SCHEMA.md`](SCHEMA.md) · machine: [`../../../schemas/title_embedder_output_v1.json`](../../../schemas/title_embedder_output_v1.json) · отчёт: [`../../../docs/audit_v3/components/title_embedder_AUDIT_V3_REPORT.md`](../../../docs/audit_v3/components/title_embedder_AUDIT_V3_REPORT.md) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/title_embedder_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/title_embedder_audit_v4.md) · **L2 stats:** [`../../../../storage/audit_v4/title_embedder_l2/title_embedder_audit_v4_stats.json`](../../../../storage/audit_v4/title_embedder_l2/title_embedder_audit_v4_stats.json) (tooling: `scripts/audit_v4_npz_stats.py`)

**Audit v3 preflight**: каноническая модель эмбеддингов — **`intfloat/multilingual-e5-large`** (задаётся профилем/`model_name` в прогоне; см. `TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md`).

### Входы

- **`doc.title`** (str): заголовок видео

**Важно (production-grade)**:
- `doc.title` optional by default (valid empty), но можно сделать обязательным через `require_title=true` (fail-fast).
- `model_name` должен быть зарегистрирован в `dp_models/spec_catalog/text/*.yaml` и доступен локально под `DP_MODELS_ROOT` (no-network).

### Выходы

Экстрактор возвращает словарь с полями:

#### Основные результаты

Экстрактор сохраняет `*.npy` в per-run `text_processor/_artifacts/`, но **не возвращает пути в result** (privacy/security).
Для NPZ/analytics возвращает только `result.features_flat`:

- `tp_titleemb_present` (0/1)
- `tp_titleemb_dim`
- `tp_titleemb_norm_raw`
- `tp_titleemb_l2_norm`
- `tp_titleemb_title_present` (0/1)
- `tp_titleemb_require_title_enabled` (0/1)
- `tp_titleemb_compute_enabled` (0/1)
- `tp_titleemb_write_artifact_enabled` (0/1)
- `tp_titleemb_artifact_written` (0/1)
- `tp_titleemb_cache_enabled` (0/1)
- `tp_titleemb_cache_hit` (0/1 или NaN если кеш выключен)
- `tp_titleemb_fp16` (0/1)
- `tp_titleemb_device_cuda` (0/1)
- `tp_titleemb_model_digest_u24` (int)
- `tp_titleemb_encode_ms` (float, миллисекунды)
- `tp_titleemb_compute_raw_norm` (0/1)

Для детерминированного доступа downstream‑экстракторами в рамках этого же run используется in-memory реестр (только если `write_artifact=true`):
`doc.tp_artifacts["embeddings"]["title"]["relpath"]` (`title_embedding.npy`).

#### Метаданные

- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `version`: версия экстрактора (`"1.2.0"`)
- `model_name`, `model_version`, `weights_digest`
- `system`: системные метрики (pre_init, post_init, post_process, peaks)
- `timings_s.encode`: время кодирования (секунды)
- `timings_s.total`: общее время обработки (секунды)
- `error`: ошибка (если есть, иначе `None`)

### Алгоритм

1. **Нормализация текста**: удаление лишних пробелов из заголовка
2. **Хеширование**: вычисление SHA256 хеша от `model_name|weights_digest + "||" + normalized_text`
3. **Проверка кеша**: загрузка эмбеддинга и нормы из кеша (если доступны)
4. **Кодирование**: если не в кеше, кодирование через модель sentence transformers
   - Батчинг для обработки нескольких заголовков
   - Получение необработанных векторов (`normalize_embeddings=False`)
   - Вычисление L2-норм необработанных векторов
   - Нормализация векторов (деление на норму)
5. **Сохранение в кеш**: атомарное сохранение нормализованного вектора и нормы
6. **Сохранение артефактов**: сохранение эмбеддинга в .npy файл (atomic tmp→replace)
   - JSON sidecar `.meta.json` **не используется** (запрещён в per-run artifacts; model meta идёт через `models_used`/manifest).

### Конфигурация

```python
TitleEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Модель для эмбеддингов
    cache_dir=None,                                       # Путь к кешу (по умолчанию: default_cache_dir/embed_cache)
    cache_enabled=False,                                  # Включить/выключить дисковый cache (default off)
    cache_ttl_days=30.0,                                  # TTL кеша (None = без TTL)
    cache_max_items=200_000,                              # Лимит количества файлов кеша (None = без лимита)
    cache_max_bytes=2_000_000_000,                        # Лимит размера кеша (None = без лимита)
    cache_cleanup_on_init=True,                           # Best-effort уборка кеша при старте
    cache_cleanup_max_seconds=0.2,                        # Бюджет времени на уборку кеша
    device="cpu",                                         # "cpu" | "cuda"
    fp16=True,                                           # Использовать float16 на GPU
    batch_size=128,                                      # Размер батча для обработки
    artifacts_dir=None,                                   # Путь для сохранения артефактов
    require_title=False,                                   # fail-fast если title пустой
    compute_embedding=True,                                # считать ли эмбеддинг
    write_artifact=True,                                   # писать ли per-run `.npy` артефакт (и регистрировать relpath)
    write_embedding_artifact=True,                         # (deprecated alias)
    compute_raw_norm=True,                                # Считать ли norm_raw (иначе NaN)
    emit_extra_metrics=False,                              # Зарезервировано: в v1.2.0 не меняет features_flat (см. SCHEMA.md)
)
```

**`emit_extra_metrics`:** принимается конструктором / YAML для совместимости API; **`tp_titleemb_encode_ms`** и остальные ключи `features_flat` на успешном пути заполняются **независимо** от этого флага. Дополнительных ключей при `True` не появляется.

### Поддерживаемые модели

Только модели, зарегистрированные в `dp_models/spec_catalog/text/*.yaml` и доступные локально:
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, default)
- (другие — добавляются через `dp_models` как отдельные spec + локальные артефакты)

### Кеширование

Компонент использует двухуровневое кеширование:

1. **В памяти**: модель загружается через `model_registry` и переиспользуется между экземплярами
2. **На диске**: эмбеддинги кешируются по SHA256 хешу текста и модели
**Cache policy (prod)**:
- TTL (`cache_ttl_days`) + лимиты (`cache_max_items`, `cache_max_bytes`)
- Best-effort уборка на старте (`cache_cleanup_on_init`) с бюджетом времени (`cache_cleanup_max_seconds`)
   - Векторы сохраняются в `.npy` файлы
   - Нормы сохраняются отдельно в `.norm.npy` файлы
   - Атомарное сохранение через временные файлы (`.tmp.npy`)

**Структура кеша:**
```
cache_dir/
  ├── {hash}.npy          # Нормализованный эмбеддинг
  └── {hash}.norm.npy     # L2-норма необработанного вектора
```

### Батчинг

При обработке нескольких заголовков через `embed_titles()` или `embed_titles_with_norms()`:
- Заголовки обрабатываются батчами размера `batch_size`
- Кеш проверяется для каждого заголовка индивидуально
- Только отсутствующие в кеше заголовки отправляются в модель

### GPU поддержка

- **Автоматическое определение**: если `device="cuda"` и CUDA доступна, используется GPU
- **FP16 режим**: опциональное использование float16 для экономии памяти (только на GPU)
- **Метрики GPU**: отслеживание использования памяти GPU в системных метриках

### Методы API

#### `embed_titles(titles: List[str], use_cache: bool = True) -> np.ndarray`
Возвращает L2-нормализованные эмбеддинги для списка заголовков.

#### `embed_titles_with_norms(titles: List[str], use_cache: bool = True, return_norms: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]`
Возвращает кортеж `(embeddings, norms)`:
- `embeddings`: нормализованные эмбеддинги, shape `(N, D)`
- `norms`: L2-нормы необработанных векторов, shape `(N,)` (если `return_norms=True`)

#### `extract_batch(docs: List[VideoDocument]) -> List[Dict[str, Any]]`
Батчевая обработка нескольких документов. Оптимизирует кодирование через единый батч, сохраняя per-document артефакты и метрики. Поддерживается через `supports_batch=True`.

### Особенности

- **L2-нормализация**: все эмбеддинги нормализуются для использования косинусного сходства
- **Сохранение норм**: сохраняются L2-нормы необработанных векторов (полезно для анализа)
- **Атомарное сохранение**: кеш сохраняется через временные файлы для предотвращения коррупции
- **Переиспользование моделей**: модели загружаются через общий реестр для экономии памяти
- **Безопасная нормализация**: защита от деления на ноль при нормализации
- **Многоязычность**: поддержка любых языков через соответствующие модели

### Обработка ошибок

- **Ошибки кеша**: игнорируются, выполняется пересчет
- **Ошибки сохранения артефактов**: логируются в поле `error`, но не прерывают выполнение
- **Пустые заголовки**: valid empty по умолчанию; fail-fast при `require_title=true`
- **Отсутствие модели**: ошибка при инициализации, если модель недоступна

### Архитектура

1. **Инициализация**: загрузка модели через `model_registry` (переиспользование между экземплярами)
2. **Хеширование**: вычисление уникального хеша для текста и модели
3. **Проверка кеша**: загрузка вектора и нормы из дискового кеша
4. **Кодирование**: батчевая обработка через модель sentence transformers
5. **Вычисление норм**: L2-нормы необработанных векторов
6. **Нормализация**: деление векторов на их нормы
7. **Кеширование**: атомарное сохранение результатов на диск
8. **Сохранение артефактов**: сохранение эмбеддинга в .npy с метаданными модели

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (зависит от модели, обычно 1-5 секунд на заголовок)
- **GPU**: опционально (значительно быстрее, особенно с fp16)
- **RAM**: зависит от модели (от ~100MB для MiniLM до ~500MB для больших моделей)
- **Disk**: кеш занимает ~4KB на заголовок (float32 вектор)

**Параметры производительности**:
- `batch_size`: большие батчи → быстрее, но больше памяти
- `fp16`: на GPU экономит ~50% памяти и может ускорить обработку
- Кеширование значительно ускоряет повторную обработку

**Estimated duration**:
- Первое кодирование: 0.5-2.0 секунды (зависит от модели и устройства)
- Из кеша: <0.01 секунды

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **VideoDocument**: схема входного документа
- **model_registry**: реестр моделей для переиспользования
- **sentence-transformers**: библиотека для создания эмбеддингов
- **normalize_whitespace**: утилита для нормализации текста

### Зависимости

- `torch`: PyTorch для работы моделей
- `sentence-transformers`: библиотека для эмбеддингов
- `numpy`: работа с массивами
- `transformers`: базовые модели Hugging Face

### Примечания

1. **Размерность эмбеддингов**: зависит от модели (MiniLM-L6-v2: 384, mpnet-base-v2: 768)
2. **L2-нормализация**: все эмбеддинги имеют норму ~1.0 после нормализации
3. **Косинусное сходство**: нормализованные эмбеддинги можно сравнивать через скалярное произведение
4. **Кеш по модели**: разные модели создают разные хеши для одного текста
5. **Атомарность**: кеш сохраняется атомарно для предотвращения коррупции при прерывании
6. **Многоязычность**: для многоязычных данных рекомендуется использовать многоязычные модели
7. **Нормы**: `title_embedding_norm` содержит норму необработанного вектора (до нормализации), что может быть полезно для анализа качества эмбеддинга
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
