## `embedding_pair_topk_extractor` (Similarity Search)

### Назначение

Вычисляет **топ-K наиболее похожих чанков транскрипта** для заголовка видео на основе косинусного сходства эмбеддингов. Также вычисляет косинусное сходство между заголовком и описанием. Cross-encoder в конструкторе **запрещён** (политика приватности / отсутствие сырых текстов чанков).

**Версия**: 1.3.0  
**Категория**: similarity search  
**GPU**: опционально по полю **`device`** (на практике FAISS/numpy чаще CPU; поле передаётся пайплайном)

**Контракт (Audit v3)**: [`SCHEMA.md`](SCHEMA.md) · machine: [`schemas/embedding_pair_topk_extractor_output_v1.json`](../../schemas/embedding_pair_topk_extractor_output_v1.json)  
**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_embedding_pair_topk_extractor_text_npz.py`](utils/validate_embedding_pair_topk_extractor_text_npz.py)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/embedding_pair_topk_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/embedding_pair_topk_extractor_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/embedding_pair_topk_extractor_l2/`

### Входы

Экстрактор читает артефакты **детерминированно** через `doc.tp_artifacts` (без `glob+mtime`):

- `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (от `title_embedder`)
- `doc.tp_artifacts["embeddings"]["description"]["relpath"]` (от `description_embedder`)
- **Canonical**: `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]` (от `transcript_chunk_embedder`)
- **Legacy fallback**: `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]` (ставится `tp_embpair_used_legacy_key_flag=1`)
  - Приоритет источников задаётся параметром `transcript_source_priority` (default: `whisper → youtube_auto`)

### Выходы

Экстрактор возвращает только `result.features_flat` (NPZ-friendly скаляры) со **стабильной схемой** Audit v3: **69** ключей (`allow_extra_keys: false` в machine JSON). **Ровно 8** слотов `top1..top8`; **`top_k_slots`** из конфига **клампится** до 8 (**`tp_embpair_top_k_slots_requested`** / **`tp_embpair_top_k_slots_clamped`**).

Ключевые фичи (новый неймспейс `tp_embpair_*`):

**Основные метрики**:
- `tp_embpair_present` (0/1) — присутствует ли хотя бы одна вычисленная метрика
- `tp_embpair_title_desc_cosine` — косинусное сходство заголовок↔описание
- `tp_embpair_title_transcript_topk_max` — максимум среди топ-K сходств (если `export_topk_summary=true`)
- `tp_embpair_title_transcript_topk_mean` — среднее среди топ-K сходств (если `export_topk_summary=true`)

**Флаги присутствия**:
- `tp_embpair_title_present` (0/1) — присутствует ли эмбеддинг заголовка
- `tp_embpair_desc_present` (0/1) — присутствует ли эмбеддинг описания
- `tp_embpair_transcript_chunks_present` (0/1) — присутствует ли матрица эмбеддингов чанков транскрипта
- `tp_embpair_title_desc_present` (0/1) — успешно ли вычислено сходство заголовок↔описание
- `tp_embpair_title_transcript_topk_present` (0/1) — успешно ли вычислен топ-K поиск

**Топ-K слоты** (всегда **8** ключей `top1..top8`; заполнение — если `export_topk_slots=true` / `export_topk_indices=true`):
- `tp_embpair_title_transcript_top{i}` — сходства (NaN если слот не экспортирован или нет данных)
- `tp_embpair_title_transcript_top{i}_idx` — индексы чанков (privacy-safe)

**Feature-gating флаги**:
- `tp_embpair_enabled` (0/1) — включен ли экстрактор
- `tp_embpair_disabled_by_policy` (0/1) — отключён ли экстрактор (`enabled=false`)
- `tp_embpair_compute_title_desc_enabled` (0/1) — включено ли вычисление заголовок↔описание
- `tp_embpair_compute_title_transcript_topk_enabled` (0/1) — включен ли топ-K поиск
- `tp_embpair_export_topk_slots_enabled` (0/1) — включен ли экспорт слотов
- `tp_embpair_export_topk_indices_enabled` (0/1) — включен ли экспорт индексов
- `tp_embpair_export_topk_summary_enabled` (0/1) — включен ли экспорт сводки (max/mean)

**Диагностические флаги**:
- `tp_embpair_dim_mismatch_flag` (0/1) — несовпадение размерностей эмбеддингов
- `tp_embpair_unsafe_relpath_flag` (0/1) — небезопасный relpath (path traversal)
- `tp_embpair_nan_inf_flag` (0/1) — обнаружены NaN/Inf в эмбеддингах
- `tp_embpair_zero_norm_flag` (0/1) — обнаружены вырожденные векторы (норма ~0)
- `tp_embpair_used_legacy_key_flag` (0/1) — использовался ли legacy ключ для транскрипта

**Конфигурационные параметры**:
- `tp_embpair_top_k` — размер поиска по чанкам (K для retrieval)
- `tp_embpair_top_k_slots` — эффективное число слотов экспорта (после клампа ≤ **`tp_embpair_schema_slots_max`** = 8)
- `tp_embpair_top_k_slots_requested` — значение из конфига до клампа
- `tp_embpair_schema_slots_max` — **8** (жёсткий потолок схемы Audit v3)
- `tp_embpair_use_faiss_mode_auto` (0/1) — режим FAISS: auto
- `tp_embpair_use_faiss_mode_never` (0/1) — режим FAISS: never
- `tp_embpair_use_faiss_mode_always` (0/1) — режим FAISS: always
- `tp_embpair_min_corpus_for_faiss` — минимальный размер корпуса для использования FAISS (в auto режиме)
- `tp_embpair_require_faiss_enabled` (0/1) — требуется ли FAISS
- `tp_embpair_require_title_embedding_enabled` (0/1) — требуется ли эмбеддинг заголовка
- `tp_embpair_require_description_embedding_enabled` (0/1) — требуется ли эмбеддинг описания
- `tp_embpair_require_transcript_chunks_enabled` (0/1) — требуется ли матрица чанков транскрипта

**Дополнительные метрики** (ключи **всегда** в `features_flat`; при **`emit_extra_metrics=false`** — **NaN**):
- `tp_embpair_n_chunks` — число чанков (строк матрицы)
- `tp_embpair_transcript_source_whisper` / `youtube_auto` / **`combined`** — 0/1 при включённых extra (иначе **NaN**)
- `tp_embpair_use_faiss_mode` — скаляр режима (0=never, 0.5=auto, 1=always), не путать с `tp_embpair_use_faiss_mode_auto` и т.д.
- `tp_embpair_require_faiss` — дублирует политику require_faiss при включённых extra

**Back-compat алиасы** (`tp_pairtopk_*`):
- `tp_pairtopk_present` — **legacy**: только **`tp_embpair_title_transcript_topk_present`**, не общий `tp_embpair_present`
- `tp_pairtopk_top_k` — алиас для `tp_embpair_top_k`
- `tp_pairtopk_title_desc_cosine` — алиас для `tp_embpair_title_desc_cosine`
- `tp_pairtopk_title_transcript_topk_max` — алиас для `tp_embpair_title_transcript_topk_max`
- `tp_pairtopk_title_transcript_topk_mean` — алиас для `tp_embpair_title_transcript_topk_mean`
- `tp_pairtopk_title_transcript_top{1..8}` — алиасы для слотов топ‑K

#### Метаданные

- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `version`: версия экстрактора
- `model_name`, `model_version`, `weights_digest`: **`null`** (эмбеддинги считаются в upstream-экстракторах)

#### Системные метрики

- `system.pre_init`: снимок системы до инициализации
- `system.post_init`: снимок системы после инициализации
- `system.post_process`: снимок системы после обработки
- `system.peaks.ram_peak_mb`: пиковое использование RAM (MB)
- `system.peaks.gpu_peak_mb`: пиковое использование GPU памяти (MB, обычно 0)

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `error`: описание ошибки (если произошла) или `None`

### Алгоритм обработки

#### 1. Загрузка эмбеддингов

- Берём relpath из `doc.tp_artifacts` и загружаем `*.npy` из per-run `text_processor/_artifacts/`.
- **Защита от path traversal**: используется `_safe_join_artifacts_dir()` для проверки, что relpath не выходит за пределы `artifacts_dir`. При обнаружении небезопасного пути устанавливается `tp_embpair_unsafe_relpath_flag=1` и загрузка не выполняется.

#### 2. Косинусное сходство заголовок-описание

- Если доступны оба эмбеддинга, вычисляется:
  ```
  cosine = dot(title, description) / (norm(title) * norm(description))
  ```

#### 3. Топ-K поиск (заголовок → транскрипт)

- **FAISS** (если доступен): эффективный поиск через `IndexFlatIP` с L2-нормализацией
- **Fallback**: прямое вычисление косинусной матрицы через NumPy
- Возвращаются топ-K индексов и соответствующих скоров

#### 4. Cross-encoder reranking (опционально)

**Примечание (prod)**: `use_cross_encoder` запрещён по умолчанию.
Rerank требует raw chunk texts + отдельного privacy-gating и `dp_models` спецификации для cross-encoder.
Сейчас при `use_cross_encoder=true` extractor делает fail-fast с понятной ошибкой.

### Конфигурация

```python
{
    "artifacts_dir": None,                                    # Путь к артефактам (по умолчанию: default_artifacts_dir())
    "enabled": True,                                          # feature-gating: включен ли экстрактор
    "top_k": 10,                                              # Кол-во кандидатов для topK поиска
    "top_k_slots": 5,                                         # Стабильное число slots в features_flat (top1..topKSlots)
    "transcript_source_priority": "whisper,youtube_auto",   # Приоритет источников транскрипта (list/CSV)
    "compute_title_desc": True,                               # Вычислять ли косинусное сходство заголовок↔описание
    "compute_title_transcript_topk": True,                    # Выполнять ли топ-K поиск заголовок→транскрипт
    "export_topk_slots": True,                                # Экспортировать ли слоты топ-K (top1..topKSlots)
    "export_topk_indices": True,                              # Экспортировать ли индексы топ-чанков (privacy-safe, без текста)
    "export_topk_summary": True,                              # Экспортировать ли сводку (max/mean) топ-K
    "use_faiss_mode": "auto",                                 # auto | never | always
    "min_corpus_for_faiss": 512,                              # Порог для auto режима (минимум чанков для использования FAISS)
    "require_faiss": False,                                   # Требовать ли FAISS (fail-fast если недоступен)
    "use_cross_encoder": False,                               # Запрещено по умолчанию (privacy + dp_models, fail-fast)
    "temperature": 0.1,                                       # Температура для softmax в cross-encoder (не используется)
    "device": "cpu",                                          # "cpu" | "cuda" (для cross-encoder, не используется)
    "require_title_embedding": False,                         # fail-fast если нет title embedding
    "require_description_embedding": False,                  # fail-fast если нужен title-desc, но нет description embedding
    "require_transcript_chunks": False,                       # fail-fast если нужен title→transcript topk, но нет chunks
    "emit_extra_metrics": False                               # Включать ли дополнительные метрики (n_chunks, source, etc.)
}
```

**Параметры**:
- `artifacts_dir`: директория для поиска файлов эмбеддингов
- `enabled`: feature-gating для всего экстрактора
- `top_k`: количество кандидатов для топ-K поиска
- `top_k_slots`: стабильное количество слотов в `features_flat` (всегда присутствуют, даже если не заполнены)
- `transcript_source_priority`: приоритет источников транскрипта (по умолчанию: `"whisper,youtube_auto"`)
- `compute_*`: feature-gating для отдельных вычислений
- `export_*`: feature-gating для экспорта различных метрик
- `use_faiss_mode`: режим использования FAISS:
  - `"auto"`: использовать FAISS если корпус >= `min_corpus_for_faiss` (default)
  - `"never"`: всегда использовать NumPy fallback
  - `"always"`: всегда пытаться использовать FAISS (fallback на NumPy при ошибке)
- `min_corpus_for_faiss`: минимальный размер корпуса (количество чанков) для использования FAISS в auto режиме
- `require_faiss`: fail-fast если FAISS требуется, но недоступен
- `use_cross_encoder`: **запрещено по умолчанию** (fail-fast с понятной ошибкой, требует raw chunk texts + dp_models spec)
- `require_*`: fail-fast политики для обязательных входов
- `emit_extra_metrics`: включать ли дополнительные метрики (размер корпуса, источник транскрипта, etc.)

### Особенности

- **Эффективный поиск**: использование FAISS для быстрого поиска в больших корпусах
- **Fallback на NumPy**: если FAISS недоступен, используется прямое вычисление
- **Приоритет источников**: предпочтение whisper транскриптам перед youtube_auto
- **Cross-encoder**: опциональный reranking для улучшения качества (отключено по умолчанию)
- **Температура**: настраиваемая температура для softmax в cross-encoder
- **Обработка NaN**: автоматическая санитизация logits и вероятностей

### Архитектура

1. Читает relpath из `doc.tp_artifacts` (canonical → legacy fallback)
2. Safe-join и загрузка `.npy` из per-run `artifacts_dir`
3. Sanity: NaN/Inf + zero-norm + dim mismatch → NaN + flags (no fake metrics)
4. Косинус title↔description (gated)
5. Top‑K title→transcript chunks (FAISS/NumPy в зависимости от `use_faiss_mode` и `min_corpus_for_faiss`)
6. Возврат только `features_flat` (privacy-safe)

### Обработка ошибок

- **Отсутствующие артефакты**: valid empty (NaN + `*_present=0`) по умолчанию; fail-fast при `require_*`
- **Ошибка загрузки**: valid empty (без PII)
- **Path traversal**: небезопасный relpath обнаруживается через `_safe_join_artifacts_dir()`, устанавливается `tp_embpair_unsafe_relpath_flag=1`, загрузка не выполняется
- **FAISS недоступен**: автоматический fallback на NumPy (fail-fast только если `require_faiss=True`)
- **Cross-encoder**: запрещён (fail-fast), т.к. требует raw chunk texts + dp_models spec + privacy gating
- **Несоответствие размерностей**: устанавливается `tp_embpair_dim_mismatch_flag=1`, соответствующие метрики становятся NaN
- **Вырожденные векторы (норма ~0)**: устанавливается `tp_embpair_zero_norm_flag=1`, соответствующие метрики становятся NaN
- **NaN/Inf в эмбеддингах**: устанавливается `tp_embpair_nan_inf_flag=1`, соответствующие метрики становятся NaN

### Performance characteristics

**Resource costs**:
- **CPU**: низкие-умеренные (векторные операции)
- **GPU**: опционально (только для cross-encoder)
- **Estimated duration**: ~0.01-0.1 секунд для типичного поиска

**Параметры производительности**:
- `top_k`: большие значения → больше вычислений, но обычно незначительно
- **FAISS**: значительно быстрее для больших корпусов (>1000 чанков)
- **Cross-encoder**: медленнее, но точнее (если включен)

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **title_embedder**: создаёт `title_embedding_*.npy`
- **description_embedder**: создаёт `description_embedding_*.npy`
- **transcript_chunk_embedder**: создаёт `transcript_{source}_embedding_*.npy`
- **FAISS**: библиотека для эффективного поиска (опционально)
- **sentence-transformers CrossEncoder**: для reranking (опционально)

### Примечания

1. **Зависимости**: требует выполнения предыдущих экстракторов (title_embedder, description_embedder, transcript_chunk_embedder)
2. **Приватность**: cross-encoder отключен по умолчанию, так как тексты чанков не сохраняются
3. **FAISS**: рекомендуется для больших корпусов, но не обязателен
4. **Нормализация**: эмбеддинги должны быть L2-нормализованы для корректной работы FAISS
5. **Топ-K**: если чанков меньше чем `top_k`, возвращаются все доступные
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
