## `embedding_pair_topk_extractor` (Similarity Search)

### Назначение

Вычисляет **топ-K наиболее похожих чанков транскрипта** для заголовка видео на основе косинусного сходства эмбеддингов. Также вычисляет косинусное сходство между заголовком и описанием. Опционально поддерживает reranking через cross-encoder (отключено по умолчанию из соображений приватности).

**Версия**: 1.2.0  
**Категория**: similarity search  
**GPU**: не требуется (опционально для cross-encoder)

### Входы

Экстрактор читает артефакты **детерминированно** через `doc.tp_artifacts` (без `glob+mtime`):

- `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (от `title_embedder`)
- `doc.tp_artifacts["embeddings"]["description"]["relpath"]` (от `description_embedder`)
- **Canonical**: `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]` (от `transcript_chunk_embedder`)
- **Legacy fallback**: `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]` (ставится `tp_embpair_used_legacy_key_flag=1`)
  - Приоритет источников задаётся параметром `transcript_source_priority` (default: `whisper → youtube_auto`)

### Выходы

Экстрактор возвращает только `result.features_flat` (NPZ-friendly скаляры) со **стабильной схемой** `tp_embpair_*` (ключи всегда присутствуют; при empty → NaN + flags).

Ключевые фичи (новый неймспейс):
- `tp_embpair_present`
- `tp_embpair_title_desc_present`
- `tp_embpair_title_transcript_topk_present`
- `tp_embpair_title_desc_cosine`
- `tp_embpair_title_transcript_topk_max`
- `tp_embpair_title_transcript_topk_mean`
- `tp_embpair_title_transcript_top{1..top_k_slots}` (если `export_topk_slots=true`)
- `tp_embpair_title_transcript_top{1..top_k_slots}_idx` (если `export_topk_indices=true`, privacy-safe индексы чанков)

Back-compat алиасы: `tp_pairtopk_*`.

#### Метаданные

- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `version`: версия экстрактора
- `model_version`: название cross-encoder модели (если используется) или `None`

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
    "enabled": True,                                          # feature-gating
    "top_k": 10,                                              # Кол-во кандидатов для topK поиска
    "top_k_slots": 5,                                        # Стабильное число slots в features_flat (top1..topKSlots)
    "transcript_source_priority": "whisper,youtube_auto",      # Приоритет источников транскрипта
    "compute_title_desc": True,
    "compute_title_transcript_topk": True,
    "export_topk_slots": True,
    "export_topk_indices": True,                               # экспорт индексов top-чанков (без текста)
    "export_topk_summary": True,
    "use_faiss_mode": "auto",                                  # auto | never | always
    "min_corpus_for_faiss": 512,                               # порог для auto (минимум чанков для FAISS)
    "require_faiss": False,
    "use_cross_encoder": False,                               # запрещено по умолчанию (privacy + dp_models)
    "temperature": 0.1,                                       # Температура для softmax в cross-encoder
    "device": "cpu",                                          # "cpu" | "cuda"
    "require_title_embedding": False,                          # fail-fast если нет title embedding
    "require_description_embedding": False,                    # fail-fast если нужен title-desc
    "require_transcript_chunks": False,                        # fail-fast если нужен title->transcript topk
    "emit_extra_metrics": False
}
```

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
- **FAISS недоступен**: автоматический fallback на NumPy
- **Cross-encoder**: запрещён (fail-fast), т.к. требует raw chunk texts + dp_models spec + privacy gating

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









