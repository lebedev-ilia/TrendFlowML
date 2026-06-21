## `embedding_shift_indicator_extractor` (Semantic Shift Detection)

**Версия**: 1.3.0 · **Контракт Audit v3**: [SCHEMA.md](./SCHEMA.md) · machine: [`schemas/embedding_shift_indicator_extractor_output_v1.json`](../../schemas/embedding_shift_indicator_extractor_output_v1.json)  
**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_embedding_shift_indicator_extractor_text_npz.py`](utils/validate_embedding_shift_indicator_extractor_text_npz.py)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/embedding_shift_indicator_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/embedding_shift_indicator_extractor_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/embedding_shift_indicator_extractor_l2/`

### Назначение

Обнаруживает **семантический сдвиг** в транскрипте видео путём сравнения эмбеддингов начала и конца транскрипта. Вычисляет косинусное сходство между усреднёнными эмбеддингами начальных и конечных чанков и устанавливает флаг сдвига, если сходство ниже порога.

**Категория**: semantic analysis  
**GPU**: не требуется

### Входы

Экстрактор читает матрицу эмбеддингов чанков транскрипта **детерминированно** через `doc.tp_artifacts` (без `glob+mtime`):
- **Canonical**: `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]` (достаточно словаря **`transcripts`**; **`transcript_chunks`** не обязателен)
- **Legacy fallback**: `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]` / `["embeddings_path"]` (ставится `tp_embshift_used_legacy_key_flag=1`)
- приоритет источников задаётся параметром `transcript_source_priority` (default: `whisper → youtube_auto`)

### Выходы

Экстрактор возвращает только `result.features_flat` (privacy-safe, NPZ-friendly), **27** фиксированных ключей **`tp_embshift_*`** (полный список — [SCHEMA.md](./SCHEMA.md)):
- `tp_embshift_present`
- `tp_embshift_disabled_by_policy`, `tp_embshift_enabled`
- `tp_embshift_require_transcript_chunks_enabled`, `tp_embshift_require_min_chunks`
- `tp_embshift_n_chunks`, `tp_embshift_n_window_chunks`, `tp_embshift_dim`
- `tp_embshift_cosine_begin_end` (NaN если данных недостаточно / zero-norm / NaN/Inf)
- `tp_embshift_shift_flag` (NaN если `compute_shift_flag=false` или cosine invalid)
- `tp_embshift_cosine_threshold`
- `tp_embshift_margin` (= cosine_begin_end - cosine_threshold, NaN если cosine invalid)
- (gated) `tp_embshift_cosine_first_last`, `tp_embshift_mean_cosine_last_to_start_window`
- feature flags: `tp_embshift_emit_extra_metrics_enabled`, `tp_embshift_compute_shift_flag_enabled`, `tp_embshift_compute_extra_cosines_enabled`
- источники: `tp_embshift_source_used_whisper`, `tp_embshift_source_used_youtube_auto`, `tp_embshift_used_legacy_key_flag`
- safety flags: `tp_embshift_unsafe_relpath_flag`, `tp_embshift_chunk_embed_missing_flag`, `tp_embshift_dim_mismatch_flag`, `tp_embshift_zero_norm_flag`, `tp_embshift_nan_inf_flag`
- timings: `tp_embshift_load_ms`, `tp_embshift_compute_ms` — **NaN**, если **`emit_extra_metrics=False`**

#### Метаданные

- `device`: устройство обработки (`"cpu"`)
- `version`: версия экстрактора
- `model_name` / `model_version` / `weights_digest`: **`null`**

#### Системные метрики

- `system.pre_init` / `post_init`: снимки после **`__init__`** экстрактора
- `system.post_process`: снимок системы после обработки
- `system.peaks.ram_peak_mb`: пиковое использование RAM (MB)
- `system.peaks.gpu_peak_mb`: пиковое использование GPU памяти (MB, всегда 0)

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `error`: описание ошибки (если произошла) или `None`

### Алгоритм обработки

#### 1. Загрузка эмбеддингов

- Берём relpath из `doc.tp_artifacts` (canonical → legacy fallback) и загружаем `*.npy` из per-run `text_processor/_artifacts/`.

#### 2. Валидация данных

- Если транскрипта нет → valid empty (`tp_embshift_present=0`, NaN метрики), либо fail-fast при `require_transcript_chunks=true`.
- Если `n_chunks < require_min_chunks` → valid empty (не ошибка), либо fail-fast при `require_transcript_chunks=true`.

#### 3. Вычисление окон

- Размер окна: `min(n_window_chunks, max(1, n_chunks // 2))`
- Начальное окно: первые `win` чанков
- Конечное окно: последние `win` чанков

#### 4. Усреднение эмбеддингов

- **Начальное усреднение**: `start_emb = mean(embeddings[:win], axis=0)`
- **Конечное усреднение**: `end_emb = mean(embeddings[-win:], axis=0)`

#### 5. Косинусное сходство

- Вычисление косинусного сходства:
  ```
  cosine = dot(start_emb, end_emb) / (norm(start_emb) * norm(end_emb))
  ```

#### 6. Определение сдвига

- Флаг сдвига: `shift_flag = (cosine < cosine_threshold)`
- Высокое сходство (близко к 1.0) → нет сдвига
- Низкое сходство (далеко от 1.0) → есть сдвиг

### Конфигурация

```python
{
    "n_window_chunks": 2,                                    # Количество чанков в окне для усреднения
    "cosine_threshold": 0.85,                                # Порог косинусного сходства для определения сдвига
    "transcript_source_priority": "whisper,youtube_auto",     # Приоритет источников транскрипта
    "enabled": True,                                         # feature-gating
    "require_transcript_chunks": False,                       # fail-fast при отсутствии/невалидности входа
    "require_min_chunks": 2,                                  # минимальное число чанков для расчёта
    "compute_shift_flag": True,                              # Можно выключить бинарный флаг (оставить только cosine)
    "compute_extra_cosines": False,                           # Доп. метрики (gated)
    "emit_extra_metrics": False                              # Доп. метрики/тайминги
}
```

### Особенности

- **Скользящее окно**: использование нескольких чанков для более стабильного усреднения
- **Адаптивный размер окна**: автоматическая корректировка при малом количестве чанков
- **Приоритет источников**: предпочтение whisper транскриптам перед youtube_auto
- **Пороговая логика**: простой и интерпретируемый критерий сдвига
- **Обработка граничных случаев**: корректная обработка малого количества чанков

### Архитектура

1. Читает relpath из `doc.tp_artifacts` (canonical → legacy fallback)
2. Safe-join и загрузка `*.npy` из per-run `artifacts_dir`
3. Валидация shape + NaN/Inf
4. Формирование start/end window и расчёт cosine
5. (опционально) доп. косинусы
6. Возврат только `features_flat`

### Обработка ошибок

По умолчанию отсутствующие/недостаточные данные → valid empty (NaN + flags), без `error`.  
Fail-fast возможен только при `require_transcript_chunks=true`.

### Performance characteristics

**Resource costs**:
- **CPU**: очень низкие (только векторные операции NumPy)
- **GPU**: не используется
- **Estimated duration**: ~0.001-0.01 секунд

**Параметры производительности**:
- `n_window_chunks`: незначительное влияние на производительность
- Размер матрицы эмбеддингов: линейная сложность по количеству чанков

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **transcript_chunk_embedder**: создаёт `transcript_{source}_embedding_*.npy`
- **default_artifacts_dir**: утилита для определения директории артефактов

### Примечания

1. **Зависимости**: требует выполнения `transcript_chunk_embedder` перед этим экстрактором
2. **Интерпретация**: высокое `cosine_begin_end` (близко к 1.0) означает семантическую консистентность, низкое — сдвиг темы
3. **Порог**: `cosine_threshold=0.85` — эмпирически подобранное значение, может требовать настройки
4. **Окна**: использование нескольких чанков для усреднения уменьшает влияние шума
5. **Граничные случаи**: при `n_chunks < 2` невозможно вычислить сдвиг

### Примеры использования

**Высокое сходство (нет сдвига)**:
```json
{
  "cosine_begin_end": 0.92,
  "shift_flag": false,
  "n_chunks": 10,
  "n_window_chunks": 2
}
```

**Низкое сходство (есть сдвиг)**:
```json
{
  "cosine_begin_end": 0.65,
  "shift_flag": true,
  "n_chunks": 15,
  "n_window_chunks": 2
}
```
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
