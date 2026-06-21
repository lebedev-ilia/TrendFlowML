## `title_to_hashtag_cosine_extractor` (Similarity metric)

### Назначение

**Cosine similarity** между эмбеддингами **title** и **hashtag** по **`doc.tp_artifacts`** (детерминированные `relpath`, без glob/mtime). Векторы **L2-нормируются** внутри экстрактора; затем скалярное произведение.

**Версия**: 1.2.0  
**Категория**: similarity metric  
**GPU**: не требуется

**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_title_to_hashtag_cosine_extractor_text_npz.py`](utils/validate_title_to_hashtag_cosine_extractor_text_npz.py)

**Контракт Audit v3**: [SCHEMA.md](./SCHEMA.md) · machine: [`schemas/title_to_hashtag_cosine_extractor_output_v1.json`](../../schemas/title_to_hashtag_cosine_extractor_output_v1.json) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/title_to_hashtag_cosine_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/title_to_hashtag_cosine_extractor_audit_v4.md) · **L2 stats:** [`../../../../storage/audit_v4/title_to_hashtag_cosine_extractor_l2/title_to_hashtag_cosine_extractor_audit_v4_stats.json`](../../../../storage/audit_v4/title_to_hashtag_cosine_extractor_l2/title_to_hashtag_cosine_extractor_audit_v4_stats.json) (tooling: `scripts/audit_v4_npz_stats.py`)

### Входы

- `doc.tp_artifacts["embeddings"]["title"]["relpath"]` ← **TitleEmbedder**
- `doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]` ← **HashtagEmbedder**

### Выходы

`result.features_flat` — **ровно 11** ключей `tp_titlehashcos_*` (порядок фиксирован, `allow_extra_keys: false` в machine schema).

Кратко:

- `tp_titlehashcos_present`, `tp_titlehashcos_cosine`
- `tp_titlehashcos_require_*_enabled` (зеркала opt-in fail-fast)
- `tp_titlehashcos_title_present`, `tp_titlehashcos_hashtag_present`
- `tp_titlehashcos_unsafe_relpath_flag` — только **path traversal** при join
- `tp_titlehashcos_title_embed_missing_flag`, `tp_titlehashcos_hashtag_embed_missing_flag` — безопасный путь, но файл отсутствует / не читается / пустой вектор
- `tp_titlehashcos_dim_mismatch_flag`, `tp_titlehashcos_zero_norm_flag`

См. [SCHEMA.md](./SCHEMA.md).

### Верхний уровень ответа

`model_name` / `model_version` / `weights_digest`: **`null`**. `system`: снимки из **`__init__`** и **`post_process`**, **`gpu_peak_mb`**.

### Конфигурация

```python
TitleToHashtagCosineExtractor(
    artifacts_dir=None,
    require_title_embedding=False,
    require_hashtag_embedding=False,
)
```

Параметр **`enabled`** в **`__init__`** отсутствует: включение/выключение экстрактора — на уровне **конфига прогона** (список extractors). Неизвестные ключи в kwargs (в т.ч. устаревший `enabled`) **игнорируются**.

### Алгоритм

1. Прочитать `relpath` из `tp_artifacts`
2. Безопасный join → при исключении: **`unsafe_relpath_flag`**
3. `np.load`, `float32` flatten; отсутствие файла / ошибка чтения: **`_*_embed_missing_flag`**
4. Сверка размерности, нормы → cosine

### Архитектура

Один проход: загрузка двух векторов, валидация, нормализация, dot-product. **Нет** выбора «самого свежего» файла — только явные `relpath` в `tp_artifacts`.

### Связанные компоненты

- **TitleEmbedder**, **HashtagEmbedder**, **TagsExtractor** (хэштеги в документе до hashtag embedder)
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
