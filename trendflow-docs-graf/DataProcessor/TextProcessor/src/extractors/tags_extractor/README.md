## `tags_extractor` (TagsExtractor)

### Назначение

Извлекает **хэштеги** из `doc.title`/`doc.description`, опционально **объединяет** с `doc.hashtags` из входного JSON, удаляет токены `#<tag>` из title/description и (опционально) делает **in-memory мутации** документа для downstream extractor’ов.

**Версия**: 1.2.0  
**Категория**: text  
**GPU**: не требуется  

**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_tags_extractor_text_npz.py`](utils/validate_tags_extractor_text_npz.py)

**Контракт**: [`SCHEMA.md`](SCHEMA.md) · machine: [`../../schemas/tags_extractor_output_v1.json`](../../schemas/tags_extractor_output_v1.json) (`allow_extra_keys: true` — `tp_tags_top{i}_*`, **i=1..top_k_slots`) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/tags_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/tags_extractor_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/tags_extractor_l2/`

### Контракт входа

- **`doc.title`** (optional by default):
  - если пустой/отсутствует → валидная пустота (`tp_tags_title_present=0`)
  - если `require_title=true` → fail-fast (`RuntimeError`)
- **`doc.description`** (optional)
- **`doc.hashtags`** (optional list): при `merge_json_hashtags=true` теги без дубликата к inline добавляются в конец (нормализация: trim, снятие ведущего `#`, `casefold` всей строки)

### Контракт выхода

- **Основной слой (NPZ/dataset/UI)**: `result.features_flat` (только числовые скаляры `tp_tags_*`)
- **In-memory propagation для downstream**: `mutations` (не персистится оркестратором в `result`)
- **Raw/debug output**: только при явном включении `export_*_mode`

### Unicode правила для хэштегов

- Unicode normalisation (default): `unicode_normalization="NFKC"` (можно `NONE|NFKC|NFC|NFKD|NFD`)
- Допустимые символы тега:
  - первый символ: только `L/M/N`
  - последующие: `L/M/N` + `_` + `-`
- Boundary правило: `#` должен быть в начале строки или после **не‑hashtag** символа (не матчим `abc#tag`)
- Нормализация тега: `casefold()`

### Область очистки `#...`

- **Обрабатываются**: только `title` и `description`.
- **Не изменяются**: `comments`, транскрипт, `video_description_by_neuro`, `trend_words` (сохранение семантики UGC / других контрактов).

### Порядок лимитов

1. Скан хэштегов на префиксе поля длиной до **`max_parse_chars`** (default `200_000`, не ниже `max_text_chars`).
2. Очищенные строки **`max_text_chars`** (default `5000`) перед записью в `doc` → `tp_tags_title_truncated_flag` / `tp_tags_description_truncated_flag`.

### Конфигурация (prod)

- `enable_extract_hashtags` (bool, default `true`)
- `require_title` (bool, default `false`)
- `mutate_doc_clean_texts` (bool, default `true`)
- `mutate_doc_hashtags` (bool, default `true`)
- `merge_json_hashtags` (bool, default `true`) — merge платформенного списка с inline
- `unicode_normalization` (str, default `"NFKC"`)
- `max_text_chars` (int, default `5000`) — лимит **сохранённых** очищенных строк
- `max_parse_chars` (int, default `200_000`) — лимит **сканирования** `#tag` на поле
- `max_tags_total` (int, default `64`) → `tp_tags_hashtags_truncated_flag`
- `top_k_slots` (int, default `5`) → фиксированные top‑K слоты (privacy-safe)
- `export_cleaned_texts_mode` (str, default `"none"`): `none|raw`
- `export_hashtags_mode` (str, default `"none"`): `none|raw|hashed`
- (deprecated) `export_cleaned_texts` / `export_hashtags`: если `True` → соответствующий `*_mode="raw"`
- `max_tag_len` (int, default `64`)

### Ошибки мутаций

Сбой записи `doc.title` / `doc.description` / `doc.hashtags` или `doc.tp_artifacts["tags"]` → **`RuntimeError`** и запись стека в лог (`logging.exception`).

### `features_flat` (ключевые фичи)

См. полный перечень в [`SCHEMA.md`](SCHEMA.md). Кратко:

- Политики + export one-hot, `tp_tags_group_merge_json_hashtags_enabled`
- `tp_tags_title_parse_capped_flag`, `tp_tags_description_parse_capped_flag`
- `tp_tags_title_truncated_flag`, `tp_tags_description_truncated_flag` (storage)
- `tp_tags_json_hashtag_merged_count`
- Счётчики, плотности, top‑K hash/len слоты

### Downstream зависимости

- `HashtagEmbedder` читает `doc.hashtags` после мутаций (inline + merge JSON при политике выше).
- `TitleEmbedder` / `DescriptionEmbedder`: очищенные `doc.title` / `doc.description` при `mutate_doc_clean_texts=true`; дублирующая очистка в embedder **не требуется**.

**In-memory маркеры** (через `doc.tp_artifacts`, не персистится):

- `doc.tp_artifacts["tags"]["hashtags_disabled_by_policy"]` (float, 0/1)

Рекомендуемый порядок: `TagsExtractor` выполняется **до** зависимых экстракторов (см. `MainProcessor` / `component_graph.yaml`).

### Performance characteristics / resource costs

- CPU-only, без моделей
- Время: \(O(n)\) по длине текста (срез до `max_parse_chars`)
- Память: \(O(n)\) (строковые операции)
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
