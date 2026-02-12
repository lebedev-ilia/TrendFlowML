## `tags_extractor` (TagsExtractor)

### Назначение

Извлекает **хэштеги** из `doc.title`/`doc.description`, удаляет токены `#<tag>` из текста и (опционально) делает **in-memory мутации** документа для downstream extractor’ов.

**Версия**: 1.1.0  
**Категория**: text  
**GPU**: не требуется

### Контракт входа

- **`doc.title`** (optional by default):
  - если пустой/отсутствует → валидная пустота (`tp_tags_title_present=0`)
  - если `require_title=true` → fail-fast (`RuntimeError`)
- **`doc.description`** (optional)

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

### Конфигурация (prod)

- `enable_extract_hashtags` (bool, default `true`)
- `require_title` (bool, default `false`)
- `mutate_doc_clean_texts` (bool, default `true`)
- `mutate_doc_hashtags` (bool, default `true`): применяется **только** если `enable_extract_hashtags=true` (иначе `tp_tags_hashtags_disabled_by_policy=1`)
- `unicode_normalization` (str, default `"NFKC"`)
- `max_text_chars` (int, default `5000`) → `tp_tags_*_truncated_flag`
- `max_tags_total` (int, default `64`) → `tp_tags_hashtags_truncated_flag`
- `top_k_slots` (int, default `5`) → фиксированные top‑K слоты (privacy-safe)
- `export_cleaned_texts_mode` (str, default `"none"`): `none|raw`
- `export_hashtags_mode` (str, default `"none"`): `none|raw|hashed`
- (deprecated) `export_cleaned_texts` / `export_hashtags`: если `True` → соответствующий `*_mode="raw"`
- `max_tag_len` (int, default `64`)

### `features_flat` (ключевые фичи)

- Presence/политики:
  - `tp_tags_title_present`, `tp_tags_description_present`
  - `tp_tags_group_extract_enabled`, `tp_tags_group_mutate_*_enabled`
  - `tp_tags_require_title_enabled`, `tp_tags_hashtags_disabled_by_policy`
  - `tp_tags_export_*_mode_*`
- Counts/densities:
  - `tp_tags_title_hashtag_found_count`, `tp_tags_description_hashtag_found_count`, `tp_tags_hashtag_total_found_count`
  - `tp_tags_hashtag_unique_count`
  - `tp_tags_title_hashtag_density_per_char`, `tp_tags_description_hashtag_density_per_char`
  - `tp_tags_hashtag_avg_len`, `tp_tags_hashtag_max_len`
- Safety:
  - `tp_tags_title_truncated_flag`, `tp_tags_description_truncated_flag`, `tp_tags_hashtags_truncated_flag`
- Privacy-safe fixed top‑K:
  - `tp_tags_topk_slots`
  - `tp_tags_top{i}_present`, `tp_tags_top{i}_hash01`, `tp_tags_top{i}_len` для `i=1..top_k_slots`

### Downstream зависимости

- `HashtagEmbedder` читает `doc.hashtags` → зависит от `mutate_doc_hashtags=true` и `enable_extract_hashtags=true`
- `TitleEmbedder/DescriptionEmbedder` используют `doc.title/doc.description` → если хотим embeddings “без хэштегов”, включаем `mutate_doc_clean_texts=true`

Рекомендуемый порядок: `TagsExtractor` должен выполняться **до** downstream extractor’ов.

### Performance characteristics / resource costs

- CPU-only, без моделей
- Время: \(O(n)\) по длине текста
- Память: \(O(n)\) (строковые операции)
