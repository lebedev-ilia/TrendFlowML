# `tags_extractor` — описание фич и артефактов

**Компонент:** `TagsExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **28** базовых скаляров + **3×K** слотов `tp_tags_top{i}_*` ( **K = `top_k_slots`**, по умолчанию **5**), префикс `tp_tags_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/tags_extractor_output_v1.json`](../../../../schemas/tags_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · **allow_extra_keys: true** (динамическое **K**). · [`../README.md`](../README.md).

**Опционально в `result` (не в `features_flat`):** `cleaned_texts` / `hashtags` / `hashtags_hashed` при `export_*_mode`.

**Версия:** 1.2.0 (`TagsExtractor.VERSION`).

---

## 1. Базовые поля (стабильные имена)

| Группа | Ключи |
|--------|--------|
| Наличие | `title_present`, `description_present` (0/1) |
| Политика | `group_extract_enabled`, `group_mutate_clean_texts_enabled`, `group_mutate_hashtags_enabled`, `group_merge_json_hashtags_enabled`, `require_title_enabled` |
| | `hashtags_disabled_by_policy` = `not enable_extract` (0/1) |
| Export one-hot | `export_cleaned_texts_mode_{none,raw}` — ровно один **1** |
| | `export_hashtags_mode_{none,raw,hashed}` — ровно один **1** |
| Парсинг/хранение | `title_parse_capped_flag`, `description_parse_capped_flag` (длина поля > `max_parse_chars`) |
| | `title_truncated_flag`, `description_truncated_flag` (после очистки > `max_text_chars`) |
| JSON merge | `json_hashtag_merged_count` — теги, добавленные **только** из `doc.hashtags`, без inline-совпадения |
| Счётчики | `hashtags_truncated_flag` (inline-список > `max_tags_total`) |
| | `title_hashtag_found_count`, `description_hashtag_found_count` — **все** вхождения `#…` (до cap уникалов) |
| | `hashtag_total_found_count` = **сумма** title + desc (**только** inline) |
| | `hashtag_unique_count` = длина итогового списка после merge title→desc→JSON (может **>** inline-only) |
| | `hashtag_avg_len`, `hashtag_max_len` — по уникальным; **NaN** если уникальных **0** |
| Плотности | `title_hashtag_density_per_char` = `title_found / len(title_for_extract)`; **NaN** при нуле длины |
| | `description_hashtag_density_per_char` — аналогично |
| Top-K | `topk_slots` = **K** (float) |

---

## 2. Слоты `tp_tags_top{i}_*` (i = 1..K)

Порядок тегов = merged список (title, затем desc, затем JSON-only).

| Поле | Смысл |
|------|--------|
| `present` | **1** если в слоте есть i-й тег |
| `hash01` | `u64 / 2^64` из **SHA-256** от UTF-8 тега: **[0, 1)**, детерминированно; **NaN** при `present=0` |
| `len` | длина нормализованного тега (символы); **NaN** при `present=0` |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 | флаги presence, group_*, `hashtags_disabled*`, `*_parse_capped*`, `*_truncated*`, `hashtags_truncated*` |
| One-hot | cleaned: **сумма 1**; hashtags: **сумма 1** |
| Счётчики | **≥ 0**; `total_found` = `title` + `desc` |
| | `json_merged` + inline-уникальные в сумме согласованы с `unique` (см. код) |
| `hash01` (finite) | **≥ 0**, **&lt; 1** (теоретически [0,1) у float) |
| `topk_slots` (finite) | **≥ 1** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_tags_extractor_text_npz.py`](../utils/validate_tags_extractor_text_npz.py)

---

## 5. Чеклист

1. Срез `tp_tags_*`: **28** базовых + **3K** top-слотов; `topk_slots` согласован с **K**.  
2. One-hot и **total = title + desc** на успешных прогонах.  
3. Top-слоты: префикс по рангу (заполнены первые **min(K, N)**)
---

## Навигация

[README (root)](../README.md) · [SCHEMA (root)](../SCHEMA.md) · [TextProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
