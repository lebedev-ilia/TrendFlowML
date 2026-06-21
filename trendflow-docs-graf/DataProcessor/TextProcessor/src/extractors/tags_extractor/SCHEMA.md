# `tags_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `tags_extractor` |
| Класс | `TagsExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/tags_extractor_output_v1.json` |
| `schema_version` (логический контракт выхода) | `tags_extractor_output_v1` |
| Версия реализации | `1.2.0` (см. `TagsExtractor.VERSION`) |

Артефакт: вклад в агрегированный `text_features.npz` (`text_npz_v1`) через плоские скаляры `tp_tags_*` в `feature_names` / `feature_values`. Отдельного NPZ у экстрактора нет.

## Назначение

Извлечь хэштеги из **`doc.title`** и **`doc.description`**, опционально **объединить** с `doc.hashtags` из входного JSON, нормализовать и при политике мутаций записать очищенные строки и итоговый список тегов в документ для downstream.

## Область текста (очистка `#...`)

- **В scope**: `title`, `description` (in-memory мутации при `mutate_doc_clean_texts` / логика извлечения при `enable_extract_hashtags`).
- **Вне scope** (намеренно): `comments[].text`, `video_description_by_neuro`, `trend_words`, транскрипт — UGC и контракты эмбеддеров не должны молча терять `#` в этих полях.

## Порядок обработки (качество / детерминизм)

1. Unicode + whitespace-нормализация полного поля (`unicode_normalization`, по умолчанию NFKC).
2. Скан `#tag` на префиксе длины **`min(len(field), max_parse_chars)`** (по умолчанию `max_parse_chars=200_000`, не ниже `max_text_chars`), чтобы теги в хвосте длинного текста не терялись из-за лимита хранения.
3. Удаление токенов `#<tag>` и формирование очищенной строки.
4. Усечение очищенной строки до **`max_text_chars`** перед записью в `doc` → флаги `tp_tags_*_truncated_flag` (storage).
5. Превышение длины нормализованного поля над `max_parse_chars` → `tp_tags_*_parse_capped_flag` (хвост не сканировался).

## Hashtags: inline + JSON

- Сначала уникальные теги из title, затем из description (**inline**, `casefold`, порядок первого появления).
- Если **`merge_json_hashtags=true`**, к списку добавляются записи из входного **`doc.hashtags`**, которых ещё нет (нормализация: trim, снятие одного ведущего `#`, `casefold` всей строки тега).
- Итог при **`mutate_doc_hashtags`**: `doc.hashtags = merged` (не сырой JSON), если включено извлечение из текста **или** осознанно включён только merge JSON при непустом списке из входа.

## Мутации и ошибки

При включённых мутациях сбой присвоения `title` / `description` / `hashtags` или обновлении `doc.tp_artifacts["tags"]` → **fail-fast** с логированием (`logging.exception`) и `RuntimeError` (не проглатывать).

## Выход `extract()`

Структура как у других экстракторов: `device`, `version`, `system`, `timings_s`, `result`, `mutations`, `error`.

### `result.features_flat` (все ключи, `float32`-скаляры в агрегате)

**Presence / политики**

| Ключ | Смысл |
|------|--------|
| `tp_tags_title_present` | Заголовок непустой после нормализации |
| `tp_tags_description_present` | Описание непустое |
| `tp_tags_group_extract_enabled` | `enable_extract_hashtags` |
| `tp_tags_group_mutate_clean_texts_enabled` | `mutate_doc_clean_texts` |
| `tp_tags_group_mutate_hashtags_enabled` | `mutate_doc_hashtags` |
| `tp_tags_group_merge_json_hashtags_enabled` | `merge_json_hashtags` |
| `tp_tags_require_title_enabled` | `require_title` |
| `tp_tags_hashtags_disabled_by_policy` | `not enable_extract_hashtags` |
| `tp_tags_export_cleaned_texts_mode_*` | one-hot режима export title/desc |
| `tp_tags_export_hashtags_mode_*` | one-hot режима export тегов |

**Парсинг / усечение**

| Ключ | Смысл |
|------|--------|
| `tp_tags_title_parse_capped_flag` | Title длиннее `max_parse_chars` до сканирования |
| `tp_tags_description_parse_capped_flag` | Description длиннее `max_parse_chars` |
| `tp_tags_title_truncated_flag` | Очищенный title усечён до `max_text_chars` |
| `tp_tags_description_truncated_flag` | Очищенное описание усечено |
| `tp_tags_json_hashtag_merged_count` | Число тегов, добавленных только из JSON (не было в inline) |

**Счётчики / плотность**

| Ключ | Смысл |
|------|--------|
| `tp_tags_hashtags_truncated_flag` | Превышен `max_tags_total` при извлечении |
| `tp_tags_title_hashtag_found_count` | Все вхождения `#tag` в title (до уникализации) |
| `tp_tags_description_hashtag_found_count` | Аналогично для description |
| `tp_tags_hashtag_total_found_count` | Сумма найденных |
| `tp_tags_hashtag_unique_count` | Уникальные после merge inline+JSON |
| `tp_tags_hashtag_avg_len` | Средняя длина уникального тега |
| `tp_tags_hashtag_max_len` | Макс. длина |
| `tp_tags_title_hashtag_density_per_char` | found / len(parse window title) |
| `tp_tags_description_hashtag_density_per_char` | found / len(parse window desc) |
| `tp_tags_topk_slots` | Параметр `top_k_slots` |

**Top-K (privacy-safe)**

Для `i = 1 .. top_k_slots`:

- `tp_tags_top{i}_present`
- `tp_tags_top{i}_hash01` — детерминированный [0,1) из SHA256 тега
- `tp_tags_top{i}_len`

Пустые слоты: `present=0`, `hash01`/`len` = NaN.

### Опционально в `result` (не в `features_flat`)

- `cleaned_texts` — только при `export_cleaned_texts_mode=raw`
- `hashtags` — при `export_hashtags_mode=raw`
- `hashtags_hashed` — при `export_hashtags_mode=hashed`

### `mutations`

Словарь для отладки оркестратора: при успешной мутации могут присутствовать ключи `cleaned_texts`, `hashtags` (сырые строки списка — только внутри процесса, не для NPZ по умолчанию).

## Downstream

- **`TitleEmbedder` / `DescriptionEmbedder`**: опираются на очищенные `doc.title` / `doc.description`, если в ранне есть `TagsExtractor` с `mutate_doc_clean_texts` (дублирующая очистка в embedder не требуется).
- **`HashtagEmbedder`**: читает `doc.hashtags` после мутаций.

## Версионирование

Изменение набора ключей или смысла флагов → bump **`tags_extractor_output_v2`** + запись в `RUN_LOG.md` и отчёт компонента.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
