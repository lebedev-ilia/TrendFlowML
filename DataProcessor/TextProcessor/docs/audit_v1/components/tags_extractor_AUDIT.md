# Audit: `tags_extractor` (TagsExtractor)

**Дата**: 2026-01-29  
**Статус**: `done`  
**Критерии**: `TextProcessor/docs/audit_v1/TP_AUDIT_CRITERIA.md`

---

## 1) Summary

Компонент приведён к A‑policy:
- **privacy/no‑raw по умолчанию**: raw тексты/списки тегов не попадают в `result` без явного `export_*_mode`
- **in-memory mutations**: чистит `doc.title/doc.description` и (если не отключено политикой) заполняет `doc.hashtags` для downstream extractor’ов
- **stable flat features**: `features_flat` (`tp_tags_*`) для dataset/UI + фиксированные privacy‑safe top‑K слоты
- **Unicode-aware парсер**: `unicode_normalization` (default NFKC), `casefold()`, boundary‑правило, запрет `#_`/`#-` как первого символа
- **hardening**: лимиты `max_text_chars`, `max_tags_total` + `*_truncated_flag`

---

## 2) Контракт входа

- `doc.title` (optional by default):
  - valid empty при отсутствии/пустоте (`tp_tags_title_present=0`)
  - fail-fast при `require_title=true`
- `doc.description` (optional)

---

## 3) Контракт выхода (features_flat)

Префикс: `tp_tags_`

Примеры:
- `tp_tags_hashtag_unique_count`
- `tp_tags_title_hashtag_found_count`, `tp_tags_description_hashtag_found_count`
- `tp_tags_*_density_per_char`
- `tp_tags_*_enabled` flags / mode flags
- privacy-safe fixed top‑K: `tp_tags_top{i}_hash01`, `tp_tags_top{i}_len`, `tp_tags_top{i}_present`

Raw outputs (debug-only, gated):
- `export_cleaned_texts_mode="raw"` → `result.cleaned_texts`
- `export_hashtags_mode="raw"` → `result.hashtags`
- `export_hashtags_mode="hashed"` → `result.hashtags_hashed` (short sha256 prefixes)

---

## 4) Downstream зависимости

Зависимости по данным:
- `HashtagEmbedder` читает `doc.hashtags` → зависит от `mutate_doc_hashtags=true`
- `TitleEmbedder/DescriptionEmbedder` используют `doc.title/doc.description` → если хотим embeddings “без хэштегов”, нужен `mutate_doc_clean_texts=true`

Рекомендуемый порядок: `TagsExtractor` должен выполняться до этих extractor’ов в профиле.

---

## 5) Privacy / Observability

- Raw данные не пишутся в output по умолчанию.
- Мутации применяются через поле `mutations` (не persisted в `result`).

---

## 6) Fixtures

- `src/extractors/tags_extractor/fixtures/doc_tags_basic.json`

---

## 7) Resource costs

- CPU-only, без моделей.
- Время: \(O(n)\) по длине текста (title+description).
- Память: \(O(n)\) (строковые операции).


