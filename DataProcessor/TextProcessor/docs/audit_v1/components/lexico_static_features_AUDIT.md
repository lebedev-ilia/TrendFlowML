# Audit: `lexico_static_features` (LexicalStatsExtractor)

**Дата**: 2026-01-29  
**Статус**: `done`  
**Критерии**: `TextProcessor/docs/audit_v1/TP_AUDIT_CRITERIA.md`

---

## 1) Summary

Компонент приведён к production‑политике TextProcessor:
- добавлен `enabled` gating и стабильный `features_flat` для UI/dataset даже при отключении компонента
- transcript берётся из AudioProcessor (`VideoDocument.asr.segments[].text`) по явной политике `transcript_source_policy` (`asr_only|asr_then_legacy|legacy_only`)
- пустой transcript — **валидный empty**: transcript‑фичи становятся `NaN`, при этом есть masks/presence флаги
- добавлены cost-control лимиты `max_*_chars` + `*_truncated_flag`
- **удалены** `spacy/langdetect` из компонента (NLP‑модели должны быть отдельными extractor’ами через `dp_models`)
- добавлен **feature‑gating по группам** и стабильные **flat features** (`tp_lex_*`) для dataset/UI
- убраны `print`‑логирования из runtime кода (no noisy stdout)
- добавлен placeholder `resource_costs` JSON (без выдуманных метрик)

---

## 2) Контракт входа

`VideoDocument`:
- `title` (optional)
- `description` (optional)
- `asr` (optional, preferred transcript source)
- `transcripts` (legacy, optional; запрещён по умолчанию)

---

## 3) Контракт выхода (features_flat)

Префикс: `tp_lex_`

Ключевые группы:
- presence/masks: `tp_lex_present_*`
- group enable flags: `tp_lex_group_*_enabled`
- title/description/transcript/combined scalars (см. `README.md`)

Требование: `features_flat` содержит только числовые скаляры (float/bool casted to float), пригодные для NPZ/dataset.

---

## 4) Feature gating и зависимости

Параметры конструктора:
- `enable_title`, `enable_description`, `enable_transcript`
- `enable_emoji` (если true и `emoji` не установлен → error)
- `enable_clickbait_heuristic`
- `allow_legacy_transcripts` (legacy fallback)

Зависимости групп:
- clickbait → title
- transcript features → `asr.segments` (или legacy transcripts, если разрешено)
- emoji → наличие `emoji` пакета

---

## 5) Privacy / no-network

- не пишет raw текст в артефакты/логи
- не скачивает модели
- `spacy/langdetect` вынесены в отдельные компоненты (если понадобятся)

---

## 6) Quality validation (sanity)

Рекомендуемые проверки:
- ratio‑фичи в \([0..1]\), если не NaN
- если `tp_lex_present_transcript=0` → transcript‑фичи NaN

Fixtures (без PII):
- `src/extractors/lexico_static_features/fixtures/doc_basic_no_asr.json`
- `.../doc_with_asr_segments.json`

---

## 7) Открытые задачи для закрытия аудита

Нет.


