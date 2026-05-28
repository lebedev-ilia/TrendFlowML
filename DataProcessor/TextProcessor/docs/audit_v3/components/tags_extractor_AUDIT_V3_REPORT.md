# Audit v3 — `tags_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `TagsExtractor.VERSION = 1.2.0`  
**Machine schema**: `tags_extractor_output_v1`  
**Human schema**: [`src/extractors/tags_extractor/SCHEMA.md`](../../../src/extractors/tags_extractor/SCHEMA.md)

## TL;DR

Экстрактор приводит хэштеги из title/description к детерминированному списку, **мерджит** платформенный `doc.hashtags` из JSON, муциирует документ для downstream; парсинг выполняется на окне до `max_parse_chars`, запись в doc усечена `max_text_chars`. Мутации и `tp_artifacts` — **fail-fast** с логом. Комментарии и прочие поля **не** очищаются от `#`.

## Входы / выходы

- Входы: `VideoDocument.title`, `description`, опционально `hashtags` (список строк из JSON).
- Выход: `result.features_flat` (`tp_tags_*`), опциональные raw/hashed блоки по флагам export; `mutations` для отладки.

## Принятые решения (владелец + исполнение)

1. Допускаются разные комбинации флагов аудита; контракт ключей фиксирует **one-hot** на активные режимы.
2. **Merge JSON + inline**: порядок тегов — сначала уникальные inline (title, затем desc), затем теги из JSON без дубликата по `casefold`.
3. Ошибки мутаций: **fail-fast** + `logging.exception`.
4. Scope очистки: только **title/description**; UGC-комментарии без изменения.
5. Схема: **`tags_extractor_output_v1.json`** + `SCHEMA.md`.
6. Порядок парсинга/лимитов: `max_parse_chars` (default 200k, ≥ `max_text_chars`) затем storage-truncation cleaned текста в `max_text_chars`.
7. Downstream: **единый источник правды** — мутации TagsExtractor; embedder’ы не дублируют strip `#`.

## Acceptance

- [x] Документация и machine schema в реестре.
- [x] Smoke: fixture + ручной вызов `extract()` под `.tp_venv`.
- [ ] Полный E2E с ASR в preflight-профиле — вне этого PR; зафиксировать в `RUN_LOG.md` при прогоне.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/tags_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/tags_extractor_output_v1.json`
