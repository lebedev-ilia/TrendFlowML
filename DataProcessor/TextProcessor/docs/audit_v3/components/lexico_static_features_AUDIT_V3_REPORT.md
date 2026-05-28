# Audit v3 — `lexico_static_features` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `LexicalStatsExtractor.VERSION = 1.2.0`  
**Machine schema**: `lexico_static_features_output_v1`  
**Human schema**: [`src/extractors/lexico_static_features/SCHEMA.md`](../../../src/extractors/lexico_static_features/SCHEMA.md)

## TL;DR

Детерминированные лексические фичи без heavy NLP; транскрипт по умолчанию **только ASR** (`asr_only`). Baseline: **`enable_emoji=true`**, **`emoji_policy=optional`**. Добавлен опциональный **`require_transcript`** для строгих прогонов. Прокси-метрики помечены как **analytics** в machine schema и SCHEMA.md. Title/description в стандартном ранне — после **TagsExtractor**.

## Принятые решения (рекомендации исполнителя)

| Тема | Решение |
|------|---------|
| Политика аудита full | Канон: **`asr_only`**, fallback только для dev — в `RUN_LOG` |
| Пустой транскрипт | Default мягкий; **`require_transcript`** для fail-fast |
| `max_*_chars` | Default **`None`** |
| Эмодзи | **`enable_emoji=true`**, **`optional`** |
| Прокси | Документировать как analytics / эвристики |
| Схемы | `SCHEMA.md` + JSON + отчёт |
| Порядок с tags | Зафиксировано в SCHEMA.md |

## Acceptance

- [x] Код, документация, реестр схем, индексы
- [x] Совместимость: `run_cli` на шаблоне smoke (при наличии ASR в документе — transcript present)
- [ ] Полный E2E Segmenter→ASR→Text — по мере общего preflight

## Файлы

- `DataProcessor/TextProcessor/src/extractors/lexico_static_features/main.py`
- `DataProcessor/TextProcessor/schemas/lexico_static_features_output_v1.json`
