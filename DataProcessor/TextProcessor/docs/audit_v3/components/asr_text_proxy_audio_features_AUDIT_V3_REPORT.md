# Audit v3 — `asr_text_proxy_audio_features` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `ASRTextProxyExtractor.VERSION = 1.2.0`  
**Machine schema**: `asr_text_proxy_audio_features_output_v1`  
**Human schema**: [`src/extractors/asr_text_proxy_audio_features/SCHEMA.md`](../../../src/extractors/asr_text_proxy_audio_features/SCHEMA.md)

## TL;DR

Экстрактор строит скалярные proxy по тексту ASR из **`doc.asr`** (или legacy **`transcripts_meta`**) с обязательной валидной длительностью (`doc.audio_duration_sec` или деградация на duration из payload с флагом). **Не** читает `doc.transcripts`. Опции **`require_asr_text`** и **`strict_document_duration`** задают fail-fast для строгого preflight. Token-id путь Audit v3 декодируется транзиентно; сбой декода → **`tp_asrproxy_token_decode_failed_flag`**. Добавлено **`tp_asrproxy_speech_rate_wpm_ratio_to_baseline`**.

## Входы / выходы

- Входы: `VideoDocument.audio_duration_sec` (желательно), `doc.asr` / `transcripts_meta`, feature-gating `enabled`, `enable_*`, пороги и лимиты.
- Выход: `result.features_flat` (`tp_asrproxy_*`, 37 ключей), вложенный `result.asr_text_proxy.metrics` — дубликат для отладки.

## Принятые решения

1. Валидный empty по умолчанию при отсутствии текста; строгий транскрипт — opt-in.
2. Длительность из payload допустима как fallback с явным флагом деградации; запрет — через `strict_document_duration`.
3. Сегментный путь имеет приоритет, если есть хотя бы один dict-сегмент; иначе пробуется token-id путь (если поля согласованы по длине).
4. Контракт: **`asr_text_proxy_audio_features_output_v1.json`** + `SCHEMA.md`.

## Acceptance

- [x] Документация и machine schema в реестре; ключи ↔ `features_flat` (37).
- [x] Smoke: `extract()` под `.tp_venv` на минимальном `VideoDocument`; проверены `strict_document_duration`, `require_asr_text`, fallback duration из payload.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/asr_text_proxy_audio_features/main.py`
- `DataProcessor/TextProcessor/schemas/asr_text_proxy_audio_features_output_v1.json`
