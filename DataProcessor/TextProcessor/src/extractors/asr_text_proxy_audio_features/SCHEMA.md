# `asr_text_proxy_audio_features` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `asr_text_proxy_audio_features` |
| Класс | `ASRTextProxyExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/asr_text_proxy_audio_features_output_v1.json` |
| `schema_version` (логический контракт выхода) | `asr_text_proxy_audio_features_output_v1` |
| Версия реализации | `1.2.0` (см. `ASRTextProxyExtractor.VERSION`) |

Артефакт: вклад в агрегированный `text_features.npz` (`text_npz_v1`) через плоские скаляры `tp_asrproxy_*` в `feature_names` / `feature_values`. Отдельного NPZ у экстрактора нет.

## Назначение

Вычислить **audio-like proxy** по **тексту ASR** (качество/шумность/ритм/интонационные эвристики) **без** анализа волны. Это не WER и не акустические фичи.

## Источники текста и политика документа

- **Канон входа**: `doc.asr` (структурированный payload от AudioProcessor).
- **Legacy**: `doc.transcripts_meta` (временный alias; будет удалён после аудита AudioProcessor).
- Экстрактор **не** читает `doc.transcripts` и **не** мутирует текстовые поля документа. Порядок тегов в DAG **не** меняет ASR input для этого компонента.
- **`require_asr_text=false` (default)**: пустой текст после join сегментов — валидный empty (`tp_asrproxy_present=0`, метрики `NaN` где уместно).
- **`require_asr_text=true`**: пустой транскрипт — **fail-fast** (`RuntimeError`).

## Длительность

- По умолчанию: `VideoDocument.audio_duration_sec`, иначе при наличии — **fallback** на поле duration из `doc.asr` / `transcripts_meta` → **`tp_asrproxy_duration_from_payload_flag=1`** (деградированный режим относительно полного контракта Segmenter→doc).
- **`strict_document_duration=true`**: если `audio_duration_sec` отсутствует — **fail-fast**, даже если duration есть в payload.

## Audit v3 token-only path

Если в `doc.asr` нет непустого списка сегментов с dict, но есть согласованные **`token_ids_by_segment`** (или `token_ids`) + **`segment_start_sec`** / **`segment_end_sec`**, экстрактор **транзиентно** декодирует id в текст через **`shared_tokenizer_v1`** (`dp_models` + `tokenizers`). Текст **не** персистится в артефактах. При исключении на этом пути: **`tp_asrproxy_token_decode_failed_flag=1`**, сегменты пустые, дальше — как empty transcript (subject to `require_asr_text`).

## Выход `extract()`

Структура как у других экстракторов: `device`, `version`, `system`, `timings_s`, `result`, `error`.

### `result.features_flat`

Source of truth по ключам: `main.py` → `_stable_template()` + заполнение; machine JSON — тот же набор (37 ключей в v1.2.0).

**Новые / характерные ключи v1.2.0**

| Ключ | Смысл |
|------|--------|
| `tp_asrproxy_require_asr_text_enabled` | Отражает `require_asr_text` |
| `tp_asrproxy_strict_document_duration_enabled` | Отражает `strict_document_duration` |
| `tp_asrproxy_token_decode_failed_flag` | Ошибка token-decode path |
| `tp_asrproxy_speech_rate_wpm_ratio_to_baseline` | `speech_rate_wpm / words_per_minute_baseline` |

Полный перечень полей см. `asr_text_proxy_audio_features_output_v1.json`.

## Downstream

- **`LexicalStatsExtractor`**: отдельный лексический контур; транскрипт там может следовать политике `transcript_source_policy`; ASR proxy здесь **только** из `doc.asr` / legacy meta, без `doc.transcripts`.
- Ритм-метрики нормируются на duration и baseline WPM из конфига.

## Версионирование

Изменение набора ключей или смысла флагов → bump **`asr_text_proxy_audio_features_output_v2`** + запись в `RUN_LOG.md` и отчёт компонента.
