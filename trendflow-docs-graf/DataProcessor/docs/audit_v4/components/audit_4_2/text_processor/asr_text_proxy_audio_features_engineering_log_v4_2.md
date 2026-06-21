# Audit v4.2 — engineering log: `asr_text_proxy_audio_features`

**Дата:** 2026-04-14  
**Компонент:** `asr_text_proxy_audio_features` (TextProcessor; табличный срез `tp_asrproxy_*` в `text_processor/text_features.npz`)

## Цель

Подготовить контур Audit v4 **L2** (несколько прогонов A+B) для среза `tp_asrproxy_*` и зафиксировать качество `result_store` на текущем B-наборе.

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/asr_text_proxy_audio_features/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/asr_text_proxy_audio_features_l2/asr_text_proxy_audio_features_audit_v4_stats.json`
- Опции: `--seed 0`; явные `--npz` при необходимости; графики строятся только при **≥3** OK-прогонах с полным срезом.

## Наблюдения по `result_store` (youtube)

Скан показал: **2** файла `text_processor/text_features.npz` с `meta.status=ok`, **3** с `meta.status=error`. На error-файлах **`feature_names` пустой** — экстрактор ASR-proxy не получил шанса записать строку (пайплайн упал раньше).

Типичная причина на B (пример `meta.error`): не удалось загрузить **TitleEmbedder** (`SentenceTransformer` / **CUDA OOM** на `intfloat/multilingual-e5-large`). Это **не** баг `asr_text_proxy_audio_features`, но **блокирует** эмпирический L2 по объединённому `text_features.npz` на этих run.

## Что в JSON

- `dataset_quality`: счётчики OK vs error, пояснение.
- `per_file`: для каждого пути — `tabular_proxy` (только `tp_asrproxy_*`), `meta_flat`, усечённый `text_processor_error` при ошибке.
- `aggregate`: по всем 5 строкам матрицы (много NaN из-за пустых прогонов).
- `aggregate_ok_subset`: перцентили **только по OK** строкам (сейчас **n_rows=2**).
- `correlation_tabular.low_sample_warning`: true при `n_ok < 3`.

## Следующие шаги (разблокировка L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` с устойчивой конфигурацией эмбеддеров (CPU, меньшая модель, или достаточный GPU).
2. Повторить скрипт статистики; ожидается **5** строк с `proxy_slice_ok=true`, затем осмысленные корреляции и heatmap.
3. При необходимости — golden §4.8 по фиксированному reference NPZ (A).
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
