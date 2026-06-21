# Оркестратор AudioProcessor — телеметрия ресурсов и профилирование (каркас)

Назначение: единый **контур наблюдаемости** вокруг `run_extractors` и **batch** (`MainProcessor.run_batch` / `process_video`): на каждый экстрактор фиксируются wall-time, снимки RAM (и при доступности — GPU через PyTorch), дельты между «до/после». Данные отдаются **внешним оркестраторам** через `scheduler_runtime_report.json` без изменения контрактов NPZ экстракторов.

## Включение

| Переменная окружения | Значение | Эффект |
|---------------------|----------|--------|
| `AP_ORCHESTRATOR_TELEMETRY` | `1` / `true` / `on` | Сбор событий по каждому успешно стартовавшему экстрактору |
| `AP_ORCHESTRATOR_TELEMETRY_CHILDREN` | `1` | Дополнительно суммировать RSS дочерних процессов (`psutil`, может быть дороже) |
| `AP_ORCHESTRATOR_TELEMETRY_LOG` | `1` | Каждое событие — строка `INFO` в лог (JSON) |

По умолчанию телеметрия **выключена** (нулевой оверхед на проверках env).

## Где пишется результат

При завершении прогона `run_cli.py` в `finally` формируется:

`{run_rs_path}/_reports/scheduler_runtime_report.json`

Если `AP_ORCHESTRATOR_TELEMETRY=1`, в корень объекта добавляется ключ:

```json
"orchestrator_telemetry": {
  "schema_version": "orchestrator_telemetry_v1",
  "host": "...",
  "include_children": false,
  "events": [
    {
      "extractor_key": "asr",
      "wall_ms": 12345.0,
      "success": true,
      "snap_before": { "rss_mb": ..., "gpu_allocated_mb": ... },
      "snap_after": { ... },
      "delta": { "rss_mb": ..., "gpu_allocated_mb": ... },
      "context": { "batch_kind": "gpu_segments", "n_files": 3, "family": "asr" }
    }
  ]
}
```

Поле `context` опционально: например `source: process_video`, `batch_kind: gpu_segments`, `scope_suffix` / `file_id` при параллельном CPU batch (ключ события тогда вида `mel::<file_id>`). Поле `delta` вычисляется как разность **числовых** метрик `snap_after − snap_before` (при отсутствии значения — `null`).

Глобальные пики **RSS / VRAM** по всему процессу по-прежнему пишутся модулем `resource_monitor.ResourceMonitor` (фоновый сэмплер) в те же `rss_peak_mb` / `gpu_used_peak_mb` внутри `per_processor.audio` — это **дополняет**, а не дублирует покадровую точность по экстракторам.

## Код

| Файл | Роль |
|------|------|
| [`src/core/orchestrator_telemetry.py`](../src/core/orchestrator_telemetry.py) | `OrchestratorTelemetryCollector`, снимки `_resource_snapshot`, флаги env |
| [`src/core/extractor_runner.py`](../src/core/extractor_runner.py) | Вызовы `mark_extractor_start` / `mark_extractor_end` вокруг `run_single_extractor` |
| [`src/core/main_processor.py`](../src/core/main_processor.py) | Batch: GPU `extract_batch_segments`; CPU путь — `process_video` + опциональный `telemetry_scope_suffix` |
| [`run_cli.py`](../run_cli.py) | Создание коллектора, передача в `run_extractors` и в `run_batch`, слияние в `scheduler_runtime_report.json` |
| [`src/core/resource_monitor.py`](../src/core/resource_monitor.py) | Глобальный пик RSS/GPU за весь прогон (без изменения контракта) |

## Подводка компонентов (roadmap)

1. **AudioProcessor extractors (single-file)** — покрыты `run_extractors` + `run_cli.py`.
2. **Batch mode** (`processor.run_batch`) — события на GPU-батч (`extract_batch_segments`) и на каждый `extractor.run` в CPU-ветке (`process_video`); коллектор потокобезопасен для параллельных воркеров.
3. **Субпроцессы** — если экстрактор сам спавнит процесс, имеет смысл включить `AP_ORCHESTRATOR_TELEMETRY_CHILDREN=1` на отладочных прогонах.
4. **DataProcessor API / VisualProcessor** — аналогичный каркас: общий модуль в `DataProcessor/` (реэкспорт или копия лёгкого API) + запись в свой runtime-отчёт; см. план Audit 4.2 §12.2.

## Версионирование

- `schema_version`: `orchestrator_telemetry_v1` — первый стабильный формат событий; при несовместимых изменениях — bump и запись в этом файле.
---

## Навигация

[README](README.md) · [Module README](../README.md) · [AudioProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
