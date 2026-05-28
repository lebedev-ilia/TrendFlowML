# Инвентаризация Prometheus labels (батч 60+ / чеклист п. 4.5)

**Цель:** понять, какие `processor` / `component` реально попадают в гистограмму и счётчик ошибок при прогоне через **API + worker**.

## Объявленные метрики

Источник: `DataProcessor/api/services/metrics.py`

| Метрика | Labels | Назначение |
|---------|--------|------------|
| `dataprocessor_queue_length` | `priority` | длина очереди |
| `dataprocessor_queue_wait_seconds` | (без labels) | ожидание в очереди |
| `dataprocessor_processing_seconds` | `processor`, `component` | время обработки **одного** run (как снимает worker) |
| `dataprocessor_failures_total` | `processor`, `component`, `error_type` | ошибки |
| `dataprocessor_memory_bytes` | `run_id` | память (отдельные вызовы из `processor.py`) |
| `dataprocessor_active_runs` | — | активные run |
| `dataprocessor_crashed_runs_total` | — | crashed (recovery) |

## Где выставляются labels

### `dataprocessor_processing_seconds`

Файл: `DataProcessor/api/services/worker.py` — после `processor_service.run_processing(request)`:

- `processor = result.get("processor") or "unknown"`
- `component = result.get("component") or "unknown"`

### Обновление кода (2026-04-15)

`DataProcessor/api/services/processor.py` обогащает все возвращаемые из `_run_main_py_async` / legacy `_run_main_py_sync` словари полями (через `setdefault`, чтобы не перетирать будущие уточнения):

- `processor="pipeline"`
- `component="main_py"`

Это **сквозной** subprocess-запуск `main.py` (весь пайплайн одним процессом), а не отдельные модули Visual/Audio/Text. В Grafana по-прежнему **нет** per-component latency из этой гистограммы без дополнительной инструментации внутри `main.py` или отдельных задач воркера.

**Исторический контекст:** раньше `processor`/`component` в ответе отсутствовали → в Prometheus попадало `unknown`/`unknown`. После патча ожидаются `pipeline`/`main_py` — **подтвердить на пилоте** (чеклист п. 4.5).

### `dataprocessor_failures_total`

Тот же `worker.py`: при неуспехе берутся `processor`, `component`, `error_type` из `result`. После патча в `processor.py` для ошибок subprocess и ошибок сохранения профиля также задаются **`pipeline` / `main_py`** (если не переопределены выше по стеку).

### `dataprocessor_memory_bytes`

Обновляется из `processor.py` (импорт `memory_usage`) — там label `run_id`; к разрезу по компонентам не относится.

## Рекомендации перед батчем (закрытие п. 4.5–4.6)

1. **Пилот:** в Prometheus проверить `dataprocessor_processing_seconds_bucket` и `dataprocessor_failures_total` после 1–2 run — ожидаются `processor="pipeline", component="main_py"` для маршрута API → worker → `main.py`.
2. **Дашборды:** не ожидать из этой гистограммы разрез по внутренним экстракторам; для этого нужны отдельные метрики или этапы воркера.
3. **CLI без API:** если батч идёт только через CLI, метрики worker не обновляются — отдельный путь (exporter, Pushgateway, обязательный API) — см. чеклист **4.6**.

## Ссылки

- [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) п. 4.5–4.6  
- [monitoring/README.md](../../monitoring/README.md)  
- Пример «ожидаемых» labels в комментарии: `DataProcessor/api/endpoints/metrics.py`
