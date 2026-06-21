# E2E и DataProcessor: фиксы после ребута и cold cache (2026-04)

Документ фиксирует известные поломки полной цепочки **Backend → Fetcher → DataProcessor** после перезагрузки машины или «холодного» кеша и соответствующие правки в репозитории.

## 1. Backend: таймаут на `POST /api/v1/process`

**Симптом:** `ingestion_status=failed`, в логах Backend worker — `httpx.ReadTimeout` / «Failed to connect to DataProcessor API» при вызове DataProcessor.

**Причина:** DataProcessor в рамках одного HTTP-запроса может **скачивать `video_url` в локальный кеш** (десятки секунд на больших файлах). У httpx-клиента в Backend стояло **30 s** на весь ответ — соединение обрывалось до `202 Accepted`.

**Правка:**

- `backend/app/config.py`: `dataprocessor_enqueue_timeout_seconds` (по умолчанию **600**), env `TF_BACKEND_DATAPROCESSOR_ENQUEUE_TIMEOUT_SECONDS`.
- `backend/app/services/dataprocessor.py`: для `run_dataprocessor_async` этот таймаут передаётся в `httpx.AsyncClient`.

**Практика:** в `backend/scripts/e2e_env.sh` экспортировано значение по умолчанию 600 s.

## 2. Старт E2E-стека на медленном диске

**Симптом:** `start_e2e_stack.sh` падает с «backend-api did not open 127.0.0.1:8001 in time», хотя через минуту uvicorn поднимается.

**Причина:** импорт `app.main` на томах с высокой латентностью может занимать **~1 мин**; ожидание порта было **120 s** без запаса.

**Правка:** в `backend/scripts/start_e2e_stack.sh` функция `wait_for_port` ждёт до **300 s**.

## 3. GlobalConfig → AudioProcessor CLI (`config_parser.py`)

Раньше часть флагов из `global_config.yaml` превращалась в **несуществующие** аргументы argparse или в режимы, запрещённые Audit v3:

| Область | Проблема | Решение |
|--------|----------|---------|
| **emotion** | `--emotion-enable-ids` и т.п. (в CLI только `--emotion-disable-*` для default-on) | Явное сопоставление флагов с `cli_args.py` |
| **quality** | `--quality-enable-basic-metrics` не существует | Только `--quality-disable-basic-metrics` при `enable_basic_metrics: false` |
| **chroma** | Audit v3 запрещает basic/extended stats в экстракторе | Не пробрасывать stats-флаги; только audio norm + time series |
| **emotion** | `process_full_audio: true` → `--emotion-process-full-audio` | Не генерировать флаг; в YAML `process_full_audio: false` |

Подробная логика — в `DataProcessor/configs/config_parser.py`.

## 4. AudioProcessor: NPZ / схемы / память GPU

Отдельный прогон «полного» аудио по `global_config.yaml` выявлял:

- **Mel (`run_segments`):** глобальные `mel_mean` / `mel_std` / … не попадали в payload при включённой статистике → расхождение размерности `M` с `mel_mean_by_segment` при валидации схемы.
- **Chroma (сегменты):** `chroma_frames = -1` ломал локальную валидацию (`must be non-negative`).
- **JSON-схемы NPZ:** размерности вида `"12"` или `"2"` не парсятся валидатором (ожидаются **числа** или символы `N`, `M`, …).
- **Pitch:** `pitch_octave_distribution` сохранялся как массив формы `(1,)` вместо скаляра `object` для схемы `shape: []`.
- **Key:** в NPZ попадал лишний ключ `payload` из `atomic_save_npz`.
- **HPSS:** проверка «схемы энергий» требовала `share_h + share_p ≈ 1`, хотя доли считаются от **полной** энергии STFT и сумма может быть **меньше 1** (нормально для HPSS).
- **GPU:** на картах ~6 GiB параллельные тяжёлые модели давали OOM — в `global_config.yaml` снижены `batch_size` для `emotion_diarization` и `source_separation`.
- **Speaker diarization:** если Segmenter отдаёт много окон в `families.diarization.segments`, экстрактор раньше падал (`expected exactly 1 … segment`); теперь окна **схлопываются** в одно по правилу `[min(start_sec), max(end_sec)]` и выполняется один проход pyannote на всём охвате.

- **Рендер (не влияет на NPZ, но шумит в логах / ломает HTML):** в `pitch_extractor/utils/render.py` была строка с неэкранированными кавычками в `class="grid"` (syntax error при импорте); в `band_energy_extractor/utils/render.py` HTML использовал `:.2f` / `:.3f` для значений, которые в meta бывают `None`.

Детали правок — в диффе соответствующих файлов (`mel_extractor/main.py`, `chroma_extractor/main.py`, `npz_savers/*.py`, `schemas/*.json`, `hpss_extractor/main.py`, `global_config.yaml`, `*_extractor/utils/render.py`).

## 5. Команды для повторения полного E2E

```bash
# Инфра + стек
./backend/scripts/start_e2e_stack.sh --with-infra

cd backend && source scripts/e2e_env.sh
export DP_MODELS_ROOT="/abs/path/to/TrendFlowML/DataProcessor/dp_models/bundled_models"
export TRITON_HTTP_URL=http://127.0.0.1:8010   # или --with-triton-docker в скрипте ниже

.venv/bin/python -u scripts/e2e_full_max_run.py --with-triton-docker --offline-example --timeout 7200
```

После смены кода **перезапустите** `dataprocessor-worker` (или весь стек), чтобы подтянуть `config_parser.py` и экстракторы.

**Ожидаемый результат и необязательный Embedding Service (8005):** см. [E2E_RUNBOOK.md § 9 — Полный max-E2E](E2E_RUNBOOK.md#9-полный-max-e2e-e2e_full_max_runpy).

## 6. Отдельный прогон только AudioProcessor

С тем же `global_config.yaml` аргументы CLI собираются так:

```bash
cd DataProcessor
./AudioProcessor/.ap_venv/bin/python -c "
from pathlib import Path
import sys
sys.path.insert(0, '.')
from configs.config_parser import GlobalConfigParser
p = GlobalConfigParser(Path('configs/global_config.yaml'))
print(' '.join(p.get_audio_cli_args()))
" | ...  # добавить к run_cli.py --frames-dir ... --run-rs-path ...
```

**Важно:** в `get_audio_cli_args()` есть значения вроде `--extractor-parallelism-config` / `--extractor-config` с JSON, который содержит пробелы. Подставлять их через оболочку (`shlex`, `$(python -c ...)` без кавычек) **нельзя** — JSON разобьётся на отдельные аргументы. Надёжно: один вызов Python, `subprocess.run([sys.executable, 'AudioProcessor/run_cli.py', ...] + p.get_audio_cli_args(), ...)` или эквивалент с списком argv.

Или см. `backend/scripts/e2e_dataprocessor_audio_smoke.py` после готового Fetcher run.

---

**Связанные документы:** [E2E_RUNBOOK.md](E2E_RUNBOOK.md) (в т.ч. § 9 max-E2E), [E2E_FULL_CHECKLIST.md](E2E_FULL_CHECKLIST.md) § 4.1, [E2E_PIPELINE_NO_TEXT.md](E2E_PIPELINE_NO_TEXT.md), `backend/scripts/e2e_env.sh`, `DataProcessor/configs/global_config.yaml`.
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
