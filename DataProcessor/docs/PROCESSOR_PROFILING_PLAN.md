# План профилирования процессоров (Audio / Text / Visual / Segmenter)

## Цель

Сделать сопоставимым **суммарное время подкомпонентов** и **wall-time процессора** в E2E/API: явно учитывать инициализацию, пост-обработку артефактов, оркестрацию DAG и ожидание I/O.

## Текущее состояние (после последних изменений)

| Источник | Что уже пишется |
|----------|-----------------|
| **AudioProcessor** `run_cli.py` | В `manifest.json` → `run.audio_processor_timings_ms`: `wall_subprocess_before_internal_t0_ms`, `run_extractors_wall_ms`, `sum_extractor_wall_ms`, `save_npz_stage_ms`, `extractors_orchestration_overhead_ms`, при запуске из `DataProcessor/main.py` — `audio_subprocess_wall_ms`, `subprocess_residual_ms`. Лог: `AudioProcessor \| wall timing breakdown`. |
| **TextProcessor** `run_cli.py` | `run.text_processor_timings_ms`: фазы загрузки документа, `MainProcessor.run`, NPZ/validate/render, сумма `timings_by_extractor`, orchestration overhead. Лог: `TextProcessor \| wall timing breakdown`. |
| **Segmenter / Visual** | Отдельный этап: добавить симметричные `*_processor_timings_ms` в manifest или в `state_*.json` по тому же контракту. |

Окружение для полного wall субпроцесса:

- `DP_AUDIO_WALL_T0` — выставляет `DataProcessor/main.py` перед AudioProcessor.
- `DP_TEXT_WALL_T0` — перед TextProcessor.

## Этапы работ (приоритет)

1. **Единая схема JSON** для `manifest.run` (или `run.processor_timings`): `schema_version`, `stages_ms`, `sum_subcomponents_ms`, `overhead_ms`, `subprocess_wall_ms`, краткий `note`.
2. **VisualProcessor** — верхний уровень: время импорта/конфига, загрузка кадров, DAG modules/cores (уже частично в событиях); агрегировать в один объект при финальном flush manifest (как audio/text).
3. **Segmenter** — wall от старта subprocess до записи `frames_dir` + разбивка по шагам, если есть в CLI.
4. **Субкомпоненты** — опционально: в каждом extractor/module добавлять `stage_timings_ms` в meta NPZ (как у `ocr_extractor`); не делать всё сразу — вести реестр компонентов и мигрировать пакетами.
5. **API / StateReader** — опционально: прокидывать `run.processor_timings` (или выбранные ключи) в ответ `/status` для E2E без чтения manifest с диска.

## Ограничения

- Сумма длительностей подкомпонентов **не обязана** совпадать с wall: параллелизм (Visual batch), ожидание GPU, повторные проходы и пост-обработка идут в `overhead_ms` / отдельные стадии.
- Политика **offline**/отсутствующих моделей: ошибки модулей visual остаются диагностируемыми по `manifest` / логам конкретного модуля; общий план не заменяет починку каждой модели.

## Связанные файлы

- `DataProcessor/AudioProcessor/run_cli.py`
- `DataProcessor/TextProcessor/run_cli.py`
- `DataProcessor/main.py` (`DP_*_WALL_T0`)
- `DataProcessor/api/services/state_reader.py` (merge manifest ↔ API)
- `DataProcessor/VisualProcessor/utils/manifest.py` (`RunManifest`)
