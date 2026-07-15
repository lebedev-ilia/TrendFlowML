# RUN_RESULT: action_recognition (2026-07-02)

Исполнитель: Cursor  
RUN_SPEC: `DataProcessor/docs/component_reports/action_recognition/RUN_SPEC.md`

## Что сделано

- Проверен preflight по варианту A (`SlowFast R50`): локальный checkpoint уже был в
  `DataProcessor/dp_models/bundled_models/visual/action_recognition/slowfast_r50/slowfast_r50.pyth`.
- Формат checkpoint совместим с `utils/action_recognition_slowfast.py` без правок:
  объект — `dict` с ключом `model_state`, префикса `module.` нет, дополнительная обёртка
  в `{"state_dict": ...}` не потребовалась.
- Выход компонента не менялся: использовался текущий `action_recognition_npz_v2`
  (`256d` L2-норм. embedding на трек + metric fields).
- Прогон выполнен по минимальной цепочке:
  `Segmenter -> core_object_detections -> action_recognition`.

## Входная матрица

Локальная матрица собрана в
`DataProcessor/docs/component_reports/action_recognition/artifacts/input_videos/`
из уже скачанных `example/hf_videos11/*.mp4`.

Важно: для покрытия длин 1m/2m/4m/8m использовались локальные склейки/повторы этих
реальных коротких роликов. Фактические длительности зафиксированы ниже и в
`artifacts/input_videos/matrix_manifest.json`.

## Статусы по видео

| video_id | duration_sec | run_id | status | empty_reason | track_count | total_clips | tracks_with_multi_clips | visual_wall | peak_vram |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| `ar_10s_talking` | 10.733 | `ar_validation_01` | `ok` |  | 1 | 1 | 0 | 22.1 s | 1418 MB |
| `ar_10s_control_no_people` | 10.633 | `ar_validation_02` | `ok` |  | 1 | 1 | 0 | 16.0 s | 1418 MB |
| `ar_30s_person_a` | 29.633 | `ar_validation_03` | `ok` |  | 4 | 5 | 1 | 16.8 s | 1418 MB |
| `ar_30s_person_b` | 49.933 | `ar_validation_04` | `ok` |  | 3 | 8 | 1 | 38.6 s | 1418 MB |
| `ar_1m_person_a` | 59.267 | `ar_validation_05` | `ok` |  | 7 | 11 | 2 | 20.6 s | 1418 MB |
| `ar_1m_person_b` | 79.567 | `ar_validation_06` | `ok` |  | 9 | 11 | 1 | 27.5 s | 1418 MB |
| `ar_2m_person_a` | 118.533 | `ar_validation_07` | `ok` |  | 28 | 28 | 0 | 26.4 s | 1596 MB |
| `ar_2m_person_b` | 138.833 | `ar_validation_08` | `ok` |  | 110 | 110 | 0 | 46.8 s | 1596 MB |
| `ar_4m_person` | 237.067 | `ar_validation_09` | `ok` |  | 205 | 205 | 0 | 54.5 s | 1596 MB |
| `ar_8m_person` | 503.767 | `ar_validation_10` | `ok` |  | 199 | 199 | 0 | 52.8 s | 1596 MB |

## Пути к ключевым артефактам

- Основная сводка batch:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/run_summary.json`
- Видео-матрица:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/input_videos/matrix_manifest.json`
- Batch CSV / JSONL:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/action_recognition_batch.csv`
  `DataProcessor/docs/component_reports/action_recognition/artifacts/action_recognition_batch.jsonl`
- Health audit:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/action_recognition_health.json`
  `DataProcessor/docs/component_reports/action_recognition/artifacts/action_recognition_health.csv`
  `DataProcessor/docs/component_reports/action_recognition/artifacts/action_recognition_health.md`
- §0.2 validator outputs:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/output_quality_results.json`
- Per-video logs:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/logs/*`
- Per-video configs:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/configs/*`
- Per-video run_store:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/run_store/youtube/<video_id>/<run_id>/`

## Тайминги и ресурсы

- Суммарное время batch runner: ~`479.7 s` на 10 видео.
- `core_object_detections` по batch:
  min `6149 ms`, max `15096 ms`, avg `11530 ms`
- `action_recognition` по batch:
  min `6372 ms`, max `36078 ms`, avg `17250 ms`
- Внутренний `action_recognition process` stage по метаданным:
  - `ar_10s_talking`: `870.8 ms`
  - `ar_30s_person_b`: `1805.3 ms`
  - `ar_1m_person_b`: `2360.5 ms`
  - `ar_2m_person_a`: `4485.7 ms`
  - `ar_2m_person_b`: `21312.9 ms`
  - `ar_4m_person`: `29324.0 ms`
  - `ar_8m_person`: `28968.1 ms`
- Пиковая VRAM по wrapper-monitoring:
  - короткие / до ~80s: `1418 MB`
  - 2m+ ролики: `1596 MB`
- Peak RSS по wrapper-monitoring рос до ~`3.0 GB` на `ar_30s_person_b` и ~`2.3 GB` на части длинных роликов
  (точные числа есть в `run_summary.json`).

## Версии

- `dataprocessor_version`: `action_recognition_validation_runner`
- `sampling_policy_version`: `action_recognition_validation_v1`
- `producer_version` (`action_recognition`): `2.0`
- `schema_version`: `action_recognition_npz_v2`
- `model_signature`:
  - `model_name`: `slowfast_r50_action_recognition`
  - `model_version`: `v1`
  - `weights_digest`: `887a8958d34c7177f8956ea280bb3c31a79d3d4a3039cec8635a25380a214c43`
  - `runtime`: `inprocess`
  - `engine`: `torch`
  - `precision`: `fp32`
  - `device`: `cuda`

## Аномалии и наблюдения

- Контрольное видео `ar_10s_control_no_people` **не дало `empty`**:
  `action_recognition` вернул `status=ok`, `track_count=1`, `total_clips=1`.
  Это расходится с ожиданием RUN_SPEC для "без людей".
- На длинных synthetic-монтажах `track_count` резко растёт:
  - `ar_2m_person_b`: `110` треков / `110` клипов
  - `ar_4m_person`: `205` / `205`
  - `ar_8m_person`: `199` / `199`
  При этом `tracks_with_multi_clips=0`, то есть длинные прогоны часто распадаются
  на множество одно-клиповых треков/сегментов.
- На более коротких роликах multi-clip треки всё же есть:
  - `ar_30s_person_a`: `1`
  - `ar_30s_person_b`: `1`
  - `ar_1m_person_a`: `2`
  - `ar_1m_person_b`: `1`
- `feature_quality_audit` отработал без явных красных флагов; нижние score в markdown —
  в основном по meta-полям (`80-90`, severity=`low`), не по runtime-ошибкам фич.
- `e2e_validate_output_quality.py` на этом профиле системно падает для всех run не из-за
  самого `action_recognition`, а потому что инструмент заточен под full E2E manifest:
  - `manifest run.status=None`
  - `text_processor/text_features.npz missing`
  Это ожидаемо для isolated visual-only profile; stderr/stdout сохранены в
  `artifacts/output_quality_results.json`.

## Технические заметки по прогону

- Первый заход упал из-за отсутствия `ffmpeg` в PATH; повторный batch выполнен с
  `tools/bin/ffmpeg` и `tools/bin/ffprobe` из репозитория.
- Для reproducibility добавлен runner:
  `DataProcessor/docs/component_reports/action_recognition/artifacts/run_action_recognition_validation.py`
  Он создаёт configs, запускает `Segmenter` и `VisualProcessor`, собирает логи и
  wrapper-метрики RAM/VRAM.
