# RUN_RESULT: action_recognition v2 (2026-07-04)

Исполнитель: Cursor  
RUN_SPEC: `DataProcessor/docs/component_reports/action_recognition/RUN_SPEC_v2.md`  
Статус: **DONE** (5 прогонов: 3 видео + golden ×2 на 4:35)

## Краткий вердикт

v3-стек отработал на реальных фикстурах. Контракт `action_recognition_npz_v3` проходит валидатор на всех прогонах.
Контроль без людей — валидный `empty (no_person_detections)`. Фрагментация треков **существенно снижена**
(26 det-треков на 4:35 vs 205 surrogate-треков в v1 на ~4m). Golden на 4:35 **идентичен**.
`mean_clips_per_track` остаётся **1.0** из‑за капа `max_windows=48` (1 клип на окно) — открытый вопрос для 200k.

## Патчи при прогоне (LOGIC_ERRORS-стиль)

1. **`appearance_tracker` importlib**: `sys.modules[spec.name] = mod` перед `exec_module` — иначе `@dataclass` падает на Py3.14 (`'NoneType' object has no attribute '__dict__'`).
2. **`VisualProcessor/main.py`**: исключить `window_hop_s`, `max_windows`, `render` из CLI `action_recognition` (Segmenter-only ключи).
3. **`Segmenter/segmenter.py`**: проброс `clip_len` / `window_hop_s` / `max_windows` из visual cfg в extractor_configs.
4. **Диск**: первый прогон 8:00 оборвался при **100% fill** на `/` (~22 GB кэша кадров). После удаления `artifacts/v2/frames` и v1 `artifacts/frames` — повтор успешен.

## Профиль

- Цепочка: `Segmenter` → `core_object_detections` (tracking **histogram**) → `action_recognition` v3
- `clip_len=32`, `window_hop_s=2.0`, `max_windows=48`
- `sampling_policy_version`: **`action_recognition_validation_v2`**
- `track_embedder`: **histogram** (`histogram_hsv_v1`), OSNet не подключён

## Статусы по видео

| video_id | duration_sec | run_id | status | empty_reason | clip_count | det_num_tracks | mean_track_len | validator | peak_vram |
|---|---:|---|---|---|---:|---:|---:|---|---:|
| `ar_real_2m47_control_nopeople` | 167.4 | `ar_v2_01` | `empty` | `no_person_detections` | 0 | 0 | — | pass | 1418 MB |
| `ar_real_4m35_people` | 275.0 | `ar_v2_02` | `ok` | — | 22 | 26 | 14.7 | pass | 1418 MB |
| `ar_real_8m00_people` | 480.6 | `ar_v2_03` | `ok` | — | 19 | 31 | 16.0 | pass | 1550 MB |

## Golden (4:35 ×2)

| метрика | результат |
|---|---|
| `clip_embeddings` digest | **идентичны** (`2d19bd74443b6f48…`) |
| `detections.npz:track_ids` digest | **идентичны** (`048cf9af2b9584df…`) |
| run_a / run_b | `ar_v2_golden_a` / `ar_v2_golden_b` |

## Сравнение с v1 (фрагментация)

| ось | v1 (synthetic ~4m) | v2 (real 4:35) | v2 (real 8:00) |
|---|---:|---:|---:|
| det / surrogate track_count | 205 | **26** | **31** |
| action_recognition clips | 205 | **22** | **19** |
| tracks_with_multi_clips | 0 | 0 | 0 |
| `mean_clips_per_track` | ~1 | **1.0** | **1.0** |
| `frac_single_len` (tracker) | n/a | 0.19 | 0.13 |

## Прокси-метрики трекера (histogram embedder)

| video | intra_cos (mean) | inter_cos (mean) | n_tracks |
|---|---:|---:|---:|
| 4:35 | 0.581 | 0.208 | 26 |
| 8:00 | 0.578 | 0.300 | 31 |

Разделимость id: **intra ≫ inter** на 4:35; на 8:00 inter выше, но intra всё ещё ~2×.

Re-ID на 8:00: трек `id=0` — **len=94** детекций (длинная когерентность через уходы из кадра); рендер боксов не генерировался (`enable_render=false`).

## Тайминги (wall + стадии)

| video | segmenter_wall | visual_wall | cod_process_frames | cod_tracking | ar_process |
|---|---:|---:|---:|---:|---:|
| 2:47 control | 13.6 s | 24.1 s | ~8.5 s | n/a (0 persons) | 5 ms |
| 4:35 | 309 s | 75.6 s | 53.6 s | **2.3 s** | 4.0 s |
| 8:00 | 222 s | 39.5 s | ~24 s | ~2 s (log) | 3.0 s |

Overhead трекинга на 4:35: **~4%** от `process_frames` (2.3s / 53.6s).

## Версии / модели

| поле | значение |
|---|---|
| `action_recognition` schema | `action_recognition_npz_v3` |
| `core_object_detections` schema | `core_object_detections_npz_v3` |
| `sampling_policy_version` | `action_recognition_validation_v2` |
| SlowFast | `slowfast_r50_action_recognition` v1, digest `887a8958…` |
| `model_signature` (AR) | `82b22efc347a6cf3…` |
| `classes_available` | `true` (people videos); labels = `action_<id>` (нет `kinetics400_labels.txt`) |
| tracker embedder | `histogram_hsv_v1` |

## Артефакты

База: `DataProcessor/docs/component_reports/action_recognition/artifacts/v2/`

| артефакт | путь |
|---|---|
| Сводка прогонов | `v2/run_summary.json` |
| Golden | `v2/golden_compare.json` |
| Batch CSV | `v2/action_recognition_batch.csv` |
| Health MD/CSV/JSON | `v2/action_recognition_health.{md,csv,json}` |
| NPZ (примеры) | `v2/run_store/youtube/<video_id>/<run_id>/` |
| Логи | `v2/logs/` |
| Runner | `artifacts/run_action_recognition_validation_v2.py` |

## Валидатор v3

Все NPZ: `validate_action_recognition_npz.py` → **pass** (включая valid-empty на контроле).

## Аномалии / открытые вопросы (для REPORT)

1. **`mean_clips_per_track=1.0`**: кап `max_windows=48` → ~1 SlowFast-клип на окно; трекер даёт длинные треки (mean_len 14–16), но AR не агрегирует несколько окон на трек. Нужен ли подъём `max_windows` / multi-clip-per-track?
2. **Метки Kinetics**: `class_names` = placeholder `action_<id>` — положить `kinetics400_labels.txt` для сверки владельцем.
3. **Embedder**: histogram достаточен для cosine-разделимости; OSNet — по метрике на следующем прогоне.
4. **Диск**: dense-окна + union-кадры → **~10 GB/видео** кэша frames на 8:00; для 200k нужен policy не хранить frames после NPZ или сэмплировать реже.
5. **DoD §5**: фрагментация ↓ и golden ✓; `mean_clips_per_track>1` и осмысленные class names — **ещё нет**; стоимость 8:00 ~4.4 min wall — приемлемо при текущих капах.
