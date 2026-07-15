# RUN_RESULT: action_recognition v3 — итерация 2 (R1+R2)

Исполнитель: Cursor  
Основание: `RUN_NOTE_FOR_CURSOR.md` (блок «⟳ Итерация 2»), `RUN_SPEC_v2.md`  
Статус: **DONE** (5 прогонов: 3 видео + golden ×2 на 4:35)

Прервано перезагрузкой ПК после 3/5 прогонов; догонены `ar_v3_golden_b` и `ar_v3_03`.

## Preflight

| шаг | результат |
|---|---|
| Kinetics labels | `provision_kinetics_labels.py` → 403 на pytorchvideo URL; записано вручную из `pyslowfast/.../kinetics_classnames.json` → `dp_models/visual/action_recognition/kinetics400_labels.txt` (400 строк) + копия в `dp_models/bundled_models/visual/action_recognition/` |
| `DP_MODELS_ROOT` в runner | выставлен `dp_models/bundled_models` для subprocess action_recognition (прогоны до golden_b имели placeholder `action_<id>` в NPZ) |
| Диск | dense-кадры ~**10.7 GB/видео** (4:35/8:00); после каждого прогона frames удаляются; пик заполнения ~92% до очистки |

## Профиль

`Segmenter` → `core_object_detections` (tracking **histogram**) → `action_recognition` v3  
`sampling_policy_version`: **`action_recognition_validation_v3`**  
`clip_len=32`, `window_hop_s=2.0`, `max_windows=48`, `min_clip_real_frames=16`

## Статусы по видео

| video_id | run_id | status | clip_count | num_tracks (AR) | mean_clips_per_track | det_num_tracks | det_mean_track_len | validator | frames_disk_mb |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `ar_real_2m47_control_nopeople` | `ar_v3_01` | `empty` | 0 | 0 | 0 | 0 | 0 | ✅ | 1148 |
| `ar_real_4m35_people` | `ar_v3_02` | `ok` | 35 | 35 | **1.0** | 30 | **77.5** | ✅ | 10721 |
| `ar_real_8m00_people` | `ar_v3_03` | `ok` | 45 | 45 | **1.0** | 35 | **85.7** | ✅ | 10795 |

## Golden (4:35 ×2)

| метрика | `ar_v3_golden_a` / `ar_v3_golden_b` |
|---|---|
| `clip_embeddings` | **идентичны** |
| `track_ids` | **идентичны** |
| validator | ✅ оба |

## Top-5 dominant actions (Kinetics, для сверки владельцем)

**4:35** (`ar_v3_02` / golden — одинаковые id/probs):

| id | prob | label |
|---:|---:|---|
| 225 | 0.436 | playing clarinet |
| 233 | 0.162 | playing harmonica |
| 244 | 0.079 | playing saxophone |
| 232 | 0.070 | playing guitar |
| 392 | 0.045 | whistling |

**8:00** (`ar_v3_03`):

| id | prob | label |
|---:|---:|---|
| 59 | 0.155 | clean and jerk |
| 88 | 0.101 | deadlifting |
| 398 | 0.088 | yoga |
| 318 | 0.062 | snatch weight lifting |
| 140 | 0.035 | giving or receiving award |

`classes_available=true` на people-видео; `class_names` в NPZ с реальными метками (после фикса `DP_MODELS_ROOT`).

## Сравнение v2 → v3 (итерация 2)

| метрика | v2 (4:35) | v3 (4:35) | v2 (8:00) | v3 (8:00) |
|---|---:|---:|---:|---:|
| clip_count | 22 | **35** | 19 | **45** |
| det_mean_track_len | 14.7 | **77.5** | 16.0 | **85.7** |
| det_num_tracks | 26 | 30 | 31 | 35 |
| mean_clips_per_track | 1.0 | 1.0 | 1.0 | 1.0 |
| cod process (manifest ms) | ~58 s | **~180 s** | ~27 s | **~143 s** |
| AR process (meta ms) | ~4 s | **~29 s** | ~3 s | **~30 s** |
| segmenter wall | 309 s | 186 s | 222 s | 381 s |
| visual wall | 76 s | 229 s | 39 s | 188 s |
| frames cache | ~не логировался | **~10.7 GB** | — | **~10.8 GB** |

**R1 (dense det + min_clip_real_frames):** треки стали **полными** (`mean_track_len` ~78–86 vs ~15 в v2), клипов больше (35/45 vs 22/19), паддинг по детекции устранён. **`mean_clips_per_track` всё ещё 1.0** — по-прежнему 1 клип на окно (`max_windows=48`), multi-clip на трек через re-ID между окнами не проявился в агрегате.

**R2 (метки):** реальные Kinetics-имена в top-действиях (см. таблицы).

## Тайминги / стоимость (200k-ось)

- **Детекция на dense-кадрах** — главный рост: cod ~3× на 4:35, ~5× на 8:00 vs v2.
- **AR process** ~7× на 4:35 (больше клипов + полные треки).
- **Трекинг overhead** (из логов 4:35 v2-стиля): ~2–4% от `process_frames`; на v3 абсолютно больше из-за большего числа кадров.
- **Диск R3:** без удаления frames после NPZ — ~11 GB/4:35, ~11 GB/8:00; на 200k нужна политика не хранить кэш.

## Контроль

`ar_v3_01`: `empty` / `no_person_detections`, `person_fp_proxy=0`, validator ✅.

## Артефакты

База: `DataProcessor/docs/component_reports/action_recognition/artifacts/v3/`

| файл | описание |
|---|---|
| `run_summary.json` | полная сводка 5 прогонов |
| `golden_compare.json` | golden digest |
| `action_recognition_batch.csv` | batch report (3 основных run) |
| `action_recognition_health.{md,csv,json}` | feature quality audit |
| `run_store/youtube/<video>/<run_id>/` | NPZ + manifest |
| `logs/` | segmenter + visual |
| `run_action_recognition_validation_v3.py` | runner |

## Вердикт для REPORT (честно)

| критерий итерации 2 | результат |
|---|---|
| valid-empty контроль | ✅ |
| validator v3 | ✅ все прогоны |
| golden идентичен | ✅ |
| det mean_track_len ≈ clip_len на длинных | ✅ (~78–86) |
| mean_clips_per_track > 1 | ❌ остаётся 1.0 |
| осмысленные class_names | ✅ (после DP_MODELS_ROOT) |
| стоимость/диск приемлемы | ⚠️ cod×3–5, ~11 GB frames/видео |
