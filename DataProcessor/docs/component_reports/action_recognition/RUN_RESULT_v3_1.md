# RUN_RESULT: action_recognition v3.1 — итерация 3 (ASSESSMENT)

Исполнитель: Cursor  
Основание: `RUN_NOTE_FOR_CURSOR.md` (блок «⟳ Итерация 3»)  
Статус: **DONE** (7 прогонов: 3 фикстуры + golden ×2 + group 30s + osnet 8:00)

Первый проход runner остановился после `ar_v3_1_03` (диск 100% — frames 8:00 не удалены). Group + osnet догнаны отдельно; frames очищены вручную (`rm -rf artifacts/v3_1/frames/*`).

## Preflight

| шаг | результат |
|---|---|
| Kinetics labels | `provision_kinetics_labels.py` → **403** (pytorchvideo URL); используется существующий `dp_models/visual/action_recognition/kinetics400_labels.txt` (**400** строк, реальные имена) |
| SlowFast weights | `dp_models/bundled_models/.../slowfast_r50/slowfast_r50.pyth` — на месте |
| Penultimate hook | лог: `penultimate-hook установлен на голову (blocks[-1].proj)`; `meta.embedding_mode=penultimate` (не `projection_fallback`) |

## Профиль v3.1

`Segmenter` → `core_object_detections` (tracking **histogram**, опц. osnet) → `action_recognition` v3.1  
`sampling_policy_version`: **`action_recognition_validation_v3_1`**

Флаги (явно в runner cfg + дефолты CLI):

- `--embedding-mode penultimate`
- `--localization track_anchored`
- `--tubelet-crop true`
- `--min-clip-real-frames 16`
- `--precision fp32`

## Статусы по видео

| video_id | run_id | status | clip_count | mean_clips_per_track | num_action_segments | embedding_mode | embedding_dim | validator_in | validator_out | metrics.json |
|---|---|---|---:|---:|---:|---|---:|---|---|---|
| `ar_real_2m47_control_nopeople` | `ar_v3_1_01` | `empty` | 0 | 0 | 0 | penultimate | 256¹ | ✅ | ✅ | ✅ |
| `ar_real_4m35_people` | `ar_v3_1_02` | `ok` | 31 | **1.0** | **31** | penultimate | **2304** | ✅ | ✅ | ✅ |
| `ar_real_8m00_people` | `ar_v3_1_03` | `ok` | 43 | **1.0** | **43** | penultimate | **2304** | ✅ | ✅ | ✅ |
| `ar_30s_person_b` (group) | `ar_v3_1_group` | `ok` | 20 | 1.0 | 20 | penultimate | 2304 | ✅ | ✅ | ✅ |

¹ на empty-run dim в meta = CLI default 256 (клипов нет, hook не обновлял размерность); на people-run **2304**.

## Golden (4:35 ×2)

| метрика | `ar_v3_1_golden_a` / `ar_v3_1_golden_b` |
|---|---|
| `clip_embeddings` | **идентичны** (digest `357106b5…`) |
| `track_ids` | **идентичны** (digest `bd815884…`) |
| validator_in / validator_out | ✅ оба |
| `embedding_dim` | 2304, L2 mean = **1.0** |

## Эмбеддинги (§1.1)

| проверка | результат |
|---|---|
| `meta.embedding_mode` | `penultimate` на всех people-прогонах |
| `meta.embedding_dim` | **2304** (`clip_embeddings` shape `(C, 2304)`) |
| L2-норма | mean = **1.0** (валидатор pass) |
| `projection_fallback` | **нет** |

## Localization + tubelet (§1.2)

| проверка | результат |
|---|---|
| `mean_clips_per_track > 1` | ❌ **1.0** на 4:35 / 8:00 / group (1 клип на трек) |
| `num_action_segments > 0` | ✅ (= `clip_count` на people-видео) |
| `clip_segment_id` в NPZ | ✅ присутствует |
| tubelet: разные треки → разные top-действия | ✅ |

**4:35** (примеры per-track top): track 3/4 → `playing clarinet`; track 7/8 → `playing guitar`.

**Group `ar_30s_person_b`** (20 треков, 7 уникальных top-label): `playing poker`, `using remote controller`, `texting`, `singing`, `laughing`, `tasting food`, `bowling` — `diverse_top_actions=true`.

## Top-5 dominant actions (Kinetics)

**4:35** (`ar_v3_1_02`):

| id | prob | label |
|---:|---:|---|
| 225 | 0.435 | playing clarinet |
| 233 | 0.172 | playing harmonica |
| 244 | 0.090 | playing saxophone |
| 232 | 0.044 | playing guitar |
| 392 | 0.043 | whistling |

**8:00** (`ar_v3_1_03`):

| id | prob | label |
|---:|---:|---|
| 59 | 0.161 | clean and jerk |
| 88 | 0.105 | deadlifting |
| 398 | 0.090 | yoga |
| 318 | 0.066 | snatch weight lifting |
| 259 | 0.027 | punching person (boxing) |

`classes_available=true`; метки реальные (не `action_<id>`).

## OSNet vs histogram (8:00, опц.)

`ar_v3_1_03_osnet`: в cfg `track_embedder=osnet`, в meta tracking `embedder=osnet`, но **фактически fallback→histogram** — `torchreid` не установлен в venv:

```
OSNetBoxEmbedder требует torch + torchreid … No module named 'torchreid'
```

| метрика | histogram (`ar_v3_1_03`) | osnet run (фактически histogram) |
|---|---:|---:|
| intra_cos_mean | 0.5649 | 0.5649 |
| inter_cos_mean | 0.2912 | 0.2912 |
| n_tracks | 35 | 35 |

Для сравнения osnet нужен `pip install torchreid` в env `core_object_detections`.

## Сравнение v3 → v3.1

| метрика | v3 (4:35) | v3.1 (4:35) | v3 (8:00) | v3.1 (8:00) |
|---|---:|---:|---:|---:|
| clip_count | 35 | **31** | 45 | **43** |
| embedding_dim | 256 (projection) | **2304** (penultimate) | 256 | **2304** |
| mean_clips_per_track | 1.0 | 1.0 | 1.0 | 1.0 |
| num_action_segments | — | 31 | — | 43 |
| det_mean_track_len | 77.5 | 77.5 | 85.7 | 85.7 |
| AR process (meta ms) | ~29 s | **~77 s** | ~30 s | **~75 s** |
| segmenter wall | 186 s | 186 s | 381 s | 289 s |
| visual wall | 229 s | 220 s | 188 s | 218 s |
| frames cache peak | ~10.7 GB | ~10.7 GB | ~10.8 GB | ~10.8 GB |

Tubelet + penultimate 2304-d увеличивают AR process ~2.5×; clip_count чуть ниже (track_anchored сегментация).

## Тайминги / диск

| run | seg wall (s) | visual wall (s) | frames peak (MB) | cod process (manifest ms) | AR process (meta ms) |
|---|---:|---:|---:|---:|---:|
| `ar_v3_1_01` | 5 | 47 | 1148 | ~30500 | ~27 |
| `ar_v3_1_02` | 186 | 220 | 10721 | ~180000 | ~77000 |
| `ar_v3_1_03` | 289 | 218 | 10795 | ~143000 | ~75200 |
| `ar_v3_1_group` | ~15 | ~45 | ~200 | — | — |

**Диск:** без удаления frames после NPZ — **~11 GB/длинное видео**; runner упал на 100% диска. Рекомендация: `cleanup_frames_after_npz.py` + **обязательный** `rm` кэша `artifacts/v3_1/frames/<video_id>` (кадры лежат вне `run_store`, скрипт их не видит).

## Контроль

`ar_v3_1_01`: `empty` / `no_person_detections`, `person_fp_proxy=0`, validator_in ✅ (valid-empty expected), validator_out ✅.

## Артефакты

База: `DataProcessor/docs/component_reports/action_recognition/artifacts/v3_1/`

| файл | описание |
|---|---|
| `run_summary.json` | полная сводка 7 прогонов |
| `golden_compare.json` | golden digest |
| `run_store/youtube/<video>/<run_id>/` | NPZ + manifest + `metrics.{json,prom}` |
| `logs/` | segmenter + visual |
| `run_action_recognition_validation_v3_1.py` | runner |

## Вердикт для REPORT (честно)

| критерий итерации 3 | результат |
|---|---|
| `embedding_mode=penultimate` | ✅ |
| `embedding_dim≈2304` (people) | ✅ |
| эмбеддинги L2 | ✅ |
| классы Kinetics осмысленны | ✅ |
| контроль valid-empty | ✅ |
| golden идентичен | ✅ |
| validator вход + выход | ✅ все прогоны |
| `metrics.json` / `metrics.prom` | ✅ |
| `mean_clips_per_track > 1` | ❌ остаётся **1.0** |
| `num_action_segments > 0` | ✅ |
| tubelet: разные треки → разные действия | ✅ (group + 4:35) |
| osnet intra/inter vs histogram | ⚠️ не измерено (fallback→histogram) |
| диск / стоимость | ⚠️ ~11 GB frames/видео, AR ~2.5× vs v3 |

**Для владельца:** сверить top-действия с роликами; при необходимости multi-clip на трек — отдельная итерация (re-ID между окнами / несколько presence-интервалов на track_id).
