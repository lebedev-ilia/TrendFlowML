# RUN_RESULT: action_recognition v3.2 — итерация 4 (window_len_mult)

Исполнитель: Cursor  
Основание: `RUN_NOTE_FOR_CURSOR.md` (блок «⟳ Итерация 4»)  
Статус: **DONE** (4 прогона: 4:35 + 8:00 + golden ×2)

Первый проход прервался (reboot + CLI `--window-len-mult` пробрасывался в action_recognition). Исправлено: `VisualProcessor/main.py` — `window_len_mult`/`windows_per_min` в exclusions; `Segmenter/segmenter.py` — проброс `window_len_mult` в extractor_configs. Догонены `golden_b` и `ar_v3_2_02`.

## Фикс (косто-нейтральный)

| параметр | v3.1 | v3.2 |
|---|---:|---:|
| `window_len_mult` | 1 (неявно) | **3** |
| кадров в окне | 32 | **96** |
| `max_windows` (cfg) | 48 | 48 → **eff 16** (`max/3`) |
| бюджет кадров AR | 48×32 = **1536** | 16×96 = **1536** |
| `ar_num_windows` (validator_in) | 48 | **16** |
| `ar_num_frames` | 1536 | **1536** |

Segmenter сам берёт `window_len_mult=3` из cfg; компонент скольжением (`clip_len=32`, `stride=16`) даёт **~5 клипов** на трек-присутствие в 96-кадровом окне.

## Профиль (без изменений vs v3.1)

`sampling_policy_version`: **`action_recognition_validation_v3_2`**  
`--embedding-mode penultimate`, `--localization track_anchored`, `--tubelet-crop true`, `--min-clip-real-frames 16`, `--precision fp32`

## Статусы по видео

| video_id | run_id | status | clip_count | num_tracks | mean_clips_per_track | num_action_segments | embedding_dim | validator_in | validator_out | metrics |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| `ar_real_4m35_people` | `ar_v3_2_01` | ok | **44** | 11 | **4.0** | 11 | 2304 | ✅ | ✅ | ✅ |
| `ar_real_8m00_people` | `ar_v3_2_02` | ok | **65** | 16 | **4.0625** | 16 | 2304 | ✅ | ✅ | ✅ |

**Per-track clips (4:35):** min=1, median=**5**, max=**5** — совпадает с ожиданием ~5.

## Golden (4:35 ×2)

| метрика | `ar_v3_2_golden_a` / `ar_v3_2_golden_b` |
|---|---|
| `clip_embeddings` | **идентичны** (digest `63306dee…`) |
| `track_ids` | **идентичны** (digest `61d004ff…`) |
| `mean_clips_per_track` | **4.0** оба |
| validator_in / validator_out | ✅ оба |

## Главный критерий Итерации 4

| критерий | v3.1 | v3.2 | результат |
|---|---:|---:|---|
| `mean_clips_per_track` | 1.0 | **4.0 / 4.06** | ✅ |
| `clip_count` (multi-clip) | 31 / 43 | **44 / 65** | ✅ рост при том же бюджете кадров |
| `num_action_segments` vs tracks | ≈ clip_count (1 clip/track) | **11/11**, **16/16** | ⚠️ segments = change-point id (≈1 на трек); **clip_count >> tracks** (44 vs 11) |

`num_action_segments` считает уникальные `clip_segment_id` (смена действия внутри трека). При одном доминирующем действии на трек segments ≈ tracks; multi-clip эффект виден в **`clip_count` и `mean_clips_per_track`**, не в segments.

## Сравнение v3.1 → v3.2

| метрика | v3.1 (4:35) | v3.2 (4:35) | v3.1 (8:00) | v3.2 (8:00) |
|---|---:|---:|---:|---:|
| `ar_num_frames` | 1536 | 1536 | 1536 | 1536 |
| `ar_num_windows` | 48 | **16** | 48 | **16** |
| clip_count | 31 | **44** | 43 | **65** |
| mean_clips_per_track | 1.0 | **4.0** | 1.0 | **4.06** |
| embedding_dim | 2304 | 2304 | 2304 | 2304 |
| frames peak (MB) | ~10721 | **~10703** | ~10795 | **~10783** |

**Диск/бюджет:** не вырос (~10.7 GB/видео, 1536 AR-кадров). Runner чистит `frames/<video_id>` после NPZ.

## Top-5 dominant actions (Kinetics)

**4:35** (`ar_v3_2_01`):

| id | prob | label |
|---:|---:|---|
| 225 | 0.501 | playing clarinet |
| 233 | 0.133 | playing harmonica |
| 244 | 0.053 | playing saxophone |
| 392 | 0.033 | whistling |
| 221 | 0.031 | playing bass guitar |

**8:00** (`ar_v3_2_02`):

| id | prob | label |
|---:|---:|---|
| 398 | 0.092 | yoga |
| 192 | 0.073 | marching |
| 252 | 0.063 | playing xylophone |
| 140 | 0.061 | giving or receiving award |
| 59 | 0.055 | clean and jerk |

Классы осмысленны (совпадают по семантике с v3.1, top prob чуть сдвинуты из‑за агрегации multi-clip).

## Регрессии (v3.1 baseline)

| проверка | результат |
|---|---|
| `embedding_mode=penultimate` | ✅ |
| `embedding_dim≈2304`, L2=1.0 | ✅ |
| классы Kinetics | ✅ |
| контроль valid-empty | ✅ не перепрогонялся (scope Ит.4); v3.1 `ar_v3_1_01` empty + оба валидатора ✅ |
| golden идентичен | ✅ |
| validator_in + validator_out | ✅ все 4 прогона |
| `metrics.json` / `metrics.prom` | ✅ |

## Тайминги

| run | seg wall (s) | visual wall (s) | AR process (meta ms) |
|---|---:|---:|---:|
| `ar_v3_2_01` | 283 | 211 | ~77000 (≈ v3.1) |
| `ar_v3_2_02` | 316 | 401 | ~75000+ |

AR process ~как v3.1 (больше клипов, но меньше окон SlowFast на входе).

## Артефакты

База: `DataProcessor/docs/component_reports/action_recognition/artifacts/v3_2/`

| файл | описание |
|---|---|
| `run_summary.json` | сводка 4 прогонов |
| `golden_compare.json` | golden digest |
| `run_store/youtube/<video>/<run_id>/` | NPZ + manifest + metrics |
| `logs/` | segmenter + visual |
| `run_action_recognition_validation_v3_2.py` | runner |

## Вердикт для REPORT

| критерий Итерации 4 | результат |
|---|---|
| **`mean_clips_per_track > 1`** | ✅ **4.0 / 4.06** |
| бюджет кадров/диск ≈ v3.1 | ✅ 1536 frames, ~10.7 GB |
| регрессии v3.1 | ✅ |
| golden | ✅ |

**Готово к штампу v3** с точки зрения `mean_clips_per_track`. Для владельца: сверить top-действия на 4:35/8:00 (multi-clip агрегация).
