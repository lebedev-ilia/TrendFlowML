# Audit v4 — журнал прогонов и отчётов

Назначение: фиксировать **какие run’ы** и **какие каталоги `result_store`** использовались для эмпирической статистики (Audit v4), чтобы отчёты можно было воспроизвести.

## Шаблон записи (копировать блок)

```text
### <component_name> — YYYY-MM-DD
- Report level: L1 (draft, набор A) | L2 (A+B) | L3 (A+B+C, полный DoD)
- Status: draft | in_progress | passed | blocked
  - `passed` только при **L3** и закрытом **§8** в `AUDIT_4_CRITERIA_AND_PLAN.md`
- Git commit: <hash>
- Stats tooling: путь к скрипту / команда / ноутбук + **seed** для subsample
- Python / numpy (если нестандартно): <версии>
- E2E log (optional): <path or CI link>
- Reference run (набор A):
  - platform_id/video_id/run_id: ...
  - Artifact dir: `result_store/.../run_id/<component_name>/`
- Diversity set (набор B): (список run_id или «TODO»)
- Edge set (набор C): (список или «TODO»)
- Report: `docs/audit_v4/components/audio_processor/…` или `components/visual_processor/modules|core/…` (или `reports/…`)
- Regression / golden stats (§4.8): ссылка на JSON или «hash сигнатуры» для набора A
- Notes: кратко (например «L2: корреляции на B — приложены в отчёте»)
```

## Записи

### Сквозной E2E (чеклист батча 60+) — 2026-04-16
- Назначение: операционный **повтор опорного A** после фиксов саверов/tabular по **AudioProcessor** (чеклист [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) п. **2.5**); п. **2.6** — выборочно **`spectral_extractor`** в **сегментном** режиме.
- Видео (**A**): `youtube / -Q6fnPIybEI`
- `run_id`: `437dd2f0-a239-424a-ad36-0026f63e094e`
- `manifest.json`: `storage/result_store/youtube/-Q6fnPIybEI/437dd2f0-a239-424a-ad36-0026f63e094e/manifest.json` — **`run.status=success`**; все компоненты с **`kind=audio`** → **`status=ok`** (21 экстрактор); **`text_processor`** → **`ok`**. Строк визуальных компонентов в манифесте **нет** (режим `local_visual_no_triton` / см. п. **2.2** чеклиста).
- **`spectral_extractor` (2.6):** `…/spectral_extractor/spectral_extractor_features.npz` — ось **`segment_*`**: **12** окон, **`segment_mask`**: **10** `True`; tabular: **`hop_length=512`**, **`n_fft=2048`**, **`duration`** ~12.03 s, **`segments_count=12`** (конечные значения, см. аудит по прежним NaN в [spectral_extractor_audit_v4.md](components/audio_processor/spectral_extractor_audit_v4.md)).
- **`meta.features_enabled` (2.7):** **`speech_analysis_extractor`** — в **`meta`** только **`asr_metrics`**, в tabular **8** скаляров **без** pitch-колонок, **`pitch_distribution`** = **{}**. **`pitch_extractor`** — в **`meta`** только **`basic_stats`**, в NPZ **нет** ряда **`f0_series`** (нет флага **`time_series`**). Чеклист: [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) п. **2.7**.
- Артефакты/логи оркестратора: `storage/e2e_full_max/20260416-120234_utc/` (`summary.json`, `e2e_stack_logs/dataprocessor-worker/process.log`).

### VisualProcessor: minimal `core_object_detections` + `action_recognition` — 2026-04-22
- Назначение: **локальная** верификация цепочки для чеклиста [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) п. **2.0.2** (без full-max E2E); воспроизведение `detections.npz` → `action_recognition_features.npz` при том же `PYTORCH_CUDA_ALLOC_CONF` из родительского shell, что и E2E.
- **Root cause (E2E exit 4):** в `action_recognition` подпроцессе `torch` из `.action_recognition_venv` не принимал **`expandable_segments`** в `PYTORCH_CUDA_ALLOC_CONF` (см. `backend/scripts/e2e_env.sh`) — падение на `_cuda_init` до инференса. **Код:** `modules/action_recognition/main.py` — удаление сегмента `expandable_segments` из `PYTORCH_CUDA_ALLOC_CONF` до загрузки SlowFast.
- **Интерпретатор:** `DataProcessor/VisualProcessor/.vp_venv/bin/python` (не системный `python3` без зависимостей).
- **Конфиг:** [`DataProcessor/configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml`](../../configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml) (включает `global.frames_dir` / `global.rs_path`).
- Команда (из `DataProcessor/`): `VisualProcessor/.vp_venv/bin/python VisualProcessor/main.py --cfg-path configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml`
- **Прогон:** `youtube / -Q6fnPIybEI`, `run_id` **`ar_minimal_cli_001`** (из YAML).
- **Артефакты:** `storage/result_store_ar_minimal/youtube/-Q6fnPIybEI/ar_minimal_cli_001/core_object_detections/detections.npz`, `…/action_recognition/action_recognition_features.npz`.
- **Заметка:** это **не** замена B4 (п. **5.1** чеклиста) — путь `CLI → main.py`, не API-очередь; для закрытия **2.0.2** в полном смысле — повтор **full-max E2E** с обновлённым кодом.

### VisualProcessor: minimal `core_object_detections` + `core_face_landmarks` + `micro_emotion` — 2026-04-22
- Назначение: отдельная проверка **`micro_emotion`** (чеклист **2.0.3** — ветка с лицами / сохранение NPZ) без full-max E2E.
- **Конфиг:** [`DataProcessor/configs/audit_v3/visual/visual_minimal_micro_emotion.yaml`](../../configs/audit_v3/visual/visual_minimal_micro_emotion.yaml) — цепочка **YOLO → face landmarks (person-mask) → OpenFace**; интерпретатор: `VisualProcessor/.vp_venv/bin/python`.
- **Прогон:** `youtube / -Q6fnPIybEI`, `run_id` **`me_minimal_cli_001`**.
- **Артефакты:** `storage/result_store_me_minimal/youtube/-Q6fnPIybEI/me_minimal_cli_001/` — `core_object_detections/detections.npz`, `core_face_landmarks/landmarks.npz`, `micro_emotion/micro_emotion.npz`.
- **Лог:** предупреждения OpenFace mapping (invalid rows) — best-effort; **`micro_emotion.npz` записан**, exit 0.

### Отдельные прогоны «empty-компонентов» — 2026-04-22
- **Цель:** увидеть фактический **`meta.status` / `empty_reason`** вне full-max E2E, на одном `video_id` **`-Q6fnPIybEI`**.
- **`car_semantics`** — конфиг [`visual_minimal_car_semantics.yaml`](../../configs/audit_v3/visual/visual_minimal_car_semantics.yaml), `VisualProcessor/.vp_venv`, `run_id` **`car_minimal_cli_001`**. **Результат:** `status=empty`, **`empty_reason=no_car_proposals`**, в meta **`dets_present=0`** (нет валидных car-crop/детекций класса *car* для поиска в embedding DB) — согласуется с [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) **W6**. Артефакт: `storage/result_store_car_minimal/youtube/-Q6fnPIybEI/car_minimal_cli_001/car_semantics/car_semantics.npz`.
- **`place_semantics`** — [`visual_minimal_place_semantics.yaml`](../../configs/audit_v3/visual/visual_minimal_place_semantics.yaml), `run_id` **`place_minimal_cli_001`**. **На этой машине:** `core_clip` не выполнился (**Triton `127.0.0.1:8010` connection refused**), `embeddings.npz` не создан, `place_semantics` упал на отсутствии артефакта — это **инфраструктурный** сбой, а не «пустой каталог мест». Для проверки именно **`no_places_detected`** нужны поднятые **Triton (CLIP)** и **embedding service :8005** (как в E2E).
- **`source_separation_extractor`** — только этот экстрактор, `AudioProcessor/run_cli.py`, `run_id` **`sep_minimal_cli_001`**:  
  `AudioProcessor/.ap_venv/bin/python AudioProcessor/run_cli.py --frames-dir storage/frames_dir/-Q6fnPIybEI --run-rs-path storage/result_store_sep_minimal/youtube/-Q6fnPIybEI/sep_minimal_cli_001 --platform-id youtube --video-id=-Q6fnPIybEI --run-id sep_minimal_cli_001 --extractors source_separation --device cuda`  
  **Результат:** `source_separation_extractor_features.npz` с **`meta.status=ok`** (на семействе `source_separation` в `segments.json` окно **~12 s** — выше порога 5 s; **empty не воспроизведён**). Повторение **`audio_too_short` / `audio_silent`** из E2E — на роликах с короткой/тихой веткой `source_separation` в Segmenter.

### Политика *empty* на full-max E2E (связь с батчем 60+) — 2026-04-22
- Чеклист: [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) п. **2.0.4–2.0.5**; waivers: [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) **W4–W6** (`place_semantics`, `source_separation_extractor`, `car_semantics`).
- Имеется в виду: **`status=empty` с документированным `empty_reason`** (нет матчей / нет car proposals / тихое или слишком короткое аудио) **≠** сбой пайплайна. На пилоте 15/70 сверять NPZ/manifest, а не только бейдж *empty* в heartbeats.

### pitch_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `AudioProcessor/src/extractors/pitch_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON + PNG в `storage/audit_v4/pitch_extractor_l2/`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, ≥5 видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- Edge set (**C**): TODO (≥2 кейса)
- Report: [components/audio_processor/pitch_extractor_audit_v4.md](components/audio_processor/pitch_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/pitch_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/pitch_extractor_engineering_log_v4_2.md)
- Regression / golden stats (§4.8): TODO после повторного прогона A (с чистым `meta.backend`)
- Notes: L2 stats собраны по 5 run из `result_store` (A+B). `backend` строкой хранится в `meta` (не в tabular).

### asr_extractor — 2026-04-07 (Audit 4.2, закрыт **L2**)
- Report level: **L2** (наборы **A + B**; **5** прогонов из `result_store`)
- Status: **in_progress** (не `passed` до **L3** и §8)
- Stats tooling: `DataProcessor/AudioProcessor/src/extractors/asr_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON + PNG в `storage/audit_v4/asr_extractor_l2/`
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c` (зафиксировать актуальный при merge)
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (набор **A**, исторический):
  - `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
  - Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/asr_extractor/asr_extractor_features.npz`
- Diversity set (набор **B**): 5 run (mock e2e), `video_id`: `-Q6fnPIybEI`, `-5EYUqIlyJU`, `-7Ei8e05x30`, `-15jH8mtfJw`, `-Ga4edhrfog` — пути в `asr_extractor_audit_v4_stats.json`
- Edge set (набор **C**): TODO (тишина / empty / короткое аудио)
- Report: [components/audio_processor/asr_extractor_audit_v4.md](components/audio_processor/asr_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md)
- Regression / golden stats (§4.8): TODO (JSON-сигнатура по **A** после договорённости по полям)
- Notes: на B **NaN** в tabular нет; языки **ms**/**en**; `token_total` 6…219; корреляции tabular — heatmap в `figures/`. **Этап 2** (профилирование/ускорение): см. `AUDIT_4_CRITERIA_AND_PLAN.md` §12.2.

### band_energy_extractor — 2026-04-06 (L2: 2026-04-06)
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- **§12.4.4 Gate:** цель **B** — OK; скрипт **A** (JSON+figures); профилирование/оптимизации — отдельной итерацией; см. `AUDIT_4_CRITERIA_AND_PLAN.md` §12.4.2–12.4.4
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/band_energy_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1-путь `4c3bf25b-…` в `result_store` отсутствует)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/band_energy_extractor_l2/band_energy_extractor_audit_v4_stats.json`, `storage/audit_v4/band_energy_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/band_energy_extractor_audit_v4.md](components/audio_processor/band_energy_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/band_energy_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/band_energy_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (tooling готово, без запуска): `AudioProcessor/src/extractors/band_energy_extractor/scripts/audit_v4_npz_stats.py --golden-npz <A.npz> --golden-out storage/audit_v4/band_energy_extractor_l2/golden_A.json --golden-round 8`
- Notes: на **B** **NaN** в tabular **0** на всех пяти файлах; корреляции tabular — `figures/tabular_corr_heatmap.png` (ожидаемая зависимость трёх долей). `meta.duration` = None в L2 артефактах — см. отчёт / engineering log.

### chroma_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/chroma_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/chroma_extractor_l2/chroma_extractor_audit_v4_stats.json`, `storage/audit_v4/chroma_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/chroma_extractor_audit_v4.md](components/audio_processor/chroma_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/chroma_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/chroma_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **A** режим `run_segments` + `time_series`: есть `chroma_mean_by_segment`, ключа `chroma` нет; `chroma_time_series_omitted=false`. Документация — `chroma_extractor/docs/README.md`, `SCHEMA.md`.

### clap_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/clap_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/clap_extractor_l2/clap_extractor_audit_v4_stats.json`, `storage/audit_v4/clap_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/clap_extractor_audit_v4.md](components/audio_processor/clap_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/clap_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/clap_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **A** N=12 окон, без NaN в sequence; порядок tabular = `npz_savers/clap.py`; в meta `device_used` может расходиться с `models_used[].device` — см. отчёт.

### emotion_diarization_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/emotion_diarization_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/emotion_diarization_extractor_l2/emotion_diarization_extractor_audit_v4_stats.json`, `storage/audit_v4/emotion_diarization_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/emotion_diarization_extractor_audit_v4.md](components/audio_processor/emotion_diarization_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/emotion_diarization_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/emotion_diarization_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **A** N=7, C=4; dict-поля в NPZ — 0-dim `object`, читать `.item()`; `rms`/`peak` только в payload, не в NPZ; `meta.model_name`/`weights_digest` на старом артефакте были null — в код добавлено прокидывание в payload для meta.

### key_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/key_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/key_extractor_l2/key_extractor_audit_v4_stats.json`, `storage/audit_v4/key_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/key_extractor_audit_v4.md](components/audio_processor/key_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/key_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/key_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона
- Notes: на старом **A** 5/10 tabular NaN из-за строк в `feature_values`; фикс — `npz_savers/key.py` + meta optional keys в JSON; `chroma_reused=true`; README: исправлена ссылка на схему.

### loudness_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/loudness_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/loudness_extractor_l2/loudness_extractor_audit_v4_stats.json`, `storage/audit_v4/loudness_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/loudness_extractor_audit_v4.md](components/audio_processor/loudness_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/loudness_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/loudness_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **A** tabular **F=18**, NaN **0**; **N=48** (`primary`); `lufs_present=true`; код не менялся.

### hpss_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/hpss_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/hpss_extractor_l2/hpss_extractor_audit_v4_stats.json`, `storage/audit_v4/hpss_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/hpss_extractor_audit_v4.md](components/audio_processor/hpss_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/hpss_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/hpss_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона A (старый NPZ: 17 NaN в tabular, пустые series при `time_series` в meta — см. отчёт; фикс в `hpss_extractor/main.py` + `npz_savers/hpss.py`).

### mel_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/mel_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/mel_extractor_l2/mel_extractor_audit_v4_stats.json`, `storage/audit_v4/mel_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/mel_extractor_audit_v4.md](components/audio_processor/mel_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/mel_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/mel_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона A (старый NPZ: **1 NaN** в tabular по **`device_used`** — строка через `as_float`; фикс: убрано из `npz_savers/mel.py`, `device_used` только в **`meta`**; `docs/README.md`, `docs/SCHEMA.md` обновлены).

### mfcc_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/mfcc_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/mfcc_extractor_l2/mfcc_extractor_audit_v4_stats.json`, `storage/audit_v4/mfcc_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/mfcc_extractor_audit_v4.md](components/audio_processor/mfcc_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/mfcc_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/mfcc_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона A (старый NPZ: **1 NaN** в tabular по **`device_used`**; фикс: `npz_savers/mfcc.py` + fallback в `utils/render.py`; `docs/README.md`, `docs/SCHEMA.md`).

### onset_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/onset_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `video_id` `-15jH8mtfJw` (`run_id` `30e1183d-…`), `-5EYUqIlyJU` (`b9761f4a-…`), `-7Ei8e05x30` (`45c451ad-…`), `-Ga4edhrfog` (`e2dc8851-…`), `-Q6fnPIybEI` (`e2bc964f-…` — совпадает с **A**)
- JSON + figures: `storage/audit_v4/onset_extractor_l2/onset_extractor_audit_v4_stats.json`, `storage/audit_v4/onset_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/onset_extractor_audit_v4.md](components/audio_processor/onset_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/onset_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/onset_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона A (старый NPZ: **1 NaN** в tabular по **`backend`**; фикс: `npz_savers/onset.py`, **`meta.backend`**, `schemas/onset_extractor_npz_v2.json`, `docs/README.md`, `docs/SCHEMA.md`, `utils/render.py`).

### quality_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/quality_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/quality_extractor_l2/quality_extractor_audit_v4_stats.json`, `storage/audit_v4/quality_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/quality_extractor_audit_v4.md](components/audio_processor/quality_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/quality_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/quality_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона A (старый NPZ: **1 NaN** в tabular по **`device_used`**; фикс: `npz_savers/quality.py`; `docs/README.md`, `docs/SCHEMA.md`).

### rhythmic_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): TODO
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/rhythmic_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/rhythmic_extractor_l2/rhythmic_extractor_audit_v4_stats.json`, `storage/audit_v4/rhythmic_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/rhythmic_extractor_audit_v4.md](components/audio_processor/rhythmic_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/rhythmic_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/rhythmic_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **A+B** tabular **F=9**, NaN **0**; `backend` / `device_used` в **meta**; **`duration_sec`** = сумма длительностей окон — задокументировано в `SCHEMA.md` / `README.md`.

### source_separation_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): TODO
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/source_separation_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/source_separation_extractor_l2/source_separation_extractor_audit_v4_stats.json`, `storage/audit_v4/source_separation_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/source_separation_extractor_audit_v4.md](components/audio_processor/source_separation_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/source_separation_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/source_separation_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на 4/5 прогонов `status=empty` (`empty_reason=audio_silent`) → NaN в tabular по долям/доминированию; на **A** (ok) tabular **F=11**, NaN **0**; строки модели / **`device_used`** в **meta**; **N=1** по оси `source_separation`.

### speaker_diarization_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): TODO
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/speaker_diarization_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/speaker_diarization_extractor_l2/speaker_diarization_extractor_audit_v4_stats.json`, `storage/audit_v4/speaker_diarization_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/speaker_diarization_extractor_audit_v4.md](components/audio_processor/speaker_diarization_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/speaker_diarization_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/speaker_diarization_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **A+B** tabular **F=10**, NaN **0**; **`device_used`** / модель в **meta**; **N=1** по `segment_*`; turn-массивы плоские; `SCHEMA.md` / `README.md` — полный список tabular как в савере.

### spectral_entropy_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): TODO
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/spectral_entropy_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/spectral_entropy_extractor_l2/spectral_entropy_extractor_audit_v4_stats.json`, `storage/audit_v4/spectral_entropy_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/spectral_entropy_extractor_audit_v4.md](components/audio_processor/spectral_entropy_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/spectral_entropy_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/spectral_entropy_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **A+B** tabular **F=2**, NaN **0**; **N=12** (`spectral`); `features_enabled`: **`basic_stats`** только; код не менялся; `SCHEMA.md` / `README.md` — Audit v4.

### spectral_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): TODO
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/spectral_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/spectral_extractor_l2/spectral_extractor_audit_v4_stats.json`, `storage/audit_v4/spectral_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/spectral_extractor_audit_v4.md](components/audio_processor/spectral_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/spectral_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/spectral_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона A (старый NPZ: **4 NaN** — `device_used` + отсутствие `hop_length`/`n_fft`/`duration` в payload `run_segments`; фикс: `npz_savers/spectral.py`, `main.py`, `utils/render.py`; `docs/SCHEMA.md`, `README.md`).

### speech_analysis_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): TODO
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/speech_analysis_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/speech_analysis_extractor_l2/speech_analysis_extractor_audit_v4_stats.json`, `storage/audit_v4/speech_analysis_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/speech_analysis_extractor_audit_v4.md](components/audio_processor/speech_analysis_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/speech_analysis_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/speech_analysis_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона A (старый NPZ: **6 NaN** в pitch при **`pitch_enabled=0`** и **`pitch_metrics`** в meta — фикс: `speech_analysis_extractor/main.py`, `_features_enabled`; `docs/SCHEMA.md`, `README.md`).

### tempo_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): TODO
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/tempo_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/tempo_extractor_l2/tempo_extractor_audit_v4_stats.json`, `storage/audit_v4/tempo_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/tempo_extractor_audit_v4.md](components/audio_processor/tempo_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/tempo_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/tempo_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **A+B** tabular **F=11**, NaN **0**; **`device_used`** в **meta**; `SCHEMA.md` / `README.md` — Audit v4.

### voice_quality_extractor — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit (tooling + отчёт): TODO
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/voice_quality_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Набор **B** (5 видео, mock e2e): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…`
- JSON + figures: `storage/audit_v4/voice_quality_extractor_l2/voice_quality_extractor_audit_v4_stats.json`, `storage/audit_v4/voice_quality_extractor_l2/figures/`
- Набор **C**: TODO
- Report: [components/audio_processor/voice_quality_extractor_audit_v4.md](components/audio_processor/voice_quality_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/audio_processor/voice_quality_extractor_engineering_log_v4_2.md](components/audit_4_2/audio_processor/voice_quality_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO после повторного прогона A (старый NPZ: **1 NaN** — **`f0_method`**; фикс: `npz_savers/voice_quality.py`; `docs/SCHEMA.md`, `README.md`).

### action_recognition — 2026-04-13
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/action_recognition/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/action_recognition_l2/action_recognition_audit_v4_stats.json`
- Набор **C**: TODO (`no_person_detections`, короткие треки/клипы)
- Report: [components/visual_processor/modules/action_recognition_audit_v4.md](components/visual_processor/modules/action_recognition_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/action_recognition_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/action_recognition_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (выбрать формат golden; базовые ключи стабильны: `tracks`, `embeddings`, `results_json`, `meta` + 8× `metric__*`)
- Notes: по текущим 5 run (A+B) эмпирика вырождена по временной оси: **`num_clips=1` на всех треках** → метрики переходов/стабильности не проверяют динамику; нужен отдельный B-поднабор с `num_clips>1`.
- **Follow-up 2026-04-15:** пункт чеклиста **2.4** закрыт **письменным waiver** [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) **W3** (критерий «не вырождена ось на пилоте» не выполнен; зафиксировано исключение для батча 60+ до расширения реестра или отдельной проверки).

### behavioral — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/behavioral/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/behavioral_l2/behavioral_audit_v4_stats.json`
- Набор **C**: TODO (нет лиц/очень мало кадров)
- Report: [components/visual_processor/modules/behavioral_audit_v4.md](components/visual_processor/modules/behavioral_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/behavioral_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/behavioral_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на 5 run (A+B) суммарно **N=1250**, `landmarks_present` True **142** (**~11.36%**); NaN в `seq_*` в основном строго привязаны к маске, но вторичные NaN при `landmarks_present=True` для части mouth/pose полей требуют явной политики.

### color_light — 2026-04-13
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/color_light/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/color_light_l2/color_light_audit_v4_stats.json`
- Набор **C**: TODO (`after_filt_empty`, нет `scene_classification`, нет timestamps)
- Report: [components/visual_processor/modules/color_light_audit_v4.md](components/visual_processor/modules/color_light_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/color_light_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/color_light_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на 5 run (A+B) суммарно **M_total=142**, диапазон **M=18…36**; `video_features` стабильно **543** ключа, NaN-ключи стабильны (**7**): `color_distribution_gini`, `nima_*`, `laion_*`, `cinematic_lighting_score`, `professional_look_score`.

### cut_detection — 2026-04-13
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/cut_detection/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/cut_detection_l2/cut_detection_audit_v4_stats.json`
- Набор **C**: TODO (ошибки deps: нет `core_optical_flow`, невалидные timestamps, sampling gaps > cap)
- Report: [components/visual_processor/modules/cut_detection_audit_v4.md](components/visual_processor/modules/cut_detection_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/cut_detection_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/cut_detection_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (учитывать динамические имена файлов; golden фиксировать по `manifest.json` / JSON stats)
- Notes: на 5 run (A+B) суммарно **N_total=543**, **pairs_total=538**, **E_total=53**; `deep_valid_ratio=0` (ветка deep не активна), `ssim_valid_ratio_mean≈0.254`, `flow_valid_ratio=1.0`.

### detalize_face — 2026-04-13
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/detalize_face/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/detalize_face_l2/detalize_face_audit_v4_stats.json`
- Набор **C**: TODO (`no_faces_in_video` → `status=empty`, частичное покрытие core_face_landmarks vs axis)
- Report: [components/visual_processor/modules/detalize_face_audit_v4.md](components/visual_processor/modules/detalize_face_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/detalize_face_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/detalize_face_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на 5 run (A+B) суммарно **N_total=1250**, `primary_valid` True **73** (**~5.84%**) → `compact_zero_row_ratio≈94.16%`; опциональные `primary_*` curves отсутствуют на всех (write_primary_curves=false).

### emotion_face — 2026-04-13
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/emotion_face/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/emotion_face_l2/emotion_face_audit_v4_stats.json`
- Набор **C**: TODO (`no_faces_in_video` → `status=empty`, тайминг/ось, low-quality gating)
- Report: [components/visual_processor/modules/emotion_face_audit_v4.md](components/visual_processor/modules/emotion_face_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/emotion_face_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/emotion_face_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на 5 run (A+B) суммарно **N_total=1000**, `face_present` True **42** (**4.2%**), `processed_mask` True **12** (**1.2%**); `keyframes_total=0` на всех 5; `dominant_emotion_id=-1` на **988/1000** кадров (вне processed_mask).

### frames_composition — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/frames_composition/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/frames_composition_l2/frames_composition_audit_v4_stats.json`
- Набор **C**: TODO
- Report: [components/visual_processor/modules/frames_composition_audit_v4.md](components/visual_processor/modules/frames_composition_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/frames_composition_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/frames_composition_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на 5 run (A+B) суммарно **N_total=543**, **D=32**, **F=217**; `present_ratio` совпадает с долей finite (max_abs_diff ~2e-8); video-level корреляции на B подтверждают ожидаемую избыточность (например, `negative_space_ratio` vs `object_bbox_coverage_ratio`).

### high_level_semantic — 2026-04-13
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/high_level_semantic/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/high_level_semantic_l2/high_level_semantic_audit_v4_stats.json`
- Набор **C**: TODO (кейсы с включённым `require_*` для audio/text + отсутствие `cut_detection_model_facing`)
- Report: [components/visual_processor/modules/high_level_semantic_audit_v4.md](components/visual_processor/modules/high_level_semantic_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/high_level_semantic_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/high_level_semantic_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на 5 run (A+B) суммарно **N_total=543**, **D=512**, **F=8**; **S** варьирует **2…8**; `loudness_dbfs/tempo_bpm` — 100% NaN на всех 5; на части run `T=0` (нет `text_processor` артефакта), поэтому корреляции `text_feature_*` требуют отдельного B/C.

### micro_emotion — 2026-04-13
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **blocked** (не хватает 5-го OK артефакта: один run в B завершился `status=error`)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/micro_emotion/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, целевые 4 доп. видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats (OK артефакты): `storage/audit_v4/micro_emotion_l2/micro_emotion_audit_v4_stats.json` (**4** NPZ; `-Ga4edhrfog` отсутствует)
- B-run error (нет NPZ): `youtube / -Ga4edhrfog / e2dc8851-6c51-43c0-9757-3c0fed803348` → `micro_emotion` в `manifest.json`: `status=error`, причина: `n_components=3 must be between 0 and min(n_samples, n_features)=2`
- **Follow-up 2026-04-15:** в дереве кода `compute_au_pca` ограничивает `n_components` и дополняет матрицу до фиксированной ширины — регрессия sklearn для вырожденного `n_samples` не ожидается. Проверка: `VisualProcessor/.vp_venv/bin/python`, синтетический DF из **2** строк с колонками `AU##_r`, `pca_components=3` — без исключения, фактически `PCA(n_components=2)` + padding. Повторный полный прогон `-Ga4edhrfog` для обновления L2 JSON не выполнялся.
- Набор **C**: TODO
- Report: [components/visual_processor/modules/micro_emotion_audit_v4.md](components/visual_processor/modules/micro_emotion_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/micro_emotion_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/micro_emotion_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: по 4 OK run суммарно **N_total=1000**, `face_present_any` True **70** (**7%**), `K_total=2` micro-events; video-vector **V=75**, NaN_total=**16**; пары корреляций (top) в JSON.

### optical_flow — 2026-04-13
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)**
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/optical_flow/scripts/audit_v4_npz_stats.py`
- Reference run (**A**): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/optical_flow_l2/optical_flow_audit_v4_stats.json`
- Набор **C**: TODO
- Report: [components/visual_processor/modules/optical_flow_audit_v4.md](components/visual_processor/modules/optical_flow_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/optical_flow_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/optical_flow_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на 5 run (A+B) суммарно **N_total=1250**, `missing_ratio_curve_mean≈0.886`, `missing_ratio_matrix_mean≈0.890`; разница долей NaN между кривой и матрицей стабильно мала (~0.00375) и объясняется тем, что матрица учитывает все 16 колонок.

### scene_classification — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/scene_classification/scripts/audit_v4_npz_stats.py`
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/scene_classification_l2/scene_classification_audit_v4_stats.json`
- Набор **C**: TODO (`status=empty`, экстремально короткие/длинные сцены, edge-case по `cut_detection`)
- Report: [components/visual_processor/modules/scene_classification_audit_v4.md](components/visual_processor/modules/scene_classification_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/scene_classification_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/scene_classification_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (зафиксировать сигнатуру A: N/S + диапазоны `frame_topk_probs` сумм + ключи/NaN)
- Notes: на 5 run (A+B) суммарно **N_total=543**; `label_fusion=places` на всех; **S** варьирует **2…7**; сумма top-5 по кадру **не 1** (глобально min≈**0.186**, max≈**0.999996**) — это expected «срез» распределения.

### shot_quality — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/shot_quality/scripts/audit_v4_npz_stats.py`
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/shot_quality_l2/shot_quality_audit_v4_stats.json`
- Набор **C**: TODO (`status=empty` deps, экстремальные NaN по face-ROI, edge-case по `cut_detection`)
- Report: [components/visual_processor/modules/shot_quality_audit_v4.md](components/visual_processor/modules/shot_quality_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/shot_quality_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/shot_quality_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (зафиксировать сигнатуру A: N/S/F/P/K + стабильные all-NaN фичи + диапазоны сумм probs)
- Notes: на 5 run (A+B) суммарно **N_total=543**; `quality_probs` сумма по строке стабильно **~1** (min≈**0.9998169**, max≈**1.000122**); shot top‑K сумма **не 1** (min≈**0.3037**, max≈**0.3084**); **union fully‑NaN features** стабилен (**4**): `vignetting_level`, `chromatic_aberration_level`, `lens_sharpness_drop_off`, `rolling_shutter_artifacts_score`.

### similarity_metrics — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/similarity_metrics/scripts/audit_v4_npz_stats.py`
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/similarity_metrics_l2/similarity_metrics_audit_v4_stats.json`
- Набор **C**: TODO (кейс с `reference_present=True`, + проверка optional modality present flags)
- Report: [components/visual_processor/modules/similarity_metrics_audit_v4.md](components/visual_processor/modules/similarity_metrics_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/similarity_metrics_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/similarity_metrics_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (зафиксировать сигнатуру A: N/F + диапазоны centroid/temporal + долю finite в feature_values)
- Notes: на 5 run (A+B) суммарно **N_total=543**, **F=39**; `reference_present=False` на всех 5; `feature_values` finite **75/195** (то есть ровно **15/39** на каждый run) — ожидаемо без reference.

### story_structure — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/story_structure/scripts/audit_v4_npz_stats.py`
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/story_structure_l2/story_structure_audit_v4_stats.json`
- Набор **C**: TODO (кейс с `topic_shift_curve_present=True`, стабильные пики/климакс на длинном видео)
- Report: [components/visual_processor/modules/story_structure_audit_v4.md](components/visual_processor/modules/story_structure_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/story_structure_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/story_structure_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (зафиксировать сигнатуру A: N/F/P + диапазоны кривых + экстремумы `hook_to_avg_energy_ratio`)
- Notes: по 5 run (A+B) суммарно **N_total=467**, **F=22**; `topic_shift_curve_present=False` на всех 5 (T_set=[0]); пики story-energy **P** варьируют **2…5**; `hook_to_avg_energy_ratio` на A+B может быть экстремальным (min≈**−8.6e5**, max≈**6.9e5**) — это требует политики в downstream (клиппинг/robust transforms).

### text_scoring — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/text_scoring/scripts/audit_v4_npz_stats.py`
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/text_scoring_l2/text_scoring_audit_v4_stats.json`
- Набор **C**: TODO (`status=empty` при отсутствии OCR после фильтра; кейсы `store_debug_objects=true` + включённые peaks/entropy/speed)
- Report: [components/visual_processor/modules/text_scoring_audit_v4.md](components/visual_processor/modules/text_scoring_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/text_scoring_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/text_scoring_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (зафиксировать сигнатуру A: N/F + текстовые счётчики + NaN-профиль feature_values)
- Notes: по 5 run (A+B) суммарно **N_total=600**, **F=35**; `text_present=True` на всех; `text_presence` ratio варьирует **0.025…0.1333**, `text_count_sum_total=77`. `feature_values` NaN **50/175** (то есть **10/35** на каждый run), `ocr_raw`/`ocr_unique_elements` на всех 5 пустые (privacy defaults).

### uniqueness — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/uniqueness/scripts/audit_v4_npz_stats.py`
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/uniqueness_l2/uniqueness_audit_v4_stats.json`
- Набор **C**: TODO (edge по max_frames, видео с низкой повторяемостью)
- Report: [components/visual_processor/modules/uniqueness_audit_v4.md](components/visual_processor/modules/uniqueness_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/uniqueness_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/uniqueness_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (зафиксировать сигнатуру A: N/F + threshold used + repetition/diversity)
- Notes: по 5 run (A+B) суммарно **N_total=467**, **F=20**; `repeat_threshold_is_otsu=1` на всех; `repeat_threshold_used` в диапазоне **~0.90…0.973**; `repetition_ratio` варьирует **~0.792…0.967** (зависит от контента).

### video_pacing — 2026-04-06
- Report level: **L2** (product stats, **A+B**; **C** и §8 — не закрыты)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Stats tooling: `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `VisualProcessor/modules/video_pacing/scripts/audit_v4_npz_stats.py`
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/video_pacing_l2/video_pacing_audit_v4_stats.json`
- Набор **C**: TODO (кейсы с включёнными entropy/histograms/peaks/periodicity/bursts, и edge по shot boundaries)
- Report: [components/visual_processor/modules/video_pacing_audit_v4.md](components/visual_processor/modules/video_pacing_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/modules/video_pacing_engineering_log_v4_2.md](components/audit_4_2/visual_processor/modules/video_pacing_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO (зафиксировать сигнатуру A: N/S/F + finite профиль feature_values + диапазоны кривых)
- Notes: по 5 run (A+B) суммарно **N_total=467**, **F=57**; shot boundaries **S** варьирует **3…9**; `feature_values` finite **218/285** (в среднем **~44/57** на run), что согласуется с выключенными optional блоками (NaN-поля стабильны по флагам).

### core_clip — 2026-04-06
- Report level: **L2** (VisualProcessor core; наборы **A + B**; **5** прогонов из `result_store`)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: TODO
- Stats tooling: `DataProcessor/VisualProcessor/core/model_process/core_clip/scripts/audit_v4_npz_stats.py` → `storage/audit_v4/core_clip_l2/core_clip_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/core_clip_l2/core_clip_audit_v4_stats.json`
- Набор **C**: TODO
- Report: [components/visual_processor/core/core_clip_audit_v4.md](components/visual_processor/core/core_clip_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/core/core_clip_engineering_log_v4_2.md](components/audit_4_2/visual_processor/core/core_clip_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: по 5 run (A+B) суммарно **N_total=543**, **D=512**, **K=5**; L2 нормы `frame_embeddings` **≈1**; `consecutive_cosine_prev` NaN строго **idx 0** на каждом run (**5** всего). Суммы `shot_quality_scores` и `places365_topk_scores` по строкам **не 1** (это ожидаемые similarity/logit‑скоры и top‑K срез).

### core_depth_midas — 2026-04-06
- Report level: **L2** (VisualProcessor core; наборы **A + B**; **5** прогонов из `result_store`)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: TODO
- Stats tooling: `DataProcessor/VisualProcessor/core/model_process/core_depth_midas/scripts/audit_v4_npz_stats.py` → `storage/audit_v4/core_depth_midas_l2/core_depth_midas_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/core_depth_midas_l2/core_depth_midas_audit_v4_stats.json`
- Набор **C**: TODO
- Report: [components/visual_processor/core/core_depth_midas_audit_v4.md](components/visual_processor/core/core_depth_midas_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/core/core_depth_midas_engineering_log_v4_2.md](components/audit_4_2/visual_processor/core/core_depth_midas_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: по 5 run (A+B) суммарно **N_total=543**, **H=W=256**, **K=10**; `depth_maps_norm` строго **[0,1]**, NaN/Inf **0**; `preview_frame_indices ⊆ frame_indices` на всех 5.

### core_face_landmarks — 2026-04-06
- Report level: **L2** (VisualProcessor core; наборы **A + B**; **5** прогонов из `result_store`)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: TODO
- Stats tooling: `DataProcessor/VisualProcessor/core/model_process/core_face_landmarks/scripts/audit_v4_npz_stats.py` → `storage/audit_v4/core_face_landmarks_l2/core_face_landmarks_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/core_face_landmarks_l2/core_face_landmarks_audit_v4_stats.json`
- Набор **C**: TODO (кейсы `no_person_detections`, **FACES>1**)
- Report: [components/visual_processor/core/core_face_landmarks_audit_v4.md](components/visual_processor/core/core_face_landmarks_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/core/core_face_landmarks_engineering_log_v4_2.md](components/audit_4_2/visual_processor/core/core_face_landmarks_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: по 5 run (A+B) суммарно **N_total=543**, `FACES=1`, `HANDS=2`; NaN‑политика соблюдена: «absent ⇒ all‑NaN», «present ⇒ no‑NaN» для face/hands слотов (violations=0). `face_mesh_ran ∧ ¬face_present` встречается (гейтинг по person‑mask, но face не найден).

### core_object_detections — 2026-04-06
- Report level: **L2** (VisualProcessor core; наборы **A + B**; **5** прогонов из `result_store`)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: TODO
- Stats tooling: `DataProcessor/VisualProcessor/core/model_process/core_object_detections/scripts/audit_v4_npz_stats.py` → `storage/audit_v4/core_object_detections_l2/core_object_detections_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/core_object_detections_l2/core_object_detections_audit_v4_stats.json`
- Набор **C**: TODO
- Report: [components/visual_processor/core/core_object_detections_audit_v4.md](components/visual_processor/core/core_object_detections_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/core/core_object_detections_engineering_log_v4_2.md](components/audit_4_2/visual_processor/core/core_object_detections_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: по 5 run (A+B) суммарно **N_total=543**, **M=100**, `class_names_len=41`; `det_count_matches_mask_all=true`. Эмпирически разделение по threshold ~0.6 наблюдается, но downstream должен читать только `valid_mask` (слоты padding могут иметь ненулевые score).

### core_optical_flow — 2026-04-06
- Report level: **L2** (VisualProcessor core; наборы **A + B**; **5** прогонов из `result_store`)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: TODO
- Stats tooling: `DataProcessor/VisualProcessor/core/model_process/core_optical_flow/scripts/audit_v4_npz_stats.py` → `storage/audit_v4/core_optical_flow_l2/core_optical_flow_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/core_optical_flow_l2/core_optical_flow_audit_v4_stats.json`
- Набор **C**: TODO
- Report: [components/visual_processor/core/core_optical_flow_audit_v4.md](components/visual_processor/core/core_optical_flow_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/core/core_optical_flow_engineering_log_v4_2.md](components/audit_4_2/visual_processor/core/core_optical_flow_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: по 5 run (A+B) суммарно **N_total=543**, **K=10**, preview map **64×64**; `frame_indices` строго возрастают на всех 5; NaN‑политика: flow‑зависимые ряды NaN только на idx 0 (`flow_dep_nan_at_0_only_all=true`), `motion_norm_per_sec_mean[0]=0` и без NaN.

### ocr_extractor — 2026-04-06
- Report level: **L2** (VisualProcessor core; наборы **A + B**; **5** прогонов из `result_store`)
- Status: **in_progress (v4 L2)** (не `passed` до **L3** и §8)
- Git commit: TODO
- Stats tooling: `DataProcessor/VisualProcessor/core/model_process/ocr_extractor/scripts/audit_v4_npz_stats.py` → `storage/audit_v4/ocr_extractor_l2/ocr_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`)
- Diversity set (**B**, 4 дополнительных видео): `-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2`, `-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53`, `-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45`, `-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348`
- JSON stats: `storage/audit_v4/ocr_extractor_l2/ocr_extractor_audit_v4_stats.json`
- Набор **C**: TODO
- Report: [components/visual_processor/core/ocr_extractor_audit_v4.md](components/visual_processor/core/ocr_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/visual_processor/core/ocr_extractor_engineering_log_v4_2.md](components/audit_4_2/visual_processor/core/ocr_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: по 5 run (A+B) суммарно **N_total=543**, **R_total=776**; `engine=ppocr_rec_onnx` на всех 5; privacy defaults: `retain_raw_ocr_text=false`, сырого текста нет (`raw_text_keys_present_any=false`); привязка строк: `frame` ⊆ `frame_indices` на всех 5.

### asr_text_proxy_audio_features (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез `tp_asrproxy_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error` на том же B-наборе видео)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/asr_text_proxy_audio_features/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/asr_text_proxy_audio_features_l2/asr_text_proxy_audio_features_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190` — `text_processor/text_features.npz` с **37** `tp_asrproxy_*`
- Diversity set (**B**,4 доп. видео, те же `run_id`, что у Visual L2): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (пустой ASR, `require_asr_text`, token-only / decode fail)
- Report: [components/text_processor/asr_text_proxy_audio_features_audit_v4.md](components/text_processor/asr_text_proxy_audio_features_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/asr_text_proxy_audio_features_engineering_log_v4_2.md](components/audit_4_2/text_processor/asr_text_proxy_audio_features_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах `tp_asrproxy_*` **37** ключей, **0** NaN внутри среза; на **3** error-NPZ `feature_names` пустой — падение пайплайна **до** заполнения таблицы (типично: **TitleEmbedder** / CUDA OOM на `intfloat/multilingual-e5-large`, см. `meta.error` в JSON `per_file`). Для закрытия L2: перепрогон `text_processor` на B с CPU/меньшей моделью или освобождённым GPU.

### comments_embedder (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез `tp_commentsemb_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/comments_embedder/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/comments_embedder_l2/comments_embedder_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (нет comments, `compute_embeddings=false`, cache on/off, CPU vs CUDA)
- Report: [components/text_processor/comments_embedder_audit_v4.md](components/text_processor/comments_embedder_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/comments_embedder_engineering_log_v4_2.md](components/audit_4_2/text_processor/comments_embedder_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах `tp_commentsemb_*` **18** ключей; `comments_embeddings.npy` существует; `tp_commentsemb_artifact_written` = NaN при `emit_extra_metrics=false` (ожидаемо). На **3** error-NPZ `feature_names` пустой, `.npy` отсутствует — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### comments_aggregator (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/comments_aggregator/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/comments_aggregator_l2/comments_aggregator_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (нет эмбеддингов, `require_comment_embeddings=true`, неверный relpath)
- Report: [components/text_processor/comments_aggregator_audit_v4.md](components/text_processor/comments_aggregator_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/comments_aggregator_engineering_log_v4_2.md](components/audit_4_2/text_processor/comments_aggregator_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (39 ключей) и присутствуют `comments_agg_mean.npy`/`comments_agg_median.npy`/`comments_selected_indices.npy`. На **3** error-NPZ `feature_names` пустой и артефакты отсутствуют — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### description_embedder (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез `tp_descemb_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/description_embedder/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/description_embedder_l2/description_embedder_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (пустое описание, `compute_embedding=false`, cache on/off, CPU vs CUDA)
- Report: [components/text_processor/description_embedder_audit_v4.md](components/text_processor/description_embedder_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/description_embedder_engineering_log_v4_2.md](components/audit_4_2/text_processor/description_embedder_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (19 ключей) и присутствует `description_embedding.npy`; на **3** error-NPZ `feature_names` пустой и артефакт отсутствует — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### cosine_metrics_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез `tp_cos_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/cosine_metrics_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/cosine_metrics_extractor_l2/cosine_metrics_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (matrix-режим, require_* fail-fast, dim mismatch, смена transcript_source_priority)
- Report: [components/text_processor/cosine_metrics_extractor_audit_v4.md](components/text_processor/cosine_metrics_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/cosine_metrics_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/cosine_metrics_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (39 ключей); на **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### embedding_pair_topk_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез `tp_embpair_*`/`tp_pairtopk_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/embedding_pair_topk_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/embedding_pair_topk_extractor_l2/embedding_pair_topk_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (M чанков ≥ top_k_slots, FAISS/auto, dim mismatch, пустые эмбеддинги)
- Report: [components/text_processor/embedding_pair_topk_extractor_audit_v4.md](components/text_processor/embedding_pair_topk_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/embedding_pair_topk_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/embedding_pair_topk_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (56 canon + 13 legacy = 69 ключей); на **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### embedding_shift_indicator_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез `tp_embshift_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/embedding_shift_indicator_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/embedding_shift_indicator_extractor_l2/embedding_shift_indicator_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (≥2 чанка, `compute_extra_cosines=true`, require_transcript_chunks=true + отсутствующий файл)
- Report: [components/text_processor/embedding_shift_indicator_extractor_audit_v4.md](components/text_processor/embedding_shift_indicator_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/embedding_shift_indicator_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/embedding_shift_indicator_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (27 ключей); на **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### embedding_source_id_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_embid_*` + nested `embedding_source_id` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/embedding_source_id_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/embedding_source_id_extractor_l2/embedding_source_id_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (`title_first`, `strict_missing_primary`, пустой `tp_artifacts`)
- Report: [components/text_processor/embedding_source_id_extractor_audit_v4.md](components/text_processor/embedding_source_id_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/embedding_source_id_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/embedding_source_id_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (13 ключей) + nested `payload["embedding_source_id"]` + успешная сверка `vector_id`; на **3** error-NPZ `feature_names` пустой и nested отсутствует — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### embedding_stats_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез `tp_embstats_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/embedding_stats_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/embedding_stats_extractor_l2/embedding_stats_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (≥2 чанка + variance path, `emit_extra_metrics=true`, topic missing/invalid + require_topic_distribution=true)
- Report: [components/text_processor/embedding_stats_extractor_audit_v4.md](components/text_processor/embedding_stats_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/embedding_stats_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/embedding_stats_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (39 ключей); на **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### hashtag_embedder (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_hashemb_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/hashtag_embedder/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/hashtag_embedder_l2/hashtag_embedder_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (пустые хештеги, `require_hashtags=true`, `write_artifact=false`, `use_frequencies=true`, `aggregation` ≠ mean, cache hit)
- Report: [components/text_processor/hashtag_embedder_audit_v4.md](components/text_processor/hashtag_embedder_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/hashtag_embedder_engineering_log_v4_2.md](components/audit_4_2/text_processor/hashtag_embedder_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (23 ключа) и присутствует `hashtag_embedding.npy` (**1024**, `float32`, L2≈1); на **3** error-NPZ `feature_names` пустой и артефакт отсутствует — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### lexico_static_features (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** срез `tp_lex_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/lexico_static_features/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/lexico_static_features_l2/lexico_static_features_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (legacy transcript, `require_transcript=true`, выключенные группы, пустые поля, усечение `max_*_chars`, `emoji_policy=required` без пакета)
- Report: [components/text_processor/lexico_static_features_audit_v4.md](components/text_processor/lexico_static_features_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/lexico_static_features_engineering_log_v4_2.md](components/audit_4_2/text_processor/lexico_static_features_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (67 ключей); на **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### qa_embedding_pairs_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_qa_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/qa_embedding_pairs_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/qa_embedding_pairs_extractor_l2/qa_embedding_pairs_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759` (на этом run `tp_qa_present=1`, `num_questions=2`, есть `qa_question_embeddings.npy`)
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190` (валидный пустой исход, `present=0`)
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (`require_min_questions>0` fail-fast, `emit_extra_metrics=true`, опциональные `qa_question_hashes.npy`/`qa_question_source_ids.npy`)
- Report: [components/text_processor/qa_embedding_pairs_extractor_audit_v4.md](components/text_processor/qa_embedding_pairs_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/qa_embedding_pairs_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/qa_embedding_pairs_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (34 ключа); на одном OK-run `present=1` и есть `qa_question_embeddings.npy` (shape согласован с `num_questions`/`dim`), на одном OK-run валидный пустой исход `present=0` без артефакта; на **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### semantic_cluster_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_semclust_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/semantic_cluster_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/semantic_cluster_extractor_l2/semantic_cluster_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (fallback на другой слот, `require_faiss=true` при отсутствии FAISS, пропуски эмбеддингов / dim mismatch, `emit_extra_metrics=true`)
- Report: [components/text_processor/semantic_cluster_extractor_audit_v4.md](components/text_processor/semantic_cluster_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/semantic_cluster_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/semantic_cluster_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (31 ключ); на **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### semantics_topics_keyphrases (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_topics_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/semantics_topics_keyphrases/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/semantics_topics_keyphrases_l2/semantics_topics_keyphrases_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (`export_keyphrases_mode=hashed` заполняет `kp_top*`, `emit_extra_metrics=true`, пустой текст, режим `raw`, выключенные keyphrases/embeddings)
- Report: [components/text_processor/semantics_topics_keyphrases_audit_v4.md](components/text_processor/semantics_topics_keyphrases_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/semantics_topics_keyphrases_engineering_log_v4_2.md](components/audit_4_2/text_processor/semantics_topics_keyphrases_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (116 ключей) и валидный `tp_topics_keyphrase_embeddings.npy` (**(10,1024)**); на **3** error-NPZ `feature_names` пустой, но `.npy` может присутствовать как частичный выход — успех определять по `meta.status` и наличию tabular slice (см. `text_processor_error` в JSON).

### speaker_turn_embeddings_aggregator (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_spkemb_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/speaker_turn_embeddings_aggregator_l2/speaker_turn_embeddings_aggregator_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (diar+ASR с таймингами, legacy `doc.speakers`, `require_input=true` при пустоте, `emit_extra_metrics=true`)
- Report: [components/text_processor/speaker_turn_embeddings_aggregator_audit_v4.md](components/text_processor/speaker_turn_embeddings_aggregator_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/speaker_turn_embeddings_aggregator_engineering_log_v4_2.md](components/audit_4_2/text_processor/speaker_turn_embeddings_aggregator_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный срез (17 ключей); на этих run `tp_spkemb_present=0` и `speaker_spk*.npy` отсутствуют (валидный пустой исход). На **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON). Для содержательного L2 по компоненту нужен B-run с diar+ASR (таймкоды) или legacy speakers.

### tags_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_tags_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/tags_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/tags_extractor_l2/tags_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (`export_hashtags_mode_raw/hashed`, `merge_json_hashtags`, усечения, `require_title`, пустые поля, `enable_extract_hashtags=false`)
- Report: [components/text_processor/tags_extractor_audit_v4.md](components/text_processor/tags_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/tags_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/tags_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK файлах полный набор `tp_tags_*` для `top_k_slots=5` (**43** ключа = 28 базовых + 15 slot‑ключей top1..5). На **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### title_embedder (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_titleemb_*` + `title_embedding.npy` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/title_embedder/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/title_embedder_l2/title_embedder_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (`require_title`, `compute_embedding=false`, CPU + cache hit, batch encode)
- Report: [components/text_processor/title_embedder_audit_v4.md](components/text_processor/title_embedder_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/title_embedder_engineering_log_v4_2.md](components/audit_4_2/text_processor/title_embedder_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK run `tp_titleemb_*` содержит **16** ключей, `title_embedding.npy` есть (**(1024,)** float32, \(L2\approx1\), `dim` согласован). На **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### title_embedding_cluster_entropy_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_titleclent_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/title_embedding_cluster_entropy_extractor_l2/title_embedding_cluster_entropy_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (`emit_extra_metrics=true`, `export_topk_distribution=true`, `top_k_slots>8` (clamp), `require_faiss=true` без FAISS, пропажа relpath/файла, dim_mismatch)
- Report: [components/text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md](components/text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/title_embedding_cluster_entropy_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/title_embedding_cluster_entropy_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK run `tp_titleclent_*` содержит **24** ключа; upstream `title_embedding.npy` присутствует (**(1024,)**). На **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### title_to_hashtag_cosine_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_titlehashcos_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/title_to_hashtag_cosine_extractor_l2/title_to_hashtag_cosine_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (только title или только hashtag → `present=0` + cosine NaN, `unsafe` relpath, `dim_mismatch`, `zero norm`, `require_*` fail-fast)
- Report: [components/text_processor/title_to_hashtag_cosine_extractor_audit_v4.md](components/text_processor/title_to_hashtag_cosine_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/title_to_hashtag_cosine_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/title_to_hashtag_cosine_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK run `tp_titlehashcos_*` содержит **11** ключей; upstream `title_embedding.npy` и `hashtag_embedding.npy` присутствуют (**(1024,)**). Для `present=1` cosine, пересчитанный из `.npy`, совпадает с `tp_titlehashcos_cosine` (см. `consistency.abs_diff` в JSON). На **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### topk_similar_titles_extractor (TextProcessor) — 2026-04-14
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_topktitles_*` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/topk_similar_titles_extractor/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/topk_similar_titles_extractor_l2/topk_similar_titles_extractor_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Edge set (**C**): TODO (`enabled=false`, `export_topk_mode=none/ids_only`, FAISS on/off, большой корпус без FAISS + policy, missing title emb / dim mismatch / NaN/Inf query)
- Report: [components/text_processor/topk_similar_titles_extractor_audit_v4.md](components/text_processor/topk_similar_titles_extractor_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/topk_similar_titles_extractor_engineering_log_v4_2.md](components/audit_4_2/text_processor/topk_similar_titles_extractor_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK run `tp_topktitles_*` содержит **29** ключей; upstream `title_embedding.npy` присутствует (**(1024,)**). На **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### transcript_chunk_embedder (TextProcessor) — 2026-04-06
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_tchunk_*` + `.npy` артефакт только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/transcript_chunk_embedder/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/transcript_chunk_embedder_l2/transcript_chunk_embedder_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Матрица: `…/_artifacts/transcript_whisper_chunk_embeddings.npy` — **(1, 1024)** на OK subset
- Render: `…/text_processor/_render/transcript_chunk_embedder_report.html`
- Наборы B / C: TODO (youtube_auto, multi-chunk, `emit_extra_metrics=true`, disk cache; require_asr/empty transcript)
- Report: [components/text_processor/transcript_chunk_embedder_audit_v4.md](components/text_processor/transcript_chunk_embedder_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/transcript_chunk_embedder_engineering_log_v4_2.md](components/audit_4_2/text_processor/transcript_chunk_embedder_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK run `tp_tchunk_*` содержит **16** ключей; `transcript_whisper_chunk_embeddings.npy` присутствует (**(1, 1024)** float32), L2 нормы строк ~1, `embedding_dim` согласован. На **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### transcript_aggregator (TextProcessor) — 2026-04-15
- Report level: **L2** (целевые наборы **A+B**, **5** путей из `result_store`; **фактически** `tp_tragg_*` + ожидаемые `.npy` только на **2** OK `text_processor` — см. JSON `dataset_quality`)
- Status: **blocked** для полного L2 (нужны **5** успешных `text_processor`; сейчас **3** run с `meta.status=error`)
- Git commit: TODO
- Stats tooling: `DataProcessor/TextProcessor/src/extractors/transcript_aggregator/scripts/audit_v4_npz_stats.py` — `--seed 0`, JSON в `storage/audit_v4/transcript_aggregator_l2/transcript_aggregator_audit_v4_stats.json`
- Python / numpy: `DataProcessor/.data_venv/bin/python`, numpy **2.2.6**
- Reference run (**A**, воспроизводимый, L2-строка в JSON): `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Исторический L1-run (старый отчёт): `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`
- Diversity set (**B**,4 доп. видео): `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`
- Артефакт (tabular): `…/text_processor/text_features.npz` — **19** имён `tp_tragg_*`; схема **`transcript_aggregator_output_v1`**, **`allow_extra_keys: false`**
- Агрегаты (по флагам в `tp_tragg_*`): `…/_artifacts/transcript_{whisper|youtube_auto}_agg_{mean,max}.npy`, `transcript_combined_agg_{mean,max}.npy` (на OK subset **youtube_auto** нет → только whisper + combined)
- Render: `…/text_processor/_render/transcript_aggregator_report.html`
- Edge set (**C**): TODO (youtube_auto + combined, `emit_extra_metrics=true`, `compute_std=true`, `write_artifacts=false`, missing chunks)
- Report: [components/text_processor/transcript_aggregator_audit_v4.md](components/text_processor/transcript_aggregator_audit_v4.md)
- Engineering log 4.2: [components/audit_4_2/text_processor/transcript_aggregator_engineering_log_v4_2.md](components/audit_4_2/text_processor/transcript_aggregator_engineering_log_v4_2.md)
- Regression / golden (§4.8): TODO
- Notes: на **2** OK run полный `tp_tragg_*` и **4** ожидаемых агрегата (**(1024,)**), \(L2\approx1\). На **3** error-NPZ `feature_names` пустой — падение `text_processor` до табличного слоя (см. `text_processor_error` в JSON).

### Batch 60+ — подготовка (документация и код, не E2E батч) — 2026-04-15
- Назначение: артефакты и правки под [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) / [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md); **массовый прогон не выполнялся**.
- Целевой размер батча (чеклист п. **0.4**, обновление): **70** уникальных `video_id` в реестре (`VIDEO_REGISTRY_60PLUS.yaml` → `target_video_count: 70`; минимум по плану **60** сохраняется).
- Git commit (зафиксировать актуальный при merge): `4c45b917c5c799c3e938ae0da78f5bcce0479b8c`
- Код: `DataProcessor/api/services/processor.py` — во все ответы `run_processing` / `_run_main_py_async` (и legacy `_run_main_py_sync`) добавлены метки для worker/Prometheus: `processor="pipeline"`, `component="main_py"` (`setdefault`, чтобы не перетирать будущие уточнения). Исправлена обработка результата в `_run_main_py_sync` (раньше ошибочно использовались атрибуты вместо dict).
- Скрипт: `DataProcessor/docs/audit_v4/scripts/validate_video_registry_60plus.py` — валидация [VIDEO_REGISTRY_60PLUS.yaml](VIDEO_REGISTRY_60PLUS.yaml); флаг `--strict-count` для проверки длины списка против `target_video_count`.
- Документы: [METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md), [monitoring/README.md](../../monitoring/README.md) § «Батч 60+», журнал в чеклисте.
- Notes: разрез по внутренним экстракторам в `dataprocessor_processing_seconds` по-прежнему **не** обеспечивается этим патчем (только сквозной `main.py`).

### Наблюдаемость: Prometheus + Grafana (локальный E2E) — 2026-04-22
- Назначение: зафиксировать для батча 60+ / чеклист **фаза C (§5)** — **рабочий** стек `start_e2e_stack.sh --with-infra` (Prometheus + Grafana в Docker, DataProcessor API/worker на хосте, **два** scrape target, `DP_WORKER_METRICS_PORT=8003` в `e2e_env.sh`).
- Документ: [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md), расшифровка метрик: [METRICS_REFERENCE.md](../../monitoring/METRICS_REFERENCE.md).
- **Прод-URL** Prometheus/Grafana — **TBD** (владелец наблюдаемости, чеклист п. **4.2** / **4.7**).
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
