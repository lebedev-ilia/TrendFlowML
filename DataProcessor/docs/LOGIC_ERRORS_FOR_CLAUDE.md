# Логические ошибки DataProcessor (для Claude)

Дата: 2026-07-02. Контекст: прогон на реальных коротких видео из [Ilialebedev/videos11](https://huggingface.co/datasets/Ilialebedev/videos11) (~10–30 с), E2E без `core_identity`.

---

## L1 — `scene_classification` + Triton batch_size (исправлено в E2E-патче)

**Симптом:** `scene_classification(triton) | infer failed: HTTP 400`, каскад в `color_light` (нет артефакта scene).

**Причина:** Places365 ensemble в Triton имеет `max_batch_size=0` (фиксированный batch=1). При `--real-video` E2E-патч не выставлял `scene_classification.batch_size=1`, уходил batch>1 → HTTP 400.

**Исправление:** `e2e_full_max_run.py` — всегда `batch_size=1` для scene_classification в E2E (не только mock).

**Правильный fix в продукте:** дефолт `batch_size=1` в `global_config.yaml` для Triton Places365 или авто-clamp в модуле.

---

## L2 — `story_structure` / `video_pacing` min_frames на коротких роликах (исправлено в E2E-патче)

**Симптом:** `too few frames: N=12 < min_frames=30 (no-fallback)` на 10.7 с видео.

**Причина:** Segmenter даёт ~12 union-кадров на коротком клипе; модули требуют `min_frames=30` абсолютно, без привязки к длительности или sampling policy.

**Исправление:** E2E-патч `min_frames=8` для обоих модулей (как для mock).

**Правильный fix:** `min_frames = min(config_min, max(8, int(0.25 * duration_sec * fps_sample)))` или status=`empty` + `empty_reason=too_few_frames` вместо hard error.

---

## L3 — `uniqueness` падает на вырожденном распределении (исправлено)

**Симптом:** `All-NaN slice encountered` в Otsu-подобном пороге (`np.nanargmax(sigma_b2)`).

**Файл:** `VisualProcessor/modules/uniqueness/utils/uniqueness.py` (`_otsu_threshold_quality`).

**Причина:** При плоской гистограмме similarity все `sigma_b2` → NaN; нет guard перед `nanargmax`.

**Исправление:** `if not np.any(np.isfinite(sigma_b2)): return default_thr, 0.0`.

---

## L4 — `similarity_metrics`: status=ok при ~60% NaN в `feature_values`

**Симптом:** Валидатор §0.2: finite ratio 38.5% при `status=ok`.

**Причина:** Контракт фичи — NaN для отсутствующих модальностей (нет reference video, нет emotion/place и т.д.). Это **задуманное** разрежение, не crash.

**Действие:** §0.2 понижает до warning для `similarity_metrics`; нужен schema-aware validator (`validate_similarity_metrics.py`), а не порог 50% finite.

---

## L9 — Fetcher worker не видит `--mock-video-dir` (исправлено в HF runner)

**Симптом:** HF video_id в URL, но Segmenter `duration_sec≈3.0` (mock tone), emotion/source_separation → `audio_too_short`.

**Причина:** `FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_DIR` задаётся в `e2e_env.sh` для worker при старте (`example/example_videos`). Без `{id}.mp4` Fetcher берёт `sample_{hash%N}.mp4` (3 s mock).

**Исправление:** `e2e_run_hf_videos11.py` копирует `example/hf_videos11/{id}.mp4` → `example/example_videos/{id}.mp4` + чистит кеш Segmenter; проверка `duration_sec` после прогона.

---

## L5 — manifest `empty_reason` (исправлено)

**Файл:** `AudioProcessor/run_cli.py` — `empty_reason` вычислялся, но не передавался в `ManifestComponent`.

---

## L6 — `face_identity` manifest ok vs NPZ empty (core_identity, вне scope HF-теста)

**Симптом:** manifest `face_identity` status=ok, NPZ `meta.status=empty`, `no_faces_in_video`.

**Причина:** Рассинхрон статуса оркестратора VisualProcessor (manifest `face_identity` ok, artifacts=[]) и meta в NPZ (`core_face_identity/face_identity.npz`).

**E2E workaround (2026-07-02):** `e2e_validate_output_quality.py` — valid empty NPZ (`no_faces_in_video`, …) → warning `manifest_ok_npz_valid_empty`, не error.

**Fix (2026-07-02):** `VisualProcessor/main.py` — `_component_artifact_dir`: manifest name `face_identity` → rs_path `core_face_identity/` для сбора NPZ и sync `status`/`empty_reason` из meta (L6).

## L12 — `face_identity` n_frames>0 но total_faces_processed=0 (2026-07-02)

**Симптом:** NPZ `n_frames=5`, `total_faces_processed=0`, ранее `empty_reason=no_faces_in_video` при наличии лиц в `core_face_landmarks`.

**Корневая причина (исправлено):** `_crop_face` трактовал **normalized** landmarks [0,1] как пиксели → пустой crop → ES test fail → `embedding_service_available=False`.

**Доп. причина:** Embedding Service ArcFace — `protobuf`/`ml_dtypes`/`onnx` mismatch в `.data_venv` (см. `embedding_service/requirements-e2e.txt`).

**Fix:** `face_identity/main.py` — scale bbox по размеру кадра; `empty_reason` = `embedding_service_unavailable` / `no_faces_processed`; §0.2 valid empty (warning L6). Перезапуск `:8005` после `pip install -r requirements-e2e.txt`.

---

## L10 — stale NPZ при повторном E2E на том же video_id (исправлено в HF runner)

**Симптом:** Segmenter `duration_sec≈10.8`, но `emotion_diarization` NPZ `empty_reason=audio_too_short`, `segments_total=2` (как на 3 s mock); mtime NPZ старше текущего прогона.

**Причина:** `cold-ingestion` чистит только Fetcher DB; `storage/result_store/youtube/{id}/` сохраняется. Backend может переиспользовать run_id; DataProcessor не перезаписывает старые артефакты.

**Исправление:** `e2e_run_hf_videos11._clear_caches` — `shutil.rmtree` для `result_store/youtube/{id}` и `state/youtube/{id}` перед каждым прогоном.

---

## L7 — `emotion_diarization` empty на 10+ с реальном аудио

**Симптом:** `empty_reason=audio_too_short` при `duration_sec≈10.7`.

**Часто это L10 (stale NPZ), не баг порога:** при чистом прогоне `max(ends)` из family `emotion` в `segments.json` > 5 s. Порог <5 s — контракт Audit v3.

**Если воспроизводится на свежем run_id:** проверить `segments_total` в NPZ vs `families.emotion.segments` в Segmenter.

---

## L13 — scene_classification: cuDNN NOT_INITIALIZED (torch cu121 не портируем между GPU) — фикс валидирован 2026-07-23

**Симптом:** в 1000-видео прогоне (Gate 3, под RTX 2000 Ada) `scene_classification` упал на **497 из ~996**
видео с `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` (rc=4, стадия `_infer_batch_compact`,
resnet50 GPU-инференс). 499 «OK» — только видео из Gate 2 (делались на поде RTX A4500, resumable не трогал).

**ЧЕСТНАЯ причина (первый вывод «N=8 concurrency» был ОШИБКОЙ, отозван):** resnet50 GPU-forward падает с
той же ошибкой **даже при N=1 в полностью изолированном процессе** на этом поде → это **несовместимость
cuDNN, поставляемого с `torch==2.4.1+cu121`, с этим GPU/драйвером (RTX 2000 Ada, Ada Lovelace sm_89)**, а
НЕ конкуренция за GPU. Следствие: **«OPT-2 scene-GPU» (перевод scene на CUDA) НЕ портируем** — на одних
подах cuDNN инициализируется (A4500 в Gate 2), на других нет (RTX 2000 Ada в Gate 3).

**Проверено на поде:** `torch.backends.cudnn.enabled=False` → resnet50 на GPU работает (native convs);
CPU тоже работает (1.9с/16 кадров). `cuda.is_available()=True`, падает именно cuDNN-conv.

**Исправление** (`modules/scene_classification/utils/scene_classification.py`, сразу после `self.device=…`):
cuDNN-guard — проба `Conv2d` на устройстве при инициализации; при исключении → `torch.backends.cudnn.enabled
=False` + повторная проба (GPU без cuDNN); если GPU вовсе нежизнеспособен → `self.device="cpu"`. Логируется
`logger.warning`. Портируемо на любой под без внешней настройки.

**Validated (2026-07-23):** пере-прогон полного пайплайна `1xFgqSpn1p0` с фиксом на том же поде →
`scene_classification` NPZ появился (+ все компоненты). Осталось до-прогнать ~497 видео Gate 3 со scene.
**Урок для инфры:** GPU-inprocess torch-компоненты (scene и будущие shot_quality/action_recognition/…)
должны иметь cuDNN/CPU-fallback; нельзя полагаться, что cu121-cuDNN заведётся на произвольном RunPod GPU.

---

## L12 — cut_detection: np.histogram падает на вырожденном/битом диапазоне интервалов (исправлено 2026-07-23)

**Симптом:** в 500-видео прогоне (Gate 2) ровно 1 видео (`1xFgqSpn1p0`) уронило `cut_detection` с
`ValueError: Too many bins for data range. Cannot create 3 finite-sized bins`, а зависимые `scene_classification`
и `video_pacing` каскадно не запустились («Обязательная зависимость 'cut_detection' не найдена»). Итог по
компонентам: 499/1 у трёх компонентов — но корневая причина ОДНА.

**Причина:** `_compute_interval_features` / `compute_shot_length_stats` считали `np.histogram(intervals, bins=N)`
(N=`min(20,max(2,size))` и `bins=8`) без защиты диапазона. numpy 2.x бросает эту ошибку, когда `max-min`
не разбивается на N финитных краёв — при: (а) inf/NaN в данных (range non-finite), (б) нулевом span, (в)
**финитных, но огромных значениях** с крошечным относительным span (напр. битые timestamp `[1.7e12, 1.7e12+1]`
или `[1e20, 1e20+2]`), где float-шаг превышает span. Видео дало вырожденные shot-интервалы (вероятно битый
timestamp от сегментации/декода).

**Исправление:** `cut_detection.py::_safe_histogram(data, bins, range=None)` — (1) фильтрует не-финитные
значения `arr[np.isfinite(arr)]`; (2) при вырожденном/малом-относительно-магнитуды span подставляет явный
`range` с относительным min-span (`|v|·eps·N·8`); (3) try/except-фолбэк на 1 финитный бин. Заменены все
4 уязвимых вызова (`intervals`×2, `durations_s`×2). Семантика сохранена: константный сигнал → один
заполненный бин → низкая энтропия (как и задумано).

**Validated (2026-07-23):** на numpy пода 2.4.6 все краш-кейсы проходят: `exact_1e20`, `inf`, `nan`, `const`,
`empty`, `normal`, `ts_ms` → `ALL PASS`. Полная ре-валидация на самом видео — при следующем прогоне (Gate 3).
**Follow-up (не баг cut_detection):** откуда у `1xFgqSpn1p0` вырожденные/огромные интервалы — вопрос к
upstream (segmenter/декод timestamp); cut_detection теперь к такому входу робастен независимо от причины.

---

## L11 — EmoNet vendor + OpenFace docker (E2E infra gap)

**Симптом:** `emotion_face` error: `EmoNet source file not found: .../dp_models/emonet/emonet/models/emonet.py` (веса `emonet_8.pth` есть). `micro_emotion` error: docker pull `openface/openface:latest` denied. Каскад: `high_level_semantic` missing emotion_face NPZ.

**Исправление E2E:** `e2e_full_max_run._resolve_openface_docker_image()` — `OPENFACE_DOCKER_IMAGE` + fallback (`algebr/openface:latest` и др.); патч `micro_emotion.docker_image` или auto-disable.

**Продуктовый fix:** `python DataProcessor/scripts/vendor_emonet.py` (в `bootstrap.sh` Phase 4); OpenFace: `./backend/scripts/setup_e2e_openface.sh` (тегирует `algebr/openface` → `openface/openface:latest`). В `openface_analyzer.py`: путь `/home/openface-build/...`, `--entrypoint bash`, toleration segfault при наличии CSV.

**E2E validated (2026-07-02):** HF `-4WRepA-bss` run `4a4bea25-…` — `micro_emotion` success, OpenFace 4/5 face frames, §0.1+§0.2 PASS.

---

## L8 — ASR → TextProcessor: transcript не прокидывается (исправлено в HF/E2E real-video)

**Симптом:** `text_features` payload `transcript_len_chars=0` при успешном `asr_extractor`.

**Причина:** E2E с `--offline-use-example-text-file` подставлял fixture `video_document_1.json` вместо autogen из ASR NPZ.

**Исправление:** `--real-video` → не задавать `processors.text.input_json`; ASR autogen в `main.py` + merge title/desc из `input_metadata_json` (fixture metadata). §0.2 проверяет `asr_wired_to_text`.

**Доп. баг (исправлено):** `_get_any("token_ids_by_segment")` для single-segment ASZ возвращал плоский `ndarray` → autogen с пустым `asr`. Fix: нормализация в `main.py` перед сборкой `token_ids_clean`.

**Примечание:** `transcript_len_chars` может быть 0 при token-only контракте — смотреть `transcripts_token_ids.whisper` в autogen JSON.

---

## Прогон HF (итог)

**2026-07-02 — финальный прогон** (L9+L10+L8 ASR autogen, EmoNet vendored, micro_emotion disabled без OpenFace).  
Лог: `backend/.e2e/logs/hf_videos11_l8_full5.log`

| video_id | duration | run_id | ASR tokens | §0.1 | §0.2 | audio |
|----------|----------|--------|------------|------|------|-------|
| `-4WRepA-bss` | 10.7s | `f1af42e0-…` | 4 | PASS | PASS | ok |
| `-8WeWWOpxHk` | 10.6s | `de18dcf1-…` | 4 | PASS | PASS | ok |
| `-3Mbinqzig4` | 17.3s | `f162bc8a-…` | 291 | PASS | PASS | ok |
| `-4RHVBIikn8` | 29.6s | `fe4e1220-…` | 459 | PASS | PASS | ok |
| `-0InsUQNwIQ` | 16.3s | `79536d73-…` | 85 | PASS | PASS | ok |

Предыдущий прогон (без L8, fixture text): `hf_videos11_real_run_all.log` — superseded.

Команда: `python backend/scripts/e2e_run_hf_videos11.py --count 5 --with-triton-docker`  
Видео: `example/hf_videos11/` из [Ilialebedev/videos11](https://huggingface.co/datasets/Ilialebedev/videos11).  
Пошаговый журнал (git clone → green): [`backend/docs/E2E_RUNBOOK.md` §0.15](../backend/docs/E2E_RUNBOOK.md#015-worklog-20-quality--hf-videos11-2026-07-02).

**Типичные warnings (не fail):** manifest без `empty_reason`, sparse `similarity_metrics`, NaN-слоты в `text_features`.

**Остаётся empty (не error) на части visual/audio:** ocr, action_recognition без детекций/действий; emotion/source_separation/speech/voice_quality — см. L5/L7.
