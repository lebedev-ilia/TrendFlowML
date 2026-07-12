# Учёт времени тестов и затраченных ресурсов (по компонентам)

Ведётся по ходу автономных прогонов. GPU-под: **RunPod RTX A4500 20GB**, CUDA 550, torch 2.4.1+cu124,
48 vCPU, Python 3.11. Стоимость пода ~$0.2–0.35/ч (Running).

## Легенда
- **wall** — реальное время стадии; **peak_vram** — пик VRAM; **дев-время** — сколько сессии ушло.
- Стадии цепочки: `trim` (ffmpeg) · `segmenter` · `detections`(+tracker) · `component`.

## action_recognition (заштампован v3, 2026-07-05)

| прогон | клип | устройство | segmenter | detections(+tracker) | action_recognition | итог/заметки |
|---|---|---|---:|---:|---:|---|
| GPU smoke (Claude сам) | 4:35 → 20s @25fps | RTX A4500 | 22.3 s | 33.5 s | 20.3 s | ✅ оба валидатора; clip_count=11, mean_clips/track=5.5, emb (11,2304) penultimate |
| CPU-профиль (v3.1, Cursor) | 4:35 полный | CPU | 309 s | 53.6 s | ~77 s (penultimate+tubelet) | — |
| overhead трекера (v2) | 4:35 | — | — | 2.3 s (~4% детекции) | — | histogram-эмбеддер |

Ресурсы action_recognition: пик VRAM ~1.4–1.6 GB (SlowFast fp32, batch 8); кэш frames dense-окон
~10–11 GB/видео (dense windows) → политика очистки `cleanup_frames_after_npz.py` обязательна для 200k.
Стоимость ∝ числу person-треков/окон; капы: `max_windows`, `min_clip_real_frames`, `window_len_mult`.

## scene_classification (в работе, 2026-07-05, GPU RTX A4500)

**Бенчмарк модели Places365 ResNet50** (inprocess, torchvision, веса CSAIL, load: missing 0/unexp 0):

| precision | batch | ms/batch | throughput | пик VRAM |
|---|---:|---:|---:|---:|
| fp32 | 8 | 7.7 | 1034 img/s | — |
| fp32 | 32 | 26.8 | 1194 img/s | — |
| fp32 | 64 | 51.0 | 1256 img/s | — |
| **fp16** | 32 | 15.0 | 2138 img/s | — |
| **fp16** | 64 | 29.2 | **2188 img/s** (×1.7 к fp32) | **812 MB** |

**Вывод (оптимизация):** канонично **inprocess ResNet50 + fp16 + batch=64** (~2200 img/s → сотни
кадров видео инференсятся <1 c). Это подтверждает выбор inprocess vs **Triton batch=1** (L1, HTTP 400
при batch>1) — Triton здесь узкое место. Рекоменд. дефолт: inprocess, fp16, batch 32–64.

Данные видео (HF `Ilialebedev/videos11`, 328 mp4 на поде): размеры 0.9–102 MB (разные разрешения/
длительности) — метаданные соберутся на прогоне цепочки.

**Модель-слой компонента (ModelManager, inprocess)**: `places365_resnet50` resolve+load **2.3 c**,
forward (8,365) ✅ на GPU. Реальные предсказания на HF-видео осмысленны (music_studio, ice_cream_parlor,
arena/performance). Backbone-C (ConvNeXt/ViT/EfficientNet) — specs готовы, нужны только Places365-веса.

**ПОЛНАЯ ЦЕПОЧКА inprocess БЕЗ Triton (2026-07-05, GPU A4500, видео ~112 кадров @6fps):**
| стадия | wall |
|---|---:|
| Segmenter | 37 s |
| core_clip (inprocess ViT-B/32) | 14 s |
| cut_detection (farneback, no-clip) | 31 s |
| scene_classification (Places365 inprocess) | 13 s |
| **итого** | **~95 s** |
Результат: 112 кадров, 4 сцены; `frame_scene_embedding (112,512)` из core_clip; метки осмысленны
(beauty_salon/hospital_room/laundromat). **label_fusion places и clip — одинаковые метки** на этом
видео (согласованно). Оба валидатора (вход+выход) ✅. **Triton НЕ нужен** (ключевая находка сессии):
core_clip inprocess (`clip.load ViT-B/32`) + cut_detection `--no-use-clip --no-require-core-optical-flow`
(внутренний farneback). Прямой вызов компонентов (обход dep-проверки оркестратора) — раннер
`scripts/run_scene_local.py`. Оптимизация: Places365 fp16 bs=64 = 2188 img/s (модель ничтожна в цепочке;
узкие места — Segmenter + cut_detection farneback).

## shot_quality (в работе, 2026-07-05, GPU A4500, видео @6fps)

Самая тяжёлая цепочка (5 deps). Прогон inprocess без Triton (depth — обход `midas_depth_inprocess.py`):
| стадия | wall | rc |
|---|---:|---|
| Segmenter | 45 s | ✅ |
| core_clip (inprocess ViT-B/32) | 34 s | ✅ |
| core_object_detections (ultralytics) | 57 s | ✅ |
| core_face_landmarks (mediapipe) | 19 s | ❌ mediapipe 0.10.35 убрал `mp.solutions` |
| cut_detection (farneback, no-clip) | **137 s** | ✅ (узкое место цепочки) |
| **core_depth_midas (inprocess MiDaS bypass)** | 44 s | ✅ (Triton обойдён!) |
| shot_quality (numpy) | — | ждёт face |

**ФИНАЛ ✅ (2026-07-05):** все 6 rc=0 (mediapipe даунгрейд 0.10.14 → face_landmarks ✅; поверх
кэшированных deps на Network Volume ~1 мин). `frame_features (37,48)`, 3 шота, 6 face-ROI = NaN by
design, оба валидатора ✅. **shot_quality заштампован.** Узкое место по времени — cut_detection farneback
(137 s); для 200k рассмотреть RAFT/оптический поток или переиспользование готового core_optical_flow.
shot_quality сам CPU/numpy (дёшев). Видео: HF `videos11`, 30 шт разной длины (0.9–563 МБ).

## core_object_detections (+ appearance-трекер, 2026-07-05, GPU A4500, 4 видео разной длины @6fps)

| видео | Segmenter | object_det(YOLO+трекер) | tracks | mean_len | max_len | person_dets | frac_single |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q3TcAJfnuWw | 71.8s | 58.7s | 3 | 100.7 | 295 | 302 | 0.0 |
| 70_Id7JQot0 | 42.0s | 53.9s | 2 | 127.5 | 233 | 255 | 0.0 |
| ODMGyEO8nf4 | 38.0s | 50.4s | 3 | 52.3 | 137 | 157 | 0.33 |
| 1Bm8VSlH_nU | 29.9s | 45.1s | 2 | 63.0 | 72 | 126 | 0.0 |

**Трекер (мой appearance-embedding, histogram):** когерентные длинные треки (mean 52–127, max до 295),
**низкая фрагментация** (frac_single в осн. 0.0), 2–3 трека/видео. Выходной валидатор ✅ (schema v3).
Стоимость object_det ~45–59s (YOLO + трекер) на разреженной выборке. Нюанс: прогон на `yolo11l.pt`
(COCO); канонично `yolo11x_41_best.pt` (41-класс) — на трекинг не влияет (person=class 0).

## core_clip / cut_detection / core_face_landmarks (2026-07-05, GPU A4500)
- **core_clip** (inprocess ViT-B/32): ~14–34c/видео; `frame_embeddings (N,512)` L2/finite; **golden diff 0.0**.
  Ключ 200k: считать CLIP ОДИН раз, переиспользовать всеми (scene/shot/cut/similarity).
- **cut_detection**: 137c = **farneback (CPU) из-за Triton-free обхода**; прод → reuse core_optical_flow
  (RAFT/GPU, ~бесплатно) ИЛИ даунсэмпл-farneback. Границы: 3 hard+2 motion+1 soft — осмысленно.
- **core_face_landmarks**: mediapipe **<0.10.15** (0.10.35 удалил mp.solutions). FaceMesh 468; face-present
  245/245 кадров на людях; валидный empty без лиц. Нужен core_object_detections (person-боксы) до него.

## core_depth_midas (2026-07-06, GPU RTX 2000 Ada 16GB, 7 видео 5.7c…847.7c @fps6/w480, inprocess MiDaS_small 256×256)

| видео | dur,c | N | Segmenter | depth_stage | init(torch.hub) | infer | ms/кадр |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMhLOCzZSmY | 5.7 | 23 | 29.0s | 28.0s | 20.2s | 139.6ms | 6.07 |
| FtnNsejZOfQ | 10.0 | 40 | 31.7s | 35.5s | 27.2s | 87.1ms | 2.18 |
| Hb3Z1HmgYKw | 30.0 | 120 | 35.5s | 34.1s | 21.6s | 163.8ms | 1.36 |
| LogDeH7V6bM | 59.5 | 200 | 41.5s | 36.8s | 20.7s | 155.6ms | 0.78 |
| ItRcDFKFiSU | 149.9 | 200 | 48.4s | 41.6s | 24.3s | 160.4ms | 0.80 |
| QfzWnPhdg3g | 344.1 | 200 | 51.0s | 39.1s | 23.2s | 153.2ms | 0.77 |
| Hb0mM9YLQOY | 847.7 | 200 | 70.2s | 42.0s | 24.4s | 156.8ms | 0.78 |

- Чистый инференс MiDaS_small **0.77–0.80 мс/кадр** на 200-кадровых роликах (RTX 2000 Ada). Раздутый depth_stage
  на весь ролик — из-за **разовой torch.hub.load ~20–27с** (импорт+кеш+перенос на GPU) на КАЖДЫЙ подпроцесс раннера.
  В проде модель грузится один раз на воркер → не относить на стоимость видео.
- **Golden побайтово идентичен на GPU** (max|Δ|=0.0, ×7): bicubic MiDaS + inference_mode детерминированы.
- Segmenter depth {min120/target200/max400}: короткие берут все кадры (23/40/120), с ~60c упирается в target=200.

## Заметки по оптимизации/масштабу (200k)
- action_recognition: dense-окна → диск главный лимит; detection ×3–5 на dense-кадрах (компромисс качества).
- SSH/rsync-нюансы настройки пода — `runpod_ssh/POD_SETUP_LOG.md`.

## ocr_extractor (2026-07-11, RTX 4000 Ada, ppocr_rec_onnx CPU-ORT, синтет-фикстура)
| ролик | кадров | боксов→строк | время main.py, c |
|---|---:|---:|---:|
| vidFshort | 3 | 1 | 8.6 |
| vidA | 12 | 4 | 10.6 |
| vidElong | 200 | 3 | 9.1 |
- Время почти не зависит от числа кадров (3..200) — доминирует init ONNX-сессии + загрузка словаря/модели
  на КАЖДЫЙ subprocess (~8с). Само распознавание — единицы боксов, дёшево. В проде сессия ppocr грузится один
  раз на воркер → per-video стоимость OCR = только кропы×inference (мс). onnxruntime — CPU (GPU не требовался).

## behavioral (2026-07-12, RTX 4000 Ada 20GB, CPU-only numpy)
- Стадия behavioral: 3.0–4.0 с/видео (N=34…300, чистый numpy CPU). process ~136–605мс + pack ~35–63мс.
- Upstream (на том же поде): Segmenter ~19–22с; core_object_detections(YOLO11l) ~34–35с; core_face_landmarks(pose+hands+face-mesh) ~16–44с.
- Вывод: behavioral — дешёвая CPU-стадия; стоимость определяется upstream (OD+landmarks). Для 200k батчить landmarks, behavioral почти бесплатен.

## core_optical_flow (2026-07-12, RTX 2000 Ada 16GB, inprocess raft_small/large, fps=4, width=480)
- init (загрузка raft-весов на GPU) ~9–10.7 с/подпроцесс — доминирует; per-video процесс платит каждый раз (в проде — разовая стоимость воркера).
- flow_inference_total: raft_256 ~83–160 мс/ролик (N≤300, ~0.58 мс/пара); raft_384 148–200 мс; raft_512 161–307 мс. raft_large на 256 сопоставим со small.
- Стадии на ролик: segmenter ~24–30с, core_optical_flow ~17–28с (в осн. init), validate ~2.4–4.8с.
- Вывод: пресет raft_256_small — оптимум (2× быстрее 512, качество агрегатов не хуже). Для 200k: держать один прогретый воркер, батчить пары кадров, переиспользовать RAFT для cut_detection.

## color_light (2026-07-12, RTX 2000 Ada, CPU-only модуль; deps CLIP+scene на GPU)
- color_light сам CPU-only: v1short 133 кадров ~1.5с обработки; vshort 23 кадра <1с; vlong 250 кадров ~3–4с.
- Полная цепочка per-video (Segmenter→core_clip→cut_detection→scene_classification→color_light):
  короткое (23f) ~1.5 мин, среднее (133f) ~2 мин, длинное (250f) ~2–3 мин; доминируют CLIP+scene (GPU), не color_light.
- ffmpeg обязателен для Segmenter (audio extract) — на свежем контейнере ставить (apt install ffmpeg).
- Golden: пиннить OMP_NUM_THREADS=1 для побайтового детерминизма (иначе ≤1 ULP дрожание BLAS).
- Для 200k: color_light дёшев, переиспользовать per-frame HSV/LAB из общего декодера; узкое место — scene/CLIP deps.

## micro_emotion (2026-07-13)
- GPU-прогон НЕ выполнялся: OpenFace только через Docker --gpus all, docker-in-docker на поде RunPod недоступен.
  Логика постобработки валидирована СИНТЕТИЧЕСКИ локально (CPU, /tmp/me_venv). Пода не поднимал.
- Тайминги постобработки (из summary реального ok-NPZ, producer 2.0.2): openface_run_ms≈14767 (docker OpenFace,
  39 face-кадров), micro_emotion_features_ms≈23 (чистая постобработка на 39 строк). Т.е. стоимость = сам OpenFace,
  постобработка пренебрежима. Реальные прод-тайминги docker-OpenFace замерить на отдельном docker-этапе.
