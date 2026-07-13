# Чеклист валидации компонентов + подтверждённых фич

Процесс — [`COMPONENT_VALIDATION_PROTOCOL.md`](COMPONENT_VALIDATION_PROTOCOL.md).
Статусы: ⬜ не начат · 🔄 прогон/анализ · 🔁 цикл доработки · ✅ заштампован (vN, прод-готов).
Назначение выхода: **seq** (для Models/Encoder) · **agg** (для аналитиков) · **both**.

> Отчёты по компонентам складываются в `DataProcessor/docs/component_reports/<component>/`.

## VisualProcessor — core/model_process

| Компонент | Выход | Статус | Версия | Отчёт |
|---|---|---|---|---|
| core_clip | seq (frame embeddings) + text-emb | ✅ | v-07-05 | [REPORT 2026-07-05](component_reports/core_clip/REPORT_2026-07-05.md): CLIP-хаб; `frame_embeddings (N,512)` L2/finite + text-эмбеддинги промптов (places365/scene/quality/…) + proxy-scores; валидатор ✅, **golden идентичен** (diff 0.0), подтверждён потребителями (scene/shot_quality). Inprocess ViT-B/32 |
| core_depth_midas | seq (per-frame агрегаты) + maps | ✅ | v3 (07-06) | [REPORT 2026-07-06](component_reports/core_depth_midas/REPORT_2026-07-06.md): 7 видео 5.7c…847.7c, inprocess MiDaS_small 256×256; валидатор ✅ rc=0, raw/norm NaN=0 (983 кадра), norm∈[0,1]; **golden побайтово идентичен на GPU (max\|Δ\|=0.0)**; различимы range_robust(std=105.7)/fg_bg(std=0.194)/complexity(CV=18.8%). C2 переопределён на CV≥0.10 (масштабо-инвариантно). Инференс 0.78 мс/кадр. seq для Encoder = per-frame агрегаты (карты 256² тяжелы) |
| core_optical_flow | seq | ✅ | v3 (07-12) | [REPORT 2026-07-12](component_reports/core_optical_flow/REPORT_2026-07-12.md): inprocess raft_small (Triton-free, схема v3 2.2), RANSAC-seed фикс → cam_* детерминир. Матрица 13 видео (N 23–300) + пресеты 3×{256/384/512, small/large}. Все гейты H1–H6 PASS: validate rc=0, ось времени (dt[0]=NaN), finite=1.0, `<2` кадров→RuntimeError, **golden 3/3 identical diff=[]**. C1 CV=0.752/p95÷med 12.8–16.6; C2 динамика/статика 4.9×; C3 диапазоны+`consistency=1/(1+div)` err 3e-8. Пресет-инвариантность агрегатов; **дефолт raft_256_small** (~120мс/ролик, 2× быстрее 512). Заметки: `bg_ratio≈0.40 by design`; batch-путь не пишет audit-v3 (синхр. перед прод) |
| core_object_detections | seq (proposals + track_ids) | ✅ | v3 (07-05) | [REPORT 2026-07-05](component_reports/core_object_detections/REPORT_2026-07-05.md): 4 видео — appearance-трекер даёт когерентные длинные person-треки (mean 52–127, max 295, frac_single~0), выходной валидатор ✅, **golden идентичен** (track_ids+boxes). Канонично `yolo11x_41_best.pt` (таксономия владельца). Остаток: метрики Prometheus |
| core_face_landmarks | seq (landmarks) | ✅ | v-07-05 | [REPORT 2026-07-05](component_reports/core_face_landmarks/REPORT_2026-07-05.md): FaceMesh 468×3 + поза/руки; **face-present ✅** (245/245 кадров) + **валидный empty без лиц ✅** (для shot_quality); валидатор ✅. Deploy: mediapipe<0.10.15 |
| ocr_extractor | seq/agg | ✅ | v-07-11 | [REPORT 2026-07-11](component_reports/ocr_extractor/REPORT.md): изоляция на синтет-фикстуре (реального text_region-детектора нет — зона владельца), движок ppocr_rec_onnx. Все гейты PASS: U1 validate_ocr ✅VALID×6/rc=0; ось int32/float32, axis_match, nan=0; **golden max\|Δconf\|=0.0** (ONNX детерминирован); privacy retain=false→только sha256+len; C1 frame-binding 100%; C3 R_varies. **Найден+исправлен логический баг expected-empty**: при отсутствии класса text_region OCR обрабатывал ВСЕ боксы вместо empty → фикс skip (reason=proposal_class_not_in_taxonomy), Dnobox→status=empty. TODO: перенести фикс main.py в git; rec_confidence на синтетике не показателен (OOD) |
| core_identity/brand_semantics | seq/agg | ⬜ | — | — |
| core_identity/car_semantics | seq/agg | ⬜ | — | — |
| core_identity/content_domain | agg | ⬜ | — | — |
| core_identity/face_identity | seq/agg | ⬜ | — | — |
| core_identity/franchise_recognition | seq/agg | ⬜ | — | — |
| core_identity/place_semantics | seq/agg | ⬜ | — | — |

## VisualProcessor — modules

| Компонент | Выход | Статус | Версия | Отчёт |
|---|---|---|---|---|
| scene_classification | seq (emb+classes) + agg | ✅ | v2+emb (07-05) | [REPORT 2026-07-05](component_reports/scene_classification/REPORT_2026-07-05.md): **полная цепочка на GPU без Triton** (core_clip inprocess + cut_detection farneback + scene) rc=0, ~95c; scene-эмбеддинг (112,512), 4 сцены, метки осмысленны, оба label_fusion согласованы, вход+выход валидаторы ✅. backbone resnet50/152/clip; fp16 2188 img/s |
| shot_quality | seq (frame_features) + agg | ✅ | v3 (07-05) | [REPORT 2026-07-05](component_reports/shot_quality/REPORT_2026-07-05.md): **полная 6-компонентная цепочка на GPU без Triton** (depth — inprocess MiDaS-обход) rc=0; `frame_features (37,48)`, 3 шота, 6 face-ROI фич = NaN by design, оба валидатора ✅. mediapipe<0.10.15. frame_features=токен качества |
| action_recognition | seq (emb) + classes | ✅ | **v3 (штамп 07-05)** | [FINAL REPORT 2026-07-05](component_reports/action_recognition/REPORT_2026-07-05_FINAL.md): penultimate-эмбеддинг (2304), Kinetics-классы, appearance-tracker, tubelet, localization, `mean_clips_per_track`=4.0/4.06, golden ✅, оба валидатора+metrics ✅, контроль empty ✅. Оценка: [ASSESSMENT](component_reports/action_recognition/ASSESSMENT_action_recognition.md). Остаток — плановые апгрейды (OSNet/VideoMAEv2/seg-tuning), не блокеры |
| color_light | seq (compact M×16) + agg | ✅ | **v2 (штамп 07-12)** | [REPORT 2026-07-12](component_reports/color_light/REPORT_2026-07-12.md): CPU-only, dep scene_classification. Все гейты U1–U6 + C1–C4 PASS на 3 видео (23/133/250 кадров) + expected-empty. **Найден+исправлен БАГ C4** (`color_distribution_gini=NaN`/entropy≈0: hue брался `frame.get` вместо `frame["features"]` → фикс getf, commit **f99742e**): после фикса gini=0.073/entropy=2.49, NaN сузился до 6 by-design aesthetic. Golden bit-identical при OMP_NUM_THREADS=1 (иначе ≤1 ULP). validate rc=0 везде |
| cut_detection | seq (events) | ✅ | v-07-05 | [REPORT 2026-07-05](component_reports/cut_detection/REPORT_2026-07-05.md): hard/motion/jump cuts + soft_events осмысленны, features-npz валиден ✅. **Q3-стоимость:** 137c farneback = артефакт Triton-free обхода; прод переиспользует core_optical_flow (RAFT/GPU, почти бесплатно) или даунсэмпл-farneback |
| video_pacing | seq/agg | ⬜ | — | — |
| story_structure | seq/agg | ⬜ | — | — |
| high_level_semantic | seq/agg | ⬜ | — | — |
| emotion_face | seq/agg | ⬜ | — | — |
| micro_emotion | seq (compact22) + agg (V=75) | ✅ | **v2.0.2 (штамп 07-13)** | [REPORT 2026-07-13](component_reports/micro_emotion/REPORT_2026-07-13.md): OpenFace(Docker)→AU/pose/gaze/landmarks. Компонент был МЁРТВ — **3 бага найдены+исправлены**: (1) leading-space в заголовках OpenFace CSV (pd.read_csv сохранял `" AU12_r"`, код читал голые имена → compact22 20/22 колонок константа, 10 PCA=NaN, microexpr=0 в 4 реальных ok-NPZ; фикс: strip колонок); (2) перепутанные метки compact22 (порядок сборки ≠ COMPACT22_FEATURE_NAMES; фикс: сборка по контракту); (3) валидатор падал на empty frame_features (N,0) (фикс: F=0 при status=empty). Валидация синтетикой (docker-in-docker на поде нет): non-const 1/22→17/22, au_pca_var_explained_1 None→0.3246, microexpr 0→8 (плоский→0), фикс==идеал побайтово, детерминизм max\|Δ\|=0.0; validate 4 ok+empty rc=0. Гейты U1–U6 + C1–C4 PASS. Остаток (не блокеры): реальный docker-OpenFace прогон, 3 placeholder-колонки, перегенерация NPZ |
| detalize_face | seq/agg | ⬜ | — | — |
| behavioral | seq/agg | ✅ | v2.0.1 (07-12) | [REPORT 2026-07-12](component_reports/behavioral/REPORT_2026-07-12.md): полная цепочка 6 видео (34–300c), U1–U6 PASS, C1–C4 PASS. Ключевой фикс: body_lean *5.0 дефект (была константа 1.0). std=0.239, 546 уник/575. Pose 57.22%/mouth 56.18% NaN структурны (co-NaN by design). Авто-штамп при 100% PASS |
| frames_composition | seq/agg | ✅ | **v2.0.1 (штамп 07-14)** | [REPORT 2026-07-13](component_reports/frames_composition/REPORT_2026-07-13.md): CPU-only модуль (numpy+opencv). **2 логических бага найдены+исправлены**: (1) `neg_space_balance_lr=0.0` при no_obj → 1.0 (идеальный баланс); (2) depth нормализация в style_probs: `ds=clip01(depth_std)` давал всегда 1.0 → `ds=clip01(depth_std/depth_mean)` (CV dimensionless), `bokeh_proxy=(p95-p05)/1024.0` (MiDaS scale). Все гейты U1–U6 + C1–C4 PASS на батче 24 NPZ + pod (N=43..215, 5.0×). Golden max|Δ|=0.0, различимость 5/5 ключевых фич. |
| optical_flow (module) | seq/agg | ⬜ | — | — |
| similarity_metrics | agg | 🔁 | — | L4 (NaN policy) |
| uniqueness | agg | 🔁 | — | L3 (NaN fix) |
| text_scoring | seq/agg | ⬜ | — | — |

## AudioProcessor — extractors

| Компонент | Выход | Статус | Версия | Отчёт |
|---|---|---|---|---|
| asr_extractor | seq | ⬜ | — | — |
| clap_extractor | seq (emb) | ⬜ | — | — |
| speaker_diarization_extractor | seq (events) | ⬜ | — | — |
| emotion_diarization_extractor | seq | ⬜ | — | — |
| source_separation_extractor | seq | ⬜ | — | — |
| speech_analysis_extractor | agg | ⬜ | — | — |
| pitch_extractor | seq | 🔄 | — | правился Cursor |
| loudness_extractor | seq | ⬜ | — | — |
| spectral_extractor | seq | ⬜ | — | — |
| spectral_entropy_extractor | seq | ⬜ | — | — |
| mel_extractor | seq | ⬜ | — | — |
| mfcc_extractor | seq | ⬜ | — | — |
| chroma_extractor | seq | ⬜ | — | — |
| tempo_extractor | seq/agg | ⬜ | — | — |
| rhythmic_extractor | seq/agg | ⬜ | — | — |
| onset_extractor | seq (events) | ⬜ | — | — |
| key_extractor | agg | ⬜ | — | — |
| band_energy_extractor | seq | ⬜ | — | — |
| quality_extractor | agg | ⬜ | — | — |
| voice_quality_extractor | seq/agg | ⬜ | — | — |
| hpss_extractor | seq | ⬜ | — | — |

## TextProcessor — extractors

| Компонент | Выход | Статус | Версия | Отчёт |
|---|---|---|---|---|
| title_embedder | emb | ⬜ | — | — |
| description_embedder | emb | ⬜ | — | — |
| hashtag_embedder | emb | ⬜ | — | — |
| tags_extractor | agg | ⬜ | — | — |
| transcript_aggregator | agg | ⬜ | — | — |
| transcript_chunk_embedder | seq (emb) | ⬜ | — | — |
| comments_aggregator | agg | ⬜ | — | — |
| comments_embedder | emb | ⬜ | — | — |
| speaker_turn_embeddings_aggregator | seq/agg | ⬜ | — | — |
| asr_text_proxy_audio_features | agg | ⬜ | — | — |
| lexico_static_features | agg | ⬜ | — | — |
| semantics_topics_keyphrases | agg | ⬜ | — | — |
| semantic_cluster_extractor | agg | ⬜ | — | — |
| cosine_metrics_extractor | agg | ⬜ | — | — |
| embedding_stats_extractor | agg | ⬜ | — | — |
| embedding_shift_indicator_extractor | agg | ⬜ | — | — |
| embedding_source_id_extractor | agg | ⬜ | — | — |
| embedding_pair_topk_extractor | agg | ⬜ | — | — |
| qa_embedding_pairs_extractor | agg | ⬜ | — | — |
| title_embedding_cluster_entropy_extractor | agg | ⬜ | — | — |
| title_to_hashtag_cosine_extractor | agg | ⬜ | — | — |
| topk_similar_titles_extractor | agg | ⬜ | — | — |

> **Segmenter** входит в цикл доработки каждого компонента (подгонка sampling budget);
> отдельного «выхода-фичи» не имеет, версионируется через `sampling_policy_version`.

## Ledger подтверждённых фич

Заполняется ПО ХОДУ валидации каждого компонента (источник списка фич —
`<component>/docs/FEATURE_DESCRIPTION.md` + `utils/validate_*`). Шаблон на компонент:

```
### <component> (v<N>)
| фича | тип (seq/agg) | назначение (model/analyst) | подтверждено | заметка |
|---|---|---|---|---|
| <feature_1> | seq | model | ✅/❌ | health=..., time-axis ок |
| <feature_2> | agg | analyst | ✅/❌ | |
```

### action_recognition (v3, штамп 2026-07-05)
| фича | тип | назначение | подтверждено | заметка |
|---|---|---|---|---|
| `clip_embeddings (C,2304)` | seq | model | ✅ | penultimate-фичи backbone, L2, finite; токен для Encoder |
| `clip_topk_action_ids/probs` + `class_names` | seq/agg | analyst+model | ✅ | Kinetics-400, осмысленны (сверено с роликами) |
| `clip_track_id` | seq | model | ✅ | из appearance-tracker; ~4–5 клипов/трек |
| `clip_times_s`/`clip_frame_indices` | seq | model | ✅ | ось времени ⊆ union |
| `video_action_hist (400)`, `dominant_action_*` | agg | analyst | ✅ | распределение действий видео |
| `clip_segment_id` | seq | analyst | ⚠ | сейчас 1 сегмент/трек (change-point не дробит); тюнится `seg_cos_threshold` |
| `num_tracks`, `mean_clips_per_track` | agg | analyst | ✅ | 11/16 треков, 4.0/4.06 |

(Фич сотни — ledger растёт покомпонентно; не заполняем авансом, только после прогона+сверки.)

## Стартовый backlog логических багов (из LOGIC_ERRORS_FOR_CLAUDE.md)

| ID | Компонент | Суть | Прод-fix |
|---|---|---|---|
| L1 | scene_classification | Triton Places365 batch=1 (иначе HTTP 400) | дефолт `batch_size=1` / авто-clamp |
| L2 | story_structure, video_pacing | `min_frames=30` рушит короткие ролики | min_frames по duration или status=empty |
| L3 | uniqueness | All-NaN в Otsu-пороге | guard перед `nanargmax` (исправлено) |
| L4 | similarity_metrics | status=ok при ~60% NaN | schema-aware validator, NaN by design |
| L9 | (ingest) | mock 3s вместо реального видео | HF runner копирует `{id}.mp4` |

Полный список и детали — [`LOGIC_ERRORS_FOR_CLAUDE.md`](LOGIC_ERRORS_FOR_CLAUDE.md).
