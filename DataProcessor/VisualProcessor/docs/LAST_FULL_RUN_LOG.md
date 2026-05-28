NOTE (Audit v3): этот лог относится к запуску **до** обновления `core_object_detections` до schema v2 и до фиксации политики “tracking removed”.
WARNING строки про `tracks not found` для semantic heads здесь ожидаемы и не являются актуальным требованием контракта.

================================================================================
  VisualProcessor
================================================================================
  [✓ OK] Запуск VisualProcessor
    2026-02-14 10:17:31,220 INFO: VisualProcessor | main | Начало обработки
    2026-02-14 10:17:31,789 INFO: VisualProcessor | main | parallelism: max_parallel_modules=1 gpu_max_concurrent=1
    2026-02-14 10:17:31,789 INFO: VisualProcessor | main | Текущие core_providers:
    2026-02-14 10:17:31,789 INFO:             core_clip
    2026-02-14 10:17:31,789 INFO:             core_depth_midas
    2026-02-14 10:17:31,789 INFO:             core_optical_flow
    2026-02-14 10:17:31,789 INFO:             core_object_detections
    2026-02-14 10:17:31,789 INFO:             ocr_extractor
    2026-02-14 10:17:31,789 INFO:             core_face_landmarks
    2026-02-14 10:17:31,789 INFO:             content_domain
    2026-02-14 10:17:31,789 INFO:             franchise_recognition
    2026-02-14 10:17:31,789 INFO:             brand_semantics
    2026-02-14 10:17:31,789 INFO:             car_semantics
    2026-02-14 10:17:31,789 INFO:             face_identity
    2026-02-14 10:17:31,789 INFO:             place_semantics
    2026-02-14 10:17:31,789 INFO: VisualProcessor | main | Текущие модули:
    2026-02-14 10:17:31,789 INFO:             cut_detection
    2026-02-14 10:17:31,789 INFO:             scene_classification
    2026-02-14 10:17:31,789 INFO:             video_pacing
    2026-02-14 10:17:31,789 INFO:             uniqueness
    2026-02-14 10:17:31,789 INFO:             shot_quality
    2026-02-14 10:17:31,789 INFO:             story_structure
    2026-02-14 10:17:31,789 INFO:             detalize_face
    2026-02-14 10:17:31,790 INFO:             emotion_face
    2026-02-14 10:17:31,790 INFO:             behavioral
    2026-02-14 10:17:31,790 INFO:             optical_flow
    2026-02-14 10:17:31,790 INFO:             action_recognition
    2026-02-14 10:17:31,790 INFO:             color_light
    2026-02-14 10:17:31,790 INFO:             frames_composition
    2026-02-14 10:17:31,790 INFO:             high_level_semantic
    2026-02-14 10:17:31,790 INFO:             micro_emotion
    2026-02-14 10:17:31,790 INFO:             similarity_metrics
    2026-02-14 10:17:31,790 INFO:             text_scoring
    2026-02-14 10:17:31,790 INFO: VisualProcessor | main | PR-6: executing by DAG order (len=14)
    2026-02-14 10:17:31,790 INFO: VisualProcessor | main | core_provider core_clip start
    2026-02-14 10:17:31,791 INFO: VisualProcessor | main | core core_clip | GPU slot acquired
    2026-02-14 10:17:35,628 INFO: core_clip | sampled frames: 115 / total=515
    2026-02-14 10:17:37,425 INFO: core_clip | processed 16/115
    2026-02-14 10:18:01,564 INFO: core_clip | Determined model_size=224 from preset=openai_clip_224
    2026-02-14 10:18:01,585 INFO: core_clip | Checking cache: dp_models/bundled_models/cache/core_clip_text_embeddings/size_224/7cbe1469be01babc.npz (model_size=224, cache_key=7cbe1469...)
    2026-02-14 10:18:01,648 INFO: core_clip | Loaded 413 text embeddings from cache (dp_models/bundled_models/cache/core_clip_text_embeddings/size_224/7cbe1469be01babc.npz)
    2026-02-14 10:18:01,648 INFO: core_clip | embeddings computed | shape: (115, 512)
    2026-02-14 10:18:01,652 INFO: core_clip | stage timings (ms): image_embeddings_total=7045.0, image_frame_loading=105.8, image_inference=6048.3, image_preprocessing=882.3, initialization=3.6, text_embeddings_prep=35.9, text_inference=0.0, triton_init=25.1
    2026-02-14 10:18:01,753 INFO: core_clip | Saved result: dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_clip/embeddings.npz
    core_clip | Timing summary:
    Stage                              Time (s)        %
    ----------------------------------------------------
    image_embeddings_total               7.0450    49.4%
    image_inference                      6.0483    42.5%
    image_preprocessing                  0.8823     6.2%
    image_frame_loading                  0.1058     0.7%
    saving                               0.1014     0.7%
    text_embeddings_prep                 0.0359     0.3%
    triton_init                          0.0251     0.2%
    initialization                       0.0036     0.0%
    text_inference                       0.0000     0.0%
    2026-02-14 10:18:04,880 INFO: VisualProcessor | main | core core_clip | GPU slot released
    2026-02-14 10:18:05,203 INFO: Saved Core CLIP HTML render to /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_clip/_render/render.html
    2026-02-14 10:18:05,221 INFO: VisualProcessor | Render generated for core_clip (HTML: True)
    2026-02-14 10:18:05,235 INFO: VisualProcessor | main | core_provider content_domain start
    2026-02-14 10:18:05,241 INFO: VisualProcessor | main | core content_domain | GPU slot acquired
    2026-02-14 10:18:20,415 INFO: content_domain | wrote /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/content_domain/content_domain.npz
    2026-02-14 10:18:21,146 INFO: VisualProcessor | main | core content_domain | GPU slot released
    2026-02-14 10:18:21,198 INFO: Saved Content Domain HTML render to dp_results/youtube/test_video_1/test_run_1_no_optimizations/content_domain/_render/render.html
    2026-02-14 10:18:21,199 INFO: VisualProcessor | Render generated for content_domain (HTML: True)
    2026-02-14 10:18:21,203 INFO: VisualProcessor | main | core_provider core_depth_midas start
    2026-02-14 10:18:21,205 INFO: VisualProcessor | main | core core_depth_midas | GPU slot acquired
    2026-02-14 10:18:21,770 INFO: core_depth_midas | main | sampled frames: 115 / total=515
    2026-02-14 10:18:21,772 INFO: core_depth_midas | main | FrameManager initialized (chunk_size=64, cache_size=2)
    2026-02-14 10:18:41,703 INFO: core_depth_midas | stage timings (ms): depth_inference_total=3998.6, initialization=30.6, total=19933.8
    2026-02-14 10:18:43,626 INFO: core_depth_midas | main | Saved NPZ artifact: dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_depth_midas/depth.npz | created_at=2026-02-14T00:18:41.703840
    2026-02-14 10:18:43,788 INFO: VisualProcessor | main | core core_depth_midas | GPU slot released
    2026-02-14 10:18:46,590 INFO: Saved Core Depth MiDaS HTML render to dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_depth_midas/_render/render.html
    2026-02-14 10:18:46,729 INFO: VisualProcessor | Render generated for core_depth_midas (HTML: True)
    2026-02-14 10:18:46,734 INFO: VisualProcessor | main | core_provider core_object_detections start
    2026-02-14 10:18:46,735 INFO: VisualProcessor | main | core core_object_detections | GPU slot acquired
    2026-02-14 10:18:47,269 INFO: core_object_detections | sampled frames: 115 / total=515
    2026-02-14 10:18:51,390 INFO: core_object_detections | Resolved model path via DP_MODELS_ROOT: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_models/bundled_models/visual/yolo/yolo11x_41_best.pt
    2026-02-14 10:18:58,868 INFO: core_object_detections | YOLO | loading model: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_models/bundled_models/visual/yolo/yolo11x_41_best.pt
    2026-02-14 10:19:06,323 INFO: core_object_detections | YOLO | processed 16/115
    2026-02-14 10:19:06,935 INFO: core_object_detections | YOLO | processed 32/115
    2026-02-14 10:19:07,586 INFO: core_object_detections | YOLO | processed 48/115
    2026-02-14 10:19:08,123 INFO: core_object_detections | YOLO | processed 64/115
    2026-02-14 10:19:08,722 INFO: core_object_detections | YOLO | processed 80/115
    2026-02-14 10:19:09,330 INFO: core_object_detections | YOLO | processed 96/115
    2026-02-14 10:19:09,911 INFO: core_object_detections | YOLO | processed 112/115
    2026-02-14 10:19:10,212 INFO: core_object_detections | YOLO | processed 115/115
    2026-02-14 10:19:10,234 INFO: core_object_detections | stage timings (ms): initialization=23.0, load_deps=5.5, process_frames=18817.6, saving=18.5, total=22987.8
    2026-02-14 10:19:10,241 INFO: core_object_detections | saved NPZ artifact: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_object_detections/detections.npz
    2026-02-14 10:19:11,662 INFO: VisualProcessor | main | core core_object_detections | GPU slot released
    2026-02-14 10:19:11,837 INFO: Saved Core Object Detections HTML render to /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_object_detections/_render/render.html
    2026-02-14 10:19:11,839 INFO: VisualProcessor | Render generated for core_object_detections (HTML: True)
    2026-02-14 10:19:11,845 INFO: VisualProcessor | main | core_provider core_face_landmarks start
    2026-02-14 10:19:26,087 INFO: core_face_landmarks | sampled frames: 115 / total=515
    2026-02-14 10:19:26,135 INFO: core_face_landmarks | person-mask | person_id=0 person_frames=90 face_mesh_frames=90 radius=0
    2026-02-14 10:19:26,147 INFO: core_face_landmarks | person-mask | primary_frames=115 face_mesh_frames=90
    2026-02-14 10:19:27,787 INFO: core_face_landmarks | Models initialized
    2026-02-14 10:19:43,814 INFO: core_face_landmarks | processed 115/115 frames (parallel)
    2026-02-14 10:19:44,767 INFO: core_face_landmarks | Profiling summary:
    2026-02-14 10:19:44,767 INFO:   io.frame_load: total=0.138s, mean=0.001s, count=115
    2026-02-14 10:19:44,767 INFO:   inference.pose: total=17.985s, mean=0.156s, count=115
    2026-02-14 10:19:44,767 INFO:   inference.hands: total=7.228s, mean=0.063s, count=115
    2026-02-14 10:19:44,767 INFO:   inference.face: total=1.075s, mean=0.012s, count=90
    2026-02-14 10:19:44,767 INFO:   postproc.store: total=0.049s, mean=0.049s, count=1
    2026-02-14 10:19:44,767 INFO:   total: total=16.279s, mean=16.279s, count=1
    2026-02-14 10:19:44,767 INFO:   postproc.temporal_filter: total=0.698s, mean=0.698s, count=1
    2026-02-14 10:19:44,938 INFO: core_face_landmarks | Saved result: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_face_landmarks/landmarks.npz
    2026-02-14 10:19:45,886 INFO: VisualProcessor | Render generated for core_face_landmarks (HTML: True)
    2026-02-14 10:19:45,895 INFO: VisualProcessor | main | core_provider core_optical_flow start
    2026-02-14 10:19:45,896 INFO: VisualProcessor | main | core core_optical_flow | GPU slot acquired
    2026-02-14 10:19:46,253 INFO: core_optical_flow | sampled frames: 115 / total=515
    2026-02-14 10:19:58,831 INFO: core_optical_flow | processed 49/115
    2026-02-14 10:20:12,575 INFO: core_optical_flow | processed 115/115
    2026-02-14 10:20:12,586 INFO: core_optical_flow | stage timings (ms): flow_inference_total=1879.8, initialization=28.5, saving=4.6, total=26335.0
    2026-02-14 10:20:12,590 INFO: core_optical_flow | Saved result: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_optical_flow/flow.npz
    2026-02-14 10:20:12,644 INFO: VisualProcessor | main | core core_optical_flow | GPU slot released
    2026-02-14 10:20:12,696 INFO: Saved Core Optical Flow HTML render to dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_optical_flow/_render/render.html
    2026-02-14 10:20:12,696 INFO: VisualProcessor | Render generated for core_optical_flow (HTML: True)
    2026-02-14 10:20:12,725 INFO: VisualProcessor | main | module cut_detection start
    2026-02-14 10:20:12,728 INFO: VisualProcessor | main | module cut_detection | GPU slot acquired
    2026-02-14 10:20:42,376 INFO: cut_detection | Log level set to: INFO
    2026-02-14 10:20:42,380 INFO: cut_detection | Начало обработки 115 кадров
    2026-02-14 10:20:43,188 INFO: CLIP detector initialized (Triton + core_clip prompts)
    2026-02-14 10:20:43,189 INFO: Frame embeddings success | Time: 0.0
    2026-02-14 10:20:43,194 INFO: cut_detection | using core_optical_flow motion curve (aligned frame_indices)
    2026-02-14 10:20:44,304 INFO: Hard cuts success | Time: 1.12
    2026-02-14 10:20:44,853 INFO: Soft cuts success | Time: 0.55
    2026-02-14 10:20:45,691 INFO: Motion-based cuts success | Time: 0.84
    2026-02-14 10:20:45,691 INFO: CLIP candidate-first: 48/114 windows selected
    2026-02-14 10:21:15,822 INFO: Stylized transitions via CLIP success | Time: 30.13
    2026-02-14 10:21:15,839 INFO: Jump cuts success | Time: 0.02
    2026-02-14 10:21:15,839 INFO: Shots segmentation success | Time: 0.0
    2026-02-14 10:21:27,317 INFO: Audio processing success | Time: 11.48
    2026-02-14 10:21:27,318 INFO: Scenes grouping success | Time: 0.0
    2026-02-14 10:21:27,355 INFO: Audio assisted success | Time: 0.04
    2026-02-14 10:21:27,357 INFO: Compose success | Time: 0.0
    2026-02-14 10:21:27,358 INFO: Scene transition success | Time: 0.0
    2026-02-14 10:21:27,425 INFO: Scene whoosh transition success | Time: 0.07
    2026-02-14 10:21:27,425 INFO: stylistic edit classification success | Time: 0.0
    2026-02-14 10:21:27,445 INFO: cut_detection | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/cut_detection/cut_detection_features_2026-02-14_00-21-27-428817_48eecaf7.npz
    2026-02-14 10:21:27,446 INFO: cut_detection | model-facing NPZ saved: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/cut_detection/cut_detection_model_facing_2026-02-14_00-21-27-428817_48eecaf7.npz
    2026-02-14 10:21:27,455 INFO: cut_detection | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/cut_detection/cut_detection_features_2026-02-14_00-21-27-447595_8284ab0d.npz
    2026-02-14 10:21:27,455 INFO: cut_detection | Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/cut_detection/cut_detection_features_2026-02-14_00-21-27-447595_8284ab0d.npz
    2026-02-14 10:21:27,457 INFO: Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/cut_detection/cut_detection_features_2026-02-14_00-21-27-447595_8284ab0d.npz
    2026-02-14 10:21:28,633 INFO: VisualProcessor | main | module cut_detection | GPU slot released
    2026-02-14 10:21:28,664 INFO: VisualProcessor | Render generated for cut_detection (HTML: True)
    2026-02-14 10:21:28,668 INFO: VisualProcessor | main | core_provider franchise_recognition start
    2026-02-14 10:21:28,670 INFO: VisualProcessor | main | core franchise_recognition | GPU slot acquired
    ⚠ 2026-02-14 10:21:29,510 WARNING: franchise_recognition | No franchise embeddings found in Embedding Service, falling back to image search
    2026-02-14 10:21:30,226 INFO: franchise_recognition | Embedding Service test request successful, proceeding with all frames
    2026-02-14 10:21:36,863 INFO: franchise_recognition | Saved results: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/franchise_recognition/franchise_recognition.npz (franchises=0, frames=115)
    2026-02-14 10:21:36,938 INFO: VisualProcessor | main | core franchise_recognition | GPU slot released
    2026-02-14 10:21:36,980 INFO: Saved Franchise Recognition HTML render to dp_results/youtube/test_video_1/test_run_1_no_optimizations/franchise_recognition/_render/render.html
    2026-02-14 10:21:36,980 INFO: VisualProcessor | Render generated for franchise_recognition (HTML: True)
    2026-02-14 10:21:36,984 INFO: VisualProcessor | main | core_provider ocr_extractor start
    2026-02-14 10:23:23,882 INFO: ocr_extractor | wrote /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/ocr_extractor/ocr.npz (rows=87)
    2026-02-14 10:23:23,956 INFO: Saved OCR Extractor HTML render to dp_results/youtube/test_video_1/test_run_1_no_optimizations/ocr_extractor/_render/render.html
    2026-02-14 10:23:23,956 INFO: VisualProcessor | Render generated for ocr_extractor (HTML: True)
    2026-02-14 10:23:23,961 INFO: VisualProcessor | main | module scene_classification start
    2026-02-14 10:23:23,964 INFO: VisualProcessor | main | module scene_classification | GPU slot acquired
    2026-02-14 10:23:37,966 INFO: scene_classification | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/scene_classification/scene_classification_features.npz
    2026-02-14 10:23:37,970 INFO: Готово. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/scene_classification/scene_classification_features.npz
    2026-02-14 10:23:38,656 INFO: VisualProcessor | main | module scene_classification | GPU slot released
    2026-02-14 10:23:38,728 INFO: VisualProcessor | Render generated for scene_classification (HTML: True)
    2026-02-14 10:23:38,736 INFO: VisualProcessor | main | module shot_quality start
    2026-02-14 10:23:39,233 INFO: shot_quality | device=cuda requested, but implementation is numpy-only (CPU).
    2026-02-14 10:23:39,233 INFO: shot_quality | process | Начало обработки: frames=115, device=cuda
    2026-02-14 10:23:39,536 INFO: shot_quality | process | Загружены зависимости: 5
    2026-02-14 10:23:39,536 INFO: shot_quality | process | Зависимости найдены: core_clip=True, core_depth=True, core_det=True, core_lm=True, cut_det=True
    2026-02-14 10:23:39,536 INFO: shot_quality | process | Валидация frame_indices для всех core providers
    2026-02-14 10:23:39,537 INFO: shot_quality | process | core_face_landmarks: has_any_face=True, faces_detected=72/115
    2026-02-14 10:23:39,538 INFO: shot_quality | process | Вычисление CLIP-based quality probabilities
    2026-02-14 10:23:39,538 INFO: shot_quality | process | CLIP matmul: frames=115, prompts=10, chunk_size=2048, device=cuda
    2026-02-14 10:23:39,538 INFO: shot_quality | process | CLIP quality probabilities вычислены: shape=(115, 10)
    2026-02-14 10:23:39,540 INFO: shot_quality | process | Начало извлечения per-frame features: frames=115
    2026-02-14 10:23:44,290 INFO: shot_quality | process | Обработка кадров: 11/115 (9.6%), rate=2.3 fps, elapsed=4.8s
    2026-02-14 10:23:48,072 INFO: shot_quality | process | Обработка кадров: 22/115 (19.1%), rate=2.6 fps, elapsed=8.5s
    2026-02-14 10:23:51,947 INFO: shot_quality | process | Обработка кадров: 33/115 (28.7%), rate=2.7 fps, elapsed=12.4s
    2026-02-14 10:23:55,526 INFO: shot_quality | process | Обработка кадров: 44/115 (38.3%), rate=2.8 fps, elapsed=16.0s
    2026-02-14 10:23:59,047 INFO: shot_quality | process | Обработка кадров: 55/115 (47.8%), rate=2.8 fps, elapsed=19.5s
    2026-02-14 10:24:02,934 INFO: shot_quality | process | Обработка кадров: 66/115 (57.4%), rate=2.8 fps, elapsed=23.4s
    2026-02-14 10:24:06,657 INFO: shot_quality | process | Обработка кадров: 77/115 (67.0%), rate=2.8 fps, elapsed=27.1s
    2026-02-14 10:24:10,444 INFO: shot_quality | process | Обработка кадров: 88/115 (76.5%), rate=2.8 fps, elapsed=30.9s
    2026-02-14 10:24:14,063 INFO: shot_quality | process | Обработка кадров: 99/115 (86.1%), rate=2.9 fps, elapsed=34.5s
    2026-02-14 10:24:17,866 INFO: shot_quality | process | Обработка кадров: 110/115 (95.7%), rate=2.9 fps, elapsed=38.3s
    2026-02-14 10:24:19,507 INFO: shot_quality | process | Обработка кадров: 115/115 (100.0%), rate=2.9 fps, elapsed=40.0s
    2026-02-14 10:24:19,834 INFO: shot_quality | process | Per-frame features извлечены: frames=115, elapsed=40.29s, avg=350.4ms/frame
    2026-02-14 10:24:19,835 INFO: shot_quality | process | Построение shot boundaries из cut_detection
    2026-02-14 10:24:19,835 INFO: shot_quality | process | Shot segmentation: shots=5, frames_per_shot_avg=23.0
    2026-02-14 10:24:19,835 INFO: shot_quality | process | Агрегация per-shot features: shots=5, feature_dim=48
    2026-02-14 10:24:19,838 INFO: shot_quality | process | Обработка завершена: frames=115, shots=5, features=48, quality_prompts=10, faces_available=True
    2026-02-14 10:24:19,855 INFO: shot_quality | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/shot_quality/shot_quality.npz
    2026-02-14 10:24:19,858 INFO: Готово. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/shot_quality/shot_quality.npz
    2026-02-14 10:24:20,108 INFO: VisualProcessor | Render generated for shot_quality (HTML: True)
    2026-02-14 10:24:20,114 INFO: VisualProcessor | main | module story_structure start
    2026-02-14 10:24:40,991 INFO: story_structure | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/story_structure/story_structure.npz
    2026-02-14 10:24:40,994 INFO: Готово. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/story_structure/story_structure.npz
    2026-02-14 10:24:41,800 INFO: VisualProcessor | Render generated for story_structure (HTML: True)
    2026-02-14 10:24:41,805 INFO: VisualProcessor | main | module uniqueness start
    2026-02-14 10:24:42,203 INFO: uniqueness | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/uniqueness/uniqueness.npz
    2026-02-14 10:24:42,211 INFO: Готово. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/uniqueness/uniqueness.npz
    2026-02-14 10:24:42,363 INFO: VisualProcessor | Render generated for uniqueness (HTML: True)
    2026-02-14 10:24:42,378 INFO: VisualProcessor | main | module video_pacing start
    2026-02-14 10:24:45,339 INFO: video_pacing | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/video_pacing/video_pacing_features.npz
    2026-02-14 10:24:45,341 INFO: Готово. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/video_pacing/video_pacing_features.npz
    2026-02-14 10:24:45,509 INFO: VisualProcessor | Render generated for video_pacing (HTML: True)
    ⚠ 2026-02-14 10:24:45,514 WARNING: VisualProcessor | main | PR-6: exec_order missing enabled components: ['action_recognition', 'behavioral', 'brand_semantics', 'car_semantics', 'color_light', 'detalize_face', 'emotion_face', 'face_identity', 'frames_composition', 'high_level_semantic', 'micro_emotion', 'optical_flow', 'place_semantics', 'similarity_metrics', 'text_scoring']
    2026-02-14 10:24:45,514 INFO: VisualProcessor | main | module action_recognition start
    2026-02-14 10:24:45,540 INFO: VisualProcessor | main | module action_recognition | GPU slot acquired
    2026-02-14 10:25:20,862 INFO: Инициализация SlowFast R50 через ModelManager (local-only)
    2026-02-14 10:25:31,909 INFO: action_recognition готов | clip_len=32 stride=16 batch_size=8 embedding_dim=256 device=cuda
    2026-02-14 10:25:31,945 INFO: action_recognition | Начало обработки 250 кадров
    2026-02-14 10:25:31,964 INFO: action_recognition | _prepare_tracks: начало загрузки detections.npz
    2026-02-14 10:25:31,996 INFO: action_recognition | _prepare_tracks: detections_data загружен, ключи: ['frame_indices', 'boxes', 'scores', 'class_ids', 'valid_mask', 'times_s', 'class_names', 'meta']
    2026-02-14 10:25:31,996 INFO: action_recognition | _prepare_tracks: загружено 115 кадров детекций, class_ids shape=(115, 100), valid_mask shape=(115, 100)
    2026-02-14 10:25:31,997 INFO: action_recognition | _prepare_tracks: найдено 26 кадров с person детекциями
    2026-02-14 10:25:31,998 INFO: action_recognition | _prepare_tracks: создано 18 сегментов из кадров с person детекциями
    2026-02-14 10:25:32,068 INFO: Начинаем извлечение эмбеддингов: clips=18 batch_size=8
    2026-02-14 10:25:46,651 INFO: Progress: 44.4% (8/18 clips)
    2026-02-14 10:25:48,197 INFO: Progress: 88.9% (16/18 clips)
    2026-02-14 10:25:48,547 INFO: Progress: 100.0% (18/18 clips)
    2026-02-14 10:25:48,550 INFO: Обработано треков: 18
    2026-02-14 10:25:48,618 INFO: action_recognition | Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/action_recognition/action_recognition_emb.npz
    2026-02-14 10:25:48,618 INFO: Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/action_recognition/action_recognition_emb.npz
    2026-02-14 10:25:51,625 INFO: VisualProcessor | main | module action_recognition | GPU slot released
    2026-02-14 10:25:51,692 INFO: VisualProcessor | Render generated for action_recognition (HTML: True)
    2026-02-14 10:25:51,725 INFO: VisualProcessor | main | module behavioral start
    2026-02-14 10:25:52,201 INFO: behavioral | Начало обработки 250 кадров
    2026-02-14 10:25:52,280 INFO: behavioral | Обработано кадров: 20/250 | Time: 0.04
    ⚠ 2026-02-14 10:25:52,313 WARNING: behavioral | process | 216 кадров отсутствуют в core_face_landmarks. Заполнены NaN и отмечены masks.
    2026-02-14 10:25:52,363 INFO: behavioral | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/behavioral/behavioral_features.npz
    2026-02-14 10:25:52,363 INFO: behavioral | Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/behavioral/behavioral_features.npz
    2026-02-14 10:25:52,486 INFO: VisualProcessor | Render generated for behavioral (HTML: True)
    2026-02-14 10:25:52,495 INFO: VisualProcessor | main | core_provider brand_semantics start
    ⚠ 2026-02-14 10:25:54,249 WARNING: brand_semantics | tracks not found in detections.npz. Generating per-detection track IDs - this may produce incorrect results. Each detection will be treated as a separate track. Consider ensuring core_object_detections provides proper tracking.
    ✗ 2026-02-14 10:25:54,412 WARNING: brand_semantics | Embedding Service test request failed: Embedding Service search failed after 1 attempts: 500 Server Error: Internal Server Error for url: http://localhost:8005/search. Skipping all tracks to avoid repeated errors. Check Embedding Service status and ensure category 'brand' is configured.
    ⚠ 2026-02-14 10:25:54,412 WARNING: brand_semantics | Embedding Service unavailable, skipping 6 tracks
    2026-02-14 10:25:54,428 INFO: brand_semantics | Saved results: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/brand_semantics/brand_semantics.npz (tracks=0, brands=0, frames=115)
    2026-02-14 10:25:54,559 INFO: VisualProcessor | Render generated for brand_semantics (HTML: True)
    2026-02-14 10:25:54,576 INFO: VisualProcessor | main | core_provider car_semantics start
    ⚠ 2026-02-14 10:25:55,386 WARNING: car_semantics | car class not found in taxonomy, using all detections
    ⚠ 2026-02-14 10:25:55,386 WARNING: car_semantics | tracks not found in detections.npz. Generating per-detection track IDs - this may produce incorrect results. Each detection will be treated as a separate track. Consider ensuring core_object_detections provides proper tracking.
    ✗ 2026-02-14 10:25:55,504 WARNING: car_semantics | Embedding Service test request failed: Embedding Service search failed after 1 attempts: 500 Server Error: Internal Server Error for url: http://localhost:8005/search. Skipping all tracks to avoid repeated errors. Check Embedding Service status and ensure category 'car' is configured.
    ⚠ 2026-02-14 10:25:55,504 WARNING: car_semantics | Embedding Service unavailable, skipping 289 tracks
    2026-02-14 10:25:55,513 INFO: car_semantics | Saved results: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/car_semantics/car_semantics.npz (tracks=0, cars=0, frames=115)
    2026-02-14 10:25:55,589 INFO: Saved Car Semantics HTML render to /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/car_semantics/_render/render.html
    2026-02-14 10:25:55,589 INFO: VisualProcessor | Render generated for car_semantics (HTML: True)
    2026-02-14 10:25:55,598 INFO: VisualProcessor | main | module color_light start
    2026-02-14 10:26:02,158 INFO: color_light | Начало обработки 250 кадров
    2026-02-14 10:26:03,322 INFO: Сцена 1/4 | Кадр 1/19 обработан
    2026-02-14 10:26:03,659 INFO: Сцена 1/4 | Кадр 2/19 обработан
    2026-02-14 10:26:04,015 INFO: Сцена 1/4 | Кадр 3/19 обработан
    2026-02-14 10:26:04,277 INFO: Сцена 1/4 | Кадр 4/19 обработан
    2026-02-14 10:26:04,528 INFO: Сцена 1/4 | Кадр 5/19 обработан
    2026-02-14 10:26:04,825 INFO: Сцена 1/4 | Кадр 6/19 обработан
    2026-02-14 10:26:05,035 INFO: Сцена 1/4 | Кадр 7/19 обработан
    2026-02-14 10:26:05,453 INFO: Сцена 1/4 | Кадр 8/19 обработан
    2026-02-14 10:26:05,811 INFO: Сцена 1/4 | Кадр 9/19 обработан
    2026-02-14 10:26:06,076 INFO: Сцена 1/4 | Кадр 10/19 обработан
    2026-02-14 10:26:06,382 INFO: Сцена 1/4 | Кадр 11/19 обработан
    2026-02-14 10:26:06,580 INFO: Сцена 1/4 | Кадр 12/19 обработан
    2026-02-14 10:26:06,823 INFO: Сцена 1/4 | Кадр 13/19 обработан
    2026-02-14 10:26:07,077 INFO: Сцена 1/4 | Кадр 14/19 обработан
    2026-02-14 10:26:07,335 INFO: Сцена 1/4 | Кадр 15/19 обработан
    2026-02-14 10:26:07,611 INFO: Сцена 1/4 | Кадр 16/19 обработан
    2026-02-14 10:26:07,826 INFO: Сцена 1/4 | Кадр 17/19 обработан
    2026-02-14 10:26:08,195 INFO: Сцена 1/4 | Кадр 18/19 обработан
    2026-02-14 10:26:08,444 INFO: Сцена 1/4 | Кадр 19/19 обработан
    2026-02-14 10:26:08,449 INFO: Сцена 1/4 | Scene-level фичи извлечены
    2026-02-14 10:26:08,705 INFO: Сцена 2/4 | Кадр 1/4 обработан
    2026-02-14 10:26:08,969 INFO: Сцена 2/4 | Кадр 2/4 обработан
    2026-02-14 10:26:09,237 INFO: Сцена 2/4 | Кадр 3/4 обработан
    2026-02-14 10:26:09,492 INFO: Сцена 2/4 | Кадр 4/4 обработан
    2026-02-14 10:26:09,498 INFO: Сцена 2/4 | Scene-level фичи извлечены
    2026-02-14 10:26:09,734 INFO: Сцена 3/4 | Кадр 1/5 обработан
    2026-02-14 10:26:10,034 INFO: Сцена 3/4 | Кадр 2/5 обработан
    2026-02-14 10:26:10,307 INFO: Сцена 3/4 | Кадр 3/5 обработан
    2026-02-14 10:26:10,684 INFO: Сцена 3/4 | Кадр 4/5 обработан
    2026-02-14 10:26:10,978 INFO: Сцена 3/4 | Кадр 5/5 обработан
    2026-02-14 10:26:10,983 INFO: Сцена 3/4 | Scene-level фичи извлечены
    2026-02-14 10:26:11,240 INFO: Сцена 4/4 | Кадр 1/6 обработан
    2026-02-14 10:26:11,491 INFO: Сцена 4/4 | Кадр 2/6 обработан
    2026-02-14 10:26:12,066 INFO: Сцена 4/4 | Кадр 3/6 обработан
    2026-02-14 10:26:12,317 INFO: Сцена 4/4 | Кадр 4/6 обработан
    2026-02-14 10:26:12,620 INFO: Сцена 4/4 | Кадр 5/6 обработан
    2026-02-14 10:26:13,001 INFO: Сцена 4/4 | Кадр 6/6 обработан
    2026-02-14 10:26:13,006 INFO: Сцена 4/4 | Scene-level фичи извлечены
    2026-02-14 10:26:13,015 INFO: Видео фичи извлечены
    2026-02-14 10:26:13,034 INFO: color_light | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/color_light/color_light_features.npz
    2026-02-14 10:26:13,037 INFO: color_light | Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/color_light/color_light_features.npz
    2026-02-14 10:26:13,038 INFO: Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/color_light/color_light_features.npz
    2026-02-14 10:26:13,050 INFO: Ключевые метрики:
    2026-02-14 10:26:13,050 INFO:   - Cinematic Lighting Score: nan
    2026-02-14 10:26:13,050 INFO:   - Professional Look Score: nan
    2026-02-14 10:26:13,051 INFO:   - Teal & Orange Style: 0.126
    2026-02-14 10:26:13,051 INFO:   - Color Distribution Entropy: 0.000
    2026-02-14 10:26:13,325 INFO: VisualProcessor | Render generated for color_light (HTML: True)
    2026-02-14 10:26:13,330 INFO: VisualProcessor | main | module detalize_face start
    2026-02-14 10:26:16,226 INFO: DetalizeFaceExtractorRefactored | init | Используем core_face_landmarks (max_faces = 4)
    2026-02-14 10:26:16,226 INFO: DetalizeFaceExtractorRefactored | init | load modules...
    2026-02-14 10:26:16,226 INFO: DetalizeFaceExtractorRefactored | init | Загружен модуль: geometry
    2026-02-14 10:26:16,226 INFO: DetalizeFaceExtractorRefactored | init | Загружен модуль: pose
    2026-02-14 10:26:16,226 INFO: DetalizeFaceExtractorRefactored | init | Загружен модуль: quality
    2026-02-14 10:26:16,226 INFO: DetalizeFaceExtractorRefactored | init | Загружен модуль: eyes
    2026-02-14 10:26:16,227 INFO: DetalizeFaceExtractorRefactored | init | Загружен модуль: motion
    2026-02-14 10:26:16,227 INFO: DetalizeFaceExtractorRefactored | init | Загружен модуль: structure
    2026-02-14 10:26:16,227 INFO: DetalizeFaceExtractorRefactored | init | Загружен модуль: lip_reading
    2026-02-14 10:26:16,243 INFO: DetalizeFaceModule | core_face_landmarks loaded: frames_with_faces=72
    2026-02-14 10:26:16,244 INFO: VisualProcessor | detalize_face | main | Запуск module (modules=auto)
    2026-02-14 10:26:16,257 INFO: DetalizeFaceExtractorRefactored | extract | processing 72 frames with 7 modules
    2026-02-14 10:26:16,469 INFO: DetalizeFaceExtractorRefactored | extract | completed: 72 frames in 211.6ms (avg 2.94ms/frame, 7 modules)
    2026-02-14 10:26:16,488 INFO: DetalizeFaceModule | Обработка завершена: обработано 72 кадров, найдено лиц: 72
    2026-02-14 10:26:16,502 INFO: detalize_face | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/detalize_face/detalize_face.npz
    2026-02-14 10:26:16,504 INFO: VisualProcessor | detalize_face | main | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/detalize_face/detalize_face.npz
    2026-02-14 10:26:16,759 INFO: VisualProcessor | Render generated for detalize_face (HTML: True)
    2026-02-14 10:26:16,763 INFO: VisualProcessor | main | module emotion_face start
    2026-02-14 10:26:16,764 INFO: VisualProcessor | main | module emotion_face | GPU slot acquired
    2026-02-14 10:26:38,062 INFO: emotion_face | Loading EmoNet from fallback path: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_models/bundled_models/visual/emonet/emonet_8.pth
    2026-02-14 10:26:42,387 INFO: emotion_face | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/emotion_face/emotion_face.npz
    2026-02-14 10:26:42,390 INFO: Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/emotion_face/emotion_face.npz
    2026-02-14 10:26:43,771 INFO: VisualProcessor | main | module emotion_face | GPU slot released
    2026-02-14 10:26:43,816 INFO: VisualProcessor | Render generated for emotion_face (HTML: True)
    2026-02-14 10:26:43,825 INFO: VisualProcessor | main | core_provider face_identity start
    ✗ 2026-02-14 10:26:59,450 WARNING: core_face_identity | Embedding Service test request failed: Embedding Service search failed after 1 attempts: 500 Server Error: Internal Server Error for url: http://localhost:8005/search. Skipping all frames to avoid repeated errors. Check Embedding Service status and ensure category 'face' is configured.
    ⚠ 2026-02-14 10:26:59,450 WARNING: core_face_identity | Embedding Service unavailable, skipping 72 frames
    2026-02-14 10:26:59,458 INFO: core_face_identity | Saved results: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/core_face_identity/face_identity.npz (frames=72, faces_processed=0)
    2026-02-14 10:26:59,510 INFO: VisualProcessor | main | module frames_composition start
    2026-02-14 10:26:59,952 INFO: frames_composition | Начало обработки 115 кадров
    2026-02-14 10:27:02,945 INFO: frames_composition | processed frames=115 workers=4 elapsed_ms=2718
    2026-02-14 10:27:02,969 INFO: frames_composition | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/frames_composition/frames_composition.npz
    2026-02-14 10:27:02,969 INFO: frames_composition | Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/frames_composition/frames_composition.npz
    2026-02-14 10:27:02,970 INFO: Done. Saved: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/frames_composition/frames_composition.npz
    2026-02-14 10:27:03,088 INFO: VisualProcessor | Render generated for frames_composition (HTML: True)
    2026-02-14 10:27:03,093 INFO: VisualProcessor | main | module high_level_semantic start
    2026-02-14 10:27:03,442 INFO: high_level_semantic | Начало обработки 115 кадров
    2026-02-14 10:27:03,590 INFO: high_level_semantic | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/high_level_semantic/high_level_semantic.npz
    2026-02-14 10:27:03,590 INFO: high_level_semantic | Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/high_level_semantic/high_level_semantic.npz
    2026-02-14 10:27:03,590 INFO: Обработка завершена. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/high_level_semantic/high_level_semantic.npz
    2026-02-14 10:27:03,670 INFO: VisualProcessor | Render generated for high_level_semantic (HTML: True)
    2026-02-14 10:27:03,678 INFO: VisualProcessor | main | module micro_emotion start
    2026-02-14 10:27:03,680 INFO: VisualProcessor | main | module micro_emotion | GPU slot acquired
    ⚠ 2026-02-14 10:27:24,382 WARNING: micro_emotion | OpenFace mapping contains 1 invalid rows (frame_union < 0); dropping these rows and continuing best-effort
    ⚠ 2026-02-14 10:27:24,395 WARNING: micro_emotion | partial OpenFace results: missing=[0] extra=[] (continuing best-effort)
    2026-02-14 10:27:24,507 INFO: micro_emotion | done: primary=250 face_frames=21 openface_rows=20
    2026-02-14 10:27:24,531 INFO: micro_emotion | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/micro_emotion/micro_emotion.npz
    2026-02-14 10:27:24,762 INFO: VisualProcessor | main | module micro_emotion | GPU slot released
    2026-02-14 10:27:24,847 INFO: VisualProcessor | Render generated for micro_emotion (HTML: True)
    2026-02-14 10:27:24,857 INFO: VisualProcessor | main | module optical_flow start
    ⚠ 2026-02-14 10:27:25,188 WARNING: optical_flow | 216/250 frame_indices not found in core_optical_flow. Using NaN for missing frames (will be ignored in statistics).
    2026-02-14 10:27:25,213 INFO: optical_flow | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/optical_flow/optical_flow.npz
    2026-02-14 10:27:25,215 INFO: Готово. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/optical_flow/optical_flow.npz
    2026-02-14 10:27:25,279 INFO: VisualProcessor | Render generated for optical_flow (HTML: True)
    2026-02-14 10:27:25,285 INFO: VisualProcessor | main | core_provider place_semantics start
    ✗ 2026-02-14 10:27:25,992 WARNING: place_semantics | Embedding Service test request failed: Embedding Service search failed after 1 attempts: 500 Server Error: Internal Server Error for url: http://localhost:8005/search. Skipping all frames to avoid repeated errors. Check Embedding Service status and ensure category 'place' is configured.
    ⚠ 2026-02-14 10:27:25,992 WARNING: place_semantics | Embedding Service unavailable, filling 115 frames with empty results
    2026-02-14 10:27:26,001 INFO: place_semantics | Saved results: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/place_semantics/place_semantics.npz (tracks=0, places=0, frames=115)
    2026-02-14 10:27:26,089 INFO: Saved Place Semantics HTML render to dp_results/youtube/test_video_1/test_run_1_no_optimizations/place_semantics/_render/render.html
    2026-02-14 10:27:26,089 INFO: VisualProcessor | Render generated for place_semantics (HTML: True)
    2026-02-14 10:27:26,093 INFO: VisualProcessor | main | module similarity_metrics start
    ⚠ 2026-02-14 10:27:26,459 WARNING: similarity_metrics | frame_indices from metadata (120 unique) differ from core_clip frame_indices (115 unique). Using core_clip indices as source-of-truth.
    ⚠ /media/ilya/Новый том1/TrendFlowML/DataProcessor/VisualProcessor/modules/similarity_metrics/similarity_metrics.py:436: RuntimeWarning: Mean of empty slice
    v = np.nanmean(np.asarray(feats_mean, dtype=np.float32), axis=0)
    2026-02-14 10:27:26,549 INFO: similarity_metrics | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/similarity_metrics/results.npz
    2026-02-14 10:27:26,549 INFO: Готово. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/similarity_metrics/results.npz
    2026-02-14 10:27:26,672 INFO: VisualProcessor | Render generated for similarity_metrics (HTML: True)
    2026-02-14 10:27:26,677 INFO: VisualProcessor | main | module text_scoring start
    2026-02-14 10:27:28,069 INFO: text_scoring | Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/text_scoring/text_scoring.npz
    2026-02-14 10:27:28,071 INFO: Готово. Результаты сохранены: /media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results/youtube/test_video_1/test_run_1_no_optimizations/text_scoring/text_scoring.npz
    2026-02-14 10:27:28,238 INFO: VisualProcessor | Render generated for text_scoring (HTML: True)
  [✓ OK] VisualProcessor завершен
        время: 621882ms

---

*Примечание (апрель 2026):* артефакт `action_recognition` сохраняется как `action_recognition_features.npz`; в логах до этого изменения может фигурировать `action_recognition_emb.npz`.