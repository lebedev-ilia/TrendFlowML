ilya@ilya-B450M-DS3H:/media/ilya/Новый том1/TrendFlowML/DataProcessor$ # auth token export removed from log
ilya@ilya-B450M-DS3H:/media/ilya/Новый том1/TrendFlowML/DataProcessor$ export DP_MODELS_ROOT="/media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_models/bundled_models"
ilya@ilya-B450M-DS3H:/media/ilya/Новый том1/TrendFlowML/DataProcessor$ python3 main.py   --video-path "/media/ilya/Новый том1/TrendFlowML/example/example_videos/video1.mp4"   --global-config configs/global_config.yaml   --run-audio   --platform-id youtube   --video-id test_video_1   --run-id test_run_1_no_optimizations --output-dir "/media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results"

================================================================================
  Segmenter
================================================================================
  [✓ OK] Запуск Segmenter
        .../TrendFlowML/example/example_videos/video1.mp4
  → Segmenter: построено 29 конфигов из /tmp/visual_cfg_3fw9eeea.yaml
  → Segmenter: обработка .../TrendFlowML/example/example_videos/video1.mp4
    primary sampling group budget: total_frames_source=863 fps=30.000 duration_s=28.8 requested_max=500 target_gap_sec=0.25 rate_fps=4.0 budget_n=115 chosen_n=115
    primary sampling group: set core_clip.frame_indices_source = N=115
    primary sampling group: set core_object_detections.frame_indices_source = N=115
    primary sampling group: set core_depth_midas.frame_indices_source = N=115
    primary sampling group: set core_face_landmarks.frame_indices_source = N=115
    primary sampling group: set core_optical_flow.frame_indices_source = N=115
    primary sampling group: set shot_quality.frame_indices_source = N=115
    primary sampling group: set frames_composition.frame_indices_source = N=115
    core_optical_flow reuse policy: set cut_detection.frame_indices_source = core_optical_flow (N=115)
    deps sampling align: ocr_extractor ⊆ core_object_detections | 250 -> 115 (parent=115)
    deps sampling align: content_domain ⊆ core_clip | 250 -> 115 (parent=115)
    deps sampling align: franchise_recognition ⊆ core_clip | 250 -> 115 (parent=115)
    deps sampling align: scene_classification ⊆ core_clip | 250 -> 115 (parent=115)
    deps sampling align: video_pacing ⊆ core_optical_flow | 120 -> 115 (parent=115)
    deps sampling align: uniqueness ⊆ core_clip | 120 -> 115 (parent=115)
    deps sampling align: story_structure ⊆ core_clip | 120 -> 115 (parent=115)
    deps sampling align: high_level_semantic ⊆ cut_detection | 250 -> 115 (parent=115)
    ✓ batch_00000.npy
    ✓ batch_00001.npy
    ✓ batch_00002.npy
    ✓ batch_00003.npy
    ✓ batch_00004.npy
    ✓ batch_00005.npy
    ✓ batch_00006.npy
    ✓ batch_00007.npy
    ✓ batch_00008.npy
    ✓ Segmenter/data/test_video_1/audio/audio.wav
    ✓ .../data/test_video_1/audio/audio.wav', 'duration_sec': 28.8, 'sample_rate': 22050, 'total_samples': 635040}
    ✓ Сохранено: Segmenter/data/test_video_1/audio/segments.json
    union mode done: union_frames=515 -> /media/ilya/Новый том1/TrendFlowML/DataProcessor/Segmenter/data/test_video_1/video/metadata.json
  [✓ OK] Segmenter завершен
        время: 92560ms

================================================================================
  AudioProcessor
================================================================================
  [✓ OK] Запуск AudioProcessor
    AudioProcessor | Initializing MainProcessor and extractors...
    AudioProcessor | MainProcessor initialized (0.00s, 0 extractors)
    AudioProcessor | [  5%] Loading input (0.00s, total: 0.00s)
    AudioProcessor | [ 10%] Running extractors (0.00s, total: 0.00s)
    AudioProcessor | [ 10%] mel: Processing segment 1/29
    AudioProcessor | [ 10%] mel: Processing segment 3/29
    AudioProcessor | [ 13%] mel: Completed
    AudioProcessor | [ 13%] mel: Running extractors (3.38s, total: 5.48s)
    AudioProcessor | [ 13%] onset: Processing segment 1/29
    AudioProcessor | [ 13%] onset: Processing segment 3/29
    AudioProcessor | [ 16%] onset: Processing segment 27/29
    AudioProcessor | [ 16%] onset: Running extractors (88.11s, total: 93.70s)
    Added OmegaConf classes to torch safe globals (runtime workaround).
    AudioProcessor | [ 20%] speaker_diarization: Running extractors (269.31s, total: 366.95s)
    AudioProcessor | [ 20%] mfcc: Processing segment 1/29
    AudioProcessor | [ 20%] mfcc: Processing segment 7/29
    AudioProcessor | [ 23%] mfcc: Completed
    AudioProcessor | [ 23%] mfcc: Running extractors (1.51s, total: 368.72s)
    AudioProcessor | [ 25%] emotion_diarization: Loading audio (1.50s, total: 565.12s)
    AudioProcessor | [ 28%] emotion_diarization: Preprocessing (3.00s, total: 566.62s)
    AudioProcessor | [ 30%] emotion_diarization: Preprocessing (4.50s, total: 568.12s)
    AudioProcessor | [ 33%] emotion_diarization: Inference: 6.0s (6.01s, total: 569.62s)
    AudioProcessor | [ 36%] emotion_diarization: Inference: 7.5s (7.51s, total: 571.12s)
    AudioProcessor | [ 38%] emotion_diarization: Inference: 9.0s (9.01s, total: 572.62s)
    AudioProcessor | [ 41%] emotion_diarization: Inference: 10.5s (10.51s, total: 574.12s)
    AudioProcessor | [ 44%] emotion_diarization: Inference: 12.0s (12.01s, total: 575.62s)
    AudioProcessor | [ 46%] emotion_diarization: Inference: 13.5s (13.51s, total: 577.13s)
    AudioProcessor | [ 49%] emotion_diarization: Inference: 15.0s (15.01s, total: 578.63s)
    AudioProcessor | [ 51%] emotion_diarization: Inference: 16.5s (16.51s, total: 580.13s)
    AudioProcessor | [ 54%] emotion_diarization: Inference: 18.0s (18.02s, total: 581.63s)
    AudioProcessor | [ 57%] emotion_diarization: Inference: 19.5s (19.52s, total: 583.13s)
    AudioProcessor | [ 59%] emotion_diarization: Inference: 21.0s (21.02s, total: 584.63s)
    AudioProcessor | [ 62%] emotion_diarization: Inference: 22.5s (22.52s, total: 586.13s)
    AudioProcessor | [ 65%] emotion_diarization: Inference: 24.0s (24.02s, total: 587.64s)
    AudioProcessor | [ 67%] emotion_diarization: Inference: 25.5s (25.52s, total: 589.14s)
    AudioProcessor | [ 70%] emotion_diarization: Inference: 27.0s (27.02s, total: 590.64s)
    AudioProcessor | [ 72%] emotion_diarization: Inference: 28.5s (28.53s, total: 592.14s)
    AudioProcessor | [ 75%] emotion_diarization: Inference: 30.0s (30.03s, total: 593.64s)
    AudioProcessor | [ 78%] emotion_diarization: Inference: 31.5s (31.53s, total: 595.14s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 33.0s (33.03s, total: 596.64s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 34.5s (34.53s, total: 598.15s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 36.0s (36.03s, total: 599.65s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 37.5s (37.53s, total: 601.15s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 39.0s (39.04s, total: 602.65s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 40.5s (40.54s, total: 604.15s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 42.0s (42.04s, total: 605.65s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 43.5s (43.54s, total: 607.15s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 45.0s (45.04s, total: 608.65s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 46.5s (46.54s, total: 610.16s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 48.0s (48.04s, total: 611.66s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 49.5s (49.54s, total: 613.16s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 51.0s (51.05s, total: 614.66s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 52.5s (52.55s, total: 616.16s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 54.1s (54.05s, total: 617.66s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 55.6s (55.55s, total: 619.17s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 57.1s (57.05s, total: 620.67s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 58.6s (58.55s, total: 622.17s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 60.1s (60.05s, total: 623.67s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 61.6s (61.56s, total: 625.17s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 63.1s (63.06s, total: 626.67s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 64.6s (64.56s, total: 628.17s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 66.1s (66.06s, total: 629.67s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 67.6s (67.56s, total: 631.18s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 69.1s (69.06s, total: 632.68s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 70.6s (70.56s, total: 634.18s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 72.1s (72.07s, total: 635.68s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 73.6s (73.57s, total: 637.18s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 75.1s (75.07s, total: 638.68s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 76.6s (76.57s, total: 640.18s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 78.1s (78.07s, total: 641.69s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 79.6s (79.57s, total: 643.19s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 81.1s (81.07s, total: 644.69s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 82.6s (82.58s, total: 646.19s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 84.1s (84.08s, total: 647.69s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 85.6s (85.58s, total: 649.19s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 87.1s (87.08s, total: 650.69s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 88.6s (88.58s, total: 652.20s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 90.1s (90.08s, total: 653.70s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 91.6s (91.58s, total: 655.20s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 93.1s (93.08s, total: 656.70s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 94.6s (94.59s, total: 658.20s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 96.1s (96.09s, total: 659.70s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 97.6s (97.59s, total: 661.20s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 99.1s (99.09s, total: 662.70s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 100.6s (100.59s, total: 664.21s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 102.1s (102.09s, total: 665.71s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 103.6s (103.59s, total: 667.21s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 105.1s (105.10s, total: 668.71s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 106.6s (106.60s, total: 670.21s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 108.1s (108.10s, total: 671.71s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 109.6s (109.60s, total: 673.22s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 111.1s (111.10s, total: 674.72s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 112.6s (112.60s, total: 676.22s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 114.1s (114.11s, total: 677.72s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 115.6s (115.61s, total: 679.22s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 117.1s (117.11s, total: 680.72s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 118.6s (118.61s, total: 682.22s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 120.1s (120.11s, total: 683.72s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 121.6s (121.61s, total: 685.23s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 123.1s (123.11s, total: 686.73s)
    AudioProcessor | [ 80%] emotion_diarization: Inference: 124.6s (124.61s, total: 688.23s)
    AudioProcessor | [ 26%] emotion_diarization: Running extractors (124.86s, total: 688.61s)
    AudioProcessor | [ 26%] tempo: Starting tempo estimation: 0/30 segments (0%)
    AudioProcessor | [ 27%] tempo: Processing segments: 6/30 (20%)
    AudioProcessor | [ 27%] tempo: Processing segments: 12/30 (40%)
    ⚠ /media/ilya/Новый том1/TrendFlowML/DataProcessor/AudioProcessor/.ap_venv/lib/python3.10/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=2048 is too large for input signal of length=1
    ⚠ warnings.warn(
    AudioProcessor | [ 29%] tempo: Processing segments: 24/30 (80%)
    AudioProcessor | [ 30%] tempo: Running extractors (1.84s, total: 690.65s)
    AudioProcessor | [ 30%] chroma: Processing segment 1/29
    ⚠ /media/ilya/Новый том1/TrendFlowML/DataProcessor/AudioProcessor/.ap_venv/lib/python3.10/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=1024 is too large for input signal of length=690
    ⚠ warnings.warn(
    ⚠ /media/ilya/Новый том1/TrendFlowML/DataProcessor/AudioProcessor/.ap_venv/lib/python3.10/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=1024 is too large for input signal of length=345
    ⚠ warnings.warn(
    AudioProcessor | [ 30%] chroma: Processing segment 7/29
    AudioProcessor | [ 31%] chroma: Processing segment 17/29
    AudioProcessor | [ 32%] chroma: Processing segment 27/29
    ⚠ /media/ilya/Новый том1/TrendFlowML/DataProcessor/AudioProcessor/.ap_venv/lib/python3.10/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=1024 is too large for input signal of length=735
    ⚠ warnings.warn(
    ⚠ /media/ilya/Новый том1/TrendFlowML/DataProcessor/AudioProcessor/.ap_venv/lib/python3.10/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=1024 is too large for input signal of length=368
    ⚠ warnings.warn(
    AudioProcessor | [ 33%] chroma: Running extractors (3.44s, total: 694.28s)
    AudioProcessor | [ 33%] rhythmic: Processed 1/29 segments
    AudioProcessor | [ 36%] rhythmic: Processed 29/29 segments
    AudioProcessor | [ 36%] rhythmic: Running extractors (2.87s, total: 697.38s)
    AudioProcessor | [ 36%] hpss: Processed 2/29 segments
    AudioProcessor | [ 40%] hpss: Running extractors (3.27s, total: 700.86s)
    AudioProcessor | [ 40%] quality: Processing segment 1/29
    AudioProcessor | [ 43%] quality: Completed
    AudioProcessor | [ 43%] quality: Running extractors (0.08s, total: 701.14s)
    AudioProcessor | [ 43%] clap: Starting preprocessing: 0/29 segments (0%)
    AudioProcessor | [ 46%] clap: Preprocessing segments: 29/29 (100%)
    AudioProcessor | [ 44%] clap: Running inference: 1/2 batches (50%)
    AudioProcessor | [ 46%] clap: Running inference: 2/2 batches (100%)
    AudioProcessor | [ 46%] clap: Running extractors (29.43s, total: 738.08s)
    AudioProcessor | [ 46%] pitch: Processing segment 1/29
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Processing segment 7/29
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 47%] pitch: Running PYIN
    AudioProcessor | [ 50%] pitch: Selecting best method
    AudioProcessor | [ 50%] pitch: Running extractors (18.03s, total: 756.34s)
    AudioProcessor | [ 50%] source_separation: Processing batches: Loaded 2 segments (0.1s)
    AudioProcessor | [ 53%] source_separation: Processing batches: Inference: 1/1 batches (100%, 3.6s)
    AudioProcessor | [ 53%] source_separation: Running extractors (3.70s, total: 763.94s)
    AudioProcessor | [ 56%] asr: Running extractors (5.75s, total: 780.01s)
    AudioProcessor | [ 56%] loudness: Starting loudness computation: 0/115 segments (0%)
    AudioProcessor | [ 58%] loudness: Processing segments: 58/115 (50%)
    AudioProcessor | [ 60%] loudness: Running extractors (1.14s, total: 781.37s)
    AudioProcessor | [ 60%] spectral: Processing segment 1/29
    AudioProcessor | [ 61%] spectral: Computing rolloff
    AudioProcessor | [ 63%] spectral: Running extractors (0.76s, total: 782.35s)
    [   INFO   ] MusicExtractorSVM: no classifier models were configured by default
    AudioProcessor | [ 63%] key: Loading segments
    AudioProcessor | [ 66%] key: Complete
    AudioProcessor | [ 66%] key: Running extractors (0.17s, total: 784.41s)
    AudioProcessor | [ 66%] voice_quality: Processed 2/29 segments
    AudioProcessor | [ 68%] voice_quality: Processed 20/29 segments
    AudioProcessor | [ 70%] voice_quality: Running extractors (3.39s, total: 788.02s)
    AudioProcessor | [ 73%] speech_analysis: ASR result loaded from dependency
    AudioProcessor | [ 73%] speech_analysis: Running extractors (0.00s, total: 788.21s)
    AudioProcessor | [ 73%] spectral_entropy: Processing segment 1/29
    AudioProcessor | [ 76%] spectral_entropy: Processing segment 29/29
    AudioProcessor | [ 76%] spectral_entropy: Running extractors (0.12s, total: 788.54s)
    AudioProcessor | [ 76%] band_energy: load_segments: Loading segments
    AudioProcessor | [ 79%] band_energy: aggregate: Aggregating results
    AudioProcessor | [ 80%] band_energy: Running extractors (0.23s, total: 788.99s)
    AudioProcessor | [ 80%] Saving NPZ artifacts (0.00s, total: 789.06s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 789.06s)
    AudioProcessor | [ 90%] Validating artifact (0.27s, total: 789.33s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 789.35s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 789.35s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 789.35s)
    AudioProcessor | [ 90%] Validating artifact (0.03s, total: 789.38s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 789.39s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 789.39s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 789.39s)
    AudioProcessor | [ 90%] Validating artifact (0.09s, total: 789.48s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 789.49s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 789.50s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 789.50s)
    AudioProcessor | [ 90%] Validating artifact (0.18s, total: 789.68s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 789.80s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 789.80s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 789.80s)
    AudioProcessor | [ 90%] Validating artifact (0.04s, total: 789.85s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 789.86s)
    AudioProcessor | [ 98%] Updating manifest (0.01s, total: 789.87s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 789.87s)
    AudioProcessor | [ 90%] Validating artifact (0.04s, total: 789.91s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 789.92s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 789.92s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 789.92s)
    AudioProcessor | [ 90%] Validating artifact (0.11s, total: 790.03s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 790.12s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 790.12s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 790.12s)
    AudioProcessor | [ 90%] Validating artifact (0.04s, total: 790.17s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 790.17s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 790.18s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 790.18s)
    AudioProcessor | [ 90%] Validating artifact (0.04s, total: 790.21s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 790.22s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 790.22s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 790.22s)
    AudioProcessor | [ 90%] Validating artifact (0.03s, total: 790.25s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 790.26s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 790.27s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 790.27s)
    AudioProcessor | [ 90%] Validating artifact (0.04s, total: 790.30s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 790.32s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 790.32s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 790.32s)
    AudioProcessor | [ 90%] Validating artifact (0.03s, total: 790.35s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 790.36s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 790.36s)
    ✗ AudioProcessor | ERROR: Status is 'error' for pitch_extractor, setting overall_ok=False
    ✗ AudioProcessor | ERROR: pitch_extractor error details: pitch | Ошибка извлечения pitch (error_code=pitch_all_methods_failed): pitch | all segments produced empty pitch (error_code=pitch_all_methods_failed)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 790.36s)
    AudioProcessor | [ 90%] Validating artifact (0.03s, total: 790.39s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 790.40s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 790.41s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 790.41s)
    AudioProcessor | [ 90%] Validating artifact (0.56s, total: 790.96s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 791.72s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 791.72s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 791.72s)
    AudioProcessor | [ 90%] Validating artifact (0.04s, total: 791.76s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 791.78s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 791.78s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 791.78s)
    AudioProcessor | [ 90%] Validating artifact (0.02s, total: 791.80s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 791.82s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 791.82s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 791.82s)
    AudioProcessor | [ 90%] Validating artifact (0.02s, total: 791.84s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 791.85s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 791.86s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 791.86s)
    AudioProcessor | [ 90%] Validating artifact (0.03s, total: 791.89s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 791.89s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 791.90s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 791.90s)
    AudioProcessor | [ 90%] Validating artifact (0.02s, total: 791.92s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 791.92s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 791.93s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 791.93s)
    AudioProcessor | [ 90%] Validating artifact (0.02s, total: 791.95s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 791.96s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 791.96s)
    AudioProcessor | [ 85%] Validating artifact (0.00s, total: 791.96s)
    AudioProcessor | [ 90%] Validating artifact (0.04s, total: 792.00s)
    AudioProcessor | [ 95%] Updating manifest (0.00s, total: 792.01s)
    AudioProcessor | [ 98%] Updating manifest (0.00s, total: 792.01s)
    AudioProcessor | [ 85%] Saving NPZ artifacts (2.95s, total: 792.01s)
    AudioProcessor | [100%] Complete (792.01s, total: 792.01s)
    AudioProcessor | [100%] Complete
    ✗ AudioProcessor | ERROR: overall_ok=False, returning exit code 2
    ✗ AudioProcessor | Extractors with errors: pitch_extractor (key=pitch)
  [✗ ERROR] AudioProcessor завершен
        exit code: 2
Traceback (most recent call last):
  File "/media/ilya/Новый том1/TrendFlowML/DataProcessor/main.py", line 863, in <module>
    raise RuntimeError(f"AudioProcessor failed for required=true (exit={r.returncode})")
RuntimeError: AudioProcessor failed for required=true (exit=2)
ilya@ilya-B450M-DS3H:/media/ilya/Новый том1/TrendFlowML/DataProcessor$
---

## Навигация

[README](README.md) · [Module README](../README.md) · [AudioProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
