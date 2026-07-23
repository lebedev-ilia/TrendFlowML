# Baseline v0-real (cleaned) - leakage audit + importance

- leakage audit passed: **True** (862 feats, 0 leaked / 0 forbidden / 0 unmapped)
- schema: v0-real cleaned (971->882 feat cols; -89 dead/poison/const-by-design)
- importance = |Spearman(feature, target)| on test (fast proxy; sklearn permutation too slow on py3.14)

## Top 20 by |Spearman| (views_21d, test)

- `scene_classification__frame_entropy__p50`: 0.868
- `core_object_detections__sum_person_area_frac__mean`: 0.865
- `core_depth_midas__foreground_background_separation_proxy__min`: 0.841
- `clap_extractor__clap_magnitude_mean`: 0.828
- `video_pacing__color_change_rate_std`: 0.828
- `core_clip__places365_topk_scores__std`: 0.824
- `cut_detection__edit_style_whip_pan_prob`: 0.819
- `scene_classification__top1_vs_top2_gap_mean__mean`: 0.819
- `clap_extractor__clap_magnitude_std`: 0.811
- `scene_classification__length_frames__max`: 0.811
- `core_face_landmarks__times_s__mean`: 0.797
- `video_pacing__color_change_rate_mean`: 0.789
- `cut_detection__edit_style_luma_wipe_transition_prob`: 0.784
- `scene_classification__scene_change_score__max`: 0.775
- `scene_classification__scene_change_score__mean`: 0.775
- `core_clip__places365_video_topk_indices__p50`: 0.772
- `core_clip__places365_video_topk_scores__p90`: 0.770
- `core_depth_midas__preview_depth_maps_norm__p90`: 0.770
- `core_object_detections__max_person_area_frac__std`: 0.770
- `shot_quality__shot_quality_topk_probs__std`: 0.770