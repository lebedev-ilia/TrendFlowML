# Реестр контрактов между компонентами (producer → consumer)

Заполняется по мере валидации (протокол §7). Для каждой связи: какой артефакт/поля
читаются, shape/dtype, обязательность, поведение при отсутствии, валидация на входе.

Легенда обязательности: **hard** (нет входа → error/valid-empty), **soft** (можно без него).

## Segmenter → (все)
| Поле | Тип | Обязат. | Примечание |
|---|---|---|---|
| `frames_dir/metadata.json.union_timestamps_sec` | (U,) float32 | hard | SoT времени (visual) |
| `metadata[<component>].frame_indices` | (N,) int32 (⊆ union) | hard | персональная выборка компонента |
| `frames_dir/audio/segments.json` | schema audio_segments_v1 | hard (audio) | окна аудио |

## core_object_detections → action_recognition
| Поле | Тип | Обязат. | Поведение при отсутствии |
|---|---|---|---|
| person-детекции (`detections.npz`: boxes/scores/class_ids person) | (N,MAX,·) | hard | нет person → `action_recognition` = valid `empty` (`no_person_detections`) |
| `track_ids` (schema v3, appearance-tracker) | (N,MAX) int32 | hard (для клип↔трек) | `-1` = не-трекаемый/невалидный слот |
| общий `frame_indices` (shared sampling group) | (N,) | hard | должен совпадать с сэмплом action_recognition |

Валидация на входе action_recognition: есть person-детекции и `track_ids`; клип строится
**по окну Segmenter** (≥`clip_len`), `clip_track_id` = мода `track_ids` в окне; при нехватке — `empty`, не падение.

✅ **Решение (2026-07-03, владелец):** трекинг возвращается как **свой appearance-embedding
трекер** внутри `core_object_detections` (эмбеддинг бокса + сверка между кадрами) →
`detections.npz` снова несёт **реальный** `track_ids` (schema `_v3`). Дизайн:
[`design/EMBEDDING_TRACKER.md`](design/EMBEDDING_TRACKER.md). Это чинит корень фрагментации
из прогона 2026-07-02. Параллельно Segmenter даёт action_recognition **плотные окна ≥32 кадров**
([`design/ACTION_RECOGNITION_V3.md`](design/ACTION_RECOGNITION_V3.md), раздел B) — иначе SlowFast = 1 клип/трек.

## action_recognition → Models/Encoder (schema v3)
| Поле | Тип | Обязат. | Примечание |
|---|---|---|---|
| `clip_embeddings` | (C,D) float32 L2 | hard | плоский time-ordered seq-токен. **v3.1:** D — размерность **penultimate-фич** backbone (для slowfast_r50 ≈2304), не 256; фактический D в `meta.embedding_dim`, режим в `meta.embedding_mode` (`penultimate`\|`projection_fallback`). Encoder читает D динамически. |
| `clip_times_s` | (C,) float32 | hard | ⊆ `union_timestamps_sec`, монотонно |
| `clip_topk_action_ids/probs` + `class_names` | (C,K)/(400,) | soft (аналитика) | классы Kinetics-400 |
| `clip_track_id` | (C,) int32 | soft | привязка к треку (`-1` если нет) |
| `video_action_hist` + `dominant_action_*` | (400,)/(top,) | soft (аналитика) | агрегаты из stream |

## core_clip → scene_classification (hard, no-fallback)
| Поле | Тип | Обязат. | Примечание |
|---|---|---|---|
| `core_clip/embeddings.npz`: `frame_indices` | (M,) int32 | hard | `scene.frame_indices ⊆ core_clip.frame_indices` (иначе fail-fast) |
| `frame_embeddings` | (M,D) float32 | hard | CLIP-эмбеддинги кадров; **переиспользуются как scene-эмбеддинг** (v3) |
| `places365_text_embeddings` | (365,D) | hard при `label_fusion=clip` | zero-shot по 365 меткам |
| `scene_*_text_embeddings` (aesthetic/luxury/atmosphere) | (P,D) | soft | advanced-семантика |

## cut_detection → scene_classification (hard)
| Поле | Тип | Обязат. | Примечание |
|---|---|---|---|
| `shot_boundaries_frame_indices` | (S,) int32 | hard | границы шотов → сегментация сцен (+ `min_scene_seconds`) |

## scene_classification → Models/Encoder (schema v2 + v3-доп)
| Поле | Тип | Обязат. | Примечание |
|---|---|---|---|
| `frame_topk_ids/probs` | (N,K) | hard | top-K Places365 (мягкая метка сцены по кадрам, seq) |
| `frame_entropy`/`frame_top1_prob`/`frame_top1_top2_gap` | (N,) | hard | скаляры уверенности сцены |
| `frame_scene_embedding` | (N,D) float32 L2 | **v3** | плотный scene-токен (reuse core_clip CLIP) — для Encoder |
| scene-сегменты (`scene_label`, `mean_*`, эстетика/атмосфера/стабильность) | (S,·) | soft | аналитика |

## (5 deps) → shot_quality  [самая тяжёлая цепочка; ВСЕ deps с ОДИНАКОВЫМИ frame_indices]
`_ensure_same_indices` требует, чтобы у всех core-провайдеров были **те же** `frame_indices`, что у
shot_quality (Segmenter aligned sampling group). Иначе — error (no-fallback).

| provider | ключи (обязат.) | примечание |
|---|---|---|
| core_clip | `frame_indices`, `frame_embeddings` | + CLIP-quality промпты/эмбеддинги |
| core_depth_midas | `frame_indices`, `depth_maps (N,H,W)` | Triton-only (MiDaS); прогон — inprocess-обход `midas_depth_inprocess.py` |
| core_object_detections | `frame_indices`, `boxes`, `valid_mask`, `class_ids` | ultralytics inprocess |
| core_face_landmarks | `frame_indices`, `face_landmarks`, `face_present`, `has_any_face`, `empty_reason` | mediapipe; пустота ок → face-ROI фичи = NaN |
| cut_detection | `detections`/`cut_detection_model_facing_*` | границы шотов для shot-агрегаций |

## shot_quality → Models/Encoder (schema v3)
| Поле | Тип | Обязат. | Примечание |
|---|---|---|---|
| `frame_features` | (N,F) float32 | hard | **готовый seq-токен качества** для Encoder (эмбеддинг не нужен — вектор фич и есть представление); face-ROI фичи = NaN by design |
| `feature_names` | (F,) str | hard | имена фич (sharpness/exposure/contrast/depth/clip-quality/face-ROI) |
| shot-агрегации + CLIP-quality | (S,·) | soft | аналитика (по шотам из cut_detection) |

_(далее пополняется по каждому провалидированному компоненту)_
