## TrendFlow — Model contracts (v1.0) — FINAL

Этот документ — **source-of-truth** по нашим обучаемым моделям: Encoder / Baseline / v1 / v2.

---

### **1) Что именно “наши обучаемые модели”**

- **Encoder (AudioEncoder + VisualEncoder)**:
  - v0: deterministic (необучаемый) encoder
  - v1: trainable encoder, обучаемый end-to-end вместе с v1 transformer
- **Baseline predictor**:
  - boosting/ensemble, multi-horizon/multi-target
- **v1 predictor**:
  - multimodal transformer system (включает trainable encoder)
- **v2 predictor**:
  - v1 prediction + ContextAdjustmentModel + ContextBuilder (контекст как артефакт)

---

### **2) Prediction time и входные данные**

- **Prediction time**: предсказываем “в любой момент времени”.
- **Вход модели**: `snapshot_0` (состояние на момент сбора/анализа).
- **Таргеты**: future snapshots 7d/14d/21d как deltas относительно snapshot_0.
- **Age buckets**: 8 buckets по возрасту видео (используются как фичи и для анализа качества).

---

### **3) Targets и нормализация**

- **Таргеты**: только `views` и `likes`.
- **Горизонты**:
  - 7d: optional (masked)
  - 14d: required
  - 21d: required
- **Функция таргета**:
  - \(y = \log(1 + (x_h - x_0))\) для `views` и `likes`.

---

### **4) Split / metrics / golden**

- **Split**: hybrid time-split по `publishedAt` + channel-group split по `channel_id`.
- **North star metric**: Spearman на \(\log1p(\Delta)\).
- **Secondary metrics**: MAE на \(\log1p(\Delta)\) + Spearman по age buckets.
- **Golden**:
  - holdout: 2000 видео (фиксированные снапшоты 0/7/14/21)
  - regression mini: 200 видео

---

### **5) Baseline predictor (boosting)**

- **Компоненты baseline**:
  - Visual modules (7): `cut_detection`, `optical_flow`, `scene_classification`, `shot_quality`, `story_structure`, `uniqueness`, `video_pacing`
  - Audio extractors (3): `clap_extractor`, `loudness_extractor`, `tempo_extractor`
  - + `snapshot_0` metadata (7 полей + comments list ≤100)
  - + required `core_*` providers (см. ниже)
- **Required core providers for baseline**:
  - `core_brand_semantics`
  - `core_car_semantics`
  - `core_clip`
  - `core_face_identity`
  - `core_face_landmarks`
  - `core_optical_flow`
  - `core_place_semantics`
  - `core_depth_midas`
  - `core_object_detections`
- **Outputs baseline**:
  - 2 модели: views + likes
  - каждая multi-output на горизонты 7/14/21 (7d masked)
- **Freeze policy**:
  - после начала baseline dataset collection нельзя менять алгоритмы/выходы компонент, которые дали baseline фичи;
  - улучшения — только через новые компоненты/ветки артефактов + bump `feature_schema_version`.
- **Schema staging**:
  - `feature_schema_version=v0` допускается как “движущийся”
  - baseline финально обучаем, когда `feature_schema_version=v1` объявлен frozen
- **Resources envelope**: GPU 16GB (допускаем 24/32), RAM 16GB, CPU 4.

---

### **6) Encoder contract (v1.0)**

**Encoder output tensors (per modality)**:
- `global_embedding (D,)`
- `summary_tokens (K, D)`
- `summary_times_s (K,)` = centers of uniform time bins по [0..duration]
- `summary_mask (K,)`

**Time-axis**:
- VisualEncoder: `frame_indices/times_s` aligned на `union_timestamps_sec`
- AudioEncoder: `segment_centers_sec/times_sec`
- Encoder не делает общий time-join между модальностями (fusion учитывает time embeddings).

**Complexity constraint**:
- encoder ≤ O(N) по длине исходной последовательности.

**Budgets (default preset = quality)**:
- `D = 768`
- `K_visual` и `K_audio` выбираются адаптивно по `duration_sec`:
  - duration_sec < 90 → K=64
  - 90 ≤ duration_sec < 600 → K=96
  - duration_sec ≥ 600 → K=128
- Фактические `K_visual/K_audio` и `duration_sec` фиксируются в meta encoder output.

**Encoder v0 (deterministic)**:
- uniform time-binning → per-bin stats (mean/max/quantiles) → linear projection → `summary_tokens`.

**Encoder v1 (trainable)**:
- обучаем end-to-end вместе с v1 transformer.

---

### **7) v1 (transformers) architecture & uncertainty**

**Fusion**:
- cross-attention fusion.

**Time encoding**:
- `time_pos_emb = MLP(t_center / duration_sec)` добавляется к каждому token.

**Text/comments**:
- raw текст не храним.
- embeddings per-comment → агрегируем в несколько text tokens (Kc=4..8) → участвуют в fusion transformer.

**Outputs**:
- 6 значений: views_7d/14d/21d + likes_7d/14d/21d, masked loss на 7d.

**Uncertainty**:
- quantile heads (минимум p10/p50/p90 для каждого из 6 выходов).

**Loss balancing**:
- uncertainty weighting (обучаемые веса горизонтов) + safety cap [0.2..5.0].

**Compute budget**:
- 30–50M params
- inference latency после готовых encoder токенов: 2–5s.

---

### **8) v2 (context) contract**

- v2 = v1 prediction + `ContextAdjustmentModel`.
- `context_features`:
  - формат: набор именованных фичей (таблично) + `context_schema_version`
  - сохраняется как артефакт run (для воспроизводимости)
- TTL контекста: 48h
- деградация:
  - если контекст недоступен/просрочен → fallback на v1 с `prediction_status="degraded"`.
---

## Навигация

[Models](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
