## Schema: Semantic Heads NPZ (v1)

Цель: зафиксировать **единый контракт** для semantic heads (brands/cars/places/face identity), чтобы:
- encoder мог читать артефакты без спец‑кода под каждую голову,
- scheduler мог использовать `model_signature` для cross-video batching/caching,
- соблюдался **fail-fast / no-fallback**.

Документ описывает **v1**; конкретные head’ы могут иметь дополнительные поля, но **не должны нарушать общие правила**.

---

### 0) Общие правила (обязательные)

- **Time-axis source-of-truth**: `frames_dir/metadata.json.union_timestamps_sec`.
- Любой head, который работает по sampled кадрам, обязан писать:
  - `frame_indices (N,) int32` в union-domain
  - `times_s (N,) float32` где `times_s[i] == union_timestamps_sec[frame_indices[i]]`
- **No gating by thresholds**: output top‑K **никогда не обрезается** порогами.
  - Threshold используется только для флагов `*_is_confident_top1`.
- **Fail-fast**:
  - если компонент включён и отсутствуют required базы/галереи/модели/артефакты upstream → **RuntimeError**.
  - нельзя писать “ok empty” из-за отсутствия базы/галереи/модели.
- **NaN-policy** (важно для encoder):
  - `*_topk_scores` заполняются `NaN` там, где прогноз не вычислен/не применим.
  - `*_topk_ids` заполняются `-1` там, где id отсутствует/не применим.
  - `*_present_mask` определяет, вычислялась ли семантика для сущности (track/scene/slot).

---

### 1) Общие ключи NPZ (v1)

#### 1.1 Time-axis

- `frame_indices (N,) int32`
- `times_s (N,) float32`

#### 1.2 Label space

Label space задаётся offline базой и должен быть **стабильным** в пределах `db_digest`.

- `semantic_label_names (A,) str`: строки вида `"id:name"` (id = int)
- `threshold_per_label_arr (A,) float32`: aligned с `semantic_label_names` (NaN если нет)

Примечание:
- Некоторые head’ы имеют несколько label spaces (например cars: make/model/segment/…); см. §2.

#### 1.3 Track-level output (retrieval over tracks)

Типовой интерфейс (если head работает по track’ам):
- `track_ids (T,) int32`
- `track_present_mask (T,) bool`
- `track_topk_ids (T, K) int32`
- `track_topk_scores (T, K) float32`
- `track_is_confident_top1 (T,) bool`

#### 1.4 Frame-level output (retrieval per frame / per detection / per face slot)

Если head возвращает per-frame результаты, он пишет соответствующие массивы:
- `frame_topk_ids (N, K) int32`
- `frame_topk_scores (N, K) float32`
- `frame_is_confident_top1 (N,) bool`

Если head возвращает per-detection/per-face-slot результаты — см. §2.

#### 1.5 Meta (обязательный dict, object scalar)

`meta` должен содержать минимум:
- `producer` (str): canonical component name
- `producer_version` (str)
- `schema_version` (str): `<component>_npz_v1`
- `created_at` (ISO str)
- `status` (str): `ok|empty` (empty допускается только если нет валидных proposals/slots при корректных входах)
- `empty_reason` (str|null)
- `models_used` (list) и `model_signature` (dict) — см. `Models/docs/contracts/MODEL_SYSTEM_RULES.md`

DB provenance (если есть база):
- `db_name`, `db_version`, `db_digest`, `db_path`

Upstream chaining (best-effort, но в v1 **желательно**):
- `<upstream>_model_signature`: например `core_clip_model_signature`

---

### 2) Специфика head’ов (v1)

#### 2.1 `core_brand_semantics`

Upstream:
- proposals из `core_object_detections/detections.npz`
- вычисляет CLIP embeddings через Triton (clip_image + clip_text)

Output:
- uses общие поля §1.1–§1.5
- дополнительно per-detection:
  - `det_topk_ids (N, MAX, K) int32`
  - `det_topk_scores (N, MAX, K) float32`
  - `det_is_confident_top1 (N, MAX) bool`

#### 2.2 `core_car_semantics`

Upstream:
- proposals из `core_object_detections/detections.npz`
- CLIP image embedding через Triton (clip_image)

Label spaces (несколько):
- `make_label_names (M,) str`
- `model_label_names (L,) str`
- `segment_label_names (S,) str`
- `body_type_label_names (B,) str`
- `price_bucket_names (P,) str`

Per-axis topK (K=3 в v1):
- `track_make_topk_ids (T,3)`, `track_make_topk_scores (T,3)`
- `track_model_topk_ids (T,3)`, `track_model_topk_scores (T,3)`
- `track_segment_topk_ids (T,3)`, `track_segment_topk_scores (T,3)`
- `track_body_type_topk_ids (T,3)`, `track_body_type_topk_scores (T,3)`
- `track_price_bucket_topk_ids (T,3)`, `track_price_bucket_topk_scores (T,3)`

Примечание:
- В cars v1 threshold flags не обязательны (в отличие от brand/place/face), но **NaN/-1** правила всё равно применяются.

#### 2.3 `core_place_semantics`

Upstream:
- frame embeddings из `core_clip/embeddings.npz` (must cover all required frame_indices)

Output:
- uses общие поля §1.1–§1.5
- `track_ids (1,)` и `track_*` описывают scene-level aggregate (v1: max over time per label)
- per-frame:
  - `frame_topk_ids (N,K)`, `frame_topk_scores (N,K)`
  - `frame_is_confident_top1 (N,)`

#### 2.4 `core_face_identity`

Upstream:
- face landmarks из `core_face_landmarks/landmarks.npz`
- face embedding model через Triton

Output:
- uses общие поля §1.1–§1.5
- face slot axis = `FACES` (кол-во слотов landmarks модели)
- per-frame/per-slot:
  - `frame_face_topk_ids (N, FACES, K)`
  - `frame_face_topk_scores (N, FACES, K)`
  - `frame_face_is_confident_top1 (N, FACES)`
- track axis = `FACES` (one track per slot):
  - `track_ids (FACES,)`, `track_present_mask (FACES,)`, `track_* (FACES, K)`


