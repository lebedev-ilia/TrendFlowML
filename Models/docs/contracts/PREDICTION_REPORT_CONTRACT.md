## Prediction report contract (v1.0)

Этот документ фиксирует machine-readable формат **`prediction_report.json`** — единый отчёт, который backend отдаёт frontend, чтобы красиво показать пользователю прогон модели, его этапы и результаты.

Принципы:
- report **не содержит raw текст/комменты/OCR** (см. `DataProcessor/docs/contracts/PRIVACY_AND_RETENTION.md`).
- report включает `model_signature`/версии и достаточно информации для аудита/воспроизводимости (см. `MODEL_SYSTEM_RULES.md`).
- значения для пользователя отображаются в **натуральных единицах**; log-scale — только internal/debug.

---

### 1) Top-level fields (required)

- **`schema_version`**: `"prediction_report_v1"`
- **`job_id`**: string (UUID/ULID)
- **`platform_id`**: string (например `"youtube"`)
- **`video_id`**: string
- **`run_ids`**: string[] (1..N) — DataProcessor run_id(ы), которые использовались (обычно 1)
- **`created_at`**: ISO8601 UTC string

- **`timestamps`**:
  - `data_collection_at`: ISO8601 | null
  - `dataprocessor_run_created_at`: ISO8601 | null
  - `inference_started_at`: ISO8601
  - `inference_finished_at`: ISO8601

- **`status`**: `"ok" | "error"`
- **`errors`**: array (может быть пустым при ok)

---

### 2) Stages (required)

**`stages[]`** — список этапов с таймингами и статусом.

Stage object:
- **`name`**: one of:
  - `DataCollection`
  - `DataProcessor`
  - `TextEmbedding`
  - `Encoder`
  - `Fusion`
  - `Heads`
  - `Postprocess`
- **`status`**: `"ok" | "skipped" | "error"`
- **`duration_ms`**: number
- **`artifacts_used`**: string[] (пути/URI; без секретов)
- **`notes`**: string (коротко; без PII)
- **`errors`**: array (если status=error)

---

### 3) Models used (required)

**`models_used[]`**: список моделей, участвовавших в prediction.

Каждый элемент:
- `model_name`
- `model_version` (pinned)
- `weights_digest` (sha256)
- `engine`: `torch|onnx|tensorrt|triton`
- `precision`: `fp32|fp16|bf16`
- `device`: `cpu|cuda:0|...`
- `model_signature` (строка, derived от всего выше; см. `MODEL_SYSTEM_RULES.md`)

---

### 4) Inputs summary (required)

**`inputs_summary`**:
- `frames_used`: int
- `audio_segments_used`: int
- `comments_used`: int
- `modalities_present`:
  - `visual`: bool
  - `audio`: bool
  - `text`: bool
- `warnings`: string[] (например `"7d_masked"`, `"comments_missing"`)

---

### 5) Outputs (required)

Выходы должны быть представлены как 2×3 heads:
- `views`: horizons 7/14/21
- `likes`: horizons 7/14/21

**`outputs`**:
- `units`: `"delta"` (дельта к snapshot_0) и/или `"absolute"` (если вычисляется)
- `scale`: `"natural"`
- `heads`:
  - `views_7d`, `views_14d`, `views_21d`, `likes_7d`, `likes_14d`, `likes_21d`

Head object:
- `masked`: bool (true для 7d при отсутствии таргета/невалидности)
- `p10`: number | null
- `p50`: number | null
- `p90`: number | null
- `delta_p10/p50/p90`: number | null (если показываем delta)
- `absolute_p10/p50/p90`: number | null (если есть `snapshot_0` и хотим показать абсолют)

---

### 6) Explainability (optional, internal/debug)

**Важно**: это не “причинное объяснение”, а диагностический сигнал.

`explainability`:
- `mode`: `"none" | "evidence" | "attribution_lite"`
- `evidence`:
  - `top_modalities`: list (например `["visual","text"]`)
  - `sanity`: например `{"p10<=p50<=p90_rate":0.99}`
- `attribution_lite` (если включено):
  - `ablation`: вклад модальностей через отключение/zeroing (delta на p50)
  - `top_time_bins`: индексы/времена с наибольшим вниманием (если доступно)

---

### 7) Example (truncated)

```json
{
  "schema_version": "prediction_report_v1",
  "job_id": "01J...ULID",
  "platform_id": "youtube",
  "video_id": "abc123",
  "run_ids": ["run_001"],
  "created_at": "2026-01-14T12:00:00Z",
  "timestamps": {
    "data_collection_at": null,
    "dataprocessor_run_created_at": "2026-01-14T11:58:00Z",
    "inference_started_at": "2026-01-14T11:59:30Z",
    "inference_finished_at": "2026-01-14T12:00:00Z"
  },
  "status": "ok",
  "errors": [],
  "stages": [
    {"name": "DataProcessor", "status": "ok", "duration_ms": 82000, "artifacts_used": ["result_store/.../manifest.json"], "notes": "", "errors": []},
    {"name": "Inference", "status": "ok", "duration_ms": 1200, "artifacts_used": [], "notes": "", "errors": []}
  ],
  "models_used": [
    {"model_name":"v1","model_version":"v1_0","weights_digest":"sha256:...","engine":"torch","precision":"bf16","device":"cuda:0","model_signature":"..."}
  ],
  "inputs_summary": {
    "frames_used": 512,
    "audio_segments_used": 48,
    "comments_used": 100,
    "modalities_present": {"visual": true, "audio": true, "text": true},
    "warnings": ["7d_masked"]
  },
  "outputs": {
    "units": "delta",
    "scale": "natural",
    "heads": {
      "views_14d": {"masked": false, "p10": null, "p50": null, "p90": null, "delta_p50": 120000, "absolute_p50": 450000}
    }
  },
  "explainability": {"mode": "evidence", "evidence": {"top_modalities": ["visual", "text"], "sanity": {"p10<=p50<=p90_rate": 0.99}}}
}
```


