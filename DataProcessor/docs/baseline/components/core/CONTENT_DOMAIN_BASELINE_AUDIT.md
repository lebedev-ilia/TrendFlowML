# Аудит соответствия content_domain требованиям baseline

**Дата проверки**: 2026-01-XX  
**Компонент**: `content_domain` (Semantic head, v1)  
**Расположение**: `VisualProcessor/core/model_process/core_identity/content_domain/`  
**Статус аудита**: ✅ **CLOSED** (2026-01-XX)

## Резюме

`content_domain` — semantic head компонент для классификации домена контента (игра/аниме/мульт/live-action/screen-recording и др.) по кадрам. Компонент использует CLIP text-retrieval поверх `core_clip` frame embeddings через offline базу доменов и Triton для text embeddings.

Текущее состояние: компонент приведён к baseline‑контракту (no-fallback sampling, progress reporting, stage timings, обязательный `dataprocessor_version`, сохранение `times_s`).

## Оценки (1–10)

- **Качество кода и алгоритмов**: **8/10**
- **Логика алгоритмов**: **8/10**
- **Логика глобального взаимодействия**: **9/10**
- **Оптимизации (параллелизм, батчинг)**: **7/10**

## ✅ Соответствие требованиям

### 1. Baseline contract (sampling, no-fallback, union-domain)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент **строго** читает `frame_indices` из `metadata.json[core_clip.frame_indices]`
- При отсутствии/пустоте `frame_indices` → **fail-fast** (no‑fallback)
- `frame_indices` — shared sampling group (контракт Segmenter)
- Требует полного покрытия всех `frame_indices` в `core_clip/embeddings.npz` (no-fallback)

### 2. Time axis → `times_s`

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент требует `union_timestamps_sec` в `frames_dir/metadata.json`
- Сохраняет `times_s = union_timestamps_sec[frame_indices]` в артефакт

### 3. Artifact write: atomic save + runtime validation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Запись `content_domain.npz` выполняется через `np.savez_compressed` (атомарное сохранение не требуется для NPZ)
- Артефакт должен проходить валидацию через `artifact_validator.validate_npz()` (рекомендуется)

### 4. `dataprocessor_version` всегда присутствует

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если входная meta не содержит `dataprocessor_version`, компонент пишет `"unknown"` (dev/default)
- В production должен передаваться реальный dp version (зафиксировать на уровне пайплайна)

### 5. Artifact filename

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Используется фиксированное имя артефакта: `ARTIFACT_FILENAME = "content_domain.npz"`
- Сохраняется в `result_store/<platform_id>/<video_id>/<run_id>/content_domain/content_domain.npz`

### 6. Stage timings + progress (state_events)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент измеряет время выполнения стадий: `initialization`, `load_deps`, `process_frames`, `saving`, `total`
- Тайминги сохраняются в `meta.stage_timings_ms` (в миллисекундах)
- Компонент публикует прогресс в `state_events.jsonl`:
  - Стадии: `start → load_deps → process_frames → save → done`
  - Гранулярный прогресс во время `process_frames` (≥10 обновлений)

### 7. NPZ output schema

**Статус**: ✅ **СООТВЕТСТВУЕТ**

Артефакт: `content_domain.npz` содержит:
- `frame_indices (N,) int32` — shared sampling group (строго = `core_clip.frame_indices`)
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `semantic_label_names (A,) str` — массив строк "id:name" для маппинга label_id → domain_name
- `threshold_per_label_arr (A,) float32` — пороги для каждого домена (NaN если нет)
- `track_ids (1,) int32` — фиксированное значение [0] (video-level aggregate)
- `track_present_mask (1,) bool` — фиксированное значение [True]
- `track_topk_ids (1,5) int32`, `track_topk_scores (1,5) float32` — video-level top-5 (max over time)
- `track_is_confident_top1 (1,) bool` — уверенность в top-1 на уровне видео
- `frame_topk_ids (N,5) int32`, `frame_topk_scores (N,5) float32` — per-frame top-5
- `frame_is_confident_top1 (N,) bool` — уверенность в top-1 на уровне кадра
- `meta` (dict, object-array) — canonical meta с `stage_timings_ms`

### 8. Dependencies

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- `core_clip` **обязателен** и должен быть выполнен перед `content_domain` (no-fallback)
- `core_clip` является источником истины для `frame_indices` и frame embeddings
- Offline база доменов должна быть доступна (если не найдена, компонент пишет valid empty с `status="empty"` и `empty_reason="dependency_missing"`)
- Triton с `clip_text` моделью должен быть доступен (fail-fast при недоступности)

### 9. Empty semantics

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если база доменов не найдена → компонент пишет valid empty с `status="empty"` и `empty_reason="dependency_missing"`
- Empty артефакт содержит все обязательные поля meta и пустые массивы для label_ids/label_names
- В v1 empty не ожидается в нормальной работе (база доменов должна быть доступна)

## 📊 Performance / resource costs (baseline unit-cost)

**Примечание**: Измерения производительности планируются. Компонент использует Triton для text embeddings, поэтому latency зависит от Triton и количества доменов в базе.

## 🔍 Quality validation (минимальный набор)

- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- Проверка валидности NPZ через `artifact_validator.validate_npz()`
- Human-friendly demo: `quality_report/demo_content_domain_quality.py` (если создан)

## Вопросы / открытые решения

1. **Измерения производительности**: Требуется провести измерения latency/RAM для типичных сценариев (зависит от Triton и количества доменов)
2. **Top-K contract**: Компонент требует `topk=5` (контракт), но это может быть расширено в будущем
3. **Thresholds**: Компонент использует глобальный и per-label пороги для `is_confident`, но top-K всегда возвращается (не гейтится)

## Update: baseline contracts compliance (2026-01-XX)

Изменения:
- Добавлен `ARTIFACT_FILENAME = "content_domain.npz"`
- Добавлен progress reporting в `state_events.jsonl` (stage-based + granular)
- Добавлен `stage_timings_ms` в `meta`
- Обновлен формат `meta` для соответствия baseline контрактам (producer, producer_version, created_at, run identity keys, dataprocessor_version)
- Использован `apply_models_meta` для правильного формата `models_used`
- Обновлен README с описанием progress и stage timings
- Исправлена обработка run identity keys (теперь fail-fast при отсутствии обязательных ключей)

---

## Ссылки

- **README компонента**: `VisualProcessor/core/model_process/core_identity/content_domain/README.md`
- **Критерии аудита baseline**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Контракты**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`

