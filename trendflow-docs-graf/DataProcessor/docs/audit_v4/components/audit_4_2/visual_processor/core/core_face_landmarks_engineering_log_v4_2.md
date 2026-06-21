# Audit v4.2 — engineering log: `core_face_landmarks`

**Дата:** 2026-04-13  
**Компонент:** `core_face_landmarks` (VisualProcessor core)  
**Цель:** закрыть Audit v4 **L2 (A+B)** и добавить наблюдаемость ресурсов + IO hygiene.

## Изменения кода (после L1)

### 1) Env-gated resource profiling (RSS + CUDA)

Добавлено best-effort поле `meta.resource_profile_before`, которое записывается **только** при включении:

- `VP_RESOURCE_PROFILE=1|true|yes|y|on`

Иначе поле отсутствует.

Содержимое (best-effort):

- `rss_bytes`, `rss_mib` (через `psutil`)
- `cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes` (через `torch.cuda`, если доступно)

Файл:

- `DataProcessor/VisualProcessor/core/model_process/core_face_landmarks/main.py`

### 2) NPZ IO hygiene

`_load_npz()` (используется для чтения dependency `core_object_detections/detections.npz`) теперь закрывает `np.load(...)` handle через `try/finally`.

## L2 статистика (A+B, 5 прогонов)

JSON:

- `storage/audit_v4/core_face_landmarks_l2/core_face_landmarks_audit_v4_stats.json`

Ключевые итоги:

- `N_total=543`, `FACES_set=[1]`, `HANDS_set=[2]`
- NaN‑политика слотов **строго соблюдена** (violations=0):
  - `absent ⇒ all‑NaN` для `face_landmarks`/`hands_landmarks`
  - `present ⇒ no‑NaN` для тех же слотов
- `face_mesh_ran ∧ ¬face_present` встречается (в L2 суммарно 100 кадров): FaceMesh запускался, но лицо не обнаружено — ожидаемо при гейтинге по person‑mask/детектору.

## Что осталось (DoD)

- Набор **C**: кейсы `no_person_detections`, устойчивость на видео без людей; отдельный сценарий **FACES>1**.
- **§4.8 golden**: TODO (зафиксировать сигнатуру A: N/FACES/HANDS + ranges по `face_present`/`hands_present` и NaN‑инварианты).
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
