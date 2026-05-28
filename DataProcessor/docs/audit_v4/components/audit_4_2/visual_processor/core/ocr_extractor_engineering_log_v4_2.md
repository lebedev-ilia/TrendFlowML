# Audit v4.2 — engineering log: `ocr_extractor`

**Дата:** 2026-04-13  
**Компонент:** `ocr_extractor` (VisualProcessor core)  
**Цель:** довести отчёт Audit v4 по компоненту до **L2 (A+B)** и добавить наблюдаемость ресурсов/IO hygiene без изменения контракта.

## Изменения кода (после L1)

### 1) Env-gated resource profiling (RSS + CUDA)

Добавлено best-effort поле `meta.resource_profile_before`, которое записывается **только** при включении переменной окружения:

- `VP_RESOURCE_PROFILE=1|true|yes|y|on` → в meta появится `resource_profile_before`
- иначе поле отсутствует

Содержимое (best-effort):

- `rss_bytes`, `rss_mib` (через `psutil`)
- `cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes` (через `torch.cuda`, если доступно)

Файл:

- `DataProcessor/VisualProcessor/core/model_process/ocr_extractor/main.py`

### 2) NPZ IO hygiene

Функция `_load_npz()` теперь явно закрывает `np.load(...)` в `finally` (best-effort), чтобы не держать открытый file handle при длительных прогонов/многократных загрузках.

Файл:

- `DataProcessor/VisualProcessor/core/model_process/ocr_extractor/main.py`

## L2 статистика (A+B, 5 прогонов)

JSON:

- `storage/audit_v4/ocr_extractor_l2/ocr_extractor_audit_v4_stats.json`

Ключевые итоги (по агрегатам JSON):

- `N_total=543`, `R_total=776`, `max_rows_per_frame_max=5`
- `engine_set=["ppocr_rec_onnx"]`
- Privacy: `retain_raw_ocr_text_set=[false]`, `raw_text_keys_present_any=false`
- Привязка строк к оси кадров: `frames_subset_ok_all=true`

## Наблюдения/риски для downstream

- При `retain_raw_ocr_text=false` строки несут только хэш/длину (`text_sha256`, `text_len`), bbox/conf и служебные поля; сырого текста нет.

## Что осталось (DoD)

- Набор **C** (edge) + **§4.8 golden**: TODO (в т.ч. сценарий `retain_raw_ocr_text=true` как отдельный dev/debug режим).

