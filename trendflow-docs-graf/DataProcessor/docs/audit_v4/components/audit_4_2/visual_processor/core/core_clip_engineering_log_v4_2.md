# Audit v4.2 — engineering log: `core_clip`

**Дата:** 2026-04-13  
**Компонент:** `core_clip` (VisualProcessor core)  
**Цель:** довести отчёт Audit v4 по компоненту до **L2 (A+B)** и добавить наблюдаемость ресурсов без изменения контракта.

## Изменения кода (после L1)

### 1) Env-gated resource profiling (RSS + CUDA)

Добавлено best-effort поле `meta.resource_profile_before`, которое записывается **только** при включении переменной окружения:

- `VP_RESOURCE_PROFILE=1|true|yes|y|on` → в meta появится `resource_profile_before`
- иначе поле отсутствует

Содержимое (best-effort):

- `rss_bytes`, `rss_mib` (через `psutil`)
- `cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes` (через `torch.cuda`, если доступно)

Файл:

- `DataProcessor/VisualProcessor/core/model_process/core_clip/main.py`

## L2 статистика (A+B, 5 прогонов)

JSON:

- `storage/audit_v4/core_clip_l2/core_clip_audit_v4_stats.json`

Ключевые итоги (по агрегатам JSON):

- `N_total=543`
- `D_set=[512]`
- `K_places365_set=[5]`
- `consecutive_cosine_prev_nan_total=5` (ожидаемо: первый кадр каждого run)
- `frame_embeddings` L2 нормы по строкам: mean в диапазоне \(\approx 1.00000002 … 1.00000004\)

## Наблюдения/риски для downstream

- `*_scores` и `places365_*_topk_scores` — **не вероятности** (не обязаны суммироваться в 1); трактовать как similarity/logit-подобные значения согласно продюсеру.
- `consecutive_cosine_prev[0]=NaN` — ожидаемый маркер «нет предыдущего кадра», downstream должен корректно обрабатывать.

## Что осталось (DoD)

- Набор **C** (edge) + **§4.8 golden**: TODO (зафиксировать «golden signature» по A и минимальный набор инвариантов).
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
