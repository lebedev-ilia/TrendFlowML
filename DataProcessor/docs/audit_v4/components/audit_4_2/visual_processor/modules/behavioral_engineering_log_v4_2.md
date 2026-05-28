# Audit 4.2 — engineering log: `behavioral` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/behavioral`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/behavioral_audit_v4.md`](../../visual_processor/modules/behavioral_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/behavioral_npz_v1.json`](../../../../../VisualProcessor/schemas/behavioral_npz_v1.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/behavioral/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/behavioral/docs/SCHEMA.md)

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/behavioral_l2/behavioral_audit_v4_stats.json`
- Итог (A+B): суммарно **N=1250**, `landmarks_present=True` **142** (**~11.36%**).

## Что поменялось / инженерные заметки (4.2)

### 1) Лёгкие оптимизации CPU-path

- `StressAnalyzer` больше не создаёт `HandGestureClassifier()` на каждом кадре — переиспользует один инстанс.

### 2) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1` (RSS через `psutil`).

### 3) Лог таймингов

- Исправлен вывод таймингов: раньше лог пытался читать `stage_timings_ms["total"]`, которого в meta нет; теперь логирует вычисленные `save_ms` и `total_ms` без ожидания ключа в NPZ meta.

## Что осталось сделать (следующий шаг)

1. **Набор C (edge)**: кейсы без лиц/с очень малым числом опорных кадров.
2. **Golden (§4.8)**: сигнатуры по A (доля `landmarks_present`, ключевые агрегаты из `aggregated`).
3. Уточнить/документировать **missing policy** для mouth/pose полей при `landmarks_present=True` (иерархия опор).

