# Критерии приёмки: core_identity/face_identity

**Согласовано:** 2026-07-16  
**Компонент:** face_identity (core_face_identity v0.2, schema v2)  
**Зависимости:** core_face_landmarks → Embedding Service (arcface)

---

## Ограничения

- **Embedding Service недоступен локально** → U5/U6 пропускаются (одобрено)
- GPU-прогон отложен: first pass — offline-валидация 24 NPZ из storage
- База лиц: HF-seed лица (тестовые), не реальные знаменитости

---

## Универсальные гейты (U1–U6)

| Гейт | Критерий | Применимость |
|------|----------|-------------|
| U1 | validate_core_face_identity_npz.py --struct rc=0 на всех 24 NPZ | ✅ проверяется |
| U2 | frame_indices возрастают при status=ok; пусты (shape=0) при status=empty | ✅ проверяется |
| U3 | face_similarities ∈ [0,1], face_ids ∈ [-1, A) при status=ok | ✅ проверяется |
| U4 | status=empty → face_ids shape=(0,5), times_s shape=(0,) — valid NPZ, не exception | ✅ проверяется |
| U5 | Golden детерминизм (повтор → diff=0) | ⚪ SKIP (нет ES локально) |
| U6 | Разные длины без падений | ⚪ SKIP (нет ES / GPU-прогона) |

## Специфические критерии (C1–C3)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | status=ok → top-1 similarity > 0.0 хотя бы на части кадров (не все нули) | ≥1 кадр с sim>0 |
| C2 | face_bbox_xyxy — каждая строка полностью NaN или полностью finite (нет полу-NaN) | 0 плохих строк |
| C3 | validate --ranges rc=0 (sim∈[0,1], ids в label-space, K==top_k, bbox) | rc=0 |

## Исключения

- **U5/U6:** SKIP (нет Embedding Service локально; аналогично другим core_identity — одобрено)
- **Дубли в label-space** (одно имя, два UUID): дедупликация по UUID добавлена в main.py; баг в ES-кеше — зона владельца
- **face_bbox_xyxy = NaN by design** при status=empty: all-NaN строки допустимы (нет лица → нет bbox)
