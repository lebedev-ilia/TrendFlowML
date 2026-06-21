# Audit v4 — `emotion_face` (VisualProcessor)

**Дата:** 2026-04-06 (обновление: 2026-04-13)  
**Уровень отчёта (план §3.1):** **L2 — product stats** (**A + B**, 5 run).  
**Артефакт (набор A, фактический в `storage/result_store`):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/emotion_face/emotion_face.npz`  
**Контракт:** [`VisualProcessor/schemas/emotion_face_npz_v3.json`](../../../../../VisualProcessor/schemas/emotion_face_npz_v3.json) · [`modules/emotion_face/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/emotion_face/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Один NPZ | ✓ | `emotion_face.npz` |
| Зависимости | ✓ | `core_face_landmarks`, EmoNet через ModelManager (см. SCHEMA) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `emotion_face_npz_v3.json` | ✓ | Совпадение множеств; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ На **A** | **`null`** |
| **N** | ✓ | **200** (внутренняя ось после stride/cap; **`meta.face_frame_stride=4`**, **`max_frames=200`**) |
| `emotion_probs` | ✓ | **`(N, 8)`** float32 |
| `dominant_emotion_id` | ✓ | **`int8`**, **`-1`** вне inference |
| `keyframes` | ✓ | **`(0,)`** object на **A** — допустимо (**K=0**) |
| `axis_source` | ✓ | Присутствует (опциональное поле схемы) |

#### §4.1a — Маски vs NaN (критично)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Документ vs факт | ✓ | [`SCHEMA.md`](../../../../../VisualProcessor/modules/emotion_face/docs/SCHEMA.md): при **`processed_mask=false`** — **NaN** в VA/I/confidence/probs, **`dominant_emotion_id=-1`** |
| На **A** | ✓ | **`processed_mask` true: 4** кадра (**2%**); **`face_present` true: 13** (**6.5%**) |
| Промежуточный слой | ✓ | **9** кадров с лицом, но **без** inference (`face_present & ~processed_mask`): **valence — NaN**, **`dominant_emotion_id=-1`** — ожидаемо (субсемплинг среди face-кадров) |
| Без лица | ✓ | **187** кадров: NaN в сигналах, **dominant −1** |
| Вероятности | ✓ | На обработанных строках **NaN 0%**, сумма по строке **≈1** (отклонения <0.05) |

#### §4.2 — Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| VA/I/confidence/probs | ✓ | **0%** Inf |

#### §4.3 — Распределения (**A**, только `processed_mask`)

| Сигнал | min … max (наблюдение) |
|--------|-------------------------|
| `valence` | **≈ −0.02 … 0.60** |
| `arousal` | **≈ −0.20 … 0.40** |

*(N мало — перцентили L2.)*

#### §4.3b — L2 stats (A+B, 5 run)

- JSON: `storage/audit_v4/emotion_face_l2/emotion_face_audit_v4_stats.json`
- Итог по 5 run: **N_total=1000**, `face_present` True **42** (**4.2%**), `processed_mask` True **12** (**1.2%**), `keyframes_total=0`.

#### §4.4 — Analytics object-поля

| Ключ | На **A** |
|------|----------|
| `features` | **3** ключа: `valence_mean`, `arousal_mean`, `intensity_mean` (конечные скаляры) |
| `advanced_features` | **пустой dict** `{}` (gated/выключено) |
| `summary` | `sequence_length=4`, `faces_found_frames=4`, `max_faces_per_frame=2` |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices`, `times_s` | ✓ | Монотонны |

#### §4.7 — Трактовка

| Наблюдение | Вывод |
|------------|--------|
| Высокая доля NaN (**~98%**) при `status=ok` | Норма при **жёстком** gating: нужен **`processed_mask`**, не интерпретировать как сбой |
| **`meta.processed_frames`=200** | Это **длина оси N**, не число кадров с inference (**4**) — не смешивать с `summary.sequence_length` |

#### §4.10 — empty

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Нет лиц / edge | ✗ | **C** |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Локально по видео? | Да |
| `models_used` | **1** элемент (EmoNet in-process), `device=cuda` на **A** |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder | Ряды **(N,)** + маска **`processed_mask`**; при необходимости **`(N,8)` probs** |
| Таблично | **`features`** (краткое агрегирование по малым N на **A**) |

#### §6 — Verdict

**Итог L2 (A+B, 5 run):** схема и файл **совпадают**; поведение **NaN / −1** **согласовано** с `processed_mask` и документацией. На текущем A+B `keyframes` **всегда пустой** (K_total=0) — событийный слой не покрыт, нужен B-набор с более плотной лицевой осью/вариативностью эмоций.

**Оценка:** **~8.5 / 10** до закрытия **C** и **golden (§4.8)**.

#### §8 — DoD

**Не закрыт:** **C**, golden **§4.8** и полный DoD (§8).

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N | 200 |
| `processed_mask` true | 4 (2%) |
| `face_present` true | 13 (6.5%) |
| NaN в `valence` (все кадры) | 98% |
| `dominant_emotion_id` при `~processed` | **−1** |
| `keyframes` K | 0 |
| `emotion_probs` на processed | без NaN, сумма **~1** |
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
