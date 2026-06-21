# Audit v4 — `ocr_extractor` (VisualProcessor core)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A + B**; **5** прогонов).  
**Артефакты:** см. `storage/audit_v4/ocr_extractor_l2/ocr_extractor_audit_v4_stats.json` (полный список путей)  
**Контракт:** [`VisualProcessor/schemas/ocr_extractor_npz_v2.json`](../../../../../VisualProcessor/schemas/ocr_extractor_npz_v2.json) · [`core/model_process/ocr_extractor/docs/SCHEMA.md`](../../../../../VisualProcessor/core/model_process/ocr_extractor/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Роль | ✓ | Ось **N** кадров + таблица **`ocr_raw`** длиной **R** (0…много строк на кадр) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `ocr_extractor_npz_v2` | ✓ | **`frame_indices`**, **`times_s`**, **`ocr_raw`**, **`meta`**, **`meta_json`**; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A+B** (5/5) |
| **N**, **R** | ✓ | На L2: **N_set=[48,59,133,147,156]**, **R_set=[13,45,202,243,273]**, **R_total=776** |
| **`ocr_raw` rows** | ✓ | Каждая строка — **`dict`** с полями вроде **`frame`**, **`time_s`**, **`bbox`**, **`det_confidence`**, **`rec_confidence`**, **`text_len`**, **`text_sha256`**, **`lang`**, **`engine`** |
| Привязка к оси | ✓ | На L2: `frames_subset_ok_all=true` (5/5); `frames_with_ocr_total=399`, `max_rows_per_frame_max=5` (см. stats JSON) |

#### §4.4 — Privacy

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **`retain_raw_ocr_text`** | ✓ | На L2: `retain_raw_ocr_text_set=[false]`, `raw_text_keys_present_any=false` (сырого текста нет) |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают |
| `times_s` | ✓ | **0%** NaN |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| Движок | **`meta.engine`**: **`ppocr_rec_onnx`** на **A** |
| `meta.models_used` | **1** запись (**ONNX** **cuda** в поле **device** у записи) |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **совпадают**, manifest **чистый** (status=ok), ось и `ocr_raw` **согласованы** (`frame` ⊆ `frame_indices`), privacy‑редакция **включена** (сырого текста нет). **~8.8 / 10** на L2.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8; сценарий **`retain_raw_ocr_text=true`**.

---

## 1. Снимок **A+B** (L2)

| Величина | Значение |
|----------|----------|
| N_total | 543 |
| R_total | 776 |
| Engine | `ppocr_rec_onnx` (5/5) |
| `retain_raw_ocr_text` | false (5/5) |
| `frame` ⊆ `frame_indices` | true (5/5) |
| max строк OCR на кадр | 5 |

Статистика/воспроизводимость:

- JSON stats: `storage/audit_v4/ocr_extractor_l2/ocr_extractor_audit_v4_stats.json`
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
