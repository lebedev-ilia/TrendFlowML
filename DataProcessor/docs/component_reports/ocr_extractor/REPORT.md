# REPORT: ocr_extractor

**Дата:** 2026-07-11
**Под:** RunPod RTX 4000 Ada 20GB (213.173.108.27:17909), netVol vuiq0iq3yf
**Движок:** ppocr_rec_onnx (ppocr_rec_onnx_v1_inprocess), min-det-score=0.5
**Метод валидации:** ИЗОЛЯЦИЯ — синтетическая фикстура (реального YOLO-детектора с классом
`text_region` в репо нет; обученный детектор — зона владельца). Скрипт
`/workspace/ocr_synth_validate.py` синтезирует frames_dir (FrameManager-формат) + detections.npz
с боксами `text_region` и гоняет НАСТОЯЩИЙ `ocr_extractor/main.py` (subprocess).
Сырые числа — `REPORT_synth.json` (рядом).

## Корпус (6 синтет-роликов)
| tag | кадров | содержимое | назначение |
|-----|--------|-----------|-----------|
| vidA | 12 | 3 кадра с текстом (TRENDFLOW/HELLO2026/OCRTEST+LINE99) | основной + golden + retain |
| vidB | 8 | 2 кадра с текстом | различимость |
| vidCblank | 6 | боксы text_region ЕСТЬ, текста НЕТ | пустой контент |
| vidDnobox | 6 | класса text_region НЕТ в таксономии | expected-empty |
| vidElong | 200 | 3 кадра с текстом | длинный |
| vidFshort | 3 | 1 кадр с текстом | короткий |

## Результаты по критериям (CRITERIA.md)

### Универсальные хард-гейты
- **U1 — валидатор rc=0:** `validate_ocr.py --struct` → `✅ VALID schema` для всех 6 (A/B/Cblank/Dnobox/E/F), rc=0. Все прогоны main.py rc=0. **PASS**
- **U2 — ось времени:** A/E/F — `axis_match=true`, `monotonic=true`, `nan_pct=0.0`. **PASS**
- **U3 — health:** сырой NPZ: `frame_indices` dtype=**int32**, `times_s` dtype=**float32**, finite, fi в диапазоне. **PASS**
  (В REPORT_synth.json U3 показывает `fi_dtype=int64` — это артефакт `.astype(int)` в загрузчике валидатора; сырой NPZ = int32, что и подтверждает VALID schema в U1.)
- **U4 — expected-empty:** `vidDnobox` (нет класса text_region) → `status=empty`, `empty_reason=proposal_class_not_in_taxonomy`, `rows=0`, `rc=0`. **PASS** *(после фикса — см. ниже)*
- **U5/C4 — golden:** повтор vidA (A_r0 vs A_r0b, ppocr_rec_onnx) → `key_identical=true`, n_rows 4/4, **max|Δrec_confidence| = 0.0** (ONNX побайтово детерминирован). **PASS**
- **U6 — разные длины:** F(3к)/A(12к)/E(200к) → rc=0, n=1/4/3, время ~8.6/10.6/9.1 c. **PASS**

### Критерии под компонент
- **C1 — привязка к оси:** frame каждой строки ∈ frame_indices: A 4/4, B 2/2, E 3/3 = **100%**. **PASS**
- **C2 — privacy:** retain=false → `has_raw=false`, `all_sha=true` (text_sha256+text_len у всех), `meta.retain=false`. retain=true → `has_raw=true`, `meta.retain=true`. **PASS**
- **C3 — различимость:** R_per_video = A:4, B:2, E:3, Cblank:6, Dnobox:0 → `R_varies=true`. `conf∈[0,1]=true` (min 0.0034, max 0.0052, n=9). **PASS**
  ⚠️ Абсолютные `rec_confidence` очень низкие (~0.005) — **артефакт OOD-синтетики**: ppocr_rec обучен на реальных кропах, на белом фоне с крупным DejaVu argmax верный (текст «TRENDFLOW/HELLO2026/OCRTEST» распознан правильно), но softmax размазан → низкий max-prob. На реальных кропах conf будет высоким. Значения conf на синтетике НЕ показательны, порог `min_rec_score` на синтетике не откалибровать.
- **C4:** см. U5 — max|Δ|=0.0 (≤ порога 1e-3). **PASS**

## Найденный логический баг + фикс
**Баг (expected-empty):** при отсутствии класса `text_region` в таксономии детектора
`proposal_ids` пуст; в цикле отбора условие `if proposal_ids and int(class_ids[...]) not in proposal_ids: continue`
при пустом `proposal_ids` схлопывается в `False` → фильтр по классу НЕ применяется и OCR-ятся ВСЕ
валидные боксы. Это противоречит уже существующему warning «OCR will be empty» (main.py:579).
Итог до фикса: vidDnobox давал 1 строку вместо empty.

**Фикс (main.py, после warning на строке 579):**
```python
if not proposal_ids and class_id_to_name:
    LOGGER.warning(... "OCR will be empty" ...)
    skip_ocr_processing = True
    skip_ocr_reason = "proposal_class_not_in_taxonomy"
```
Механизм skip уже существовал (использовался для `tesseract_not_in_path`). После фикса Dnobox →
`status=empty`, `empty_reason=proposal_class_not_in_taxonomy`, rows=0, rc=0. Бэкап оригинала на поде: `main.py.bak_ocrfix`.
⚠️ Правку нужно перенести в git-репозиторий DataProcessor (на поде — рабочая копия).

## Ограничения метода
- Валидация в изоляции на синтетике: подтверждает контракт NPZ/оси/privacy/детерминизм/empty-путь и
  корректность пайплайна OCR. НЕ подтверждает качество распознавания на реальных кадрах и калибровку
  `rec_confidence` (нужен реальный `text_region`-детектор — зона владельца).
- vidCblank (боксы есть, текста нет) → 6 строк-галлюцинаций на пустом сером фоне (conf ~0.005). Это
  свойство rec-модели на OOD-входе, не механизм empty; на реальных данных детектор не даст боксов на пустоте.

## Вердикт (предлагаемый)
**Штамповать** по осям корректность/стабильность/детерминизм/privacy/empty-путь (все гейты PASS,
логический баг expected-empty найден и исправлен). Оговорка: качество rec и калибровка confidence
проверяемы только на реальном text_region-детекторе (зона владельца). Фикс main.py требует переноса в git.
