# ocr_extractor — описание фич (Audit)

**Компонент:** `ocr_extractor` (VisualProcessor core)  
**schema_version NPZ:** `ocr_extractor_npz_v2`  
**Путь артефакта:** `rs_path/ocr_extractor/ocr.npz`

## Назначение

OCR по кропам `text_region` из **core_object_detections** (tesseract CLI или `ppocr_rec_onnx` через ModelManager). Строки детекций — union-domain: `frame`, `time_s`, `bbox`, `det_confidence`, `engine`, при необходимости `text_raw` / `text_norm` или `text_sha256` / `text_len` (без сырого текста).

## Ключи NPZ

| Ключ | Описание |
|------|----------|
| `frame_indices` | `(N,) int32` — индексы кадров (как в detections) |
| `times_s` | `(N,) float32` — время, с |
| `ocr_raw` | `(R,) object` — список словарей-строк OCR; **R** не обязан совпадать с **N** |
| `meta` | dict: producer, schema, status, `engine`, tesseract/ppocr поля, `stage_timings_ms`, … |
| `meta_json` | JSON-строка с тем же содержанием (для совместимости) |

## Тайминги (`meta.stage_timings_ms` → `meta_timing_*` в плоском CSV)

Обычно: `initialization`, `load_deps`, `process_frames`, `saving`, `total` (в **мс** в meta).

## Пусто

`status=empty`, `empty_reason`: например `no_text_available` или `tesseract_not_in_path` (нет tesseract / пропуск обработки).

## Схема

`docs/SCHEMA.md`, `schemas/ocr_extractor_npz_v2.json`, подробно — `README.md`.
