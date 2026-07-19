# FINAL REPORT — `ocr_extractor`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `ocr_extractor` (VisualProcessor **core provider**, `core/model_process/ocr_extractor/`) |
| Версия кода | `0.2` |
| Схема NPZ | `ocr_extractor_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/ocr_extractor/ocr.npz` |
| Модель | **PP-OCR rec ONNX** (`ppocr_rec_onnx_v1_inprocess`, CTC greedy decode) via ModelManager; tesseract-опция |
| Hard dep | `core_object_detections` (боксы класса `text_region`) |
| Потребители | `franchise_recognition`, `text_scoring`, `story_structure` (sparse text-события) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → ocr_extractor ✅ (2026-07-11) |
| Отчёт валидации | [`REPORT.md`](REPORT.md), [`CRITERIA.md`](CRITERIA.md), [`REPORT_synth.json`](REPORT_synth.json) |
| Код | `DataProcessor/VisualProcessor/core/model_process/ocr_extractor/main.py` (803 строки) |

## 2. Резюме

`ocr_extractor` — **распознаватель экранного текста**: берёт боксы класса `text_region` из
`core_object_detections`, кропает их и прогоняет через PP-OCR recognizer (ONNX, CTC greedy decode), отдавая
**sparse-события текста** (`ocr_raw`: frame, bbox, распознанный текст, `rec_confidence`) с privacy-политикой
(при `retain=false` — только `text_sha256`+`text_len`, без сырого текста). Это **analytics/sparse-компонент**,
не dense-seq для Encoder. Контракт/ось/privacy/детерминизм/empty-путь **валидированы, но только в изоляции на
синтетических фикстурах** — реального `text_region`-детектора в репозитории нет (зона владельца). На реальном
storage-корпусе компонент **пуст на 100% видео** (`status=empty`, `no_text_available`, 0 строк), потому что
детектор в батче работает на COCO-весах, где класса `text_region` нет вовсе. Итог: качество распознавания и
калибровка confidence на реальных кадрах **не подтверждены ничем**.

## 3. Функционал

Core-провайдер (Tier-1), гейтится детекциями:

1. Загружает `detections.npz`, ищет боксы класса `text_region` (`--proposal-class`).
2. Кропает каждый бокс (RGB uint8), препроцессит под PP-OCR (48×320), прогоняет ONNX recognizer.
3. **CTC greedy decode** логитов → строка + `rec_confidence ∈ [0,1]` (max-prob усреднённо).
4. Privacy-обработка: при `retain=false` (прод-дефолт) хранит `text_sha256`+`text_len`, не сырой текст.
5. Пишет `ocr_raw` — sparse-события, привязанные к `frame_indices`/`times_s` (union-ось).

**Зачем продукту:** экранный текст — **важный семантический слой**: заголовки-плашки, субтитры, названия
брендов/франшиз, CTA («подпишись»), цены. Это кормит `franchise_recognition` (распознавание франшиз по тексту),
`text_scoring`, `story_structure`. Для аналитика — «какой текст на экране», для модели (косвенно) — присутствие/
плотность экранного текста как признак формата (мем/обучающее/реклама).

## 4. Вход

- **`core_object_detections`** (hard) — боксы класса `text_region`; **нет класса в таксономии** → skip →
  `status=empty, empty_reason=proposal_class_not_in_taxonomy` (баг-фикс, см. §9); **нет боксов** →
  `no_text_available`.
- **Кадры** — `FrameManager.get(idx)` для кропов.
- **`union_timestamps_sec`** + Segmenter `frame_indices` — ось (no-fallback).
- **PP-OCR веса** (`ppocr_rec_onnx_v1_inprocess`, ModelManager, fp32 cuda) + словарь символов; либо tesseract.
- **`--retain-raw-ocr-text`** (дефолт false).

## 5. Выход

- **`ocr_raw`** — sparse-массив событий: `frame`, `bbox`, `rec_confidence`, при retain=true `text_raw/text_norm`,
  всегда `text_sha256`+`text_len`. Analytics-tier (не dense-seq).
- **Ось:** `frame_indices (N,) int32`, `times_s (N,) float32` (= `union_timestamps_sec[frame_indices]`).
- **`meta`:** `status`/`empty_reason`, `engine`, `retain_raw_ocr_text`, `models_used` (digest весов), timings.
- **Плотных числовых per-frame фич для Encoder НЕТ by design** — это sparse/analyst-компонент.

## 6. Фичи (важное/неочевидное)

- **Privacy-first дизайн** — прод-дефолт `retain=false`: наружу идёт только SHA-256 хеш + длина текста.
  Downstream может матчить по хешу (франшизы), не храня PII/сырой текст. Осознанное и правильное решение.
- **`rec_confidence`** — усреднённый max-prob CTC. На синтетике ~0.005 (OOD: модель обучена на реальных
  кропах, argmax верный, но softmax размазан) → **на синтетике не калибруется**; на реальных кропах ожидается высоким.
- **Sparse-события, а не dense-матрица** — текст появляется редко (несколько кадров), поэтому контракт —
  события `(frame,bbox,text)`, а не `(N,D)`. Правильная форма для разреженного сигнала.
- **CTC greedy decode** — простой (не beam search); достаточно для коротких плашек, может ошибаться на длинных.
- **Гейтинг детектором** — OCR-ит только то, что детектор пометил как `text_region` (экономия: не гоняет OCR
  по всему кадру), но полностью зависит от наличия и качества этого детектора.

## 7. Алгоритм / архитектура

- **Движок:** PP-OCR recognizer (PaddleOCR) в ONNX, inprocess через ModelManager (weights_digest фиксирован),
  fp32/cuda. Препроцесс 48×320, CTC greedy decode по словарю символов. Альтернатива — tesseract (skip если not-in-PATH).
- **Сложность:** OCR на каждый text_region-кроп; синтетика ~8.6–10.6 c/видео (доминирует init). Линейна по числу боксов.
- **Детерминизм:** golden max|Δ rec_confidence| = **0.0** (ONNX побайтово детерминирован) — образцово.

## 8. Оптимизации

- **Гейтинг детектором** (только text_region-кропы) — не гоняет OCR по всему кадру.
- **ONNX inprocess** через ModelManager — offline, детерминирован, digest весов в meta.
- **Privacy-хеширование** — компактный выход (sha+len) вместо сырого текста.
- **Skip-механизм** (переиспользован из tesseract-пути) для empty-случаев — единый путь.

## 9. Слабые места

- **Пуст на 100% реального корпуса (главное).** Все 6 storage-видео → `status=empty, no_text_available, 0 строк`.
  Причина: детектор в батче на **COCO-весах** (нет класса `text_region`) → 0 боксов → OCR нечего распознавать.
  Компонент **не произвёл ни одной реальной OCR-строки** за весь корпус.
- **Валидирован только на синтетике в изоляции.** Реального `text_region`-детектора в репо нет (зона владельца).
  Подтверждены контракт/ось/privacy/детерминизм/empty, но **качество распознавания и калибровка confidence на
  реальных кадрах — не подтверждены ничем**.
- **Полная зависимость от несуществующего детектора** — весь компонент бесполезен, пока владелец не обучит и
  не подключит `text_region`-детектор. Это блокер продуктовой пользы.
- **Баг-фикс не в git** — expected-empty (proposal_class_not_in_taxonomy) исправлен на поде (`main.py.bak_ocrfix`),
  требует переноса в репозиторий.
- **Галлюцинации на пустых боксах** — vidCblank (боксы есть, текста нет) дал 6 строк-галлюцинаций (conf~0.005);
  на реальных данных детектор не должен давать боксы на пустоте, но защиты по `min_rec_score` на синтетике не откалибровать.
- **CTC greedy (не beam)** — субоптимально на длинном/шумном тексте.
- **Версия 0.2** — самый «сырой» из разобранных (обычно 2.x).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс., блокер] Обучить/подключить `text_region`-детектор** — без него компонент пуст навсегда. Либо
   добавить класс в 41-таксономию core_object_detections, либо отдельный text-детектор (напр. DBNet/PP-OCR det).
2. **[выс.] Пере-прогнать детектор на реальных 41-весах** (не COCO) — тогда text_region-боксы появятся и OCR
   реально заработает (общий блокер с core_object_detections storage-корпусом на COCO).
3. **[выс.] Перенести баг-фикс expected-empty в git** (сейчас только на поде).
4. **[сред.] Откалибровать `min_rec_score`** на реальных кропах — отсечь галлюцинации/мусор; синтетика не годится.
5. **[низ.] Рассмотреть beam-search/языковую модель** для длинного текста, если качество greedy окажется низким.

## 11. Рекомендации по архитектуре / связям

- **Связать с core_object_detections таксономией** — `text_region` должен быть штатным классом детектора
  (сейчас OCR ждёт класс, которого в прод-весах нет). Это единая точка отказа всей текст-ветки.
- **Хеш-контракт с downstream** (franchise_recognition/text_scoring) — матч по `text_sha256` без сырого текста
  — закрепить как приватный паттерн; убедиться, что потребители умеют работать с хешами.
- **DBNet/детекция текста** мог бы заменить зависимость от YOLO-класса `text_region` более специализированным путём.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| U1 validate struct | 6 синт-роликов | VALID schema, rc=0 | контракт NPZ корректен |
| U2/U3 ось/health | синт | axis_match, monotonic, int32/float32, 0% NaN | ось/типы ок |
| U4 expected-empty | vidDnobox | status=empty, proposal_class_not_in_taxonomy, rc=0 (после фикса) | пустой путь чинён |
| U5/C4 golden | vidA×2 | max\|Δ rec_confidence\|=**0.0** (ONNX) | образцовый детерминизм |
| U6 разные длины | 3/12/200 кадров | rc=0 | масштабируется |
| C1 привязка к оси | A/B/E | 100% строк ∈ frame_indices | ось корректна |
| C2 privacy | retain=false/true | sha256+len без raw / raw при true | privacy работает |
| C3 различимость | синт | R варьируется (0/2/3/4/6); conf∈[0,1] | но conf~0.005 (OOD-синтетика) |
| **Реальный storage (мой прогон)** | **6 видео** | **все empty (no_text_available), 0 строк** | **0 реального OCR за весь корпус** |

Вывод: **инфраструктура (контракт/privacy/детерминизм/empty) образцова**, но **качество на реальных данных не
подтверждено вообще** — компонент пуст везде, гейтится несуществующим/неактивным детектором.

## 13. Интерпретируемость

- Экранный текст — **сам по себе интерпретируем** (это буквально слова на экране). При retain=true — прямой
  показ распознанного текста; при retain=false — метаданные (сколько текста, где).
- Предложение: overlay распознанных плашек на превью-кадры; «на видео N текстовых вставок»; но всё это —
  **после** появления реального text_region-детектора.

## 14. Польза для моделей

**Потенциально заметная, фактически нулевая.** Экранный текст (плотность, присутствие CTA/цен/брендов) —
осмысленный признак формата контента. Но: (1) компонент отдаёт sparse-события, не dense-seq — прямого
Encoder-входа нет by design; (2) на реальных данных выход пуст на 100%. Фактическая польза для моделей сейчас =
0; потенциал раскроется только после подключения детектора и, вероятно, через downstream (franchise/text_scoring),
а не напрямую.

## 15. Польза для аналитиков

**Потенциально высокая, фактически нулевая.** «Какой текст на экране» (заголовки, CTA, бренды) — ценнейшая
аналитика для креатора. Но на реальных данных — пусто на всех видео. Пока не появится `text_region`-детектор,
аналитик не получает ничего. После — это станет одним из самых наглядных компонентов.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 3 | Ясная нужная роль (экранный текст), но целиком гейтится несуществующим детектором |
| 5. Выход (контракт) | 4 | Чистый sparse-контракт + privacy-хеши + ось; образцовая privacy-модель |
| 6. Фичи | 3 | Privacy/детерминизм сильны; rec_confidence не калибрsuper, greedy-decode прост |
| 8. Оптимизации | 4 | Гейтинг детектором, ONNX-inprocess, privacy-хеш, skip-путь |
| 9. Слабые места (инверсно) | 1 | Пуст на 100% реальных данных, детектора нет, качество не подтверждено, фикс не в git |
| 12. Результаты тестов | 2 | Инфра-гейты образцовы, но всё на синтетике; реальные данные все empty |
| 13. Интерпретируемость | 4 | Текст самоинтерпретируем (когда есть) |
| 14. Польза для моделей | 2 | Потенциал есть, факт=0 (пусто + sparse, не Encoder-вход) |
| 15. Польза для аналитиков | 2 | Потенциал высок, факт=0 (пусто на всех видео) |

### Итоговые оценки

- **Польза для моделей: 2/5.** Экранный текст — осмысленный признак формата, но компонент отдаёт sparse-события
  (не dense Encoder-вход) и на реальных данных пуст на 100%. Инфраструктура (privacy/детерминизм) образцова, но
  фактическая польза для моделей сейчас нулевая; раскроется только после реального text_region-детектора, и то
  скорее через downstream.
- **Польза для аналитиков: 2/5.** «Текст на экране» потенциально — одна из самых ценных и наглядных аналитик,
  но на всём реальном корпусе выход пуст (нет детектора / COCO-веса). До подключения детектора аналитик не
  получает ничего; балл отражает факт, а не потенциал (который 4–5).

## 17. Источники

- `DataProcessor/VisualProcessor/core/model_process/ocr_extractor/main.py` (803 строки), `utils/validate_ocr.py`, `render.py`
- `DataProcessor/VisualProcessor/schemas/ocr_extractor_npz_v2.json`
- `DataProcessor/dp_models/spec_catalog/vision/ppocr_rec_onnx_v1_inprocess.yaml`
- `DataProcessor/docs/component_reports/ocr_extractor/{REPORT.md,CRITERIA.md,REPORT_synth.json}`
- `DataProcessor/docs/audit_v4/components/visual_processor/core/ocr_extractor_audit_v4.md`
- `DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md` (empty-семантика ocr)
- Cross-ref: `core_object_detections` (text_region-класс, storage на COCO-весах); downstream `franchise_recognition`, `text_scoring`, `story_structure`
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/ocr_extractor/ocr.npz`
  (**все status=empty, no_text_available, 0 строк**)

## 18. Визуализации

![ocr_extractor overview](ocr_extractor_overview.png)

`ocr_extractor_overview.png`: слева — все 6 реальных видео `status=empty` (0 OCR-строк при N=12…119 кадрах);
справа — схема пайплайна (text_region → crop → PP-OCR CTC → sha256 → sparse events) и «ворота» реальных данных:
детектор в батче на COCO-весах не даёт класса `text_region` → `no_text_available` на всех видео. Подтверждает:
инфраструктура образцова (детерминизм/privacy), но реальной OCR-строки корпус не содержит ни одной.
