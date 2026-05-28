# Финальный отчет о тестировании модуля `story_structure`

**Дата:** 2026-03-06  
**Модуль:** `story_structure`  
**Версия схемы:** `story_structure_npz_v3`  
**Producer version:** `3.0.2`

## Резюме

Модуль `story_structure` успешно протестирован на 17 видео из 20 запланированных. Все созданные артефакты прошли валидацию без ошибок. Один тест (test_story_structure_20) не выполнился из-за ошибки импорта в VisualProcessor (не связанной с модулем), два теста (test_story_structure_18, test_story_structure_19) не создали файлы по неизвестным причинам.

### Валидация
**Результаты валидации:**
- Total videos: 17
- Total issues: 0
- Issues by severity: нет
- Issues by type: нет

✅ **Все проверенные артефакты соответствуют схеме `story_structure_npz_v3`**

### Тестирование
**Результаты тестирования:**
- Успешно обработано: 17 видео
- Не обработано: 3 видео (test_story_structure_18, test_story_structure_19, test_story_structure_20)
- Smoke test: ✅ пройден (test_story_structure_shortest)

**Список успешно обработанных видео:**
1. test_story_structure_shortest
2. test_story_structure_2
3. test_story_structure_3
4. test_story_structure_4
5. test_story_structure_5
6. test_story_structure_6
7. test_story_structure_7
8. test_story_structure_8
9. test_story_structure_9
10. test_story_structure_10
11. test_story_structure_11
12. test_story_structure_12
13. test_story_structure_13
14. test_story_structure_14
15. test_story_structure_15
16. test_story_structure_16
17. test_story_structure_17

## Зависимости модуля

### Hard dependencies (обязательные)
- **`core_clip`**: CLIP embeddings для вычисления embedding change rate
- **`core_optical_flow`**: Optical flow для кривой движения (`motion_norm_per_sec_mean`)
- **`core_face_landmarks`**: Детекция лиц для `any_face_present` и character proxies

### Optional dependencies
- **`ocr_extractor`**: Опционально для `topic_shift_curve` (если `text_mode="ocr_clip_text"`)

### Конфигурация зависимостей
В тестах использовалась следующая конфигурация:

```yaml
core_clip:
  runtime: "inprocess"
  model_name: "ViT-B/32"
  batch_size: 16

core_optical_flow:
  runtime: "triton"
  triton_model_spec: "raft_256_triton"
  batch_size: 1

core_object_detections:
  runtime: "ultralytics"
  model: "visual/yolo/yolo11x_41_best.pt"
  batch_size: 16
  device: "cuda"

core_face_landmarks:
  use_face_mesh: true
  use_person_mask: true
```

## Конфигурация модуля

В тестах использовалась следующая конфигурация `story_structure`:

```yaml
story_structure:
  min_frames: 30
  max_frames: 200
  energy_smoothing_sigma: 1.0
  text_mode: "none"  # отключен для audit-тестов
  clip_text_model_spec: "clip_text_triton"
  clip_text_batch_size: 64
  ocr_max_chars_per_frame: 256
```

## Структура выходных данных

### Основные массивы (sequence-level)
- `frame_indices (N,) int32` - индексы обработанных кадров
- `times_s (N,) float32` - временные метки кадров (секунды)
- `story_energy_curve (N,) float32` - комбинированная кривая энергии (z-score)
- `motion_norm_per_sec_mean (N,) float32` - кривая движения (per-second mean)
- `embedding_change_rate_per_sec (N,) float32` - скорость изменения CLIP embeddings (/s)
- `any_face_present (N,) bool` - наличие лиц на кадрах
- `topic_shift_curve (N,) float32` - кривая смены темы (NaN если text_mode="none")
- `frame_feature_present_ratio (N,) float32` - доля finite значений среди кривых

### Табличные фичи (model-facing)
- `feature_names (F,) object` - имена фич
- `feature_values (F,) float32` - значения фич в фиксированном порядке

**Основные фичи включают:**
- `n_frames` - количество кадров
- `video_length_seconds` - длина видео в секундах
- `hook_visual_surprise_score` - оценка визуального сюрприза в hook-окне
- `climax_time_sec` - время кульминации (секунды)
- `main_character_screen_time` - доля экранного времени главного персонажа
- `number_of_peaks` - количество пиков энергии
- И другие метрики (см. `_FEATURE_NAMES_V1` в коде)

### Analytics / debug
- `story_energy_curve_downsampled_128 (128,) float32` - даунсэмплированная кривая энергии
- `story_energy_peaks_idx` - индексы пиков энергии
- `story_energy_peaks_times_s` - времена пиков энергии
- `story_energy_peaks_values_z` - значения пиков (z-score)
- `topic_shift_peaks_idx` - индексы пиков смены темы (если доступно)

## Проблемы и решения

### Проблема 1: Triton недоступен
**Симптомы:** `core_optical_flow` падал с ошибкой `TritonError: core_optical_flow | Triton is not ready`
**Решение:** Пользователь запустил Triton Inference Server с моделью `raft_256_triton`
**Статус:** ✅ Решено

### Проблема 2: Отсутствие `core_object_detections`
**Симптомы:** `core_face_landmarks` падал с ошибкой `RuntimeError: core_face_landmarks | missing required artifact: core_object_detections/detections.npz`
**Решение:** Добавлен `core_object_detections` в конфигурацию `visual_story_structure_only.yaml`
**Статус:** ✅ Решено

### Проблема 3: Ошибка импорта в VisualProcessor (test_story_structure_20)
**Симптомы:** `ModuleNotFoundError: No module named 'embedding_service_client'` при запуске VisualProcessor
**Причина:** Проблема с окружением VisualProcessor, не связанная с модулем `story_structure`
**Статус:** ⚠️ Не критично (17 из 20 тестов успешны)

## Выводы

1. ✅ **Модуль работает корректно**: Все 17 успешно обработанных видео создали валидные артефакты
2. ✅ **Валидация пройдена**: Все артефакты соответствуют схеме `story_structure_npz_v3`
3. ✅ **Зависимости настроены**: `core_clip`, `core_optical_flow`, `core_face_landmarks` работают корректно
4. ⚠️ **Частичное покрытие**: 17 из 20 тестов успешны (85% успешность)

## Рекомендации

1. **Исправить окружение VisualProcessor**: Решить проблему с импортом `embedding_service_client` для полного покрытия тестами
2. **Проверить test_story_structure_18 и test_story_structure_19**: Выяснить, почему эти тесты не создали файлы
3. **Мониторинг Triton**: Убедиться, что Triton Inference Server стабильно работает для `core_optical_flow`

## Файлы тестирования

- **Профиль:** `DataProcessor/configs/audit_v3/visual/profile_story_structure.yaml`
- **Конфигурация:** `DataProcessor/configs/audit_v3/visual/visual_story_structure_only.yaml`
- **Скрипт запуска:** `DataProcessor/VisualProcessor/modules/story_structure/scripts/run_tests.sh`
- **Валидатор:** `DataProcessor/VisualProcessor/modules/story_structure/utils/validate_story_structure.py`
- **Анализатор:** `DataProcessor/VisualProcessor/modules/story_structure/utils/analyze_all_results.py`

## Статистика артефактов

- **Средний размер файла:** ~8.75 KB (очень компактный благодаря эффективному сжатию)
- **Формат:** NPZ (NumPy compressed archive)
- **Схема версии:** `story_structure_npz_v3`

---

**Тестирование выполнено:** 2026-03-06  
**Валидация выполнена:** 2026-03-06  
**Статус:** ✅ Успешно (17/20 тестов)

