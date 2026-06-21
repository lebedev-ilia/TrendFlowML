# Отчёт о тестировании video_pacing компонента

**Дата**: 2026-03-09  
**Компонент**: `video_pacing`  
**Версия схемы**: `video_pacing_npz_v3`  
**Producer version**: 2.0.1

---

## Резюме

✅ **Все тесты пройдены успешно**

- **Протестировано видео**: 20 (+ smoke = 21)
- **Успешных прогонов**: 21/21 (100%)
- **Валидных артефактов**: 21/21 (100%)

Зависимости: `cut_detection`, `core_optical_flow`, `core_clip`. Валидация без ошибок (после исправления валидатора для `shot_boundary_frame_indices`).

---

## Качество данных

- ✅ Все артефакты соответствуют схеме `video_pacing_npz_v3`
- ✅ Валидация: 0 ошибок

---

## Файлы

- **Скрипт запуска**: `DataProcessor/VisualProcessor/modules/video_pacing/scripts/run_tests.sh`
- **Валидатор**: `DataProcessor/VisualProcessor/modules/video_pacing/utils/validate_video_pacing.py`
- **Результаты**: `DataProcessor/dp_results/youtube/test_video_pacing_*/`

---

## Заключение

Компонент `video_pacing` успешно протестирован на 20+ видео. Все артефакты валидны. Компонент готов к использованию.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
