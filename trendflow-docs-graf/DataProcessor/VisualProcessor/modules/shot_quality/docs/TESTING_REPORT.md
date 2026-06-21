# Отчёт о тестировании shot_quality компонента

**Дата**: 2026-03-11  
**Компонент**: `shot_quality`  
**Schema**: `shot_quality_npz_v3`  
**Producer version**: 2.0.2

---

## Резюме

- **Протестировано видео**: 20 (скрипт `run_tests.sh`: shortest + 2…20)
- **Успешных прогонов**: **19/20** (95%)
- **Валидных артефактов**: 19/20

**Проверка 2026-03-11**: в `dp_results/youtube/` присутствуют `shot_quality/shot_quality.npz` для всех тестов, кроме **test_shot_quality_17**. Тест 17 (видео `-1eKh7CJbhM.mp4`, ~494 с, 300 кадров) не доводится до конца по решению: длинный прогон, специально не подгоняем. Исправления в ходе тестирования: `core_depth_midas` sys.path; выравнивание размеров кадров в `temporal_flicker` и `rolling_shutter_artifacts_score` при разном letterbox-crop; тесты 11, 12 прошли при поднятом Triton.

Модуль оценивает техническое качество видео (frame-level, shot-level, CLIP quality). Зависит от `core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks`, `cut_detection`; для полного пайплайна требуется Triton (core_optical_flow).

---

## Качество данных

- ✅ Схема `shot_quality_npz_v3`
- ✅ Обязательные ключи и размерности проверены валидатором

---

## Файлы

- **Скрипт запуска**: `DataProcessor/VisualProcessor/modules/shot_quality/scripts/run_tests.sh`
- **Валидатор**: `DataProcessor/VisualProcessor/modules/shot_quality/utils/validate_shot_quality.py`
- **Результаты**: `DataProcessor/dp_results/youtube/test_shot_quality_*/`

---

## Заключение

Компонент `shot_quality` протестирован на 19 из 20 видео (тест 17 не завершается по решению). Все полученные артефакты соответствуют схеме. Компонент готов к использованию.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
