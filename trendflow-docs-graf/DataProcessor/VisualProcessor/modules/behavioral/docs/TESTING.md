# Тестирование behavioral компонента

## Обзор

Этот документ описывает процесс тестирования компонента `behavioral` на нескольких видео с фокусом на качество данных, а не на скорость обработки.

## Конфигурация

### Профиль для тестирования

Используется профиль: `DataProcessor/configs/audit_v3/visual/profile_behavioral.yaml`

Этот профиль включает:
- `core_object_detections` (зависимость)
- `core_face_landmarks` (зависимость)
- `behavioral` (тестируемый компонент)

### Зависимости

Behavioral компонент требует:
1. **core_object_detections** - для person-mask gating
2. **core_face_landmarks** - для landmarks (pose, hands, face)

## Процесс тестирования

### 1. Подготовка

Убедитесь, что:
- Виртуальное окружение активировано: `DataProcessor/.data_venv`
- Конфиг профиль создан: `DataProcessor/configs/audit_v3/visual/profile_behavioral.yaml`
- Видео доступны в `example/example_videos/`

### 2. Запуск теста на одном видео

```bash
cd /path/to/TrendFlowML
DataProcessor/.data_venv/bin/python DataProcessor/main.py \
  --video-path "example/example_videos/-Q6fnPIybEI.mp4" \
  --global-config DataProcessor/configs/global_config.yaml \
  --profile-path DataProcessor/configs/audit_v3/visual/profile_behavioral.yaml \
  --platform-id youtube \
  --video-id test_behavioral_shortest \
  --run-id test_behavioral_shortest \
  --output-dir DataProcessor/dp_results
```

### 3. Проверка результатов

После успешного запуска проверьте:

1. **NPZ артефакт**:
   ```
   DataProcessor/dp_results/youtube/{video_id}/{run_id}/behavioral/behavioral_features.npz
   ```

2. **Render контекст**:
   ```
   DataProcessor/dp_results/youtube/{video_id}/{run_id}/behavioral/_render/render_context.json
   ```

3. **HTML рендер**:
   ```
   DataProcessor/dp_results/youtube/{video_id}/{run_id}/behavioral/_render/render.html
   ```

### 4. Валидация результатов

Используйте персональный валидатор:

```bash
DataProcessor/.data_venv/bin/python DataProcessor/VisualProcessor/modules/behavioral/validate_behavioral.py \
  --results-base DataProcessor/dp_results \
  --videos youtube:test_behavioral_shortest:test_behavioral_shortest \
           youtube:test_behavioral_2:test_behavioral_2
```

## Проверка качества данных

### Ключевые метрики для проверки

1. **Landmarks present ratio** - доля кадров с обнаруженными landmarks
   - Низкое значение (< 10%) может указывать на проблемы с детекцией
   - Высокое значение (> 50%) - хороший знак

2. **Avg engagement** - средний индекс вовлеченности [0, 1]
   - Должен быть в диапазоне [0, 1]
   - Сравнивайте между видео для выявления аномалий

3. **Avg confidence** - средний индекс уверенности [0, 1]
   - Должен быть в диапазоне [0, 1]

4. **Avg stress** - средний индекс стресса [0, 1]
   - Должен быть в диапазоне [0, 1]

5. **Gesture rate per sec** - частота жестов в секунду
   - Сравнивайте между видео

### Проверка рендеров

1. Откройте HTML рендер в браузере
2. Проверьте:
   - Key facts (schema_version, status, frames_count)
   - Summary метрики
   - Timeline графики (speech_activity_proxy, arm_openness, stress_proxy)
   - Distribution statistics
   - Top/Anti-top таблицы

### Выявление проблем

Валидатор проверяет:
- ✅ Статус компонента (должен быть "ok")
- ✅ Наличие всех обязательных ключей
- ✅ Согласованность размеров массивов
- ✅ Наличие агрегированных метрик
- ✅ Диапазоны значений метрик
- ✅ Аномалии при сравнении между видео

## Результаты тестирования

### Тест 1: Самое короткое видео (-Q6fnPIybEI.mp4, 3.2MB)

- **Статус**: ✅ OK
- **Frames**: 250
- **Landmarks present ratio**: 14.40%
- **Avg engagement**: 0.7795
- **Avg confidence**: 0.5924
- **Avg stress**: 0.1516
- **Gesture rate**: 1.1732 per sec

### Тест 2: Второе видео (-ZLHxCNCpdA.mp4, 6.9MB)

- **Статус**: ✅ OK
- **Frames**: 250
- **Landmarks present ratio**: 12.00%
- **Avg engagement**: 0.7265
- **Avg confidence**: 0.6406
- **Avg stress**: 0.1632
- **Gesture rate**: 0.3336 per sec

## Рекомендации

1. **Продолжить тестирование** на остальных видео по нарастанию длительности
2. **Сравнивать метрики** между видео для выявления паттернов
3. **Проверять рендеры** для визуальной оценки качества
4. **Использовать валидатор** для автоматической проверки

## Известные проблемы

1. **RuntimeWarning: Mean of empty slice** - возникает когда нет данных для ранних/поздних сегментов видео. Это нормально для коротких видео или видео с низким landmarks_present_ratio.

2. **WARNING: exec_order missing enabled components** - предупреждение о том, что behavioral не включен в exec_order. Это не критично, компонент все равно выполняется.

3. **WARNING: Failed to generate render-context for core_object_detections** - проблема с сериализацией numpy массивов в JSON. Это не влияет на behavioral компонент.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
