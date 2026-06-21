# Отчёт о тестировании voice_quality_extractor компонента

**Дата**: (обновить после прогона)  
**Компонент**: `voice_quality_extractor`  
**Версия**: `3.0.0`  
**Schema**: `voice_quality_extractor_npz_v1`

---

## Резюме

- **Протестировано видео**: 20 (от 12 сек до 759 сек)
- **Успешных прогонов**: X/20
- **Валидных артефактов**: X/20

---

## Статистика по видео

| # | Видео | Длительность | Статус |
|---|-------|--------------|--------|
| 1 | -Q6fnPIybEI.mp4 | 12 сек | |
| 2 | -7Ei8e05x30.mp4 | 14 сек | |
| ... | ... | ... | |

---

## Качество данных

### Валидация схемы

Все артефакты соответствуют схеме `voice_quality_extractor_npz_v1`:
- Обязательные ключи: feature_names, feature_values, segment_*_sec, segment_mask, meta
- Метаданные корректны

---

## Файлы

- **Валидатор**: `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/utils/validate_voice_quality.py`
- **Скрипт батч-тестирования**: `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/scripts/run_tests.sh`
- **Скрипт анализа**: `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/utils/analyze_all_results.py`
- **Результаты**: `DataProcessor/dp_results/youtube/test_voice_quality_*/`
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
