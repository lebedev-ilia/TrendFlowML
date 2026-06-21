# Отчёт о тестировании emotion_diarization_extractor компонента

**Компонент**: `emotion_diarization_extractor`
**Schema**: `emotion_diarization_extractor_npz_v1`

---

## Резюме

- **Smoke**: ✅ проходит (1 короткое видео)
- **Подготовка**: `./DataProcessor/scripts/prepare_hf_cache.sh` — добавляет `preprocessor_config.json` в WavLM cache при неполной загрузке
- **Требования**: WavLM (microsoft/wavlm-large) в HF cache; main.py передаёт HF_HOME/HF_HUB_CACHE в subprocess

---

## Файлы

- **Валидатор**: `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/utils/validate_emotion_diarization.py`
- **Скрипт тестирования**: `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/scripts/run_tests.sh`
- **Скрипт анализа**: `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/utils/analyze_all_results.py`
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
