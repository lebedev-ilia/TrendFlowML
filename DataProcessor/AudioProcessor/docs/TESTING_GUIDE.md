# Руководство по тестированию AudioProcessor

Полное руководство по запуску smoke-тестов и полного тестирования всех 21 extractors.

---

## Оглавление

1. [Обзор](#обзор)
2. [Структура результатов](#структура-результатов)
3. [Smoke-тест (1 короткое видео на компонент)](#smoke-тест)
4. [Полное тестирование (20 видео на компонент)](#полное-тестирование)
5. [Валидация и анализ](#валидация-и-анализ)
6. [Устранение ошибок](#устранение-ошибок)

---

## Обзор

| Режим | Скрипт | Видео на компонент | Результаты |
|-------|--------|-------------------|------------|
| **Smoke** | `run_smoke_all_components.sh` | 1 (короткое) | `dp_results/smoke_test/` |
| **Full** | `run_full_all_components.sh` | 20 | `dp_results/full_test/` |

**Smoke-тест** — быстрая проверка всех компонентов на одном коротком видео. Используется для выявления и исправления ошибок перед полным прогоном.

**Полное тестирование** — 20 видео на каждый компонент для валидации качества (по образцу VisualProcessor).

### Статус smoke-тестов (21/21)

Все 21 компонент проходят smoke при корректной подготовке:
- **Подготовка**: `./DataProcessor/scripts/prepare_hf_cache.sh` (emotion_diarization требует WavLM cache)
- **Валидация**: `./DataProcessor/scripts/validate_smoke_results.sh` → 21/21

---

## Структура результатов

### Smoke-тест

```
DataProcessor/dp_results/smoke_test/
└── youtube/
    ├── smoke_asr_shortest/
    │   └── smoke_asr_shortest/
    │       └── asr_extractor/
    │           └── asr_extractor_features.npz
    ├── smoke_clap_shortest/
    │   └── smoke_clap_shortest/
    │       └── clap_extractor/
    ├── smoke_tempo_shortest/
    ├── ... (21 компонент)
    └── smoke_voice_quality_shortest/
```

### Полное тестирование

```
DataProcessor/dp_results/full_test/
└── youtube/
    ├── full_asr_v1/
    │   └── full_asr_v1/
    │       └── asr_extractor/
    ├── full_asr_v2/
    ├── ...
    ├── full_asr_v20/
    ├── full_clap_v1/
    ├── ...
    └── full_voice_quality_v20/
```

Каждый run: `{rs_base}/youtube/{video_id}/{run_id}/{component_name}/`

---

## Smoke-тест

**Цель**: Проверить, что все 21 компонент запускаются без ошибок на одном коротком видео.

**Видео**: `-Q6fnPIybEI.mp4` (~12 сек)

**Перед первым запуском** (для emotion_diarization):
```bash
./DataProcessor/scripts/prepare_hf_cache.sh
```

**Запуск**:

```bash
cd "/media/ilya/Новый том/TrendFlowML"
chmod +x DataProcessor/scripts/run_smoke_all_components.sh
./DataProcessor/scripts/run_smoke_all_components.sh
```

**Логи**: `/tmp/audio_smoke_<component>_shortest.log`

**После прогона**: Исправьте ошибки в компонентах, затем повторите smoke до успешного прохождения всех 21.

---

## Полное тестирование

**Цель**: Полная валидация на 20 видео разной длительности (12 сек — 759 сек).

**Запуск** (последовательно по всем компонентам):

```bash
cd "/media/ilya/Новый том/TrendFlowML"
chmod +x DataProcessor/scripts/run_full_all_components.sh
./DataProcessor/scripts/run_full_all_components.sh
```

**Внимание**: Полный прогон занимает много времени (21 компонент × 20 видео = 420 запусков).

**Логи**: `/tmp/audio_full_<component>_<run_id>.log`

---

## Валидация и анализ

### Валидация smoke-результатов

```bash
./DataProcessor/scripts/validate_smoke_results.sh
```

### Валидация full-результатов

```bash
./DataProcessor/scripts/validate_full_results.sh
```

### Анализ по компоненту

```bash
cd DataProcessor
PYTHONPATH=AudioProcessor/src .data_venv/bin/python3 \
  AudioProcessor/src/extractors/<extractor>/utils/analyze_all_results.py \
  --rs-base dp_results/full_test \
  --run-id-prefix "full_<key>_" \
  --component-name "<extractor>"
```

---

## Устранение ошибок

1. **Smoke не прошёл** — проверьте лог `/tmp/audio_smoke_<component>_shortest.log`
2. **Зависимости** — некоторые extractors требуют другие (например, `voice_quality` → `pitch`, `speech_analysis` → `asr`+`speaker_diarization`). Обновите `configs/audit_v3/audio/profile_<key>.yaml` (корень репозитория)
3. **Модели** — убедитесь, что `DP_MODELS_ROOT` указывает на bundle с моделями
4. **GPU/CPU** — при нехватке GPU некоторые extractors могут падать; проверьте `device: "cpu"` в профиле
5. **emotion_diarization** — требует WavLM (microsoft/wavlm-large) в HuggingFace cache:
   - **Подготовка кэша**: запустите `./DataProcessor/scripts/prepare_hf_cache.sh` — скрипт добавит недостающий `preprocessor_config.json` в snapshots (при неполной загрузке)
   - Вариант A: `DP_MODELS_ROOT/hf_cache/hub` с `models--microsoft--wavlm-large` (symlink на `~/.cache/huggingface/hub` или копия)
   - Вариант B: `~/.cache/huggingface/hub/models--microsoft--wavlm-large` — main.py передаёт HF_HOME в subprocess
   - При ошибке "couldn't find them in the cached files" — скачайте модель: `huggingface-cli download microsoft/wavlm-large`, затем `prepare_hf_cache.sh`

---

## Чеклист перед первым запуском

1. **Модели**: `DP_MODELS_ROOT` указывает на `DataProcessor/dp_models/bundled_models` (или аналог)
2. **emotion_diarization**: `./DataProcessor/scripts/prepare_hf_cache.sh` — подготовка WavLM cache
3. **Видео**: `example/example_videos/-Q6fnPIybEI.mp4` (или `VIDEOS_DIR`)

## Конфиги

Профили: `configs/audit_v3/audio/` (корень репозитория). Скрипты используют `$REPO_ROOT/configs/audit_v3/audio`.

---

## Расположение скриптов

Все скрипты в `DataProcessor/scripts/`:
- `prepare_hf_cache.sh` — подготовка HF cache для emotion_diarization (WavLM)
- `run_smoke_all_components.sh`
- `run_full_all_components.sh`
- `validate_smoke_results.sh`
- `validate_full_results.sh`

---

## Связанные документы

- [TESTING_STRUCTURE.md](TESTING_STRUCTURE.md) — структура папок extractors
- [README.md](README.md) — главный индекс документации
- [MAIN_INDEX.md](MAIN_INDEX.md) — индекс всех extractors
