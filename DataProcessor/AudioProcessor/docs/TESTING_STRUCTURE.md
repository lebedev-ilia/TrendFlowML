# Структура тестирования AudioProcessor (по образцу VisualProcessor)

Архитектура папок и скриптов для каждого extractor'а соответствует VisualProcessor (action_recognition).

---

## Структура папок extractor'а

```
DataProcessor/AudioProcessor/src/extractors/<extractor_name>/
├── docs/
│   ├── README.md           # Описание компонента
│   ├── SCHEMA.md           # Схема данных
│   └── TESTING_REPORT.md   # Отчёт о тестировании (обновить после прогона)
├── scripts/
│   ├── run_tests.sh        # Запуск тестов на 20 видео (per-component)
│   └── run_analyze.sh      # Анализ результатов
├── utils/
│   ├── __init__.py
│   ├── render.py           # Рендер NPZ → HTML/JSON
│   ├── validate_<name>.py  # Валидатор NPZ артефактов
│   └── analyze_all_results.py
├── main.py
└── __init__.py
```

---

## Глобальные скрипты тестирования

Расположение: `DataProcessor/scripts/`

| Скрипт | Назначение |
|--------|------------|
| `prepare_hf_cache.sh` | Подготовка HF cache для emotion_diarization (WavLM preprocessor_config.json) |
| `run_smoke_all_components.sh` | Smoke: 1 короткое видео на каждый из 21 компонентов |
| `run_full_all_components.sh` | Full: 20 видео на каждый компонент (420 запусков) |
| `validate_smoke_results.sh` | Валидация smoke-результатов |
| `validate_full_results.sh` | Валидация full-результатов |

---

## Структура результатов

### Smoke-тест

```
DataProcessor/dp_results/smoke_test/
└── youtube/
    ├── smoke_asr_shortest/smoke_asr_shortest/asr_extractor/
    ├── smoke_clap_shortest/smoke_clap_shortest/clap_extractor/
    ├── ...
    └── smoke_voice_quality_shortest/...
```

### Полное тестирование

```
DataProcessor/dp_results/full_test/
└── youtube/
    ├── full_asr_v1/full_asr_v1/asr_extractor/
    ├── full_asr_v2/...
    ├── ...
    ├── full_asr_v20/...
    ├── full_clap_v1/...
    └── ...
```

---

## Конфиги

Профили в `configs/audit_v3/audio/` (корень репозитория):
- `profile_<key>.yaml` — один extractor на профиль (для изолированного теста)
- Скрипты `run_smoke_all_components.sh` и `run_full_all_components.sh` используют `$REPO_ROOT/configs/audit_v3/audio`

---

## Запуск тестов по компоненту

```bash
# Из extractor'а (20 видео)
./DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/scripts/run_tests.sh

# Анализ
./DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/scripts/run_analyze.sh
```

---

## Связанные документы

- [TESTING_GUIDE.md](TESTING_GUIDE.md) — полное руководство по тестированию
