# Документация AudioProcessor

Главный индекс документации.

---

## Навигация

| Документ | Описание |
|----------|----------|
| [MAIN_INDEX.md](MAIN_INDEX.md) | Индекс всех 21 extractors с ссылками на README |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | **Руководство по тестированию** — smoke, full, валидация |
| [TESTING_STRUCTURE.md](TESTING_STRUCTURE.md) | Структура папок extractors (docs, scripts, utils) |
| [BATCH_PROCESSING_PLAN.md](BATCH_PROCESSING_PLAN.md) | План батч-обработки |

---

## Тестирование (кратко)

```bash
# Подготовка HF cache (emotion_diarization) — перед первым smoke
./DataProcessor/scripts/prepare_hf_cache.sh

# Smoke: 1 короткое видео на каждый из 21 компонентов
./DataProcessor/scripts/run_smoke_all_components.sh

# Full: 20 видео на каждый компонент (420 запусков)
./DataProcessor/scripts/run_full_all_components.sh

# Валидация
./DataProcessor/scripts/validate_smoke_results.sh
./DataProcessor/scripts/validate_full_results.sh
```

**Результаты**: `DataProcessor/dp_results/smoke_test/` и `DataProcessor/dp_results/full_test/`

---

## Audit v3

Отчёты по компонентам: [audit_v3/components/](audit_v3/components/)
