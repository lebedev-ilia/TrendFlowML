## configs/ — единая точка входа для локальных прогонов (MVP)

Цель: временно держать **один “верхний” конфиг** для end‑to‑end экспериментов:
- глобальный оркестратор (DynamicBatch) → DataProcessor → VisualProcessor/AudioProcessor
- baseline DAG + scheduler batch_size overrides
- пост‑проверки (валидаторы + HTML)

### Файлы

- `configs/visual_triton_baseline_gpu_local.yaml`
  - конфиг VisualProcessor (enable flags + параметры компонентов)
  - используется DynamicBatch как `--visual-cfg-template`

- `configs/profile_triton_baseline_gpu_local.yaml`
  - analysis profile для DataProcessor:
    - включает audio tier‑0
    - указывает `visual.cfg_path` на файл выше
  - используется DynamicBatch как `--profile-path` (можно сделать дефолтом)

### Почему так

В репо было много `config_triton_*` под частные тесты; сейчас нам важнее **один каноничный** конфиг для цепочки “10 видео”.
---

## Навигация

[Vault](../docs/MAIN_INDEX.md)
