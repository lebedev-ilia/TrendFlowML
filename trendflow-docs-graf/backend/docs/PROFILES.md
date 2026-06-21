# Profiles

Профили анализа — это JSON‑конфигурации DataProcessor.
Backend хранит их в таблице `analysis_profiles`.

## 1) Нормализация профиля

При создании/обновлении профиля backend гарантирует минимум:

- если нет `visual.cfg_path`, подставляется дефолтный
  `configs/visual_triton_baseline_gpu_local.yaml`
- если нет `processors`, добавляется:
  - `audio.enabled=false, required=false`
  - `text.enabled=false, required=false`

Нормализация реализована в `routers/profiles.py::_normalize_config`.

## 2) config_hash

`config_hash` вычисляется как `sha256` от JSON с сортировкой ключей.

Это значение используется:

- для детерминированного сравнения конфигураций
- для DataProcessor (прокидывается в manifest)

## 3) Публичные профили (seed)

На старте приложения выполняется `seed_public_profiles(db, profiles_dir)` (см. `app/main.py`, `app/services/profiles.py`):

- читает YAML из `DataProcessor/profiles/*.yaml` (путь задаётся через `dataproc_root`)
- создаёт public профили (`is_public=true`) в таблице `analysis_profiles`, если профиля с таким именем (по имени файла) ещё нет
- `config_hash` вычисляется функцией `compute_config_hash` (SHA-256 от JSON с сортировкой ключей)

## 4) API

См. `backend/docs/API.md` → раздел Profiles.
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
