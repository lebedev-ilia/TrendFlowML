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

На старте приложения выполняется `seed_public_profiles(db)`:

- читает YAML из `DataProcessor/profiles/*.yaml`
- создаёт public профили (`is_public=true`) если их ещё нет в БД

## 4) API

См. `backend/docs/API.md` → раздел Profiles.

