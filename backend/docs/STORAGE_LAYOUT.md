# Storage layout

Backend использует локальный filesystem (dev‑режим) и строит единые пути
для raw, frames и result_store.

## 1) Директории (по умолчанию)

```
<repo>/storage/
  raw/                      # raw uploads
    <video_id>/video.<ext>
    tmp/<upload_id>/...
  frames_dir/
    <video_id>/video/...    # frames_dir от Segmenter
  result_store/
    <platform_id>/<video_id>/<run_id>/...

<repo>/example/example_videos/
  <video_id>.<ext>          # копия raw для удобства тестов

<repo>/storage/profiles_cache/
  <run_id>/profile.yaml     # профиль для запуска DataProcessor

<repo>/storage/state/
  <platform_id>/<video_id>/<run_id>/state_events.jsonl
```

## 2) Raw uploads

Файл из upload‑flow попадает в:

- `storage/raw/<video_id>/video.<ext>`
- копия в `example/example_videos/<video_id>.<ext>`

Дедуп реализован через таблицу `video_files` по `sha256_hex`, но
физическая копия файла остаётся в `storage/raw`.

## 3) Result store (source‑of‑truth)

`result_store/<platform_id>/<video_id>/<run_id>/manifest.json`  
`result_store/<platform_id>/<video_id>/<run_id>/<component_name>/*.npz`

Backend читает `manifest.json` и регистрирует артефакты (NPZ/JSON/HTML)
в таблице `artifacts`.

## 4) Frames dir

`frames_dir/<video_id>/video/...`  
Используется для quality‑scripts и для UI‑метрик, если компонент
запросил `--frames-dir`.

## 5) state_events.jsonl

DataProcessor пишет `state_events.jsonl`. Backend tail‑ит этот файл
в `tasks.py::_tail_state_events` и превращает записи в WS‑события.

