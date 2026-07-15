# Лог настройки пода — что сделано (2026-07-05), как воспроизвести

**Итог:** Claude сам подключился к GPU-поду и прогнал полную цепочку компонента
`Segmenter → core_object_detections(+appearance-tracker) → action_recognition v3` на GPU
(RTX A4500). Оба валидатора (вход+выход) ✅, npz+metrics созданы. **Cursor не нужен для прогонов.**

## Проверенный успешный прогон (20-сек клип 4:35, `--device cuda`)
```
stages: trim ok / segmenter ok (22s) / detections ok (33s) / action_recognition ok (20s)
result: clip_count=11, num_tracks=2, mean_clips_per_track=5.5,
        embedding (11, 2304) penultimate, classes_available=true
validators: ✅ входной контракт выполнен · ✅ соответствует schema v3
```

## Нюансы окружения пода (важно для будущих сессий)
1. **SSH из песочницы Cowork** ругается на системный `ssh_config.d` → добавляй `-F /dev/null`:
   ```
   ssh -F /dev/null -i <key> -p <PORT> root@<HOST> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ...
   ```
   Ключ и /tmp не переживают между bash-вызовами → в начале каждого вызова:
   `cp automation/runpod_ssh/id_ed25519 /tmp/k && chmod 600 /tmp/k`.
2. **Передача кода** — `rsync` (ставится на под из apt). Флаги: `-rltz --mkpath --no-perms
   --no-owner --no-group` (сетевой том не даёт chown). Excludes брать в кавычки
   (`--exclude='*.mp4'`) — иначе имя фикстуры с `-` съедается как опция.
3. **Что синхронизировать:** весь код `DataProcessor/` (кроме `*venv*`, `*/artifacts`,
   бинарных весов) + нужные веса (`yolo11l`, `slowfast_r50.pyth`) + `spec_catalog` +
   фикстуры. Веса VideoMAE/Hiera/OSNet — публичные, тянутся на поде, не грузить.
4. **Пакеты на поде** (system python, torch уже в образе) — см. `pod_setup.sh`:
   `pytorchvideo ultralytics opencv-python-headless transformers scipy scikit-learn
   scikit-image pyyaml pillow`.
5. **Вес SlowFast**: ModelManager ждёт `dp_models/visual/action_recognition/slowfast_r50/
   slowfast_r50.pyth`, а в унифицированном репо он под `bundled_models/...` → `pod_setup.sh`
   копирует его на нужный путь.

## Быстрый повтор (следующая сессия)
1. Прочитать `POD_CONNECTION.md` (Host/Port/статус).
2. rsync кода+весов (см. §2-3) на `/workspace/TrendFlowML`.
3. `ssh <pod> 'bash /workspace/TrendFlowML/automation/runpod_ssh/pod_setup.sh'`.
4. Прогон: `run_ar_local.py --video <mp4> --seconds 0 --fps 25 --device cuda`.
5. Забрать `rs/.../*.npz` + `summary.json` (`rsync` обратно) → анализ.

## Стоимость
Под **Running** — капает оплата. По завершении прогонов остановить (Stop) в консоли RunPod,
либо через API (ключ в `POD_CONNECTION.md`): `curl -X POST -H "Authorization: Bearer <KEY>"
https://rest.runpod.io/v1/pods/<id>/stop`.
