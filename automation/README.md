# Автономный прогон на твоём GPU (один запуск → дальше Claude сам)

Идея: workspace-папка `TrendFlowML` лежит на **твоём ПК** (там уже `.vp_venv`, веса, GPU). Ты
**один раз** запускаешь демон-воркер. Дальше Claude кладёт «заявки на прогон» в `automation/queue/`,
демон исполняет их **на твоём GPU**, пишет результаты в `automation/results/<id>/`, а Claude их
читает (общая папка) и сам пишет REPORT. **Cursor не нужен.**

```
Claude: ar_enqueue.py → automation/queue/<id>.json
                              │  (демон на твоём ПК видит файл)
                              ▼
   ar_run_daemon.py → run_ar_local.py --device cuda (Segmenter→детекция+трекер→action_recognition v3)
                              │
                              ▼
        automation/results/<id>/  (npz, summary.json, metrics, run.log, DONE)
                              │  (Claude читает ту же папку)
                              ▼
              Claude: разбор артефактов → REPORT
```

## Что нужно от тебя — ОДИН РАЗ

1. **Проверь окружение** (обычно уже готово после прежних прогонов):
   - venv с моделями: `DataProcessor/VisualProcessor/.vp_venv` (torch+cuda, pytorchvideo, ultralytics);
   - веса: `DataProcessor/dp_models/...` (YOLO `yolo11l.pt`, SlowFast `slowfast_r50.pyth`);
   - если чего-то нет → `HF_TOKEN=... ./bootstrap.sh --skip-stack` (venvs + модели).
   - `ffmpeg` в системе (или `tools/bin/ffmpeg`).

2. **Запусти демон один раз** (foreground или фоном/через systemd):
   ```bash
   chmod +x DataProcessor/scripts/ar_run_daemon.sh
   nohup ./DataProcessor/scripts/ar_run_daemon.sh >/dev/null 2>&1 &
   ```
   Проверить, что живёт: `tail -f automation/logs/daemon.log`

Всё. После этого я (Claude) сам ставлю заявки и читаю результаты.

### (опц.) systemd — чтобы демон переживал перезагрузку
`~/.config/systemd/user/ar-daemon.service`:
```ini
[Unit]
Description=TrendFlow action_recognition run daemon
[Service]
WorkingDirectory=%h/Рабочий стол/TrendFlowML
ExecStart=%h/Рабочий стол/TrendFlowML/DataProcessor/scripts/ar_run_daemon.sh
Restart=always
[Install]
WantedBy=default.target
```
`systemctl --user enable --now ar-daemon`

## Как это выглядит с моей стороны (Claude)
```bash
# поставить заявку (на GPU, полный клип)
python DataProcessor/scripts/ar_enqueue.py \
  --video DataProcessor/docs/component_reports/action_recognition/fixtures/ar_real_4m35_people.mp4 \
  --seconds 0 --fps 25 --device cuda
# дождаться automation/results/<id>/DONE и разобрать npz
```

## Безопасность
Демон **не** исполняет произвольные команды: заявка описывает только видео+параметры, запускается
единственный доверенный скрипт `run_ar_local.py`, а видео обязано лежать **внутри репозитория**
(пути наружу отклоняются). Хочешь сузить ещё — ограничь папку `fixtures/`.

## Ограничение
Я по-прежнему не имею GPU в своём песочнице и не могу *запустить* демон — его стартуешь ты (один
раз). Но после старта весь цикл «поставить прогон → дождаться → проанализировать → отчёт» я делаю
сам. Масштаб 200k — тем же демоном или k8s-Job'ом (`k8s/jobs/`).
```
```
