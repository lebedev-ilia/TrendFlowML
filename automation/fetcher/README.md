# Fetcher-инфраструктура — 3 постоянных CPU-пода RunPod (сбор датасета YouTube)

Отдельная система от `automation/runner/` (ML-валидация компонентов) — не связаны кодом, только
общий реестр `automation/runner/state/machines.json`, куда эти поды зарегистрированы с
`policy=persistent`, `kind=fetcher`, чтобы ML-раннер их никогда не гасил/не удалял (см.
`automation/runner/runpod_api.py`, шапка модуля, и `AGENT_CONTEXT.md` раздел 3.4).

## Инфраструктура (уже создана, 2026-07-16)

| Под | Роль | vCPU/RAM | Volume | ЦОД | $/ч |
|---|---|---|---|---|---|
| `fetcher-main` | discover (непрерывно, авто-восстановление после исчерпания квоты) + workers шард 0/3 | 2/8ГБ | 15ГБ | EU-CZ-1 | $0.08 |
| `fetcher-worker-b` | workers шард 1/3 | 2/8ГБ | 15ГБ | EU-CZ-1 | $0.08 |
| `fetcher-worker-c` | workers шард 2/3 | 2/8ГБ | 15ГБ | EU-CZ-1 | $0.08 |

Итого ~$0.24/ч compute + ~$0.09/ч эквивалент хранилища (3×15ГБ×$0.07/ГБ/мес) ≈ **$200/мес** при
непрерывной работе. ID подов/томов — `state/provision_result.json`.

Видео удаляются локально сразу после подтверждённого HF-аплоада (уже так в коде Fetcher) — 15ГБ на
под достаточно как буфер, не для хранения всего корпуса.

## Как запустить (после того, как есть секреты)

Нужно: `HF_TOKEN`, `keys.txt` (49 YouTube API-ключей), `cookies/*.txt` (для скачивания видео).

```python
# из automation/fetcher/
import deploy
HF_TOKEN = "hf_..."
for pod in ("fetcher-main", "fetcher-worker-b", "fetcher-worker-c"):
    deploy.deploy_secrets(pod, hf_token=HF_TOKEN,
                          youtube_keys_local="/path/to/keys.txt",   # только реально нужен на main (discover)
                          cookies_dir_local="/path/to/cookies")     # нужен на всех (download)
    print(pod, deploy.launch(pod, HF_TOKEN))
```

## Мониторинг

```bash
cd automation/fetcher
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env   # заполнить RUNPOD_API_KEY (тот же, что у runner), VK_TOKEN3, HF_TOKEN
python watchdog.py &        # Третий агент (Haiku) — почасовой анализ логов, брифинги при проблемах
python hourly_report.py --loop &   # часовой отчёт метрик (код, не LLM)
```

## Патч в самом Fetcher (2026-07-16)

`Fetcher/fetcher/dataset_collector/cli.py::command_discover` — при исчерпании квоты ВСЕГО пула
YouTube API-ключей (`QuotaExceededError`) процесс раньше падал необработанным исключением (см.
`FETCHER_DATASET_COLLECTOR_HANDOFF.md` §6). Теперь: сохраняет прогресс, спит чанками по ~1ч до
полуночи UTC (граница сброса квоты Google), продолжает В ТОМ ЖЕ процессе с чекпоинта. Внешний
`launch_role.sh` дополнительно оборачивает в `while true` на случай реальной (не квотной) ошибки.

## Файлы

- `config.py` — конфиг (свой .env, независимый от runner).
- `runpod_client.py` — минимальный REST/GraphQL клиент RunPod (CPU-поды, Network Volume). Специально
  НЕ переиспользует `automation/runner/runpod_api.py` — независимость систем.
- `deploy.py` — SSH-хелперы: деплой секретов, запуск/перезапуск ролей, чтение логов/summary.json.
- `launch_role.sh` — bash-скрипт, копируется на под и запускает discover+workers (main) или только
  workers (worker-b/c) через `nohup`.
- `watchdog.py` — Третий агент (Haiku), почасовой анализ логов + может чинить код и перезапускать.
- `hourly_report.py` — часовой отчёт метрик, чистый код без LLM.
- `state/provision_result.json` — ID подов/томов созданной инфраструктуры.
- `ssh/id_ed25519` — тот же ключ, что у ML-раннера (RunPod автоматически прописывает один и тот же
  паблик-ключ аккаунта во все новые поды) — скопирован для независимости модулей.
