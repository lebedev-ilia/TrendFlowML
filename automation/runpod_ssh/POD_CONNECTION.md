# RunPod — текущее подключение

> Заполняет Илья после деплоя пода. Claude (в любом Cowork-сеансе) читает этот файл
> перед тем как пытаться подключиться — здесь всегда актуальные IP/порт.

- **Статус**: stopped (2026-07-05, 5 компонентов; stop_pod.sh по API — EXITED verified)
- **Дата деплоя**: 05.08.2026 14:32 (UTC +10:00 Владивосток)
- **Host (Public IP)**: 213.173.108.211
- **Port**: 10757
- **User**: root
- **Публичный ключ**: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIApx5EZ5NLUAqSqdWjiMNg/PaZ34mx02aweR3DDWK7+w claude-cowork-trendflow
- **Приватный ключ**: `automation/runpod_ssh/id_ed25519` (относительно корня репозитория TrendFlowML, лежит рядом с этим файлом)
- **GPU**: RTX A4500 20GB
- **Network Volume**: Data center: EU-RO-1, Name: polite_magenta_coral, Volume ID: vuiq0iq3yf, Size: 100 GB
- **current-pod-id**: 557abm57yfq73m (меняется при миграции — не хардкодить)
- **RUNPOD_API_KEY**: см. `automation/runner/.env` (не храним ключи в git-доках)
- **S3_API_KEY**: см. `automation/runner/.env` (не храним ключи в git-доках)

## Команда подключения (собери из полей выше)

```bash
ssh -i automation/runpod_ssh/id_ed25519 -p <Port> root@<Host> \
  -o StrictHostKeyChecking=no -o BatchMode=yes "echo ok && nvidia-smi"
```

## История подов (для истории, необязательно)

| Дата | Host:Port | GPU | Причина деплоя | Когда остановлен |
|---|---|---|---|---|
| — | — | — | — | — |
