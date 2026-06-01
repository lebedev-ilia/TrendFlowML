# Kubernetes слой для Dataset Collector

Kubernetes не обязателен для первого Colab Free теста. Этот слой нужен как следующий шаг, когда появится нормальное хранилище `ReadWriteMany` и кластер вроде GKE/k3s.

## Сборка образа

```bash
cd Fetcher
docker build -f docker/dataset-collector.Dockerfile -t trendflow-fetcher/dataset-collector:20k .
```

## Базовый деплой

```bash
kubectl apply -f k8s/dataset-collector-20k.yaml
```

Перед применением замени Secret values:

- `HF_TOKEN`
- `FETCHER_YOUTUBE_DATA_API_KEYS`

И проверь PVC: для большого запуска нужен storage class с `ReadWriteMany`, потому что discover/workers/snapshots должны видеть один state.

## Как масштабировать

- `dataset-collector-workers` держи в `replicas: 1`, пока не будет per-queue lease на уровне отдельных queue items.
- Discover лучше запускать отдельными Jobs по категориям: меняй `--category Sport` на нужную категорию.
- Snapshot CronJob нужно продублировать для индексов `2` и `3`, когда наступят `14d` и `21d`.

## Что смотреть

- `/app/dataset_runs/dataset-20k-colab/state/inventory/summary.json`
- `/app/dataset_runs/dataset-20k-colab/state/worker_leases.json`
- `/app/dataset_runs/dataset-20k-colab/state/hf_commit_log.jsonl`
- `/app/dataset_runs/dataset-20k-colab/logs/workers/*.log`
