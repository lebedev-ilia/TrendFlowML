# GitHub Workflow for Two Devices

Этот документ фиксирует рабочие команды для параллельной работы:

- `fetcher-dev` - разработка Fetcher на ноутбуке
- `system-testing` - тестирование всей системы на основном ПК

## 1) Первичная настройка (один раз на каждой машине)

```bash
cd /path/to/TrendFlowML
git remote set-url origin https://github.com/lebedev-ilia/TrendFlowML.git
gh auth login
git fetch origin
```

## 2) Основные ветки

```bash
git checkout main
git pull origin main

git checkout fetcher-dev
git pull --rebase origin fetcher-dev

git checkout system-testing
git pull --rebase origin system-testing
```

## 3) Ноутбук: работа только с Fetcher

```bash
git checkout fetcher-dev
git pull --rebase origin fetcher-dev

# изменения
git add -A
git commit -m "Fetcher: short description"
git push origin fetcher-dev
```

## 4) Основной ПК: системное тестирование

```bash
git checkout system-testing
git pull --rebase origin system-testing

# изменения
git add -A
git commit -m "Testing: short description"
git push origin system-testing
```

## 5) Регулярная синхронизация веток с main

```bash
git checkout main
git pull origin main

git checkout fetcher-dev
git pull --rebase origin fetcher-dev
git rebase main
git push --force-with-lease origin fetcher-dev

git checkout system-testing
git pull --rebase origin system-testing
git rebase main
git push --force-with-lease origin system-testing
```

## 6) Подготовка PR в main

Для ветки `fetcher-dev`:

```bash
git checkout fetcher-dev
git pull --rebase origin fetcher-dev
git rebase main
git push --force-with-lease origin fetcher-dev
gh pr create --base main --head fetcher-dev --title "Fetcher updates" --body "See commits"
```

Для ветки `system-testing`:

```bash
git checkout system-testing
git pull --rebase origin system-testing
git rebase main
git push --force-with-lease origin system-testing
gh pr create --base main --head system-testing --title "System testing updates" --body "See commits"
```

## 7) Быстрая диагностика, если что-то пошло не так

```bash
git status
git branch -vv
git log --oneline -n 10
git fetch origin
```

Если есть локальные незакоммиченные изменения:

```bash
git stash push -u -m "wip"
# переключения / pull / rebase
git stash pop
```

## 8) Связанные команды по артефактам (HF)

```bash
export HF_TOKEN="YOUR_TOKEN"
./DataProcessor/scripts/hf_download_all.sh
./DataProcessor/scripts/hf_upload_all.sh
```
---

## Навигация

[Vault](MAIN_INDEX.md)
