#!/usr/bin/env bash
# Запуск всей автономной системы в tmux-сессии "trendflow".
# Требования: заполнен .env (VK_TOKEN, VK_TOKEN2, RUNPOD_API_KEY, CLAUDE_CDP_PORT=9222),
# и запущен Chrome с CDP + логином в claude.ai (см. ниже).
set -e
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true

SES=trendflow
tmux kill-session -t $SES 2>/dev/null || true
tmux new-session -d -s $SES -n daemon "source .venv/bin/activate; python limits_daemon.py"
tmux new-window -t $SES -n opus       "source .venv/bin/activate; python agent_runner.py --sas 999"
tmux new-window -t $SES -n assistant  "source .venv/bin/activate; python assistant.py"
echo "Запущено в tmux '$SES': daemon (лимиты), opus (компоненты), assistant (правки)."
echo "Смотреть: tmux attach -t $SES   (переключение окон: Ctrl+b затем n)"
echo "Остановить всё: tmux kill-session -t $SES"
