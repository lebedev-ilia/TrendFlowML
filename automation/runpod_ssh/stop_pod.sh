#!/usr/bin/env bash
# Остановка RunPod-пода(ов). ВАЖНО: при миграции пода его ID МЕНЯЕТСЯ, поэтому НЕ хардкодим ID —
# запрашиваем список подов из API и останавливаем ВСЕ RUNNING (чтобы деньги не капали).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONN="$ROOT/automation/runpod_ssh/POD_CONNECTION.md"
KEY="${RUNPOD_API_KEY:-$(grep -oE 'rpa_[A-Za-z0-9]+' "$CONN" 2>/dev/null | head -1)}"
[ -n "$KEY" ] || { echo "нет RUNPOD_API_KEY"; exit 1; }

echo "[stop_pod] ищу RUNNING поды..."
RUNNING=$(curl -s -m 25 -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods" 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); pods=d if isinstance(d,list) else d.get('pods',d.get('data',[]))
for p in (pods or []):
    if str(p.get('desiredStatus')).upper()=='RUNNING': print(p.get('id'), p.get('name'))
")
[ -n "$RUNNING" ] || { echo "[stop_pod] RUNNING подов нет — всё уже остановлено."; exit 0; }
echo "$RUNNING" | while read -r id name; do
  [ -n "$id" ] || continue
  echo "[stop_pod] останавливаю $id ($name)"
  curl -s -m 25 -X POST -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$id/stop" >/dev/null 2>&1
done
sleep 8
echo "[stop_pod] проверка статусов:"
curl -s -m 25 -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods" 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); pods=d if isinstance(d,list) else d.get('pods',d.get('data',[]))
for p in (pods or []): print(' ', p.get('id'), p.get('name'), '->', p.get('desiredStatus'))
bad=[p for p in (pods or []) if str(p.get('desiredStatus')).upper()=='RUNNING']
print('ИТОГ:', 'ВСЕ ОСТАНОВЛЕНЫ ✅' if not bad else 'ЕСТЬ RUNNING ❌ '+str([p.get('id') for p in bad]))
"
