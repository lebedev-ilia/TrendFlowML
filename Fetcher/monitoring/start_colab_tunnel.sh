#!/usr/bin/env bash
# Colab: wait for Grafana on :3001, set root URL, start cloudflared quick tunnel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Native .deb Grafana uses 3000; Docker compose maps 3001:3000
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
WAIT_SEC="${WAIT_SEC:-120}"

echo "==> Checking Grafana on 127.0.0.1:${GRAFANA_PORT} ..."
deadline=$((SECONDS + WAIT_SEC))
while (( SECONDS < deadline )); do
  if curl -sf -o /dev/null "http://127.0.0.1:${GRAFANA_PORT}/login"; then
    echo "    Grafana OK"
    break
  fi
  sleep 2
done

if ! curl -sf -o /dev/null "http://127.0.0.1:${GRAFANA_PORT}/login"; then
  echo "FATAL: nothing on http://127.0.0.1:${GRAFANA_PORT}/login" >&2
  echo "Start monitoring first:" >&2
  echo "  cd ${ROOT} && docker compose -f docker-compose.yml -f docker-compose.colab.yml up -d" >&2
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "Installing cloudflared ..."
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb || apt-get install -fy
fi

echo "==> Starting cloudflared (Ctrl+C to stop) ..."
echo "    After URL appears, in another terminal run:"
echo "    export GRAFANA_ROOT_URL=\"https://YOUR-SUBDOMAIN.trycloudflare.com/\""
echo "    cd ${ROOT} && docker compose -f docker-compose.yml -f docker-compose.colab.yml -f docker-compose.colab-tunnel.yml up -d --force-recreate grafana"
echo

# http2 is more stable than quic on some Colab VMs
exec cloudflared tunnel --protocol http2 --url "http://127.0.0.1:${GRAFANA_PORT}"
