#!/usr/bin/env bash
# Colab: native Prometheus + Grafana (no Docker). Port 3000 for Grafana, 9090 for Prometheus.
set -euo pipefail

FETCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITORING="${FETCHER_ROOT}/monitoring"
PROM_VERSION="${PROM_VERSION:-2.51.2}"
GRAFANA_DEB_VERSION="${GRAFANA_DEB_VERSION:-10.4.2}"

install_deps() {
  apt-get update -qq
  apt-get install -y -qq adduser libfontconfig1 musl wget
}

install_prometheus() {
  if [[ -x "/content/prometheus-${PROM_VERSION}.linux-amd64/prometheus" ]]; then
    return
  fi
  cd /content
  wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz"
  tar -xzf "prometheus-${PROM_VERSION}.linux-amd64.tar.gz"
}

install_grafana() {
  if command -v grafana-server >/dev/null 2>&1; then
    return
  fi
  cd /content
  wget -q "https://dl.grafana.com/oss/release/grafana_${GRAFANA_DEB_VERSION}_amd64.deb"
  dpkg -i "grafana_${GRAFANA_DEB_VERSION}_amd64.deb" || apt-get install -fy
}

install_cloudflared() {
  if command -v cloudflared >/dev/null 2>&1; then
    return
  fi
  cd /content
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i cloudflared-linux-amd64.deb || apt-get install -fy
}

write_prometheus_config() {
  cp "${MONITORING}/prometheus/prometheus.colab.yml" /content/prometheus.yml
}

provision_grafana() {
  mkdir -p /etc/grafana/provisioning/datasources
  mkdir -p /etc/grafana/provisioning/dashboards
  mkdir -p /var/lib/grafana/dashboards
  cp "${MONITORING}/grafana/provisioning/datasources/prometheus-native.yml" \
    /etc/grafana/provisioning/datasources/prometheus.yml
  cp "${MONITORING}/grafana/provisioning/dashboards/default.yml" \
    /etc/grafana/provisioning/dashboards/default.yml
  cp "${MONITORING}/grafana/dashboards/"*.json \
    /var/lib/grafana/dashboards/
  chown -R grafana:grafana /var/lib/grafana/dashboards 2>/dev/null || true
  chown -R grafana:grafana /etc/grafana/provisioning 2>/dev/null || true
}

stop_monitoring() {
  pkill -f 'prometheus.*--config.file=/content/prometheus.yml' 2>/dev/null || true
  pkill -f grafana-server 2>/dev/null || true
  sleep 1
}

_grafana_up() {
  curl -sf http://127.0.0.1:3000/login >/dev/null 2>&1
}

_prometheus_up() {
  curl -sf http://127.0.0.1:9090/-/ready >/dev/null 2>&1
}

start_prometheus() {
  if _prometheus_up; then
    echo "Prometheus already listening on :9090"
    return
  fi
  mkdir -p /content/prometheus-data
  setsid "/content/prometheus-${PROM_VERSION}.linux-amd64/prometheus" \
    --config.file=/content/prometheus.yml \
    --storage.tsdb.path=/content/prometheus-data \
    --web.listen-address=0.0.0.0:9090 \
    >> /content/prometheus.log 2>&1 &
  disown || true
  echo "Prometheus log: /content/prometheus.log"
}

start_grafana() {
  if _grafana_up; then
    echo "Grafana already listening on :3000"
    return
  fi
  # For trycloudflare: export GRAFANA_ROOT_URL=https://xxx.trycloudflare.com/ before start
  setsid /usr/sbin/grafana-server \
    --homepath=/usr/share/grafana \
    --config=/etc/grafana/grafana.ini \
    web \
    >> /content/grafana.log 2>&1 &
  disown || true
  echo "Grafana log: /content/grafana.log"
}

wait_ready() {
  local i
  for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:9090/-/ready >/dev/null 2>&1 \
      && curl -sf http://127.0.0.1:3000/login >/dev/null 2>&1; then
      echo "Prometheus :9090 and Grafana :3000 are up"
      return 0
    fi
    sleep 2
  done
  echo "WARN: timeout waiting for Prometheus/Grafana — check logs" >&2
  return 1
}

usage() {
  cat <<EOF
Usage: $0 [install|start|stop|restart|status]

  install  — apt packages + prometheus tarball + grafana deb + cloudflared
  start    — write config, provision dashboard, nohup prometheus + grafana
  stop     — kill prometheus/grafana processes
  restart  — stop + start
  status   — curl readiness checks

Grafana: http://127.0.0.1:3000  (admin / admin)
Prometheus: http://127.0.0.1:9090

Cloudflare tunnel (native Grafana uses port 3000):
  cloudflared tunnel --protocol http2 --url http://127.0.0.1:3000

After tunnel URL is known:
  export GRAFANA_ROOT_URL="https://YOUR.trycloudflare.com/"
  $0 restart

Collector metrics: discover --metrics-port 9095, workers --metrics-port 9096
EOF
}

cmd="${1:-start}"
case "$cmd" in
  install)
    install_deps
    install_prometheus
    install_grafana
    install_cloudflared
    ;;
  start)
    write_prometheus_config
    provision_grafana
    if ! _prometheus_up; then
      start_prometheus
    fi
    # Restart Grafana so provisioning (datasource uid=prometheus) is always loaded.
    if _grafana_up; then
      pkill -f grafana-server 2>/dev/null || true
      sleep 2
    fi
    start_grafana
    wait_ready || true
    echo ""
    echo "IMPORTANT: do not re-run the Colab start cell — Jupyter sends SIGINT and kills Grafana."
    echo "Before cloudflared: curl -sf http://127.0.0.1:3000/login && echo OK"
    ;;
  stop) stop_monitoring ;;
  restart)
    "$0" stop
    sleep 2
    "$0" start
    ;;
  status)
    curl -sf http://127.0.0.1:9090/-/ready && echo "prometheus OK" || echo "prometheus DOWN"
    curl -sf http://127.0.0.1:3000/login && echo "grafana OK" || echo "grafana DOWN"
    ;;
  *) usage; exit 1 ;;
esac
