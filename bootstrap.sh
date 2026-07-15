#!/usr/bin/env bash
# =============================================================================
# TrendFlowML bootstrap — bring the whole project up from a fresh `git clone`.
#
# Phases (each can be skipped):
#   1. prereqs   — check python3 / docker / docker compose / git
#   2. env       — DP_MODELS_ROOT + .env files
#   3. venvs     — backend/.venv, DataProcessor/.venv (+ deps)
#   4. models    — download weights+artifacts from the unified HF repo
#   5. stack     — Postgres+Redis+MinIO, DB migrations, Fetcher/Backend/DP services
#   6. smoke     — health checks + dp_models self-test
#
# Usage:
#   ./bootstrap.sh                      # full bootstrap
#   HF_TOKEN=hf_xxx ./bootstrap.sh      # private model repo
#   ./bootstrap.sh --check              # only run prereqs + model dry-run (no changes)
#   ./bootstrap.sh --skip-deps          # create venvs but don't pip install (fast)
#   ./bootstrap.sh --models-groups "audio visual"   # skip the 409 semantic images
#   ./bootstrap.sh --with-triton        # also start Triton (ONNX inference)
#   ./bootstrap.sh --with-action-backbones  # + провижн VideoMAE/VideoMAEv2/Hiera/OSNet (action_recognition)
#   ./bootstrap.sh --no-start           # set up infra+migrations but don't launch app services
#
# Flags: --skip-venvs --skip-deps --skip-models --skip-stack --skip-smoke
#        --models-groups "<g..>"  --with-triton  --no-start  --check  -h
# Idempotent: safe to re-run; finished steps are detected and skipped.
# =============================================================================
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ----- defaults -----
DO_VENVS=1; DO_DEPS=1; DO_MODELS=1; DO_STACK=1; DO_SMOKE=1
WITH_TRITON=0; NO_START=0; CHECK_ONLY=0
WITH_ACTION_BB=0
MODELS_GROUPS=""
export DP_MODELS_ROOT="${DP_MODELS_ROOT:-$REPO_ROOT/DataProcessor/dp_models}"

log()  { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  warn\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m  FATAL\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-venvs)  DO_VENVS=0 ;;
    --skip-deps)   DO_DEPS=0 ;;
    --skip-models) DO_MODELS=0 ;;
    --skip-stack)  DO_STACK=0 ;;
    --skip-smoke)  DO_SMOKE=0 ;;
    --with-triton) WITH_TRITON=1 ;;
    --with-action-backbones) WITH_ACTION_BB=1 ;;  # провижн VideoMAE/VideoMAEv2/Hiera/OSNet для action_recognition
    --no-start)    NO_START=1 ;;
    --check)       CHECK_ONLY=1 ;;
    --models-groups) shift; MODELS_GROUPS="${1:-}" ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

# ----------------------------------------------------------------------------
# Phase 1: prerequisites
# ----------------------------------------------------------------------------
log "Phase 1/6: prerequisites"
have python3 || die "python3 not found"
PYV="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
ok "python3 $PYV"
DOCKER_OK=1
if have docker; then ok "docker $(docker --version 2>/dev/null | awk '{print $3}' | tr -d ,)"; else warn "docker not found (stack phase will be skipped)"; DOCKER_OK=0; fi
if docker compose version >/dev/null 2>&1; then ok "docker compose"; else warn "docker compose plugin not found"; DOCKER_OK=0; fi
have git && ok "git $(git --version | awk '{print $3}')" || warn "git not found"
[[ -n "${HF_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}" ]] && ok "HF token present" || warn "no HF_TOKEN (needed if model repo is private)"

if [[ "$CHECK_ONLY" == 1 ]]; then
  log "--check: model download dry-run"
  python3 "$REPO_ROOT/DataProcessor/scripts/download_models.py" --dry-run || true
  log "--check complete (no changes made)"
  exit 0
fi

# ----------------------------------------------------------------------------
# Phase 2: env
# ----------------------------------------------------------------------------
log "Phase 2/6: environment"
if [[ -f "$REPO_ROOT/DataProcessor/env.example" && ! -f "$REPO_ROOT/DataProcessor/.env" ]]; then
  cp "$REPO_ROOT/DataProcessor/env.example" "$REPO_ROOT/DataProcessor/.env"
  ok "created DataProcessor/.env from env.example"
fi
# Pin DP_MODELS_ROOT for child processes that read .env
grep -q '^DP_MODELS_ROOT=' "$REPO_ROOT/DataProcessor/.env" 2>/dev/null \
  && sed -i "s#^DP_MODELS_ROOT=.*#DP_MODELS_ROOT=$DP_MODELS_ROOT#" "$REPO_ROOT/DataProcessor/.env" 2>/dev/null \
  || echo "DP_MODELS_ROOT=$DP_MODELS_ROOT" >> "$REPO_ROOT/DataProcessor/.env" 2>/dev/null || true
ok "DP_MODELS_ROOT=$DP_MODELS_ROOT"

# ----------------------------------------------------------------------------
# Phase 3: venvs + deps
# ----------------------------------------------------------------------------
mkvenv() { # <dir> <venv_subdir> <req1> [req2...]
  local d="$1"; local venv_name="$2"; shift 2
  [[ -d "$d" ]] || return 0
  local venv="$d/$venv_name"
  if [[ ! -d "$venv" ]]; then
    log "creating venv: $venv"
    python3 -m venv "$venv" || { warn "venv create failed in $d"; return 0; }
  fi
  if [[ "$DO_DEPS" == 1 ]]; then
    # shellcheck disable=SC1091
    "$venv/bin/pip" install -q --upgrade pip >/dev/null 2>&1 || true
    for req in "$@"; do
      if [[ -f "$d/$req" ]]; then
        log "pip install -r $d/$req"
        "$venv/bin/pip" install -q -r "$d/$req" || warn "deps failed: $d/$req (continue)"
      fi
    done
  fi
}
if [[ "$DO_VENVS" == 1 ]]; then
  log "Phase 3/6: venvs + deps (deps=$DO_DEPS)"
  mkvenv "$REPO_ROOT/backend" ".venv" requirements.txt
  # backend E2E fixes: email-validator + alembic
  if [[ "$DO_DEPS" == 1 && -d "$REPO_ROOT/backend/.venv" ]]; then
    "$REPO_ROOT/backend/.venv/bin/pip" install -q "pydantic[email]" alembic >/dev/null 2>&1 || true
  fi
  # E2E scripts expect .data_venv / .fetcher_venv (not generic .venv)
  mkvenv "$REPO_ROOT/DataProcessor" ".data_venv" requirements-api.txt
  mkvenv "$REPO_ROOT/Fetcher" ".fetcher_venv" requirements.txt
else
  log "Phase 3/6: venvs skipped"
fi

# ----------------------------------------------------------------------------
# Phase 4: models
# ----------------------------------------------------------------------------
if [[ "$DO_MODELS" == 1 ]]; then
  log "Phase 4/6: downloading models -> $DP_MODELS_ROOT"
  GROUP_ARGS=()
  [[ -n "$MODELS_GROUPS" ]] && GROUP_ARGS=(--groups $MODELS_GROUPS)
  python3 "$REPO_ROOT/DataProcessor/scripts/download_models.py" "${GROUP_ARGS[@]}" \
    && ok "models downloaded" || warn "model download reported failures (see above)"
  python3 "$REPO_ROOT/DataProcessor/scripts/vendor_emonet.py" \
    && ok "EmoNet sources vendored" || warn "vendor_emonet failed (emotion_face may be disabled in E2E)"
  # опц.: alt-backbone action_recognition (VideoMAE/VideoMAEv2/Hiera/OSNet) — тяжёлые, только по флагу
  if [[ "$WITH_ACTION_BB" == 1 ]]; then
    log "Phase 4/6: провижн action-backbones (VideoMAE/VideoMAEv2/Hiera/OSNet)"
    VP_PY="$REPO_ROOT/DataProcessor/VisualProcessor/.vp_venv/bin/python"
    [[ -x "$VP_PY" ]] || VP_PY="python3"
    "$VP_PY" -m pip install -q "transformers>=4.44" torchreid >/dev/null 2>&1 \
      && ok "transformers+torchreid установлены" || warn "pip transformers/torchreid failed"
    "$VP_PY" "$REPO_ROOT/DataProcessor/scripts/provision_base_models.py" \
      --only action_videomae action_videomaev2 action_hiera osnet_reid \
      && ok "action-backbones provisioned" || warn "action-backbone provision reported issues (см. выше)"
  fi
else
  log "Phase 4/6: models skipped"
fi

# ----------------------------------------------------------------------------
# Phase 5: stack (infra + migrations + services)
# ----------------------------------------------------------------------------
if [[ "$DO_STACK" == 1 && "$DOCKER_OK" == 1 ]]; then
  log "Phase 5/6: stack"
  if [[ "$NO_START" == 1 ]]; then
    log "infra + migrations only (--no-start)"
    bash "$REPO_ROOT/backend/scripts/setup_e2e_infra.sh" || warn "setup_e2e_infra reported issues"
  else
    log "starting full stack (Postgres/Redis/MinIO + migrations + Fetcher/Backend/DP)"
    bash "$REPO_ROOT/backend/scripts/start_e2e_stack.sh" --with-infra || warn "start_e2e_stack reported issues"
  fi
  if [[ "$WITH_TRITON" == 1 && -f "$REPO_ROOT/backend/scripts/e2e_triton_docker.sh" ]]; then
    log "starting Triton (ONNX inference)"
    bash "$REPO_ROOT/backend/scripts/e2e_triton_docker.sh" up || warn "triton start reported issues"
  fi
else
  log "Phase 5/6: stack skipped (docker missing or --skip-stack)"
fi

# ----------------------------------------------------------------------------
# Phase 6: smoke
# ----------------------------------------------------------------------------
if [[ "$DO_SMOKE" == 1 ]]; then
  log "Phase 6/6: smoke checks"
  check_http() { local u="$1" n="$2"; local code; code="$(curl -s -o /dev/null -w '%{http_code}' "$u" 2>/dev/null || echo 000)"; [[ "$code" =~ ^(200|301|302|404)$ ]] && ok "$n reachable ($u -> $code)" || warn "$n NOT reachable ($u -> $code)"; }
  if have curl; then
    check_http "http://127.0.0.1:8000/docs" "Fetcher API"
    check_http "http://127.0.0.1:8001/docs" "Backend API"
    check_http "http://127.0.0.1:8002/health" "DataProcessor API"
  fi
  if [[ -x "$REPO_ROOT/DataProcessor/.data_venv/bin/python" && -f "$REPO_ROOT/DataProcessor/scripts/dp_models_selftest.py" ]]; then
    log "dp_models self-test"
    (cd "$REPO_ROOT/DataProcessor" && DP_MODELS_ROOT="$DP_MODELS_ROOT" .data_venv/bin/python scripts/dp_models_selftest.py) \
      && ok "dp_models self-test passed" || warn "dp_models self-test reported issues"
  fi
fi

log "Bootstrap finished."
cat <<EOF

Next steps:
  - Backend API   : http://127.0.0.1:8001/docs
  - Fetcher API   : http://127.0.0.1:8000/docs
  - DataProcessor : http://127.0.0.1:8002
  - Stop stack    : ./backend/scripts/stop_e2e_stack.sh --with-infra
  - Re-download a group: python DataProcessor/scripts/download_models.py --groups audio
EOF
