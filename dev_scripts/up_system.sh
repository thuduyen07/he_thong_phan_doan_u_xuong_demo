#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SYSTEM_URL="http://127.0.0.1:4173"

log() {
  printf '[up_system] %s\n' "$*"
}

fail() {
  printf '[up_system] Loi: %s\n' "$*" >&2
  exit 1
}

ensure_python3() {
  command -v python3 >/dev/null 2>&1 || fail "Khong tim thay python3. Hay cai python3 truoc khi chay script nay."
}

detect_compose_command() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
    return
  fi

  fail "Khong tim thay Docker Compose. Hay cai Docker Desktop hoac docker compose truoc."
}

ensure_docker_daemon() {
  docker info >/dev/null 2>&1 || fail "Docker daemon chua chay. Hay mo Docker Desktop roi chay lai script."
}

start_compose_stack() {
  log "Dang build va khoi dong he thong bang Docker Compose..."
  (
    cd "${SYSTEM_DIR}"
    "${COMPOSE_CMD[@]}" up --build -d
  )
}

wait_for_healthcheck() {
  local attempt
  local max_attempts=90

  log "Dang doi backend san sang tai ${SYSTEM_URL} ..."
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if python3 - "${SYSTEM_URL}" <<'PY'
import sys
import urllib.request
import urllib.error

url = sys.argv[1] + "/health"
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        raise SystemExit(0 if 200 <= response.status < 300 else 1)
except (urllib.error.URLError, TimeoutError, OSError):
    raise SystemExit(1)
PY
    then
      log "He thong da san sang. Mo trinh duyet tai ${SYSTEM_URL}"
      return
    fi
    sleep 2
  done

  fail "He thong chua san sang sau thoi gian cho. Hay kiem tra log bang lenh: cd ${SYSTEM_DIR} && ${COMPOSE_CMD[*]} logs -f"
}

main() {
  ensure_python3
  detect_compose_command
  ensure_docker_daemon
  start_compose_stack
  wait_for_healthcheck
}

main "$@"
