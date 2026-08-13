#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '[down_system] %s\n' "$*"
}

fail() {
  printf '[down_system] Loi: %s\n' "$*" >&2
  exit 1
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

  fail "Khong tim thay Docker Compose de tat he thong."
}

ensure_docker_daemon() {
  docker info >/dev/null 2>&1 || fail "Docker daemon chua chay. Hay mo Docker Desktop neu muon tat stack bang script nay."
}

main() {
  detect_compose_command
  ensure_docker_daemon

  log "Dang tat he thong..."
  (
    cd "${SYSTEM_DIR}"
    "${COMPOSE_CMD[@]}" down
  )
  log "He thong da duoc tat."
}

main "$@"
