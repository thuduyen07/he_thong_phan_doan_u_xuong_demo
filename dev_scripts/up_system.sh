#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
KEY_PATH="${SYSTEM_DIR}/configs/ggserviceaccount.json"
SYSTEM_URL="http://127.0.0.1:4173"
DVC_REMOTE_NAME="gdrive_remote"
DVC_TARGETS=(
  "resources/pretrained/segformer_b0_ade_512_512/pytorch_model.bin.dvc"
  "resources/pretrained/segformer_b0_ade_512_512/model.safetensors.dvc"
  "resources/pretrained/segformer_b0_ade_512_512/tf_model.h5.dvc"
)

log() {
  printf '[up_system] %s\n' "$*"
}

fail() {
  printf '[up_system] Loi: %s\n' "$*" >&2
  exit 1
}

append_python_user_bin_to_path() {
  local user_bin
  user_bin="$(python3 - <<'PY'
import site
print(site.USER_BASE + "/bin")
PY
)"
  case ":${PATH}:" in
    *":${user_bin}:"*) ;;
    *) export PATH="${user_bin}:${PATH}" ;;
  esac
}

ensure_python3() {
  command -v python3 >/dev/null 2>&1 || fail "Khong tim thay python3. Hay cai python3 truoc khi chay script nay."
}

ensure_key_file() {
  [[ -s "${KEY_PATH}" ]] || fail "Khong tim thay file key hop le tai ${KEY_PATH}. Hay chep file service account vao dung duong dan nay truoc."

  python3 - "${KEY_PATH}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    with path.open("r", encoding="utf-8") as fh:
        json.load(fh)
except Exception as exc:
    raise SystemExit(f"File key khong hop le: {exc}")
PY
}

ensure_dvc() {
  append_python_user_bin_to_path
  if command -v dvc >/dev/null 2>&1; then
    log "Da tim thay dvc: $(dvc version | head -n 1)"
    return
  fi

  log "Chua tim thay dvc. Dang thu cai dat bang pip user..."
  python3 -m pip --version >/dev/null 2>&1 || python3 -m ensurepip --user >/dev/null 2>&1 || fail "Khong the khoi tao pip cho python3."
  python3 -m pip install --user "dvc[gdrive]"
  append_python_user_bin_to_path

  command -v dvc >/dev/null 2>&1 || fail "Da cai dvc nhung shell hien tai van chua tim thay lenh dvc."
  log "Da cai dat dvc thanh cong."
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

configure_dvc_remote() {
  (
    cd "${SYSTEM_DIR}"
    dvc remote modify --local "${DVC_REMOTE_NAME}" gdrive_service_account_json_file_path "${KEY_PATH}"
    dvc remote modify --local "${DVC_REMOTE_NAME}" gdrive_use_service_account true
  )
  log "Da cap nhat .dvc/config.local voi key hien tai."
}

pull_required_artifacts() {
  log "Dang dvc pull cac artifact can thiet..."
  (
    cd "${SYSTEM_DIR}"
    dvc pull "${DVC_TARGETS[@]}"
  )
  log "Da tai xong cac file can thiet cho he thong."
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
  ensure_key_file
  ensure_dvc
  detect_compose_command
  ensure_docker_daemon
  configure_dvc_remote
  pull_required_artifacts
  start_compose_stack
  wait_for_healthcheck
}

main "$@"
