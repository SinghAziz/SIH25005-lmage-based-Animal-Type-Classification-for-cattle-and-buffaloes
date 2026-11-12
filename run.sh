#!/usr/bin/env bash
# One-command dev runner: checks/installs Python + Flutter, then serves the
# backend (uvicorn) and the Flutter web app (web-server).
#
#   ./run.sh
#   WEB_PORT=8083 BACKEND_PORT=8082 ./run.sh
#   API_URL=http://192.168.1.20:8000/predict ./run.sh   # open from a phone
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

WEB_PORT="${WEB_PORT:-8083}"
BACKEND_PORT="${BACKEND_PORT:-8082}"
API_URL="${API_URL:-http://localhost:${BACKEND_PORT}/predict}"
FLUTTER_HOME="${FLUTTER_HOME:-$HOME/.local/flutter}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) die "Native Windows is not supported; run this from WSL." ;;
esac

for c in python3 git curl xz unzip; do
  have "$c" && continue
  die "Missing '$c'. Install it and re-run:
  Ubuntu/WSL: sudo apt install python3 python3-venv git curl xz-utils unzip
  macOS:      brew install python git curl xz unzip"
done

# ---------- Flutter ----------
if have flutter; then
  FLUTTER="$(command -v flutter)"
elif [ -x "$FLUTTER_HOME/bin/flutter" ]; then
  FLUTTER="$FLUTTER_HOME/bin/flutter"
elif [ -x "$HOME/flutter/bin/flutter" ]; then
  FLUTTER="$HOME/flutter/bin/flutter"
else
  log "Flutter not found; installing the latest stable SDK to $FLUTTER_HOME"
  case "$(uname -s)" in
    Darwin) REL=releases_macos.json ;;
    *)      REL=releases_linux.json ;;
  esac
  REL_FILE="$(mktemp)"
  curl -fsSL "https://storage.googleapis.com/flutter_infra_release/releases/$REL" -o "$REL_FILE"
  ARCHIVE="$(python3 - "$REL_FILE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
stable = d["current_release"]["stable"]
print(next(r["archive"] for r in d["releases"] if r["hash"] == stable))
PY
)"
  rm -f "$REL_FILE"
  mkdir -p "$(dirname "$FLUTTER_HOME")"
  URL="https://storage.googleapis.com/flutter_infra_release/releases/$ARCHIVE"
  log "Downloading $ARCHIVE (~1.5 GB) — this takes a few minutes"
  case "$ARCHIVE" in
    *.zip)
      curl -fL --progress-bar "$URL" -o "$FLUTTER_HOME.zip"
      log "Extracting..."
      unzip -q "$FLUTTER_HOME.zip" -d "$(dirname "$FLUTTER_HOME")"
      rm -f "$FLUTTER_HOME.zip"
      ;;
    *)
      curl -fL --progress-bar "$URL" -o "$FLUTTER_HOME.tar.xz"
      log "Extracting..."
      tar --no-same-owner -xf "$FLUTTER_HOME.tar.xz" -C "$(dirname "$FLUTTER_HOME")"
      rm -f "$FLUTTER_HOME.tar.xz"
      ;;
  esac
  FLUTTER="$FLUTTER_HOME/bin/flutter"
  # Extracted as root the SDK can end up owned by another uid; flutter needs to
  # write into bin/cache, so make it ours.
  chown -R "$(id -u):$(id -g)" "$FLUTTER_HOME" 2>/dev/null || true
fi
export PATH="$(dirname "$FLUTTER"):$PATH"

# The SDK ships a .git dir; when extracted as root (containers) git refuses to
# work on it ("dubious ownership") and every flutter command fails.
FLUTTER_ROOT="$(dirname "$(dirname "$FLUTTER")")"
git config --global --add safe.directory "$FLUTTER_ROOT" 2>/dev/null || true
[ "$(id -u)" = "0" ] && chown -R 0:0 "$FLUTTER_ROOT" 2>/dev/null || true

FLUTTER_VERSION="$("$FLUTTER" --version 2>/dev/null | head -1)"
[ -n "$FLUTTER_VERSION" ] || die "Flutter at $FLUTTER is not working. Delete $FLUTTER_ROOT and re-run."
log "Using $FLUTTER_VERSION"
log "Precaching web artifacts (one-time, a few hundred MB)"
"$FLUTTER" precache --web >/dev/null 2>&1 || true

# ---------- Backend ----------
log "Preparing backend venv and dependencies"
if [ ! -x backend/venv/bin/python ]; then
  python3 -m venv backend/venv
fi
PIP=backend/venv/bin/pip
STAMP=backend/venv/.deps-stamp
if [ ! -f "$STAMP" ] || [ backend/requirements.txt -nt "$STAMP" ]; then
  if [ "$(uname -s)" = "Linux" ] && ! have nvidia-smi &&
    ! backend/venv/bin/python -c 'import torch' 2>/dev/null; then
    # CPU-only box: avoid pip pulling multiple GB of CUDA wheels.
    "$PIP" install -q --upgrade pip
    "$PIP" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  "$PIP" install -q -r backend/requirements.txt
  touch "$STAMP"
fi

port_busy() {
  python3 -c 'import socket,sys;sys.exit(0 if socket.socket().connect_ex(("127.0.0.1",int(sys.argv[1])))==0 else 1)' "$1"
}

kill_listeners() {
  local port="$1" pids="" pid desc
  if have lsof; then
    pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
  elif have fuser; then
    pids="$(fuser -v "${port}/tcp" 2>&1 | grep -oE '[0-9]+' | tr '\n' ' ' || true)"
  fi
  [ -z "${pids// /}" ] && return 0

  for pid in $pids; do
    desc="$(ps -p "$pid" -o args= 2>/dev/null | cut -c1-60)"
    log "Port $port is busy — killing PID $pid (${desc:-unknown})"
    kill "$pid" 2>/dev/null || log "  could not kill PID $pid (not yours?)"
  done
  sleep 1
  for pid in $pids; do
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 1
}

for p in "$BACKEND_PORT" "$WEB_PORT"; do
  if port_busy "$p"; then
    kill_listeners "$p"
    port_busy "$p" && die "Port $p is still in use. Free it or run with another port."
  fi
done

BACKEND_PID=""
cleanup() {
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

log "Starting backend on :$BACKEND_PORT"
(cd backend && exec "$ROOT/backend/venv/bin/python" -m uvicorn app:app \
  --host 0.0.0.0 --port "$BACKEND_PORT") &
BACKEND_PID=$!

for _ in $(seq 1 120); do
  if curl -sf -o /dev/null "http://127.0.0.1:$BACKEND_PORT/"; then
    break
  fi
  kill -0 "$BACKEND_PID" 2>/dev/null || die "Backend exited during startup."
  sleep 1
done
curl -sf -o /dev/null "http://127.0.0.1:$BACKEND_PORT/" \
  || die "Backend did not become ready within 120s."

# ---------- Frontend ----------
log "Preparing Flutter web dependencies"
(cd frontend && "$FLUTTER" pub get)

log "Starting app on :$WEB_PORT (API: $API_URL)"
printf '\n  Backend  http://localhost:%s\n  App      http://localhost:%s\n  Stop     Ctrl+C\n\n' \
  "$BACKEND_PORT" "$WEB_PORT"

cd frontend
"$FLUTTER" run -d web-server --web-hostname=0.0.0.0 --web-port "$WEB_PORT" \
  --dart-define=API_URL="$API_URL"
