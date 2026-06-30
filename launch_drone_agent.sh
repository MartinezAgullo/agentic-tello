#!/usr/bin/env bash
#
# launch_drone_agent.sh — single launcher for the Agentic Tello system.
#
# Does the boring plumbing so you don't have to remember it:
#   1. Pre-flight checks  (uv, Ollama + VLM model, Tello WiFi, Docker)
#   2. Brings up the NodeODM photogrammetry container  (idempotent — no name clash)
#   3. Launches the agent + web dashboard              (uv run python -m agentic_tello)
#
# Cross-platform (Linux / NVIDIA DGX Spark  +  macOS M3), no platform-specific
# assumptions beyond the tools each check probes for.
#
# Usage:
#   ./launch_drone_agent.sh            # checks + NodeODM + dashboard
#   ./launch_drone_agent.sh --no-odm   # skip NodeODM (no 3D reconstruction)
#   ./launch_drone_agent.sh --check    # run pre-flight checks only, then exit
#   ./launch_drone_agent.sh --yes      # don't stop on warnings (CI / unattended)
#
set -uo pipefail

# ── tunables (match config.py / docker invocation) ──────────────────────────
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
VLM_MODEL="${VLM_MODEL:-gemma3:12b}"
TELLO_IP="${TELLO_IP:-192.168.10.1}"        # Tello/RMTT soft-AP gateway
ODM_CONTAINER="${ODM_CONTAINER:-nodeodm}"
ODM_IMAGE="${ODM_IMAGE:-opendronemap/nodeodm:latest}"
ODM_PORT="${ODM_PORT:-3000}"
WEB_PORT="${WEB_PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
RUN_DIR="${SCRIPT_DIR}/.run"        # pidfiles / logs for services we start
mkdir -p "$RUN_DIR"

# ── flags ───────────────────────────────────────────────────────────────────
RUN_ODM=1; CHECK_ONLY=0; ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --no-odm) RUN_ODM=0 ;;
    --check)  CHECK_ONLY=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg (try --help)"; exit 2 ;;
  esac
done

# ── pretty output ───────────────────────────────────────────────────────────
if [ -t 1 ]; then G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; B=$'\e[1m'; Z=$'\e[0m'
else G=""; Y=""; R=""; B=""; Z=""; fi
FAIL=0; WARN=0
ok()   { printf "  ${G}✓${Z} %s\n" "$1"; }
warn() { printf "  ${Y}!${Z} %s\n" "$1"; WARN=$((WARN+1)); }
bad()  { printf "  ${R}✗${Z} %s\n" "$1"; FAIL=$((FAIL+1)); }
have() { command -v "$1" >/dev/null 2>&1; }

printf "\n${B}── Pre-flight checks ──────────────────────────────${Z}\n"

# 1. uv (the only sanctioned runner — see CLAUDE.md)
if have uv; then ok "uv present"; else bad "uv not found — install from https://docs.astral.sh/uv/"; fi

# 2. Ollama up (auto-start if down) + VLM model pulled
ollama_up() { curl -fsS --max-time 3 "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; }
if ! ollama_up && have ollama; then
  printf "  ${Y}…${Z} Ollama down — starting 'ollama serve'\n"
  nohup ollama serve >"${RUN_DIR}/ollama.log" 2>&1 &
  echo $! > "${RUN_DIR}/ollama.pid"          # marks it as *ours* → stop script may kill it
  for _ in $(seq 1 20); do ollama_up && break; sleep 0.5; done
fi
if ollama_up; then
  ok "Ollama reachable (${OLLAMA_HOST})"
  if curl -fsS --max-time 3 "${OLLAMA_HOST}/api/tags" 2>/dev/null | grep -q "\"${VLM_MODEL}\""; then
    ok "VLM model '${VLM_MODEL}' pulled"
  else
    warn "VLM model '${VLM_MODEL}' not pulled — run: ollama pull ${VLM_MODEL}"
  fi
elif have ollama; then
  bad "Ollama failed to start — see ${RUN_DIR}/ollama.log"
else
  bad "Ollama not installed and not reachable at ${OLLAMA_HOST} — install Ollama or set OLLAMA_HOST"
fi

# 3. Tello WiFi — ping the soft-AP gateway (most reliable signal across OSes)
case "$(uname -s)" in
  Darwin) PING="ping -c1 -t1" ;;
  *)      PING="ping -c1 -W1" ;;
esac
if $PING "$TELLO_IP" >/dev/null 2>&1; then
  ok "Tello reachable at ${TELLO_IP} (WiFi connected)"
else
  warn "No Tello at ${TELLO_IP} — connect to its WiFi (SSID TELLO-* / RMTT-*). Dashboard still runs without a drone."
fi

# 4. Docker (only required when NodeODM is requested)
if [ "$RUN_ODM" = 1 ]; then
  if have docker && docker info >/dev/null 2>&1; then
    ok "Docker daemon running"
  else
    warn "Docker unavailable — 3D reconstruction (NodeODM) will be skipped"
    RUN_ODM=0
  fi
fi

# ── stop-on-trouble gate ────────────────────────────────────────────────────
if [ "$FAIL" -gt 0 ]; then
  printf "\n${R}${B}%d check(s) failed.${Z} Fix the above and re-run.\n\n" "$FAIL"
  exit 1
fi
if [ "$CHECK_ONLY" = 1 ]; then
  printf "\n${G}Checks done${Z} (%d warning(s)).\n\n" "$WARN"
  exit 0
fi
if [ "$WARN" -gt 0 ] && [ "$ASSUME_YES" = 0 ]; then
  printf "\n${Y}%d warning(s).${Z} Continue anyway? [y/N] " "$WARN"
  read -r reply </dev/tty
  case "$reply" in y|Y|yes) ;; *) echo "Aborted."; exit 1 ;; esac
fi

# ── NodeODM: idempotent bring-up (this is the fix for the name clash) ────────
if [ "$RUN_ODM" = 1 ]; then
  printf "\n${B}── NodeODM (photogrammetry :%s) ───────────────────${Z}\n" "$ODM_PORT"
  if [ -n "$(docker ps -q -f "name=^${ODM_CONTAINER}$")" ]; then
    ok "container '${ODM_CONTAINER}' already running"
  elif [ -n "$(docker ps -aq -f "name=^${ODM_CONTAINER}$")" ]; then
    docker start "$ODM_CONTAINER" >/dev/null && ok "started existing container '${ODM_CONTAINER}'" \
      || bad "could not start existing container '${ODM_CONTAINER}'"
  else
    docker run -d --name "$ODM_CONTAINER" -p "${ODM_PORT}:3000" "$ODM_IMAGE" >/dev/null \
      && ok "created container '${ODM_CONTAINER}'" \
      || bad "could not create container '${ODM_CONTAINER}'"
  fi
fi

# ── launch the system ───────────────────────────────────────────────────────
printf "\n${B}── Launching agent + dashboard ────────────────────${Z}\n"
printf "  dashboard → ${B}http://localhost:%s${Z}\n\n" "$WEB_PORT"
# --no-sync: flying means Tello WiFi (no internet), so never let uv reach out to
# pypi to re-sync/rebuild the project here. Run `uv sync` once, on a normal
# connection, after changing dependencies. See README.
exec uv run --no-sync python -m agentic_tello
