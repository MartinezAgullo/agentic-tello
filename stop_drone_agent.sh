#!/usr/bin/env bash
#
# stop_drone_agent.sh — orderly shutdown of the Agentic Tello system.
#
# Order matters — the drone comes first:
#   1. SAFE LANDING  — if the dashboard is up, command a land via its REST API
#                      (the running process owns the drone link) and wait for
#                      the drone to touch down before anything else.
#   2. Stop the agent + web dashboard (graceful SIGTERM → its shutdown hook
#      lands again as a backstop and releases the UDP sockets).
#   3. Stop the NodeODM container (kept, not removed → fast restart next launch).
#   4. Stop Ollama ONLY if this project started it (pidfile from launch).
#
# Usage:
#   ./stop_drone_agent.sh            # land + stop dashboard + NodeODM (Ollama left running)
#   ./stop_drone_agent.sh --no-land  # skip the landing step (drone already down)
#   ./stop_drone_agent.sh --stop-ollama   # also stop Ollama, if launch started it
#
set -uo pipefail

WEB_PORT="${WEB_PORT:-8000}"
WEB_HOST="${WEB_HOST:-localhost}"
ODM_CONTAINER="${ODM_CONTAINER:-nodeodm}"
LAND_SETTLE_S="${LAND_SETTLE_S:-12}"        # max seconds to wait for touchdown

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
RUN_DIR="${SCRIPT_DIR}/.run"
BASE="http://${WEB_HOST}:${WEB_PORT}"

DO_LAND=1; STOP_OLLAMA=0    # Ollama is kept warm by default — it's slow to reload
for arg in "$@"; do
  case "$arg" in
    --no-land) DO_LAND=0 ;;
    --stop-ollama) STOP_OLLAMA=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg (try --help)"; exit 2 ;;
  esac
done

if [ -t 1 ]; then G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; B=$'\e[1m'; Z=$'\e[0m'
else G=""; Y=""; R=""; B=""; Z=""; fi
ok()   { printf "  ${G}✓${Z} %s\n" "$1"; }
warn() { printf "  ${Y}!${Z} %s\n" "$1"; }
bad()  { printf "  ${R}✗${Z} %s\n" "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
dash_up() { curl -fsS --max-time 3 "${BASE}/status" >/dev/null 2>&1; }
height()  { curl -fsS --max-time 3 "${BASE}/telemetry" 2>/dev/null \
              | grep -o '"height_cm"[: ]*[0-9.-]*' | grep -o '[0-9.-]*$'; }

printf "\n${B}── Stopping Agentic Tello ─────────────────────────${Z}\n"

# ── 1. SAFE LANDING (only meaningful while the owning process is alive) ──────
if [ "$DO_LAND" = 1 ]; then
  if dash_up; then
    h="$(height)"; [ -n "$h" ] && printf "  current height: %s cm\n" "$h"
    printf "  ${Y}…${Z} commanding LAND\n"
    if curl -fsS --max-time 8 -X POST "${BASE}/control/land" >/dev/null 2>&1; then
      # wait for touchdown — height settles near 0
      for _ in $(seq 1 "$LAND_SETTLE_S"); do
        h="$(height)"
        if [ -z "$h" ] || awk "BEGIN{exit !($h <= 5)}" 2>/dev/null; then break; fi
        sleep 1
      done
      ok "land commanded (height now ${h:-?} cm)"
    else
      bad "land request failed — STOP and verify the drone manually before continuing"
    fi
  else
    warn "dashboard not reachable at ${BASE} — cannot command a landing via REST"
    warn "if the drone is airborne, land it manually NOW (controller / power) before proceeding"
  fi
else
  warn "skipping landing (--no-land)"
fi

# ── 2. Stop the agent + dashboard (graceful → its hook lands as a backstop) ──
PIDS="$( { have lsof && lsof -ti "tcp:${WEB_PORT}" 2>/dev/null; pgrep -f 'agentic_tello' 2>/dev/null; } | sort -u)"
if [ -n "$PIDS" ]; then
  # shellcheck disable=SC2086
  kill $PIDS 2>/dev/null
  for _ in $(seq 1 10); do dash_up || break; sleep 1; done
  if dash_up; then
    # shellcheck disable=SC2086
    kill -9 $PIDS 2>/dev/null; warn "dashboard force-killed (graceful stop timed out)"
  else
    ok "agent + dashboard stopped"
  fi
else
  warn "no running dashboard/agent process found"
fi

# ── 3. Stop NodeODM (stop, don't remove → next launch reuses it) ─────────────
if have docker && [ -n "$(docker ps -q -f "name=^${ODM_CONTAINER}$" 2>/dev/null)" ]; then
  docker stop "$ODM_CONTAINER" >/dev/null && ok "NodeODM container stopped" \
    || bad "could not stop NodeODM container"
else
  warn "NodeODM container not running"
fi

# ── 4. Ollama: kept warm by default; stop only on request AND if WE started it ─
if [ "$STOP_OLLAMA" = 0 ]; then
  warn "leaving Ollama running (default — kept warm; pass --stop-ollama to stop it)"
elif [ -f "${RUN_DIR}/ollama.pid" ]; then
  OPID="$(cat "${RUN_DIR}/ollama.pid")"
  if [ -n "$OPID" ] && kill -0 "$OPID" 2>/dev/null; then
    kill "$OPID" 2>/dev/null && ok "Ollama (started by launch, pid ${OPID}) stopped"
  else
    warn "Ollama pidfile is stale — not running"
  fi
  rm -f "${RUN_DIR}/ollama.pid"
else
  warn "Ollama not started by launch — leaving it as-is"
fi

printf "\n${G}${B}Shutdown complete. Fly safe 👋${Z}\n\n"
