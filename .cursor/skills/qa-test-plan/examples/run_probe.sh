#!/usr/bin/env bash
# Shared helper for the examples QA plan.
#
# Usage (must be run inside `nix develop`, from the repo root):
#   bash .cursor/skills/qa-test-plan/examples/run_probe.sh <target> <wait_s> <name...>
#
#   <target>  a sim entry (e.g. examples/ball/main.py) or DB address for `elodin run`
#   <wait_s>  seconds to let the sim compile + serve before probing
#   <name...> component names (e.g. ball.world_pos) or message logs (msg:fsw.log)
#
# It launches the REAL run path (`elodin run <target>`) in its own session,
# waits, connects to the live DB on 127.0.0.1:2240 and prints per-component
# sample counts + value ranges (and message-log counts), then HARD-kills the
# whole process group and waits for port 2240 to free. Killing the group is
# required: `elodin run` launches an s10-managed python child with an instant
# restart policy, so killing only the child respawns it.
#
# Env vars set by the caller (e.g. DBNAME, ELODIN_NBODY_MAX_TICKS) are inherited.
set -u
TARGET="$1"; WAIT="$2"; shift 2
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${QA_RUN_LOG:-/tmp/qa-run-probe.log}"

setsid elodin run "$TARGET" >"$LOG" 2>&1 &
P=$!
sleep "$WAIT"
uv run python "$DIR/probe.py" 127.0.0.1:2240 "$@"
RC=$?
kill -9 -- -"$P" 2>/dev/null
for _ in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":2240 " || break; sleep 0.5; done
exit $RC
