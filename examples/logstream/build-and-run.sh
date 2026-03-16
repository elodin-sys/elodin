#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DB_ADDRESS="${DB_ADDRESS:-127.0.0.1}"
DB_PORT="${DB_PORT:-2240}"

LOG_CLIENT_SRC="$REPO_ROOT/libs/db/examples/log-client.cpp"
LOG_CLIENT_BIN="/tmp/build-cache/elodin-log-client"

echo "logstream: compiling log-client.cpp..."
mkdir -p "$(dirname "$LOG_CLIENT_BIN")"
CXX="${CXX:-$(which clang++ 2>/dev/null || which g++)}"
$CXX --std=c++23 "$LOG_CLIENT_SRC" -o "$LOG_CLIENT_BIN"
echo "logstream: compiled -> $LOG_CLIENT_BIN"

echo "logstream: waiting for Elodin DB at $DB_ADDRESS:$DB_PORT..."
RETRIES=0
MAX_RETRIES=30
while ! nc -z "$DB_ADDRESS" "$DB_PORT" 2>/dev/null; do
    RETRIES=$((RETRIES + 1))
    if [ "$RETRIES" -ge "$MAX_RETRIES" ]; then
        echo "logstream: ERROR - DB not available after $MAX_RETRIES retries"
        exit 1
    fi
    sleep 1
done
echo "logstream: DB is up, starting log client"

exec "$LOG_CLIENT_BIN" "$DB_ADDRESS" "$DB_PORT"
