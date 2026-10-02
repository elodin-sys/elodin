#!/usr/bin/env bash
#
# Emit KDL from a Python schematic and diff it against its source KDL.
#
# Usage (from the repository root, inside `nix develop`):
#   scripts/diff-python-schematic.sh <schematic.py> <source.kdl> [diff options]

set -euo pipefail;

if [[ $# -lt 2  ]]; then
    echo "usage: $0 <schematic.kdl> <source.py> [diff options]" >&2;
    exit 2;
fi

SOURCE_KDL="$1";
shift;
PYTHON_SCHEMATIC="$1";
shift;

for path in "$PYTHON_SCHEMATIC" "$SOURCE_KDL"; do
    if [[ ! -f "$path" ]]; then
        echo "error: file not found: $path" >&2;
        exit 2;
    fi
done

set +e;
diff -u "$@" \
    --label "$SOURCE_KDL (original)" \
    --label "uv run python $PYTHON_SCHEMATIC (generated)" \
    "$SOURCE_KDL" <(uv run python "$PYTHON_SCHEMATIC")
status=$?;
set -e;

case "$status" in
    0)
        echo "KDL files match exactly." >&2;
        ;;
    1)
        echo "KDL files differ." >&2;
        ;;
    *)
        echo "error: diff failed with status $status" >&2;
        ;;
esac

exit "$status";
