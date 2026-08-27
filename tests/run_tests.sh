#!/bin/bash
# Regression test harness entry point.
#
# Usage:
#   tests/run_tests.sh [test|update] [--pgs-container NAME] [-- ...extra args]
#
# Subcommands:
#   test     Run pytest against tests/ (default).
#   update   Regenerate goldens. Extra args after `--` are passed as stems
#            to update_goldens.py; with none, all goldens are regenerated.
#
# Flags:
#   --pgs-container NAME  Docker container exposing the cohmetrix Postgres
#                         service (linked as `pgs_cohmetrix`). Default:
#                         pgs_cohmetrix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBCOMMAND="test"
PGS_CONTAINER="pgs_cohmetrix"
EXTRA_ARGS=()

# Parse the optional subcommand.
if [[ $# -gt 0 && "$1" != --* ]]; then
    SUBCOMMAND="$1"
    shift
fi

# Parse flags and collect anything after `--` into EXTRA_ARGS.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pgs-container)
            PGS_CONTAINER="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$SUBCOMMAND" in
    test)
        INNER="cd /opt/text_metrics && pip3 install --quiet pytest && python3 -m pytest tests/ ${EXTRA_ARGS[*]:-}"
        ;;
    update)
        INNER="cd /opt/text_metrics && python3 tests/update_goldens.py ${EXTRA_ARGS[*]:-}"
        ;;
    *)
        echo "Unknown subcommand: $SUBCOMMAND" >&2
        echo "Usage: $0 [test|update] [--pgs-container NAME] [-- ...extra args]" >&2
        exit 2
        ;;
esac

exec docker run --rm \
    --link "${PGS_CONTAINER}:pgs_cohmetrix" \
    -v "$REPO_ROOT":/opt/text_metrics \
    cohmetrix:noble \
    bash -c "$INNER"
