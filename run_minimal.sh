#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm --link pgs_cohmetrix:pgs_cohmetrix -v "$SCRIPT_DIR":/opt/text_metrics cohmetrix:noble bash -c "python3 run_min.py \"$1\" \"$2\""
