#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm --name cohmetrix --link pgs_cohmetrix:pgs_cohmetrix -v "$SCRIPT_DIR":/opt/text_metrics cohmetrix bash -c "python3 run156.py \"$1\" \"$2\""
