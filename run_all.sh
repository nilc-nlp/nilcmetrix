#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm --link pgs_cohmetrix:pgs_cohmetrix --link palavras:palavras -v "$SCRIPT_DIR":/opt/text_metrics cohmetrix:focal bash -c "python3 run_all.py \"$1\" \"$2\""
