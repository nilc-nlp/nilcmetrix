#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run -d --name cohmetrixp --link pgs_cohmetrix:pgs_cohmetrix -v "$SCRIPT_DIR":/opt/text_metrics --restart always cohmetrix bash -c "python3 porsimples.py"
