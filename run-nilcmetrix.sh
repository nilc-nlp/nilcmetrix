#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker stop nilcmetrix
docker rm nilcmetrix
docker run -d --name nilcmetrix -p 80:8080 --link pgs_cohmetrix:pgs_cohmetrix --link palavras:palavras -v "$SCRIPT_DIR":/opt/text_metrics --restart always nilcmetrix:$1
