#!/bin/bash
docker run --rm --link pgs_cohmetrix:pgs_cohmetrix -v /home/alefa/nilcmetrix:/opt/text_metrics cohmetrix:focal bash -c "python3 run_profiled.py \"$1\""
