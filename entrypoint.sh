#!/usr/bin/env bash
set -e

# Run once at container startup
/opt/venv/bin/python main.py || true

# Start cron in foreground
cron -f
