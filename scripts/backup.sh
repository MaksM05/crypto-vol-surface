#!/usr/bin/env bash
set -euo pipefail
out=/root/backups
mkdir -p "$out"
ts=$(date -u +%Y%m%dT%H%M%SZ)
docker compose -f /root/crypto-vol-surface/docker-compose.yml exec -T timescaledb \
  pg_dump -U volsurface volsurface | gzip > "$out/volsurface_$ts.sql.gz"
# keep the last 14 dumps, delete older
ls -1t "$out"/volsurface_*.sql.gz | tail -n +15 | xargs -r rm
