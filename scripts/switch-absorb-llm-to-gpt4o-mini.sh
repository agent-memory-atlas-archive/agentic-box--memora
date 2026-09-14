#!/usr/bin/env bash
# Switch the LIVE memora-all container (nuc8) from x-ai/grok-build-0.1 to
# openai/gpt-4o-mini for absorb classification. Absorb's classification
# latency was measured at 12-17s/call against the reasoning model (1000-1700
# reasoning tokens it ignores reasoning.effort/max_tokens for); gpt-4o-mini
# answered the same classification correctly in 3-4s in that same
# measurement.
#
# Edits MEMORA_LLM_MODEL in ~/.config/memora/credentials.mcp.json on nuc8,
# then recreates memora-all with the SAME image, mounts, ports, memory/cpu
# limits and restart policy the live container currently runs with (checked
# via docker inspect on 2026-09-14) — only MEMORA_LLM_MODEL changes. The old
# container is kept stopped under a timestamped name for rollback, same
# pattern as the earlier embeddings-backend cutover.
#
# NOT RUN by this repo or by any agent — review and run it yourself:
#   scripts/switch-absorb-llm-to-gpt4o-mini.sh
#
# Rollback (prints again at the end of a real run):
#   ssh nuc8 'docker rm -f memora-all && docker rename memora-all-grok-<ts> memora-all && docker start memora-all'
#   (and restore ~/.config/memora/credentials.mcp.json.bak-llm-<ts> if MEMORA_LLM_MODEL itself needs reverting)
set -euo pipefail

# MEMORA_DATABASES names a Cloudflare account + database ids — read from the
# git-ignored instance config rather than written into this (public) script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/instances/all.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE — need MEMORA_DATABASES for memora-all" >&2; exit 1; }
MEMORA_DATABASES="$(grep -E "^MEMORA_DATABASES=" "$ENV_FILE" | head -1 | cut -d= -f2- | sed "s/^'//;s/'\$//")"
[ -n "$MEMORA_DATABASES" ] || { echo "$ENV_FILE has no MEMORA_DATABASES" >&2; exit 1; }
# base64 over the wire: the JSON has embedded quotes that ssh's remote
# command re-join would otherwise mangle.
MEMORA_DATABASES_B64="$(printf '%s' "$MEMORA_DATABASES" | base64 | tr -d '\n')"

ssh nuc8 bash -s -- "$MEMORA_DATABASES_B64" <<'REMOTE'
set -euo pipefail
MEMORA_DATABASES="$(printf '%s' "$1" | base64 -d)"

CRED=~/.config/memora/credentials.mcp.json
[ -f "$CRED" ] || { echo "missing $CRED" >&2; exit 1; }

TS=$(date +%s)
cp -p "$CRED" "$CRED.bak-llm-$TS"

python3 - "$CRED" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
env = d["mcpServers"]["memora"]["env"]
before = env.get("MEMORA_LLM_MODEL")
env["MEMORA_LLM_MODEL"] = "openai/gpt-4o-mini"
json.dump(d, open(p, "w"), indent=2)
print(f"MEMORA_LLM_MODEL: {before!r} -> 'openai/gpt-4o-mini' (backup kept alongside)")
PY

HEALTH_TOKEN_FILE=~/.config/memora/all.health-token
[ -f "$HEALTH_TOKEN_FILE" ] || { echo "missing $HEALTH_TOKEN_FILE — refusing to mint a new one for a live container" >&2; exit 1; }
HEALTH_TOKEN=$(cat "$HEALTH_TOKEN_FILE")

# Reuse memora-all's EXISTING data volume by id — nothing about it changes.
VOLUME_ID=$(docker inspect memora-all --format '{{range .Mounts}}{{.Name}}{{end}}')
[ -n "$VOLUME_ID" ] || { echo "could not read memora-all's data volume id" >&2; exit 1; }

ENV_ARGS=()
while IFS='=' read -r key value; do
  [ -z "$key" ] && continue
  case "$key" in
    MEMORA_STORAGE_URI|MEMORA_DB_PATH|MEMORA_DATABASES|MEMORA_DEFAULT_DB) continue ;;
  esac
  ENV_ARGS+=(-e "$key=$value")
done < <(python3 -c "
import json
env = json.load(open('$CRED'))['mcpServers']['memora']['env']
for k, v in env.items():
    if v != '':
        print(f'{k}={v}')
")

docker stop memora-all
docker rename memora-all "memora-all-grok-$TS"

docker run -d --name memora-all \
  --restart unless-stopped \
  --memory 768m --cpus 4 \
  -p 0.0.0.0:8920:8000 \
  -v "$VOLUME_ID:/data" \
  -e "MEMORA_TOOL_PROFILE=leader" \
  -e "MEMORA_HEALTH_TOKEN=$HEALTH_TOKEN" \
  -e "MEMORA_HEALTH_TIMEOUT=30" \
  -e "MEMORA_HEALTH_REFRESH_INTERVAL=15" \
  -e "MEMORA_VECTOR_SCAN_PAGE_SIZE=100" \
  -e "MEMORA_ALLOW_ANY_TAG=1" \
  -e "MEMORA_DATABASES=$MEMORA_DATABASES" \
  -e "MEMORA_DEFAULT_DB=memora" \
  "${ENV_ARGS[@]}" \
  memora:latest

echo "memora-all recreated with MEMORA_LLM_MODEL=openai/gpt-4o-mini"
echo "old container kept stopped as memora-all-grok-$TS for rollback:"
echo "  docker rm -f memora-all && docker rename memora-all-grok-$TS memora-all && docker start memora-all"
REMOTE
