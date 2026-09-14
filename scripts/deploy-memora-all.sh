#!/usr/bin/env bash
# Full deploy of the live memora-all container (nuc8) to v0.4.1: fetch +
# build the tagged image, switch MEMORA_LLM_MODEL from x-ai/grok-build-0.1 to
# openai/gpt-4o-mini, recreate the container, then verify it. Absorb's
# classification latency was measured at 12-17s/call against the reasoning
# model (1000-1700 reasoning tokens it ignores reasoning.effort/max_tokens
# for); gpt-4o-mini answered the same classification correctly in 3-4s in
# that same measurement, and concurrency + the model switch together were
# measured at ~18.5x on a synthetic 7-fact absorb.
#
# Steps, all on nuc8:
#  1. git fetch + checkout the v0.4.1 tag in the nuc8 checkout, docker build.
#     The image currently tagged memora:latest is kept as memora:rollback-<ts>
#     before the new one replaces it.
#  2. Edit MEMORA_LLM_MODEL in ~/.config/memora/credentials.mcp.json.
#  3. Recreate memora-all — same image tag, mounts, ports, memory/cpu limits
#     and restart policy the live container already runs with (checked via
#     docker inspect on 2026-09-14), only MEMORA_LLM_MODEL and the image
#     content changed. Old container kept stopped as memora-all-grok-<ts>.
#  4. Wait for GET /health, then run one 3-fact dry-run memory_absorb call
#     and print its wall time, as a smoke test before calling this done.
#
# NOT RUN by this repo or any agent — review and run it yourself:
#   scripts/deploy-memora-all.sh
#
# Rollback:
#   ssh nuc8 'docker rm -f memora-all && docker rename memora-all-grok-<ts> memora-all && docker start memora-all'
#   ssh nuc8 'docker tag memora:rollback-<ts> memora:latest'   # only if the image itself needs reverting too
#   restore ~/.config/memora/credentials.mcp.json.bak-llm-<ts> if MEMORA_LLM_MODEL itself needs reverting
set -euo pipefail

TAG="v0.4.1"

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

ssh nuc8 bash -s -- "$TAG" "$MEMORA_DATABASES_B64" <<'REMOTE'
set -euo pipefail
TAG="$1"
MEMORA_DATABASES="$(printf '%s' "$2" | base64 -d)"
TS=$(date +%s)

REPO=~/repos/agentic-box/memora
[ -d "$REPO" ] || { echo "missing $REPO checkout on nuc8" >&2; exit 1; }
git -C "$REPO" fetch origin
git -C "$REPO" checkout "$TAG"

# Keep the currently-running image for rollback before building over it.
docker tag memora:latest "memora:rollback-$TS" 2>/dev/null || true
docker build -t memora:latest "$REPO"

CRED=~/.config/memora/credentials.mcp.json
[ -f "$CRED" ] || { echo "missing $CRED" >&2; exit 1; }
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

echo "memora-all recreated from $TAG with MEMORA_LLM_MODEL=openai/gpt-4o-mini"
echo "old container kept stopped as memora-all-grok-$TS; old image kept as memora:rollback-$TS"
echo "rollback: docker rm -f memora-all && docker rename memora-all-grok-$TS memora-all && docker start memora-all"

echo "waiting for /health..."
healthy=0
for i in $(seq 1 30); do
  if curl -sf -m 3 http://127.0.0.1:8920/health >/dev/null 2>&1; then
    healthy=1
    echo "healthy after about $((i * 2))s"
    break
  fi
  sleep 2
done
if [ "$healthy" -ne 1 ]; then
  echo "still not healthy after ~60s — check: docker logs memora-all" >&2
  exit 1
fi

python3 - <<'PY'
import json, time, urllib.request

BASE = "http://127.0.0.1:8920/mcp/memora"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

def _post(body, session_id=None):
    headers = dict(HEADERS)
    if session_id:
        headers["mcp-session-id"] = session_id
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        sid = resp.headers.get("mcp-session-id")
        raw = resp.read().decode()
    return sid, raw

def _parse_sse(raw):
    for line in raw.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    return json.loads(raw)

sid, _ = _post({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
               "clientInfo": {"name": "deploy-check", "version": "0"}},
})
_post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id=sid)

facts = [
    "deploy-check fact one about the gpt-4o-mini rollout",
    "deploy-check fact two about the gpt-4o-mini rollout",
    "deploy-check fact three about the gpt-4o-mini rollout",
]
t0 = time.time()
_, raw = _post({
    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
    "params": {"name": "memory_absorb", "arguments": {"facts": facts, "dry_run": True}},
}, session_id=sid)
elapsed = time.time() - t0
result = _parse_sse(raw)
print(f"3-fact dry-run absorb via memory store: {elapsed:.1f}s")
print(json.dumps(result.get("result", result), indent=2)[:2000])
PY
REMOTE
