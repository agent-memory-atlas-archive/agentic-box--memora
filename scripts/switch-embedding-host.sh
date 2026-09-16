#!/usr/bin/env bash
# Repoint memora-all's embedding endpoint (MEMORA_EMBEDDING_BASE_URL) at the
# replacement Ollama host. Same image/volume/mounts/ports/limits as the live
# container — this changes exactly one env var, nothing else. Queue item 23.
#
# ALREADY RUN (2026-09-16, nuc8): the new host answered, the switch and the
# 3-fact dry-run absorb smoke test both succeeded, and dev was pushed. This
# script is kept for reference / re-run on a future host move, not because
# there is anything left to do.
#
# POST-MORTEM: the first live run of this script's MEMORA_DATABASES line
# tried `json.load(open(credentials.mcp.json))[...]['env']['MEMORA_DATABASES']`
# — that key does not exist in credentials.mcp.json (deploy-memora-all.sh
# already knew this and reads it from instances/all.env instead; this script
# didn't follow that precedent). The KeyError's exit code was swallowed by
# bash's `$(...)` command substitution (not caught by `set -e`, a known
# gotcha — a failing command substitution only aborts the script if the
# OUTER command also fails), so `docker run` proceeded with
# MEMORA_DATABASES="" and memora-all came up with 404s on /mcp/memora for
# ~90 seconds before being caught (curl -i by hand, not silenced with -s)
# and fixed forward with a second recreate sourcing MEMORA_DATABASES from
# instances/all.env like this version now does. No data loss — the shared
# volume was never touched. Lesson kept here on purpose: never build an
# env var for `docker run` from a command substitution without checking its
# exit status explicitly (`x=$(cmd) || exit 1`, not just `set -e` and hope).
#
# Old host: 100.85.27.63:11434 (decommissioned M1 — offline in clmux's own
#   mesh peer registry too, ~/.clmux-mesh/network/peers.json on nuc8; that
#   file is clmux's, not memora's, and was out of scope here, untouched).
# New host: 100.118.64.23:11434 (Ollama 0.34.0, bge-m3, confirmed live).
#
# Verified nothing else on nuc8 or in this repo hardcodes the old IP:
#   - memora repo (incl. gitignored instances/*.env): no match.
#   - nuc8's memora checkout: no match.
#   - nuc8 home dir scripts/json/env/py/yaml/conf/service files: no match
#     except ~/.config/memora/credentials.mcp.json itself (the file this
#     script edits) and ~/.clmux-mesh/network/peers.json (see above).
#
# To re-run for a future host move, review and run yourself:
#   scripts/switch-embedding-host.sh
#
# Rollback (restores the state from immediately before this script's most
# recent run — check `docker ps -a --filter name=memora-all-embhost` on
# nuc8 for the exact <ts>, there may be more than one from repeated runs):
#   ssh nuc8 'docker rm -f memora-all && docker rename memora-all-embhost-<ts> memora-all && docker start memora-all'
#   restore ~/.config/memora/credentials.mcp.json.bak-embhost-<ts> if the credentials file itself needs reverting
set -euo pipefail

OLD_URL="http://100.85.27.63:11434/v1"
NEW_URL="http://100.118.64.23:11434/v1"

# MEMORA_DATABASES names a Cloudflare account + database ids — read from the
# git-ignored instance config rather than credentials.mcp.json, which does
# NOT carry this key (see the post-mortem above).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/instances/all.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE — need MEMORA_DATABASES for memora-all" >&2; exit 1; }
MEMORA_DATABASES="$(grep -E "^MEMORA_DATABASES=" "$ENV_FILE" | head -1 | cut -d= -f2- | sed "s/^'//;s/'\$//")"
[ -n "$MEMORA_DATABASES" ] || { echo "$ENV_FILE has no MEMORA_DATABASES" >&2; exit 1; }
MEMORA_DATABASES_B64="$(printf '%s' "$MEMORA_DATABASES" | base64 | tr -d '\n')"

ssh nuc8 bash -s -- "$OLD_URL" "$NEW_URL" "$MEMORA_DATABASES_B64" <<'REMOTE'
set -euo pipefail
OLD_URL="$1"
NEW_URL="$2"
MEMORA_DATABASES="$(printf '%s' "$3" | base64 -d)"
TS=$(date +%s)

CRED=~/.config/memora/credentials.mcp.json
[ -f "$CRED" ] || { echo "missing $CRED" >&2; exit 1; }
cp -p "$CRED" "$CRED.bak-embhost-$TS"

# Fail loudly rather than silently no-op if the base URL has already moved
# (someone else changed it, or this script already ran) or the new host
# still isn't reachable — this must not proceed on a stale assumption.
if ! curl -sf -m 3 "http://100.118.64.23:11434/" >/dev/null 2>&1; then
  echo "new embedding host 100.118.64.23:11434 is not answering — aborting, not touching credentials" >&2
  exit 1
fi

python3 - "$CRED" "$OLD_URL" "$NEW_URL" <<'PY'
import json, sys
p, old_url, new_url = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(p))
env = d["mcpServers"]["memora"]["env"]
before = env.get("MEMORA_EMBEDDING_BASE_URL")
if before != old_url:
    print(f"MEMORA_EMBEDDING_BASE_URL is {before!r}, not the expected {old_url!r} — aborting, not writing", file=sys.stderr)
    sys.exit(1)
env["MEMORA_EMBEDDING_BASE_URL"] = new_url
json.dump(d, open(p, "w"), indent=2)
print(f"MEMORA_EMBEDDING_BASE_URL: {before!r} -> {new_url!r} (backup kept alongside)")
PY

HEALTH_TOKEN_FILE=~/.config/memora/all.health-token
[ -f "$HEALTH_TOKEN_FILE" ] || { echo "missing $HEALTH_TOKEN_FILE — refusing to mint a new one for a live container" >&2; exit 1; }
HEALTH_TOKEN=$(cat "$HEALTH_TOKEN_FILE")

# Reuse memora-all's EXISTING data volume and image — nothing about either changes.
VOLUME_ID=$(docker inspect memora-all --format '{{range .Mounts}}{{.Name}}{{end}}')
[ -n "$VOLUME_ID" ] || { echo "could not read memora-all's data volume id" >&2; exit 1; }
IMAGE_ID=$(docker inspect memora-all --format '{{.Image}}')
[ -n "$IMAGE_ID" ] || { echo "could not read memora-all's image id" >&2; exit 1; }

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
docker rename memora-all "memora-all-embhost-$TS"

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
  "$IMAGE_ID"

echo "memora-all recreated with MEMORA_EMBEDDING_BASE_URL=$NEW_URL"
echo "old container kept stopped as memora-all-embhost-$TS (same image, old embedding host)"
echo "rollback: docker rm -f memora-all && docker rename memora-all-embhost-$TS memora-all && docker start memora-all"

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
    "deploy-check fact one about the embedding host switch",
    "deploy-check fact two about the embedding host switch",
    "deploy-check fact three about the embedding host switch",
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
