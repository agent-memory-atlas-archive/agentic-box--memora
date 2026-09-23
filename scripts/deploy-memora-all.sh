#!/usr/bin/env bash
# Full deploy of the live memora-all container (nuc8) to v0.4.3: fetch +
# build the tagged image and recreate the container from it, then verify it.
#
# What v0.4.3 changes (see CHANGELOG.md "0.4.3"): memory_absorb only.
#  - Far fewer D1 round trips: batched phase-1 reads, bounded supersession
#    views, SELECT 1 existence checks, batched embeddings. Modeled on a
#    9-fact update-heavy absorb: 550 D1 requests / ~120 s -> 204 / ~49 s.
#    Why: live absorb calls were hitting the caller's 300 s timeout while
#    the server kept committing.
#  - A per-leaf supersede gate (similarity floor + a narrow LLM check on the
#    exact memory being hidden, re-checked at the write boundary and for
#    concurrent siblings). Why: #1082 was superseded by unrelated #1109.
#    Each UPDATE now costs one extra LLM call per leaf it would supersede.
#  - Every absorb result carries a "profile" field (per-phase time and D1
#    request counts).
#
# NO MODEL CHANGE: MEMORA_LLM_MODEL stays openai/gpt-4o-mini (set by the
# v0.4.1 deploy); step 2 below re-writes the same value, a confirming no-op.
# No new required env vars. One new OPTIONAL one, set here:
# MEMORA_LOG_LEVEL=INFO. Without it memora configures no logging and every
# INFO line -- including absorb's per-call profile and its supersede /
# downgrade audit lines -- is silently dropped. With it they go to the
# container's stderr (docker logs memora-all). The audit lines carry up to
# 500 characters of memory text each; that log stays on nuc8.
#
# Steps, all on nuc8:
#  1. git fetch + checkout the v0.4.3 tag in the nuc8 checkout, docker build.
#     The image currently tagged memora:latest is kept as memora:rollback-<ts>
#     before the new one replaces it.
#  2. Edit MEMORA_LLM_MODEL in ~/.config/memora/credentials.mcp.json (already
#     openai/gpt-4o-mini -- a confirming no-op, see above; backup kept).
#  3. Recreate memora-all -- same image tag, mounts, ports, memory/cpu limits
#     and restart policy the live container already runs with (checked via
#     docker inspect on 2026-09-14), plus MEMORA_LOG_LEVEL=INFO. Old
#     container kept stopped as memora-all-grok-<ts> (the name predates the
#     model switch being a no-op; it still means "the container before this
#     deploy", and the rollback commands below depend on it).
#  4. Wait for GET /health, check it reports version 0.4.3 (proves the new
#     build is the one serving, not a stale image), then run one 3-fact
#     dry-run memory_absorb call, asserting no JSON-RPC error and a real
#     session id at initialize, no JSON-RPC error / isError at tools/call (a
#     JSON-RPC error rides HTTP 200 -- an HTTP-status-only check would print
#     and exit zero on a server that answers but can't actually serve
#     requests), and a result with a "decisions" list and the new "profile"
#     field, before calling this done.
#
# HARDENED (queue item 23 follow-up, sealed review msg 5698/5699): the
# credentials-env parser used to stream straight into the while loop via
# `done < <(python3 ...)` — a parser failure inside a process substitution
# is invisible to both the while loop and set -e, so ENV_ARGS could
# silently end up empty while the script still stopped/renamed/recreated
# the live container. Same root cause as switch-embedding-host.sh's own
# 2026-09-16 incident (see that script's header), caught here by review
# before ever running. Now captured to a variable and checked (exit status
# + non-empty) before any destructive step; the health-wait loop's
# `$(seq ...)` also replaced with shell arithmetic.
#
# NOT RUN by this repo or any agent — review and run it yourself:
#   scripts/deploy-memora-all.sh
#
# Rollback:
#   ssh nuc8 'docker rm -f memora-all && docker rename memora-all-grok-<ts> memora-all && docker start memora-all'
#   ssh nuc8 'docker tag memora:rollback-<ts> memora:latest'   # only if the image itself needs reverting too
#   restore ~/.config/memora/credentials.mcp.json.bak-llm-<ts> if MEMORA_LLM_MODEL itself needs reverting
set -euo pipefail

TAG="v0.4.3"

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

# Captured to a variable FIRST, not streamed straight into the while loop
# via process substitution (`done < <(python3 ...)`) — a parser failure
# inside a process substitution is invisible to both the while loop and
# set -e, so ENV_ARGS could silently end up empty and the script would
# still stop/rename/recreate the live container on the next lines (the
# same failure class as the 2026-09-16 incident this script's sibling
# switch-embedding-host.sh already post-mortems). Check exit status AND
# non-empty output explicitly, before any destructive step.
ENV_LINES="$(python3 -c "
import json
env = json.load(open('$CRED'))['mcpServers']['memora']['env']
for k, v in env.items():
    if v != '':
        print(f'{k}={v}')
")" || { echo "credentials parser failed — aborting before touching the live container" >&2; exit 1; }
[ -n "$ENV_LINES" ] || { echo "credentials parser produced no output — aborting before touching the live container" >&2; exit 1; }

ENV_ARGS=()
while IFS='=' read -r key value; do
  [ -z "$key" ] && continue
  case "$key" in
    MEMORA_STORAGE_URI|MEMORA_DB_PATH|MEMORA_DATABASES|MEMORA_DEFAULT_DB) continue ;;
  esac
  ENV_ARGS+=(-e "$key=$value")
done <<< "$ENV_LINES"

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
  -e "MEMORA_LOG_LEVEL=INFO" \
  -e "MEMORA_DATABASES=$MEMORA_DATABASES" \
  -e "MEMORA_DEFAULT_DB=memora" \
  "${ENV_ARGS[@]}" \
  memora:latest

echo "memora-all recreated from $TAG (MEMORA_LLM_MODEL=openai/gpt-4o-mini unchanged, MEMORA_LOG_LEVEL=INFO)"
echo "old container kept stopped as memora-all-grok-$TS; old image kept as memora:rollback-$TS"
echo "rollback: docker rm -f memora-all && docker rename memora-all-grok-$TS memora-all && docker start memora-all"

echo "waiting for /health..."
healthy=0
for ((i = 1; i <= 30; i++)); do
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

python3 - "${TAG#v}" <<'PY'
import json, sys, time, urllib.request

EXPECTED_VERSION = sys.argv[1]
BASE = "http://127.0.0.1:8920/mcp/memora"

# The version the RUNNING process reports -- a stale image or a failed
# rebuild would still answer /health, just with the old version.
with urllib.request.urlopen("http://127.0.0.1:8920/health", timeout=10) as resp:
    health = json.loads(resp.read().decode())
if health.get("version") != EXPECTED_VERSION:
    print(f"/health reports version {health.get('version')!r}, expected {EXPECTED_VERSION!r}", file=sys.stderr)
    sys.exit(1)
print(f"/health reports version {EXPECTED_VERSION}")
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

# A JSON-RPC error rides HTTP 200 — urllib/curl's own status checks never
# see it. Check the envelope's own "error" key and, for initialize, that a
# session id actually came back; a smoke test that only checks HTTP status
# would print and exit zero on a server that answers but can't actually
# serve requests.
sid, init_raw = _post({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
               "clientInfo": {"name": "deploy-check", "version": "0"}},
})
init_result = _parse_sse(init_raw)
if "error" in init_result:
    print(f"initialize returned a JSON-RPC error: {init_result['error']}", file=sys.stderr)
    sys.exit(1)
if not sid:
    print("initialize succeeded but no mcp-session-id header was returned", file=sys.stderr)
    sys.exit(1)

_post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id=sid)

facts = [
    "deploy-check fact one about the v0.4.3 absorb rollout",
    "deploy-check fact two about the v0.4.3 absorb rollout",
    "deploy-check fact three about the v0.4.3 absorb rollout",
]
t0 = time.time()
_, raw = _post({
    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
    "params": {"name": "memory_absorb", "arguments": {"facts": facts, "dry_run": True}},
}, session_id=sid)
elapsed = time.time() - t0
result = _parse_sse(raw)
if "error" in result:
    print(f"tools/call returned a JSON-RPC error: {result['error']}", file=sys.stderr)
    sys.exit(1)
tool_result = result.get("result", {})
if tool_result.get("isError"):
    print(f"memory_absorb reported isError=true: {json.dumps(tool_result)[:2000]}", file=sys.stderr)
    sys.exit(1)
# The absorb result dict: FastMCP sends it as JSON text content (and as
# structuredContent["result"]). Check its shape, not just the absence of an
# error: v0.4.3 must return "decisions" and the new "profile" field.
absorb = None
for item in tool_result.get("content") or []:
    if item.get("type") == "text":
        try:
            absorb = json.loads(item["text"])
        except ValueError:
            pass
        break
if absorb is None:
    absorb = (tool_result.get("structuredContent") or {}).get("result")
if not isinstance(absorb, dict) or "error" in absorb:
    print(f"memory_absorb returned no result dict or an error: {json.dumps(tool_result)[:2000]}", file=sys.stderr)
    sys.exit(1)
# Not one decision per fact: near-identical facts may be consolidated.
if not isinstance(absorb.get("decisions"), list) or not absorb["decisions"]:
    print(f"memory_absorb result has no decisions: {json.dumps(absorb)[:2000]}", file=sys.stderr)
    sys.exit(1)
profile = absorb.get("profile")
if not isinstance(profile, dict) or "total_requests" not in profile:
    print(f"memory_absorb result lacks the v0.4.3 profile field: {json.dumps(absorb)[:2000]}", file=sys.stderr)
    sys.exit(1)
print(f"3-fact dry-run absorb via memory store: {elapsed:.1f}s "
      f"({profile['total_requests']} {profile.get('request_unit', 'requests')}, "
      f"server-side {profile['total_seconds']}s)")
print("actions:", [d.get("action") for d in absorb["decisions"]])
PY
REMOTE
