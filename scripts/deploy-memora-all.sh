#!/usr/bin/env bash
# Full deploy of the live memora-all container (nuc8) to v0.4.5: fetch +
# build the tagged image and recreate the container from it, then verify it.
#
# What v0.4.5 changes (see CHANGELOG.md "0.4.5"):
#  - Project identity is explicit (issue #47): keyword-based project
#    detection is gone. A memory's project comes from an explicit `project`
#    argument, else metadata.project, else exactly one tag naming a project
#    CONFIGURED for its store (MEMORA_PROJECTS). Typed tags are
#    <project>/issues|todos|sections|documents, or bare without a project --
#    no more memora/... default.
#  - Import hardening: prepare-before-delete, D1 staged clear, per-row
#    markers with verified cleanup, one fenced import lease per store,
#    truthful partial results. A D1 replace is still NOT atomic (recover by
#    re-running it from the export file). New admin tool
#    memory_import_sweep (full profile only; this container runs `leader`,
#    so it is not exposed here); memory_stats gains `import_pending`.
#
# CONFIGURATION CHANGE -- this deploy ADDS one env var, MEMORA_PROJECTS,
# keyed by registry store name:
#   {"memora":["memora","clmux","acebar","pi"],"ob1":["ob1"],
#    "bestation":["bestation"],"re":["re"]}
# (acebar and pi write to the memora store, per their .mcp.json.) Why: with
# the keyword heuristics removed, tags only imply a project the store
# declares. WITHOUT this variable no project is inferred from tags at all,
# so only callers that pass an explicit `project` would get sections and
# project-prefixed tags, and memora/clmux-tagged content would lose its
# section conventions. With it, a project outside a store's list is rejected
# (invalid_input). It is set here, not in credentials.mcp.json (a copy there
# is ignored so this value always wins). The value is validated with the NEW
# image (memora's own parser, and its keys must equal MEMORA_DATABASES'
# store names) before anything is stopped. MEMORA_LLM_MODEL is unchanged
# (step 2 is a confirming no-op); MEMORA_CORPUS_CACHE_BUDGET_MB stays unset.
#
# SCHEMA: on first connect v0.4.5 creates one new, empty table per store,
# import_lease (CREATE TABLE IF NOT EXISTS; nothing existing changes).
#
# STARTUP SWEEP: at startup v0.4.5 sweeps every store, on a daemon thread,
# for rows whose metadata carries an `import_attempt` marker (an interrupted
# import) and completes or REMOVES them. No released memora ever wrote that
# key (it first appears in this release), so the live stores should have
# none -- but before 0.4.5 a caller could put ANY key in metadata. So step 3
# first counts, READ-ONLY, rows whose metadata contains "import_attempt" in
# each live store (through the running v0.4.4 container, raw connection, no
# schema pass) and ABORTS the deploy if any store has one, before anything
# is stopped.
#
# Steps, all on nuc8:
#  1. git fetch + checkout the v0.4.5 tag in the nuc8 checkout, docker build.
#     The image currently tagged memora:latest is kept as memora:rollback-<ts>
#     before the new one replaces it.
#  2. Edit MEMORA_LLM_MODEL in ~/.config/memora/credentials.mcp.json (already
#     openai/gpt-4o-mini -- a confirming no-op; backup kept).
#  3. Preflight, before any destructive step: validate MEMORA_PROJECTS with
#     the new image, and the read-only import_attempt count on the live
#     stores (see above). Either failing aborts with the old container
#     untouched and still serving.
#  4. Recreate memora-all -- same image tag, mounts, ports, memory/cpu limits,
#     restart policy and env as the v0.4.4 deploy, plus MEMORA_PROJECTS. Old
#     container kept stopped as memora-all-grok-<ts> (the name predates the
#     model switch being a no-op; it still means "the container before this
#     deploy", and the rollback commands below depend on it).
#  5. Wait for GET /health, check it reports version 0.4.5 (proves the new
#     build is the one serving, not a stale image), then run one 3-fact
#     dry-run memory_absorb call, one memory_semantic_search call and one
#     memory_stats call, asserting no JSON-RPC error and a real session id at
#     initialize, no JSON-RPC error / isError at each tools/call (a JSON-RPC
#     error rides HTTP 200 -- an HTTP-status-only check would print and exit
#     zero on a server that answers but can't actually serve requests), an
#     absorb result with a "decisions" list and a "profile" field, a search
#     result with a "results" list and a "profile" field, and a stats result
#     with an integer "import_pending" field. Then EVERY store in
#     MEMORA_DATABASES (not only the default one the calls above use; /health
#     itself touches no database): /health/db/<store> must reach a current
#     (not stale) 200 ok within 90 s, and memory_stats over /mcp/<store> must
#     report that store as its bound database, with an integer
#     import_pending. Any store failing is named and fails the deploy.
#     ("pi" in the memora project list is a tag project inside the memora
#     store, not a store; pi agents have no MCP config.)
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

TAG="v0.4.5"

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
# Keyed by registry store name (MEMORA_DATABASES); see the header for why.
MEMORA_PROJECTS='{"memora":["memora","clmux","acebar","pi"],"ob1":["ob1"],"bestation":["bestation"],"re":["re"]}'

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
    MEMORA_STORAGE_URI|MEMORA_DB_PATH|MEMORA_DATABASES|MEMORA_DEFAULT_DB|MEMORA_PROJECTS) continue ;;
  esac
  ENV_ARGS+=(-e "$key=$value")
done <<< "$ENV_LINES"

# Preflight 1: MEMORA_PROJECTS parses with the NEW image's own validator
# (the server refuses to start on a malformed value), and names exactly the
# registry's stores.
docker run --rm -e "MEMORA_PROJECTS=$MEMORA_PROJECTS" -e "MEMORA_DATABASES=$MEMORA_DATABASES" \
  memora:latest python -c '
import json, os, sys
from memora.storage import load_projects_config
projects = load_projects_config()
stores = set(json.loads(os.environ["MEMORA_DATABASES"]))
if not isinstance(projects, dict) or set(projects) != stores:
    sys.exit(f"MEMORA_PROJECTS stores {sorted(projects or [])} != MEMORA_DATABASES stores {sorted(stores)}")
print("MEMORA_PROJECTS ok:", json.dumps(projects, sort_keys=True))
' || { echo "MEMORA_PROJECTS preflight failed — aborting before touching the live container" >&2; exit 1; }

# Preflight 2 (READ-ONLY): the startup sweep of v0.4.5 completes or removes
# rows whose metadata carries an import_attempt marker. None should exist
# (no released memora wrote the key), but a caller could have set it. Count,
# per live store, rows whose metadata contains the string at all (a superset
# of real markers), through the running v0.4.4 container: a raw backend
# connection, so no schema pass -- one SELECT per store. Any hit, or a
# failed check, aborts before anything is stopped.
docker exec -i memora-all python - <<'PY' || { echo "import_attempt preflight failed — aborting before touching the live container" >&2; exit 1; }
import json, os, sys
from memora import storage
bad = {}
for name in json.loads(os.environ["MEMORA_DATABASES"]):
    conn = storage.backend_for(name).connect()
    try:
        rows = conn.execute(
            "SELECT id FROM memories WHERE instr(metadata, ?) > 0 LIMIT 20", ('"import_attempt"',)
        ).fetchall()
    finally:
        conn.close()
    ids = [int(r[0]) for r in rows]
    print(f"{name}: {len(ids)} row(s) with import_attempt in metadata")
    if ids:
        bad[name] = ids
if bad:
    sys.exit(f"rows the v0.4.5 startup sweep could complete or remove: {bad} -- inspect them first")
PY

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
  -e "MEMORA_PROJECTS=$MEMORA_PROJECTS" \
  "${ENV_ARGS[@]}" \
  memora:latest

echo "memora-all recreated from $TAG (MEMORA_PROJECTS set; MEMORA_LLM_MODEL=openai/gpt-4o-mini unchanged, MEMORA_LOG_LEVEL=INFO, corpus cache budget default 384 MB)"
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

python3 - "${TAG#v}" "$MEMORA_DATABASES" "$HEALTH_TOKEN" <<'PY'
import json, sys, time, urllib.error, urllib.request

EXPECTED_VERSION = sys.argv[1]
STORES = list(json.loads(sys.argv[2]))
HEALTH_TOKEN = sys.argv[3]
ROOT = "http://127.0.0.1:8920"
BASE = f"{ROOT}/mcp/memora"

# The version the RUNNING process reports -- a stale image or a failed
# rebuild would still answer /health, just with the old version.
with urllib.request.urlopen("http://127.0.0.1:8920/health", timeout=10) as resp:
    health = json.loads(resp.read().decode())
if health.get("version") != EXPECTED_VERSION:
    print(f"/health reports version {health.get('version')!r}, expected {EXPECTED_VERSION!r}", file=sys.stderr)
    sys.exit(1)
print(f"/health reports version {EXPECTED_VERSION}")
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

def _post(body, session_id=None, base=BASE):
    headers = dict(HEADERS)
    if session_id:
        headers["mcp-session-id"] = session_id
    req = urllib.request.Request(base, data=json.dumps(body).encode(), headers=headers, method="POST")
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
def _initialize(base):
    sid, init_raw = _post({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "deploy-check", "version": "0"}},
    }, base=base)
    init_result = _parse_sse(init_raw)
    if "error" in init_result:
        print(f"initialize at {base} returned a JSON-RPC error: {init_result['error']}", file=sys.stderr)
        sys.exit(1)
    if not sid:
        print(f"initialize at {base} succeeded but no mcp-session-id header was returned", file=sys.stderr)
        sys.exit(1)
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id=sid, base=base)
    return sid


sid = _initialize(BASE)

def _tool_dict(tool_result, name):
    """The tool's result dict: FastMCP sends it as JSON text content (and as
    structuredContent["result"]). Exits on a missing dict or an error key."""
    out = None
    for item in tool_result.get("content") or []:
        if item.get("type") == "text":
            try:
                out = json.loads(item["text"])
            except ValueError:
                pass
            break
    if out is None:
        out = (tool_result.get("structuredContent") or {}).get("result")
    if not isinstance(out, dict) or "error" in out:
        print(f"{name} returned no result dict or an error: {json.dumps(tool_result)[:2000]}", file=sys.stderr)
        sys.exit(1)
    return out


def _call_tool(req_id, name, arguments, session=None, base=BASE):
    t0 = time.time()
    _, raw = _post({
        "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }, session_id=session or sid, base=base)
    elapsed = time.time() - t0
    result = _parse_sse(raw)
    if "error" in result:
        print(f"{name} tools/call returned a JSON-RPC error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    tool_result = result.get("result", {})
    if tool_result.get("isError"):
        print(f"{name} reported isError=true: {json.dumps(tool_result)[:2000]}", file=sys.stderr)
        sys.exit(1)
    return _tool_dict(tool_result, name), elapsed


def _require_profile(name, out):
    profile = out.get("profile")
    if not isinstance(profile, dict) or "total_requests" not in profile:
        print(f"{name} result lacks the profile field: {json.dumps(out)[:2000]}", file=sys.stderr)
        sys.exit(1)
    return profile


facts = [
    "deploy-check fact one about the v0.4.5 project-identity rollout",
    "deploy-check fact two about the v0.4.5 project-identity rollout",
    "deploy-check fact three about the v0.4.5 project-identity rollout",
]
absorb, elapsed = _call_tool(2, "memory_absorb", {"facts": facts, "dry_run": True})
# Not one decision per fact: near-identical facts may be consolidated.
if not isinstance(absorb.get("decisions"), list) or not absorb["decisions"]:
    print(f"memory_absorb result has no decisions: {json.dumps(absorb)[:2000]}", file=sys.stderr)
    sys.exit(1)
profile = _require_profile("memory_absorb", absorb)
print(f"3-fact dry-run absorb via memory store: {elapsed:.1f}s "
      f"({profile['total_requests']} {profile.get('request_unit', 'requests')}, "
      f"server-side {profile['total_seconds']}s)")
print("actions:", [d.get("action") for d in absorb["decisions"]])

search, elapsed = _call_tool(3, "memory_semantic_search", {"query": "memora deploy", "top_k": 3})
if not isinstance(search.get("results"), list):
    print(f"memory_semantic_search result has no results list: {json.dumps(search)[:2000]}", file=sys.stderr)
    sys.exit(1)
profile = _require_profile("memory_semantic_search", search)
print(f"semantic search via memory store: {elapsed:.1f}s, {len(search['results'])} results "
      f"({profile['total_requests']} {profile.get('request_unit', 'requests')}, "
      f"server-side {profile['total_seconds']}s)")

stats, elapsed = _call_tool(4, "memory_stats", {})
pending = stats.get("import_pending")
if not isinstance(pending, int) or isinstance(pending, bool):
    print(f"memory_stats has no integer import_pending field: {json.dumps(stats)[:2000]}", file=sys.stderr)
    sys.exit(1)
print(f"memory_stats via memory store: {elapsed:.1f}s, {stats.get('total_memories')} memories, "
      f"import_pending={pending}")

# EVERY store, not just the default one: /health is liveness only (no
# database), and the calls above all went to the memora store. A store the
# new image cannot reach, or misreads, must fail the deploy by name.
#  a) /health/db/<store>: poll to a CURRENT 200 ok (not stale), bounded.
#  b) memory_stats over /mcp/<store>: a real tool call through the router,
#     bound to that very database, with an integer import_pending.
def _store_health(store):
    req = urllib.request.Request(f"{ROOT}/health/db/{store}",
                                 headers={"Authorization": f"Bearer {HEALTH_TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode() or "{}")
        except ValueError:
            return exc.code, {}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, {"error": str(exc)}

failed = []
for store in STORES:
    deadline = time.time() + 90
    healthy = False
    while not healthy:
        status, body = _store_health(store)
        healthy = status == 200 and body.get("status") == "ok" and body.get("stale") is False
        if not healthy and time.time() > deadline:
            break
        if not healthy:
            time.sleep(3)
    if not healthy:
        failed.append(f"{store}: /health/db/{store} not a current 200 ok after 90s "
                      f"(last: HTTP {status}, {json.dumps(body)[:300]})")
        continue
    store_base = f"{ROOT}/mcp/{store}"
    store_sid = _initialize(store_base)
    stats, elapsed = _call_tool(10, "memory_stats", {}, session=store_sid, base=store_base)
    if stats.get("database") != store:
        failed.append(f"{store}: memory_stats is bound to {stats.get('database')!r}, not {store!r}")
        continue
    pending = stats.get("import_pending")
    if not isinstance(pending, int) or isinstance(pending, bool):
        failed.append(f"{store}: memory_stats has no integer import_pending")
        continue
    print(f"store {store}: /health/db 200 ok (latency {body.get('latency_ms')} ms); memory_stats "
          f"{stats.get('total_memories')} memories, import_pending={pending} ({elapsed:.1f}s)")
if failed:
    for line in failed:
        print(f"STORE CHECK FAILED — {line}", file=sys.stderr)
    sys.exit(1)
print(f"all {len(STORES)} stores verified: {', '.join(STORES)}")
PY
REMOTE
