#!/usr/bin/env python3
"""Live smoke test of a running memora /api/v1 (the Phase gates).

Sends health, a search, and an absorb with one idempotency key TWICE to a
running server, and validates every response against the contract schemas
(contracts/memora-api/v1). The absorb goes to --store, so point it at a
TEST store; on a store reporting writes "unsupported" it must be 501 both
times and write nothing, and on a "transactional" one (Phase L) it must be
200 both times with identical bodies (the replay path).

The token file holds the raw bearer token (the client side of
MEMORA_API_TOKENS_FILE, which holds its sha256), and is opened with the same
checks as memora's own tokens file (O_NOFOLLOW, owner, mode, parents).

Usage:
  python scripts/memora_api_smoke.py --base http://127.0.0.1:8920 --store test \\
      --token-file ~/.config/clmux/memora.token [--project clmux] [--query "..."]
Exit status 0 when every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memora_api_contract import schema_errors  # noqa: E402
from memora.api_v1 import read_token_file  # noqa: E402


def _call(base, method, path, token, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        return exc.code, json.loads(raw) if raw else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--project", default="memora")
    ap.add_argument("--query", default="memora api smoke test")
    args = ap.parse_args(argv)

    # NOT resolved: read_token_file must see the path as given (its final
    # component opened O_NOFOLLOW, every parent lstat-checked); resolve()
    # would follow a symlink first and defeat those checks.
    token = read_token_file(os.path.abspath(os.path.expanduser(args.token_file))).decode().strip()
    problems = []

    def check(name, status, body, allowed, schema_for):
        schema = schema_for(status)
        errs = schema_errors(schema, body)
        if status not in allowed or errs:
            problems.append(f"{name}: status {status} (allowed {sorted(allowed)}), schema {schema}: {errs}")
        print(f"{name}: {status} {json.dumps(body)[:200]}")

    status, health = _call(args.base, "GET", f"/api/v1/{args.store}/health", token)
    check("health", status, health, {200, 503},
          lambda s: "health_response.json" if s in (200, 503) else "error_response.json")
    writes = (health or {}).get("writes") if status == 200 else None

    t0 = time.time()
    status, search = _call(args.base, "POST", f"/api/v1/{args.store}/search", token,
                           {"query": args.query, "top_k": 3})
    check("search", status, search, {200},
          lambda s: "search_response.json" if s == 200 else "error_response.json")
    print(f"search took {time.time() - t0:.2f}s")

    key = f"smoke:{uuid.uuid4().hex}"
    request = {"idempotency_key": key, "project": args.project, "source": "smoke",
               "facts": [f"memora api smoke test fact {key}"]}
    first = _call(args.base, "POST", f"/api/v1/{args.store}/absorb", token, request)
    second = _call(args.base, "POST", f"/api/v1/{args.store}/absorb", token, request)
    for label, (s, b) in (("absorb#1", first), ("absorb#2", second)):
        check(label, s, b, {200, 501},
              lambda st: "absorb_response.json" if st == 200 else "error_response.json")
    if writes == "transactional":
        if first[0] != 200 or first != second:
            problems.append("absorb replay: a transactional store must answer 200 twice with the same body")
    else:
        if first[0] != 501 or second[0] != 501 or (first[1] or {}).get("error") != "writes_unsupported":
            problems.append("absorb on a non-transactional store must be 501 writes_unsupported both times")

    for p in problems:
        print(f"FAIL {p}")
    print("smoke: ok" if not problems else f"smoke: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
