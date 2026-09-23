#!/usr/bin/env python3
"""Count D1 requests per memora READ tool, offline.

Never touches a live store: a scratch SQLite file behind the FakeD1Connection
double (tests/conftest.py), where one statement == one HTTPS POST, as on
Cloudflare D1. Calls the real MCP tool coroutines in memora.server, so the
connection wrapper, follow defaults and response shaping are all included.

Store: --rows background memories (default 964, the live size) with real
tfidf embeddings, three supersession chains (3, 5, 3 versions), a retired
chain, related_to crossrefs on every row, and a few memories with an empty
stored crossref list.

For each tool the first call after a write is "cold" (caches invalid) and the
second is "warm". Prints requests per call; with --d1-latency the seconds
are modeled from real sleeps per statement.

Usage: ./.venv/bin/python scripts/measure_read_roundtrips.py [--rows 964] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MEMORA_VECTOR_SCAN_PAGE_SIZE", "100")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memora  # noqa: E402
import memora.storage as storage  # noqa: E402
from memora import server  # noqa: E402
from tests.conftest import FakeD1Backend, FakeD1Connection  # noqa: E402


class LatencyFakeD1Connection(FakeD1Connection):
    latency = 0.0

    def execute(self, sql, params=None):
        cur = super().execute(sql, params)
        if self.latency and not self._is_savepoint(sql):
            time.sleep(self.latency)
        return cur


class CountingBackend(FakeD1Backend):
    def connect(self, *, check_same_thread: bool = True):
        conn = LatencyFakeD1Connection(self.db_path, transactional=self.transactional)
        self.connections.append(conn)
        return conn

    def total(self) -> int:
        return sum(c.statement_count for c in self.connections)


WORDS = ("deploy proxy cache absorb sidebar watchdog embedding graph backup "
         "tombstone lease daemon pane socket health router session").split()


def _text(i: int) -> str:
    w = [WORDS[(i * k) % len(WORDS)] for k in (1, 3, 5, 7)]
    return f"note {i} about the {w[0]} {w[1]} and {w[2]} for subsystem {i % 37} ({w[3]})"


def seed(conn, rows: int) -> dict:
    ids = []
    for i in range(rows):
        cur = conn.execute(
            "INSERT INTO memories (content, metadata, tags, created_at) VALUES (?, ?, ?, ?)",
            (_text(i), json.dumps({"type": "note"}) if i % 5 else None,
             json.dumps([f"memora/{WORDS[i % len(WORDS)]}"]), f"2026-09-{1 + i % 28:02d} 00:00:00"),
        )
        mid = int(cur.lastrowid)
        ids.append(mid)
        storage._upsert_embedding(conn, mid, storage._compute_embedding(_text(i), None, []))
    # related_to crossrefs: each row points at its two neighbours, except a
    # few rows that carry a legitimately EMPTY stored list.
    empty = set(ids[10:13])
    rows_x = []
    for n, mid in enumerate(ids):
        rel = [] if mid in empty else [
            {"id": ids[(n + 1) % len(ids)], "score": 0.5, "edge_type": "related_to"},
            {"id": ids[(n - 1) % len(ids)], "score": 0.4, "edge_type": "related_to"},
        ]
        rows_x.append((mid, rel))
    storage._store_crossrefs_bulk(conn, rows_x)
    chains = {}
    for name, length in (("chainA", 3), ("chainB", 5), ("chainC", 3), ("retired", 2)):
        cids = []
        for v in range(length):
            content = f"{name} deploy proxy setting version {v}"
            cur = conn.execute(
                "INSERT INTO memories (content, metadata, tags, created_at) VALUES (?, ?, ?, ?)",
                (content, None, json.dumps(["memora/deploy"]), f"2026-09-20 0{v}:00:00"),
            )
            mid = int(cur.lastrowid)
            storage._upsert_embedding(conn, mid, storage._compute_embedding(content, None, []))
            if cids:
                storage.add_link(conn, mid, cids[-1], edge_type="supersedes", commit=False)
            cids.append(mid)
        chains[name] = cids
    storage._retire_members_atomic(conn, chains["retired"], reason="bench", known={})
    conn.commit()
    return {"ids": ids, "chains": chains, "empty_related": sorted(empty)}


def run(backend: CountingBackend, coro_fn):
    before = backend.total()
    t0 = time.perf_counter()
    result = asyncio.run(coro_fn())
    return backend.total() - before, time.perf_counter() - t0, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=964)
    ap.add_argument("--d1-latency", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="memora-reads-"))
    backend = CountingBackend(tmp / "reads.db")
    storage.STORAGE_BACKEND = backend
    storage.EMBEDDING_MODEL = "tfidf"
    memora.TAG_WHITELIST = set()
    with storage.connect() as conn:
        info = seed(conn, args.rows)
        storage.rebuild_embeddings(conn)  # stamps integrity + model; not measured

    chains = info["chains"]
    stale = chains["chainB"][1]
    leaf = chains["chainB"][-1]
    plain = info["ids"][500]
    empty_rel = info["empty_related"][0]

    calls = [
        ("memory_semantic_search", lambda: server.memory_semantic_search("deploy proxy cache", top_k=5)),
        ("memory_semantic_search metadata filter", lambda: server.memory_semantic_search(
            "deploy proxy", top_k=5, metadata_filters={"type": "note"})),
        ("memory_hybrid_search", lambda: server.memory_hybrid_search("deploy proxy cache", limit=5)),
        ("memory_hybrid_search tags+dates", lambda: server.memory_hybrid_search(
            "deploy proxy", limit=5, tags_any=["memora/deploy"], date_from="2026-09-05")),
        ("memory_get (current)", lambda: server.memory_get(plain)),
        ("memory_get (stale -> leaf)", lambda: server.memory_get(stale)),
        ("memory_get full_history", lambda: server.memory_get(leaf, follow="full_history")),
        ("memory_list limit=20", lambda: server.memory_list(limit=20)),
        ("memory_list_compact limit=20", lambda: server.memory_list_compact(limit=20)),
        ("memory_related (stored)", lambda: server.memory_related(plain)),
        ("memory_related (empty stored list)", lambda: server.memory_related(empty_rel)),
    ]

    LatencyFakeD1Connection.latency = args.d1_latency
    rows = []
    # Cold = right after a write invalidates caches; warm = the next call.
    for name, fn in calls:
        with storage.connect() as conn:  # a write: bumps the epoch
            conn.execute("UPDATE memories SET importance = importance WHERE id = ?", (info["ids"][0],))
            conn.commit()
        cold_req, cold_s, cold_res = run(backend, fn)
        warm_req, warm_s, warm_res = run(backend, fn)
        if isinstance(warm_res, dict) and warm_res.get("error"):
            raise SystemExit(f"{name}: {warm_res}")
        rows.append({"tool": name, "cold_requests": cold_req, "warm_requests": warm_req,
                     "cold_seconds": round(cold_s, 3), "warm_seconds": round(warm_s, 3)})
    LatencyFakeD1Connection.latency = 0.0

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"rows={args.rows} d1_latency={args.d1_latency}s")
    print(f"{'tool':<38}{'cold req':>9}{'warm req':>9}{'cold s':>8}{'warm s':>8}")
    for r in rows:
        print(f"{r['tool']:<38}{r['cold_requests']:>9}{r['warm_requests']:>9}"
              f"{r['cold_seconds']:>8.2f}{r['warm_seconds']:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
