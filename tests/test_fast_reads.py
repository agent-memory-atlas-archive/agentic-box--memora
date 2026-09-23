"""Fast read paths must return exactly what the old read paths returned.

Every test runs on local SQLite and on the FakeD1 double, and compares the
new path with the pre-change reference: _search_by_vector_scan (the old
full-row scan), _get_memory_legacy, _follow_status_legacy, and a verbatim
copy of the old apply_follow (tests/legacy_reads.py). importance_score is
time-dependent, so it is frozen for the comparisons.
"""

import http.server
import json
import threading

import pytest

import memora
import memora.storage as storage
from memora.backends import D1Connection
from tests.legacy_reads import legacy_apply_follow


@pytest.fixture(params=["local_db", "fake_d1_backend"])
def db(request, monkeypatch):
    request.getfixturevalue(request.param)
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    monkeypatch.setattr(storage, "calculate_importance", lambda *a, **k: 1.0)
    storage._corpus_cache.clear()
    yield request.param
    storage._corpus_cache.clear()


def _raw_insert(conn, content, *, metadata=None, tags=(), created="2026-09-10 00:00:00", embed=True):
    cur = conn.execute(
        "INSERT INTO memories (content, metadata, tags, created_at) VALUES (?, ?, ?, ?)",
        (content, json.dumps(metadata) if metadata is not None else None, json.dumps(list(tags)), created),
    )
    mid = int(cur.lastrowid)
    if embed:
        storage._upsert_embedding(conn, mid, storage._compute_embedding(content, None, []))
    return mid


def _seed_search_store(conn):
    ids = {}
    words = "deploy proxy cache watchdog sidebar absorb graph".split()
    for i in range(60):
        ids[f"m{i}"] = _raw_insert(
            conn,
            f"{words[i % 7]} {words[(i * 3) % 7]} note {i % 9}",
            metadata=({"section": "ops", "type": "note"} if i % 3 == 0 else
                      {"hierarchy": {"path": ["a", "b"]}} if i % 3 == 1 else None),
            tags=[f"memora/{words[i % 7]}"] + (["memora/extra"] if i % 4 == 0 else []),
            created=f"2026-09-{1 + i % 20:02d} 00:00:00",
        )
    # Exact ties: identical content AND created_at, different ids.
    ids["tie1"] = _raw_insert(conn, "deploy proxy tie", created="2026-09-15 00:00:00")
    ids["tie2"] = _raw_insert(conn, "deploy proxy tie", created="2026-09-15 00:00:00")
    # A legacy row with no embedding (repaired / backfilled on first search).
    ids["missing"] = _raw_insert(conn, "deploy proxy legacy row", embed=False)
    conn.commit()
    return ids


SEARCH_CASES = [
    dict(),
    dict(top_k=None),
    dict(top_k=3, min_score=0.2),
    dict(metadata_filters={"section": "ops"}),
    dict(metadata_filters={"hierarchy_path": ["a"]}),
    dict(tags_any=["memora/extra"], tags_none=["memora/cache"]),
    dict(tags_all=["memora/deploy", "memora/extra"]),
    dict(date_from="2026-09-05", date_to="2026-09-12"),
    dict(exclude_ids="first-two"),
]


@pytest.mark.parametrize("case", SEARCH_CASES, ids=lambda c: ",".join(c) or "plain")
def test_snapshot_search_equals_full_scan(db, case):
    with storage.connect() as conn:
        ids = _seed_search_store(conn)
        q = storage._compute_embedding("deploy proxy", None, [])
        kwargs = dict(case)
        kwargs.setdefault("top_k", 10)
        if kwargs.get("exclude_ids") == "first-two":
            kwargs["exclude_ids"] = [ids["m0"], ids["m1"]]
        scan = storage._search_by_vector_scan(conn, q, **kwargs)  # backfills "missing"
        storage._corpus_cache.clear()
        fast = storage._search_by_vector(conn, q, **kwargs)
    assert fast == scan
    assert fast, "case must return something to be meaningful"


def test_snapshot_repair_replaces_inline_backfill(db):
    with storage.connect() as conn:
        ids = _seed_search_store(conn)
        q = storage._compute_embedding("deploy proxy legacy row", None, [])
        fast = storage._search_by_vector(conn, q, top_k=1)
        assert fast[0]["memory"]["id"] == ids["missing"]
        # The repair stored the vector, as the scan's backfill would have.
        assert storage._get_embeddings_for_ids(conn, [ids["missing"]])


def test_semantic_search_warm_call_is_cheap_and_identical(fake_d1_backend, monkeypatch):
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    monkeypatch.setattr(storage, "calculate_importance", lambda *a, **k: 1.0)
    with storage.connect() as conn:
        _seed_search_store(conn)
        storage.rebuild_embeddings(conn)
    with storage.connect() as conn:
        first = storage.semantic_search(conn, "deploy proxy", top_k=5)
        before = conn.request_count
        second = storage.semantic_search(conn, "deploy proxy", top_k=5)
        warm = conn.request_count - before
    assert first == second
    assert warm <= 3  # meta + hydrate (+ nothing else when no follow)


# --- follow ---------------------------------------------------------------

def _seed_graph(conn, *, malformed_refs=True):
    g = {}
    mk = lambda name: g.setdefault(name, _raw_insert(conn, f"graph {name} deploy"))  # noqa: E731
    for n in "abc":
        mk(n)
    storage.add_link(conn, g["b"], g["a"], edge_type="supersedes")
    storage.add_link(conn, g["c"], g["b"], edge_type="supersedes")
    for n in ("orig", "left", "right"):
        mk(n)
    storage.add_link(conn, g["left"], g["orig"], edge_type="supersedes")
    storage.add_link(conn, g["right"], g["orig"], edge_type="supersedes")
    for n in "xy":
        mk(n)
    storage.add_link(conn, g["x"], g["y"], edge_type="supersedes")
    storage.add_link(conn, g["y"], g["x"], edge_type="supersedes")
    for n in ("t1", "t2"):
        mk(n)
    storage.add_link(conn, g["t2"], g["t1"], edge_type="supersedes")
    storage._retire_members_atomic(conn, [g["t2"]], reason="t", known={})
    mk("legacy_tomb")
    storage._write_tombstone(conn, memory_id=g["legacy_tomb"], content="zz", reason="r")
    mk("plain")
    if not malformed_refs:
        conn.commit()
        return g
    # Crossref blobs the Python parser has to tolerate.
    odd = {
        "dangling": [{"id": 987654, "edge_type": "superseded_by"}],
        "string_id": [{"id": str(g["a"]), "edge_type": "superseded_by"}],
        "float_id": [{"id": float(g["a"]), "edge_type": "superseded_by"}],
        "non_dict": ["oops", 3, {"edge_type": "superseded_by"}],
        # JSON true: the legacy Python path treats it as memory id 1.
        "bool_id": [{"id": True, "edge_type": "superseded_by"}],
        "self_loop": None,
    }
    for name, blob in odd.items():
        mk(name)
        if blob is None:
            blob = [{"id": g[name], "edge_type": "superseded_by"}]
        storage._store_crossrefs(conn, g[name], blob)
    mk("malformed")
    conn.execute("INSERT OR REPLACE INTO memories_crossrefs(memory_id, related) VALUES (?, ?)",
                 (g["malformed"], "{not json"))
    mk("object_blob")
    conn.execute("INSERT OR REPLACE INTO memories_crossrefs(memory_id, related) VALUES (?, ?)",
                 (g["object_blob"], json.dumps({"id": g["a"], "edge_type": "superseded_by"})))
    conn.commit()
    return g


def test_follow_status_equals_legacy(db):
    with storage.connect() as conn:
        g = _seed_graph(conn)
        ids = list(g.values()) + [424242]
        assert storage._follow_status(conn, ids) == storage._follow_status_legacy(conn, ids)
        sup, ret, unsafe = storage._follow_status(conn, ids)
        assert g["a"] in sup and g["float_id"] in sup and g["string_id"] not in sup
        assert {g["t2"], g["legacy_tomb"]} <= ret
        assert unsafe == {g["string_id"], g["float_id"], g["non_dict"], g["bool_id"]}
        assert g["a"] == 1 and g["bool_id"] in sup  # True counted as #1, as legacy did


@pytest.mark.parametrize("is_search", [False, True])
def test_follow_active_with_malformed_refs_equals_legacy(db, is_search):
    with storage.connect() as conn:
        _seed_graph(conn)
        mems = storage.list_memories(conn, limit=-1)
        items = [{"score": 0.5, "memory": m} for m in mems] if is_search else mems
        assert storage.apply_follow(conn, [dict(i) for i in items], "active", is_search=is_search) == \
            legacy_apply_follow(conn, [dict(i) for i in items], "active", is_search=is_search)


def test_follow_status_one_statement_for_any_size(fake_d1_backend):
    with storage.connect() as conn:
        ids = [_raw_insert(conn, f"row {i}", embed=False) for i in range(250)]
        before = conn.request_count
        storage._follow_status(conn, ids)
        assert conn.request_count - before == 1


def test_follow_status_falls_back_when_a_tombstone_table_is_missing(db, caplog):
    with storage.connect() as conn:
        g = _seed_graph(conn)
        conn.execute("DROP TABLE tombstones")
        ids = list(g.values())
        assert storage._follow_status(conn, ids) == storage._follow_status_legacy(conn, ids)


@pytest.mark.parametrize("follow", ["active", "latest", "full_history"])
@pytest.mark.parametrize("is_search", [False, True])
def test_apply_follow_equals_legacy(db, follow, is_search):
    with storage.connect() as conn:
        g = _seed_graph(conn, malformed_refs=False)
        mems = storage.list_memories(conn, limit=-1)
        items = [{"score": 0.5, "memory": m} for m in mems] if is_search else mems
        new = storage.apply_follow(conn, [dict(i) for i in items], follow, is_search=is_search)
        old = legacy_apply_follow(conn, [dict(i) for i in items], follow, is_search=is_search)
    assert new == old


def _outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # the legacy walks crash on some malformed blobs
        return ("raised", type(exc).__name__)


@pytest.mark.parametrize("follow", ["latest", "full_history"])
def test_malformed_refs_behave_exactly_as_before(db, follow):
    """String/float ids and non-dict entries: the view declines and the
    per-row reads run, so results AND failures match the old code."""
    with storage.connect() as conn:
        g = _seed_graph(conn)
        for name in ("string_id", "float_id", "non_dict", "self_loop", "malformed", "object_blob"):
            mid = g[name]
            assert _outcome(lambda: storage.get_memory(conn, mid, follow=follow)) == \
                _outcome(lambda: storage._get_memory_legacy(conn, mid, follow=follow)), name
            mem = storage._get_memory_legacy(conn, mid)
            assert _outcome(lambda: storage.apply_follow(conn, [dict(mem)], follow)) == \
                _outcome(lambda: legacy_apply_follow(conn, [dict(mem)], follow)), name


# --- memory_get -----------------------------------------------------------

@pytest.mark.parametrize("follow", [None, "latest", "full_history"])
def test_get_memory_equals_legacy(db, follow):
    with storage.connect() as conn:
        g = _seed_graph(conn, malformed_refs=False)
        # A deleted id whose crossref row lingers: the legacy edge case.
        gone = _raw_insert(conn, "gone soon")
        storage._store_crossrefs(conn, gone, [{"id": g["c"], "edge_type": "superseded_by"}])
        conn.execute("DELETE FROM memories WHERE id = ?", (gone,))
        conn.commit()
        for mid in [*g.values(), gone, 424242]:
            assert storage.get_memory(conn, mid, follow=follow) == \
                storage._get_memory_legacy(conn, mid, follow=follow), (mid, follow)


def test_get_memory_current_is_one_statement(fake_d1_backend):
    with storage.connect() as conn:
        g = _seed_graph(conn)
        before = conn.request_count
        rec = storage.get_memory(conn, g["plain"], follow="latest")
        assert conn.request_count - before == 1
        assert rec["id"] == g["plain"]


# --- memory_related -------------------------------------------------------

def test_related_empty_stored_list_is_an_answer(db):
    with storage.connect() as conn:
        a = _raw_insert(conn, "alpha related")
        _raw_insert(conn, "beta related")
        storage._store_crossrefs(conn, a, [])  # row exists, list empty
        # _store_crossrefs stores NULL for an empty list; the row still exists.
        assert storage.get_related(conn, a) == []
        exists, _raw, _refs = storage._load_crossrefs_raw(conn, a)
        assert exists


def test_related_missing_row_is_computed_and_refresh_equals_scan(db):
    with storage.connect() as conn:
        ids = _seed_search_store(conn)
        target = ids["m5"]
        conn.execute("DELETE FROM memories_crossrefs WHERE memory_id = ?", (target,))
        computed = storage.get_related(conn, target)
        assert computed
        # The corpus-scored refresh stores what the old full-scan pass stored.
        storage._update_crossrefs_for_memory(conn, target)
        scanned = storage.get_crossrefs(conn, target)
        assert storage.get_related(conn, target, refresh=True) == scanned == computed


# --- query embedding cache -----------------------------------------------

def test_query_embedding_cache(monkeypatch):
    calls = []

    def embed(content, meta, tags):
        calls.append(content)
        return {} if content == "empty" else {"x": 1.0}

    monkeypatch.setattr(storage, "_compute_embedding", embed)
    assert storage._query_embedding("q") == storage._query_embedding("q")
    assert calls == ["q"]
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "another-model")
    storage._query_embedding("q")
    assert calls == ["q", "q"]  # model change -> miss
    storage._query_embedding("empty")
    storage._query_embedding("empty")
    assert calls.count("empty") == 2  # empty results are never cached


# --- D1 persistent transport ----------------------------------------------

class _FakeD1Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fake-d1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        srv = self.server
        srv.requests.append((self.client_address, body["sql"]))
        if srv.drop_next:
            srv.drop_next = False
            self.close_connection = True
            self.connection.close()
            return
        if srv.truncate_next:
            # Status line and headers arrive, then the body is cut short.
            srv.truncate_next = False
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "500")
            self.end_headers()
            self.wfile.write(b'{"success": tr')
            self.wfile.flush()
            self.close_connection = True
            self.connection.shutdown(2)
            return
        if body["sql"].startswith("BAD"):
            payload, status = b'{"errors":[{"message":"nope"}]}', 400
        else:
            payload = json.dumps({"success": True, "result": [
                {"results": [{"n": 1}], "meta": {"last_row_id": 7, "changes": 1}}]}).encode()
            status = 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("cf-d1-session-token", f"bm-{len(srv.requests):04d}")
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def fake_d1_http(monkeypatch):
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeD1Handler)
    srv.requests, srv.drop_next, srv.truncate_next = [], False, False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    conn = D1Connection("acct", "db", "token")
    conn.base_url = f"http://127.0.0.1:{srv.server_address[1]}/client/v4/accounts/acct/d1/database/db"
    yield conn, srv
    conn.close()
    srv.shutdown()


def test_d1_transport_reuses_one_connection(fake_d1_http):
    conn, srv = fake_d1_http
    for _ in range(3):
        assert conn.execute("SELECT 1").fetchone() is not None
    assert len({addr for addr, _ in srv.requests}) == 1  # one TCP connection
    assert conn._session_token == "bm-0003"


def test_d1_transport_retries_a_stale_read_once_never_a_write(fake_d1_http):
    conn, srv = fake_d1_http
    conn.execute("SELECT 1")
    srv.drop_next = True  # server closes the kept-alive socket without answering
    assert conn.execute("SELECT 2").fetchone() is not None
    assert [sql for _, sql in srv.requests].count("SELECT 2") == 2
    srv.drop_next = True
    with pytest.raises(Exception):
        conn.execute("INSERT INTO t VALUES (1)")
    assert [sql for _, sql in srv.requests].count("INSERT INTO t VALUES (1)") == 1


def test_d1_transport_http_error_message_unchanged(fake_d1_http):
    conn, _srv = fake_d1_http
    with pytest.raises(RuntimeError, match=r"D1 API error \(400\)"):
        conn.execute("BAD SQL")


# --- read tool profiles ---------------------------------------------------

def test_read_tools_return_a_profile(fake_d1_backend, monkeypatch):
    import asyncio
    from memora import server

    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    with storage.connect() as conn:
        ids = _seed_search_store(conn)
        storage.rebuild_embeddings(conn)
    for resp in (
        asyncio.run(server.memory_semantic_search("deploy proxy", top_k=3)),
        asyncio.run(server.memory_get(ids["m1"])),
        asyncio.run(server.memory_list(limit=5)),
        asyncio.run(server.memory_list_compact(limit=5)),
        asyncio.run(server.memory_related(ids["m1"])),
        asyncio.run(server.memory_hybrid_search("deploy proxy", limit=3)),
    ):
        prof = resp["profile"]
        assert prof["request_unit"] == "d1_requests" and prof["total_requests"] >= 1
        assert "seconds" in next(iter(prof["phases"].values()))


@pytest.mark.parametrize("follow", [None, "latest", "full_history"])
def test_get_memory_track_access_equals_legacy(tmp_path, monkeypatch, follow):
    """track_access writes, so compare on two identically seeded stores."""
    from memora.backends import LocalSQLiteBackend

    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    monkeypatch.setattr(storage, "EMBEDDING_MODEL", "tfidf")
    monkeypatch.setattr(storage, "calculate_importance", lambda *a, **k: 1.0)
    outcomes = []
    for name, fn in (("new", storage.get_memory), ("old", storage._get_memory_legacy)):
        monkeypatch.setattr(storage, "STORAGE_BACKEND", LocalSQLiteBackend(tmp_path / f"{name}-{follow}.db"))
        with storage.connect() as conn:
            g = _seed_graph(conn, malformed_refs=False)
            got = [fn(conn, g[k], track_access=True, follow=follow) for k in ("a", "c", "orig", "plain")]
            counts = [conn.execute("SELECT access_count FROM memories WHERE id = ?", (g[k],)).fetchone()[0]
                      for k in ("a", "b", "c", "orig", "left", "right", "plain")]
            outcomes.append((got, counts))
    for got in outcomes[0][0] + outcomes[1][0]:
        if got:
            got.pop("last_accessed", None)
            for h in got.get("history", []):
                h.pop("last_accessed", None)
    assert outcomes[0] == outcomes[1]


def test_follow_latest_past_view_bounds_equals_legacy(db, monkeypatch):
    monkeypatch.setattr(storage, "_SUPERSESSION_VIEW_MAX_NODES", 2)
    with storage.connect() as conn:
        _seed_graph(conn, malformed_refs=False)
        mems = storage.list_memories(conn, limit=-1)
        assert storage.apply_follow(conn, [dict(m) for m in mems], "latest") == \
            legacy_apply_follow(conn, [dict(m) for m in mems], "latest")
        for m in mems:
            for follow in ("latest", "full_history"):
                assert storage.get_memory(conn, m["id"], follow=follow) == \
                    storage._get_memory_legacy(conn, m["id"], follow=follow)


def test_cold_search_repairs_more_than_100_missing_embeddings(fake_d1_backend, monkeypatch):
    """FakeD1 enforces D1's 100-bound-parameter cap (conftest)."""
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    with storage.connect() as conn:
        ids = [_raw_insert(conn, f"deploy note {i}", embed=False) for i in range(101)]
        conn.commit()
        storage._corpus_cache.clear()
        q = storage._compute_embedding("deploy note", None, [])
        out = storage._search_by_vector(conn, q, top_k=None)
        assert {r["memory"]["id"] for r in out} == set(ids)
        assert len(storage._get_embeddings_for_ids(conn, ids)) == 101


def test_malformed_tags_blob_does_not_abort_search_or_absorb(db, caplog, monkeypatch):
    monkeypatch.setattr(storage, "_bad_tags_warned", set())  # warn-once is per process
    with storage.connect() as conn:
        _seed_search_store(conn)
        bad = _raw_insert(conn, "zzz unrelated bad tags row")
        conn.execute("UPDATE memories SET tags = ? WHERE id = ?", ("{not json", bad))
        conn.commit()
        storage._corpus_cache.clear()
        q = storage._compute_embedding("deploy proxy", None, [])
        # A tag filter treats the bad row as untagged instead of raising.
        out = storage._search_by_vector(conn, q, top_k=5, tags_none=["memora/deploy"])
        assert out and bad not in {r["memory"]["id"] for r in out}
        assert "unparseable tags JSON" in caplog.text
        storage._corpus_cache.clear()
        result = storage.absorb_memory(conn, ["a brand new fact about lighthouses"])
        assert result["created"] == 1


@pytest.mark.parametrize("min_score", [None, 0.0])
def test_fresh_empty_repair_scores_zero_once_like_the_old_backfill(tmp_path, monkeypatch, min_score):
    """A missing embedding that repairs to an EMPTY vector (punctuation-only
    content): the old inline backfill scored it 0 on that call, then it was
    certified empty and skipped. Compared on two identical stores, since
    both paths write the repair."""
    from memora.backends import LocalSQLiteBackend

    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    monkeypatch.setattr(storage, "EMBEDDING_MODEL", "tfidf")
    monkeypatch.setattr(storage, "calculate_importance", lambda *a, **k: 1.0)
    runs = {}
    for name in ("new", "old"):
        monkeypatch.setattr(storage, "STORAGE_BACKEND", LocalSQLiteBackend(tmp_path / f"{name}.db"))
        storage._corpus_cache.clear()
        with storage.connect() as conn:
            _raw_insert(conn, "deploy proxy one")
            empty_id = _raw_insert(conn, "!!! ...", embed=False)
            _raw_insert(conn, "cache sidebar two")
            conn.commit()
            q = storage._compute_embedding("deploy proxy", None, [])
            fn = storage._search_by_vector if name == "new" else storage._search_by_vector_scan
            first = fn(conn, q, top_k=None, min_score=min_score)
            second = fn(conn, q, top_k=None, min_score=min_score)
            runs[name] = (first, second)
    assert runs["new"] == runs["old"]
    first, second = runs["new"]
    assert empty_id in {r["memory"]["id"] for r in first}
    assert [r["score"] for r in first if r["memory"]["id"] == empty_id] == [0.0]
    assert empty_id not in {r["memory"]["id"] for r in second}


def test_d1_transport_never_retries_after_response_bytes(fake_d1_http):
    """Headers arrived, then the connection died mid-body: even a SELECT on
    a reused socket is NOT re-sent (it could observe a newer state)."""
    conn, srv = fake_d1_http
    conn.execute("SELECT 1")  # socket now reused
    srv.truncate_next = True
    with pytest.raises(Exception):
        conn.execute("SELECT 3")
    assert [sql for _, sql in srv.requests].count("SELECT 3") == 1
    # The transport recovers on the next call with a fresh socket.
    assert conn.execute("SELECT 4").fetchone() is not None
