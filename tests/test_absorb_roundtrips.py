"""Absorb D1 round-trip reductions: equivalence and request-count bounds.

The batched reads (phase-1 prep, supersession views, retirement lookups,
embedding batches) must answer exactly what the per-row reads they replace
answered. These tests pin both halves: same answers, fewer requests.
"""

import pytest

import memora
import memora.storage as storage


def _mk(conn, text):
    return storage.add_memory(conn, content=text)["id"]


def _legacy(monkeypatch):
    """Force the per-row graph reads (the pre-view code path)."""
    monkeypatch.setattr(storage, "_load_supersession_view", lambda *a, **k: None)


def _graph_answers(conn, ids):
    out = {}
    for mid in ids:
        out[mid] = (
            storage._component_live_leaves(conn, mid),
            storage._resolve_absorb_supersedes_target(conn, mid),
            sorted(storage._get_full_history(conn, mid)),
        )
    return out


def _seed_graphs(conn):
    ids = []
    # chain a <- b <- c
    a, b, c = (_mk(conn, f"chain node {n} extra words") for n in "abc")
    storage.add_link(conn, b, a, edge_type="supersedes")
    storage.add_link(conn, c, b, edge_type="supersedes")
    ids += [a, b, c]
    # fork: orig <- left, orig <- right
    o, l, r = (_mk(conn, f"fork node {n} extra words") for n in ("orig", "left", "right"))
    storage.add_link(conn, l, o, edge_type="supersedes")
    storage.add_link(conn, r, o, edge_type="supersedes")
    ids += [o, l, r]
    # merge: m supersedes both p and q
    p, q, m = (_mk(conn, f"merge node {n} extra words") for n in "pqm")
    storage.add_link(conn, m, p, edge_type="supersedes")
    storage.add_link(conn, m, q, edge_type="supersedes")
    ids += [p, q, m]
    # cycle: x <-> y
    x, y = (_mk(conn, f"cycle node {n} extra words") for n in "xy")
    storage.add_link(conn, x, y, edge_type="supersedes")
    storage.add_link(conn, y, x, edge_type="supersedes")
    ids += [x, y]
    # dangling successor edge to a memory id that does not exist
    d = _mk(conn, "dangling node extra words")
    storage._upsert_crossref_edge(conn, d, 987654, "superseded_by")
    ids += [d]
    # retired member inside a chain
    t1, t2 = (_mk(conn, f"retired chain {n} extra words") for n in (1, 2))
    storage.add_link(conn, t2, t1, edge_type="supersedes")
    storage._retire_members_atomic(conn, [t2], reason="test", known={})
    ids += [t1, t2]
    # singleton with an unrelated related_to edge
    s = _mk(conn, "singleton node extra words")
    storage.add_link(conn, s, a, edge_type="related_to")
    ids += [s]
    conn.commit()
    return ids


@pytest.mark.parametrize("backend", ["local_db", "fake_d1_backend"])
def test_supersession_view_matches_per_row_walk(backend, request, monkeypatch):
    request.getfixturevalue(backend)
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    with storage.connect() as conn:
        ids = _seed_graphs(conn)
        viewed = _graph_answers(conn, ids + [987654])
        _legacy(monkeypatch)
        legacy = _graph_answers(conn, ids + [987654])
    assert viewed == legacy


def test_supersession_view_reads_crossrefs_of_a_deleted_seed(fake_d1_backend):
    with storage.connect() as conn:
        a = _mk(conn, "seed alpha extra words")
        b = _mk(conn, "seed beta extra words")
        storage.add_link(conn, b, a, edge_type="supersedes")
        # Remove only the memory row: the crossref row remains, as it can in
        # the window of a concurrent delete.
        conn.execute("DELETE FROM memories WHERE id = ?", (b,))
        view = storage._load_supersession_view(conn, [b])
        assert view.crossrefs(b) == storage.get_crossrefs(conn, b)
        assert not view.exists(b) and view.exists(a)


def test_supersession_view_bounded_requests_and_fallback(fake_d1_backend, monkeypatch):
    real_loader = storage._load_supersession_view
    with storage.connect() as conn:
        chain = [_mk(conn, f"long chain {i} extra words") for i in range(6)]
        for newer, older in zip(chain[1:], chain):
            storage.add_link(conn, newer, older, edge_type="supersedes")
        before = conn.request_count
        storage._component_live_leaves(conn, chain[-1])
        viewed = conn.request_count - before
        _legacy(monkeypatch)
        before = conn.request_count
        storage._component_live_leaves(conn, chain[-1])
        legacy = conn.request_count - before
    # One query per BFS level (6) + two retirement queries.
    assert viewed == len(chain) + 2
    assert legacy > 4 * viewed
    monkeypatch.setattr(storage, "_load_supersession_view", real_loader)
    monkeypatch.setattr(storage, "_SUPERSESSION_VIEW_MAX_NODES", 3)
    with storage.connect() as conn:
        assert storage._load_supersession_view(conn, [chain[-1]]) is None
        assert storage._component_live_leaves(conn, chain[-1]) == ([chain[-1]], False)


def test_lookup_tombstones_by_hash_batch_matches_single(fake_d1_backend):
    with storage.connect() as conn:
        a = _mk(conn, "tomb alpha extra words")
        b = _mk(conn, "tomb beta extra words")
        # Two component markers with one hash: newest created_at, then
        # highest memory_id, wins — same tie-break as the single lookup.
        h = storage.content_tombstone_hash("Shared   CONTENT")
        conn.execute(
            "INSERT INTO tombstone_components(memory_id, content_hash, reason, created_at) "
            "VALUES (?, ?, ?, ?), (?, ?, ?, ?)",
            (a, h, "older", "2026-01-01 00:00:00", b, h, "newer", "2026-02-01 00:00:00"),
        )
        # A legacy-table-only hash.
        storage._write_tombstone(conn, memory_id=a, content="legacy only", reason="legacy")
        contents = ["shared content", "legacy only", "never deleted"]
        batch = storage._lookup_tombstones_by_hash_batch(conn, contents)
        for c in contents:
            assert batch.get(storage.content_tombstone_hash(c)) == storage._lookup_tombstone_by_hash(conn, c)
        assert batch[h] == "newer"


def test_retired_ids_among_matches_is_tombstoned(fake_d1_backend):
    with storage.connect() as conn:
        ids = [_mk(conn, f"retire probe {i} extra words") for i in range(4)]
        storage._retire_members_atomic(conn, [ids[1]], reason="x", known={})
        storage._write_tombstone(conn, memory_id=ids[2], content="whatever", reason="y")
        assert storage._retired_ids_among(conn, ids) == {
            m for m in ids if storage._is_tombstoned_id(conn, m)
        } == {ids[1], ids[2]}


def test_add_link_existence_checks_are_select_one(fake_d1_backend):
    with storage.connect() as conn:
        a = _mk(conn, "link a extra words")
        b = _mk(conn, "link b extra words")
        seen = []
        real = conn.execute

        def spy(sql, params=None):
            seen.append(sql)
            return real(sql, params)

        conn.execute = spy
        storage.add_link(conn, a, b, edge_type="related_to")
        with pytest.raises(ValueError):
            storage.add_link(conn, a, 424242, edge_type="related_to")
    assert not any(s.lstrip().upper().startswith("SELECT ID, CONTENT") for s in seen)
    assert sum("SELECT 1 FROM memories WHERE id" in s for s in seen) == 4


def _fake_vec(text):
    return {"tok:" + text.split()[0]: 1.0}


def test_phase1_requests_do_not_grow_with_fact_count(fake_d1_backend, monkeypatch):
    monkeypatch.setattr(storage, "_compute_embedding", lambda c, m, t: _fake_vec(c))
    monkeypatch.setattr(storage, "_get_llm_client", lambda: None)
    with storage.connect() as conn:
        for i in range(8):
            storage.add_memory(conn, content=f"topic{i} existing memory body", embedding=_fake_vec(f"topic{i}"))
        one = storage.absorb_memory(conn, ["topic0 a fresh angle"], dry_run=True)
        many = storage.absorb_memory(
            conn, [f"topic{i} a fresh angle" for i in range(8)], dry_run=True,
        )
    p1 = one["profile"]["phases"]["phase1_prep"]["requests"]
    p8 = many["profile"]["phases"]["phase1_prep"]["requests"]
    # tombstone hash (2 tables) + one hydration + retirement (2 tables).
    assert p1 == p8 == 5


def test_absorb_uses_one_embedding_batch_per_phase(fake_d1_backend, monkeypatch):
    monkeypatch.setattr(storage, "EMBEDDING_MODEL", "openai")
    monkeypatch.setattr(storage, "_get_llm_client", lambda: None)
    batches = []

    def fake_batch(entries, model):
        batches.append(len(entries))
        return [_fake_vec(e["content"]) for e in entries]

    monkeypatch.setattr(storage, "_compute_embeddings_batch", fake_batch)
    monkeypatch.setattr(
        storage, "_compute_embedding",
        lambda *a, **k: pytest.fail("per-text embedding on the dense backend"),
    )
    with storage.connect() as conn:
        result = storage.absorb_memory(
            conn, ["alpha one fact", "beta two fact", "gamma three fact"],
        )
    assert result["created"] == 3
    assert batches == [3, 3]
    assert result["profile"]["counters"]["embedding_requests"] == 2


def test_absorb_embedding_batch_failure_falls_back_per_fact(fake_d1_backend, monkeypatch):
    monkeypatch.setattr(storage, "EMBEDDING_MODEL", "openai")
    monkeypatch.setattr(storage, "_get_llm_client", lambda: None)

    def broken_batch(entries, model):
        raise ValueError("batch endpoint hiccup")

    def single(content, meta, tags):
        if content.startswith("bad"):
            raise ValueError("this one input is bad")
        return _fake_vec(content)

    monkeypatch.setattr(storage, "_compute_embeddings_batch", broken_batch)
    monkeypatch.setattr(storage, "_compute_embedding", single)
    with storage.connect() as conn:
        prepared = storage._absorb_phase1_prepare_batch(
            ["good first fact", "bad second fact"], conn, storage.get_corpus_snapshot(conn),
        )
    assert prepared[0]["kind"] == "pending"
    assert prepared[1]["kind"] == "decision"
    assert "embedding/search failed" in prepared[1]["decision"]["reason"]


def test_absorb_strict_embedding_failure_still_propagates(fake_d1_backend, monkeypatch):
    from memora.embeddings import EmbeddingProviderError

    monkeypatch.setattr(storage, "EMBEDDING_MODEL", "openai")

    def broken_batch(entries, model):
        raise EmbeddingProviderError("provider down")

    monkeypatch.setattr(storage, "_compute_embeddings_batch", broken_batch)
    with storage.connect() as conn:
        with pytest.raises(EmbeddingProviderError):
            storage.absorb_memory(conn, ["one fact here", "two fact here"])
