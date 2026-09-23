"""scripts/preview_backfill_47.py: a READ-ONLY preview with an approval list."""

import hashlib
import json
import os

import pytest

import memora
import memora.storage as storage
from scripts import preview_backfill_47 as preview

PROJECTS = ["memora", "clmux", "acebar", "pi"]
CLMUX_TEXT = "clmux TUI sidebar tmux pane workspace switching in the clmux daemon clmuxd"
NEUTRAL_TEXT = "A neutral note about scheduling that names no project at all."


@pytest.fixture
def store(local_db, monkeypatch):
    monkeypatch.setenv("MEMORA_PROJECTS", json.dumps(PROJECTS))
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    return storage.STORAGE_BACKEND.db_path


def _row(conn, content, *, tags, metadata):
    """Insert as the legacy write path left it (section already assigned)."""
    cur = conn.execute("INSERT INTO memories (content, metadata, tags) VALUES (?, ?, ?)",
                       (content, json.dumps(metadata), json.dumps(tags)))
    return cur.lastrowid


def _seed(conn):
    ids = {}
    ids["legacy_issue"] = _row(conn, "**clmux: workspace rename does not work** " + CLMUX_TEXT,
                               tags=["memora/issues"], metadata={"type": "issue", "section": "clmux"})
    ids["conflict"] = _row(conn, "Findings about the socket server. " + CLMUX_TEXT,
                           tags=["memora/issues", "memora/absorb"], metadata={"type": "issue", "section": "clmux"})
    ids["keyword"] = _row(conn, "Observability matters because " + CLMUX_TEXT,
                          tags=["architecture"], metadata={"section": "clmux", "subsection": "architecture"})
    ids["typed_only"] = _row(conn, NEUTRAL_TEXT, tags=["memora/todos"],
                             metadata={"type": "todo", "section": "memora"})
    ids["unknown"] = _row(conn, NEUTRAL_TEXT + " Second.", tags=[], metadata={"section": "memora"})
    ids["explicit"] = _row(conn, CLMUX_TEXT, tags=["clmux/tui"], metadata={"section": "clmux"})
    conn.commit()
    return ids


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preview_groups_proposals_and_needs_human(store, tmp_path):
    with storage.connect() as conn:
        ids = _seed(conn)
    outdir = tmp_path / "out"
    outdir.mkdir()
    before = (_digest(store), sorted(os.listdir(store.parent)))
    out = outdir / "preview.json"
    md = outdir / "preview.md"
    assert preview.main(["--out", str(out), "--markdown", str(md)]) == 0
    # Nothing written, nothing created next to the store.
    assert (_digest(store), sorted(os.listdir(store.parent))) == before

    data = json.loads(out.read_text())
    rows = {r["id"]: r for group in ("contradictions", "keyword_only") for r in data[group]}
    assert all(r["approved"] is False for r in rows.values())
    assert ids["explicit"] not in rows  # already consistent: not listed

    legacy = rows[ids["legacy_issue"]]
    assert legacy in data["contradictions"] and legacy["status"] == "proposed"
    assert legacy["proposal"]["set_metadata_project"] == "clmux"
    assert legacy["proposal"]["retag_typed"] == {"memora/issues": "clmux/issues"}

    conflict = rows[ids["conflict"]]
    assert conflict in data["contradictions"] and conflict["status"] == "needs-human"
    assert "disagrees" in conflict["reason"]

    keyword = rows[ids["keyword"]]
    assert keyword in data["keyword_only"] and keyword["status"] == "proposed"
    assert keyword["evidence"] == [{"source": "content keywords", "project": "clmux"}]
    assert keyword["proposal"]["set_metadata_project"] == "clmux"
    assert keyword["proposal"]["alternative_marker_tag"] == "clmux/architecture"

    typed_only = rows[ids["typed_only"]]
    assert typed_only in data["keyword_only"] and typed_only["status"] == "needs-human"
    assert "typed tag" in typed_only["reason"]

    unknown = rows[ids["unknown"]]
    assert unknown["status"] == "needs-human" and "no non-typed source" in unknown["reason"]

    s = data["summary"]
    assert s["contradictions"] == {"total": 2, "proposed": 1, "needs_human": 1}
    assert s["keyword_only"] == {"total": 3, "proposed": 1, "needs_human": 2}
    assert "| id | status |" in md.read_text()


def test_preview_skips_import_pending_rows(store, tmp_path):
    with storage.connect() as conn:
        _row(conn, CLMUX_TEXT, tags=["memora/issues"],
             metadata={"type": "issue", "section": "clmux", "import_attempt": "x:1:0"})
        conn.commit()
    out = tmp_path / "p.json"
    preview.main(["--out", str(out)])
    data = json.loads(out.read_text())
    assert data["contradictions"] == [] and data["keyword_only"] == []
