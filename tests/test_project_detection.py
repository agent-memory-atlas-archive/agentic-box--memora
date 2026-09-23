"""Issue #47: a memory's project comes from explicit markers, never keywords.

Resolution order: an explicit `project` argument, else metadata.project, else
exactly one tag naming a project configured for the store (MEMORA_PROJECTS),
else no project. Section/subsection assignment and generic-tag prefixing act
only on a resolved project; LLM-suggested tags are filtered by the configured
tag policy, not a hardcoded memora/clmux prefix list.
"""

import asyncio
import io
import json
import sys

import pytest

import memora
import memora.storage as storage

# Modelled on memora #1082 (parked Claude Mods design) and #1109 (pi channel
# work): both mention clmux vocabulary, so the old detector put both in clmux.
PARKED_DESIGN = (
    "Claude Mods design idea (parked, not started): a mod loader that lets users "
    "bundle prompt snippets, hooks and statusline widgets. Open question: whether "
    "mods ship through clmux agent delivery or a separate registry."
)
PI_CHANNEL_WORK = (
    "pi channel work: the pi agent now receives inbox doorbells over the clmux "
    "agent delivery channel instead of pane injection."
)
GENERIC_TECH = (
    "The daemon keeps one workspace per socket and the sidebar shows embedding "
    "latency for the absorb pipeline."
)


@pytest.fixture
def projects(monkeypatch):
    def set_projects(value):
        if value is None:
            monkeypatch.delenv("MEMORA_PROJECTS", raising=False)
        else:
            monkeypatch.setenv("MEMORA_PROJECTS", json.dumps(value))
    return set_projects


@pytest.fixture(params=["local_db", "fake_d1_backend"])
def db(request, monkeypatch):
    request.getfixturevalue(request.param)
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    return request.param


def _add(conn, content, **kw):
    return storage.add_memory(conn, content=content, **kw)


# --- resolution -------------------------------------------------------------

def test_generic_technical_text_stays_unclassified(db, projects):
    projects(["memora", "clmux", "pi"])
    with storage.connect() as conn:
        rec = _add(conn, GENERIC_TECH, tags=["architecture"])
    assert rec["tags"] == ["architecture"]
    assert not (rec["metadata"] or {}).get("section")
    assert "project" not in (rec["metadata"] or {})


def test_1082_and_1109_no_longer_share_a_project_by_keyword(db, projects):
    projects(["memora", "clmux", "pi"])
    assert storage._resolve_project(None, [], None) is None
    with storage.connect() as conn:
        old = _add(conn, PARKED_DESIGN, tags=["architecture"])
        new = _add(conn, PI_CHANNEL_WORK, tags=["architecture"], project="pi")
    assert old["tags"] == ["architecture"] and not (old["metadata"] or {}).get("section")
    assert new["tags"] == ["pi/architecture"]
    assert new["metadata"]["section"] == "pi" and new["metadata"]["project"] == "pi"


def test_multi_project_store_resolves_each_memory_from_its_own_markers(db, projects):
    projects(["memora", "clmux", "pi"])
    with storage.connect() as conn:
        by_tag = _add(conn, "sidebar refresh cadence", tags=["clmux/tui", "design"])
        by_meta = _add(conn, "doorbell over channel", metadata={"project": "pi"}, tags=["research"])
        explicit = _add(conn, "absorb gate calibration", tags=["analysis"], project="memora")
        ambiguous = _add(conn, "shared note", tags=["clmux/tui", "pi/channels", "plan"])
    assert by_tag["metadata"]["section"] == "clmux" and by_tag["metadata"]["subsection"] == "tui"
    assert "clmux/design" in by_tag["tags"]
    assert by_meta["metadata"]["section"] == "pi" and by_meta["tags"] == ["pi/research"]
    assert explicit["metadata"]["section"] == "memora" and explicit["tags"] == ["memora/analysis"]
    # Two configured projects in the tags: no guess.
    assert not (ambiguous["metadata"] or {}).get("section") and "plan" in ambiguous["tags"]


def test_unconfigured_store_infers_nothing_from_tags_but_accepts_explicit(db, projects):
    projects(None)
    with storage.connect() as conn:
        tagged = _add(conn, "sidebar refresh", tags=["clmux/tui", "design"])
        explicit = _add(conn, "sidebar refresh", tags=["design"], project="clmux")
    assert tagged["tags"] == ["clmux/tui", "design"] and not (tagged["metadata"] or {}).get("section")
    assert explicit["tags"] == ["clmux/design"] and explicit["metadata"]["section"] == "clmux"


def test_explicit_project_outside_the_configured_list_is_rejected(db, projects):
    projects(["memora", "clmux"])
    with storage.connect() as conn:
        with pytest.raises(storage.ProjectConfigError):
            _add(conn, "x y z note", project="pi")
        with pytest.raises(ValueError):
            _add(conn, "x y z note", project="Not A Name")


def test_per_store_projects_config(db, projects, monkeypatch):
    projects({"default": ["pi"], "other": ["memora"]})
    monkeypatch.setattr(storage, "effective_database_name", lambda: None)
    assert storage.configured_projects() == ("pi",)
    assert storage.configured_projects("other") == ("memora",)
    assert storage.configured_projects("missing") == ()


def test_malformed_projects_config_fails_loudly(projects):
    projects({"default": ["Bad Name"]})
    with pytest.raises(storage.ProjectConfigError):
        storage.configured_projects("default")


def test_update_memory_keeps_prefixing_on_its_explicit_project(db, projects):
    projects(["memora", "clmux", "pi"])
    with storage.connect() as conn:
        rec = _add(conn, "pi note", tags=["plan"], project="pi")
        updated = storage.update_memory(conn, rec["id"], tags=["design"])
    assert updated["tags"] == ["pi/design"]


# --- suggested tags -----------------------------------------------------------

def test_allowlisted_third_project_tags_survive_filtering(monkeypatch, projects):
    projects(["memora", "clmux", "pi"])
    monkeypatch.setattr(memora, "TAG_WHITELIST", {"pi/*", "memora/*"})
    kept = storage._filter_suggested_tags(["pi/research", "clmux/architecture", "memora/notes", "bare"])
    assert kept == ["pi/research", "memora/notes"]  # clmux/* not allowed; bare not project-prefixed


def test_allow_any_tag_keeps_any_prefixed_suggestion(monkeypatch, projects):
    projects(None)
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    assert storage._filter_suggested_tags(["trader/strategy", "pi/channels", "x"]) == [
        "trader/strategy", "pi/channels",
    ]


def test_suggestions_naming_another_configured_project_are_dropped(monkeypatch, projects):
    projects(["memora", "clmux", "pi"])
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())
    assert storage._filter_suggested_tags(
        ["clmux/architecture", "pi/channels", "other/x"], project="pi",
    ) == ["pi/channels", "other/x"]


# --- absorb -----------------------------------------------------------------

def test_explicit_project_on_absorb_drives_section_and_tag_prefixing(db, projects, monkeypatch):
    projects(["memora", "clmux", "pi"])
    monkeypatch.setattr(storage, "_compute_embedding", lambda *a, **k: {"x": 1.0})
    # The classifier suggests a clmux tag for pi work (what happened to #1109).
    monkeypatch.setattr(storage, "_search_snapshot_full", lambda *a, **k: [])
    with storage.connect() as conn:
        result = storage.absorb_memory(
            conn, [PI_CHANNEL_WORK], tags=["architecture"], project="pi",
        )
        (decision,) = result["decisions"]
        mem = storage.get_memory(conn, decision["memory_id"])
    assert mem["tags"] == ["pi/architecture"]
    assert mem["metadata"]["section"] == "pi" and mem["metadata"]["project"] == "pi"


def test_absorb_drops_suggested_tags_of_another_project(db, projects, monkeypatch):
    projects(["memora", "clmux", "pi"])
    monkeypatch.setattr(storage, "_compute_embedding", lambda *a, **k: {"x": 1.0})
    with storage.connect() as conn:
        seed = _add(conn, "pi inbox baseline", tags=["pi/channels"], project="pi")
    monkeypatch.setattr(storage, "_search_snapshot_full", lambda *a, **k: [{"score": 0.6, "memory": seed}])
    monkeypatch.setattr(
        storage, "_classify_fact_against_matches",
        lambda fact, matches: ([{"memory_id": seed["id"], "relationship": "RELATED", "reason": "r"}],
                               ["clmux/architecture", "pi/research"]),
    )
    with storage.connect() as conn:
        result = storage.absorb_memory(conn, [PI_CHANNEL_WORK], project="pi")
        (decision,) = result["decisions"]
        mem = storage.get_memory(conn, decision["memory_id"])
    assert "clmux/architecture" not in mem["tags"] and "pi/research" in mem["tags"]


def test_absorb_rejects_an_unknown_project_before_any_work(db, projects, monkeypatch):
    projects(["memora"])
    monkeypatch.setattr(storage, "_compute_embedding", lambda *a, **k: pytest.fail("no work"))
    with storage.connect() as conn:
        with pytest.raises(storage.ProjectConfigError):
            storage.absorb_memory(conn, ["some fact here"], project="pi")


def test_classify_prompt_no_longer_seeds_memora_or_clmux(monkeypatch):
    seen = {}

    class Completions:
        def create(self, **kw):
            seen["prompt"] = kw["messages"][-1]["content"]
            raise RuntimeError("stop")

    from types import SimpleNamespace
    monkeypatch.setattr(storage, "_get_llm_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    storage._classify_fact_against_matches("fact", [{"id": 1, "content": "c", "score": 0.5, "tags": []}])
    assert "memora/research" not in seen["prompt"] and "clmux/architecture" not in seen["prompt"]
    assert "Do not guess a project" in seen["prompt"]


# --- MCP tools and CLI ----------------------------------------------------------

def test_mcp_tools_accept_project(db, projects, monkeypatch):
    from memora import server

    projects(["memora", "clmux", "pi"])
    monkeypatch.setattr(storage, "_search_snapshot_full", lambda *a, **k: [])
    created = asyncio.run(server.memory_create("pi mailbox note", tags=["plan"], project="pi"))
    assert created["memory"]["tags"] == ["pi/plan"]
    issue = asyncio.run(server.memory_create_issue("pi bug", project="pi"))
    assert issue["memory"]["tags"] == ["pi/issues"] and issue["memory"]["metadata"]["project"] == "pi"
    todo = asyncio.run(server.memory_create_todo("pi task", project="pi"))
    assert todo["memory"]["tags"] == ["pi/todos"]
    bad = asyncio.run(server.memory_create("x y z", project="nope"))
    assert bad["error"] == "invalid_input"
    absorbed = asyncio.run(server.memory_absorb(["pi fact number one"], project="pi", dry_run=True))
    assert "error" not in absorbed
    rejected = asyncio.run(server.memory_absorb(["pi fact"], project="nope"))
    assert rejected["error"] == "invalid_input"


def test_cli_absorb_project_flag(db, projects, monkeypatch, capsys):
    from memora import cli

    projects(["pi"])
    monkeypatch.setattr(storage, "_search_snapshot_full", lambda *a, **k: [])
    monkeypatch.setattr(sys, "stdin", io.StringIO("cli pi fact text"))
    monkeypatch.setattr(sys, "argv", ["memora.cli", "absorb", "--project", "pi", "--tags", "plan"])
    cli.main()
    out = json.loads(capsys.readouterr().out)
    with storage.connect() as conn:
        mem = storage.get_memory(conn, out["decisions"][0]["memory_id"])
    assert mem["tags"] == ["pi/plan"]
    monkeypatch.setattr(sys, "stdin", io.StringIO("cli pi fact text two"))
    monkeypatch.setattr(sys, "argv", ["memora.cli", "absorb", "--project", "nope"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "invalid_input"


# --- dry-run report -------------------------------------------------------------

def test_report_lists_memories_the_keywords_would_have_classified(db, projects, capsys):
    sys.path.insert(0, "scripts")
    import report_project_detection as report

    projects(["memora", "clmux", "pi"])
    with storage.connect() as conn:
        # As the OLD code would have stored it: keyword -> clmux section + prefix.
        guessed = _add(conn, "The daemon keeps one workspace per socket; the sidebar lags.",
                       metadata={"section": "clmux"}, tags=["clmux/architecture"])
        explicit = _add(conn, "pi mailbox", tags=["plan"], project="pi")
        plain = _add(conn, "a recipe for bread", tags=["plan"])
        before = conn.execute("SELECT COUNT(*), MAX(id) FROM memories").fetchone()
    assert report.main(["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    ids = {m["id"]: m for m in data["memories"]}
    assert guessed["id"] in ids and explicit["id"] not in ids and plain["id"] not in ids
    row = ids[guessed["id"]]
    assert row["old_keyword_project"] == "clmux" and row["new_project"] is None
    assert row["changed"]["section"] == {"stored": "clmux", "new": None}
    assert row["changed"]["tags"] == {"stored": ["clmux/architecture"], "new": ["architecture"]}
    assert "clmux" in row["keyword_indicators"]
    assert data["summary"]["reported"] == 1
    assert report.main(["--json", "--all"]) == 0
    everything = json.loads(capsys.readouterr().out)
    assert everything["summary"]["reported"] >= 1 and plain["id"] not in {
        m["id"] for m in everything["memories"]
    }
    with storage.connect() as conn:  # read-only: nothing changed
        assert conn.execute("SELECT COUNT(*), MAX(id) FROM memories").fetchone() == before
        assert storage.get_memory(conn, guessed["id"])["tags"] == ["clmux/architecture"]
