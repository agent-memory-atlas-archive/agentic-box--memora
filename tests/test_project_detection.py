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


# --- round 2: metadata.project obeys the same rule (review 7030 HIGH 1) --------

def test_metadata_project_cannot_bypass_the_configured_list(db, projects):
    from memora import server

    projects(["memora"])
    with storage.connect() as conn:
        with pytest.raises(storage.ProjectConfigError):
            _add(conn, "a pi fact", metadata={"project": "pi"}, tags=["analysis"])
        # import: the entry fails, nothing tagged pi/ is written
        result = storage.import_memories(
            conn, [{"content": "imported pi fact", "metadata": {"project": "pi"}, "tags": ["analysis"]}],
        )
        assert result.get("errors") and not any(
            "pi/analysis" in (m.get("tags") or []) for m in storage.list_memories(conn)
        )
    bad = asyncio.run(server.memory_create("a pi fact", metadata={"project": "pi"}, tags=["analysis"]))
    assert bad["error"] == "invalid_input"


def test_update_rejects_a_supplied_project_but_tolerates_a_stored_one(db, projects):
    projects(None)
    with storage.connect() as conn:
        legacy = _add(conn, "legacy memory", metadata={"project": "target"}, tags=["plan"])
    projects(["memora"])
    with storage.connect() as conn:
        with pytest.raises(storage.ProjectConfigError):
            storage.update_memory(conn, legacy["id"], metadata={"project": "pi"})
        # The stored legacy value is not configured: tolerated, but no prefixing.
        updated = storage.update_memory(conn, legacy["id"], tags=["design"])
    assert updated["tags"] == ["design"]


# --- round 2: typed tags follow the project, no memora default (HIGH 2) --------

def test_typed_tags_follow_the_project_with_no_memora_default(db, projects, monkeypatch):
    from memora import server

    projects(["memora", "clmux", "pi"])
    monkeypatch.setattr(storage, "_search_snapshot_full", lambda *a, **k: [])
    assert asyncio.run(server.memory_create_issue("an issue"))["memory"]["tags"] == ["issues"]
    assert asyncio.run(server.memory_create_todo("a task"))["memory"]["tags"] == ["todos"]
    assert asyncio.run(server.memory_create_section("Arch"))["memory"]["tags"] == ["sections"]
    section = asyncio.run(server.memory_create_section("Arch", project="pi"))["memory"]
    assert section["tags"] == ["pi/sections"] and section["metadata"]["project"] == "pi"


def test_documents_take_the_project(db, projects):
    from memora import server

    projects(["memora", "pi"])
    doc = "# Plan\n\n1. first step\n2. second step\n"
    out = asyncio.run(server.memory_store_document(doc, "pi/plan-doc", project="pi"))
    with storage.connect() as conn:
        root = storage.get_memory(conn, out["root_id"])
        frags = [storage.get_memory(conn, i) for ids in out["node_map"].values() for i in ids]
    assert "pi/documents" in root["tags"] and root["metadata"]["project"] == "pi"
    assert frags and all("pi/documents" in f["tags"] and f["metadata"]["project"] == "pi" for f in frags)
    plain = asyncio.run(server.memory_store_document(doc, "plain-doc"))
    with storage.connect() as conn:
        assert "documents" in storage.get_memory(conn, plain["root_id"])["tags"]
        assert "memora/documents" not in storage.get_memory(conn, plain["root_id"])["tags"]


def test_create_suggestions_use_the_memorys_own_project(db, projects, monkeypatch):
    from memora import server

    projects(["memora", "pi"])
    monkeypatch.setattr(storage, "_search_snapshot_full", lambda *a, **k: [])
    pi = asyncio.run(server.memory_create("TODO: wire the pi inbox", project="pi"))
    assert pi["suggestions"]["tags"] == ["pi/todos"]
    none = asyncio.run(server.memory_create("TODO: something generic"))
    assert none["suggestions"]["tags"] == ["todos"]
    # memora-tagged content still resolves to memora under MEMORA_PROJECTS.
    mem = asyncio.run(server.memory_create("TODO: absorb gate", tags=["memora/absorb"]))
    assert mem["suggestions"]["tags"] == ["memora/todos"]


def test_graph_issue_filter_accepts_any_project_issues_tag(graph_request, projects):
    projects(["pi"])
    with storage.connect() as conn:
        tagged = _add(conn, "tag-only pi issue", tags=["pi/issues"])
        bare = _add(conn, "tag-only bare issue", tags=["issues"])
        other = _add(conn, "not an issue", tags=["pi/notes"])
    status, api = graph_request("GET", "/api/memories?type=issue&limit=50")
    assert status == 200
    ids = {m["id"] for m in api.get("memories", api if isinstance(api, list) else [])}
    assert {tagged["id"], bare["id"]} <= ids and other["id"] not in ids


# --- round 2: full MEMORA_PROJECTS validation at startup ------------------------

@pytest.mark.parametrize("raw", [
    '{"memora": ["memora"], "unused": ["Bad Name"]}',
    '{"memora": "memora"}',
    '{"Bad Store": ["memora"]}',
    '"memora"',
    '["ok", 3]',
    "not json",
])
def test_malformed_projects_config_fails_at_startup(monkeypatch, raw, capsys):
    from memora import server

    monkeypatch.setenv("MEMORA_PROJECTS", raw)
    with pytest.raises(storage.ProjectConfigError):
        storage.load_projects_config()
    with pytest.raises(SystemExit) as exit_info:
        server.main(["--transport", "stdio"])
    assert exit_info.value.code == 2
    assert "MEMORA_PROJECTS" in capsys.readouterr().err


def test_report_says_it_is_a_preview_not_the_backfill(db, projects, capsys):
    sys.path.insert(0, "scripts")
    import report_project_detection as report

    projects(["clmux"])
    assert report.main([]) == 0
    out = capsys.readouterr().out
    assert "REMEDIATION PREVIEW" in out and "backfill_tags does NOT perform" in out
    assert report.main(["--json"]) == 0
    assert "does NOT perform" in json.loads(capsys.readouterr().out)["summary"]["kind"]


# --- round 3: typed tags under the DEFAULT tag policy (review 7040 HIGH 1) ------

@pytest.fixture(params=["local_db", "fake_d1_backend"])
def default_policy_db(request, monkeypatch):
    """The real out-of-the-box policy: memora.DEFAULT_TAGS, no ALLOW_ANY."""
    request.getfixturevalue(request.param)
    monkeypatch.setattr(memora, "TAG_WHITELIST", set(memora.DEFAULT_TAGS))
    monkeypatch.setattr(storage, "_search_snapshot_full", lambda *a, **k: [])
    return request.param


def test_typed_tools_work_under_the_default_policy(default_policy_db, projects):
    from memora import server

    projects(["pi"])
    for tool, kind in ((server.memory_create_issue, "issues"), (server.memory_create_todo, "todos"),
                       (server.memory_create_section, "sections")):
        bare = asyncio.run(tool("typed thing"))
        assert "error" not in bare, bare
        assert bare["memory"]["tags"] == [kind]
        pi = asyncio.run(tool("typed pi thing", project="pi"))
        assert "error" not in pi, pi
        assert pi["memory"]["tags"] == [f"pi/{kind}"]


def test_user_supplied_tags_are_still_enforced_under_the_default_policy(default_policy_db, projects):
    from memora import server

    projects(["pi"])
    # A caller cannot hand-apply a typed tag: only memora's own are exempt.
    denied = asyncio.run(server.memory_create("x y z", tags=["pi/issues"]))
    assert denied["error"] == "invalid_input"
    denied = asyncio.run(server.memory_create("x y z", tags=["not-allowed"]))
    assert denied["error"] == "invalid_input"
    with storage.connect() as conn:
        with pytest.raises(ValueError):  # a system tag for ANOTHER project
            _add(conn, "x y z", project="pi", system_tags=["clmux/issues"])
        with pytest.raises(ValueError):  # not a typed kind
            _add(conn, "x y z", system_tags=["anything"])


def test_explicit_project_keeps_allowed_generic_tags_bare_under_the_default_policy(default_policy_db, projects):
    projects(["pi"])
    with storage.connect() as conn:
        rec = _add(conn, "pi plan text", tags=["plan", "analysis"], project="pi")
    # pi/plan is not in the default policy: stays "plan" instead of failing.
    assert rec["tags"] == ["plan", "analysis"] and rec["metadata"]["section"] == "pi"


def test_documents_work_under_the_default_policy(default_policy_db, projects):
    from memora import server

    projects(["pi"])
    doc = "# Plan\n\n1. first step\n2. second step\n"
    out = asyncio.run(server.memory_store_document(doc, "pi/default-policy", project="pi"))
    assert "error" not in out, out
    with storage.connect() as conn:
        assert "pi/documents" in storage.get_memory(conn, out["root_id"])["tags"]


# --- round 3: documents resolve their project before the plan (HIGH 2) ----------

@pytest.mark.parametrize("how", ["metadata", "tag"])
def test_document_project_inferred_from_metadata_or_tag(db, projects, how):
    from memora import server

    projects(["memora", "pi"])
    doc = "# Plan\n\n1. first step\n2. second step\n"
    kwargs = {"metadata": {"project": "pi"}} if how == "metadata" else {"tags": ["pi/notes"]}
    out = asyncio.run(server.memory_store_document(doc, f"inferred-{how}", **kwargs))
    assert "error" not in out, out
    with storage.connect() as conn:
        root = storage.get_memory(conn, out["root_id"])
        frags = [storage.get_memory(conn, i) for ids in out["node_map"].values() for i in ids]
    for mem in [root, *frags]:
        assert "pi/documents" in mem["tags"] and "documents" not in mem["tags"], mem["tags"]


def test_document_rejects_an_unconfigured_metadata_project(db, projects):
    from memora import server

    projects(["memora"])
    out = asyncio.run(server.memory_store_document("# T\n\ntext\n", "bad", metadata={"project": "pi"}))
    assert out["error"] == "invalid_input"


# --- round 3: the digest recognises any typed tag (MEDIUM) ------------------------

def test_digest_buckets_include_tag_only_typed_entries(db, projects):
    from memora import server

    projects(None)
    with storage.connect() as conn:
        entries = {
            "todos": _add(conn, "routing digest todo bare", tags=["todos", "routing"]),
            "pi/todos": _add(conn, "routing digest todo pi", tags=["pi/todos", "routing"]),
            "pi/issues": _add(conn, "routing digest issue pi", tags=["pi/issues", "routing"]),
            "memora/issues": _add(conn, "routing digest issue legacy", tags=["memora/issues", "routing"]),
            "issues": _add(conn, "routing digest issue bare", tags=["issues", "routing"]),
        }
        noise = _add(conn, "routing digest plain note", tags=["routing"])
    digest = asyncio.run(server.memory_digest("routing digest", k=20))
    todo_ids = {item["id"] for item in digest["todos"]}
    issue_ids = {item["id"] for item in digest["issues"]}
    assert {entries["todos"]["id"], entries["pi/todos"]["id"]} <= todo_ids
    assert {entries["pi/issues"]["id"], entries["memora/issues"]["id"], entries["issues"]["id"]} <= issue_ids
    assert noise["id"] not in todo_ids | issue_ids


# --- round 4: system tags are internal-only and round-trip (review 7044) ---------

def test_public_batch_cannot_smuggle_system_tags(default_policy_db, projects):
    from memora import server

    projects(["pi"])
    out = asyncio.run(server.memory_create_batch([
        {"content": "hand-applied typed tag", "project": "pi",
         "metadata": {"type": "issue"}, "system_tags": ["pi/issues"]},
    ]))
    assert out["error"] == "invalid_batch"
    with storage.connect() as conn:
        assert not storage.list_memories(conn)
    # The internal typed tools still work under the default policy.
    issue = asyncio.run(server.memory_create_issue("real issue", project="pi"))
    assert issue["memory"]["tags"] == ["pi/issues"]


def test_system_tags_are_bound_to_metadata_type(default_policy_db, projects):
    projects(["pi"])
    with storage.connect() as conn:
        with pytest.raises(ValueError):
            _add(conn, "a todo, not an issue", project="pi",
                 metadata={"type": "todo"}, system_tags=["pi/issues"])
        with pytest.raises(ValueError):
            _add(conn, "a note", project="pi", system_tags=["pi/documents"])


def test_update_keeps_a_typed_tag_but_rejects_a_foreign_one(default_policy_db, projects):
    from memora import server

    projects(["pi"])
    issue = asyncio.run(server.memory_create_issue("editable issue", project="pi"))["memory"]
    with storage.connect() as conn:
        kept = storage.update_memory(conn, issue["id"], content="edited issue", tags=["pi/issues", "plan"])
        assert set(kept["tags"]) == {"pi/issues", "plan"}
        with pytest.raises(ValueError):  # a typed tag this issue never had
            storage.update_memory(conn, issue["id"], tags=["pi/issues", "pi/todos"])
        note = _add(conn, "plain note", project="pi", tags=["plan"])
        with pytest.raises(ValueError):  # hand-applying a typed tag via update
            storage.update_memory(conn, note["id"], tags=["plan", "pi/issues"])
    via_tool = asyncio.run(server.memory_update(issue["id"], tags=["pi/issues", "note"]))
    assert "error" not in via_tool, via_tool


def _typed_fixture(server):
    asyncio.run(server.memory_create_issue("rt issue", project="pi"))
    asyncio.run(server.memory_create_todo("rt todo"))
    asyncio.run(server.memory_create_section("rt section", project="pi"))
    asyncio.run(server.memory_store_document("# RT\n\n1. one\n2. two\n", "rt-doc", project="pi"))
    with storage.connect() as conn:
        _add(conn, "rt plain", tags=["plan"])


def _tags_by_content(conn):
    return {m["content"]: sorted(m["tags"]) for m in storage.list_memories(conn, limit=-1)}


def test_export_import_round_trips_typed_tags(default_policy_db, projects):
    from memora import server

    projects(["pi"])
    _typed_fixture(server)
    with storage.connect() as conn:
        before = _tags_by_content(conn)
        exported = storage.export_memories(conn)
        assert any(r["system_tags"] == ["pi/issues"] for r in exported)
        result = storage.import_memories(conn, exported, strategy="replace")
        assert result["replaced"] is True and result["total_errors"] == 0, result
        assert _tags_by_content(conn) == before


def test_replace_import_with_one_bad_entry_deletes_nothing(default_policy_db, projects):
    from memora import server

    projects(["pi"])
    _typed_fixture(server)
    with storage.connect() as conn:
        before = _tags_by_content(conn)
        exported = storage.export_memories(conn)
        bad = exported + [{"content": "bad entry", "tags": ["not-allowed"]}]
        result = storage.import_memories(conn, bad, strategy="replace")
        assert result["replaced"] is False and result["imported"] == 0 and result["total_errors"] == 1
        assert _tags_by_content(conn) == before  # nothing deleted, nothing added


def test_import_refuses_forged_system_tags(default_policy_db, projects):
    projects(["pi"])
    with storage.connect() as conn:
        result = storage.import_memories(conn, [
            {"content": "not really an issue", "metadata": {"type": "note"},
             "tags": ["pi/issues"], "system_tags": ["pi/issues"], "project": "pi"},
        ])
        assert result["imported"] == 0 and result["total_errors"] == 1
        assert not storage.list_memories(conn)
