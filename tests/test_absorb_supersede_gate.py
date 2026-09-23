"""The supersede gate: a classifier UPDATE supersedes only when it is the same
project and entity and the old claim is fully replaced.

Driven end to end through a fake LLM client that answers both prompts absorb
sends (the classify prompt and the supersede-verify prompt), so these tests
exercise the real _classify_fact_against_matches parsing, the gate, and the
phase-3 write path.
"""

import json
import logging
from types import SimpleNamespace

import pytest

import memora
import memora.storage as storage


@pytest.fixture(autouse=True)
def _no_tag_whitelist(monkeypatch):
    monkeypatch.setattr(memora, "TAG_WHITELIST", set())


# Modelled on memora #1082 / #1109 (2026-09-23): a parked design idea was
# superseded by unrelated work that only shared "clmux agent delivery".
PARKED_DESIGN = (
    "Claude Mods design idea (parked, not started): a mod loader that lets "
    "users bundle prompt snippets, hooks and statusline widgets as installable "
    "mods, with a manifest and per-mod enable/disable. Open question: whether "
    "mods ship through clmux agent delivery or a separate registry. Parked "
    "until the plugin API stabilises."
)
PI_CHANNEL_WORK = (
    "pi channel work: the pi agent now receives inbox doorbells over the clmux "
    "agent delivery channel instead of pane injection; registry_recv drains "
    "the durable inbox and the doorbell is one line."
)


class FakeLLM:
    """Answers absorb's two prompts; records every prompt it saw."""

    def __init__(self, *, classify, verify=None):
        self.classify = classify
        self.verify = verify
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("Decide whether a NEW fact should REPLACE"):
            if self.verify is None:
                raise AssertionError("supersede verifier must not be called here")
            answer = self.verify(prompt)
        else:
            answer = self.classify(prompt)
        if isinstance(answer, Exception):
            raise answer
        content = answer if isinstance(answer, str) else json.dumps(answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def verify_prompts(self):
        return [p for p in self.prompts if p.startswith("Decide whether a NEW fact")]


def _update(target_id, reason="newer information on clmux agent delivery"):
    return lambda prompt: {
        "classifications": [{"memory_id": target_id, "relationship": "UPDATE", "reason": reason}],
        "suggested_tags": [],
    }


def _verdict(same_project, same_entity, fully_replaces, related=True, reason="checked"):
    return lambda prompt: {
        "same_project": same_project, "same_entity": same_entity,
        "fully_replaces": fully_replaces, "related": related, "reason": reason,
    }


@pytest.fixture()
def seeded(fake_d1_backend, monkeypatch):
    monkeypatch.setattr(storage, "_compute_embedding", lambda *a, **k: {"x": 1.0})
    with storage.connect() as conn:
        old = storage.add_memory(conn, content=PARKED_DESIGN, tags=["clmux/ideas"])
    return old


def _absorb(monkeypatch, llm, old, fact, *, score=0.72, **kwargs):
    monkeypatch.setattr(storage, "_get_llm_client", lambda: llm)
    monkeypatch.setattr(
        storage, "_search_snapshot_full",
        lambda *a, **k: [{"score": score, "memory": old}],
    )
    with storage.connect() as conn:
        result = storage.absorb_memory(conn, [fact], **kwargs)
        active = {m["id"] for m in storage.list_memories(conn, follow="active")}
        crossrefs = storage.get_crossrefs(conn, old["id"])
    return result, active, crossrefs


def test_regression_1082_parked_design_not_superseded_by_unrelated_work(
    seeded, monkeypatch, caplog,
):
    old = seeded
    llm = FakeLLM(
        classify=_update(old["id"]),
        verify=_verdict(True, False, False, reason="different piece of work in the same area"),
    )
    with caplog.at_level(logging.INFO, logger="memora.storage"):
        result, active, crossrefs = _absorb(monkeypatch, llm, old, PI_CHANNEL_WORK)

    (decision,) = result["decisions"]
    assert decision["action"] == "linked"
    assert decision["downgraded_from"] == "UPDATE"
    assert decision["supersede_check"]["gate"] == "llm"
    assert decision["supersede_check"]["same_entity"] is False
    assert result["superseded"] == 0 and result["linked"] == 1
    # The parked idea stays visible and gains no successor.
    assert old["id"] in active
    assert all(r.get("edge_type") != "superseded_by" for r in crossrefs)
    assert {"id": decision["memory_id"], "score": 1.0, "edge_type": "related_to"} in crossrefs
    # The verifier saw both texts in full plus the old memory's tags.
    (vp,) = llm.verify_prompts()
    assert PARKED_DESIGN in vp and PI_CHANNEL_WORK in vp and "clmux/ideas" in vp
    # Audit line: old text, new text, score and both reasons.
    line = next(r.getMessage() for r in caplog.records if "update_downgraded" in r.getMessage())
    assert PARKED_DESIGN[:200] in line and PI_CHANNEL_WORK[:120] in line
    assert "score=0.72" in line and "different piece of work" in line
    assert "newer information on clmux agent delivery" in line


def test_confirmed_update_supersedes_and_logs_audit(seeded, monkeypatch, caplog):
    old = seeded
    fact = PARKED_DESIGN.replace("Parked until the plugin API stabilises.", "Unparked: work started 2026-09-23.")
    llm = FakeLLM(classify=_update(old["id"], "same idea, now started"), verify=_verdict(True, True, True))
    with caplog.at_level(logging.INFO, logger="memora.storage"):
        result, active, _ = _absorb(monkeypatch, llm, old, fact)

    (decision,) = result["decisions"]
    assert decision["action"] == "superseded"
    assert decision["target_id"] == old["id"]
    assert decision["score"] == 0.72
    assert decision["supersede_check"]["verdict"] == "supersede"
    assert "downgraded_from" not in decision
    assert old["id"] not in active and decision["memory_id"] in active
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("absorb supersede:"))
    assert f"target=#{old['id']}" in line and "score=0.72" in line
    assert PARKED_DESIGN[:200] in line and "Unparked: work started" in line
    assert "same idea, now started" in line


def test_low_similarity_update_is_downgraded_without_llm_check(seeded, monkeypatch):
    old = seeded
    llm = FakeLLM(classify=_update(old["id"]))  # verify=None: must not be asked
    result, active, _ = _absorb(monkeypatch, llm, old, PI_CHANNEL_WORK, score=0.45)
    (decision,) = result["decisions"]
    assert decision["action"] == "linked"
    assert decision["supersede_check"]["gate"] == "score"
    assert old["id"] in active


def test_cross_project_update_is_downgraded_without_llm_check(seeded, monkeypatch):
    old = seeded
    llm = FakeLLM(classify=_update(old["id"]))
    result, active, _ = _absorb(
        monkeypatch, llm, old, PI_CHANNEL_WORK, tags=["pi/channels"],
    )
    (decision,) = result["decisions"]
    assert decision["action"] == "linked"
    assert decision["supersede_check"]["gate"] == "project"
    assert old["id"] in active


@pytest.mark.parametrize("bad_answer", [
    RuntimeError("provider 502"),
    "I think these are the same thing, yes.",
    {"same_project": "true", "same_entity": "true"},  # fully_replaces missing
])
def test_uncertain_or_failed_check_never_supersedes(seeded, monkeypatch, bad_answer):
    old = seeded
    llm = FakeLLM(classify=_update(old["id"]), verify=lambda prompt: bad_answer)
    result, active, _ = _absorb(monkeypatch, llm, old, PI_CHANNEL_WORK)
    (decision,) = result["decisions"]
    assert decision["action"] == "linked"
    assert decision["downgraded_from"] == "UPDATE"
    assert old["id"] in active


def test_unrelated_verdict_creates_without_link(seeded, monkeypatch):
    old = seeded
    llm = FakeLLM(classify=_update(old["id"]), verify=_verdict(False, False, False, related=False))
    result, active, crossrefs = _absorb(monkeypatch, llm, old, PI_CHANNEL_WORK)
    (decision,) = result["decisions"]
    assert decision["action"] == "created"
    assert old["id"] in active
    assert all(r.get("id") != decision["memory_id"] or r.get("edge_type") == "related_to"
               for r in crossrefs)
    assert all(r.get("edge_type") not in ("superseded_by", "supersedes") for r in crossrefs)


def test_verify_prompt_carries_context_and_untruncated_text(seeded, monkeypatch):
    old = seeded
    long_old = PARKED_DESIGN + " " + ("detail " * 120)
    old = {**old, "content": long_old}
    llm = FakeLLM(classify=_update(old["id"]), verify=_verdict(True, False, False))
    _absorb(monkeypatch, llm, old, PI_CHANNEL_WORK, context="session on pi agent inbox wiring", dry_run=True)
    (vp,) = llm.verify_prompts()
    assert long_old[:storage._SUPERSEDE_VERIFY_MAX_CHARS] in vp
    assert len(long_old) > 800
    assert "session on pi agent inbox wiring" in vp
    # The classifier sees more than the old 300 characters too.
    classify_prompt = next(p for p in llm.prompts if p.startswith("Compare this new fact"))
    assert long_old[:800] in classify_prompt


def test_dry_run_reports_the_check(seeded, monkeypatch):
    old = seeded
    llm = FakeLLM(classify=_update(old["id"]), verify=_verdict(True, True, True))
    result, active, _ = _absorb(monkeypatch, llm, old, PI_CHANNEL_WORK + " v2", dry_run=True)
    (decision,) = result["decisions"]
    assert decision["action"] == "supersede"
    assert decision["supersede_check"]["verdict"] == "supersede"
    assert old["id"] in active  # dry run wrote nothing


def test_concurrent_path_is_gated_too(fake_d1_backend, monkeypatch):
    monkeypatch.setattr(storage, "_compute_embedding", lambda *a, **k: {"x": 1.0})
    monkeypatch.setenv("MEMORA_ABSORB_CONCURRENCY", "4")
    with storage.connect() as conn:
        a = storage.add_memory(conn, content="setting alpha is 5 extra words", tags=["memora/config"])
        b = storage.add_memory(conn, content="setting beta is 7 extra words", tags=["memora/config"])
    targets = {"alpha": a, "beta": b}

    def classify(prompt):
        mem = targets["alpha" if "alpha is 6" in prompt else "beta"]
        return {"classifications": [{"memory_id": mem["id"], "relationship": "UPDATE", "reason": "r"}]}

    def verify(prompt):
        # Confirm alpha's update, reject beta's.
        return _verdict(True, "alpha is 6" in prompt, "alpha is 6" in prompt)(prompt)

    llm = FakeLLM(classify=classify, verify=verify)
    monkeypatch.setattr(storage, "_get_llm_client", lambda: llm)
    monkeypatch.setattr(
        storage, "_search_snapshot_full",
        lambda conn, corpus, vector, **k: [{"score": 0.7, "memory": a}, {"score": 0.7, "memory": b}],
    )
    with storage.connect() as conn:
        result = storage.absorb_memory(conn, ["setting alpha is 6 now", "setting beta is 8 now"])
    actions = [d["action"] for d in result["decisions"]]
    assert sorted(actions) == ["linked", "superseded"]
    assert result["profile"]["counters"]["llm_supersede_checks"] == 2


def test_llm_bool_coercion_is_strict():
    assert storage._coerce_llm_bool(True) is True
    assert storage._coerce_llm_bool("yes") is True
    for v in (False, "false", "no", None, 1, "maybe", ""):
        assert storage._coerce_llm_bool(v) is False
