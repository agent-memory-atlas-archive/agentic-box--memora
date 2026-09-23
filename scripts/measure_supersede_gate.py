#!/usr/bin/env python3
"""Calibrate absorb's supersede gate on labelled (old, new) pairs.

Runs _absorb_check_supersede — the per-leaf gate: score floor, project tag
prefixes, then the LLM verifier — on each pair of
tests/fixtures/supersede_gate_pairs.json and reports, per category, how
often it supersedes and which stage decided.

It touches NO store: pairs are passed to the gate directly.

  --scores live   real bge-m3 cosine via the configured embedding endpoint
                  (the old side embedded like a stored memory: content +
                  metadata + tags; the new side as absorb's phase 1 does:
                  the bare fact). Needs MEMORA_EMBEDDING_* / OPENAI_* env.
  --scores fixed  every pair scored --fixed-score (default 0.7): isolates
                  the project rule and the LLM.
  --llm yes       fake verifier that always says supersede: shows what the
                  deterministic stages alone block (the recall ceiling).
  --llm live      the real verifier on MEMORA_LLM_MODEL (OPENAI_API_KEY /
                  OPENAI_BASE_URL).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memora import storage  # noqa: E402
from memora.embeddings import cosine_similarity  # noqa: E402

DEFAULT_PAIRS = ROOT / "tests" / "fixtures" / "supersede_gate_pairs.json"
STORED_META = {"source": "manual", "confidence": 0.8}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--scores", choices=("live", "fixed"), default="fixed")
    ap.add_argument("--fixed-score", type=float, default=0.7)
    ap.add_argument("--llm", choices=("yes", "live"), default="yes")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    pairs = json.loads(args.pairs.read_text())
    if args.llm == "yes":
        storage._verify_absorb_supersede_llm = lambda *a, **k: {
            "verdict": "supersede", "same_project": True, "same_entity": True,
            "fully_replaces": True, "related": True, "reason": "fake yes",
        }
    elif storage._get_llm_client() is None:
        print("--llm live needs OPENAI_API_KEY", file=sys.stderr)
        return 2

    rows = []
    for p in pairs:
        if args.scores == "live":
            old_vec = storage._compute_embedding(p["old"], STORED_META, p["old_tags"])
            new_vec = storage._compute_embedding(p["new"], None, [])
            score = cosine_similarity(new_vec, old_vec)
        else:
            score = args.fixed_score
        leaf = {"id": 1, "content": p["old"], "tags": p["old_tags"], "created_at": "2026-09-01", "score": score}
        check = storage._absorb_check_supersede(p["new"], leaf, [], caller_tags=p["new_tags"])
        rows.append({
            "id": p["id"], "category": p["category"], "should_supersede": p["should_supersede"],
            "score": round(score, 3), "gate": check["gate"], "verdict": check["verdict"],
            "supersede": check["verdict"] == "supersede", "reason": check.get("reason"),
            "flags": {k: check.get(k) for k in ("same_project", "same_entity", "fully_replaces")},
        })

    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    print(f"scores={args.scores} llm={args.llm} model={storage.LLM_MODEL if args.llm == 'live' else '-'}")
    print(f"{'id':<22}{'category':<26}{'want':>5}{'got':>5}{'score':>7}  gate     reason")
    for r in rows:
        mark = "" if r["supersede"] == r["should_supersede"] else "  <-- WRONG"
        print(f"{r['id']:<22}{r['category']:<26}{'Y' if r['should_supersede'] else 'n':>5}"
              f"{'Y' if r['supersede'] else 'n':>5}{r['score']:>7.3f}  {r['gate']:<8} {str(r['reason'])[:70]}{mark}")
    by_cat = defaultdict(Counter)
    for r in rows:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["correct"] += r["supersede"] == r["should_supersede"]
        by_cat[r["category"]][f"gate:{r['gate']}"] += 1
    print()
    for cat, c in by_cat.items():
        gates = ", ".join(f"{k[5:]}={v}" for k, v in sorted(c.items()) if k.startswith("gate:"))
        print(f"{cat:<26} correct {c['correct']}/{c['n']}   decided by: {gates}")
    tp = sum(r["supersede"] and r["should_supersede"] for r in rows)
    fp = sum(r["supersede"] and not r["should_supersede"] for r in rows)
    fn = sum(not r["supersede"] and r["should_supersede"] for r in rows)
    print(f"\nsupersede precision {tp}/{tp + fp}  recall {tp}/{tp + fn}  false supersedes {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
