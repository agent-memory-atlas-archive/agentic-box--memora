#!/usr/bin/env python3
"""Dry-run report for issue #47: which memories got a section or a
project-prefixed tag from the removed keyword heuristics?

READ-ONLY. It runs SELECTs only and never writes; it exists so a backfill
(or re-tag) can be decided from facts. Point it at a store with the usual
memora configuration (MEMORA_DATABASES + --db NAME, or MEMORA_STORAGE_URI),
and set MEMORA_PROJECTS as the server will run with it.

Method, per memory:
  1. Undo what the old write path could have added on its own: a section
     equal to the old keyword-detected project (and the subsection that came
     with it), and "<project>/<generic-tag>" tags back to "<generic-tag>".
  2. Run the NEW pipeline (storage._resolve_project and friends) on that raw
     state, with no explicit project argument -- only what the memory itself
     carries (metadata.project, configured project tags), as a backfill would.
  3. Report the memory when its STORED section, subsection or tags differ
     from the new result -- exactly what a backfill would change -- AND the
     old content keyword indicators fired for it (scripts/_legacy_project_
     detection.py, verbatim from 795c40e): those are the keyword-caused cases
     issue #47 is about. --all also lists differences with no keyword hit
     (e.g. the old write path derived subsections from tags before
     prefixing them, a reconstruction after, so explicit-project memories
     can differ on subsection alone).

Step 1 is a heuristic: a section a user set by hand that happens to equal
the detected project looks auto-assigned. The report says "may", not "did".

Usage:
  python scripts/report_project_detection.py [--db NAME] [--json] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _legacy_project_detection as legacy  # noqa: E402
from memora import storage  # noqa: E402


def _indicators(content: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    text = content.lower()
    if metadata:
        text = f"{text} {str(metadata.get('section', '')).lower()} {str(metadata.get('context', '')).lower()}"
    hits: Dict[str, List[str]] = {}
    for project, patterns in legacy._PROJECT_INDICATORS.items():
        fired = [p for p in patterns if re.search(p, text)]
        if fired:
            hits[project] = fired
    return hits


def _raw_state(content: str, metadata: Optional[Dict[str, Any]], tags: List[str]):
    """Undo what the old write path may have added by itself (step 1)."""
    old_project = legacy._detect_project(content, metadata, tags)
    raw_meta = dict(metadata) if isinstance(metadata, dict) else None
    raw_tags = list(tags)
    if old_project:
        if raw_meta and raw_meta.get("section") == old_project:
            raw_meta.pop("section", None)
            raw_meta.pop("subsection", None)
        prefix = f"{old_project}/"
        raw_tags = [
            t[len(prefix):] if t.startswith(prefix) and t[len(prefix):] in legacy._GENERIC_TAGS_TO_PREFIX else t
            for t in raw_tags
        ]
    return raw_meta, raw_tags


def assess(memory_id: int, content: str, metadata: Optional[Dict[str, Any]], tags: List[str]) -> Optional[Dict[str, Any]]:
    """A report row when the STORED section/subsection/tags differ from what
    the new rules give for this memory (what a backfill would change);
    None otherwise."""
    raw_meta, raw_tags = _raw_state(content, metadata, tags)
    new_project = storage._resolve_project(None, raw_tags, raw_meta)
    new_meta = storage._auto_assign_section(raw_meta, raw_tags, new_project) or {}
    new_tags = storage._normalize_tags(raw_tags, new_project)

    stored_meta = metadata if isinstance(metadata, dict) else {}
    stored = {
        "section": stored_meta.get("section"),
        "subsection": stored_meta.get("subsection"),
        "tags": list(tags),
    }
    new = {
        "section": new_meta.get("section"),
        "subsection": new_meta.get("subsection"),
        "tags": list(new_tags),
    }
    changed = {k: {"stored": stored[k], "new": new[k]} for k in stored if stored[k] != new[k]}
    if not changed:
        return None
    return {
        "id": memory_id,
        "changed": changed,
        "old_keyword_project": legacy._detect_project(content, metadata, tags),
        "new_project": new_project,
        "keyword_indicators": _indicators(content, raw_meta),
        "preview": content[:120],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="registry store name (MEMORA_DATABASES)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="report at most N memories (0 = all)")
    ap.add_argument("--all", action="store_true",
                    help="also list differences where no content keyword fired")
    args = ap.parse_args(argv)

    token = storage.CURRENT_DB.set(args.db) if args.db else None
    try:
        conn = storage.connect()
        try:
            rows = conn.execute("SELECT id, content, metadata, tags FROM memories ORDER BY id").fetchall()
        finally:
            conn.close()
    finally:
        if token is not None:
            storage.CURRENT_DB.reset(token)

    report = []
    differ_total = 0
    for row in rows:
        memory_id, content = row[0], row[1] or ""
        try:
            metadata = json.loads(row[2]) if row[2] else None
        except json.JSONDecodeError:
            metadata = None
        tags, _ok = storage._parse_tags_json(row[3], memory_id)
        entry = assess(memory_id, content, metadata if isinstance(metadata, dict) else None,
                       tags if isinstance(tags, list) else [])
        if entry is None:
            continue
        differ_total += 1
        if not args.all and not entry["keyword_indicators"]:
            continue
        report.append(entry)
        if args.limit and len(report) >= args.limit:
            break

    summary = {
        "scanned": len(rows),
        "differ_from_new_rules": differ_total,
        "reported": len(report),
        "reported_filter": "all differences" if args.all else "keyword indicators fired",
        "configured_projects": list(storage.configured_projects(args.db)) if args.db
        else list(storage.configured_projects()),
    }
    if args.json:
        print(json.dumps({"summary": summary, "memories": report}, indent=1, ensure_ascii=False))
        return 0
    print(f"scanned {summary['scanned']} memories; {summary['differ_from_new_rules']} differ from "
          f"the new rules; {summary['reported']} reported ({summary['reported_filter']}); "
          f"configured projects: {summary['configured_projects'] or 'none'}")
    for entry in report:
        changes = "; ".join(f"{k}: {v['stored']!r} -> {v['new']!r}" for k, v in entry["changed"].items())
        hits = ", ".join(f"{p}:{len(v)}" for p, v in entry["keyword_indicators"].items()) or "-"
        print(f"#{entry['id']}: {changes}  [keywords {hits}]  {entry['preview']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
