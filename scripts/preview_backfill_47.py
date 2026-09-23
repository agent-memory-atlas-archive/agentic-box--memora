#!/usr/bin/env python3
"""Issue #47 backfill PREVIEW with an explicit approval list. READ-ONLY.

Writes NOTHING and creates nothing; it may REFUSE. A local SQLite store opens
read-only (mode=ro, or immutable for a WAL file with no sidecars); a local
WAL store WITH sidecars is in use by a writer process and is refused (exit
non-zero: stop the server, or use --db against D1). D1 opens through a raw
connection with no schema pass. SELECTs only. Its
output is a preview FILE the user reviews: every row carries
"approved": false, and a later apply step (a separate item) may act only on
rows the user flipped to true.

Two groups, per memory (import-pending rows are skipped):

  contradictions  The stored section (a project, assigned by the removed
                  keyword heuristics) differs from the project its TAGS gave
                  under the old rule, where a typed tag (<p>/issues, todos,
                  sections, documents, knowledge) counted as project evidence
                  -- e.g. a clmux issue whose only memora tag is the old
                  default memora/issues. The target project is proposed from
                  NON-TYPED evidence only: metadata.project, non-typed
                  "<project>/..." tags, and the stored section. Evidence that
                  disagrees, or none at all, is "needs-human".

  keyword_only    No explicit project (no metadata.project, no non-typed
                  project tag) but a stored section naming a configured
                  project. Proposed only where that section came from a
                  NON-TYPED source (the old content keyword detector gives
                  the same project); where the section is explained by a
                  typed tag alone, or by nothing found, "needs-human".

A proposal declares the project with metadata.project (the strongest
explicit marker, touching no tags and no tag policy) and re-prefixes the
memory's own typed tags to it. As an ALTERNATIVE the file also lists a
"<section>/<subsection>" marker tag where a subsection exists; the apply
step uses whichever the reviewer approves.

Usage:
  python scripts/preview_backfill_47.py --out preview.json [--db NAME] [--markdown preview.md]
Set MEMORA_PROJECTS as the server runs with it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _legacy_project_detection as legacy  # noqa: E402
from memora import storage  # noqa: E402
from memora.backends import LocalSQLiteBackend  # noqa: E402


class StoreInUse(SystemExit):
    pass


def open_read_only():
    """A connection that cannot write and creates nothing -- or a refusal.

    Local SQLite: a WAL database with a -wal or -shm present has a writer in
    another process (e.g. the memora server): reading it would need its
    locks, and if that writer closed between our check and our open the read
    would recreate its sidecars. So it is REFUSED. With no sidecars it opens
    immutable (no locks, nothing created); a rollback-journal database opens
    mode=ro. D1 and others: a raw connection with no schema pass.
    Guarantee: creates nothing, may refuse.
    """
    backend = storage.current_backend()
    if isinstance(backend, LocalSQLiteBackend):
        path = backend.db_path
        if not path.is_file():
            raise SystemExit(f"no database at {path}")
        with open(path, "rb") as fh:
            header = fh.read(20)  # only the header, never the whole database
        wal = len(header) >= 20 and (header[18] == 2 or header[19] == 2)
        if wal and (Path(f"{path}-wal").exists() or Path(f"{path}-shm").exists()):
            raise StoreInUse(
                f"store in use by a writer ({path} has WAL sidecars); stop the server, "
                "or use --db against D1")
        params = "mode=ro&immutable=1" if wal else "mode=ro"
        return sqlite3.connect(f"file:{quote(str(path.resolve()))}?{params}", uri=True)
    return backend.connect()  # D1 and others: a raw connection, no schema pass


def _old_tag_projects(tags: List[str], known: List[str]) -> set:
    """The configured projects the tags named under the OLD rule (typed tags counted)."""
    return {t.split("/", 1)[0] for t in tags if isinstance(t, str) and t.split("/", 1)[0] in known}


def _non_typed_tag_projects(tags: List[str], known: List[str]) -> set:
    return {
        t.split("/", 1)[0] for t in tags
        if isinstance(t, str) and "/" in t and t.split("/", 1)[0] in known
        and storage._typed_tag_kind(t) is None
    }


def _typed_retag(tags: List[str], target: str, metadata: Dict[str, Any]) -> Dict[str, str]:
    own = storage._existing_system_tags(tags, metadata)
    return {t: storage.project_tag(target, storage._typed_tag_kind(t)) for t in own
            if t != storage.project_tag(target, storage._typed_tag_kind(t))}


def assess(memory_id: int, content: str, metadata: Dict[str, Any], tags: List[str],
           known: List[str]) -> Optional[Dict[str, Any]]:
    section = metadata.get("section") if isinstance(metadata.get("section"), str) else None
    subsection = metadata.get("subsection") if isinstance(metadata.get("subsection"), str) else None
    meta_project = metadata.get("project") if metadata.get("project") in known else None
    non_typed = _non_typed_tag_projects(tags, known)
    old_tags = _old_tag_projects(tags, known)
    old_explicit = next(iter(old_tags)) if len(old_tags) == 1 else None
    new_project = storage._resolve_project(None, tags, metadata, strict=False)
    row: Dict[str, Any] = {
        "id": memory_id,
        "preview": content[:120],
        "stored": {"section": section, "subsection": subsection, "tags": list(tags),
                   "metadata_project": metadata.get("project"), "type": metadata.get("type")},
        "approved": False,
    }

    if section in known and old_explicit and old_explicit != section:
        row["group"] = "contradictions"
        evidence = []
        if meta_project:
            evidence.append({"source": "metadata.project", "project": meta_project})
        evidence += [{"source": "non-typed tag", "project": p} for p in sorted(non_typed)]
        evidence.append({"source": "section", "project": section})
        row["evidence"] = evidence
        targets = {e["project"] for e in evidence}
        if len(targets) != 1:
            row.update(status="needs-human", reason=f"non-typed evidence disagrees: {sorted(targets)}")
            return row
        target = targets.pop()
    elif section in known and new_project is None:
        row["group"] = "keyword_only"
        keyword = legacy._detect_project(content, None, [])
        if keyword == section:
            row["evidence"] = [{"source": "content keywords", "project": keyword}]
            target = section
        elif old_explicit == section:
            row["evidence"] = [{"source": "typed tag only", "project": old_explicit}]
            row.update(status="needs-human",
                       reason="the section is explained only by a typed tag (the old default), not by content")
            return row
        else:
            row["evidence"] = [{"source": "none found", "project": None}]
            row.update(status="needs-human", reason="no non-typed source explains the stored section")
            return row
    else:
        return None

    row["status"] = "proposed"
    row["proposal"] = {
        "set_metadata_project": target,
        "retag_typed": _typed_retag(tags, target, metadata),
        "alternative_marker_tag": (
            f"{target}/{subsection}" if subsection and storage._typed_tag_kind(f"{target}/{subsection}") is None
            else None
        ),
    }
    return row


def build_preview(conn, known: List[str]) -> Dict[str, Any]:
    rows = conn.execute("SELECT id, content, metadata, tags FROM memories ORDER BY id").fetchall()
    out: Dict[str, List[Dict[str, Any]]] = {"contradictions": [], "keyword_only": []}
    for memory_id, content, raw_meta, raw_tags in rows:
        if storage._import_pending(raw_meta):
            continue
        try:
            metadata = json.loads(raw_meta) if raw_meta else {}
        except (TypeError, ValueError):
            metadata = {}
        tags, _ok = storage._parse_tags_json(raw_tags, memory_id)
        row = assess(int(memory_id), content or "", metadata if isinstance(metadata, dict) else {},
                     tags if isinstance(tags, list) else [], known)
        if row is not None:
            out[row.pop("group")].append(row)
    summary = {
        "kind": "issue #47 backfill PREVIEW (read-only; nothing was written). Flip approved to true "
                "for rows to apply; the apply step is a separate item.",
        "scanned": len(rows),
        "configured_projects": known,
    }
    for group, items in out.items():
        summary[group] = {"total": len(items),
                          "proposed": sum(1 for r in items if r["status"] == "proposed"),
                          "needs_human": sum(1 for r in items if r["status"] == "needs-human")}
    return {"summary": summary, **out}


def to_markdown(preview: Dict[str, Any]) -> str:
    s = preview["summary"]
    lines = [f"# Issue 47 backfill preview (read-only)", "", s["kind"], "",
             f"Scanned {s['scanned']}; configured projects: {', '.join(s['configured_projects']) or 'none'}.", ""]
    for group in ("contradictions", "keyword_only"):
        g = s[group]
        lines += [f"## {group}: {g['total']} ({g['proposed']} proposed, {g['needs_human']} needs-human)", "",
                  "| id | status | target / reason | evidence | typed retag | preview |", "|---|---|---|---|---|---|"]
        for r in preview[group]:
            prop = r.get("proposal") or {}
            target = prop.get("set_metadata_project") or r.get("reason", "")
            ev = "; ".join(f"{e['source']}={e['project']}" for e in r.get("evidence", []))
            retag = ", ".join(f"{a}->{b}" for a, b in (prop.get("retag_typed") or {}).items()) or "-"
            prev = r["preview"].replace("|", "/").replace("\n", " ")[:80]
            lines.append(f"| {r['id']} | {r['status']} | {target} | {ev} | {retag} | {prev} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="registry store name (MEMORA_DATABASES)")
    ap.add_argument("--out", required=True, help="preview JSON file to write (the approval list)")
    ap.add_argument("--markdown", help="also write a markdown table here")
    args = ap.parse_args(argv)

    token = storage.CURRENT_DB.set(args.db) if args.db else None
    try:
        known = list(storage.configured_projects(args.db) if args.db else storage.configured_projects())
        conn = open_read_only()
        try:
            preview = build_preview(conn, known)
        finally:
            conn.close()
    finally:
        if token is not None:
            storage.CURRENT_DB.reset(token)
    Path(args.out).write_text(json.dumps(preview, indent=1, ensure_ascii=False) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(to_markdown(preview) + "\n")
    s = preview["summary"]
    print(f"PREVIEW (read-only) written to {args.out}: scanned {s['scanned']}; "
          f"contradictions {s['contradictions']}; keyword_only {s['keyword_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
