"""Verbatim pre-change read code (memora fe09773), the reference the new
read paths must equal. Test-only; resolves storage primitives from
memora.storage, which still behave per-row when no view is passed."""

from typing import Any, Dict, List, Optional

import sqlite3

from memora.storage import (  # noqa: F401
    _get_full_history,
    _resolve_latest,
    _serialise_memory_for_follow,
    _superseded_ids_batch,
    retired_memory_ids,
    validate_follow,
)


def legacy_apply_follow(
    conn: sqlite3.Connection,
    results: List[Dict[str, Any]],
    follow: str,
    is_search: bool = False,
    seen_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """Apply lineage-aware post-processing to retrieval results.

    Args:
        conn: Database connection
        results: List of memory dicts (or search results with {score, memory} envelope)
        follow: Follow mode — "latest", "active", or "full_history"
        is_search: If True, results are {score, memory} envelopes
        seen_ids: Optional shared set so windowed list scans can dedupe
            latest leaves across successive candidate windows.

    Returns:
        Transformed results list

    Raises:
        ValueError: If follow mode is invalid
    """
    validate_follow(follow)

    if not results:
        return results

    def _get_mem(item: Dict) -> Dict:
        return item["memory"] if is_search else item

    def _get_id(item: Dict) -> int:
        return _get_mem(item)["id"]

    def _wrap(mem: Dict, score: float) -> Dict:
        return {"score": score, "memory": mem} if is_search else mem

    if follow == "active":
        # Pre-existing forks: storage follow=active (and digest) still
        # surfaces EVERY live leaf until the next absorb UPDATE collapses
        # them. Graph-only quarantine (authority_unknown on multi-leaf tips)
        # is the approved middle scope; storage-side quarantine is a follow-up.
        retired_ids = retired_memory_ids(conn)
        superseded = _superseded_ids_batch(conn, [_get_id(item) for item in results])
        return [
            item for item in results
            if _get_id(item) not in superseded
            and _get_id(item) not in retired_ids
        ]

    if follow == "latest":
        if seen_ids is None:
            seen_ids = set()
        retired_ids = retired_memory_ids(conn)
        out: List[Dict[str, Any]] = []
        for item in results:
            leaf_ids = _resolve_latest(conn, _get_id(item), retired_ids)
            for latest_id in leaf_ids:
                if latest_id in seen_ids:
                    continue
                seen_ids.add(latest_id)
                if latest_id == _get_id(item):
                    out.append(item)
                else:
                    latest_mem = _serialise_memory_for_follow(conn, latest_id)
                    if latest_mem:
                        out.append(_wrap(latest_mem, item.get("score", 0) if is_search else 0))
        return out

    if follow == "full_history":
        seen_ids: set[int] = set()
        out: List[Dict[str, Any]] = []
        for item in results:
            mid = _get_id(item)
            if mid in seen_ids:
                continue
            chain_ids = _get_full_history(conn, mid)
            for chain_id in chain_ids:
                if chain_id in seen_ids:
                    continue
                seen_ids.add(chain_id)
                if chain_id == mid:
                    out.append(item)
                else:
                    mem = _serialise_memory_for_follow(conn, chain_id)
                    if mem:
                        out.append(_wrap(mem, item.get("score", 0) if is_search else 0))
        return out

    return results
