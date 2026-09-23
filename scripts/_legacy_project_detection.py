"""Verbatim pre-issue-#47 project detection (memora 795c40e, memora/storage.py).

Kept ONLY for scripts/report_project_detection.py and its tests: it shows
what the removed keyword heuristics would have assigned, so a backfill can
be decided. Nothing in memora imports this.
"""

import re
from typing import Any, Dict, List, Optional

_PROJECT_INDICATORS: Dict[str, List[str]] = {
    "memora": [
        r"\bmemora\b", r"\bmemory.server\b", r"\bmcp.server\b",
        r"\bstorage\.py\b", r"\babsorb\b", r"\bembedding", r"\bcrossref",
        r"\bgraph.visualization\b", r"\bknowledge.graph\b",
        r"\bmemory_create\b", r"\bmemory_absorb\b", r"\bmemory_search\b",
    ],
    "clmux": [
        r"\bclmux\b", r"\btmux.workspace\b", r"\bmultiplexer\b",
        r"\btmux\b", r"\bpane\b", r"\bworkspace\b", r"\bsidebar\b",
        r"\btui\b", r"\bdaemon\b", r"\bsocket.server\b",
    ],
}

# Tags that imply a project (checked when content detection fails)
_TAG_PROJECT_MAP: Dict[str, str] = {
    "clmux": "clmux",
    "tui": "clmux",
    "tmux": "clmux",
    "memora": "memora",
}

_GENERIC_TAGS_TO_PREFIX = {
    "plan", "analysis", "research", "architecture", "roadmap",
    "design", "status", "reference",
}

# Valid project prefixes for LLM-suggested tag filtering
_KNOWN_PROJECT_PREFIXES = tuple(f"{p}/" for p in _PROJECT_INDICATORS)


def _detect_project(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    context: Optional[str] = None,
) -> Optional[str]:
    """Detect which project content belongs to. Returns None if ambiguous or unknown."""
    text = content.lower()
    if metadata:
        section = str(metadata.get("section", "")).lower()
        meta_context = str(metadata.get("context", "")).lower()
        text = f"{text} {section} {meta_context}"
    if context:
        text = f"{text} {context.lower()}"

    matched = set()
    for project, patterns in _PROJECT_INDICATORS.items():
        if any(re.search(p, text) for p in patterns):
            matched.add(project)

    # If content is ambiguous or unknown, check tags for project hints
    if len(matched) != 1 and tags:
        tag_projects = set()
        for tag in tags:
            # Check direct tag match
            if tag in _TAG_PROJECT_MAP:
                tag_projects.add(_TAG_PROJECT_MAP[tag])
            # Check slash-prefixed tag (e.g., "memora/todos" → memora)
            if "/" in tag:
                prefix = tag.split("/", 1)[0]
                if prefix in _PROJECT_INDICATORS:
                    tag_projects.add(prefix)
            # Check hyphen-prefixed tag (e.g., "clmux-architecture" → clmux)
            if "-" in tag:
                hyphen_prefix = tag.split("-", 1)[0]
                if hyphen_prefix in _PROJECT_INDICATORS:
                    tag_projects.add(hyphen_prefix)
        if len(tag_projects) == 1:
            return tag_projects.pop()

    if len(matched) == 1:
        return matched.pop()
    return None  # ambiguous (multiple) or unknown (none)


def _normalize_tags(
    tags: List[str],
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Normalize generic tags to project-prefixed form when context is unambiguous.

    Idempotent: tags already containing '/' are never touched.
    Returns the normalized tag list.
    """
    if not tags:
        return tags

    project = _detect_project(content, metadata, tags)
    if not project:
        return tags

    normalized = []
    seen: set = set()
    for tag in tags:
        if tag in _GENERIC_TAGS_TO_PREFIX and "/" not in tag:
            prefixed = f"{project}/{tag}"
            if prefixed not in seen:
                normalized.append(prefixed)
                seen.add(prefixed)
        else:
            if tag not in seen:
                normalized.append(tag)
                seen.add(tag)
    return normalized


def _filter_suggested_tags(suggested: List[str]) -> List[str]:
    """Filter LLM-suggested tags to only known project prefixes + known suffixes."""
    filtered = []
    for tag in suggested:
        if not isinstance(tag, str) or "/" not in tag:
            continue
        prefix, _, suffix = tag.partition("/")
        if f"{prefix}/" in _KNOWN_PROJECT_PREFIXES and suffix in _GENERIC_TAGS_TO_PREFIX:
            filtered.append(tag)
    return filtered


def _auto_assign_section(
    metadata: Optional[Dict[str, Any]],
    content: str,
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Auto-assign metadata.section and subsection based on detected project and tags."""
    project = _detect_project(content, metadata, tags)
    if not project:
        return metadata

    has_section = metadata and metadata.get("section")
    has_subsection = metadata and metadata.get("subsection")

    if has_section and has_subsection:
        return metadata  # fully assigned

    updated = dict(metadata) if metadata else {}

    if not has_section:
        updated["section"] = project

    # Derive subsection from the most specific project-prefixed tag
    if not has_subsection and tags:
        prefix = f"{project}/"
        subsections = [
            tag[len(prefix):] for tag in tags
            if tag.startswith(prefix) and tag != project
        ]

        # Fallback: check bare tags as subsection candidates
        if not subsections:
            # Known topic tags that map to subsections
            _SUBSECTION_TAGS = {
                "tui", "architecture", "research", "roadmap", "bugfix",
                "design-decisions", "skills", "knowledge", "changelog",
                "overview", "risks",
            }
            subsections = [t for t in tags if t in _SUBSECTION_TAGS]

        if subsections:
            # Pick the most descriptive one (prefer non-type tags over issues/todos/sections)
            type_tags = {"issues", "todos", "sections"}
            content_subs = [s for s in subsections if s not in type_tags]
            best = content_subs[0] if content_subs else subsections[0]
            updated["subsection"] = best

    return updated


