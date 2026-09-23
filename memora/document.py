"""Structured document parser and chunker for Memora.

Parses markdown documents into a root memory + typed fragment memories,
preserving document structure while making individual pieces searchable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Fragment:
    """A single document fragment ready for storage."""
    content: str
    node_kind: str  # claim, plan_item, reference, section_chunk, risk
    ordinal: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentPlan:
    """Parsed document ready for batch storage."""
    root_content: str
    root_metadata: Dict[str, Any]
    root_tags: List[str]
    fragments: List[Fragment]
    # memora's own typed tags ("<project>/documents"): stored on root and
    # fragments, exempt from the tag allowlist (issue #47).
    system_tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")
_URL_RE = re.compile(r"https?://[^\s\)>]+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_NUMBERED_ITEM_RE = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*[—–-]\s*(.+)", re.DOTALL)
_BOLD_ITEM_RE = re.compile(r"^-\s+\*\*(.+?)\*\*\s*[—–-]\s*(.+)", re.DOTALL)
_RISK_KEYWORDS = {"risk", "question", "caveat", "concern", "unknown", "unresolved", "open"}


def parse_document(
    content: str,
    document_key: str,
    version: int = 1,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_fragment_crossrefs: bool = True,
    project: Optional[str] = None,
) -> DocumentPlan:
    """Parse a markdown document into a root + typed fragments.

    Chunking is structure-aware:
    - Split by markdown headings
    - Tables: one section_chunk for raw table + one claim per data row
    - Numbered lists: one plan_item per item
    - Reference lists (URLs): one reference per URL
    - Long prose: split by paragraph (>2000 chars threshold)
    - Atomic units (table rows, items, refs) are never overlapped

    Args:
        content: Full markdown document text
        document_key: Stable document identifier
        version: Document version number
        tags: Tags to apply to root and fragments
        metadata: Extra metadata merged into each fragment
        skip_fragment_crossrefs: Whether fragments skip crossref computation
        project: The document's project: the root tag is "<project>/documents",
            or bare "documents" without one (issue #47)

    Returns:
        DocumentPlan with root info and ordered fragment list
    """
    tags = tags or []
    metadata = metadata or {}

    # Build root metadata
    root_metadata = {
        "type": "document_root",
        "document_key": document_key,
        "document_version": version,
        "hierarchy": {"path": document_key.split("/")},
        **metadata,
    }
    documents_tag = f"{project}/documents" if project else "documents"
    root_tags = [t for t in tags if t != documents_tag]
    system_tags = [documents_tag]

    # Split document into heading-delimited sections
    sections = _split_by_headings(content)

    # Parse each section into fragments
    fragments: List[Fragment] = []
    ordinal = 0

    for section in sections:
        heading = section["heading"]
        body = section["body"].strip()
        if not body:
            continue

        section_frags = _parse_section(body, heading)
        for frag in section_frags:
            ordinal += 1
            frag.ordinal = ordinal
            # Apply shared metadata + fragment identity
            frag.metadata.update({
                "type": "document_fragment",
                "document_key": document_key,
                "document_version": version,
                "node_kind": frag.node_kind,
                "ordinal": frag.ordinal,
                "hierarchy": {"path": document_key.split("/")},
            })
            if skip_fragment_crossrefs:
                frag.metadata["indexing"] = {"skip_crossrefs": True}
            # Merge caller metadata (without overwriting fragment-specific keys)
            for k, v in metadata.items():
                if k not in frag.metadata:
                    frag.metadata[k] = v
            fragments.append(frag)

    # Update root with fragment count
    root_metadata["fragment_count"] = len(fragments)

    return DocumentPlan(
        root_content=content,
        root_metadata=root_metadata,
        root_tags=root_tags,
        fragments=fragments,
        system_tags=system_tags,
    )


def _split_by_headings(content: str) -> List[Dict[str, str]]:
    """Split markdown into sections by headings.

    Returns list of {heading: str, body: str} dicts.
    The first section (before any heading) gets heading="".
    """
    sections: List[Dict[str, str]] = []
    matches = list(_HEADING_RE.finditer(content))

    if not matches:
        return [{"heading": "", "body": content}]

    # Content before first heading
    pre = content[:matches[0].start()].strip()
    if pre:
        sections.append({"heading": "", "body": pre})

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        sections.append({"heading": heading, "body": body})

    return sections


def _parse_section(body: str, heading: str) -> List[Fragment]:
    """Parse a section body into typed fragments.

    Tries specialized parsers first, then falls back to prose chunking.
    Mixed-content sections (e.g., prose + numbered list) are split: the
    specialized parser handles its portion, remaining prose becomes chunks.
    """
    fragments: List[Fragment] = []

    # Check if body contains a table — tables get special treatment
    lines = body.split("\n")
    table_lines = [l for l in lines if _TABLE_ROW_RE.match(l.strip())]

    if len(table_lines) >= 3:  # header + separator + at least 1 data row
        fragments.extend(_parse_table_section(body, heading))
        return fragments

    # Check if heading indicates a risk/open-question section
    risk_items, remaining_prose = _extract_risks(body, heading)
    if risk_items:
        fragments.extend(risk_items)
        if remaining_prose.strip():
            fragments.extend(_chunk_prose(remaining_prose.strip(), heading))
        return fragments

    # For numbered lists and references, extract the structured items
    # and chunk any remaining prose separately
    numbered_items, remaining_prose = _extract_numbered_items_with_remainder(body)
    if numbered_items:
        fragments.extend(numbered_items)
        if remaining_prose.strip():
            fragments.extend(_chunk_prose(remaining_prose.strip(), heading))
        return fragments

    ref_items, remaining_prose = _extract_references_with_remainder(body)
    if ref_items:
        fragments.extend(ref_items)
        if remaining_prose.strip():
            fragments.extend(_chunk_prose(remaining_prose.strip(), heading))
        return fragments

    # Default: section_chunk, possibly split if long
    fragments.extend(_chunk_prose(body, heading))
    return fragments


def _parse_table_section(body: str, heading: str) -> List[Fragment]:
    """Parse a section containing a markdown table.

    Creates one section_chunk for the full table + one claim per data row.
    """
    fragments: List[Fragment] = []
    lines = body.split("\n")

    # Find table boundaries
    table_start = None
    table_end = None
    header_cells: List[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if _TABLE_ROW_RE.match(stripped):
            if table_start is None:
                table_start = i
                header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            table_end = i
        elif table_start is not None and not _TABLE_SEPARATOR_RE.match(stripped):
            if not stripped:
                continue
            break

    if table_start is None:
        return [Fragment(content=body, node_kind="section_chunk", ordinal=0,
                         metadata={"section_heading": heading})]

    # Full table as section_chunk
    table_text = "\n".join(lines[table_start:table_end + 1])
    fragments.append(Fragment(
        content=f"{heading}\n\n{table_text}" if heading else table_text,
        node_kind="section_chunk",
        ordinal=0,
        metadata={"section_heading": heading},
    ))

    # Individual data rows as claims
    separator_seen = False
    for line in lines[table_start:table_end + 1]:
        stripped = line.strip()
        if _TABLE_SEPARATOR_RE.match(stripped):
            separator_seen = True
            continue
        if not separator_seen:
            continue  # skip header row
        if not _TABLE_ROW_RE.match(stripped):
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        claim_content = _format_claim_from_cells(header_cells, cells)
        if claim_content:
            claim_meta: Dict[str, Any] = {"section_heading": heading}
            # Try to extract confidence from cells
            for cell in cells:
                low = cell.lower().strip("*").strip()
                if low in ("high", "medium", "low"):
                    claim_meta["confidence"] = low
                    break
            fragments.append(Fragment(
                content=claim_content,
                node_kind="claim",
                ordinal=0,
                metadata=claim_meta,
            ))

    # Non-table content in the section
    non_table = []
    if table_start > 0:
        non_table.append("\n".join(lines[:table_start]).strip())
    if table_end is not None and table_end + 1 < len(lines):
        non_table.append("\n".join(lines[table_end + 1:]).strip())
    for chunk in non_table:
        if chunk:
            fragments.extend(_chunk_prose(chunk, heading))

    return fragments


def _format_claim_from_cells(headers: List[str], cells: List[str]) -> str:
    """Format a table row as a structured claim string."""
    if not cells or all(not c.strip() for c in cells):
        return ""

    parts = []
    for i, cell in enumerate(cells):
        cell = cell.strip()
        if not cell:
            continue
        # Strip markdown bold
        cell = re.sub(r"\*\*(.+?)\*\*", r"\1", cell)
        header = headers[i].strip() if i < len(headers) else f"Column {i + 1}"
        header = re.sub(r"\*\*(.+?)\*\*", r"\1", header)
        parts.append(f"{header}: {cell}")

    return "\n".join(parts) if parts else ""


_PLAIN_NUMBERED_RE = re.compile(r"^(\d+)\.\s+(.+)", re.DOTALL)


def _extract_numbered_items_with_remainder(body: str) -> tuple[List[Fragment], str]:
    """Extract numbered list items as plan_item fragments.

    Supports both bold-style (1. **Title** — desc) and plain (1. Text) items.
    Returns (items, remaining_prose) so callers can handle mixed content.
    """
    items: List[Fragment] = []
    current_num: Optional[int] = None
    current_lines: List[str] = []
    pre_lines: List[str] = []  # prose before the first numbered item
    post_lines: List[str] = []  # prose after the last numbered item
    finished_items = False  # True once we see non-continuation after items

    for line in body.split("\n"):
        stripped = line.strip()

        if finished_items:
            post_lines.append(line)
            continue

        # Try bold-style first, then plain numbered
        match = _NUMBERED_ITEM_RE.match(stripped) or _PLAIN_NUMBERED_RE.match(stripped)
        if match:
            if current_num is not None and current_lines:
                items.append(_make_plan_item(current_num, "\n".join(current_lines)))
            current_num = int(match.group(1))
            current_lines = [stripped]
        elif current_num is not None:
            # Check if this is continuation (indented/blank) or new content
            if not stripped or line.startswith("   ") or line.startswith("\t"):
                current_lines.append(line)
            else:
                # Non-continuation line after items — flush and switch to post
                items.append(_make_plan_item(current_num, "\n".join(current_lines)))
                current_num = None
                current_lines = []
                finished_items = True
                post_lines.append(line)
        else:
            # Content before any numbered item
            pre_lines.append(line)

    if current_num is not None and current_lines:
        items.append(_make_plan_item(current_num, "\n".join(current_lines)))

    # Only treat as plan items if we found at least 2
    if len(items) < 2:
        return [], body

    remainder_parts = []
    pre = "\n".join(pre_lines).strip()
    post = "\n".join(post_lines).strip()
    if pre:
        remainder_parts.append(pre)
    if post:
        remainder_parts.append(post)
    remainder = "\n\n".join(remainder_parts)
    return items, remainder


def _make_plan_item(num: int, text: str) -> Fragment:
    """Create a plan_item fragment from a numbered item."""
    return Fragment(
        content=f"Plan item {num}: {text}",
        node_kind="plan_item",
        ordinal=0,
        metadata={"item_number": num},
    )


def _extract_references_with_remainder(body: str) -> tuple[List[Fragment], str]:
    """Extract URL-containing list items as reference fragments.

    Handles multiple markdown links per line. Returns (refs, remaining_prose).
    """
    refs: List[Fragment] = []
    non_ref_lines: List[str] = []

    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        md_links = _MD_LINK_RE.findall(stripped)
        bare_urls = _URL_RE.findall(stripped)

        if md_links:
            # Create one reference per markdown link on this line
            for title, url in md_links:
                # Get context around this specific link
                remainder = _MD_LINK_RE.sub("", stripped).strip(" -—–|,;")
                content = f"Reference: {title}\nURL: {url}"
                if remainder:
                    content += f"\nRelevance: {remainder}"
                refs.append(Fragment(
                    content=content,
                    node_kind="reference",
                    ordinal=0,
                    metadata={"source_urls": [url]},
                ))
        elif bare_urls:
            for url in bare_urls:
                content = f"Reference: {stripped}\nURL: {url}"
                refs.append(Fragment(
                    content=content,
                    node_kind="reference",
                    ordinal=0,
                    metadata={"source_urls": [url]},
                ))
        else:
            non_ref_lines.append(line)

    # Only treat as reference section if majority of lines have URLs
    if refs and len(non_ref_lines) <= len(refs):
        remainder = "\n".join(non_ref_lines)
        return refs, remainder
    return [], body


def _extract_risks(body: str, heading: str) -> tuple[List[Fragment], str]:
    """Extract risk/open question items.

    Returns (risks, remaining_prose) so trailing prose is not dropped.
    """
    heading_lower = heading.lower()
    is_risk_section = any(kw in heading_lower for kw in _RISK_KEYWORDS)

    if not is_risk_section:
        return [], body

    risks: List[Fragment] = []
    current_lines: List[str] = []
    post_lines: List[str] = []
    in_items = False  # True once we've seen the first bullet
    finished_items = False

    for line in body.split("\n"):
        stripped = line.strip()

        if finished_items:
            post_lines.append(line)
            continue

        # New bullet item
        if stripped.startswith("- **") or stripped.startswith("- ["):
            in_items = True
            if current_lines:
                risks.append(Fragment(
                    content="\n".join(current_lines),
                    node_kind="risk",
                    ordinal=0,
                    metadata={"section_heading": heading},
                ))
            current_lines = [stripped]
        elif in_items and current_lines:
            # Continuation lines (indented or blank)
            if not stripped or line.startswith("  ") or line.startswith("\t"):
                current_lines.append(line)
            else:
                # Non-continuation — flush current item, collect remainder
                risks.append(Fragment(
                    content="\n".join(current_lines),
                    node_kind="risk",
                    ordinal=0,
                    metadata={"section_heading": heading},
                ))
                current_lines = []
                finished_items = True
                post_lines.append(line)
        # Skip preamble text before the first bullet

    if current_lines:
        risks.append(Fragment(
            content="\n".join(current_lines),
            node_kind="risk",
            ordinal=0,
            metadata={"section_heading": heading},
        ))

    if len(risks) < 2:
        return [], body

    remainder = "\n".join(post_lines).strip()
    return risks, remainder


_MAX_CHUNK_CHARS = 2000
_OVERLAP_CHARS = 100


def _chunk_prose(body: str, heading: str) -> List[Fragment]:
    """Split prose into section_chunk fragments.

    Short sections (<2000 chars) become a single chunk.
    Long sections are split by paragraph with ~100 char overlap.
    """
    if len(body) <= _MAX_CHUNK_CHARS:
        content = f"{heading}\n\n{body}" if heading else body
        return [Fragment(
            content=content,
            node_kind="section_chunk",
            ordinal=0,
            metadata={"section_heading": heading},
        )]

    # Split by paragraphs (double newline)
    paragraphs = re.split(r"\n{2,}", body)
    chunks: List[Fragment] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_len + len(para) > _MAX_CHUNK_CHARS and current:
            chunk_text = "\n\n".join(current)
            prefix = f"{heading}\n\n" if heading and not chunks else ""
            chunks.append(Fragment(
                content=prefix + chunk_text,
                node_kind="section_chunk",
                ordinal=0,
                metadata={"section_heading": heading, "chunk_index": len(chunks)},
            ))
            # Overlap: keep last paragraph's tail
            last = current[-1]
            overlap = last[-_OVERLAP_CHARS:] if len(last) > _OVERLAP_CHARS else last
            current = [overlap, para]
            current_len = len(overlap) + len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunk_text = "\n\n".join(current)
        chunks.append(Fragment(
            content=chunk_text,
            node_kind="section_chunk",
            ordinal=0,
            metadata={"section_heading": heading, "chunk_index": len(chunks)},
        ))

    return chunks
