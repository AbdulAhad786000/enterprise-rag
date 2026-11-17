
import hashlib

import tiktoken

from app.ingestion.schemas import (
    ChildChunk,
    ParentChunk,
    ParsedDocument,
)


TARGET_TOKENS = 300
MAX_TOKENS = 400
OVERLAP_TOKENS = 40

# GPT cl100k_base is used as a practical approximation; Gemini token counts differ.
# Chunk limits are therefore approximate, not exact Gemini token limits.
ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the token count for a text string."""
    return len(ENCODING.encode(text))


def make_id(*parts: str) -> str:
    """Create a deterministic ID from input values."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def create_parent_chunks(
    document: ParsedDocument,
) -> list[ParentChunk]:
    """Group parsed blocks by section path into parent chunks."""

    grouped: dict[tuple[str, ...], list] = {}

    for block in document.blocks:
        section_key = tuple(block.section_path)
        grouped.setdefault(section_key, []).append(block)

    parents: list[ParentChunk] = []

    for section_path, section_blocks in grouped.items():
        parent_text = "\n".join(block.text for block in section_blocks).strip()
        page_numbers = [
            block.page_number for block in section_blocks
            if block.page_number is not None
        ]

        if not parent_text:
            continue

        parent_id = make_id(
            document.metadata.document_id,
            "/".join(section_path),
            parent_text,
        )

        parents.append(
            ParentChunk(
                parent_id=parent_id,
                document_id=document.metadata.document_id,
                text=parent_text,
                section_path=list(section_path),
                metadata=document.metadata,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
            )
        )

    return parents


def split_into_child_chunks(
    text: str,
) -> list[str]:
    """Split text into overlapping token-aware child chunks."""

    tokens = ENCODING.encode(text)

    if not tokens:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + TARGET_TOKENS, len(tokens))

        # Never exceed the maximum token limit.
        end = min(end, start + MAX_TOKENS)

        chunk_text = ENCODING.decode(tokens[start:end]).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start = end - OVERLAP_TOKENS

    return chunks


def create_child_chunks(
    parent: ParentChunk,
) -> list[ChildChunk]:
    """Create searchable child chunks from a parent."""

    text_chunks = split_into_child_chunks(parent.text)

    children: list[ChildChunk] = []

    for index, text in enumerate(text_chunks):
        child_id = make_id(parent.parent_id, str(index), text)

        children.append(
            ChildChunk(
                child_id=child_id,
                parent_id=parent.parent_id,
                document_id=parent.document_id,
                text=text,
                section_path=parent.section_path.copy(),
                metadata=parent.metadata,
                token_count=count_tokens(text),
                page_start=parent.page_start,
                page_end=parent.page_end,
            )
        )

    return children


def chunk_document(
    document: ParsedDocument,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Create parent and child chunks for one parsed document."""

    parents = create_parent_chunks(document)

    children: list[ChildChunk] = []

    for parent in parents:
        children.extend(create_child_chunks(parent))

    return parents, children
