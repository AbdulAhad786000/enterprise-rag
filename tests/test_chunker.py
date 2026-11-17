
from pathlib import Path

from app.ingestion.chunker import (
    MAX_TOKENS,
    chunk_document,
    count_tokens,
    split_into_child_chunks,
)
from app.ingestion.pipeline import ingest_docx_directory


def test_child_chunks_respect_token_limit():
    text = " ".join(["employee policy benefits leave"] * 300)

    chunks = split_into_child_chunks(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert count_tokens(chunk) <= MAX_TOKENS


def test_parent_child_relationships():
    documents = ingest_docx_directory(Path("data/raw"))

    for document in documents:
        parents, children = chunk_document(document)

        parent_ids = {parent.parent_id for parent in parents}

        for child in children:
            assert child.parent_id in parent_ids
            assert child.document_id == document.metadata.document_id


def test_chunk_ids_are_stable():
    documents = ingest_docx_directory(Path("data/raw"))

    first_parents, first_children = chunk_document(documents[0])
    second_parents, second_children = chunk_document(documents[0])

    assert [p.parent_id for p in first_parents] == [
        p.parent_id for p in second_parents
    ]

    assert [c.child_id for c in first_children] == [
        c.child_id for c in second_children
    ]
