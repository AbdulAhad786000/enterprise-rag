import faiss
import numpy as np

from app.ingestion.schemas import ChildChunk, DocumentMetadata, ParentChunk
from app.retrieval import DenseRetriever


class FixedEmbedder:
    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(queries), 1))


def test_bm25_can_promote_exact_identifier_match():
    metadata = DocumentMetadata(
        document_id="demo",
        title="Demo Policy",
        department="HR",
        version="1",
        effective_date="2025-01-01",
        source_file="demo.docx",
        file_type="docx",
    )
    texts = [
        "General benefits information.",
        "Annual leave policy HR-POL-004 allows 20 days.",
        "General workplace information and employee guidance.",
        "General travel information and employee guidance.",
        "General security information and employee guidance.",
    ]
    parents = [
        ParentChunk(f"p-{index}", "demo", text, [], metadata)
        for index, text in enumerate(texts)
    ]
    children = [
        ChildChunk(f"c-{index}", f"p-{index}", "demo", text, [], metadata, len(text.split()))
        for index, text in enumerate(texts)
    ]
    index = faiss.IndexFlatIP(2)
    vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    index.add(vectors)
    retriever = DenseRetriever(parents, children, index, FixedEmbedder())

    hits = retriever._fuse_query_rankings(
        "What is the annual leave policy HR-POL-004?",
        candidate_count=2,
        include_bm25=True,
    )

    assert hits[0].child.child_id == "c-1"
    assert hits[0].rrf_score is not None
