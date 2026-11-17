import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import faiss
import numpy as np

from app.embeddings.contextualize import build_embedding_text
from app.embeddings.gemini import GeminiEmbedder, MODEL_NAME
from app.ingestion.schemas import (
    ChildChunk,
    DocumentMetadata,
    ParentChunk,
)


INDEX_DIR = Path("artifacts/retrieval")
RRF_K = int(os.getenv("RRF_K", "60"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "40"))
_RERANKER = None
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "me", "of",
    "on", "or", "please", "tell", "that", "the", "their", "this", "to",
    "what", "when", "where", "which", "who", "why", "with", "would",
}


def build_query_variants(query: str) -> list[str]:
    """Return original wording and a compact keyword form, with no extra API call."""
    original = " ".join(query.split())
    terms = re.findall(r"[\w./-]+", original, flags=re.UNICODE)
    keywords = [
        term for term in terms
        if term.casefold() not in _QUERY_STOPWORDS and len(term) > 1
    ]
    keyword_query = " ".join(keywords)
    if len(keywords) >= 2 and keyword_query.casefold() != original.casefold():
        return [original, keyword_query]
    return [original]


def _get_reranker():
    """Load the configured local cross-encoder once per process."""
    global _RERANKER
    if _RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "MiniLM reranking is enabled but sentence-transformers is missing. "
                "Install requirements.txt or set ENABLE_RERANKER=false in .env."
            ) from exc
        print(f"Loading local reranker: {RERANKER_MODEL}...", flush=True)
        _RERANKER = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _RERANKER


@dataclass
class ChildHit:
    """A child chunk with final and intermediate retrieval scores."""

    child: ChildChunk
    score: float
    rrf_score: float | None = None
    dense_score: float | None = None


@dataclass
class ParentContext:
    """A parent section plus the child chunks that matched the query."""

    parent: ParentChunk
    matched_children: list[ChildChunk]
    score: float


class DenseRetriever:
    """FAISS search over child chunks, with parent-section expansion."""

    def __init__(
        self,
        parents: list[ParentChunk],
        children: list[ChildChunk],
        index: faiss.Index,
        embedder: GeminiEmbedder,
    ) -> None:
        self.parents_by_id = {
            parent.parent_id: parent
            for parent in parents
        }
        self.children = children
        self.index = index
        self.embedder = embedder

        missing_parent_ids = {
            child.parent_id
            for child in children
            if child.parent_id not in self.parents_by_id
        }
        if missing_parent_ids:
            raise ValueError(
                f"Children refer to missing parents: {sorted(missing_parent_ids)}"
            )

        if self.index.ntotal != len(self.children):
            raise ValueError(
                "FAISS vector count does not match the number of child chunks."
            )

    @classmethod
    def build(
        cls,
        parents: list[ParentChunk],
        children: list[ChildChunk],
        embedder: GeminiEmbedder,
    ) -> "DenseRetriever":
        """Embed all child chunks and create the FAISS index."""

        if not children:
            raise ValueError("Cannot build a retrieval index with no children.")

        # Embed the contextualized version. Original child.text stays unchanged.
        embedding_texts = [
            build_embedding_text(child)
            for child in children
        ]

        vectors = embedder.embed_documents(embedding_texts)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or len(vectors) != len(children):
            raise ValueError("Embedding count or dimensions do not match child count.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding vectors contain a non-finite value.")

        # Normalized inner product gives cosine-style similarity in FAISS.
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        return cls(
            parents=parents,
            children=children,
            index=index,
            embedder=embedder,
        )

    def save(self, directory: Path) -> None:
        """Persist vectors, sidecar records, then a manifest for validation."""

        directory.mkdir(parents=True, exist_ok=True)
        index_temp = directory / "children.faiss.tmp"
        chunks_temp = directory / "chunks.json.tmp"
        manifest_temp = directory / "manifest.json.tmp"

        faiss.write_index(self.index, str(index_temp))
        payload = {
            "embedding_model": MODEL_NAME,
            "parents": [asdict(parent) for parent in self.parents_by_id.values()],
            "children": [asdict(child) for child in self.children],
        }
        chunks_temp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        child_ids = sorted(child.child_id for child in self.children)
        manifest = {
            "index_version": hashlib.sha256("\n".join(child_ids).encode("utf-8")).hexdigest()[:16],
            "built_at": datetime.now(timezone.utc).isoformat(),
            "corpus_hash": hashlib.sha256("\n".join(child_ids).encode("utf-8")).hexdigest(),
            "embedding_model": MODEL_NAME,
            "dimensions": int(self.index.d),
            "distance_metric": "inner_product",
            "normalized_vectors": True,
            "document_count": len({child.document_id for child in self.children}),
            "parent_count": len(self.parents_by_id),
            "child_count": len(self.children),
        }
        manifest_temp.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        index_temp.replace(directory / "children.faiss")
        chunks_temp.replace(directory / "chunks.json")
        manifest_temp.replace(directory / "manifest.json")

    @classmethod
    def load(
        cls,
        directory: Path,
        embedder: GeminiEmbedder,
    ) -> "DenseRetriever":
        """Load a saved FAISS index without re-embedding the documents."""

        payload = json.loads(
            (directory / "chunks.json").read_text(encoding="utf-8")
        )

        if payload["embedding_model"] != MODEL_NAME:
            raise ValueError(
                "The saved index uses a different embedding model. "
                "Rebuild it with the configured model before searching."
            )

        parents = []
        for item in payload["parents"]:
            item["metadata"] = DocumentMetadata(**item["metadata"])
            parents.append(ParentChunk(**item))

        children = []
        for item in payload["children"]:
            item["metadata"] = DocumentMetadata(**item["metadata"])
            children.append(ChildChunk(**item))

        index = faiss.read_index(
            str(directory / "children.faiss")
        )
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("embedding_model") != MODEL_NAME:
                raise ValueError("Index manifest embedding model does not match the configured model.")
            if manifest.get("child_count") != len(children) or manifest.get("child_count") != index.ntotal:
                raise ValueError("Index manifest, FAISS vectors, and child sidecar counts do not match.")
            if manifest.get("dimensions") != index.d:
                raise ValueError("Index manifest dimension does not match the FAISS index.")

        return cls(
            parents=parents,
            children=children,
            index=index,
            embedder=embedder,
        )

    def search_children(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[ChildHit]:
        """Embed a question and return the most similar child chunks."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.embedder.embed_query(query)
        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32,
        )
        if query_vector.ndim != 2 or query_vector.shape[1] != self.index.d:
            raise ValueError("Query vector dimension does not match the saved FAISS index.")
        faiss.normalize_L2(query_vector)

        count = min(top_k, len(self.children))
        scores, positions = self.index.search(query_vector, count)

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position >= 0:
                results.append(
                    ChildHit(
                        child=self.children[int(position)],
                        score=float(score),
                    )
                )

        return results

    def _fuse_query_rankings(
        self,
        query: str,
        candidate_count: int,
    ) -> list[ChildHit]:
        """Retrieve variants and combine their ranked lists with RRF."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if candidate_count < 1:
            raise ValueError("candidate_count must be at least 1.")

        variants = build_query_variants(query)
        vectors = np.ascontiguousarray(
            self.embedder.embed_queries(variants), dtype=np.float32
        )
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError("Query vector dimension does not match the saved FAISS index.")
        faiss.normalize_L2(vectors)
        count = min(candidate_count, len(self.children))
        dense_scores, positions = self.index.search(vectors, count)

        fused: dict[int, float] = {}
        best_dense: dict[int, float] = {}
        for row_scores, row_positions in zip(dense_scores, positions):
            for rank, (score, position) in enumerate(
                zip(row_scores, row_positions), start=1
            ):
                if position < 0:
                    continue
                pos = int(position)
                fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank)
                best_dense[pos] = max(best_dense.get(pos, float("-inf")), float(score))

        candidates = [
            ChildHit(
                child=self.children[pos],
                score=rrf,
                rrf_score=rrf,
                dense_score=best_dense[pos],
            )
            for pos, rrf in fused.items()
        ]
        candidates.sort(key=lambda hit: hit.rrf_score or 0.0, reverse=True)
        return candidates

    @staticmethod
    def rerank_candidates(query: str, candidates: list[ChildHit]) -> list[ChildHit]:
        """Reorder a dense/RRF shortlist with the local MiniLM cross-encoder."""
        if not candidates:
            return []
        shortlist = candidates[:min(RERANKER_CANDIDATES, len(candidates))]
        model = _get_reranker()
        pairs = [(query, hit.child.text) for hit in shortlist]
        scores = model.predict(pairs, show_progress_bar=False)
        reranked = [
            ChildHit(
                child=hit.child,
                score=float(score),
                rrf_score=hit.rrf_score,
                dense_score=hit.dense_score,
            )
            for hit, score in zip(shortlist, scores)
        ]
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return reranked + candidates[len(shortlist):]

    def search_children_fused(
        self,
        query: str,
        candidate_count: int = 20,
        use_reranker: bool | None = None,
    ) -> list[ChildHit]:
        """Fuse query-variant rankings with RRF and optionally rerank them."""
        candidates = self._fuse_query_rankings(query, candidate_count)
        rerank_enabled = ENABLE_RERANKER if use_reranker is None else use_reranker
        if rerank_enabled:
            candidates = self.rerank_candidates(query, candidates)
        return candidates[:candidate_count]

    def retrieve_with_parents(
        self,
        query: str,
        candidate_count: int = 20,
        max_parents: int = 5,
    ) -> list[ParentContext]:
        """Find children, then return their unique parent sections."""

        child_hits = self.search_children_fused(
            query=query,
            candidate_count=candidate_count,
        )

        grouped: dict[str, ParentContext] = {}

        for hit in child_hits:
            parent_id = hit.child.parent_id

            if parent_id not in grouped:
                grouped[parent_id] = ParentContext(
                    parent=self.parents_by_id[parent_id],
                    matched_children=[],
                    score=hit.score,
                )

            result = grouped[parent_id]
            result.matched_children.append(hit.child)
            result.score = max(result.score, hit.score)

        # Dict order preserves the order in which parents first appeared
        # in the ranked child results.
        return list(grouped.values())[:max_parents]



def write_manifest_from_existing_index(directory: Path = INDEX_DIR) -> dict:
    """Validate saved FAISS/sidecar artifacts and write their manifest offline."""
    index_path = directory / "children.faiss"
    chunks_path = directory / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Saved FAISS index and chunks.json are both required.")
    index = faiss.read_index(str(index_path))
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    children = payload.get("children", [])
    parents = payload.get("parents", [])
    if index.ntotal != len(children):
        raise ValueError("FAISS vector count does not match chunks.json child count.")
    if payload.get("embedding_model") != MODEL_NAME:
        raise ValueError("Saved sidecar embedding model does not match this code.")
    child_ids = sorted(child["child_id"] for child in children)
    content_hash = hashlib.sha256("\n".join(child_ids).encode("utf-8")).hexdigest()
    manifest = {
        "index_version": content_hash[:16],
        "built_at": datetime.fromtimestamp(index_path.stat().st_mtime, timezone.utc).isoformat(),
        "manifest_written_at": datetime.now(timezone.utc).isoformat(),
        "corpus_hash": content_hash,
        "embedding_model": MODEL_NAME,
        "dimensions": int(index.d),
        "distance_metric": "inner_product",
        "normalized_vectors": True,
        "document_count": len({child["document_id"] for child in children}),
        "parent_count": len(parents),
        "child_count": len(children),
    }
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    return manifest

def build_index() -> None:
    """Run ingestion and chunking, then build and save the vector index."""

    from app.ingestion.chunker import chunk_document
    from app.ingestion.pipeline import ingest_directory

    documents = ingest_directory(Path("data/raw"))

    parents = []
    children = []

    for document in documents:
        document_parents, document_children = chunk_document(document)
        parents.extend(document_parents)
        children.extend(document_children)

    retriever = DenseRetriever.build(
        parents=parents,
        children=children,
        embedder=GeminiEmbedder(),
    )
    retriever.save(INDEX_DIR)

    print(f"Documents: {len(documents)}")
    print(f"Parents: {len(parents)}")
    print(f"Children: {len(children)}")
    print(f"FAISS vectors: {retriever.index.ntotal}")
    print(f"Index saved in: {INDEX_DIR}")


def search_index(query: str, top_k: int, max_parents: int) -> None:
    """Load the saved index and print child matches with parent context."""

    retriever = DenseRetriever.load(
        directory=INDEX_DIR,
        embedder=GeminiEmbedder(),
    )

    results = retriever.retrieve_with_parents(
        query=query,
        candidate_count=top_k,
        max_parents=max_parents,
    )

    for rank, result in enumerate(results, start=1):
        section = " > ".join(result.parent.section_path) or "(no section heading)"

        print(f"\nResult {rank} | rerank/RRF score: {result.score:.4f}")
        print(f"Document: {result.parent.metadata.title}")
        print(f"Section: {section}")
        print("Matched child:")
        print(result.matched_children[0].text)
        print("Parent section context:")
        print(result.parent.text)


def _metric_totals(
    ranked_ids: list[str],
    relevant_ids: set[str],
    count: int,
) -> tuple[float, float, float, float]:
    hit = float(bool(relevant_ids.intersection(ranked_ids[:5])))
    reciprocal_rank = 0.0
    for rank, child_id in enumerate(ranked_ids[:5], start=1):
        if child_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break
    recall = len(relevant_ids.intersection(ranked_ids[:10])) / len(relevant_ids)
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, child_id in enumerate(ranked_ids[:10], start=1)
        if child_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), 10)
    ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return hit / count, reciprocal_rank / count, recall / count, ndcg / count


def evaluate_index(questions_path: Path) -> None:
    """Compare dense, multi-query/RRF, and reranked retrieval."""

    retriever = DenseRetriever.load(
        directory=INDEX_DIR,
        embedder=GeminiEmbedder(),
    )
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("The evaluation question set is empty.")

    totals = {
        "Dense baseline": [0.0, 0.0, 0.0, 0.0],
        "Multi-query + RRF": [0.0, 0.0, 0.0, 0.0],
        "Multi-query + RRF + MiniLM": [0.0, 0.0, 0.0, 0.0],
    }
    count = len(questions)
    for item in questions:
        relevant_ids = set(item["relevant_child_ids"])
        if not relevant_ids:
            raise ValueError(f"No relevant child IDs for: {item['query']}")

        query = item["query"]
        dense_hits = retriever.search_children(query, top_k=20)
        rrf_pool = retriever._fuse_query_rankings(query, candidate_count=20)
        rrf_hits = rrf_pool[:20]
        reranked_hits = retriever.rerank_candidates(query, rrf_pool)[:20]
        ranked_sets = (
            ("Dense baseline", dense_hits),
            ("Multi-query + RRF", rrf_hits),
            ("Multi-query + RRF + MiniLM", reranked_hits),
        )
        for label, hits in ranked_sets:
            ranked_ids = [hit.child.child_id for hit in hits]
            values = _metric_totals(ranked_ids, relevant_ids, count)
            for index, value in enumerate(values):
                totals[label][index] += value

    metric_names = ("Hit Rate@5", "MRR@5", "Recall@10", "nDCG@10")
    print(f"Questions: {count}")
    for label, values in totals.items():
        print(f"{label}:")
        for name, value in zip(metric_names, values):
            print(f"  {name}: {value:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or search the local RAG retrieval index."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "build",
        help="Create document embeddings and save the FAISS index.",
    )

    search_parser = commands.add_parser(
        "search",
        help="Search the saved index with a question.",
    )
    search_parser.add_argument("query", help="The question to search for.")
    search_parser.add_argument("--top-k", type=int, default=20)
    search_parser.add_argument("--max-parents", type=int, default=5)

    commands.add_parser("manifest", help="Validate saved index artifacts and write a manifest offline.")

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Compare dense and enhanced retrieval against labeled questions.",
    )
    evaluate_parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/retrieval_questions.json"),
    )

    args = parser.parse_args()

    if args.command == "build":
        build_index()
    elif args.command == "search":
        search_index(
            query=args.query,
            top_k=args.top_k,
            max_parents=args.max_parents,
        )
    elif args.command == "evaluate":
        evaluate_index(args.questions)
    elif args.command == "manifest":
        print(json.dumps(write_manifest_from_existing_index(INDEX_DIR), indent=2))
if __name__ == "__main__":
    main()
