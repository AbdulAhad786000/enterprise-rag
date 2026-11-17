"""Local FastAPI wrapper for policy retrieval and grounded answers."""

from contextlib import asynccontextmanager
import json
import logging
import re
import time
from uuid import uuid4
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.embeddings.gemini import GeminiEmbedder
from app.generation import GeminiAnswerGenerator
from app.retrieval import DenseRetriever, INDEX_DIR


logger = logging.getLogger("enterprise_rag.api")
FRONTEND_FILE = Path(__file__).with_name("static") / "index.html"


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1500)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    candidate_count: int = Field(default=20, ge=1, le=60)
    max_parents: int = Field(default=5, ge=1, le=10)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=8)


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    rating: int = Field(ge=1, le=5)
    correction: str | None = Field(default=None, max_length=500)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (INDEX_DIR / "children.faiss").exists() or not (INDEX_DIR / "chunks.json").exists():
        raise RuntimeError("Retrieval index is missing. Build it with `python -m app.retrieval build`.")
    embedder = GeminiEmbedder()
    app.state.retriever = DenseRetriever.load(INDEX_DIR, embedder)
    app.state.generator = GeminiAnswerGenerator()
    yield


app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    version="0.1.0",
    description="Local, evidence-grounded answers over the configured policy corpus.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the local question-and-answer page."""
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend page is not installed.")
    return FileResponse(FRONTEND_FILE, media_type="text/html")


@app.get("/health")
def health() -> dict:
    """Report whether the local index artifacts are present and validated."""
    manifest_path = INDEX_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_ready = (INDEX_DIR / "children.faiss").exists() and (INDEX_DIR / "chunks.json").exists()
    return {
        "status": "ready" if index_ready else "not_ready",
        "index_ready": index_ready,
        "index_version": manifest.get("index_version"),
        "child_count": manifest.get("child_count"),
    }


@app.get("/v1/index")
def index_info() -> dict:
    """Return safe index metadata without local source paths."""
    manifest_path = INDEX_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Index manifest is unavailable; rebuild the index.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@app.post("/v1/query")
def query(request: QueryRequest) -> dict:
    """Retrieve evidence, generate a grounded answer, and return citations."""
    request_id = str(uuid4())
    request_started = time.perf_counter()
    retrieval_seconds = 0.0
    try:
        retrieval_started = time.perf_counter()
        previous_user_questions = [
            message.content
            for message in request.history
            if message.role == "user"
        ]
        retrieval_question = request.question
        short_follow_up = len(request.question.split()) <= 8
        if previous_user_questions and short_follow_up:
            retrieval_question = (
                f"{previous_user_questions[-1]}\nFollow-up question: {request.question}"
            )

        contexts = app.state.retriever.retrieve_with_parents(
            query=retrieval_question,
            candidate_count=request.candidate_count,
            max_parents=request.max_parents,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
        result = app.state.generator.answer(
            request.question,
            contexts,
            conversation_history=[message.model_dump() for message in request.history],
        )
    except Exception as exc:
        logger.exception("query_failed request_id=%s", request_id)
        # Keep provider details, prompts, filesystem paths, and secrets out of responses.
        raise HTTPException(
            status_code=503,
            detail=f"The query could not be completed. Reference: {request_id}",
        ) from exc

    referenced_ids = set(re.findall(r"\[(S\d+)\]", result.answer))
    citations = [
        {
            "citation_id": source.citation_id,
            "child_id": source.child_id,
            "document_id": source.document_id,
            "title": source.title,
            "section": source.section,
            "page_start": source.page_start,
            "page_end": source.page_end,
        }
        for source in result.sources
        if source.citation_id in referenced_ids
    ]
    logger.info(
        "query_completed request_id=%s model=%s retrieval_seconds=%.3f "
        "total_seconds=%.3f contexts=%d citations_valid=%s",
        request_id,
        result.model_used,
        retrieval_seconds,
        time.perf_counter() - request_started,
        len(contexts),
        result.citation_validation.valid,
    )
    return {
        "request_id": request_id,
        "answer": result.answer,
        "abstained": result.model_used == "not_called",
        "citations": citations,
        "citation_validation": result.citation_validation.valid,
        "model": result.model_used,
        "index_version": _index_version(),
    }


def _index_version() -> str | None:
    manifest_path = INDEX_DIR / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8")).get("index_version")


@app.post("/v1/feedback")
def feedback(request: FeedbackRequest) -> dict:
    """Append a small rating/correction record for prototype review."""
    record = {
        "request_id": request.request_id,
        "rating": request.rating,
        "correction": request.correction,
        "received_at": time.time(),
    }
    feedback_path = INDEX_DIR.parent / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"accepted": True}
