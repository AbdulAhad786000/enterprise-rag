# Enterprise Knowledge Intelligence Platform

A local prototype that ingests company policy documents, retrieves evidence, and generates cited answers.

## Architecture and data flow

### Offline indexing

1. `app/ingestion/loader.py` finds PDF, DOCX, and Markdown files.
2. `app/ingestion/parser.py` extracts text and structure. PDF blocks retain page numbers; DOCX tables are converted to labeled rows; Markdown headings become section paths.
3. `app/ingestion/chunker.py` creates parent sections and smaller child passages. The child is the precise retrieval unit; its parent supplies surrounding context.
4. `app/embeddings/contextualize.py` prefixes document title, ID, department, version, date, and heading path to the embedding input. The original child text stays unchanged for citations.
5. `app/embeddings/gemini.py` embeds child passages with `gemini-embedding-2`.
6. `app/retrieval.py` normalizes vectors and saves them to FAISS, with chunk metadata in `chunks.json` and build information in `manifest.json`.

### Query and answer

1. The original question and a compact keyword version are embedded together.
2. FAISS searches both vectors. Reciprocal Rank Fusion combines the lists using `1 / (60 + rank)`, so it combines positions without comparing raw vector scores.
3. The optional local `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker reads each shortlisted question/passage pair together and sorts candidates by relevance. It runs locally on CPU; its first use downloads model weights, then later runs use the local cache. It is disabled by default because the current 15-question comparison showed no recall gain and slightly lower MRR/nDCG; keep it enabled only if later evaluation supports it.
4. Retrieval expands selected children to their parent sections and removes duplicate child passages.
5. Gemini generation uses only that evidence, with inline citation labels. Citation validation checks that labels exist and each answer sentence cites a supplied source. This is citation-reference validation, not proof that a claim is semantically supported.

There is deliberately no BM25 component in this architecture. Dense search, multi-query/RRF, and MiniLM reranking are the agreed retrieval stages.

## Gemini fallback

The generation order is configured in `.env`: one primary model and four fallbacks. A 429 rate/quota response skips that model immediately and places it on a five-minute cooldown persisted under `artifacts/retrieval/`, so the next CLI run also skips it. A transient 5xx response is retried once, then the next model is tried. A timeout/network failure moves to the next model. Authentication and malformed-request errors stop immediately because another model cannot fix them. The chosen model and failover reason appear in terminal progress.

Fallback applies to **generation**. Embeddings remain on one model because vectors from different embedding models must not be mixed with this FAISS index. Switching embedding models requires rebuilding the entire index.

## Setup (PowerShell)

Run commands from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Paste the Gemini key after `GEMINI_API_KEY=` and save. `.env` is ignored by Git. The current index was built with `gemini-embedding-001`, so after switching to Embedding 2, rebuild it before searching or generating answers. The sample `.env.example` is safe to commit. If `.env` already exists, do not overwrite it with `Copy-Item`.

## Commands

After the index already uses the configured embedding model, you can validate its saved FAISS artifacts and write a manifest offline (no Gemini call):

```powershell
python -m app.retrieval manifest
```

Build or refresh the FAISS index (calls Gemini embeddings for all chunks):

```powershell
python -m app.retrieval build
```

Search and show ranked evidence:

```powershell
python -m app.retrieval search "How many annual leave days are available?"
```

Compare dense, multi-query + RRF, and multi-query + RRF + MiniLM on the labeled question set:

```powershell
python -m app.retrieval evaluate
```

Generate a grounded answer:

```powershell
python -m app.generation "How many annual leave days are available?"
```

Run the local API, bound to localhost:

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the chat page. It keeps up to eight recent turns in browser memory while the tab stays open; short follow-up questions use the preceding user question to retrieve the right policy section. Use **New chat** to clear the conversation. The app also provides `GET /health`, `GET /v1/index`, `POST /v1/query` with `{"question":"..."}`, and `POST /v1/feedback` with a request ID and 1–5 rating. Interactive API docs are at `/docs`. This prototype has no authentication; keep it bound to localhost.

## Configuration

- `GEMINI_API_KEY`: local secret, never commit it.
- `GEMINI_EMBEDDING_MODEL` and `GEMINI_EMBEDDING_DIMENSIONS`: embedding-space configuration; changing either requires rebuilding every vector.
- `GEMINI_GENERATION_PRIMARY_MODEL` and `GEMINI_GENERATION_FALLBACK_MODELS`: ordered generation cascade.
- `GEMINI_REQUEST_TIMEOUT_MS`: timeout per Gemini HTTP request.
- `GEMINI_MODEL_COOLDOWN_SECONDS`: generation model cooldown after quota errors.
- `EVIDENCE_TOKEN_BUDGET`: maximum approximate source-evidence tokens sent to generation.
- `RRF_K`: RRF rank constant, default 60.
- `RERANKER_MODEL`, `RERANKER_CANDIDATES`, `ENABLE_RERANKER`: local reranking settings.

## Evaluation limits

Retrieval results depend on the labeled question set. The existing 15-question set is a small development set; its previous 1.000 metrics are not a general accuracy guarantee. Review answer correctness, source support, abstention, latency, and fallback frequency manually before making quality claims. Do not treat a reranker score as a probability.
