# Submission validation report

**Verified:** 2026-11-17  
**Project:** Enterprise Knowledge Intelligence Platform  
**Environment:** Local Windows virtual environment, five policy documents, 44 child passages

## Automated tests

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
8 passed, 10 subtests passed
```

The suite covers chunk limits, stable chunk IDs, parent-child links, DOCX parsing, PDF typography-based section paths, and exact-term BM25 ranking.

## Retrieval evaluation

Command:

```powershell
.\.venv\Scripts\python.exe -m app.retrieval evaluate
```

The evaluation used 15 labeled queries from `data/evaluation/retrieval_questions.json`.

| Retrieval method | Hit Rate@5 | MRR@5 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense baseline | 1.000 | 1.000 | 1.000 | 1.000 |
| Multi-query dense + RRF | 1.000 | 1.000 | 1.000 | 1.000 |
| Dense + BM25 + RRF | 1.000 | 1.000 | 1.000 | 1.000 |
| Dense + BM25 + RRF + MiniLM | 1.000 | 0.967 | 1.000 | 0.975 |

On this small evaluation set, BM25 tied the dense baseline; it did not increase these aggregate metrics. MiniLM reranking slightly reduced ranking quality, so it remains off by default. These results should not be treated as a guarantee on unseen documents or queries.

## Grounded answer example

Question: `How many annual leave days are available?`

The CLI retrieved five parent sections and generated an answer using the configured Gemini fallback chain. It ended on `gemini-3.5-flash-lite` after temporary server errors from earlier models. Citation reference validation returned `True`.

```text
Every full-time employee receives 20 annual leave days per calendar year [S1].
Part-time and contract employee entitlements are determined by their individual
employment agreements [S1].

[S1] HR Employee Handbook | 3. Annual Leave Policy
```

## Local chat/API check

- `GET /health`: `ready`, 44 child passages.
- `GET /`: HTTP 200, chat page served.
- Follow-up question `What about part-time?` with the annual-leave question in conversation history returned the relevant employment-agreement answer with citation validation `True`.
- API generation used `gemini-3.5-flash-lite` after the configured fallback chain.

## Reproduce for a live demo

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m app.retrieval evaluate
python -m app.generation "How many annual leave days are available?"
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/` and ask the sample question. For a fresh clone, first install `requirements.txt`, create `.env` with a Gemini API key, add policy documents to `data/raw/`, and build the index with `python -m app.retrieval build`. The raw policy documents, `.env`, and generated index are excluded from this public repository.
