import os
from pathlib import Path
import time

import httpx2

import numpy as np
from google import genai
from google.genai import errors, types


def _load_project_env() -> None:
    """Load simple KEY=VALUE entries from the project-root .env file."""

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_project_env()


MODEL_NAME = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
OUTPUT_DIMENSIONALITY = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "3072"))
BATCH_SIZE = 16
REQUEST_TIMEOUT_MS = int(os.getenv("GEMINI_REQUEST_TIMEOUT_MS", "45000"))


class GeminiEmbedder:
    """Generate Gemini embeddings for retrieval documents and questions."""

    def __init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to the project-root .env file."
            )

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )

    def _embed(
        self,
        texts: list[str],
        task_type: str,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty list.")

        vectors: list[list[float]] = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]

            response = None
            for attempt in range(2):
                try:
                    if MODEL_NAME == "gemini-embedding-2":
                        if task_type == "RETRIEVAL_QUERY":
                            prepared = [
                                f"task: question answering | query: {text}"
                                for text in batch
                            ]
                        else:
                            prepared = batch
                        # Content objects keep one embedding per input in Embedding 2.
                        contents = [
                            types.Content(parts=[types.Part.from_text(text=text)])
                            for text in prepared
                        ]
                        config = types.EmbedContentConfig(
                            output_dimensionality=OUTPUT_DIMENSIONALITY
                        )
                    elif MODEL_NAME == "gemini-embedding-001":
                        contents = batch
                        config = types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=OUTPUT_DIMENSIONALITY,
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported Gemini embedding model: {MODEL_NAME}"
                        )

                    response = self.client.models.embed_content(
                        model=MODEL_NAME,
                        contents=contents,
                        config=config,
                    )
                    break
                except errors.APIError as exc:
                    if exc.code not in {429, 500, 502, 503, 504} or attempt == 1:
                        raise RuntimeError(
                            f"Embedding request failed with HTTP {exc.code}. "
                            "The index and query must use the same embedding model."
                        ) from exc
                    time.sleep(2**attempt)
                except httpx2.TransportError as exc:
                    if attempt == 1:
                        raise RuntimeError(
                            "Embedding request timed out or lost its network connection. "
                            "The index and query must use the same embedding model."
                        ) from exc
                    time.sleep(2**attempt)

            if response is None or not response.embeddings:
                raise RuntimeError("Gemini returned no embeddings.")

            vectors.extend(embedding.values for embedding in response.embeddings)

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} embeddings, received {len(vectors)}."
            )

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError("Gemini returned vectors with inconsistent dimensions.")
        result = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("Gemini returned an embedding with a non-finite value.")
        return result

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed child chunks for later retrieval."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed one or more query variants in the query vector space."""
        return self._embed(queries, task_type="RETRIEVAL_QUERY")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one user's question for retrieval."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        return self.embed_queries([query])