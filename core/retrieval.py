"""Retrieval: question -> embedding -> top-K similar chunks (Day 3).

Returns a ``RetrievalResponse`` whose ``status`` says clearly what happened,
so Day 4's answer generator can decide when to use the safe fallback instead
of guessing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .config import DEFAULT_MIN_SIMILARITY, DEFAULT_TOP_K
from .embeddings import Embedder
from .vector_store import ChromaVectorStore, SearchResult, VectorStoreError

STATUS_OK = "ok"  # at least one relevant chunk found
STATUS_EMPTY_QUERY = "empty_query"
STATUS_QUERY_TOO_LONG = "query_too_long"
STATUS_NO_DOCUMENTS = "no_documents"  # nothing indexed to search
STATUS_NO_RELEVANT = "no_relevant_results"  # chunks exist, but none similar enough

MAX_QUERY_CHARS = 1000


@dataclass
class RetrievalResponse:
    query: str
    status: str
    message: str = ""
    results: List[SearchResult] = field(default_factory=list)
    best_score: Optional[float] = None  # best score found, even if below the threshold
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SIMILARITY
    elapsed_seconds: float = 0.0

    @property
    def found_relevant(self) -> bool:
        return self.status == STATUS_OK and bool(self.results)


class Retriever:
    def __init__(
        self,
        store: ChromaVectorStore,
        embedder: Embedder,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SIMILARITY,
    ) -> None:
        if embedder.model_name != store.embedding_model:
            raise VectorStoreError(
                "The embedding model and the search index do not match "
                f"('{embedder.model_name}' vs '{store.embedding_model}')."
            )
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        doc_hashes: Optional[Sequence[str]] = None,
    ) -> RetrievalResponse:
        """Find the chunks most relevant to ``query``.

        ``doc_hashes`` limits the search to those documents (the app passes the
        current session's documents). Raises EmbeddingError / VectorStoreError
        if the model or the index fails.
        """
        top_k = self.top_k if top_k is None else top_k
        min_score = self.min_score if min_score is None else min_score
        text = query.strip() if isinstance(query, str) else ""
        response = RetrievalResponse(query=text, status=STATUS_OK, top_k=top_k, min_score=min_score)

        if not text:
            response.status = STATUS_EMPTY_QUERY
            response.message = "Please type a question first."
            return response
        if len(text) > MAX_QUERY_CHARS:
            response.status = STATUS_QUERY_TOO_LONG
            response.message = f"Please keep the question under {MAX_QUERY_CHARS} characters."
            return response

        stored = self.store.document_hashes()
        searchable = stored if doc_hashes is None else stored.intersection(doc_hashes)
        if not searchable:
            response.status = STATUS_NO_DOCUMENTS
            response.message = (
                "There are no indexed documents to search yet. "
                "Add documents and tap 'Create embeddings and index'."
            )
            return response

        started = time.time()
        vector = self.embedder.embed_query(text)
        found = self.store.search(vector, top_k=top_k, doc_hashes=sorted(searchable))
        response.elapsed_seconds = time.time() - started

        if found:
            response.best_score = found[0].score
        relevant = [r for r in found if r.score >= min_score]
        # Re-number so ranks stay 1..n after filtering.
        response.results = [
            SearchResult(**{**r.__dict__, "rank": i + 1}) for i, r in enumerate(relevant)
        ]
        if not relevant:
            response.status = STATUS_NO_RELEVANT
            response.message = (
                "No part of your documents seems related to this question."
            )
        return response
