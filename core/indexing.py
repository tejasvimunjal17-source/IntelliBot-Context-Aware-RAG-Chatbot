"""Indexing: chunks -> embeddings -> ChromaDB (Day 3).

Connects the Day 2 pipeline (chunking + embeddings) to the vector store.
Documents that are already stored unchanged are skipped, so they are not
embedded or written again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .chunking import Chunk
from .embeddings import Embedder, EmbeddingCache, embed_chunks
from .vector_store import ChromaVectorStore, VectorStoreError


@dataclass
class IndexReport:
    documents_total: int = 0
    documents_indexed: int = 0  # new or changed -> embedded and stored
    documents_unchanged: int = 0  # already in the store -> skipped
    chunks_total: int = 0  # chunks in all given documents
    chunks_written: int = 0
    newly_embedded: int = 0  # unique texts sent to the embedding model
    reused_vectors: int = 0  # vectors taken from the in-memory cache
    dimension: int = 0
    store_total_chunks: int = 0  # chunks now in the store (all documents)


def index_chunks(
    chunks: Sequence[Chunk],
    store: ChromaVectorStore,
    embedder: Embedder,
    cache: Optional[EmbeddingCache] = None,
) -> IndexReport:
    """Embed and store chunks. Raises EmbeddingError / VectorStoreError on failure."""
    if embedder.model_name != store.embedding_model:
        raise VectorStoreError(
            "The embedding model and the search index do not match "
            f"('{embedder.model_name}' vs '{store.embedding_model}')."
        )
    report = IndexReport(chunks_total=len(chunks))
    by_document: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.doc_hash, []).append(chunk)
    report.documents_total = len(by_document)

    stale: List[Chunk] = []
    for doc_chunks in by_document.values():
        if store.is_document_current(doc_chunks):
            report.documents_unchanged += 1
        else:
            stale.extend(doc_chunks)

    if stale:
        embedded = embed_chunks(stale, embedder, cache)
        added = store.add_embedded_chunks(embedded.items)
        report.documents_indexed = added.documents_added + added.documents_replaced
        report.documents_unchanged += added.documents_unchanged
        report.chunks_written = added.chunks_written
        report.newly_embedded = embedded.newly_embedded
        report.reused_vectors = embedded.reused
        report.dimension = embedded.dimension
    report.store_total_chunks = store.count()
    return report
