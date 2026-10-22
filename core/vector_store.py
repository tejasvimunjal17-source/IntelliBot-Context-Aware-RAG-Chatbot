"""Persistent vector store for IntelliBot (Day 3), built on ChromaDB.

This module talks to the real ``chromadb`` package (``PersistentClient``,
``get_or_create_collection``, ``upsert``, ``query``, ``get``, ``delete``).
If ChromaDB is not installed, ``ChromaVectorStore`` raises
``VectorStoreUnavailableError`` with install instructions. There is no
substitute store and no fake data.

Key behaviours
* Persistent: chunks, vectors and metadata are saved in a folder on disk and
  reused the next time the app starts.
* One collection per embedding model, so vectors from different models (which
  have different sizes and meanings) are never mixed.
* Cosine similarity. Chroma returns a *distance*; we report
  ``score = 1 - distance`` (1.0 = identical direction, higher is better).
* No duplicates: each document is identified by its file hash. Indexing the same
  document again does nothing; if the same file is re-chunked differently
  (other chunk size, edited settings) its old chunks are replaced, never
  piled up next to the new ones.
* Isolation: ``search(..., doc_hashes=[...])`` only looks inside the documents
  the caller names. The app passes the documents of the current browser
  session, so one visitor cannot retrieve another visitor's files.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .chunking import Chunk
from .config import DEFAULT_TOP_K, MAX_TOP_K
from .embeddings import EmbeddedChunk

logger = logging.getLogger(__name__)

_WRITE_BATCH_SIZE = 256


class VectorStoreError(Exception):
    """A vector-store problem. The message is safe to show to the user."""


class VectorStoreUnavailableError(VectorStoreError):
    """ChromaDB is not installed or could not be started."""


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk with its similarity score and metadata."""

    rank: int  # 1 = best match
    chunk_id: str
    text: str
    score: float  # cosine similarity, higher = more similar
    source: str  # document file name
    doc_hash: str
    file_type: str
    chunk_index: int
    total_chunks: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int


@dataclass
class AddReport:
    documents_added: int = 0  # new to the store
    documents_replaced: int = 0  # already stored, but the chunks changed
    documents_unchanged: int = 0  # already stored, identical -> nothing written
    chunks_written: int = 0


def collection_name_for(embedding_model: str) -> str:
    """A valid, unique Chroma collection name for an embedding model."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", embedding_model).strip("_")[:30] or "model"
    digest = hashlib.sha1(embedding_model.encode("utf-8")).hexdigest()[:8]
    return f"intellibot_{slug}_{digest}"


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _check_vector(vector: Any, what: str) -> List[float]:
    try:
        values = [float(x) for x in vector]
    except Exception:
        raise VectorStoreError(f"The {what} is not a list of numbers.")
    if not values:
        raise VectorStoreError(f"The {what} is empty.")
    if any(not math.isfinite(x) for x in values):
        raise VectorStoreError(f"The {what} contains invalid numbers.")
    return values


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ChromaVectorStore:
    """Persistent ChromaDB collection for one embedding model."""

    def __init__(self, persist_dir: Path, embedding_model: str) -> None:
        if not embedding_model or not str(embedding_model).strip():
            raise VectorStoreError("An embedding model name is required.")
        self.embedding_model = str(embedding_model).strip()
        self.collection_name = collection_name_for(self.embedding_model)
        self.persist_dir = Path(persist_dir)
        self._lock = threading.RLock()

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise VectorStoreUnavailableError(
                "ChromaDB is not installed, so documents cannot be indexed or searched. "
                "Install it with: pip install chromadb"
            ) from exc

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.warning("ChromaDB could not start: %s", type(exc).__name__)
            hint = ""
            if "sqlite" in str(exc).lower():
                hint = (
                    " Your system's SQLite is too old for ChromaDB (see the README, "
                    "'Troubleshooting')."
                )
            raise VectorStoreUnavailableError(
                "ChromaDB could not open its storage folder." + hint
            ) from exc

        # Safety check: a collection created earlier must also use cosine distance.
        existing_space = (getattr(self._collection, "metadata", None) or {}).get("hnsw:space")
        if existing_space and existing_space != "cosine":
            raise VectorStoreError(
                "The saved index uses a different similarity measure. "
                "Use 'Clear index' and index your documents again."
            )

    # ------------------------------------------------------------------ helpers
    def _call(self, action: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a ChromaDB call; turn any failure into a friendly VectorStoreError."""
        try:
            return func(*args, **kwargs)
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.warning("ChromaDB call failed (%s): %s", action, type(exc).__name__)
            raise VectorStoreError(
                f"The search index could not {action}. If this keeps happening, "
                "use 'Clear index' and index your documents again."
            ) from exc

    def _doc_fingerprint(self, doc_hash: str) -> Dict[str, str]:
        """Stored {chunk_id: text_hash} for one document ({} if not stored)."""
        found = self._call(
            "read stored documents",
            self._collection.get,
            where={"doc_hash": doc_hash},
            include=["metadatas"],
        )
        ids = found.get("ids") or []
        metas = found.get("metadatas") or []
        return {
            chunk_id: str((meta or {}).get("text_hash", ""))
            for chunk_id, meta in zip(ids, metas)
        }

    @staticmethod
    def _new_fingerprint(chunks: Sequence[Chunk]) -> Dict[str, str]:
        fingerprint = {c.chunk_id: text_fingerprint(c.text) for c in chunks}
        if len(fingerprint) != len(chunks):
            raise VectorStoreError("Two chunks of one document have the same ID.")
        return fingerprint

    @staticmethod
    def _group_by_document(chunks: Sequence[Chunk]) -> Dict[str, List[Chunk]]:
        groups: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            groups.setdefault(chunk.doc_hash, []).append(chunk)
        return groups

    # ------------------------------------------------------------------ status
    def count(self) -> int:
        """Total number of chunks stored."""
        with self._lock:
            return int(self._call("count stored chunks", self._collection.count))

    def sources(self) -> Dict[str, int]:
        """Stored chunk count per source file name."""
        with self._lock:
            found = self._call(
                "list stored documents", self._collection.get, include=["metadatas"]
            )
        counts: Dict[str, int] = {}
        for meta in found.get("metadatas") or []:
            name = str((meta or {}).get("source", "unknown"))
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def document_hashes(self) -> Set[str]:
        """File hashes of all stored documents."""
        with self._lock:
            found = self._call(
                "list stored documents", self._collection.get, include=["metadatas"]
            )
        return {
            str((meta or {})["doc_hash"])
            for meta in (found.get("metadatas") or [])
            if meta and "doc_hash" in meta
        }

    def is_document_current(self, chunks: Sequence[Chunk]) -> bool:
        """True if these chunks (all from ONE document) are already stored unchanged."""
        if not chunks:
            return True
        groups = self._group_by_document(chunks)
        if len(groups) != 1:
            raise VectorStoreError("is_document_current expects chunks of a single document.")
        doc_hash = next(iter(groups))
        with self._lock:
            return self._doc_fingerprint(doc_hash) == self._new_fingerprint(chunks)

    # ------------------------------------------------------------------ writing
    def add_embedded_chunks(self, items: Sequence[EmbeddedChunk]) -> AddReport:
        """Store embedded chunks. Safe to call repeatedly with the same documents."""
        report = AddReport()
        if not items:
            return report

        prepared: List[Tuple[Chunk, List[float]]] = []
        for item in items:
            if not isinstance(item, EmbeddedChunk) or not isinstance(item.chunk, Chunk):
                raise VectorStoreError("Only embedded chunks can be stored.")
            if not item.chunk.text or not item.chunk.text.strip():
                raise VectorStoreError("A chunk with empty text cannot be stored.")
            prepared.append((item.chunk, _check_vector(item.vector, "embedding vector")))
        dimensions = {len(vector) for _, vector in prepared}
        if len(dimensions) != 1:
            raise VectorStoreError("The embedding vectors have different sizes.")

        groups: Dict[str, List[Tuple[Chunk, List[float]]]] = {}
        for chunk, vector in prepared:
            groups.setdefault(chunk.doc_hash, []).append((chunk, vector))

        with self._lock:
            for doc_hash, pairs in groups.items():
                chunks = [chunk for chunk, _ in pairs]
                new_fp = self._new_fingerprint(chunks)
                stored_fp = self._doc_fingerprint(doc_hash)
                if stored_fp == new_fp:
                    report.documents_unchanged += 1
                    continue
                if stored_fp:  # same file, different chunks: replace, never mix old and new
                    self._call(
                        "replace an old version of a document",
                        self._collection.delete,
                        ids=list(stored_fp),
                    )
                    report.documents_replaced += 1
                else:
                    report.documents_added += 1

                for start in range(0, len(pairs), _WRITE_BATCH_SIZE):
                    batch = pairs[start : start + _WRITE_BATCH_SIZE]
                    metadatas = []
                    for chunk, _ in batch:
                        meta = dict(chunk.metadata())
                        meta["text_hash"] = text_fingerprint(chunk.text)
                        meta["embedding_model"] = self.embedding_model
                        metadatas.append(meta)
                    self._call(
                        "store chunks",
                        self._collection.upsert,
                        ids=[chunk.chunk_id for chunk, _ in batch],
                        embeddings=[vector for _, vector in batch],
                        documents=[chunk.text for chunk, _ in batch],
                        metadatas=metadatas,
                    )
                    report.chunks_written += len(batch)
        return report

    def delete_document(self, doc_hash: str) -> int:
        """Remove one document's chunks. Returns how many were removed."""
        with self._lock:
            ids = list(self._doc_fingerprint(doc_hash))
            if ids:
                self._call("delete a document", self._collection.delete, ids=ids)
            return len(ids)

    def clear(self) -> None:
        """Delete everything in this model's collection (it is recreated empty)."""
        with self._lock:
            self._call(
                "clear the index", self._client.delete_collection, name=self.collection_name
            )
            self._collection = self._call(
                "recreate the index",
                self._client.get_or_create_collection,
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    # ------------------------------------------------------------------ searching
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = DEFAULT_TOP_K,
        min_score: Optional[float] = None,
        doc_hashes: Optional[Sequence[str]] = None,
    ) -> List[SearchResult]:
        """Return the ``top_k`` most similar chunks, best first.

        * ``min_score``  - drop results with a lower cosine similarity.
        * ``doc_hashes`` - only search inside these documents. ``None`` searches
          everything; an empty list searches nothing.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not (1 <= top_k <= MAX_TOP_K):
            raise VectorStoreError(f"top_k must be a whole number from 1 to {MAX_TOP_K}.")
        vector = _check_vector(query_vector, "query vector")
        if doc_hashes is not None:
            doc_hashes = list(dict.fromkeys(str(h) for h in doc_hashes))
            if not doc_hashes:
                return []

        with self._lock:
            total = self.count()
            if total == 0:
                return []
            kwargs: Dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": min(top_k, total),
                "include": ["documents", "metadatas", "distances"],
            }
            if doc_hashes is not None:
                kwargs["where"] = (
                    {"doc_hash": doc_hashes[0]}
                    if len(doc_hashes) == 1
                    else {"doc_hash": {"$in": doc_hashes}}
                )
            raw = self._call("search", self._collection.query, **kwargs)

        try:
            ids = raw["ids"][0]
            documents = raw["documents"][0]
            metadatas = raw["metadatas"][0]
            distances = raw["distances"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise VectorStoreError("The search index returned an unexpected answer.") from exc

        hits = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            hits.append((score, str(chunk_id), text, meta or {}))
        hits.sort(key=lambda h: (-h[0], h[1]))  # best first; ties by ID for stable output

        results: List[SearchResult] = []
        for score, chunk_id, text, meta in hits:
            if min_score is not None and score < min_score:
                continue
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    chunk_id=chunk_id,
                    text=text or "",
                    score=score,
                    source=str(meta.get("source", "unknown")),
                    doc_hash=str(meta.get("doc_hash", "")),
                    file_type=str(meta.get("file_type", "")),
                    chunk_index=_as_int(meta.get("chunk_index")),
                    total_chunks=_as_int(meta.get("total_chunks")),
                    page_start=_as_int(meta.get("page_start"), 1),
                    page_end=_as_int(meta.get("page_end"), 1),
                    char_start=_as_int(meta.get("char_start")),
                    char_end=_as_int(meta.get("char_end")),
                )
            )
        return results
