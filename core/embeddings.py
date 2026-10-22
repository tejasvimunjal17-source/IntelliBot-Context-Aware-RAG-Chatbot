"""Embeddings for IntelliBot (Day 2).

An *embedding* turns a piece of text into a list of numbers (a vector).
Texts with similar meaning get similar vectors, which is how Day 3's
vector search will find the chunks that match a question.

Design:
* ``Embedder`` is a tiny interface: ``embed_documents`` (for chunks) and
  ``embed_query`` (for a user question). Day 3's retrieval code only depends on
  this interface, so the embedding model can be swapped without other changes.
* ``SentenceTransformerEmbedder`` is the real implementation. It runs a free
  model on your own machine - no API key. The first use downloads the model
  (about 90 MB for the default) and needs the ``sentence-transformers`` package.
* If the package or the model is not available, a clear
  ``EmbeddingUnavailableError`` is raised. We NEVER return fake vectors.
* ``embed_chunks`` embeds Chunk objects and reuses earlier results through an
  ``EmbeddingCache``, so unchanged text is not embedded twice.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from .chunking import Chunk
from .config import DEFAULT_EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

Vector = List[float]


class EmbeddingError(Exception):
    """Embedding failed. The message is safe to show to the user."""


class EmbeddingUnavailableError(EmbeddingError):
    """The embedding library or model is not available (not installed / not downloadable)."""


class Embedder(Protocol):
    """What the rest of IntelliBot needs from an embedding model."""

    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        """Embed many texts (document chunks). Returns one vector per text, same order."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a single user question."""
        ...


def sentence_transformers_installed() -> bool:
    """True if the sentence-transformers package can be imported (model not checked)."""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def _check_vectors(vectors: object, expected_count: int) -> List[Vector]:
    """Make sure a model returned the right number of finite, same-length vectors."""
    try:
        rows = [[float(x) for x in row] for row in vectors]  # type: ignore[union-attr]
    except Exception as exc:
        raise EmbeddingError("The embedding model returned unreadable results.") from exc
    if len(rows) != expected_count:
        raise EmbeddingError(
            f"The embedding model returned {len(rows)} results for {expected_count} texts."
        )
    if rows:
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise EmbeddingError("The embedding model returned vectors of different sizes.")
        if any(not math.isfinite(x) for row in rows for x in row):
            raise EmbeddingError("The embedding model returned invalid numbers.")
    return rows


class SentenceTransformerEmbedder:
    """Local embeddings with the SentenceTransformers library.

    The model is loaded the first time it is needed (not when the object is
    created), so importing/creating this class is cheap and never downloads.
    Vectors are normalised to length 1, so cosine similarity (Day 3) is simply
    a dot product.

    ``model_loader`` exists for tests: it lets a test supply a stand-in model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        model_loader: Optional[Callable[[str], object]] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self._model_loader = model_loader
        self._model: Optional[object] = None

    # -- loading ----------------------------------------------------------
    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        loader = self._model_loader
        if loader is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingUnavailableError(
                    "The 'sentence-transformers' package is not installed, so embeddings "
                    "cannot be created. Install it with: pip install sentence-transformers"
                ) from exc
            loader = SentenceTransformer
        try:
            self._model = loader(self.model_name)
        except Exception as exc:
            logger.warning("Embedding model load failed: %s", type(exc).__name__)
            raise EmbeddingUnavailableError(
                f"The embedding model '{self.model_name}' could not be loaded. "
                "The first run needs an internet connection to download it (about 90 MB "
                "for the default model). Check the model name and your connection."
            ) from exc
        return self._model

    # -- embedding --------------------------------------------------------
    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        texts = list(texts)
        if not texts:
            return []  # nothing to do; no need to load the model
        model = self._load_model()
        try:
            raw = model.encode(  # type: ignore[attr-defined]
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
        except Exception as exc:
            logger.warning("Embedding failed: %s", type(exc).__name__)
            raise EmbeddingError(
                "Creating embeddings failed. Please try again with fewer or smaller documents."
            ) from exc
        return _check_vectors(raw, len(texts))

    def embed_query(self, text: str) -> Vector:
        if not text or not text.strip():
            raise EmbeddingError("The question is empty.")
        return self.embed_documents([text])[0]


# --------------------------------------------------------------------------
# Cache + pipeline
# --------------------------------------------------------------------------


class EmbeddingCache:
    """In-memory store of text -> vector, separate for each model.

    Keys are hashes of (model name + text), so editing a document only
    re-embeds the chunks whose text actually changed. (Day 3 stores vectors
    permanently in ChromaDB; this cache avoids repeat work within a session.)
    """

    def __init__(self) -> None:
        self._vectors: Dict[str, Vector] = {}

    @staticmethod
    def key(model_name: str, text: str) -> str:
        return hashlib.sha256(f"{model_name}\x00{text}".encode("utf-8")).hexdigest()

    def get(self, model_name: str, text: str) -> Optional[Vector]:
        return self._vectors.get(self.key(model_name, text))

    def put(self, model_name: str, text: str, vector: Vector) -> None:
        self._vectors[self.key(model_name, text)] = vector

    def __len__(self) -> int:
        return len(self._vectors)


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: Vector


@dataclass
class EmbeddingResult:
    items: List[EmbeddedChunk] = field(default_factory=list)
    newly_embedded: int = 0  # unique texts sent to the model in this run
    reused: int = 0  # chunks served from the cache

    @property
    def dimension(self) -> int:
        return len(self.items[0].vector) if self.items else 0


def embed_chunks(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    cache: Optional[EmbeddingCache] = None,
) -> EmbeddingResult:
    """Embed chunks, reusing cached vectors. Order of ``chunks`` is preserved.

    Raises EmbeddingError (or EmbeddingUnavailableError) if the model cannot
    do the job. On failure the cache is left unchanged.
    """
    if not chunks:
        return EmbeddingResult()
    cache = cache if cache is not None else EmbeddingCache()
    model_name = embedder.model_name

    vectors: List[Optional[Vector]] = [cache.get(model_name, c.text) for c in chunks]
    reused = sum(1 for v in vectors if v is not None)

    # Unique texts that still need a vector (identical chunks are embedded once).
    missing: List[str] = []
    seen = set()
    for chunk, vector in zip(chunks, vectors):
        if vector is None and chunk.text not in seen:
            seen.add(chunk.text)
            missing.append(chunk.text)

    fresh: Dict[str, Vector] = {}
    if missing:
        produced = embedder.embed_documents(missing)
        if len(produced) != len(missing):
            raise EmbeddingError("The embedding model returned the wrong number of results.")
        fresh = dict(zip(missing, produced))
        for text, vector in fresh.items():
            cache.put(model_name, text, vector)

    items = [
        EmbeddedChunk(chunk=chunk, vector=vector if vector is not None else fresh[chunk.text])
        for chunk, vector in zip(chunks, vectors)
    ]
    return EmbeddingResult(items=items, newly_embedded=len(missing), reused=reused)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (1 = same direction, 0 = unrelated)."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
