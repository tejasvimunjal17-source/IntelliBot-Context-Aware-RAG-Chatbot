"""Shared helpers for tests.

FakeEmbedder is a MOCK. It turns words into numbers with a simple hashing
trick (a "bag of words"). It only matches shared WORDS, not meaning, so it is
NOT a real embedding model. Tests that use it check IntelliBot's plumbing
(order, caching, metadata, errors), never embedding quality.
"""

import hashlib
import math
import re
from typing import List, Sequence

from core.document_loader import LoadedDocument, PageText


class FakeEmbedder:
    def __init__(self, model_name: str = "fake-bag-of-words", dim: int = 64):
        self.model_name = model_name
        self.dim = dim
        self.calls: List[List[str]] = []  # every batch of texts it was asked to embed

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


def make_document(name: str = "doc.txt", pages=None, file_type: str = "txt",
                  content_hash: str = "ab" * 32) -> LoadedDocument:
    """Build a LoadedDocument by hand (no file needed)."""
    pages = pages if pages is not None else ["Hello world."]
    return LoadedDocument(
        name=name,
        file_type=file_type,
        pages=[PageText(i + 1, t) for i, t in enumerate(pages)],
        size_bytes=sum(len(t) for t in pages),
        content_hash=content_hash,
        parser="text",
    )
