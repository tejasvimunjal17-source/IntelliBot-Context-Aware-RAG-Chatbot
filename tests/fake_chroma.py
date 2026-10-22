"""MOCK of the `chromadb` package - FOR TESTS ONLY.

This file imitates the small part of ChromaDB's Python API that IntelliBot
uses (PersistentClient, get_or_create_collection, upsert, query, get, delete,
count, delete_collection). It exists because the build environment could not
install the real ChromaDB.

IMPORTANT
* It is written from the author's knowledge of the Chroma API. It may differ
  from the real library in details, so passing tests that use it do NOT prove
  that IntelliBot works with real ChromaDB.
* Production code never imports this file.
* The tests marked OPTIONAL in test_vector_store.py run the same checks
  against the REAL chromadb package when it is installed.

It is deliberately strict in a few places where real Chroma is strict (metadata
value types, empty "$in" lists, vector sizes) so that mistakes are caught.
"""

import contextlib
import math
import sys
import types
from typing import Any, Dict, List, Optional

# Imitates data saved on disk: {path: {collection_name: FakeCollection}}.
# Because it outlives individual client objects, it lets tests check that data
# survives "restarting" (creating a new client for the same folder).
_DISK: Dict[str, Dict[str, "FakeCollection"]] = {}


def reset_disk() -> None:
    _DISK.clear()


def _matches(metadata: Optional[dict], where: Optional[dict]) -> bool:
    if not where:
        return True
    for key, condition in where.items():
        value = (metadata or {}).get(key)
        if isinstance(condition, dict):
            for operator, operand in condition.items():
                if operator == "$in":
                    if not isinstance(operand, list) or not operand:
                        raise ValueError("Expected where value for $in to be a non-empty list")
                    if value not in operand:
                        return False
                else:
                    raise ValueError(f"Unsupported operator in mock: {operator}")
        elif value != condition:
            return False
    return True


def _cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - (dot / norm if norm else 0.0)


class FakeCollection:
    def __init__(self, name: str, metadata: Optional[dict]):
        self.name = name
        self.metadata = dict(metadata or {})
        self._rows: Dict[str, dict] = {}
        self._dim: Optional[int] = None

    def count(self) -> int:
        return len(self._rows)

    def upsert(self, ids, embeddings=None, documents=None, metadatas=None):
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("Lengths of ids, embeddings, documents and metadatas differ")
        if len(set(ids)) != len(ids):
            raise ValueError("Expected IDs to be unique")
        for meta in metadatas:
            for key, value in meta.items():
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError(f"Expected metadata value for {key} to be str/int/float/bool")
        for vector in embeddings:
            if self._dim is None:
                self._dim = len(vector)
            elif len(vector) != self._dim:
                raise ValueError(
                    f"Embedding dimension {len(vector)} does not match collection dimensionality {self._dim}"
                )
        for i, doc_id in enumerate(ids):
            self._rows[doc_id] = {
                "embedding": [float(x) for x in embeddings[i]],
                "document": documents[i],
                "metadata": dict(metadatas[i]),
            }

    def get(self, ids=None, where=None, include=None, limit=None):
        include = include if include is not None else ["metadatas", "documents"]
        chosen = [
            doc_id
            for doc_id, row in self._rows.items()
            if (ids is None or doc_id in ids) and _matches(row["metadata"], where)
        ]
        if limit is not None:
            chosen = chosen[:limit]
        out: Dict[str, Any] = {"ids": chosen}
        out["metadatas"] = [self._rows[i]["metadata"] for i in chosen] if "metadatas" in include else None
        out["documents"] = [self._rows[i]["document"] for i in chosen] if "documents" in include else None
        return out

    def query(self, query_embeddings, n_results=10, where=None, include=None):
        include = include or ["metadatas", "documents", "distances"]
        ids_all, docs_all, metas_all, dists_all = [], [], [], []
        for query in query_embeddings:
            if self._dim is not None and len(query) != self._dim:
                raise ValueError(
                    f"Embedding dimension {len(query)} does not match collection dimensionality {self._dim}"
                )
            scored = [
                (_cosine_distance(query, row["embedding"]), doc_id)
                for doc_id, row in self._rows.items()
                if _matches(row["metadata"], where)
            ]
            scored.sort()
            scored = scored[:n_results]  # like real Chroma: fewer rows if fewer match
            ids_all.append([d for _, d in scored])
            dists_all.append([dist for dist, _ in scored])
            docs_all.append([self._rows[d]["document"] for _, d in scored])
            metas_all.append([self._rows[d]["metadata"] for _, d in scored])
        return {"ids": ids_all, "documents": docs_all, "metadatas": metas_all, "distances": dists_all}

    def delete(self, ids=None, where=None):
        if ids is None and where is None:
            raise ValueError("Either ids or where must be provided")
        doomed = [
            doc_id
            for doc_id, row in self._rows.items()
            if (ids is None or doc_id in ids) and _matches(row["metadata"], where)
        ]
        for doc_id in doomed:
            del self._rows[doc_id]


class FakePersistentClient:
    def __init__(self, path: str, settings: Any = None):
        self._path = str(path)
        _DISK.setdefault(self._path, {})

    def get_or_create_collection(self, name: str, metadata: Optional[dict] = None):
        collections = _DISK[self._path]
        if name not in collections:
            collections[name] = FakeCollection(name, metadata)
        return collections[name]

    def delete_collection(self, name: str):
        if name not in _DISK[self._path]:
            raise ValueError(f"Collection {name} does not exist.")
        del _DISK[self._path][name]


def build_fake_modules() -> Dict[str, types.ModuleType]:
    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = FakePersistentClient
    config = types.ModuleType("chromadb.config")
    config.Settings = lambda **kwargs: dict(kwargs)
    chromadb.config = config
    return {"chromadb": chromadb, "chromadb.config": config}


@contextlib.contextmanager
def installed():
    """Temporarily make `import chromadb` return the MOCK (and wipe its fake disk)."""
    reset_disk()
    saved = {name: sys.modules.get(name) for name in ("chromadb", "chromadb.config")}
    sys.modules.update(build_fake_modules())
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        reset_disk()
