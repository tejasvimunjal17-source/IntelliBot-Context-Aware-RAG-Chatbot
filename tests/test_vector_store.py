"""Tests for core/vector_store.py.

HOW TO READ THESE TESTS
  MOCK     - run against tests/fake_chroma.py, an imitation of the chromadb API.
             They test IntelliBot's own logic (duplicates, ranking maths, metadata,
             error handling, how it calls Chroma). They do NOT prove that real
             ChromaDB works with this code.
  OPTIONAL - the SAME shared checks run against the REAL `chromadb` package. They
             are skipped automatically when chromadb is not installed (as in the
             build environment). When chromadb IS installed (e.g. in Codespaces or
             on a computer), they run for real. A skipped run means the real
             integration is UNVERIFIED.
  REAL     - plain functions that need no chromadb at all.

Vectors used here are tiny hand-made 3-number vectors, so expected rankings can
be checked by hand. The embedding stand-in (FakeEmbedder) is a MOCK.
"""

import contextlib
import importlib.util
import math
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.chunking import Chunk, ChunkingConfig, chunk_document
from core.config import SAMPLE_DOCS_DIR
from core.document_loader import load_path
from core.embeddings import EmbeddedChunk, embed_chunks
from core.vector_store import (
    ChromaVectorStore,
    SearchResult,
    VectorStoreError,
    VectorStoreUnavailableError,
    collection_name_for,
)
from tests import fake_chroma
from tests.helpers import FakeEmbedder

HASH_A = "a" * 64
HASH_B = "b" * 64


def make_chunk(source, doc_hash, index, total, text, page=1):
    return Chunk(
        text=text, source=source, doc_hash=doc_hash, file_type="pdf",
        chunk_index=index, total_chunks=total, page_start=page, page_end=page,
        char_start=index * 100, char_end=index * 100 + len(text),
    )


def doc_items(source, doc_hash, specs):
    """specs: list of (text, vector) -> list of EmbeddedChunk."""
    total = len(specs)
    return [
        EmbeddedChunk(make_chunk(source, doc_hash, i, total, text), list(vec))
        for i, (text, vec) in enumerate(specs)
    ]


X, Y, Z = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
XY = [1.0, 1.0, 0.0]


def chroma_installed() -> bool:
    try:
        return importlib.util.find_spec("chromadb") is not None
    except (ImportError, ValueError):
        return False


# ======================================================================
# Shared checks (the "contract"). Run against MOCK chromadb and real chromadb.
# ======================================================================
class StoreContract:
    MODEL = "test/model-a"

    def backend_context(self):
        return contextlib.nullcontext()

    def setUp(self):
        ctx = self.backend_context()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.tmp = tempfile.mkdtemp(prefix="intellibot_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def new_store(self, model=None):
        return ChromaVectorStore(Path(self.tmp) / "db", model or self.MODEL)

    def doc_a(self):
        return doc_items("a.pdf", HASH_A, [("alpha text", X), ("beta text", Y), ("gamma text", Z)])

    # ---- empty store ------------------------------------------------
    def test_empty_store(self):
        store = self.new_store()
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.sources(), {})
        self.assertEqual(store.document_hashes(), set())
        self.assertEqual(store.search(X, top_k=3), [])

    def test_adding_nothing_is_fine(self):
        report = self.new_store().add_embedded_chunks([])
        self.assertEqual((report.chunks_written, report.documents_added), (0, 0))

    # ---- adding and metadata ---------------------------------------------
    def test_add_chunks_and_status(self):
        store = self.new_store()
        report = store.add_embedded_chunks(self.doc_a())
        self.assertEqual((report.documents_added, report.chunks_written), (1, 3))
        self.assertEqual(store.count(), 3)
        self.assertEqual(store.sources(), {"a.pdf": 3})
        self.assertEqual(store.document_hashes(), {HASH_A})

    def test_metadata_is_preserved_in_results(self):
        store = self.new_store()
        items = doc_items("policy.pdf", HASH_A, [("first", X), ("second text here", Y)])
        items[1].chunk = make_chunk("policy.pdf", HASH_A, 1, 2, "second text here", page=4)
        store.add_embedded_chunks(items)
        result = store.search(Y, top_k=1)[0]
        original = items[1].chunk
        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.text, original.text)
        self.assertEqual(result.chunk_id, original.chunk_id)
        self.assertEqual(result.source, "policy.pdf")
        self.assertEqual(result.doc_hash, HASH_A)
        self.assertEqual(result.file_type, "pdf")
        self.assertEqual((result.chunk_index, result.total_chunks), (1, 2))
        self.assertEqual((result.page_start, result.page_end), (4, 4))
        self.assertEqual((result.char_start, result.char_end), (original.char_start, original.char_end))

    # ---- persistence ------------------------------------------------------
    def test_data_survives_reopening_the_store(self):
        first = self.new_store()
        first.add_embedded_chunks(self.doc_a())
        del first
        reopened = self.new_store()  # like restarting the app
        self.assertEqual(reopened.count(), 3)
        self.assertEqual(reopened.search(Z, top_k=1)[0].text, "gamma text")
        self.assertEqual(reopened.sources(), {"a.pdf": 3})

    def test_reopened_store_does_not_duplicate_when_same_documents_are_added(self):
        self.new_store().add_embedded_chunks(self.doc_a())
        again = self.new_store()
        report = again.add_embedded_chunks(self.doc_a())
        self.assertEqual(report.documents_unchanged, 1)
        self.assertEqual(again.count(), 3)

    # ---- top-K and ranking ---------------------------------------------------
    def test_ranking_follows_cosine_similarity(self):
        store = self.new_store()
        store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [
            ("x axis", X), ("y axis", Y), ("between x and y", XY), ("z axis", Z),
        ]))
        results = store.search([1.0, 0.2, 0.0], top_k=4)
        self.assertEqual([r.text for r in results],
                         ["x axis", "between x and y", "y axis", "z axis"])
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([r.rank for r in results], [1, 2, 3, 4])

    def test_scores_are_cosine_similarities(self):
        store = self.new_store()
        store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [
            ("same", X), ("orthogonal", Y), ("opposite", [-1.0, 0.0, 0.0]), ("diagonal", XY),
        ]))
        by_text = {r.text: r.score for r in store.search(X, top_k=4)}
        self.assertAlmostEqual(by_text["same"], 1.0, places=3)
        self.assertAlmostEqual(by_text["orthogonal"], 0.0, places=3)
        self.assertAlmostEqual(by_text["opposite"], -1.0, places=3)
        self.assertAlmostEqual(by_text["diagonal"], 1 / math.sqrt(2), places=3)

    def test_vector_length_does_not_change_the_score(self):
        store = self.new_store()
        store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [("long", [10.0, 0.0, 0.0]), ("y", Y)]))
        self.assertAlmostEqual(store.search([0.1, 0.0, 0.0], top_k=1)[0].score, 1.0, places=3)

    def test_top_k_limits_the_number_of_results(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        self.assertEqual(len(store.search(X, top_k=1)), 1)
        self.assertEqual(len(store.search(X, top_k=2)), 2)

    def test_top_k_larger_than_store_returns_everything(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        self.assertEqual(len(store.search(X, top_k=50)), 3)

    def test_min_score_filters_weak_matches(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        results = store.search(X, top_k=3, min_score=0.5)
        self.assertEqual([r.text for r in results], ["alpha text"])
        self.assertEqual(store.search(X, top_k=3, min_score=1.01), [])

    # ---- limiting the search to chosen documents (session isolation) --------------
    def test_search_can_be_limited_to_chosen_documents(self):
        store = self.new_store()
        store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [("a one", X), ("a two", Y)]))
        store.add_embedded_chunks(doc_items("b.pdf", HASH_B, [("b one", X), ("b two", Z)]))
        only_a = store.search(X, top_k=10, doc_hashes=[HASH_A])
        self.assertEqual({r.source for r in only_a}, {"a.pdf"})
        only_b = store.search(X, top_k=10, doc_hashes=[HASH_B])
        self.assertEqual({r.source for r in only_b}, {"b.pdf"})
        both = store.search(X, top_k=10, doc_hashes=[HASH_A, HASH_B])
        self.assertEqual({r.source for r in both}, {"a.pdf", "b.pdf"})
        self.assertEqual(len(store.search(X, top_k=10)), 4)  # None = everything

    def test_empty_or_unknown_document_list_finds_nothing(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        self.assertEqual(store.search(X, doc_hashes=[]), [])
        self.assertEqual(store.search(X, doc_hashes=["f" * 64]), [])

    # ---- duplicates ---------------------------------------------------------
    def test_adding_the_same_document_twice_creates_no_duplicates(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        report = store.add_embedded_chunks(self.doc_a())
        self.assertEqual(store.count(), 3)
        self.assertEqual((report.documents_unchanged, report.documents_added, report.chunks_written), (1, 0, 0))

    def test_changed_chunks_of_the_same_file_replace_the_old_ones(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())  # 3 chunks
        new_version = doc_items("a.pdf", HASH_A, [("brand new one", X), ("brand new two", Y)])
        report = store.add_embedded_chunks(new_version)
        self.assertEqual(report.documents_replaced, 1)
        self.assertEqual(store.count(), 2)  # old 3 chunks are gone, nothing stale remains
        texts = {r.text for r in store.search(X, top_k=10)}
        self.assertEqual(texts, {"brand new one", "brand new two"})

    def test_edited_text_with_same_ids_is_detected(self):
        store = self.new_store()
        store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [("old wording", X)]))
        store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [("new wording", X)]))
        self.assertEqual([r.text for r in store.search(X, top_k=5)], ["new wording"])

    def test_is_document_current(self):
        store = self.new_store()
        chunks = [i.chunk for i in self.doc_a()]
        self.assertFalse(store.is_document_current(chunks))
        store.add_embedded_chunks(self.doc_a())
        self.assertTrue(store.is_document_current(chunks))
        self.assertFalse(store.is_document_current(chunks[:2]))
        self.assertTrue(store.is_document_current([]))

    def test_is_document_current_rejects_mixed_documents(self):
        store = self.new_store()
        mixed = [i.chunk for i in self.doc_a()] + [make_chunk("b.pdf", HASH_B, 0, 1, "x")]
        with self.assertRaises(VectorStoreError):
            store.is_document_current(mixed)

    def test_two_documents_in_one_call(self):
        store = self.new_store()
        report = store.add_embedded_chunks(
            doc_items("a.pdf", HASH_A, [("a1", X)]) + doc_items("b.pdf", HASH_B, [("b1", Y)])
        )
        self.assertEqual(report.documents_added, 2)
        self.assertEqual(store.sources(), {"a.pdf": 1, "b.pdf": 1})

    # ---- delete / clear -----------------------------------------------------------
    def test_delete_document(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        store.add_embedded_chunks(doc_items("b.pdf", HASH_B, [("b1", X)]))
        self.assertEqual(store.delete_document(HASH_A), 3)
        self.assertEqual(store.document_hashes(), {HASH_B})
        self.assertEqual(store.delete_document(HASH_A), 0)
        self.assertEqual(store.delete_document("f" * 64), 0)

    def test_clear_empties_the_store_and_it_stays_usable(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        store.clear()
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.search(X), [])
        store.add_embedded_chunks(self.doc_a())
        self.assertEqual(store.count(), 3)
        self.assertEqual(self.new_store().count(), 3)  # persisted again after clear

    def test_clearing_an_empty_store_is_fine(self):
        store = self.new_store()
        store.clear()
        self.assertEqual(store.count(), 0)
        store.add_embedded_chunks(self.doc_a())  # still usable afterwards
        self.assertEqual(store.count(), 3)

    # ---- different models stay apart ------------------------------------------------
    def test_embedding_models_use_separate_collections(self):
        a = self.new_store("test/model-a")
        b = self.new_store("test/model-b")
        a.add_embedded_chunks(self.doc_a())
        self.assertEqual(b.count(), 0)
        self.assertEqual(b.search(X), [])
        self.assertNotEqual(a.collection_name, b.collection_name)

    # ---- invalid input ------------------------------------------------------------------
    def test_invalid_vectors_are_rejected_when_adding(self):
        store = self.new_store()
        chunk = make_chunk("a.pdf", HASH_A, 0, 1, "text")
        for bad in ([], [float("nan"), 1.0], [float("inf"), 0.0], ["a", "b"], None, 5):
            with self.subTest(vector=bad):
                with self.assertRaises(VectorStoreError):
                    store.add_embedded_chunks([EmbeddedChunk(chunk, bad)])
        self.assertEqual(store.count(), 0)

    def test_only_embedded_chunks_are_accepted(self):
        with self.assertRaises(VectorStoreError):
            self.new_store().add_embedded_chunks(["not a chunk"])  # type: ignore[list-item]

    def test_chunk_with_empty_text_is_rejected(self):
        with self.assertRaises(VectorStoreError):
            self.new_store().add_embedded_chunks(
                [EmbeddedChunk(make_chunk("a.pdf", HASH_A, 0, 1, "   "), X)]
            )

    def test_vectors_of_different_sizes_are_rejected(self):
        store = self.new_store()
        with self.assertRaises(VectorStoreError):
            store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [("one", X), ("two", [1.0, 0.0])]))
        self.assertEqual(store.count(), 0)

    def test_invalid_top_k_is_rejected(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        for bad in (0, -1, 51, 1000, True, "3", 2.5, None):
            with self.subTest(top_k=bad):
                with self.assertRaises(VectorStoreError):
                    store.search(X, top_k=bad)  # type: ignore[arg-type]

    def test_invalid_query_vectors_are_rejected(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        for bad in ([], [float("nan"), 0, 0], ["x"], None):
            with self.subTest(query=bad):
                with self.assertRaises(VectorStoreError):
                    store.search(bad)  # type: ignore[arg-type]

    def test_query_with_wrong_vector_size_gives_friendly_error(self):
        store = self.new_store()
        store.add_embedded_chunks(self.doc_a())
        with self.assertRaises(VectorStoreError) as ctx:
            store.search([1.0, 0.0])  # 2 numbers instead of 3
        self.assertNotIn("Traceback", str(ctx.exception))

    def test_blank_model_name_is_rejected(self):
        for bad in ("", "   ", None):
            with self.subTest(model=bad):
                with self.assertRaises(VectorStoreError):
                    ChromaVectorStore(Path(self.tmp) / "db", bad)  # type: ignore[arg-type]

    # ---- fit with the Day 1 + Day 2 pipeline -------------------------------------------
    def test_real_pipeline_objects_fit_the_store(self):
        """Chunks from the real loader/chunker + vectors from a stand-in embedder (MOCK).
        Checks that the stored records match the chunk objects. It says nothing
        about the quality of real embeddings."""
        embedder = FakeEmbedder(model_name=self.MODEL)
        chunks = []
        for path in sorted(SAMPLE_DOCS_DIR.iterdir()):
            chunks += chunk_document(load_path(path).document, ChunkingConfig(600, 100))
        store = self.new_store()
        report = store.add_embedded_chunks(embed_chunks(chunks, embedder).items)
        self.assertEqual(report.chunks_written, len(chunks))
        self.assertEqual(store.count(), len(chunks))
        self.assertEqual(len(store.sources()), 5)
        hit = store.search(embedder.embed_query("password security multi-factor authentication phishing incident"), top_k=1)[0]
        stored_chunk = next(c for c in chunks if c.chunk_id == hit.chunk_id)
        self.assertEqual(hit.text, stored_chunk.text)
        self.assertEqual(hit.source, stored_chunk.source)
        self.assertEqual((hit.page_start, hit.page_end), (stored_chunk.page_start, stored_chunk.page_end))
        self.assertEqual(hit.source, "NovaTech_IT_Security_Policy.pdf")


class MockChromaContractTests(StoreContract, unittest.TestCase):
    """MOCK: the shared checks against tests/fake_chroma.py."""

    def backend_context(self):
        return fake_chroma.installed()


@unittest.skipUnless(chroma_installed(), "OPTIONAL: real chromadb is not installed here - real integration UNVERIFIED")
class RealChromaContractTests(StoreContract, unittest.TestCase):
    """OPTIONAL: the SAME shared checks against the REAL chromadb package."""


# ======================================================================
# MOCK-only checks: how we call Chroma, and failure handling
# ======================================================================
class MockChromaBehaviourTests(unittest.TestCase):
    def setUp(self):
        ctx = fake_chroma.installed()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.tmp = tempfile.mkdtemp(prefix="intellibot_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def store(self):
        return ChromaVectorStore(Path(self.tmp) / "db", "test/model-a")

    def test_client_and_collection_are_created_as_intended(self):
        with mock.patch.object(fake_chroma.FakePersistentClient, "__init__", return_value=None) as init, \
             mock.patch.object(fake_chroma.FakePersistentClient, "get_or_create_collection") as goc:
            goc.return_value.metadata = {"hnsw:space": "cosine"}
            ChromaVectorStore(Path(self.tmp) / "db", "test/model-a")
        self.assertEqual(init.call_args.kwargs["path"], str(Path(self.tmp) / "db"))
        self.assertEqual(init.call_args.kwargs["settings"], {"anonymized_telemetry": False})
        self.assertEqual(goc.call_args.kwargs["metadata"], {"hnsw:space": "cosine"})
        self.assertEqual(goc.call_args.kwargs["name"], collection_name_for("test/model-a"))

    def test_storage_folder_is_created(self):
        self.store()
        self.assertTrue((Path(self.tmp) / "db").is_dir())

    def test_missing_chromadb_gives_install_message(self):
        # Setting a module to None makes `import chromadb` fail, on any machine.
        with mock.patch.dict(sys.modules, {"chromadb": None, "chromadb.config": None}):
            with self.assertRaises(VectorStoreUnavailableError) as ctx:
                ChromaVectorStore(Path(self.tmp) / "db", "m")
        self.assertIn("pip install chromadb", str(ctx.exception))

    def test_startup_failure_is_reported_as_unavailable(self):
        with mock.patch.object(fake_chroma.FakePersistentClient, "__init__",
                               side_effect=RuntimeError("internal detail")):
            with self.assertRaises(VectorStoreUnavailableError) as ctx:
                self.store()
        self.assertNotIn("internal detail", str(ctx.exception))

    def test_old_sqlite_gets_a_helpful_hint(self):
        with mock.patch.object(fake_chroma.FakePersistentClient, "__init__",
                               side_effect=RuntimeError("Your system has an unsupported version of sqlite3")):
            with self.assertRaises(VectorStoreUnavailableError) as ctx:
                self.store()
        self.assertIn("SQLite", str(ctx.exception))

    def test_collection_with_other_distance_measure_is_refused(self):
        client = fake_chroma.FakePersistentClient(str(Path(self.tmp) / "db"))
        client.get_or_create_collection(collection_name_for("test/model-a"), metadata={"hnsw:space": "l2"})
        with self.assertRaises(VectorStoreError) as ctx:
            self.store()
        self.assertIn("Clear index", str(ctx.exception))

    def test_every_chroma_failure_becomes_a_friendly_error(self):
        store = self.store()
        store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [("t", X)]))
        for method in ("upsert", "query", "get", "delete", "count"):
            with self.subTest(method=method):
                store2 = self.store()
                setattr(store2._collection, method, mock.Mock(side_effect=RuntimeError("secret internals")))
                actions = {
                    "upsert": lambda s: s.add_embedded_chunks(doc_items("c.pdf", "c" * 64, [("n", Y)])),
                    "query": lambda s: s.search(X),
                    "get": lambda s: s.sources(),
                    "delete": lambda s: s.delete_document(HASH_A),
                    "count": lambda s: s.count(),
                }
                with self.assertRaises(VectorStoreError) as ctx:
                    actions[method](store2)
                self.assertNotIn("secret internals", str(ctx.exception))

    def test_unexpected_query_answer_is_reported(self):
        store = self.store()
        store.add_embedded_chunks(doc_items("a.pdf", HASH_A, [("t", X)]))
        store._collection.query = mock.Mock(return_value={"unexpected": True})
        with self.assertRaises(VectorStoreError):
            store.search(X)

    def test_missing_metadata_in_old_records_does_not_crash(self):
        store = self.store()
        store._collection.upsert(ids=["old-1"], embeddings=[X], documents=["legacy text"], metadatas=[{"source": "old.txt"}])
        result = store.search(X, top_k=1)[0]
        self.assertEqual((result.source, result.chunk_index, result.page_start), ("old.txt", 0, 1))

    def test_large_batches_are_written_in_several_calls(self):
        store = self.store()
        items = doc_items("big.pdf", HASH_A, [(f"chunk {i}", [1.0, float(i % 7), 0.0]) for i in range(600)])
        spy = mock.Mock(wraps=store._collection.upsert)
        store._collection.upsert = spy
        store.add_embedded_chunks(items)
        self.assertGreaterEqual(spy.call_count, 3)
        self.assertEqual(store.count(), 600)


# ======================================================================
# REAL: need no chromadb
# ======================================================================
class CollectionNameTests(unittest.TestCase):
    def test_names_follow_chroma_rules(self):
        for model in ("sentence-transformers/all-MiniLM-L6-v2", "x", "a" * 200, "ü/日本語", "  spaced name  "):
            with self.subTest(model=model):
                name = collection_name_for(model)
                self.assertTrue(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]", name), name)
                self.assertLessEqual(len(name), 63)

    def test_deterministic_and_unique_per_model(self):
        self.assertEqual(collection_name_for("m1"), collection_name_for("m1"))
        self.assertNotEqual(collection_name_for("m1"), collection_name_for("m2"))
        self.assertNotEqual(collection_name_for("a" * 100 + "1"), collection_name_for("a" * 100 + "2"))

    def test_missing_chromadb_is_reported_without_any_substitute(self):
        """REAL when chromadb is truly absent (forced absent otherwise)."""
        with mock.patch.dict(sys.modules, {"chromadb": None, "chromadb.config": None}):
            with self.assertRaises(VectorStoreUnavailableError):
                ChromaVectorStore(Path(tempfile.gettempdir()) / "never_created_db", "m")


if __name__ == "__main__":
    unittest.main()
