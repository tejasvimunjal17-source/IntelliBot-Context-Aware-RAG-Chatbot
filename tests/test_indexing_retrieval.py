"""Tests for core/indexing.py and core/retrieval.py.

All of these run against the MOCK chromadb (tests/fake_chroma.py) with a
stand-in embedder (MOCK). They check IntelliBot's own logic: skipping unchanged
documents, replacing changed ones, retrieval statuses, thresholds, session
isolation and error propagation. They do NOT prove that real ChromaDB or a real
embedding model works. See test_vector_store.py for the OPTIONAL real-ChromaDB
checks of the store itself.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.chunking import ChunkingConfig, chunk_document, chunk_documents
from core.config import SAMPLE_DOCS_DIR
from core.document_loader import load_path
from core.embeddings import EmbeddingCache, EmbeddingError, EmbeddingUnavailableError
from core.indexing import index_chunks
from core.retrieval import (
    MAX_QUERY_CHARS,
    STATUS_EMPTY_QUERY,
    STATUS_NO_DOCUMENTS,
    STATUS_NO_RELEVANT,
    STATUS_OK,
    STATUS_QUERY_TOO_LONG,
    Retriever,
)
from core.vector_store import ChromaVectorStore, VectorStoreError
from tests import fake_chroma
from tests.helpers import FakeEmbedder, make_document
from tests.test_vector_store import HASH_A, HASH_B, X, Y, Z, make_chunk

MODEL = "test/model-a"


class TableEmbedder:
    """MOCK embedder with hand-chosen vectors: text -> vector (default: Z)."""

    def __init__(self, table=None, model_name=MODEL):
        self.model_name = model_name
        self.table = table or {}
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [list(self.table.get(t, Z)) for t in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return list(self.table.get(text, Z))


class Base(unittest.TestCase):
    def setUp(self):
        ctx = fake_chroma.installed()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.tmp = tempfile.mkdtemp(prefix="intellibot_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def new_store(self, model=MODEL):
        return ChromaVectorStore(Path(self.tmp) / "db", model)

    def sample_chunks(self, size=600, overlap=100):
        docs = [load_path(p).document for p in sorted(SAMPLE_DOCS_DIR.iterdir())]
        return chunk_documents(docs, ChunkingConfig(size, overlap))


# ----------------------------------------------------------------- indexing
class IndexingTests(Base):
    def test_sample_documents_are_embedded_and_stored(self):
        chunks, store, emb = self.sample_chunks(), self.new_store(), FakeEmbedder(MODEL)
        report = index_chunks(chunks, store, emb)
        self.assertEqual(report.documents_total, 5)
        self.assertEqual(report.documents_indexed, 5)
        self.assertEqual(report.documents_unchanged, 0)
        self.assertEqual(report.chunks_total, len(chunks))
        self.assertEqual(report.chunks_written, len(chunks))
        self.assertEqual(report.store_total_chunks, len(chunks))
        self.assertEqual(report.dimension, emb.dim)
        self.assertEqual(store.count(), len(chunks))

    def test_second_run_skips_everything_and_does_not_call_the_model(self):
        chunks, store, emb = self.sample_chunks(), self.new_store(), FakeEmbedder(MODEL)
        index_chunks(chunks, store, emb)
        calls = len(emb.calls)
        report = index_chunks(chunks, store, emb)
        self.assertEqual(len(emb.calls), calls)
        self.assertEqual((report.documents_unchanged, report.documents_indexed), (5, 0))
        self.assertEqual((report.chunks_written, report.newly_embedded), (0, 0))
        self.assertEqual(store.count(), len(chunks))

    def test_reopened_store_also_skips_unchanged_documents(self):
        chunks, emb = self.sample_chunks(), FakeEmbedder(MODEL)
        index_chunks(chunks, self.new_store(), emb)
        calls = len(emb.calls)
        reopened = self.new_store()  # like restarting the app
        report = index_chunks(chunks, reopened, emb)
        self.assertEqual(len(emb.calls), calls)
        self.assertEqual(report.documents_unchanged, 5)
        self.assertEqual(reopened.count(), len(chunks))

    def test_only_new_documents_are_embedded(self):
        chunks, store = self.sample_chunks(), self.new_store()
        first, rest = chunks[:3], chunks[3:]
        emb = FakeEmbedder(MODEL)
        index_chunks([c for c in chunks if c.source == chunks[0].source], store, emb)
        emb.calls.clear()
        report = index_chunks(chunks, store, emb)
        sent = [t for batch in emb.calls for t in batch]
        self.assertTrue(sent)
        self.assertTrue(all(c.text in sent for c in chunks if c.source != chunks[0].source))
        self.assertFalse(any(c.text in sent for c in chunks if c.source == chunks[0].source))
        self.assertEqual(report.documents_unchanged, 1)
        self.assertEqual(report.documents_indexed, 4)
        self.assertEqual(store.count(), len(chunks))

    def test_changing_chunk_settings_replaces_old_chunks(self):
        store, emb = self.new_store(), FakeEmbedder(MODEL)
        index_chunks(self.sample_chunks(600, 100), store, emb)
        new_chunks = self.sample_chunks(300, 50)
        report = index_chunks(new_chunks, store, emb)
        self.assertEqual(report.documents_indexed, 5)
        self.assertEqual(store.count(), len(new_chunks))  # no leftovers from the 600-size run
        hits = store.search(emb.embed_query("password"), top_k=50)
        self.assertTrue(all(len(h.text) <= 300 for h in hits))

    def test_empty_chunk_list(self):
        store, emb = self.new_store(), FakeEmbedder(MODEL)
        report = index_chunks([], store, emb)
        self.assertEqual((report.documents_total, report.chunks_written), (0, 0))
        self.assertEqual(emb.calls, [])

    def test_model_mismatch_is_refused_before_any_work(self):
        store, emb = self.new_store("test/model-a"), FakeEmbedder("other-model")
        with self.assertRaises(VectorStoreError):
            index_chunks(self.sample_chunks(), store, emb)
        self.assertEqual(emb.calls, [])
        self.assertEqual(store.count(), 0)

    def test_embedding_failure_stores_nothing(self):
        class Broken(FakeEmbedder):
            def embed_documents(self, texts):
                raise EmbeddingUnavailableError("model missing")

        store = self.new_store()
        with self.assertRaises(EmbeddingError):
            index_chunks(self.sample_chunks(), store, Broken(MODEL))
        self.assertEqual(store.count(), 0)

    def test_embedding_cache_is_used_for_identical_text(self):
        a = make_document("a.txt", ["Same text in two files."], content_hash="aa" * 32)
        b = make_document("b.txt", ["Same text in two files."], content_hash="bb" * 32)
        emb, cache = FakeEmbedder(MODEL), EmbeddingCache()
        report = index_chunks(chunk_documents([a, b]), self.new_store(), emb, cache)
        self.assertEqual(report.newly_embedded, 1)
        self.assertEqual(report.chunks_written, 2)


# ----------------------------------------------------------------- retrieval
class RetrievalTests(Base):
    def setUp(self):
        super().setUp()
        self.store = self.new_store()
        self.chunk_x = make_chunk("a.pdf", HASH_A, 0, 2, "chunk about leave", page=1)
        self.chunk_y = make_chunk("a.pdf", HASH_A, 1, 2, "chunk about passwords", page=2)
        self.chunk_b = make_chunk("b.pdf", HASH_B, 0, 1, "chunk about expenses", page=1)
        self.emb = TableEmbedder({
            "chunk about leave": X, "chunk about passwords": Y, "chunk about expenses": X,
            "how much leave": X, "password rules": Y, "pasta recipes": Z,
        })
        index_chunks([self.chunk_x, self.chunk_y, self.chunk_b], self.store, self.emb)
        self.retriever = Retriever(self.store, self.emb, top_k=4, min_score=0.2)

    def test_relevant_chunks_come_back_best_first_with_metadata(self):
        response = self.retriever.retrieve("password rules")
        self.assertEqual(response.status, STATUS_OK)
        self.assertTrue(response.found_relevant)
        top = response.results[0]
        self.assertEqual((top.text, top.source, top.page_start, top.chunk_index, top.total_chunks),
                         ("chunk about passwords", "a.pdf", 2, 1, 2))
        self.assertAlmostEqual(top.score, 1.0, places=3)
        self.assertEqual(response.best_score, top.score)
        self.assertEqual([r.rank for r in response.results], list(range(1, len(response.results) + 1)))

    def test_irrelevant_results_are_filtered_by_threshold(self):
        response = self.retriever.retrieve("how much leave")
        self.assertEqual({r.text for r in response.results}, {"chunk about leave", "chunk about expenses"})
        self.assertNotIn("chunk about passwords", [r.text for r in response.results])

    def test_nothing_similar_enough_is_reported_clearly(self):
        response = self.retriever.retrieve("pasta recipes")  # points at Z, nothing stored there
        self.assertEqual(response.status, STATUS_NO_RELEVANT)
        self.assertEqual(response.results, [])
        self.assertFalse(response.found_relevant)
        self.assertIsNotNone(response.best_score)  # best score is still reported
        self.assertLess(response.best_score, 0.2)
        self.assertTrue(response.message)

    def test_top_k_is_configurable(self):
        self.assertEqual(len(self.retriever.retrieve("how much leave", top_k=1).results), 1)
        self.assertEqual(Retriever(self.store, self.emb, top_k=1).retrieve("how much leave").top_k, 1)
        self.assertEqual(len(Retriever(self.store, self.emb, top_k=1).retrieve("how much leave").results), 1)

    def test_min_score_is_configurable(self):
        everything = self.retriever.retrieve("how much leave", min_score=-1.0, top_k=10)
        self.assertEqual(len(everything.results), 3)
        self.assertEqual([r.rank for r in everything.results], [1, 2, 3])
        strict = self.retriever.retrieve("how much leave", min_score=0.99, top_k=10)
        self.assertEqual(len(strict.results), 2)

    def test_ranks_are_renumbered_after_filtering(self):
        response = self.retriever.retrieve("how much leave", top_k=10)
        self.assertEqual([r.rank for r in response.results], [1, 2])

    def test_empty_query_never_touches_the_model_or_store(self):
        for q in ("", "   ", "\n\t", None):
            with self.subTest(q=q):
                response = self.retriever.retrieve(q)  # type: ignore[arg-type]
                self.assertEqual(response.status, STATUS_EMPTY_QUERY)
                self.assertEqual(response.results, [])
        self.assertEqual(self.emb.query_calls, [])

    def test_too_long_query(self):
        response = self.retriever.retrieve("x" * (MAX_QUERY_CHARS + 1))
        self.assertEqual(response.status, STATUS_QUERY_TOO_LONG)
        self.assertEqual(self.emb.query_calls, [])

    def test_query_is_stripped(self):
        self.retriever.retrieve("   password rules  \n")
        self.assertEqual(self.emb.query_calls, ["password rules"])

    def test_empty_store_reports_no_documents(self):
        empty = Retriever(self.new_store("test/empty-model"), TableEmbedder(model_name="test/empty-model"))
        response = empty.retrieve("anything")
        self.assertEqual(response.status, STATUS_NO_DOCUMENTS)
        self.assertEqual(response.results, [])

    def test_search_is_limited_to_the_sessions_documents(self):
        only_b = self.retriever.retrieve("how much leave", doc_hashes=[HASH_B], top_k=10)
        self.assertEqual({r.source for r in only_b.results}, {"b.pdf"})
        only_a = self.retriever.retrieve("how much leave", doc_hashes=[HASH_A], top_k=10)
        self.assertEqual({r.source for r in only_a.results}, {"a.pdf"})

    def test_documents_not_in_the_index_give_no_documents(self):
        for hashes in ([], ["f" * 64]):
            with self.subTest(hashes=hashes):
                response = self.retriever.retrieve("how much leave", doc_hashes=hashes)
                self.assertEqual(response.status, STATUS_NO_DOCUMENTS)
                self.assertEqual(response.results, [])

    def test_elapsed_time_is_recorded(self):
        self.assertGreaterEqual(self.retriever.retrieve("password rules").elapsed_seconds, 0.0)

    def test_embedding_errors_propagate(self):
        with mock.patch.object(self.emb, "embed_query", side_effect=EmbeddingUnavailableError("no model")):
            with self.assertRaises(EmbeddingError):
                self.retriever.retrieve("password rules")

    def test_store_errors_propagate(self):
        self.store._collection.query = mock.Mock(side_effect=RuntimeError("boom"))
        with self.assertRaises(VectorStoreError):
            self.retriever.retrieve("password rules")

    def test_retriever_refuses_mismatched_model(self):
        with self.assertRaises(VectorStoreError):
            Retriever(self.store, TableEmbedder(model_name="different"))


# ----------------------------------------------------------------- whole pipeline
class EndToEndMockTests(Base):
    """Real loader + real chunker + stand-in embedder (MOCK) + MOCK chromadb."""

    def test_documents_to_retrieval_and_back_after_restart(self):
        emb = FakeEmbedder(MODEL)
        chunks = self.sample_chunks()
        index_chunks(chunks, self.new_store(), emb)

        # "restart": new store object, new retriever, nothing re-embedded
        calls = len(emb.calls)
        store = self.new_store()
        report = index_chunks(chunks, store, emb)
        self.assertEqual(len(emb.calls), calls)
        self.assertEqual(report.documents_unchanged, 5)

        hashes = sorted({c.doc_hash for c in chunks})
        response = Retriever(store, emb, top_k=3, min_score=0.0).retrieve(
            "password multi-factor authentication phishing incident", doc_hashes=hashes
        )
        self.assertEqual(response.status, STATUS_OK)
        self.assertEqual(response.results[0].source, "NovaTech_IT_Security_Policy.pdf")
        self.assertLessEqual(len(response.results), 3)

    def test_a_session_cannot_retrieve_documents_it_did_not_load(self):
        emb = FakeEmbedder(MODEL)
        chunks = self.sample_chunks()
        store = self.new_store()
        index_chunks(chunks, store, emb)  # all 5 documents are in the shared index
        leave_hash = next(c.doc_hash for c in chunks if "Leave" in c.source)
        response = Retriever(store, emb, top_k=10, min_score=-1.0).retrieve(
            "password security phishing", doc_hashes=[leave_hash]
        )
        self.assertEqual({r.source for r in response.results}, {"NovaTech_Leave_and_Attendance_Policy.pdf"})


if __name__ == "__main__":
    unittest.main()
