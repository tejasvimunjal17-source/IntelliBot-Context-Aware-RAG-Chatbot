"""Tests for core/embeddings.py.

Test groups:
  * MOCK     - a stand-in model (FakeModel / FakeEmbedder) checks IntelliBot's own
               logic: batching, caching, ordering, error handling. These do NOT
               prove that the real SentenceTransformers model works or that
               embeddings are of good quality.
  * REAL     - run for real, no mocks (e.g. the "library is not installed" case
               is only REAL when the library truly is missing).
  * OPTIONAL - need the real sentence-transformers package AND the downloaded
               model. Skipped automatically when they are unavailable.
"""

import sys
import unittest
from unittest import mock

from core.chunking import ChunkingConfig, chunk_document, chunk_documents
from core.config import SAMPLE_DOCS_DIR
from core.document_loader import load_path
from core.embeddings import (
    EmbeddingCache,
    EmbeddingError,
    EmbeddingUnavailableError,
    SentenceTransformerEmbedder,
    cosine_similarity,
    embed_chunks,
    sentence_transformers_installed,
)
from tests.helpers import FakeEmbedder, make_document


# ----------------------------------------------------------------- stand-in model
class FakeModel:
    """Stand-in for a SentenceTransformer model (MOCK)."""

    def __init__(self, vectors=None, fail=False):
        self.encode_calls = []
        self._vectors = vectors
        self._fail = fail

    def encode(self, texts, **kwargs):
        self.encode_calls.append((list(texts), kwargs))
        if self._fail:
            raise RuntimeError("model crashed")
        if self._vectors is not None:
            return self._vectors
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class FakeNumpyLike:
    """Mimics a numpy array: has .tolist() (what real models return)."""

    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


def embedder_with(model, **kwargs):
    return SentenceTransformerEmbedder("fake/model", model_loader=lambda name: model, **kwargs)


# ----------------------------------------------------------------- MOCK: SentenceTransformerEmbedder
class SentenceTransformerEmbedderMockTests(unittest.TestCase):
    def test_model_is_not_loaded_when_object_is_created(self):
        loads = []
        SentenceTransformerEmbedder("m", model_loader=lambda n: loads.append(n))
        self.assertEqual(loads, [])

    def test_model_is_loaded_once_and_with_correct_name(self):
        loads = []

        def loader(name):
            loads.append(name)
            return FakeModel()

        emb = SentenceTransformerEmbedder("my/model", model_loader=loader)
        emb.embed_documents(["a"])
        emb.embed_documents(["b"])
        self.assertEqual(loads, ["my/model"])

    def test_empty_input_does_not_load_model(self):
        loads = []
        emb = SentenceTransformerEmbedder("m", model_loader=lambda n: loads.append(n))
        self.assertEqual(emb.embed_documents([]), [])
        self.assertEqual(loads, [])

    def test_encode_gets_batch_size_and_normalisation(self):
        model = FakeModel()
        embedder_with(model, batch_size=7).embed_documents(["one", "two"])
        texts, kwargs = model.encode_calls[0]
        self.assertEqual(texts, ["one", "two"])
        self.assertEqual(kwargs["batch_size"], 7)
        self.assertTrue(kwargs["normalize_embeddings"])
        self.assertFalse(kwargs["show_progress_bar"])

    def test_numpy_like_results_become_plain_lists(self):
        model = FakeModel(vectors=FakeNumpyLike([[0.1, 0.2], [0.3, 0.4]]))
        result = embedder_with(model).embed_documents(["a", "b"])
        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        self.assertIsInstance(result[0], list)

    def test_embed_query_returns_one_vector(self):
        vec = embedder_with(FakeModel()).embed_query("hello")
        self.assertEqual(len(vec), 3)

    def test_blank_query_is_rejected(self):
        for q in ("", "   ", None):
            with self.subTest(q=q):
                with self.assertRaises(EmbeddingError):
                    embedder_with(FakeModel()).embed_query(q)  # type: ignore[arg-type]

    def test_model_load_failure_is_reported_as_unavailable(self):
        def broken(name):
            raise OSError("no internet")

        emb = SentenceTransformerEmbedder("x/y", model_loader=broken)
        with self.assertRaises(EmbeddingUnavailableError) as ctx:
            emb.embed_documents(["a"])
        self.assertIn("x/y", str(ctx.exception))
        self.assertIn("internet", str(ctx.exception))
        self.assertNotIn("OSError", str(ctx.exception))

    def test_encode_failure_is_wrapped_in_friendly_error(self):
        with self.assertRaises(EmbeddingError) as ctx:
            embedder_with(FakeModel(fail=True)).embed_documents(["a"])
        self.assertNotIn("model crashed", str(ctx.exception))

    def test_wrong_number_of_vectors_is_an_error(self):
        with self.assertRaises(EmbeddingError):
            embedder_with(FakeModel(vectors=[[1.0, 2.0]])).embed_documents(["a", "b"])

    def test_nan_or_infinite_values_are_an_error(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(EmbeddingError):
                    embedder_with(FakeModel(vectors=[[1.0, bad]])).embed_documents(["a"])

    def test_vectors_of_different_lengths_are_an_error(self):
        with self.assertRaises(EmbeddingError):
            embedder_with(FakeModel(vectors=[[1.0, 2.0], [1.0]])).embed_documents(["a", "b"])

    def test_batch_size_is_never_below_one(self):
        self.assertEqual(SentenceTransformerEmbedder(batch_size=0).batch_size, 1)


# ----------------------------------------------------------------- availability
class AvailabilityTests(unittest.TestCase):
    def test_missing_library_gives_clear_install_message(self):
        # Forcing the import to fail makes this deterministic on any machine.
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            emb = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
            with self.assertRaises(EmbeddingUnavailableError) as ctx:
                emb.embed_documents(["hello"])
        self.assertIn("pip install sentence-transformers", str(ctx.exception))

    def test_unavailable_error_is_an_embedding_error(self):
        self.assertTrue(issubclass(EmbeddingUnavailableError, EmbeddingError))

    def test_no_fake_vectors_are_ever_returned_without_a_model(self):
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(EmbeddingError):
                SentenceTransformerEmbedder().embed_query("anything")

    def test_installed_check_returns_a_bool(self):
        self.assertIsInstance(sentence_transformers_installed(), bool)


# ----------------------------------------------------------------- pipeline + cache (MOCK embedder)
def sample_chunks(count_docs=2):
    docs = [
        make_document(f"d{i}.txt", [("Alpha sentence number %d is here. " % i) * 40],
                      content_hash=f"{i:02d}" * 32)
        for i in range(count_docs)
    ]
    return chunk_documents(docs, ChunkingConfig(300, 50))


class EmbedChunksTests(unittest.TestCase):
    def test_one_vector_per_chunk_in_order(self):
        chunks = sample_chunks()
        emb = FakeEmbedder()
        result = embed_chunks(chunks, emb)
        self.assertEqual(len(result.items), len(chunks))
        for item, chunk in zip(result.items, chunks):
            self.assertIs(item.chunk, chunk)
            self.assertEqual(len(item.vector), emb.dim)
        self.assertEqual(result.dimension, emb.dim)

    def test_metadata_travels_with_each_vector(self):
        chunks = sample_chunks()
        result = embed_chunks(chunks, FakeEmbedder())
        self.assertEqual({i.chunk.source for i in result.items}, {"d0.txt", "d1.txt"})
        self.assertEqual(result.items[0].chunk.chunk_index, 0)

    def test_empty_chunk_list(self):
        emb = FakeEmbedder()
        result = embed_chunks([], emb)
        self.assertEqual(result.items, [])
        self.assertEqual(result.dimension, 0)
        self.assertEqual(emb.calls, [])

    def test_second_run_reuses_cache_and_calls_model_zero_times(self):
        chunks, emb, cache = sample_chunks(), FakeEmbedder(), EmbeddingCache()
        first = embed_chunks(chunks, emb, cache)
        calls_after_first = len(emb.calls)
        second = embed_chunks(chunks, emb, cache)
        self.assertEqual(len(emb.calls), calls_after_first)
        self.assertEqual(second.newly_embedded, 0)
        self.assertEqual(second.reused, len(chunks))
        self.assertEqual([i.vector for i in first.items], [i.vector for i in second.items])

    def test_only_changed_text_is_re_embedded(self):
        emb, cache = FakeEmbedder(), EmbeddingCache()
        para1 = "Original paragraph one has enough words to fill a chunk on its own, honestly."
        para2_old = "The second paragraph is the one that will be edited by the author later on."
        para2_new = "A brand new second paragraph replaces the old one after the author edited it."
        doc_v1 = make_document("a.txt", [para1 + "\n\n" + para2_old])
        doc_v2 = make_document("a.txt", [para1 + "\n\n" + para2_new])
        cfg = ChunkingConfig(100, 0)
        first = chunk_document(doc_v1, cfg)
        self.assertEqual(len(first), 2)  # test premise: one chunk per paragraph
        embed_chunks(first, emb, cache)
        emb.calls.clear()
        result = embed_chunks(chunk_document(doc_v2, cfg), emb, cache)
        self.assertEqual(emb.calls, [[para2_new]])  # only the changed text was sent
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.newly_embedded, 1)
        self.assertEqual(result.reused, 1)

    def test_identical_chunks_are_embedded_once(self):
        doc_a = make_document("a.txt", ["Same text."], content_hash="aa" * 32)
        doc_b = make_document("b.txt", ["Same text."], content_hash="bb" * 32)
        emb = FakeEmbedder()
        result = embed_chunks(chunk_documents([doc_a, doc_b]), emb)
        self.assertEqual(emb.calls, [["Same text."]])
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.newly_embedded, 1)

    def test_cache_is_separate_for_each_model(self):
        chunks, cache = sample_chunks(1), EmbeddingCache()
        embed_chunks(chunks, FakeEmbedder("model-a"), cache)
        other = FakeEmbedder("model-b")
        result = embed_chunks(chunks, other, cache)
        self.assertEqual(result.reused, 0)
        self.assertTrue(other.calls)

    def test_failure_leaves_cache_unchanged_and_raises(self):
        class Broken(FakeEmbedder):
            def embed_documents(self, texts):
                raise EmbeddingError("boom")

        cache = EmbeddingCache()
        with self.assertRaises(EmbeddingError):
            embed_chunks(sample_chunks(), Broken(), cache)
        self.assertEqual(len(cache), 0)

    def test_model_returning_wrong_count_is_detected(self):
        class Short(FakeEmbedder):
            def embed_documents(self, texts):
                return super().embed_documents(texts)[:-1]

        with self.assertRaises(EmbeddingError):
            embed_chunks(sample_chunks(), Short())

    def test_unavailable_model_propagates_clear_error(self):
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(EmbeddingUnavailableError):
                embed_chunks(sample_chunks(), SentenceTransformerEmbedder())


class CosineTests(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2], [1])


class InterfaceReadinessTests(unittest.TestCase):
    """MOCK: checks the chunk -> vector -> similarity plumbing Day 3 will use.
    FakeEmbedder matches shared WORDS only; this says nothing about real
    semantic search quality."""

    def test_query_vector_is_comparable_with_chunk_vectors(self):
        chunks = []
        for path in sorted(SAMPLE_DOCS_DIR.iterdir()):
            chunks += chunk_document(load_path(path).document, ChunkingConfig(600, 100))
        emb = FakeEmbedder()
        result = embed_chunks(chunks, emb)
        query = emb.embed_query("password security multi-factor authentication phishing incident")
        best = max(result.items, key=lambda i: cosine_similarity(query, i.vector))
        # Only the source document is asserted: the toy embedder is too crude for more.
        self.assertEqual(best.chunk.source, "NovaTech_IT_Security_Policy.pdf")


# ----------------------------------------------------------------- OPTIONAL: real model
def _real_model_available() -> bool:
    if not sentence_transformers_installed():
        return False
    try:
        SentenceTransformerEmbedder().embed_query("test")
        return True
    except EmbeddingError:
        return False


@unittest.skipUnless(_real_model_available(),
                     "Real sentence-transformers + downloaded model not available here")
class RealSentenceTransformerTests(unittest.TestCase):
    def test_real_vectors_are_normalised_and_meaningful(self):
        emb = SentenceTransformerEmbedder()
        vecs = emb.embed_documents(
            ["Employees get 24 days of annual leave.",
             "Passwords must be 14 characters long.",
             "How many vacation days do I get?"]
        )
        self.assertEqual(len(vecs), 3)
        self.assertEqual(len(vecs[0]), len(vecs[1]))
        self.assertAlmostEqual(sum(x * x for x in vecs[0]), 1.0, places=3)
        self.assertGreater(cosine_similarity(vecs[2], vecs[0]), cosine_similarity(vecs[2], vecs[1]))


if __name__ == "__main__":
    unittest.main()
