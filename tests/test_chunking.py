"""Tests for core/chunking.py. All REAL (no mocks): they run the real chunker."""

import time
import unittest

from core.chunking import (
    Chunk,
    ChunkingConfig,
    chunk_document,
    chunk_documents,
    split_text,
    split_text_spans,
)
from core.config import SAMPLE_DOCS_DIR
from core.document_loader import load_path
from tests.helpers import make_document

SENTENCE = "This is a sentence about company policy. "


def long_text(paragraphs=12, sentences=8):
    return "\n\n".join((SENTENCE * sentences).strip() for _ in range(paragraphs))


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        cfg = ChunkingConfig()
        self.assertEqual((cfg.chunk_size, cfg.chunk_overlap), (600, 100))

    def test_invalid_settings_are_rejected(self):
        bad = [(0, 0), (-5, 0), (99, 0), (4001, 0), (500, -1), (500, 251), (500, 500), (600, 700)]
        for size, overlap in bad:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    ChunkingConfig(size, overlap)

    def test_boundary_values_are_accepted(self):
        for size, overlap in [(100, 0), (100, 50), (4000, 2000)]:
            cfg = ChunkingConfig(size, overlap)
            self.assertEqual((cfg.chunk_size, cfg.chunk_overlap), (size, overlap))

    def test_non_integers_are_rejected(self):
        with self.assertRaises(ValueError):
            ChunkingConfig("500", 50)  # type: ignore[arg-type]


class SplitTextTests(unittest.TestCase):
    def test_empty_and_whitespace_input_gives_no_chunks(self):
        for text in ("", "   ", "\n\n\t \n", None):
            with self.subTest(text=text):
                self.assertEqual(split_text(text), [])  # type: ignore[arg-type]

    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_text("Just one short line."), ["Just one short line."])

    def test_text_exactly_chunk_size_is_one_chunk(self):
        text = "a" * 49 + " " + "b" * 50  # 100 chars
        self.assertEqual(len(text), 100)
        self.assertEqual(len(split_text(text, ChunkingConfig(100, 0))), 1)

    def test_no_chunk_is_longer_than_chunk_size(self):
        text = long_text()
        for size, overlap in [(100, 0), (100, 50), (300, 60), (600, 100), (1000, 200)]:
            with self.subTest(size=size, overlap=overlap):
                for s, e in split_text_spans(text, ChunkingConfig(size, overlap)):
                    self.assertLessEqual(e - s, size)

    def test_chunks_are_exact_slices_without_edge_whitespace(self):
        text = long_text()
        for s, e in split_text_spans(text, ChunkingConfig(300, 50)):
            chunk = text[s:e]
            self.assertTrue(chunk)
            self.assertEqual(chunk, chunk.strip())

    def test_no_text_is_lost(self):
        """Every non-space character of the input lies inside at least one chunk."""
        text = long_text() + "\n\nA final unusual paragraph: zebra-quartz 12345."
        for size, overlap in [(100, 0), (150, 30), (600, 100)]:
            with self.subTest(size=size, overlap=overlap):
                covered = [False] * len(text)
                for s, e in split_text_spans(text, ChunkingConfig(size, overlap)):
                    for i in range(s, e):
                        covered[i] = True
                missing = [i for i, ch in enumerate(text) if not ch.isspace() and not covered[i]]
                self.assertEqual(missing, [])

    def test_chunks_follow_document_order(self):
        spans = split_text_spans(long_text(), ChunkingConfig(200, 40))
        starts = [s for s, _ in spans]
        ends = [e for _, e in spans]
        self.assertEqual(starts, sorted(set(starts)))
        self.assertEqual(ends, sorted(set(ends)))

    def test_deterministic(self):
        text = long_text()
        self.assertEqual(split_text_spans(text), split_text_spans(text))


class OverlapTests(unittest.TestCase):
    def test_neighbouring_chunks_share_text_when_overlap_is_set(self):
        text = long_text()
        spans = split_text_spans(text, ChunkingConfig(300, 80))
        self.assertGreater(len(spans), 3)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            self.assertLess(s2, e1, "next chunk should start before the previous one ends")
            self.assertGreater(e2, e1, "next chunk must add new text")

    def test_overlap_never_exceeds_setting(self):
        text = long_text()
        for overlap in (20, 50, 100, 150):
            with self.subTest(overlap=overlap):
                spans = split_text_spans(text, ChunkingConfig(300, overlap))
                for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
                    self.assertLessEqual(max(0, e1 - s2), overlap)

    def test_zero_overlap_means_no_shared_text(self):
        spans = split_text_spans(long_text(), ChunkingConfig(300, 0))
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            self.assertGreaterEqual(s2, e1)

    def test_overlapping_text_is_really_repeated(self):
        text = long_text()
        spans = split_text_spans(text, ChunkingConfig(300, 80))
        (s1, e1), (s2, e2) = spans[0], spans[1]
        shared = text[s2:e1]
        self.assertTrue(shared)
        self.assertTrue(text[s1:e1].endswith(shared))
        self.assertTrue(text[s2:e2].startswith(shared))


class BoundaryTests(unittest.TestCase):
    def test_paragraphs_are_kept_whole_when_they_fit(self):
        p1, p2 = "A" * 40 + " " + "B" * 40, "C" * 40 + " " + "D" * 40
        chunks = split_text(p1 + "\n\n" + p2, ChunkingConfig(100, 0))
        self.assertEqual(chunks, [p1, p2])

    def test_long_paragraph_splits_at_sentence_ends(self):
        text = (SENTENCE * 30).strip()
        chunks = split_text(text, ChunkingConfig(200, 0))
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertTrue(chunk.endswith("policy."), chunk[-20:])
            self.assertTrue(chunk.startswith("This is"), chunk[:20])

    def test_numbered_heading_is_not_separated_from_its_title(self):
        text = (SENTENCE * 4).strip() + "\n4. Data Classification\n" + (SENTENCE * 4).strip()
        for chunk in split_text(text, ChunkingConfig(180, 0)):
            self.assertFalse(chunk.rstrip().endswith("4."), chunk[-15:])

    def test_long_text_without_spaces_is_still_split_with_overlap(self):
        text = "X" * 2500
        spans = split_text_spans(text, ChunkingConfig(300, 50))
        self.assertGreater(len(spans), 5)
        for s, e in spans:
            self.assertLessEqual(e - s, 300)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 2500)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            self.assertEqual(e1 - s2, 50)

    def test_unicode_text_is_safe(self):
        text = ("नमस्ते दुनिया. यह एक परीक्षण है. café naïve 日本語のテキスト. 😀 " * 40).strip()
        spans = split_text_spans(text, ChunkingConfig(200, 40))
        self.assertGreater(len(spans), 1)
        for s, e in spans:
            self.assertLessEqual(e - s, 200)

    def test_windows_style_blank_lines(self):
        text = "First paragraph here.\n \nSecond paragraph here."
        self.assertEqual(len(split_text(text, ChunkingConfig(100, 0))), 1)  # fits in one
        self.assertEqual(
            split_text(text, ChunkingConfig(100, 0)), [text]
        )

    def test_large_input_is_fast(self):
        text = long_text(paragraphs=2000, sentences=10)  # ~ 800 KB
        start = time.time()
        spans = split_text_spans(text, ChunkingConfig(600, 100))
        elapsed = time.time() - start
        self.assertGreater(len(spans), 1000)
        self.assertLess(elapsed, 10, f"took {elapsed:.1f}s")


class MetadataTests(unittest.TestCase):
    def test_basic_metadata(self):
        doc = make_document("policy.txt", [long_text()], content_hash="cd" * 32)
        chunks = chunk_document(doc, ChunkingConfig(300, 50))
        self.assertGreater(len(chunks), 3)
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, Chunk)
            self.assertEqual(chunk.source, "policy.txt")
            self.assertEqual(chunk.doc_hash, "cd" * 32)
            self.assertEqual(chunk.file_type, "txt")
            self.assertEqual(chunk.chunk_index, i)
            self.assertEqual(chunk.total_chunks, len(chunks))
            self.assertEqual(chunk.text, doc.text[chunk.char_start:chunk.char_end])

    def test_chunk_ids_are_unique_and_stable(self):
        doc = make_document("a.txt", [long_text()])
        ids1 = [c.chunk_id for c in chunk_document(doc, ChunkingConfig(300, 50))]
        ids2 = [c.chunk_id for c in chunk_document(doc, ChunkingConfig(300, 50))]
        self.assertEqual(ids1, ids2)
        self.assertEqual(len(ids1), len(set(ids1)))

    def test_metadata_dict_contains_only_simple_values(self):
        chunk = chunk_document(make_document(), ChunkingConfig(300, 50))[0]
        meta = chunk.metadata()
        for key in ("source", "doc_hash", "chunk_index", "total_chunks",
                    "page_start", "page_end", "char_start", "char_end", "file_type"):
            self.assertIn(key, meta)
        for value in meta.values():
            self.assertIsInstance(value, (str, int))

    def test_page_numbers_for_multi_page_document(self):
        page1 = "Page one text. " * 30   # ~450 chars
        page2 = "Page two text. " * 30
        page3 = "Page three text. " * 30
        doc = make_document("m.pdf", [page1.strip(), page2.strip(), page3.strip()], file_type="pdf")
        chunks = chunk_document(doc, ChunkingConfig(300, 50))
        self.assertEqual(chunks[0].page_start, 1)
        self.assertEqual(chunks[-1].page_end, 3)
        for chunk in chunks:
            self.assertLessEqual(chunk.page_start, chunk.page_end)
            self.assertIn(chunk.page_start, (1, 2, 3))
        # a chunk containing only page-two text must be labelled page 2
        only_two = [c for c in chunks if "Page two" in c.text and "Page one" not in c.text
                    and "Page three" not in c.text]
        self.assertTrue(only_two)
        for c in only_two:
            self.assertEqual((c.page_start, c.page_end), (2, 2))

    def test_page_numbers_follow_original_numbers_when_pages_were_skipped(self):
        # e.g. page 2 was blank and dropped by the loader
        doc = make_document("gap.pdf", ["First page text."], file_type="pdf")
        doc.pages[0].page_number = 3
        chunk = chunk_document(doc)[0]
        self.assertEqual((chunk.page_start, chunk.page_end), (3, 3))

    def test_empty_document_gives_no_chunks(self):
        doc = make_document("empty.txt", [""])
        self.assertEqual(chunk_document(doc), [])
        self.assertEqual(chunk_document(make_document("e.txt", [])), [])

    def test_chunk_documents_restarts_numbering_per_document(self):
        a = make_document("a.txt", [long_text()], content_hash="aa" * 32)
        b = make_document("b.txt", [long_text()], content_hash="bb" * 32)
        chunks = chunk_documents([a, b], ChunkingConfig(300, 50))
        by_source = {}
        for c in chunks:
            by_source.setdefault(c.source, []).append(c)
        self.assertEqual(set(by_source), {"a.txt", "b.txt"})
        for source_chunks in by_source.values():
            self.assertEqual([c.chunk_index for c in source_chunks], list(range(len(source_chunks))))
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_chunk_documents_with_no_documents(self):
        self.assertEqual(chunk_documents([]), [])


class SampleDocumentTests(unittest.TestCase):
    def test_all_sample_documents_chunk_correctly(self):
        for path in sorted(SAMPLE_DOCS_DIR.iterdir()):
            with self.subTest(file=path.name):
                doc = load_path(path).document
                chunks = chunk_document(doc, ChunkingConfig(600, 100))
                self.assertGreaterEqual(len(chunks), 2)
                for c in chunks:
                    self.assertTrue(c.text.strip())
                    self.assertLessEqual(len(c.text), 600)
                    self.assertEqual(c.source, path.name)

    def test_key_fact_survives_chunking_intact(self):
        doc = load_path(SAMPLE_DOCS_DIR / "NovaTech_IT_Security_Policy.pdf").document
        chunks = chunk_document(doc, ChunkingConfig(600, 100))
        joined = [" ".join(c.text.split()) for c in chunks]
        self.assertTrue(any("at least 14 characters long" in t for t in joined))
        self.assertTrue(any("security@novatech.example within 1 hour" in t for t in joined))


if __name__ == "__main__":
    unittest.main()
