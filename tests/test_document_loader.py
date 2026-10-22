"""Tests for core/document_loader.py.

Test groups:
  * REAL      - run the real loader on real files / real bytes.
  * MOCK      - use a FAKE 'fitz' (PyMuPDF) module to test our error handling
                logic. These do NOT prove the real PyMuPDF library works.
  * OPTIONAL  - only run when the real PyMuPDF is installed (skipped otherwise).
"""

import io
import sys
import types
import unittest
from unittest import mock

from core import document_loader as dl
from core.config import SAMPLE_DOCS_DIR
from core.document_loader import load_document, load_documents, load_path, safe_filename


def _pypdf_available() -> bool:
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False


def _pymupdf_available() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


def _sample_pdf_bytes() -> bytes:
    return (SAMPLE_DOCS_DIR / "NovaTech_IT_Security_Policy.pdf").read_bytes()


# ---------------------------------------------------------------- REAL: TXT
class TxtLoadingTests(unittest.TestCase):
    def test_plain_txt_loads(self):
        r = load_document("notes.txt", b"Hello world.\nSecond line.")
        self.assertTrue(r.ok)
        self.assertEqual(r.document.file_type, "txt")
        self.assertEqual(r.document.page_count, 1)
        self.assertIn("Second line.", r.document.text)

    def test_utf8_with_bom_is_cleaned(self):
        r = load_document("a.txt", "\ufeffHello".encode("utf-8"))
        self.assertEqual(r.document.text, "Hello")

    def test_utf16_txt_loads(self):
        r = load_document("a.txt", "Hello UTF16".encode("utf-16"))
        self.assertTrue(r.ok)
        self.assertEqual(r.document.text, "Hello UTF16")

    def test_non_utf8_txt_falls_back_with_warning(self):
        r = load_document("a.txt", "caf\xe9 menu".encode("cp1252"))
        self.assertTrue(r.ok)
        self.assertIn("café", r.document.text)
        self.assertTrue(r.document.warnings)

    def test_binary_file_named_txt_is_rejected(self):
        r = load_document("a.txt", b"\x00\x01\x02\x03binary")
        self.assertFalse(r.ok)
        self.assertIn("binary", r.error)

    def test_whitespace_only_txt_is_rejected(self):
        r = load_document("a.txt", b"   \n\n\t  \n")
        self.assertFalse(r.ok)
        self.assertIn("no readable text", r.error)

    def test_windows_line_endings_and_blank_lines_are_normalized(self):
        r = load_document("a.txt", b"One\r\n\r\n\r\n\r\nTwo   spaced")
        self.assertEqual(r.document.text, "One\n\nTwo spaced")


# ---------------------------------------------------------------- REAL: validation
class ValidationTests(unittest.TestCase):
    def test_unsupported_extension(self):
        r = load_document("report.docx", b"data")
        self.assertFalse(r.ok)
        self.assertIn(".docx", r.error)
        self.assertIn("PDF or TXT", r.error)

    def test_no_extension(self):
        r = load_document("README", b"data")
        self.assertFalse(r.ok)
        self.assertIn("no extension", r.error)

    def test_uppercase_extension_is_accepted(self):
        self.assertTrue(load_document("NOTES.TXT", b"hello").ok)

    def test_empty_file(self):
        r = load_document("a.txt", b"")
        self.assertFalse(r.ok)
        self.assertIn("empty", r.error)

    def test_file_too_large(self):
        r = load_document("a.txt", b"x" * 2048, max_bytes=1024)
        self.assertFalse(r.ok)
        self.assertIn("too large", r.error)

    def test_file_exactly_at_limit_is_allowed(self):
        self.assertTrue(load_document("a.txt", b"x" * 1024, max_bytes=1024).ok)

    def test_folder_parts_are_removed_from_filename(self):
        self.assertEqual(safe_filename("../../etc/passwd.txt"), "passwd.txt")
        self.assertEqual(safe_filename("C:\\Users\\me\\file.pdf"), "file.pdf")
        self.assertEqual(safe_filename(""), "unnamed")
        r = load_document("../../evil.txt", b"hello")
        self.assertEqual(r.document.name, "evil.txt")

    def test_content_hash_is_stable_and_content_specific(self):
        a1 = load_document("a.txt", b"same").document.content_hash
        a2 = load_document("b.txt", b"same").document.content_hash
        b = load_document("a.txt", b"different").document.content_hash
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)

    def test_load_never_raises_on_non_bytes_garbage(self):
        r = load_document("a.txt", None)  # type: ignore[arg-type]
        self.assertFalse(r.ok)  # handled as an error result, no exception


# ---------------------------------------------------------------- REAL: PDF
@unittest.skipUnless(_pypdf_available(), "pypdf not installed")
class PdfLoadingTests(unittest.TestCase):
    """These run through whichever real PDF engine is installed
    (PyMuPDF if present, otherwise pypdf)."""

    def test_all_bundled_sample_documents_load(self):
        files = sorted(SAMPLE_DOCS_DIR.iterdir())
        self.assertGreaterEqual(len(files), 3)
        for path in files:
            with self.subTest(file=path.name):
                r = load_path(path)
                self.assertTrue(r.ok, r.error)
                self.assertGreater(r.document.char_count, 200)

    def test_pdf_text_content_is_extracted(self):
        r = load_document("sec.pdf", _sample_pdf_bytes())
        self.assertTrue(r.ok)
        self.assertEqual(r.document.file_type, "pdf")
        self.assertIn("14 characters", r.document.text.replace("\n", " "))
        self.assertIn("security@novatech.example", r.document.text)

    def test_multiple_files_one_bad_does_not_stop_others(self):
        results = load_documents(
            [
                ("good.pdf", _sample_pdf_bytes()),
                ("bad.pdf", b"%PDF-1.4 this is not really a pdf"),
                ("notes.txt", b"hello there"),
                ("sheet.xlsx", b"zzz"),
            ]
        )
        self.assertEqual([r.ok for r in results], [True, False, True, False])

    def test_file_without_pdf_header_is_rejected(self):
        r = load_document("fake.pdf", b"This is just text pretending to be a PDF.")
        self.assertFalse(r.ok)
        self.assertIn("could not be read", r.error)

    def test_corrupted_pdf_is_rejected_with_friendly_message(self):
        r = load_document("bad.pdf", b"%PDF-1.4\n1 0 obj\n<< garbage >>\nendobj\n")
        self.assertFalse(r.ok)
        self.assertNotIn("Traceback", r.error)

    def test_truncated_pdf_does_not_crash(self):
        data = _sample_pdf_bytes()
        r = load_document("cut.pdf", data[: len(data) // 3])
        self.assertIsNotNone(r)  # must return a result (ok or error) - never raise
        if not r.ok:
            self.assertTrue(r.error)

    def test_pdf_with_blank_page_only_reports_no_readable_text(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        buf = io.BytesIO()
        writer.write(buf)
        r = load_document("blank.pdf", buf.getvalue())
        self.assertFalse(r.ok)
        self.assertIn("No readable text", r.error)

    def test_password_protected_pdf_gives_clear_message(self):
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(_sample_pdf_bytes()))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        try:
            writer.encrypt("user-secret", "owner-secret")
        except Exception as exc:  # e.g. 'cryptography' package missing
            self.skipTest(f"Cannot create an encrypted test PDF here: {type(exc).__name__}")
        buf = io.BytesIO()
        writer.write(buf)
        r = load_document("locked.pdf", buf.getvalue())
        self.assertFalse(r.ok)
        self.assertIn("password", r.error.lower())


# ---------------------------------------------------------------- MOCK: PyMuPDF
def _fake_fitz(open_behaviour):
    module = types.ModuleType("fitz")
    module.open = open_behaviour
    return module


class FakePage:
    def __init__(self, text=None, fail=False):
        self._text, self._fail = text, fail

    def get_text(self, _mode):
        if self._fail:
            raise RuntimeError("boom")
        return self._text


class FakeDoc:
    def __init__(self, pages, needs_pass=False):
        self._pages = pages
        self.needs_pass = needs_pass
        self.closed = False

    @property
    def page_count(self):
        return len(self._pages)

    def load_page(self, i):
        return self._pages[i]

    def close(self):
        self.closed = True


class MockPyMuPdfTests(unittest.TestCase):
    """MOCK tests: a fake PyMuPDF checks OUR logic only (not the real library)."""

    PDF_BYTES = b"%PDF-1.7 fake"

    def _load(self, fake_open):
        with mock.patch.dict(sys.modules, {"fitz": _fake_fitz(fake_open)}):
            return load_document("x.pdf", self.PDF_BYTES)

    def test_mock_text_is_used_and_parser_is_pymupdf(self):
        doc = FakeDoc([FakePage("Page one text"), FakePage("Page two text")])
        r = self._load(lambda **kw: doc)
        self.assertTrue(r.ok)
        self.assertEqual(r.document.parser, "pymupdf")
        self.assertEqual([p.page_number for p in r.document.pages], [1, 2])
        self.assertTrue(doc.closed)

    def test_mock_encrypted_pdf(self):
        doc = FakeDoc([FakePage("secret")], needs_pass=True)
        r = self._load(lambda **kw: doc)
        self.assertFalse(r.ok)
        self.assertIn("password", r.error.lower())
        self.assertTrue(doc.closed)

    def test_mock_open_failure_is_reported_as_unreadable(self):
        def broken_open(**kw):
            raise RuntimeError("cannot open")

        r = self._load(broken_open)
        self.assertFalse(r.ok)
        self.assertIn("could not be read", r.error)

    def test_mock_one_failing_page_is_skipped_with_warning(self):
        doc = FakeDoc([FakePage("Good page"), FakePage(fail=True), FakePage("Another good page")])
        r = self._load(lambda **kw: doc)
        self.assertTrue(r.ok)
        self.assertEqual(r.document.page_count, 2)
        self.assertTrue(any("Page 2" in w for w in r.document.warnings))

    def test_mock_scanned_pdf_has_no_text(self):
        doc = FakeDoc([FakePage(""), FakePage("  ")])
        r = self._load(lambda **kw: doc)
        self.assertFalse(r.ok)
        self.assertIn("scanned", r.error)

    def test_no_pdf_engine_installed_gives_helpful_message(self):
        # Setting a module to None in sys.modules makes 'import x' raise ImportError.
        with mock.patch.dict(sys.modules, {"fitz": None, "pypdf": None}):
            r = load_document("x.pdf", self.PDF_BYTES)
        self.assertFalse(r.ok)
        self.assertIn("No PDF reader", r.error)

    def test_pypdf_is_used_when_pymupdf_is_missing(self):
        if not _pypdf_available():
            self.skipTest("pypdf not installed")
        with mock.patch.dict(sys.modules, {"fitz": None}):
            r = load_document("sec.pdf", _sample_pdf_bytes())
        self.assertTrue(r.ok)
        self.assertEqual(r.document.parser, "pypdf")


# ---------------------------------------------------------------- OPTIONAL: real PyMuPDF
@unittest.skipUnless(_pymupdf_available(), "Real PyMuPDF not installed in this environment")
class RealPyMuPdfTests(unittest.TestCase):
    def test_real_pymupdf_reads_sample_pdf(self):
        r = load_document("sec.pdf", _sample_pdf_bytes())
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.document.parser, "pymupdf")
        self.assertIn("NovaVPN", r.document.text)

    def test_real_pymupdf_rejects_garbage(self):
        r = load_document("bad.pdf", b"%PDF-1.4 garbage garbage")
        self.assertFalse(r.ok)


if __name__ == "__main__":
    unittest.main()
