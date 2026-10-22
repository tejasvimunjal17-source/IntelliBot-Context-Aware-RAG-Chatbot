"""Document loading for IntelliBot (Day 1).

Turns uploaded PDF / TXT files into clean text, page by page.

Design rules:
* One bad file must never crash the app. ``load_document`` never raises; it
  returns a ``LoadResult`` that holds either a document or a friendly error.
* Error messages are written for end users (no tracebacks, no internals).
* PyMuPDF is the main PDF engine (as required by the brief). If it is not
  installed, ``pypdf`` is used as a backup so the app still works.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """A problem with one file. The message is safe to show to the user."""


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str


@dataclass
class LoadedDocument:
    name: str
    file_type: str  # "pdf" or "txt"
    pages: List[PageText]
    size_bytes: int
    content_hash: str  # sha256 of the file; lets later days skip unchanged files
    parser: str  # which engine read it: "pymupdf", "pypdf" or "text"
    warnings: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


@dataclass
class LoadResult:
    filename: str
    document: Optional[LoadedDocument] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.document is not None and self.error is None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def safe_filename(name: str) -> str:
    """Keep only the file name part (no folders) and strip odd characters."""
    base = PurePath(str(name).replace("\\", "/")).name.strip()
    base = re.sub(r"[\x00-\x1f]", "", base)
    return base or "unnamed"


def _normalize_text(text: str) -> str:
    """Clean extracted text without changing its meaning."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


# --------------------------------------------------------------------------
# TXT
# --------------------------------------------------------------------------


def _decode_text_bytes(data: bytes) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), warnings
        except UnicodeDecodeError:
            raise DocumentLoadError(
                "This text file could not be read. Please save it as UTF-8 and try again."
            )
    if b"\x00" in data:
        raise DocumentLoadError(
            "This file looks like a binary file, not plain text. "
            "Please upload a real .txt file."
        )
    try:
        return data.decode("utf-8-sig"), warnings
    except UnicodeDecodeError:
        pass
    try:
        text = data.decode("cp1252")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    warnings.append(
        "File is not UTF-8; it was read using a Western-European encoding. "
        "Some special characters may look different."
    )
    return text, warnings


def _extract_txt(data: bytes) -> Tuple[List[PageText], List[str], str]:
    raw, warnings = _decode_text_bytes(data)
    text = _normalize_text(raw)
    pages = [PageText(1, text)] if text else []
    return pages, warnings, "text"


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

_ENCRYPTED_MSG = (
    "This PDF is password-protected. Please remove the password and upload it again."
)
_MALFORMED_MSG = (
    "This PDF could not be read. It may be damaged or not a real PDF file."
)


def _extract_pdf_pymupdf(data: bytes) -> Tuple[List[PageText], List[str]]:
    import fitz  # PyMuPDF  (ImportError is handled by the caller)

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types
        logger.warning("PyMuPDF could not open a file: %s", type(exc).__name__)
        raise DocumentLoadError(_MALFORMED_MSG) from exc

    pages: List[PageText] = []
    warnings: List[str] = []
    try:
        if getattr(doc, "needs_pass", False):
            raise DocumentLoadError(_ENCRYPTED_MSG)
        for index in range(doc.page_count):
            try:
                raw = doc.load_page(index).get_text("text")
            except Exception as exc:
                logger.warning("PyMuPDF failed on a page: %s", type(exc).__name__)
                warnings.append(f"Page {index + 1} could not be read and was skipped.")
                continue
            pages.append(PageText(index + 1, _normalize_text(raw)))
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return pages, warnings


def _extract_pdf_pypdf(data: bytes) -> Tuple[List[PageText], List[str]]:
    from pypdf import PdfReader  # ImportError is handled by the caller

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        logger.warning("pypdf could not open a file: %s", type(exc).__name__)
        raise DocumentLoadError(_MALFORMED_MSG) from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")  # many PDFs use an empty user password
        except Exception as exc:
            raise DocumentLoadError(_ENCRYPTED_MSG) from exc
        if not unlocked:
            raise DocumentLoadError(_ENCRYPTED_MSG)

    pages: List[PageText] = []
    warnings: List[str] = []
    try:
        total = len(reader.pages)
    except Exception as exc:
        raise DocumentLoadError(_MALFORMED_MSG) from exc
    for index in range(total):
        try:
            raw = reader.pages[index].extract_text() or ""
        except Exception as exc:
            logger.warning("pypdf failed on a page: %s", type(exc).__name__)
            warnings.append(f"Page {index + 1} could not be read and was skipped.")
            continue
        pages.append(PageText(index + 1, _normalize_text(raw)))
    return pages, warnings


def _extract_pdf(data: bytes) -> Tuple[List[PageText], List[str], str]:
    if b"%PDF" not in data[:1024]:
        raise DocumentLoadError(_MALFORMED_MSG)
    try:
        pages, warnings = _extract_pdf_pymupdf(data)
        return pages, warnings, "pymupdf"
    except ImportError:
        pass
    try:
        pages, warnings = _extract_pdf_pypdf(data)
        return pages, warnings, "pypdf"
    except ImportError:
        raise DocumentLoadError(
            "No PDF reader is installed on this server. "
            "Install the requirements (PyMuPDF) and restart the app."
        )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def load_document(
    filename: str,
    data: bytes,
    max_bytes: Optional[int] = None,
    allowed_extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
) -> LoadResult:
    """Read one uploaded file. Never raises; errors come back in the result."""
    name = safe_filename(filename)
    try:
        extension = Path(name).suffix.lower()
        if extension not in allowed_extensions:
            shown = extension or "(no extension)"
            raise DocumentLoadError(
                f"Unsupported file type {shown}. Please upload PDF or TXT files only."
            )
        if not data:
            raise DocumentLoadError("This file is empty.")
        if max_bytes is not None and len(data) > max_bytes:
            raise DocumentLoadError(
                f"This file is too large ({_format_mb(len(data))}). "
                f"The limit is {_format_mb(max_bytes)}."
            )

        if extension == ".pdf":
            pages, warnings, parser = _extract_pdf(data)
            file_type = "pdf"
        else:
            pages, warnings, parser = _extract_txt(data)
            file_type = "txt"

        readable = [page for page in pages if page.text]
        skipped_blank = len(pages) - len(readable)
        if not readable:
            if file_type == "pdf":
                raise DocumentLoadError(
                    "No readable text was found in this PDF. It may be a scanned "
                    "image, which IntelliBot cannot read yet."
                )
            raise DocumentLoadError("This text file has no readable text.")
        if skipped_blank:
            warnings.append(f"{skipped_blank} page(s) had no readable text and were left out.")

        document = LoadedDocument(
            name=name,
            file_type=file_type,
            pages=readable,
            size_bytes=len(data),
            content_hash=hashlib.sha256(data).hexdigest(),
            parser=parser,
            warnings=list(warnings),
        )
        return LoadResult(filename=name, document=document)

    except DocumentLoadError as exc:
        return LoadResult(filename=name, error=str(exc))
    except Exception as exc:  # last safety net: never crash the app
        logger.exception("Unexpected error while loading a file (%s)", type(exc).__name__)
        return LoadResult(
            filename=name,
            error="Something went wrong while reading this file. Please try another file.",
        )


def load_path(
    path: Path,
    max_bytes: Optional[int] = None,
    allowed_extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
) -> LoadResult:
    """Read a file from disk (used for the bundled sample documents)."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError:
        return LoadResult(
            filename=safe_filename(path.name),
            error="This file could not be opened.",
        )
    return load_document(path.name, data, max_bytes, allowed_extensions)


def load_documents(
    items: Iterable[Tuple[str, bytes]],
    max_bytes: Optional[int] = None,
    allowed_extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
) -> List[LoadResult]:
    """Load many (filename, bytes) pairs. One failure never stops the others."""
    return [
        load_document(name, data, max_bytes, allowed_extensions) for name, data in items
    ]
