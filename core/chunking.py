"""Text chunking for IntelliBot (Day 2).

Splits a loaded document into overlapping chunks that are small enough to
embed and search, while keeping useful metadata (source file, chunk number,
page numbers, character positions).

How the splitter works (the same idea as LangChain's
RecursiveCharacterTextSplitter, written here without the extra dependency):

1. Try to split on the biggest natural boundary first: paragraphs.
2. Any piece that is still too big is split on the next boundary: sentences,
   then lines, then words, and finally single characters as a last resort.
3. Small neighbouring pieces are merged back together up to ``chunk_size``.
4. Each new chunk starts with a little text from the end of the previous chunk
   (the *overlap*), so facts at a boundary are not cut off.

Sizes are measured in characters. Chunk text is always an exact slice of the
document text (``text[char_start:char_end]``), which makes chunks easy to
verify and to cite later.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, Iterable, List, Pattern, Sequence, Tuple, Union

from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    chunk_settings_are_valid,
)
from .document_loader import LoadedDocument

Span = Tuple[int, int]  # (start, end) positions in the original text

# Boundaries from largest to smallest. The last entry (None) means
# "cut at a fixed number of characters" and is only a last resort.
_BOUNDARIES: Sequence[Union[Pattern[str], None]] = (
    re.compile(r"\n\s*\n"),  # paragraph break
    # Sentence end. The punctuation stays with its sentence, and a numbered
    # heading such as "4. Title" is NOT split after its number.
    re.compile(r"(?<=[.!?])(?<!^\d\.)(?<!^\d\d\.)(?<!\n\d\.)(?<!\n\d\d\.)\s+"),
    re.compile(r"\n"),  # line break
    re.compile(r"\s+"),  # any whitespace (word boundary)
    None,  # fixed-size cut
)


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk size and overlap, both in characters."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_size, int) or not isinstance(self.chunk_overlap, int):
            raise ValueError("chunk_size and chunk_overlap must be whole numbers.")
        if not chunk_settings_are_valid(self.chunk_size, self.chunk_overlap):
            raise ValueError(
                f"Invalid chunk settings: chunk_size must be {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} "
                f"and chunk_overlap must be between 0 and half of chunk_size "
                f"(got size={self.chunk_size}, overlap={self.chunk_overlap})."
            )


@dataclass(frozen=True)
class Chunk:
    """One searchable piece of a document, with its metadata."""

    text: str
    source: str  # document file name, e.g. "NovaTech_IT_Security_Policy.pdf"
    doc_hash: str  # sha256 of the original file
    file_type: str  # "pdf" or "txt"
    chunk_index: int  # 0-based position inside its document
    total_chunks: int  # number of chunks in its document
    page_start: int  # 1-based page where the chunk starts
    page_end: int  # 1-based page where the chunk ends
    char_start: int  # position inside document.text
    char_end: int

    @property
    def chunk_id(self) -> str:
        """Unique, stable ID (same file + same settings -> same IDs)."""
        return f"{self.doc_hash[:16]}-{self.chunk_index:04d}"

    def metadata(self) -> Dict[str, Union[str, int]]:
        """Plain text/number metadata (the form vector databases accept)."""
        return {
            "source": self.source,
            "doc_hash": self.doc_hash,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def _strip_span(text: str, start: int, end: int) -> Span:
    """Shrink (start, end) so it does not begin or end with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _atomic_spans(
    text: str, start: int, end: int, level: int, chunk_size: int, grain: int
) -> List[Span]:
    """Cut text[start:end] into pieces that are each <= chunk_size long."""
    start, end = _strip_span(text, start, end)
    if start >= end:
        return []
    if end - start <= chunk_size:
        return [(start, end)]

    pattern = _BOUNDARIES[level]
    if pattern is None:
        # Last resort: fixed-size cuts. ``grain`` is small (the overlap size) so
        # that overlap still works for very long text without spaces.
        return [
            (pos, min(pos + grain, end)) for pos in range(start, end, grain)
        ]

    segment = text[start:end]
    pieces: List[Span] = []
    cursor = 0
    for match in pattern.finditer(segment):
        if match.end() == match.start():
            continue
        pieces.append((start + cursor, start + match.start()))
        cursor = match.end()
    pieces.append((start + cursor, end))

    if len(pieces) == 1:  # this boundary does not occur; try a smaller one
        return _atomic_spans(text, start, end, level + 1, chunk_size, grain)

    spans: List[Span] = []
    for piece_start, piece_end in pieces:
        spans.extend(
            _atomic_spans(text, piece_start, piece_end, level + 1, chunk_size, grain)
        )
    return spans


def split_text_spans(text: str, config: ChunkingConfig = ChunkingConfig()) -> List[Span]:
    """Split text into overlapping chunks. Returns (start, end) positions."""
    if not text or not text.strip():
        return []
    size, overlap = config.chunk_size, config.chunk_overlap
    grain = overlap if overlap > 0 else size
    atoms = _atomic_spans(text, 0, len(text), 0, size, grain)

    groups: List[List[Span]] = []
    current: List[Span] = []
    for span in atoms:
        if current and span[1] - current[0][0] > size:
            groups.append(current)
            # Carry the tail of the finished chunk into the next one (overlap).
            carry: List[Span] = []
            last_end = current[-1][1]
            for previous in reversed(current):
                if last_end - previous[0] <= overlap:
                    carry.insert(0, previous)
                else:
                    break
            while carry and span[1] - carry[0][0] > size:
                carry.pop(0)
            current = carry
        current.append(span)
    if current:
        groups.append(current)

    return [(group[0][0], group[-1][1]) for group in groups]


def split_text(text: str, config: ChunkingConfig = ChunkingConfig()) -> List[str]:
    """Convenience wrapper: the chunk strings only."""
    return [text[s:e] for s, e in split_text_spans(text, config)]


# --------------------------------------------------------------------------
# Documents -> chunks with metadata
# --------------------------------------------------------------------------


def _page_starts(document: LoadedDocument) -> List[int]:
    """Start position of each page inside ``document.text`` (pages joined by a blank line)."""
    starts: List[int] = []
    position = 0
    for page in document.pages:
        starts.append(position)
        position += len(page.text) + 2  # the two newlines used to join pages
    return starts


def chunk_document(
    document: LoadedDocument, config: ChunkingConfig = ChunkingConfig()
) -> List[Chunk]:
    """Split one loaded document into Chunk objects with metadata."""
    text = document.text
    spans = split_text_spans(text, config)
    if not spans:
        return []

    starts = _page_starts(document)
    page_numbers = [page.page_number for page in document.pages]

    def page_at(position: int) -> int:
        index = max(0, bisect_right(starts, position) - 1)
        return page_numbers[index]

    total = len(spans)
    return [
        Chunk(
            text=text[start:end],
            source=document.name,
            doc_hash=document.content_hash,
            file_type=document.file_type,
            chunk_index=index,
            total_chunks=total,
            page_start=page_at(start),
            page_end=page_at(end - 1),
            char_start=start,
            char_end=end,
        )
        for index, (start, end) in enumerate(spans)
    ]


def chunk_documents(
    documents: Iterable[LoadedDocument], config: ChunkingConfig = ChunkingConfig()
) -> List[Chunk]:
    """Chunk several documents. Chunk numbering restarts for each document."""
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, config))
    return chunks
