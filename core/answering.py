"""Grounded answer generation (Day 4).

Flow for one question
  1. Retrieve relevant chunks (Day 3). Retrieval problems stop here with their own status.
  2. If nothing relevant was found: return the safe fallback. The LLM is NOT called,
     so it cannot make up an answer.
  3. Otherwise build a grounded prompt and call the LLM. LLM problems have their own status.
  4. Check the reply before showing it:
       * the "NOT_IN_DOCUMENTS" reply becomes the honest fallback,
       * citations like [S2] must refer to passages that were really retrieved;
         unknown ones are removed and reported,
       * an answer with no valid citation is not shown (status UNVERIFIED),
       * a reply that leaks the secret marker of the system prompt is blocked,
       * numbers that do not occur in the sources are reported as a warning,
       * file names in the answer that were not retrieved are reported as a warning,
       * markdown images/links are stripped before display.
  5. Return an ``Answer`` with the text, the sources (file, pages, excerpt, score)
     and any warnings.

``AnswerEngine.answer`` does not raise for expected failures: they come back as a status.
These checks reduce, but cannot eliminate, wrong or manipulated answers (see README).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

from .embeddings import EmbeddingError
from .llm import LLMClient, LLMError
from .prompting import (
    CANARY,
    NOT_IN_DOCUMENTS_TOKEN,
    build_messages,
    find_injection_signs,
    passage_id,
)
from .retrieval import (
    STATUS_EMPTY_QUERY,
    STATUS_NO_DOCUMENTS,
    STATUS_NO_RELEVANT,
    STATUS_OK,
    STATUS_QUERY_TOO_LONG,
    RetrievalResponse,
    Retriever,
)
from .vector_store import SearchResult, VectorStoreError

logger = logging.getLogger(__name__)

# ---- answer statuses
ANSWERED = "answered"  # a verified, cited answer
NOT_IN_DOCUMENTS = "not_in_documents"  # safe fallback: documents do not contain the answer
EMPTY_QUERY = "empty_query"
QUERY_TOO_LONG = "query_too_long"
NO_DOCUMENTS = "no_documents"  # nothing indexed to search
RETRIEVAL_ERROR = "retrieval_error"  # embedding model / vector store problem
LLM_UNAVAILABLE = "llm_unavailable"  # no API key / package missing
LLM_ERROR = "llm_error"  # provider call failed
UNVERIFIED = "unverified"  # model reply had no valid citation -> not shown
BLOCKED = "blocked"  # model reply leaked the system prompt marker -> not shown

FALLBACK_MESSAGE = "I couldn't find the answer to this in your uploaded documents."
UNVERIFIED_MESSAGE = (
    "I could not verify the model's reply against your documents (it did not cite a "
    "source), so I am not showing it. Please try rephrasing your question."
)
BLOCKED_MESSAGE = (
    "The reply was blocked because it appeared to reveal internal instructions. "
    "Please ask a question about your documents."
)

_CITATION_GROUP = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_FILE_NAME = re.compile(r"[\w\-]+(?:\.[\w\-]+)*\.(?:pdf|txt|docx?|xlsx?|pptx?|csv|md)\b", re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?:)?//[^)]*\)", re.IGNORECASE)

EXCERPT_CHARS = 350


@dataclass
class SourceReference:
    """A passage that was really retrieved and shown to the LLM."""

    label: str  # "S1", "S2", ... as used in citations
    source: str  # file name
    page_start: int
    page_end: int
    chunk_index: int
    total_chunks: int
    score: float
    excerpt: str
    text: str  # full passage
    chunk_id: str
    cited: bool = False  # True if the answer cites it
    injection_signs: List[str] = field(default_factory=list)

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return f"page {self.page_start}"
        return f"pages {self.page_start}-{self.page_end}"


@dataclass
class Answer:
    question: str
    status: str
    text: str  # the answer, or the message to show for non-answer statuses
    sources: List[SourceReference] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retrieval: Optional[RetrievalResponse] = None
    model: str = ""
    llm_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def is_answer(self) -> bool:
        return self.status == ANSWERED

    @property
    def cited_sources(self) -> List[SourceReference]:
        return [s for s in self.sources if s.cited]


# --------------------------------------------------------------------------
# Pure helper functions (easy to test)
# --------------------------------------------------------------------------


def make_excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " …"


def is_not_in_documents_reply(text: str) -> bool:
    return text.strip().strip("\"'`*. !").upper().startswith(NOT_IN_DOCUMENTS_TOKEN)


def rewrite_citations(text: str, valid_labels: Set[str]) -> Tuple[str, Set[str], List[str]]:
    """Keep only citations that point to real passages.

    Returns (new_text, cited_labels, invalid_labels_removed).
    """
    cited: Set[str] = set()
    invalid: List[str] = []

    def replace(match: "re.Match[str]") -> str:
        kept = []
        for number in re.findall(r"S(\d+)", match.group(0), re.IGNORECASE):
            label = f"S{int(number)}"
            if label in valid_labels:
                cited.add(label)
                kept.append(f"[{label}]")
            else:
                invalid.append(label)
        return "".join(kept)

    new_text = _CITATION_GROUP.sub(replace, text)
    new_text = re.sub(r"[ \t]{2,}", " ", new_text)
    new_text = re.sub(r" +([.,;:!?])", r"\1", new_text).strip()
    return new_text, cited, invalid


def _normalise_number(token: str) -> str:
    return token.replace(",", "")


def numbers_not_in_sources(answer_text: str, source_text: str) -> List[str]:
    """Numbers that appear in the answer but nowhere in the sources (hallucination hint)."""
    without_citations = _CITATION_GROUP.sub(" ", answer_text)
    # ignore list numbering such as "1." at the start of a line
    without_lists = re.sub(r"(?m)^\s*\d+[.)]\s+", "", without_citations)
    known = {_normalise_number(n) for n in _NUMBER.findall(source_text)}
    missing: List[str] = []
    for token in _NUMBER.findall(without_lists):
        normalised = _normalise_number(token)
        if normalised not in known and token not in missing:
            missing.append(token)
    return missing


def file_names_not_in_sources(answer_text: str, source_names: Sequence[str]) -> List[str]:
    """File names mentioned in the answer that were not among the retrieved sources."""
    known = {name.lower() for name in source_names}
    unknown: List[str] = []
    for match in _FILE_NAME.findall(answer_text):
        if match.lower() not in known and match not in unknown:
            unknown.append(match)
    return unknown


def sanitize_for_display(text: str) -> str:
    """Remove markdown images and web links (a classic way to leak data from an answer)."""
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    return text.strip()


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class AnswerEngine:
    def __init__(
        self,
        retriever: Retriever,
        llm: Optional[LLMClient] = None,
        llm_unavailable_reason: str = "",
        require_citations: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.llm_unavailable_reason = llm_unavailable_reason or (
            "No LLM is configured, so answers cannot be generated. "
            "Add an API key to your .env file (see the README)."
        )
        self.require_citations = require_citations

    @staticmethod
    def _to_reference(label: str, result: SearchResult) -> SourceReference:
        return SourceReference(
            label=label,
            source=result.source,
            page_start=result.page_start,
            page_end=result.page_end,
            chunk_index=result.chunk_index,
            total_chunks=result.total_chunks,
            score=result.score,
            excerpt=make_excerpt(result.text),
            text=result.text,
            chunk_id=result.chunk_id,
            injection_signs=find_injection_signs(result.text),
        )

    def answer(
        self,
        question: str,
        doc_hashes: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Answer:
        started = time.time()

        def finish(answer: Answer) -> Answer:
            answer.total_seconds = time.time() - started
            return answer

        # ---- 1. retrieval (its failures are reported as RETRIEVAL_ERROR)
        try:
            retrieval = self.retriever.retrieve(
                question, top_k=top_k, min_score=min_score, doc_hashes=doc_hashes
            )
        except (EmbeddingError, VectorStoreError) as error:
            return finish(Answer(question, RETRIEVAL_ERROR, str(error)))

        asked = retrieval.query or (question if isinstance(question, str) else "")
        if retrieval.status == STATUS_EMPTY_QUERY:
            return finish(Answer(asked, EMPTY_QUERY, retrieval.message, retrieval=retrieval))
        if retrieval.status == STATUS_QUERY_TOO_LONG:
            return finish(Answer(asked, QUERY_TOO_LONG, retrieval.message, retrieval=retrieval))
        if retrieval.status == STATUS_NO_DOCUMENTS:
            return finish(Answer(asked, NO_DOCUMENTS, retrieval.message, retrieval=retrieval))
        if retrieval.status == STATUS_NO_RELEVANT or not retrieval.results:
            # Safe fallback: the LLM is never asked, so it cannot invent an answer.
            return finish(
                Answer(asked, NOT_IN_DOCUMENTS, FALLBACK_MESSAGE, retrieval=retrieval)
            )
        if retrieval.status != STATUS_OK:  # unknown status: be safe
            return finish(Answer(asked, NOT_IN_DOCUMENTS, FALLBACK_MESSAGE, retrieval=retrieval))

        # ---- 2. prompt
        bundle = build_messages(asked, retrieval.results)
        references = [
            self._to_reference(passage_id(i), result) for i, result in enumerate(bundle.passages)
        ]
        warnings: List[str] = []
        for reference in references:
            if reference.injection_signs:
                warnings.append(
                    f"Passage [{reference.label}] from {reference.source} contains text that "
                    f"looks like an instruction to an AI ({'; '.join(reference.injection_signs)}). "
                    "It was treated as plain document content and not followed."
                )
        if bundle.dropped:
            warnings.append(
                f"{bundle.dropped} lower-ranked passage(s) were left out to keep the request small."
            )

        # ---- 3. LLM (its failures are reported as LLM_UNAVAILABLE / LLM_ERROR)
        if self.llm is None:
            return finish(
                Answer(asked, LLM_UNAVAILABLE, self.llm_unavailable_reason,
                       sources=references, warnings=warnings, retrieval=retrieval)
            )
        llm_started = time.time()
        try:
            raw = self.llm.generate(bundle.messages)
        except LLMError as error:
            return finish(
                Answer(asked, LLM_ERROR, str(error), sources=references, warnings=warnings,
                       retrieval=retrieval, model=getattr(self.llm, "model", ""))
            )
        except Exception as error:  # a misbehaving client must not crash the app
            logger.warning("Unexpected LLM client failure: %s", type(error).__name__)
            return finish(
                Answer(asked, LLM_ERROR,
                       "The LLM call failed unexpectedly. Please try again.",
                       sources=references, warnings=warnings, retrieval=retrieval,
                       model=getattr(self.llm, "model", ""))
            )
        llm_seconds = time.time() - llm_started
        model = getattr(self.llm, "model", "")

        def result(status: str, text: str, sources: List[SourceReference]) -> Answer:
            return finish(
                Answer(asked, status, text, sources=sources, warnings=warnings,
                       retrieval=retrieval, model=model, llm_seconds=llm_seconds)
            )

        # ---- 4. check the reply before showing it
        if CANARY in raw:
            return result(BLOCKED, BLOCKED_MESSAGE, [])
        if is_not_in_documents_reply(raw):
            return result(NOT_IN_DOCUMENTS, FALLBACK_MESSAGE, references)

        text, cited, invalid = rewrite_citations(raw, {r.label for r in references})
        if invalid:
            warnings.append(
                "The model referred to source(s) that were not provided "
                f"({', '.join(sorted(set(invalid)))}); those references were removed."
            )
        if self.require_citations and not cited:
            return result(UNVERIFIED, UNVERIFIED_MESSAGE, references)

        text = sanitize_for_display(text)
        if not text:
            return result(UNVERIFIED, UNVERIFIED_MESSAGE, references)

        source_text = "\n".join(r.text for r in references)
        unsupported = numbers_not_in_sources(text, source_text)
        if unsupported:
            warnings.append(
                "These numbers in the answer were not found in the sources: "
                f"{', '.join(unsupported)}. Please check them against the documents."
            )
        invented_files = file_names_not_in_sources(text, [r.source for r in references])
        if invented_files:
            warnings.append(
                "The answer mentions file name(s) that were not among the retrieved sources: "
                f"{', '.join(invented_files)}. Rely only on the sources listed below the answer."
            )
        for reference in references:
            reference.cited = reference.label in cited
        return result(ANSWERED, text, references)
