"""UI helpers for the Streamlit app (Day 6).

Everything here is plain Python with NO Streamlit import, so it can be tested for
real and the UI code in ``app.py`` stays thin. It contains:

* ``describe_problem``  - friendly title / message / next step for every non-answer status;
* ``setup_steps`` / ``readiness`` - the "what do I do next?" checklist;
* ``history_limit_text`` - plain-language description of the chat-memory limit;
* ``safe_details``      - debugging facts that are safe to show (no secrets, no tracebacks);
* ``unique_sources``    - de-duplicated "file (page)" labels for the sources line;
* ``CSS``               - the one static stylesheet (mobile-friendly spacing, tap targets, wrapping).

Nothing user-supplied or secret ever goes into the CSS or into these messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answering import (
    ANSWERED,
    BLOCKED,
    EMPTY_QUERY,
    LLM_ERROR,
    LLM_UNAVAILABLE,
    NO_DOCUMENTS,
    NOT_IN_DOCUMENTS,
    QUERY_TOO_LONG,
    RETRIEVAL_ERROR,
    UNVERIFIED,
    Answer,
)
from .memory import NEEDS_CLARIFICATION

# ---------------------------------------------------------------------------
# Problems: what the user sees when an answer could not be given
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    kind: str  # "error" | "warning" | "info"
    title: str
    message: str  # already-safe text (comes from IntelliBot's own friendly messages)
    next_step: str  # what the user can do now


_NEXT_STEPS = {
    EMPTY_QUERY: "Type your question in the box below and send it.",
    QUERY_TOO_LONG: "Shorten the question and send it again.",
    NO_DOCUMENTS: (
        "Open the 📄 Documents tab, add your files (or tap “Use sample documents”) and "
        "tap “Create embeddings and index”."
    ),
    NOT_IN_DOCUMENTS: (
        "Try asking in different words, or add the document that contains the answer."
    ),
    NEEDS_CLARIFICATION: "Ask the full question again and include the topic.",
    UNVERIFIED: "Try rephrasing the question. If it keeps happening, try another answer model.",
    BLOCKED: "Ask a question about your documents.",
    RETRIEVAL_ERROR: (
        "Check that the embedding model and ChromaDB are installed, then try again. "
        "If it keeps happening, open 📄 Documents → “Manage the saved index” and index again."
    ),
    LLM_UNAVAILABLE: (
        "Add OPENAI_API_KEY (or HUGGINGFACEHUB_API_TOKEN) to your .env file or Streamlit "
        "Secrets and restart the app. Searching in the 🔎 Search tab works without a key."
    ),
    LLM_ERROR: (
        "Wait a moment and try again. If it keeps happening, check your key, the model name "
        "and your account balance."
    ),
}

_TITLES = {
    EMPTY_QUERY: ("warning", "Your question is empty"),
    QUERY_TOO_LONG: ("warning", "Your question is too long"),
    NO_DOCUMENTS: ("info", "No indexed documents yet"),
    NOT_IN_DOCUMENTS: ("warning", "Not found in your documents"),
    NEEDS_CLARIFICATION: ("info", "I need a little more information"),
    UNVERIFIED: ("warning", "I could not verify the reply"),
    BLOCKED: ("warning", "The reply was blocked"),
    RETRIEVAL_ERROR: ("error", "Searching your documents failed"),
    LLM_UNAVAILABLE: ("warning", "Written answers are not available yet"),
    LLM_ERROR: ("error", "The AI service could not answer"),
}


def describe_problem(status: str, text: str) -> Optional[Problem]:
    """Friendly view of a non-answer status. Returns None for a normal answer."""
    if status == ANSWERED or status not in _TITLES:
        return None
    kind, title = _TITLES[status]
    return Problem(kind=kind, title=title, message=text, next_step=_NEXT_STEPS[status])


def unknown_problem(text: str = "") -> Problem:
    """Fallback for an unexpected situation (never shows internals)."""
    return Problem(
        "error",
        "Something went wrong",
        text or "An unexpected problem occurred while answering.",
        "Please try again. If it keeps happening, tap Clear chat or reload the page.",
    )


# ---------------------------------------------------------------------------
# Setup checklist
# ---------------------------------------------------------------------------

READY = "ready"
NO_DOCS = "no_documents"
NOT_INDEXED = "not_indexed"
NEEDS_SETUP = "needs_setup"  # ChromaDB or the embedding library is missing


@dataclass(frozen=True)
class Step:
    label: str
    done: bool
    hint: str
    optional: bool = False

    @property
    def mark(self) -> str:
        if self.done:
            return "✅"
        return "➖" if self.optional else "⬜"


def readiness(n_documents: int, n_indexed: int, store_ok: bool, embeddings_ok: bool) -> str:
    """Where is the user in the setup? (checked in the order a beginner meets the steps)"""
    if n_documents <= 0:
        return NO_DOCS
    if not store_ok or not embeddings_ok:
        return NEEDS_SETUP
    if n_indexed < n_documents:
        return NOT_INDEXED
    return READY


def setup_steps(
    n_documents: int, n_indexed: int, store_ok: bool, embeddings_ok: bool, llm_configured: bool
) -> List[Step]:
    indexed_all = n_documents > 0 and n_indexed >= n_documents and store_ok
    return [
        Step("Add documents", n_documents > 0, "Open the 📄 Documents tab."),
        Step(
            "Index documents",
            indexed_all,
            "Tap “Create embeddings and index” in the 📄 Documents tab.",
        ),
        Step(
            "Answer model (API key)",
            llm_configured,
            "Optional for searching; needed for written answers.",
            optional=True,
        ),
    ]


_READINESS_TEXT = {
    NO_DOCS: (
        "info",
        "👋 Welcome! First add your documents: open the **📄 Documents** tab, upload PDF or TXT "
        "files (or tap “Use sample documents”), then tap “Create embeddings and index”.",
    ),
    NOT_INDEXED: (
        "warning",
        "Your documents are loaded but **not indexed yet**, so I cannot search them. "
        "Open the **📄 Documents** tab and tap “Create embeddings and index”.",
    ),
    NEEDS_SETUP: (
        "warning",
        "Chat is unavailable because ChromaDB or the embedding library is missing. "
        "Install the requirements and restart the app (see the README).",
    ),
}


def readiness_message(state: str) -> Optional[Tuple[str, str]]:
    """(kind, text) to show above the chat, or None when everything is ready."""
    return _READINESS_TEXT.get(state)


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

# Shown only while the bundled sample documents are loaded (they describe a fictional company).
EXAMPLE_QUESTIONS: Tuple[str, ...] = (
    "How many days of annual leave do employees get?",
    "How long must a password be?",
    "What does the Business plan of NovaDesk cost? (then try: And the Enterprise plan?)",
)


def history_limit_text(max_turns: int, remembered: int) -> str:
    """Plain-language description of the chat memory limit."""
    if max_turns <= 0:
        return "Chat memory is off: every question is answered on its own."
    unit = "question and answer" if max_turns == 1 else "questions and answers"
    return (
        f"I remember your last {max_turns} {unit} (this chat only). "
        f"Remembered now: {min(remembered, max_turns)}."
    )


def unique_sources(notes: Sequence[Any]) -> List[str]:
    """['file.pdf (page 1)', ...] without repeats, keeping the order of first appearance."""
    seen: List[str] = []
    for note in notes:
        label = f"{note.source} ({note.pages})"
        if label not in seen:
            seen.append(label)
    return seen


def safe_details(answer: Answer, standalone_question: str = "", method: str = "none") -> Dict[str, str]:
    """Facts useful for debugging that are safe to show: no secrets, no stack traces,
    no document text. Values are short, fixed-format strings."""
    details: Dict[str, str] = {"Status": answer.status}
    if answer.model:
        details["Answer model"] = answer.model
    retrieval = answer.retrieval
    if retrieval is not None:
        details["Search result"] = retrieval.status
        details["Passages found"] = str(len(retrieval.results))
        if retrieval.best_score is not None:
            details["Best similarity"] = f"{retrieval.best_score:.2f}"
        details["Settings"] = f"top-K {retrieval.top_k}, minimum similarity {retrieval.min_score:.2f}"
    if method and method != "none":
        details["Follow-up handling"] = method
    if answer.total_seconds:
        details["Total time"] = f"{answer.total_seconds:.2f} s"
    if answer.llm_seconds:
        details["Model time"] = f"{answer.llm_seconds:.2f} s"
    return details


# ---------------------------------------------------------------------------
# Styling: ONE static stylesheet. Never built from user input or secrets.
# ---------------------------------------------------------------------------

CSS = """
/* IntelliBot - mobile-friendly spacing, tap targets and text wrapping (static CSS) */
.block-container { padding-top: 1.2rem; padding-bottom: 5rem; max-width: 46rem; }
.stButton > button, .stFormSubmitButton > button { min-height: 2.75rem; }
[data-testid="stChatMessage"] { padding: 0.6rem 0.75rem; border-radius: 0.75rem; }
[data-testid="stText"], pre, code {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere;
    word-break: break-word;
}
[data-testid="stMarkdownContainer"] p { overflow-wrap: anywhere; }
[data-testid="stCaptionContainer"] { overflow-wrap: anywhere; }
@media (max-width: 640px) {
    .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
    h1 { font-size: 1.6rem !important; }
    h3 { font-size: 1.15rem !important; }
}
""".strip()
