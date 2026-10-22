"""Conversation memory for one chat session (Day 5).

A ``ChatSession`` remembers the last few question/answer exchanges of ONE
browser session so follow-up questions can be understood.

Privacy and isolation rules
* Every ``ChatSession`` object is independent: there is no shared or global history.
  The app keeps one per Streamlit session (``st.session_state``), so one visitor
  can never read or clear another visitor's conversation.
* History lives only in memory. Nothing is written to disk and nothing is logged.
  It disappears when the browser session ends (or when the user taps Clear chat).
* Secrets are masked before anything is stored: well-known key formats plus the
  exact API keys/tokens passed in ``redact_values``.
* The history is limited (``max_turns`` exchanges, 0 = memory off); older
  exchanges are dropped. Stored text is length-limited too.
* Earlier assistant answers are NOT evidence. They are only used to work out what
  a follow-up question refers to (see followup.py); facts always come from
  freshly retrieved document passages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from .answering import ANSWERED, NOT_IN_DOCUMENTS

NEEDS_CLARIFICATION = "needs_clarification"

MAX_STORED_ANSWER_CHARS = 4000
MAX_STORED_QUESTION_CHARS = 1000
MAX_HISTORY_ANSWER_CHARS = 600  # per assistant message used for interpretation
MAX_HISTORY_QUESTION_CHARS = 300  # per user message used for interpretation
DEFAULT_HISTORY_CHARS = 3000  # total budget for interpretation history

REDACTED = "[redacted]"
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{12,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret)\s*[:=]\s*\S{8,}"),
]
_CITATION = re.compile(r"\s*\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]", re.IGNORECASE)


@dataclass(frozen=True)
class SourceNote:
    """What the UI needs to show a past source (no secrets, short excerpt)."""

    label: str
    source: str
    pages: str
    cited: bool
    excerpt: str


@dataclass(frozen=True)
class Turn:
    """One exchange: what the user asked and what IntelliBot replied."""

    question: str
    answer_text: str
    status: str
    standalone_question: str  # the question as understood without the chat (for follow-ups)
    sources: Tuple[SourceNote, ...] = ()
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class HistoryMessage:
    role: str  # "user" or "assistant"
    text: str


def redact_secrets(text: str, extra_values: Sequence[str] = ()) -> str:
    """Mask API-key-like strings and any exact secret values given."""
    if not isinstance(text, str):
        return ""
    for value in extra_values:
        if value and len(value) >= 8:
            text = text.replace(value, REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class ChatSession:
    """History of one chat. Create one per user session; never share it."""

    def __init__(self, max_turns: int = 4, redact_values: Sequence[Optional[str]] = ()) -> None:
        self.max_turns = max(0, int(max_turns))
        self._redact_values: Tuple[str, ...] = tuple(v for v in redact_values if v)
        self._turns: List[Turn] = []  # instance attribute: never shared between sessions

    # ------------------------------------------------------------------ basics
    @property
    def turns(self) -> List[Turn]:
        """A COPY of the stored exchanges (oldest first); editing it changes nothing."""
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return not self._turns

    def clear(self) -> None:
        """Forget this session's conversation. Touches nothing else."""
        self._turns.clear()

    def mask(self, text: str) -> str:
        """Mask known secrets in ``text`` (used on a question BEFORE it is searched, sent or shown)."""
        return redact_secrets(text, self._redact_values)

    def __repr__(self) -> str:  # never prints conversation text
        return f"ChatSession(turns={len(self._turns)}, max_turns={self.max_turns})"

    # ------------------------------------------------------------------ writing
    def add_exchange(
        self,
        question: str,
        answer_text: str,
        status: str,
        standalone_question: Optional[str] = None,
        sources: Sequence[SourceNote] = (),
        warnings: Sequence[str] = (),
    ) -> bool:
        """Store one exchange (secrets masked, text limited). Returns False if memory is off."""
        if self.max_turns == 0:
            return False

        def clean(text: str, limit: int) -> str:
            return redact_secrets(_clip(text, limit), self._redact_values)

        question_text = clean(question, MAX_STORED_QUESTION_CHARS)
        turn = Turn(
            question=question_text,
            answer_text=clean(answer_text, MAX_STORED_ANSWER_CHARS),
            status=status,
            standalone_question=clean(standalone_question or question, MAX_STORED_QUESTION_CHARS)
            or question_text,
            sources=tuple(
                replace(
                    note,
                    source=clean(note.source, 200),
                    excerpt=clean(note.excerpt, 400),
                )
                for note in sources
            ),
            warnings=tuple(clean(w, 400) for w in warnings),
        )
        self._turns.append(turn)
        del self._turns[: max(0, len(self._turns) - self.max_turns)]  # drop the oldest
        return True

    # ------------------------------------------------------------------ reading
    def has_answered_turn(self) -> bool:
        """True if at least one earlier question really got an answer from the documents."""
        return any(t.status == ANSWERED for t in self._turns)

    def last_standalone_question(self) -> str:
        return self._turns[-1].standalone_question if self._turns else ""

    def interpretation_history(self, max_chars: int = DEFAULT_HISTORY_CHARS) -> List[HistoryMessage]:
        """Recent messages (oldest first) for understanding a follow-up. Size-limited.

        Assistant messages are shortened, citations removed, and replies that were
        not real answers are replaced by a neutral marker.
        """
        messages: List[HistoryMessage] = []
        used = 0
        for turn in reversed(self._turns):  # newest first, until the budget is used
            if turn.status == ANSWERED:
                reply = _clip(_CITATION.sub("", turn.answer_text), MAX_HISTORY_ANSWER_CHARS)
            elif turn.status == NOT_IN_DOCUMENTS:
                reply = "(I could not find an answer in the documents.)"
            else:
                reply = "(I asked the user to clarify the question.)"
            pair = [
                HistoryMessage("user", _clip(turn.question, MAX_HISTORY_QUESTION_CHARS)),
                HistoryMessage("assistant", reply),
            ]
            size = sum(len(m.text) for m in pair)
            if messages and used + size > max_chars:
                break
            used += size
            messages[:0] = pair
        return messages
