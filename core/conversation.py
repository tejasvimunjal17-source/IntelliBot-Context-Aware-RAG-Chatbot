"""Conversation engine (Day 5): memory + follow-ups on top of the Day 4 AnswerEngine.

``ConversationEngine.ask`` does this for one question of one session:
  1. Work out the question to search for (``resolve_followup``): a normal question
     is used as it is; a follow-up is rewritten into a standalone question using
     the session history; an unclear one gets a clarifying question back instead.
  2. Run the normal grounded pipeline on that standalone question: retrieval from
     the session's own documents, grounded prompt, answer checks, source attribution.
     Earlier assistant answers are never part of that prompt, so they cannot act as evidence.
  3. Store the exchange in the session's history (only real answers, "not in
     documents" replies and clarifications; errors and empty questions are not stored).

The engine itself keeps no per-user state: everything about a user lives in the
``ChatSession`` object passed in. Two sessions therefore cannot see each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .answering import ANSWERED, NOT_IN_DOCUMENTS, Answer, AnswerEngine
from .followup import resolve_followup
from .memory import NEEDS_CLARIFICATION, ChatSession, SourceNote

RECORDED_STATUSES = {ANSWERED, NOT_IN_DOCUMENTS, NEEDS_CLARIFICATION}


@dataclass
class ChatReply:
    answer: Answer
    original_question: str
    standalone_question: str  # what was actually searched (differs for follow-ups)
    follow_up: bool = False
    method: str = "none"  # none | llm | heuristic | clarify
    notes: List[str] = field(default_factory=list)
    recorded: bool = False  # stored in the session history?

    @property
    def was_rewritten(self) -> bool:
        return self.standalone_question != self.original_question


class ConversationEngine:
    def __init__(self, answer_engine: AnswerEngine) -> None:
        self.answer_engine = answer_engine

    def ask(
        self,
        session: ChatSession,
        question: str,
        doc_hashes: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> ChatReply:
        # A key pasted into a question is masked first, so it is never searched, sent to the
        # language model, shown again on screen or remembered.
        original = session.mask(question.strip()) if isinstance(question, str) else ""
        resolution = resolve_followup(original, session, self.answer_engine.llm)

        if resolution.clarification:
            answer = Answer(question=original, status=NEEDS_CLARIFICATION, text=resolution.clarification)
        else:
            answer = self.answer_engine.answer(
                resolution.question, doc_hashes=doc_hashes, top_k=top_k, min_score=min_score
            )
            if original:
                answer.question = original  # show the user's own words, not the rewrite

        recorded = False
        if answer.status in RECORDED_STATUSES:
            notes = (
                tuple(
                    SourceNote(s.label, s.source, s.pages, True, s.excerpt)
                    for s in answer.cited_sources
                )
                if answer.status == ANSWERED
                else ()
            )
            recorded = session.add_exchange(
                question=original,
                answer_text=answer.text,
                status=answer.status,
                standalone_question=resolution.question,
                sources=notes,
                warnings=answer.warnings if answer.status == ANSWERED else (),
            )
        return ChatReply(
            answer=answer,
            original_question=original,
            standalone_question=resolution.question,
            follow_up=resolution.follow_up,
            method=resolution.method,
            notes=list(resolution.notes),
            recorded=recorded,
        )
