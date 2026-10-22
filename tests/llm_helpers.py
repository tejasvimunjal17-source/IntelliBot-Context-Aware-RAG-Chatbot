"""Test doubles for Day 4. ALL OF THESE ARE MOCKS.

ScriptedLLM    - a pretend language model that returns text chosen by the test.
StubRetriever  - a pretend retriever that returns a prepared RetrievalResponse.
Neither says anything about how a real LLM or real retrieval behaves; they let
tests check IntelliBot's own prompt building, checking and error handling.
"""

import dataclasses
from typing import Callable, List, Optional, Union

from core.retrieval import STATUS_OK, RetrievalResponse
from core.vector_store import SearchResult


class ScriptedLLM:
    """MOCK language model."""

    def __init__(self, reply: Union[str, Callable[[list], str]] = "Answer [S1]", error: Optional[Exception] = None,
                 model: str = "scripted-model"):
        self.model = model
        self._reply = reply
        self._error = error
        self.calls: List[list] = []

    def generate(self, messages):
        self.calls.append(list(messages))
        if self._error is not None:
            raise self._error
        return self._reply(messages) if callable(self._reply) else self._reply


class StubRetriever:
    """MOCK retriever returning a fixed response (or raising a fixed error)."""

    def __init__(self, response: Optional[RetrievalResponse] = None, error: Optional[Exception] = None):
        self.response = response
        self.error = error
        self.calls: List[dict] = []

    def retrieve(self, query, top_k=None, min_score=None, doc_hashes=None):
        self.calls.append({"query": query, "top_k": top_k, "min_score": min_score, "doc_hashes": doc_hashes})
        if self.error is not None:
            raise self.error
        # Like the real Retriever, report the (stripped) question that was asked.
        asked = query.strip() if isinstance(query, str) else ""
        return dataclasses.replace(self.response, query=asked) if self.response.query else self.response


def make_result(rank: int, text: str, source: str = "policy.pdf", page: int = 1, score: float = 0.8,
                chunk_index: int = 0, total_chunks: int = 3, doc_hash: str = "a" * 64) -> SearchResult:
    return SearchResult(
        rank=rank, chunk_id=f"{doc_hash[:16]}-{chunk_index:04d}", text=text, score=score, source=source,
        doc_hash=doc_hash, file_type="pdf", chunk_index=chunk_index, total_chunks=total_chunks,
        page_start=page, page_end=page, char_start=0, char_end=len(text),
    )


def ok_response(results: List[SearchResult], query: str = "a question") -> RetrievalResponse:
    return RetrievalResponse(query=query, status=STATUS_OK, results=results,
                             best_score=results[0].score if results else None)


class TwoStepLLM:
    """MOCK language model for Day 5: it recognises the follow-up REWRITE request
    (by its system prompt) and the normal ANSWER request, and replies with text the
    test chose for each. Says nothing about how a real model would behave."""

    def __init__(self, rewrite="", answer="Answer [S1]", rewrite_error=None, answer_error=None,
                 model="scripted-model"):
        self.model = model
        self._rewrite, self._answer = rewrite, answer
        self._rewrite_error, self._answer_error = rewrite_error, answer_error
        self.rewrite_calls: List[list] = []
        self.answer_calls: List[list] = []

    def generate(self, messages):
        from core.followup import REWRITE_SYSTEM_PROMPT
        is_rewrite = messages[0]["content"] == REWRITE_SYSTEM_PROMPT
        (self.rewrite_calls if is_rewrite else self.answer_calls).append(list(messages))
        error = self._rewrite_error if is_rewrite else self._answer_error
        if error is not None:
            raise error
        reply = self._rewrite if is_rewrite else self._answer
        return reply(messages) if callable(reply) else reply
