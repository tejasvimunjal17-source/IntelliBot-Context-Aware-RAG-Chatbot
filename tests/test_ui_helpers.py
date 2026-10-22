"""Tests for core/ui_helpers.py. All REAL: pure Python, no Streamlit, no mocks."""

import re
import unittest

from core.answering import (
    ANSWERED, BLOCKED, EMPTY_QUERY, LLM_ERROR, LLM_UNAVAILABLE, NO_DOCUMENTS, NOT_IN_DOCUMENTS,
    QUERY_TOO_LONG, RETRIEVAL_ERROR, UNVERIFIED, Answer,
)
from core.memory import NEEDS_CLARIFICATION, SourceNote
from core.retrieval import STATUS_OK, RetrievalResponse
from core.ui_helpers import (
    CSS, EXAMPLE_QUESTIONS, NEEDS_SETUP, NO_DOCS, NOT_INDEXED, READY,
    describe_problem, history_limit_text, readiness, readiness_message, safe_details,
    setup_steps, unique_sources, unknown_problem,
)

NON_ANSWER_STATUSES = [EMPTY_QUERY, QUERY_TOO_LONG, NO_DOCUMENTS, NOT_IN_DOCUMENTS, NEEDS_CLARIFICATION,
                       UNVERIFIED, BLOCKED, RETRIEVAL_ERROR, LLM_UNAVAILABLE, LLM_ERROR]
SECRET = "sk-test-FAKE-0123456789abcdef"


class ProblemTests(unittest.TestCase):
    def test_every_non_answer_status_has_a_title_a_message_and_a_next_step(self):
        for status in NON_ANSWER_STATUSES:
            with self.subTest(status=status):
                problem = describe_problem(status, "some safe text")
                self.assertIsNotNone(problem)
                self.assertIn(problem.kind, {"error", "warning", "info"})
                self.assertTrue(problem.title.strip())
                self.assertEqual(problem.message, "some safe text")
                self.assertGreater(len(problem.next_step), 15)

    def test_a_normal_answer_is_not_a_problem(self):
        self.assertIsNone(describe_problem(ANSWERED, "text"))

    def test_unknown_statuses_are_not_invented(self):
        self.assertIsNone(describe_problem("something_else", "text"))

    def test_failures_are_errors_but_missing_configuration_is_only_a_warning(self):
        self.assertEqual(describe_problem(RETRIEVAL_ERROR, "x").kind, "error")
        self.assertEqual(describe_problem(LLM_ERROR, "x").kind, "error")
        self.assertEqual(describe_problem(LLM_UNAVAILABLE, "x").kind, "warning")

    def test_next_steps_tell_the_user_where_to_go(self):
        self.assertIn("Documents tab", describe_problem(NO_DOCUMENTS, "x").next_step)
        self.assertIn("OPENAI_API_KEY", describe_problem(LLM_UNAVAILABLE, "x").next_step)
        self.assertIn("Search", describe_problem(LLM_UNAVAILABLE, "x").next_step)
        self.assertIn("try again", describe_problem(LLM_ERROR, "x").next_step)
        self.assertIn("ChromaDB", describe_problem(RETRIEVAL_ERROR, "x").next_step)

    def test_retrieval_and_llm_problems_have_different_titles(self):
        self.assertNotEqual(describe_problem(RETRIEVAL_ERROR, "x").title, describe_problem(LLM_ERROR, "x").title)

    def test_the_generic_fallback_never_shows_internals(self):
        problem = unknown_problem()
        self.assertEqual(problem.kind, "error")
        self.assertNotIn("Traceback", problem.message + problem.next_step)

    def test_next_steps_contain_no_secrets_or_stack_traces(self):
        for status in NON_ANSWER_STATUSES:
            text = describe_problem(status, "x").next_step
            self.assertNotIn("Traceback", text)
            self.assertNotRegex(text, r"sk-[A-Za-z0-9]|hf_[A-Za-z0-9]")


class ReadinessTests(unittest.TestCase):
    def test_states(self):
        self.assertEqual(readiness(0, 0, True, True), NO_DOCS)
        self.assertEqual(readiness(3, 0, True, True), NOT_INDEXED)
        self.assertEqual(readiness(3, 2, True, True), NOT_INDEXED)
        self.assertEqual(readiness(3, 3, True, True), READY)
        self.assertEqual(readiness(3, 3, False, True), NEEDS_SETUP)  # ChromaDB missing
        self.assertEqual(readiness(3, 0, True, False), NEEDS_SETUP)  # embedding library missing
        self.assertEqual(readiness(0, 0, False, False), NO_DOCS)  # documents come first

    def test_messages_exist_for_every_state_that_needs_one(self):
        self.assertIsNone(readiness_message(READY))
        for state in (NO_DOCS, NOT_INDEXED, NEEDS_SETUP):
            kind, text = readiness_message(state)
            self.assertIn(kind, {"info", "warning"})
            self.assertGreater(len(text), 30)
        self.assertIn("not indexed", readiness_message(NOT_INDEXED)[1])
        self.assertIn("Documents", readiness_message(NO_DOCS)[1])

    def test_checklist(self):
        steps = setup_steps(0, 0, True, True, False)
        self.assertEqual([s.label for s in steps], ["Add documents", "Index documents", "Answer model (API key)"])
        self.assertEqual([s.mark for s in steps], ["⬜", "⬜", "➖"])  # the key is optional
        steps = setup_steps(2, 2, True, True, True)
        self.assertEqual([s.mark for s in steps], ["✅", "✅", "✅"])
        steps = setup_steps(2, 1, True, True, True)
        self.assertEqual([s.done for s in steps], [True, False, True])

    def test_indexing_is_not_done_without_a_store_even_if_counts_match(self):
        self.assertFalse(setup_steps(2, 2, False, True, True)[1].done)

    def test_every_step_has_a_hint(self):
        for step in setup_steps(0, 0, True, True, False):
            self.assertTrue(step.hint)


class TextHelperTests(unittest.TestCase):
    def test_history_limit_text(self):
        self.assertIn("last 4 questions and answers", history_limit_text(4, 0))
        self.assertIn("Remembered now: 2", history_limit_text(4, 2))
        self.assertIn("last 1 question and answer", history_limit_text(1, 1))
        self.assertIn("this chat only", history_limit_text(4, 0))
        for off in (0, -1):
            self.assertIn("memory is off", history_limit_text(off, 0))

    def test_remembered_count_never_exceeds_the_limit(self):
        self.assertIn("Remembered now: 3", history_limit_text(3, 9))

    def test_unique_sources_keeps_order_and_removes_repeats(self):
        notes = [SourceNote("S1", "a.pdf", "page 1", True, ""), SourceNote("S2", "b.txt", "page 1", True, ""),
                 SourceNote("S3", "a.pdf", "page 1", True, ""), SourceNote("S4", "a.pdf", "page 2", True, "")]
        self.assertEqual(unique_sources(notes), ["a.pdf (page 1)", "b.txt (page 1)", "a.pdf (page 2)"])
        self.assertEqual(unique_sources([]), [])

    def test_example_questions_exist_and_are_short(self):
        self.assertGreaterEqual(len(EXAMPLE_QUESTIONS), 3)
        self.assertTrue(all(len(q) < 100 for q in EXAMPLE_QUESTIONS))


class SafeDetailsTests(unittest.TestCase):
    def answer(self):
        retrieval = RetrievalResponse(query="q", status=STATUS_OK, best_score=0.8123, top_k=4, min_score=0.2, results=[])
        return Answer("q", ANSWERED, f"answer text {SECRET}", model="gpt-x", llm_seconds=1.234,
                      total_seconds=2.345, retrieval=retrieval)

    def test_contains_useful_debugging_facts(self):
        details = safe_details(self.answer(), "standalone", "llm")
        self.assertEqual(details["Status"], "answered")
        self.assertEqual(details["Answer model"], "gpt-x")
        self.assertEqual(details["Search result"], "ok")
        self.assertEqual(details["Best similarity"], "0.81")
        self.assertEqual(details["Follow-up handling"], "llm")
        self.assertEqual(details["Total time"], "2.35 s")

    def test_never_contains_answer_text_questions_or_secrets(self):
        everything = " ".join(f"{k} {v}" for k, v in safe_details(self.answer(), "my private question", "llm").items())
        self.assertNotIn(SECRET, everything)
        self.assertNotIn("answer text", everything)
        self.assertNotIn("private question", everything)
        self.assertNotIn("Traceback", everything)

    def test_minimal_answers_work(self):
        self.assertEqual(safe_details(Answer("q", "empty_query", "x")), {"Status": "empty_query"})


class CssTests(unittest.TestCase):
    def test_the_stylesheet_is_static_and_self_contained(self):
        self.assertNotRegex(CSS, r"(?i)@import|url\(|<script|javascript:|http://|https://")

    def test_it_has_the_mobile_rules(self):
        self.assertIn("@media (max-width: 640px)", CSS)
        self.assertIn("min-height: 2.75rem", CSS)  # comfortable tap targets
        self.assertIn("overflow-wrap", CSS)  # long words/paths wrap instead of scrolling sideways
        self.assertIn("pre-wrap", CSS)

    def test_braces_are_balanced(self):
        self.assertEqual(CSS.count("{"), CSS.count("}"))

    def test_no_animation(self):
        self.assertNotRegex(CSS, r"(?i)@keyframes|animation|transition")


if __name__ == "__main__":
    unittest.main()
