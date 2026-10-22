"""Tests for core/memory.py (ChatSession). All REAL: no mocks, no network."""

import os
import tempfile
import unittest

from core.answering import ANSWERED, NOT_IN_DOCUMENTS
from core.memory import (
    MAX_HISTORY_ANSWER_CHARS,
    MAX_STORED_ANSWER_CHARS,
    MAX_STORED_QUESTION_CHARS,
    NEEDS_CLARIFICATION,
    REDACTED,
    ChatSession,
    SourceNote,
    Turn,
    redact_secrets,
)

FAKE_KEY = "sk-test-FAKE-0123456789abcdef"


def add(session, n, status=ANSWERED, answer=None):
    return session.add_exchange(f"question {n}", answer or f"answer {n} [S1]", status)


class EmptySessionTests(unittest.TestCase):
    def test_a_new_session_is_empty(self):
        s = ChatSession()
        self.assertTrue(s.is_empty)
        self.assertEqual(len(s), 0)
        self.assertEqual(s.turns, [])
        self.assertEqual(s.interpretation_history(), [])
        self.assertFalse(s.has_answered_turn())
        self.assertEqual(s.last_standalone_question(), "")

    def test_clearing_an_empty_session_is_fine(self):
        s = ChatSession()
        s.clear()
        self.assertTrue(s.is_empty)


class HistoryTests(unittest.TestCase):
    def test_exchanges_are_stored_in_order_with_their_fields(self):
        s = ChatSession(max_turns=5)
        note = SourceNote("S1", "leave.pdf", "page 2", True, "Employees receive 24 days.")
        self.assertTrue(s.add_exchange("How much leave?", "24 days [S1]", ANSWERED,
                                       standalone_question="How much annual leave do employees get?",
                                       sources=[note], warnings=["w"]))
        add(s, 2)
        first, second = s.turns
        self.assertIsInstance(first, Turn)
        self.assertEqual((first.question, first.answer_text, first.status), ("How much leave?", "24 days [S1]", ANSWERED))
        self.assertEqual(first.standalone_question, "How much annual leave do employees get?")
        self.assertEqual(first.sources, (note,))
        self.assertEqual(first.warnings, ("w",))
        self.assertEqual(second.question, "question 2")
        self.assertEqual(second.standalone_question, "question 2")  # defaults to the question itself

    def test_history_is_limited_and_the_oldest_exchanges_are_dropped(self):
        s = ChatSession(max_turns=3)
        for n in range(1, 6):
            add(s, n)
        self.assertEqual(len(s), 3)
        self.assertEqual([t.question for t in s.turns], ["question 3", "question 4", "question 5"])

    def test_a_limit_of_one(self):
        s = ChatSession(max_turns=1)
        add(s, 1)
        add(s, 2)
        self.assertEqual([t.question for t in s.turns], ["question 2"])

    def test_memory_can_be_switched_off(self):
        for limit in (0, -3):
            with self.subTest(limit=limit):
                s = ChatSession(max_turns=limit)
                self.assertFalse(add(s, 1))
                self.assertTrue(s.is_empty)
                self.assertEqual(s.interpretation_history(), [])

    def test_stored_text_is_length_limited(self):
        s = ChatSession()
        s.add_exchange("q" * 5000, "a" * 20000, ANSWERED)
        turn = s.turns[0]
        self.assertLessEqual(len(turn.question), MAX_STORED_QUESTION_CHARS)
        self.assertLessEqual(len(turn.answer_text), MAX_STORED_ANSWER_CHARS)

    def test_whitespace_is_tidied(self):
        s = ChatSession()
        s.add_exchange("  a \n\n b  ", "x\ty", ANSWERED)
        self.assertEqual((s.turns[0].question, s.turns[0].answer_text), ("a b", "x y"))

    def test_turns_property_returns_a_copy(self):
        s = ChatSession()
        add(s, 1)
        snapshot = s.turns
        snapshot.clear()
        self.assertEqual(len(s), 1)

    def test_has_answered_turn_only_for_real_answers(self):
        s = ChatSession()
        add(s, 1, NOT_IN_DOCUMENTS)
        add(s, 2, NEEDS_CLARIFICATION)
        self.assertFalse(s.has_answered_turn())
        add(s, 3, ANSWERED)
        self.assertTrue(s.has_answered_turn())

    def test_last_standalone_question(self):
        s = ChatSession()
        s.add_exchange("and sick leave?", "12 days [S1]", ANSWERED, standalone_question="How many sick leave days?")
        self.assertEqual(s.last_standalone_question(), "How many sick leave days?")


class ClearTests(unittest.TestCase):
    def test_clear_removes_everything_and_the_session_stays_usable(self):
        s = ChatSession()
        add(s, 1)
        add(s, 2)
        s.clear()
        self.assertTrue(s.is_empty)
        self.assertEqual(s.interpretation_history(), [])
        self.assertFalse(s.has_answered_turn())
        add(s, 3)
        self.assertEqual([t.question for t in s.turns], ["question 3"])


class IsolationTests(unittest.TestCase):
    def test_sessions_do_not_share_history(self):
        a, b = ChatSession(), ChatSession()
        add(a, 1)
        self.assertEqual(len(a), 1)
        self.assertTrue(b.is_empty)
        self.assertEqual(b.interpretation_history(), [])

    def test_clearing_one_session_leaves_the_other_alone(self):
        a, b = ChatSession(), ChatSession()
        add(a, 1)
        add(b, 2)
        a.clear()
        self.assertTrue(a.is_empty)
        self.assertEqual([t.question for t in b.turns], ["question 2"])

    def test_no_state_is_shared_through_the_class(self):
        self.assertIsNot(ChatSession()._turns, ChatSession()._turns)
        first = ChatSession()
        add(first, 1)
        self.assertTrue(ChatSession().is_empty)  # a brand-new session never inherits history

    def test_each_session_has_its_own_limit(self):
        small, large = ChatSession(max_turns=1), ChatSession(max_turns=5)
        for n in range(3):
            add(small, n)
            add(large, n)
        self.assertEqual((len(small), len(large)), (1, 3))


class PrivacyTests(unittest.TestCase):
    def test_known_secret_formats_are_masked(self):
        samples = [
            FAKE_KEY, "hf_FAKEtoken0123456789", "AKIAFAKEFAKEFAKE1234", "ghp_" + "a" * 24,
            "Bearer abcdefghijklmnopqrstuvwxyz", "api_key = supersecretvalue123", "secret: abcdefgh1234",
        ]
        for text in samples:
            with self.subTest(text=text):
                masked = redact_secrets(f"my key is {text} ok")
                self.assertIn(REDACTED, masked)
                self.assertNotIn(text.split()[-1], masked)

    def test_exact_configured_secrets_are_masked(self):
        masked = redact_secrets("the value is Zx9-unusual-secret-VALUE here", ["Zx9-unusual-secret-VALUE"])
        self.assertEqual(masked, f"the value is {REDACTED} here")

    def test_very_short_values_are_not_masked(self):
        self.assertEqual(redact_secrets("the cat sat", ["cat"]), "the cat sat")

    def test_normal_text_is_untouched(self):
        text = "Passwords must be at least 14 characters; the secret to success is practice."
        self.assertEqual(redact_secrets(text), text)

    def test_non_text_input_gives_empty_text(self):
        self.assertEqual(redact_secrets(None), "")  # type: ignore[arg-type]

    def test_secrets_never_enter_the_stored_history(self):
        s = ChatSession(redact_values=["configured-secret-123456"])
        s.add_exchange(f"is {FAKE_KEY} valid?", "I saw configured-secret-123456 [S1]", ANSWERED,
                       standalone_question=f"valid {FAKE_KEY}",
                       sources=[SourceNote("S1", "a.pdf", "page 1", True, f"text {FAKE_KEY}")],
                       warnings=[f"warn {FAKE_KEY}"])
        stored = repr(s.turns)
        self.assertNotIn(FAKE_KEY, stored)
        self.assertNotIn("configured-secret-123456", stored)
        self.assertIn(REDACTED, s.turns[0].question)

    def test_mask_hides_configured_and_known_secrets_in_any_text(self):
        s = ChatSession(redact_values=["configured-secret-123456"])
        self.assertEqual(s.mask(f"a {FAKE_KEY} b configured-secret-123456"), f"a {REDACTED} b {REDACTED}")
        self.assertEqual(s.mask("nothing secret here"), "nothing secret here")

    def test_repr_never_shows_conversation_text(self):
        s = ChatSession()
        s.add_exchange("a private question about salaries", "private answer", ANSWERED)
        self.assertNotIn("salaries", repr(s))
        self.assertNotIn("private", repr(s))

    def test_nothing_is_written_to_disk_or_logged(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            os.chdir(folder)
            try:
                with self.assertNoLogs(level="DEBUG"):
                    s = ChatSession()
                    add(s, 1)
                    s.interpretation_history()
                    s.clear()
                self.assertEqual(os.listdir(folder), [])
            finally:
                os.chdir(original)


class InterpretationHistoryTests(unittest.TestCase):
    def test_messages_alternate_user_then_assistant_oldest_first(self):
        s = ChatSession()
        add(s, 1)
        add(s, 2)
        messages = s.interpretation_history()
        self.assertEqual([m.role for m in messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual([m.text for m in messages if m.role == "user"], ["question 1", "question 2"])

    def test_citation_markers_are_removed_from_assistant_text(self):
        s = ChatSession()
        s.add_exchange("q", "Leave is 24 days [S1][S2]. More [S3, S4].", ANSWERED)
        self.assertEqual(s.interpretation_history()[1].text, "Leave is 24 days. More.")

    def test_long_assistant_answers_are_shortened(self):
        s = ChatSession()
        s.add_exchange("q", "word " * 500, ANSWERED)
        text = s.interpretation_history()[1].text
        self.assertLessEqual(len(text), MAX_HISTORY_ANSWER_CHARS)
        self.assertTrue(text.endswith("…"))

    def test_replies_that_were_not_answers_become_neutral_markers(self):
        s = ChatSession()
        s.add_exchange("q1", "I couldn't find the answer ... INJECTED TEXT", NOT_IN_DOCUMENTS)
        s.add_exchange("q2", "Which plan do you mean? INJECTED TEXT", NEEDS_CLARIFICATION)
        replies = [m.text for m in s.interpretation_history() if m.role == "assistant"]
        self.assertEqual(replies, ["(I could not find an answer in the documents.)",
                                   "(I asked the user to clarify the question.)"])

    def test_total_size_budget_drops_the_oldest_exchanges_first(self):
        s = ChatSession(max_turns=10)
        for n in range(1, 6):
            s.add_exchange(f"question {n}", "x" * 200, ANSWERED)
        messages = s.interpretation_history(max_chars=500)
        questions = [m.text for m in messages if m.role == "user"]
        self.assertEqual(questions, ["question 4", "question 5"])  # newest kept, oldest dropped

    def test_the_newest_exchange_is_kept_even_if_it_exceeds_the_budget(self):
        s = ChatSession()
        s.add_exchange("question", "x" * 400, ANSWERED)
        self.assertEqual(len(s.interpretation_history(max_chars=10)), 2)


if __name__ == "__main__":
    unittest.main()
