"""Day 5 integration tests: conversation memory on top of the Day 1-4 pipeline.

REAL  : loader, chunker, indexing, retrieval logic, follow-up resolution, memory,
        prompt building, answer checking.
MOCK  : ChromaDB (tests/fake_chroma.py), the embedder (FakeEmbedder: shared WORDS only)
        and the language model (TwoStepLLM: says what the test tells it to).

These show that the pieces work together and that IntelliBot's own rules hold
(grounding, isolation, limits). They do NOT show that real ChromaDB, real embeddings or a
real LLM interpret follow-up questions well. No real provider has been tested.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.answering import (
    ANSWERED,
    EMPTY_QUERY,
    FALLBACK_MESSAGE,
    LLM_ERROR,
    LLM_UNAVAILABLE,
    NOT_IN_DOCUMENTS,
    QUERY_TOO_LONG,
    RETRIEVAL_ERROR,
    AnswerEngine,
)
from core.chunking import ChunkingConfig, chunk_documents
from core.config import SAMPLE_DOCS_DIR
from core.conversation import ConversationEngine
from core.document_loader import load_path
from core.embeddings import EmbeddingUnavailableError
from core.followup import NO_HISTORY_CLARIFICATION, REWRITE_SYSTEM_PROMPT
from core.indexing import index_chunks
from core.llm import LLMError
from core.memory import NEEDS_CLARIFICATION, ChatSession
from core.retrieval import MAX_QUERY_CHARS, Retriever
from core.vector_store import ChromaVectorStore
from tests import fake_chroma
from tests.helpers import FakeEmbedder
from tests.llm_helpers import ScriptedLLM, TwoStepLLM

MODEL = "test/model-a"
FAQ = "NovaDesk_Product_FAQ.pdf"
LEAVE = "NovaTech_Leave_and_Attendance_Policy.pdf"
Q_PLANS = "NovaDesk pricing plans Starter Business Enterprise per agent per month"
A_PLANS = "The Starter plan costs USD 15 per agent per month [S1]."
REWRITE_BUSINESS = "How much does the NovaDesk Business plan cost per agent per month?"
FAKE_KEY = "sk-test-FAKE-0123456789abcdef"


class Base(unittest.TestCase):
    def setUp(self):
        ctx = fake_chroma.installed()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.tmp = tempfile.mkdtemp(prefix="intellibot_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = ChromaVectorStore(Path(self.tmp) / "db", MODEL)
        self.embedder = FakeEmbedder(MODEL)
        self.docs = {p.name: load_path(p).document for p in sorted(SAMPLE_DOCS_DIR.iterdir())}
        self.chunks = chunk_documents(self.docs.values(), ChunkingConfig(600, 100))
        index_chunks(self.chunks, self.store, self.embedder)
        self.retriever = Retriever(self.store, self.embedder, top_k=3, min_score=0.0)
        self.hash = {name: doc.content_hash for name, doc in self.docs.items()}

    def make_engine(self, llm, retriever=None):
        return ConversationEngine(AnswerEngine(retriever or self.retriever, llm))

    def spy(self):
        """Record every question that reaches the retriever (behaviour unchanged)."""
        return mock.patch.object(self.retriever, "retrieve", wraps=self.retriever.retrieve)


# ============================================================ multi-turn history
class MultiTurnTests(Base):
    def test_several_turns_are_remembered_in_order_with_real_sources(self):
        llm = TwoStepLLM(answer=A_PLANS)
        engine, session = self.make_engine(llm), ChatSession(max_turns=5)
        questions = [Q_PLANS, "password security multi-factor authentication phishing incident", Q_PLANS + " refund"]
        for q in questions:
            reply = engine.ask(session, q)
            self.assertEqual(reply.answer.status, ANSWERED)
            self.assertTrue(reply.recorded)
        self.assertEqual([t.question for t in session.turns], questions)
        for turn in session.turns:
            self.assertEqual(turn.status, ANSWERED)
            self.assertTrue(turn.sources)
            self.assertTrue(all(note.source in self.docs for note in turn.sources))
            self.assertTrue(all(note.cited for note in turn.sources))

    def test_a_follow_up_is_understood_through_the_history_and_searched_as_a_standalone_question(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer=A_PLANS)
        engine, session = self.make_engine(llm), ChatSession()
        engine.ask(session, Q_PLANS)
        with self.spy() as retrieve:
            reply = engine.ask(session, "What about the second option?")
        self.assertTrue(reply.follow_up)
        self.assertEqual(reply.method, "llm")
        self.assertEqual(reply.standalone_question, REWRITE_BUSINESS)
        self.assertTrue(reply.was_rewritten)
        # the search used the rewritten question, not the vague one
        self.assertEqual(retrieve.call_args.args[0], REWRITE_BUSINESS)
        # the rewrite step was given the earlier question and answer to work out the reference
        self.assertEqual(len(llm.rewrite_calls), 1)
        history_prompt = llm.rewrite_calls[0][1]["content"]
        self.assertIn(Q_PLANS, history_prompt)
        self.assertIn("The Starter plan costs USD 15", history_prompt)
        # the answer shown carries the user's own words and real sources
        self.assertEqual(reply.answer.question, "What about the second option?")
        self.assertEqual(reply.answer.status, ANSWERED)
        self.assertEqual(reply.answer.sources[0].source, FAQ)
        # both exchanges are remembered; the second remembers what it was understood as
        self.assertEqual(len(session), 2)
        self.assertEqual(session.turns[1].standalone_question, REWRITE_BUSINESS)

    def test_a_follow_up_without_a_language_model_still_retrieves_using_the_previous_topic(self):
        session = ChatSession()
        session.add_exchange(Q_PLANS, A_PLANS, ANSWERED, standalone_question=Q_PLANS)
        engine = self.make_engine(None)
        with self.spy() as retrieve:
            reply = engine.ask(session, "And the Business plan?")
        self.assertEqual(retrieve.call_args.args[0], f"{Q_PLANS} And the Business plan?")
        self.assertEqual(reply.method, "heuristic")
        self.assertEqual(reply.answer.status, LLM_UNAVAILABLE)  # search-only mode, no answer text
        self.assertEqual(reply.answer.sources[0].source, FAQ)
        self.assertEqual(len(session), 1)  # nothing new was stored: no answer was given

    def test_a_first_question_behaves_exactly_like_the_day_4_engine(self):
        llm = TwoStepLLM(answer=A_PLANS)
        direct = AnswerEngine(self.retriever, llm).answer(Q_PLANS)
        reply = self.make_engine(llm).ask(ChatSession(), Q_PLANS)
        for field in ("status", "text", "warnings"):
            self.assertEqual(getattr(reply.answer, field), getattr(direct, field))
        self.assertEqual([(s.label, s.source, s.chunk_id, s.cited) for s in reply.answer.sources],
                         [(s.label, s.source, s.chunk_id, s.cited) for s in direct.sources])
        self.assertEqual(llm.rewrite_calls, [])  # no extra model call for a normal question
        self.assertFalse(reply.follow_up)


# ============================================================ grounding
class GroundingTests(Base):
    def test_earlier_answers_are_never_part_of_the_answer_prompt(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer="The Business plan costs USD 39 per agent per month [S1].")
        engine, session = self.make_engine(llm), ChatSession()
        session.add_exchange(Q_PLANS, "The secret plan costs USD 777 per agent [S1].", ANSWERED, standalone_question=Q_PLANS)
        engine.ask(session, "What about the second option?")
        final_user_message = llm.answer_calls[0][1]["content"]
        self.assertNotIn("USD 777", final_user_message)
        self.assertNotIn("secret plan", final_user_message)
        self.assertIn(REWRITE_BUSINESS, final_user_message)  # only the standalone question and passages

    def test_a_fact_repeated_from_an_earlier_answer_is_not_treated_as_supported(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer="The Business plan costs USD 777 per agent per month [S1].")
        engine, session = self.make_engine(llm), ChatSession()
        session.add_exchange(Q_PLANS, "The plan costs USD 777 per agent [S1].", ANSWERED, standalone_question=Q_PLANS)
        reply = engine.ask(session, "What about the second option?")
        self.assertEqual(reply.answer.status, ANSWERED)
        self.assertTrue(any("777" in w for w in reply.answer.warnings))  # flagged: not in the documents

    def test_citations_of_a_follow_up_answer_point_to_freshly_retrieved_passages(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer="Business costs USD 39 [S1][S9].")
        engine, session = self.make_engine(llm), ChatSession()
        session.add_exchange(Q_PLANS, "answer", ANSWERED, standalone_question=Q_PLANS)
        reply = engine.ask(session, "What about the second option?")
        self.assertNotIn("S9", reply.answer.text)
        self.assertTrue(any("S9" in w for w in reply.answer.warnings))
        retrieved = {(c.source, c.chunk_id) for c in self.chunks}
        for source in reply.answer.sources:
            self.assertIn((source.source, source.chunk_id), retrieved)

    def test_a_follow_up_without_cited_support_is_not_shown(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer="As I said before, it costs USD 777.")
        session = ChatSession()
        session.add_exchange(Q_PLANS, "USD 777", ANSWERED, standalone_question=Q_PLANS)
        reply = self.make_engine(llm).ask(session, "What about the second option?")
        self.assertEqual(reply.answer.status, "unverified")
        self.assertNotIn("777", reply.answer.text)
        self.assertEqual(len(session), 1)  # the unverified reply was not remembered

    def test_when_the_follow_up_finds_nothing_relevant_the_fallback_is_given_and_the_answer_model_is_not_called(self):
        llm = TwoStepLLM(rewrite="zzzz qqqq xxxx yyyy NovaDesk", answer="INVENTED [S1]")
        strict = Retriever(self.store, self.embedder, top_k=3, min_score=0.99)
        engine, session = self.make_engine(llm, strict), ChatSession()
        session.add_exchange(Q_PLANS, A_PLANS, ANSWERED, standalone_question=Q_PLANS)
        reply = engine.ask(session, "What about the second option?")
        self.assertEqual((reply.answer.status, reply.answer.text), (NOT_IN_DOCUMENTS, FALLBACK_MESSAGE))
        self.assertEqual(len(llm.rewrite_calls), 1)
        self.assertEqual(llm.answer_calls, [])
        self.assertEqual(session.turns[-1].status, NOT_IN_DOCUMENTS)


# ============================================================ ambiguity
class AmbiguityTests(Base):
    def test_a_follow_up_with_no_earlier_context_asks_for_clarification_without_searching(self):
        llm = TwoStepLLM(answer="x [S1]")
        engine, session = self.make_engine(llm), ChatSession()
        with self.spy() as retrieve:
            reply = engine.ask(session, "What about the second option?")
        self.assertEqual(reply.answer.status, NEEDS_CLARIFICATION)
        self.assertEqual(reply.answer.text, NO_HISTORY_CLARIFICATION)
        retrieve.assert_not_called()
        self.assertEqual((llm.rewrite_calls, llm.answer_calls), ([], []))
        self.assertEqual(reply.answer.sources, [])

    def test_the_user_can_clarify_and_then_gets_a_normal_grounded_answer(self):
        llm = TwoStepLLM(answer=A_PLANS)
        engine, session = self.make_engine(llm), ChatSession()
        engine.ask(session, "What about the second option?")
        reply = engine.ask(session, Q_PLANS)
        self.assertEqual(reply.answer.status, ANSWERED)
        self.assertEqual([t.status for t in session.turns], [NEEDS_CLARIFICATION, ANSWERED])

    def test_the_model_can_ask_for_clarification_and_nothing_is_searched_or_answered(self):
        llm = TwoStepLLM(rewrite="CLARIFY: Do you mean the Starter or the Business plan?", answer="x [S1]")
        engine, session = self.make_engine(llm), ChatSession()
        session.add_exchange(Q_PLANS, A_PLANS, ANSWERED, standalone_question=Q_PLANS)
        with self.spy() as retrieve:
            reply = engine.ask(session, "What about the other one?")
        self.assertEqual((reply.answer.status, reply.answer.text),
                         (NEEDS_CLARIFICATION, "Do you mean the Starter or the Business plan?"))
        retrieve.assert_not_called()
        self.assertEqual(llm.answer_calls, [])

    def test_pointing_at_an_option_when_nothing_was_ever_answered_asks_for_clarification(self):
        engine, session = self.make_engine(TwoStepLLM()), ChatSession()
        session.add_exchange("What is the CEO salary?", FALLBACK_MESSAGE, NOT_IN_DOCUMENTS)
        reply = engine.ask(session, "What about the second option?")
        self.assertEqual(reply.answer.status, NEEDS_CLARIFICATION)


# ============================================================ clear chat
class ClearChatTests(Base):
    def test_clearing_forgets_the_conversation_so_a_follow_up_needs_clarification_again(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer=A_PLANS)
        engine, session = self.make_engine(llm), ChatSession()
        engine.ask(session, Q_PLANS)
        self.assertEqual(engine.ask(session, "What about the second option?").answer.status, ANSWERED)
        session.clear()
        self.assertTrue(session.is_empty)
        reply = engine.ask(session, "What about the second option?")
        self.assertEqual(reply.answer.status, NEEDS_CLARIFICATION)

    def test_clearing_does_not_touch_the_shared_index_or_its_documents(self):
        engine, session = self.make_engine(TwoStepLLM(answer=A_PLANS)), ChatSession()
        engine.ask(session, Q_PLANS)
        before = (self.store.count(), self.store.sources(), self.store.document_hashes())
        session.clear()
        self.assertEqual((self.store.count(), self.store.sources(), self.store.document_hashes()), before)
        self.assertEqual(engine.ask(session, Q_PLANS).answer.status, ANSWERED)  # still searchable

    def test_clearing_one_session_leaves_another_sessions_history_alone(self):
        engine = self.make_engine(TwoStepLLM(answer=A_PLANS))
        a, b = ChatSession(), ChatSession()
        engine.ask(a, Q_PLANS)
        engine.ask(b, Q_PLANS + " refund")
        a.clear()
        self.assertTrue(a.is_empty)
        self.assertEqual([t.question for t in b.turns], [Q_PLANS + " refund"])


# ============================================================ session isolation
class SessionIsolationTests(Base):
    def test_one_engine_serves_two_sessions_without_mixing_their_histories(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer=A_PLANS)
        engine = self.make_engine(llm)
        alice, bob = ChatSession(), ChatSession()
        engine.ask(alice, "Alice private question about NovaDesk pricing plans Starter Business Enterprise")
        engine.ask(bob, "Bob private question about annual leave days carry forward")
        engine.ask(alice, "What about the second option?")  # a follow-up, so the rewrite prompt is built
        engine.ask(bob, "And for sick leave?")
        prompts = [m["content"] for call in llm.rewrite_calls for m in call]
        alice_prompt = next(p for p in prompts if "Alice private question" in p)
        bob_prompt = next(p for p in prompts if "Bob private question" in p)
        self.assertNotIn("Bob private", alice_prompt)
        self.assertNotIn("Alice private", bob_prompt)
        everything_bob_sent = "\n".join(m["content"] for call in llm.answer_calls + llm.rewrite_calls for m in call
                                        if "Bob private" in m["content"])
        self.assertNotIn("Alice private", everything_bob_sent)

    def test_a_new_session_cannot_resolve_a_follow_up_using_another_sessions_history(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer=A_PLANS)
        engine = self.make_engine(llm)
        engine.ask(ChatSession(), Q_PLANS)
        stranger = ChatSession()
        reply = engine.ask(stranger, "What about the second option?")
        self.assertEqual(reply.answer.status, NEEDS_CLARIFICATION)
        self.assertEqual(llm.rewrite_calls, [])

    def test_the_engine_object_holds_no_per_user_state(self):
        engine = self.make_engine(TwoStepLLM(answer=A_PLANS))
        before = set(vars(engine))
        engine.ask(ChatSession(), Q_PLANS)
        self.assertEqual(set(vars(engine)), before)
        self.assertEqual(before, {"answer_engine"})

    def test_sessions_only_search_their_own_documents(self):
        llm = TwoStepLLM(answer="ok [S1]")
        engine = self.make_engine(llm)
        a, b = ChatSession(), ChatSession()
        reply_a = engine.ask(a, Q_PLANS, doc_hashes=[self.hash[FAQ]])
        reply_b = engine.ask(b, Q_PLANS, doc_hashes=[self.hash[LEAVE]])
        self.assertEqual({s.source for s in reply_a.answer.sources}, {FAQ})
        self.assertEqual({s.source for s in reply_b.answer.sources}, {LEAVE})

    def test_a_follow_up_is_searched_only_inside_the_sessions_documents(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer="ok [S1]")
        engine, session = self.make_engine(llm), ChatSession()
        session.add_exchange(Q_PLANS, A_PLANS, ANSWERED, standalone_question=Q_PLANS)
        reply = engine.ask(session, "What about the second option?", doc_hashes=[self.hash[LEAVE]])
        self.assertEqual({s.source for s in reply.answer.sources}, {LEAVE})


# ============================================================ limits
class HistoryLimitTests(Base):
    def test_only_the_most_recent_exchanges_are_remembered_and_sent(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer=A_PLANS)
        engine, session = self.make_engine(llm), ChatSession(max_turns=2)
        for n in range(1, 5):
            engine.ask(session, f"{Q_PLANS} variant{n}")
        self.assertEqual([t.question.split()[-1] for t in session.turns], ["variant3", "variant4"])
        engine.ask(session, "What about the second option?")
        prompt = llm.rewrite_calls[0][1]["content"]
        self.assertIn("variant3", prompt)
        self.assertIn("variant4", prompt)
        self.assertNotIn("variant1", prompt)
        self.assertNotIn("variant2", prompt)
        self.assertEqual(len(session), 2)

    def test_memory_switched_off_remembers_nothing_but_still_answers_normal_questions(self):
        engine, session = self.make_engine(TwoStepLLM(answer=A_PLANS)), ChatSession(max_turns=0)
        reply = engine.ask(session, Q_PLANS)
        self.assertEqual(reply.answer.status, ANSWERED)
        self.assertFalse(reply.recorded)
        self.assertTrue(session.is_empty)
        self.assertEqual(engine.ask(session, "What about the second option?").answer.status, NEEDS_CLARIFICATION)


# ============================================================ errors
class ErrorHandlingTests(Base):
    def test_errors_and_empty_questions_are_not_remembered(self):
        session = ChatSession()
        engine = self.make_engine(TwoStepLLM(answer_error=LLMError("The LLM provider rejected the API key.")))
        reply = engine.ask(session, Q_PLANS)
        self.assertEqual(reply.answer.status, LLM_ERROR)
        self.assertEqual(reply.answer.sources[0].source, FAQ)  # retrieval worked; only the model failed
        for question, status in (("", EMPTY_QUERY), ("   ", EMPTY_QUERY), (None, EMPTY_QUERY),
                                 ("x" * (MAX_QUERY_CHARS + 1), QUERY_TOO_LONG)):
            with self.subTest(question=str(question)[:10]):
                self.assertEqual(self.make_engine(TwoStepLLM()).ask(session, question).answer.status, status)  # type: ignore[arg-type]
        self.assertTrue(session.is_empty)

    def test_a_retrieval_failure_is_reported_as_such_and_not_remembered(self):
        session = ChatSession()
        with mock.patch.object(self.embedder, "embed_query", side_effect=EmbeddingUnavailableError("model missing")):
            reply = self.make_engine(TwoStepLLM(answer=A_PLANS)).ask(session, Q_PLANS)
        self.assertEqual((reply.answer.status, reply.answer.text), (RETRIEVAL_ERROR, "model missing"))
        self.assertTrue(session.is_empty)

    def test_a_failing_rewrite_does_not_stop_the_answer(self):
        llm = TwoStepLLM(rewrite_error=LLMError("temporary"), answer=A_PLANS)
        engine, session = self.make_engine(llm), ChatSession()
        session.add_exchange(Q_PLANS, A_PLANS, ANSWERED, standalone_question=Q_PLANS)
        reply = engine.ask(session, "And the Business plan?")
        self.assertEqual(reply.method, "heuristic")
        self.assertEqual(reply.answer.status, ANSWERED)
        self.assertTrue(any("could not use the language model" in n for n in reply.notes))

    def test_when_the_provider_is_down_for_both_steps_the_error_is_clear_and_history_unchanged(self):
        down = LLMError("The LLM provider could not be reached.")
        engine, session = self.make_engine(TwoStepLLM(rewrite_error=down, answer_error=down)), ChatSession()
        session.add_exchange(Q_PLANS, A_PLANS, ANSWERED, standalone_question=Q_PLANS)
        reply = engine.ask(session, "And the Business plan?")
        self.assertEqual((reply.answer.status, reply.answer.text), (LLM_ERROR, str(down)))
        self.assertEqual(len(session), 1)

    def test_no_api_key_search_still_works_for_normal_questions(self):
        reply = self.make_engine(None).ask(ChatSession(), Q_PLANS)
        self.assertEqual(reply.answer.status, LLM_UNAVAILABLE)
        self.assertEqual(reply.answer.sources[0].source, FAQ)


# ============================================================ privacy
class ConversationPrivacyTests(Base):
    def test_a_pasted_key_is_masked_before_it_is_searched_sent_shown_or_remembered(self):
        llm = TwoStepLLM(rewrite=REWRITE_BUSINESS, answer=A_PLANS)
        engine, session = self.make_engine(llm), ChatSession(redact_values=[FAKE_KEY])
        with self.spy() as retrieve:
            reply = engine.ask(session, f"{Q_PLANS} my key is {FAKE_KEY}")
        self.assertNotIn(FAKE_KEY, retrieve.call_args.args[0])  # not searched
        self.assertNotIn(FAKE_KEY, repr(llm.answer_calls))  # not sent to the language model
        self.assertNotIn(FAKE_KEY, reply.answer.question)  # not echoed back on screen
        self.assertIn("[redacted]", reply.answer.question)
        self.assertNotIn(FAKE_KEY, repr(session.turns))  # not remembered
        engine.ask(session, "What about the second option?")
        self.assertNotIn(FAKE_KEY, repr(llm.rewrite_calls))  # not sent again for interpretation

    def test_stored_sources_are_short_excerpts_of_real_passages_without_secrets(self):
        engine, session = self.make_engine(TwoStepLLM(answer=A_PLANS)), ChatSession()
        engine.ask(session, Q_PLANS)
        for note in session.turns[0].sources:
            self.assertLessEqual(len(note.excerpt), 400)
            self.assertIn(note.source, self.docs)


if __name__ == "__main__":
    unittest.main()
