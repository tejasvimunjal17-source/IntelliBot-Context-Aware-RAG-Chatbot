"""Day 6 interface tests: the real app.py run against a MOCK of Streamlit.

WHAT IS REAL   : app.py itself, the loader, chunker, indexing, retrieval logic, conversation
                 engine, memory, masking, prompt building, answer checking, ui_helpers.
WHAT IS MOCK   : Streamlit (tests/fake_streamlit.py), ChromaDB (tests/fake_chroma.py), the
                 embedder (FakeEmbedder: shared WORDS only) and the language model (TwoStepLLM /
                 ScriptedLLM: says what the test tells it to).
WHAT IS NOT PROVEN: how the page LOOKS, whether it works in a real browser or on a phone, or
                 anything about real ChromaDB / real embeddings / a real LLM. The structural
                 "mobile" checks are proxies (layout setting, full-width buttons, no wide
                 widgets, a static stylesheet), not device tests.
OPTIONAL class at the bottom uses the real streamlit.testing.AppTest and is skipped when
Streamlit is not installed (as in the build environment).
"""

import ast
import contextlib
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.config import PROJECT_ROOT, load_settings
from core.embeddings import EmbeddingUnavailableError
from core.llm import LLMError, LLMUnavailableError
from core.retrieval import Retriever
from core.vector_store import ChromaVectorStore, VectorStoreError
from tests import fake_chroma
from tests.fake_streamlit import APP_PATH, AppRunner
from tests.helpers import FakeEmbedder
from tests.llm_helpers import ScriptedLLM, TwoStepLLM

Q1 = "NovaDesk pricing plans Starter Business Enterprise per agent per month"
A1 = "The Starter plan costs USD 15 per agent per month [S1]."
REWRITE = "How much does the NovaDesk Business plan cost per agent per month?"
FOLLOW_UP = "What about the second option?"
FAQ = "NovaDesk_Product_FAQ.pdf"
KEY = "sk-test-FAKE-UI-0123456789abcdef"
ENV_KEYS = ("OPENAI_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "LLM_PROVIDER", "CHAT_HISTORY_TURNS",
            "TOP_K", "MIN_SIMILARITY", "CHUNK_SIZE", "CHUNK_OVERLAP")


class Upload:
    """MOCK of an uploaded file object."""

    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


class UiCase(unittest.TestCase):
    """Sets up the mock services. Each test gets a fresh index folder."""

    env = {}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="intellibot_ui_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, {"VECTOR_STORE_DIR": str(Path(self.tmp) / "idx"), **self.env}))
        for name in ENV_KEYS:
            if name not in self.env:
                os.environ.pop(name, None)
        stack.enter_context(fake_chroma.installed())
        import core.embeddings as embeddings
        import core.llm as llm_module
        self.real_create_llm_client = llm_module.create_llm_client  # the genuine function, before patching
        self.embedder_error = None
        self.llm = TwoStepLLM(rewrite=REWRITE, answer=A1)
        self.llm_factory = lambda settings: self.llm  # tests may replace this
        stack.enter_context(mock.patch.object(embeddings, "sentence_transformers_installed", lambda: True))
        stack.enter_context(mock.patch.object(embeddings, "SentenceTransformerEmbedder", lambda name, bs: self.make_embedder(name)))
        stack.enter_context(mock.patch.object(llm_module, "create_llm_client", lambda settings: self.llm_factory(settings)))
        self.runner = AppRunner()

    def make_embedder(self, name):
        error = self.embedder_error
        embedder = FakeEmbedder(name)
        if error is not None:
            embedder.embed_query = mock.Mock(side_effect=error)
        return embedder

    def prepare(self, runner=None):
        """Load the sample documents and index them (as a user would)."""
        runner = runner or self.runner
        runner.run(press=("Use sample documents",))
        return runner.run(press=("Create embeddings and index",))

    def index_count(self):
        settings = load_settings()
        return ChromaVectorStore(settings.vector_store_path, settings.embedding_model).count()

    @staticmethod
    def assistant_text(fake, which=-1):
        return [b for b in fake.bubbles() if b["role"] == "assistant"][which]["text"]


# ====================================================================== A. rendering + submission
class ChatRenderingTests(UiCase):
    def test_native_chat_components_and_tabs_are_used(self):
        fake = self.prepare()
        self.assertEqual(fake.all("tabs")[0].args[0], ["💬 Chat", "📄 Documents", "🔎 Search"])
        self.assertTrue(fake.all("chat_input"))
        self.assertEqual(fake.page_config["layout"], "centered")
        self.assertEqual(fake.page_config["page_title"], "IntelliBot")

    def test_the_title_and_the_setup_checklist_are_shown(self):
        fake = self.prepare()
        self.assertIn("🤖 IntelliBot", fake.texts("title"))
        strip = [t for t in fake.texts("markdown") if "Add documents" in t and "Index documents" in t and "·" in t]
        self.assertTrue(strip)
        self.assertIn("✅ Add documents", strip[0])
        self.assertIn("✅ Index documents", strip[0])

    def test_a_question_appears_as_a_user_bubble_and_the_answer_as_an_assistant_bubble(self):
        self.prepare()
        fake = self.runner.run(chat=Q1)
        bubbles = fake.bubbles()
        self.assertEqual([b["role"] for b in bubbles], ["user", "assistant"])
        self.assertEqual(bubbles[0]["text"], Q1)
        self.assertIn("The Starter plan costs USD 15", bubbles[1]["text"])
        self.assertEqual(len(fake.all("chat_message")), 2)  # exactly one bubble per message

    def test_bubbles_have_distinct_avatars(self):
        self.prepare()
        fake = self.runner.run(chat=Q1)
        avatars = [e.kwargs.get("avatar") for e in fake.all("chat_message")]
        self.assertEqual(avatars, ["🧑", "🤖"])

    def test_the_chat_input_has_a_helpful_placeholder_and_a_length_limit(self):
        fake = self.prepare()
        element = fake.all("chat_input")[0]
        self.assertIn("Ask a question about your documents", element.text)
        self.assertEqual(element.kwargs["max_chars"], 1000)
        self.assertFalse(element.kwargs["disabled"])

    def test_example_questions_are_offered_for_the_sample_documents(self):
        fake = self.prepare()
        text = " ".join(fake.texts("caption"))
        self.assertIn("Try asking", text)
        self.assertIn("annual leave", text)

    def test_example_questions_disappear_once_the_conversation_starts(self):
        self.prepare()
        fake = self.runner.run(chat=Q1)
        self.assertNotIn("Try asking", " ".join(fake.texts("caption")))


# ====================================================================== sources
class SourceAttributionTests(UiCase):
    def test_sources_are_shown_as_a_compact_line_and_an_expander_with_the_passages(self):
        self.prepare()
        fake = self.runner.run(chat=Q1)
        assistant = [b for b in fake.bubbles() if b["role"] == "assistant"][0]
        self.assertIn(f"📎 Sources: {FAQ} (page 1)", assistant["text"])
        expanders = [e for e in assistant["elements"] if e.kind == "expander" and e.text.startswith("View sources")]
        self.assertEqual(len(expanders), 1)
        card = [e.text for e in assistant["elements"] if e.kind == "markdown"]
        self.assertTrue(any(t.startswith(f"**[S1] {FAQ}**") for t in card))
        details = [e.text for e in assistant["elements"] if e.kind == "caption" and "similarity" in e.text]
        self.assertTrue(details and "page 1" in details[0] and "chunk" in details[0])

    def test_the_passage_text_shown_is_really_from_the_document(self):
        self.prepare()
        fake = self.runner.run(chat=Q1)
        passage = [e.text for e in fake.bubbles()[1]["elements"] if e.kind == "text"][0]
        self.assertIn("NovaDesk", passage)
        faq_text = next(p for p in (PROJECT_ROOT / "data" / "sample_docs").iterdir() if p.name == FAQ)
        from core.document_loader import load_path
        self.assertIn(" ".join(passage.split()), " ".join(load_path(faq_text).document.text.split()))

    def test_only_cited_sources_are_listed_on_the_sources_line(self):
        self.prepare()
        self.llm = TwoStepLLM(answer="Starter costs USD 15 [S1].")
        fake = self.runner.run(chat=Q1)
        line = [t for t in fake.texts("caption") if t.startswith("📎 Sources:")][0]
        self.assertEqual(line.count("("), 1)  # only the one cited passage

    def test_an_answer_that_cites_nothing_real_shows_no_sources(self):
        self.prepare()
        self.llm = TwoStepLLM(answer="The price is USD 15 [S9].")
        fake = self.runner.run(chat=Q1)
        text = self.assistant_text(fake)
        self.assertNotIn("📎 Sources", text)
        self.assertIn("could not verify", text)

    def test_a_fallback_never_claims_support(self):
        self.prepare()
        self.llm = TwoStepLLM(answer="NOT_IN_DOCUMENTS")
        fake = self.runner.run(chat=Q1)
        text = self.assistant_text(fake)
        self.assertNotIn("📎 Sources", text)
        self.assertIn("Not found in your documents", text)
        self.assertIn("not used as support", " ".join(e.text for e in fake.all("expander")))

    def test_earlier_exchanges_keep_their_sources(self):
        self.prepare()
        self.runner.run(chat=Q1)
        fake = self.runner.run(chat="password security multi-factor authentication phishing incident")
        first = [b for b in fake.bubbles() if b["role"] == "assistant"][0]
        self.assertIn(f"📎 Sources: {FAQ} (page 1)", first["text"])


# ====================================================================== C. loading state
class LoadingStateTests(UiCase):
    def test_a_spinner_is_shown_inside_the_assistant_bubble_and_the_answer_comes_after_it(self):
        self.prepare()
        calls_at = {}
        fake = self.runner.run(chat=Q1, on_spinner=lambda phase: calls_at.setdefault(phase, len(self.llm.answer_calls)))
        spinner = fake.all("spinner")
        self.assertEqual(len(spinner), 1)
        self.assertIn("Searching your documents", spinner[0].text)
        self.assertEqual(spinner[0].path[-1][0], "chat_message:assistant")
        # no fake progress: the model had NOT been called when the spinner appeared, and HAD when it ended
        self.assertEqual(calls_at["enter"], 0)
        self.assertEqual(calls_at["exit"], 1)
        enter, exit_ = [(e[2]) for e in fake.spinner_events if e[0] == "enter"][0], [(e[2]) for e in fake.spinner_events if e[0] == "exit"][0]
        answer_index = next(e.index for e in fake.elements if e.kind == "write" and "Starter plan costs" in e.text)
        self.assertGreaterEqual(answer_index, exit_)  # the answer is drawn only after processing ended
        self.assertEqual(enter, exit_)  # nothing was drawn while it was working

    def test_no_spinner_when_nothing_is_asked(self):
        self.prepare()
        fake = self.runner.run()
        self.assertEqual([e for e in fake.all("spinner") if "Searching your documents and" in e.text], [])


# ====================================================================== D. error states
class ErrorStateTests(UiCase):
    def test_a_missing_api_key_gives_a_friendly_message_with_a_next_step(self):
        """The REAL create_llm_client runs here (no key is configured, so it raises for real)."""
        import core.llm as llm_module
        with mock.patch.object(llm_module, "create_llm_client", self.real_create_llm_client):
            self.prepare()
            fake = self.runner.run(chat=Q1)
        text = self.assistant_text(fake)
        self.assertIn("Written answers are not available yet", text)
        self.assertIn("OPENAI_API_KEY", text)
        self.assertIn("👉", text)
        self.assertNotIn("Traceback", fake.everything())
        self.assertTrue(self.runner.state["chat"].is_empty)  # not remembered
        self.assertTrue(any("Passages found in your documents" in e.text for e in fake.all("expander")))  # search still worked

    def test_a_missing_openai_package_is_reported_the_same_friendly_way(self):
        """Real code path: a key IS configured but the 'openai' package is genuinely absent here."""
        import core.llm as llm_module
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": KEY}), \
             mock.patch.dict(sys.modules, {"openai": None}), \
             mock.patch.object(llm_module, "create_llm_client", self.real_create_llm_client):
            self.prepare()
            fake = self.runner.run(chat=Q1)
        text = self.assistant_text(fake)
        self.assertIn("pip install openai", text)
        self.assertNotIn(KEY, fake.everything())

    def test_the_banner_above_the_chat_warns_when_no_key_is_configured(self):
        fake = self.prepare()
        chat_tab = [e.text for e in fake.in_tab("💬 Chat") if e.kind == "warning"]
        self.assertTrue(any("No API key found" in t and "Search" in t for t in chat_tab))

    def test_an_llm_failure_is_shown_as_an_ai_service_error_with_a_next_step(self):
        self.prepare()
        self.llm = TwoStepLLM(answer_error=LLMError("The LLM provider rejected the API key."))
        fake = self.runner.run(chat=Q1)
        text = self.assistant_text(fake)
        self.assertIn("The AI service could not answer", text)
        self.assertIn("rejected the API key", text)
        self.assertIn("👉", text)
        self.assertNotIn("Searching your documents failed", text)  # not confused with a retrieval error
        self.assertTrue(self.runner.state["chat"].is_empty)

    def test_a_retrieval_failure_is_shown_differently_from_an_llm_failure(self):
        self.prepare()
        self.embedder_error = EmbeddingUnavailableError("The embedding model could not be loaded.")
        fake = self.runner.run(chat=Q1)
        text = self.assistant_text(fake)
        self.assertIn("Searching your documents failed", text)
        self.assertIn("embedding model could not be loaded", text)
        self.assertIn("ChromaDB", text)  # the next step
        self.assertNotIn("The AI service could not answer", text)
        self.assertEqual(self.llm.answer_calls, [])

    def test_an_unexpected_client_exception_never_shows_internals(self):
        self.prepare()
        self.llm = TwoStepLLM(answer_error=RuntimeError(f"boom {KEY} Traceback (most recent call last)"))
        fake = self.runner.run(chat=Q1)
        self.assertNotIn(KEY, fake.everything())
        self.assertNotIn("Traceback", fake.everything())
        self.assertIn("The AI service could not answer", self.assistant_text(fake))

    def test_a_vector_store_failure_while_answering(self):
        self.prepare()
        with mock.patch.object(ChromaVectorStore, "search", side_effect=VectorStoreError("The search index could not search.")):
            fake = self.runner.run(chat=Q1)
        self.assertIn("Searching your documents failed", self.assistant_text(fake))

    def test_an_empty_question_is_refused_politely(self):
        self.prepare()
        fake = self.runner.run(chat="   ")
        self.assertIn("Your question is empty", self.assistant_text(fake))

    def test_a_too_long_question_is_refused_politely(self):
        self.prepare()
        fake = self.runner.run(chat="x" * 1500)
        self.assertIn("Your question is too long", self.assistant_text(fake))
        self.assertEqual(self.llm.answer_calls, [])

    def test_technical_details_are_available_but_safe(self):
        self.prepare()
        fake = self.runner.run(chat=Q1)
        self.assertTrue(any(e.text == "🔧 Technical details" for e in fake.all("expander")))
        lines = [e.text for e in fake.bubbles()[1]["elements"] if e.kind == "text" and ": " in e.text and e.text.split(":")[0] in
                 {"Status", "Answer model", "Search result", "Passages found", "Best similarity", "Settings", "Total time"}]
        self.assertTrue(any(l.startswith("Status: answered") for l in lines))
        self.assertTrue(any(l.startswith("Search result: ok") for l in lines))


# ====================================================================== readiness / empty index
class SetupStateTests(UiCase):
    def test_before_any_document_a_welcome_message_points_to_the_documents_tab(self):
        fake = self.runner.run()
        info = [e.text for e in fake.in_tab("💬 Chat") if e.kind == "info"]
        self.assertTrue(any("Welcome" in t and "Documents" in t for t in info))
        strip = [t for t in fake.texts("markdown") if "Add documents" in t and "·" in t][0]
        self.assertIn("⬜ Add documents", strip)

    def test_loaded_but_not_indexed_documents_are_clearly_flagged(self):
        self.runner.run(press=("Use sample documents",))
        fake = self.runner.run()
        warnings = [e.text for e in fake.in_tab("💬 Chat") if e.kind == "warning"]
        self.assertTrue(any("not indexed yet" in t for t in warnings))
        strip = [t for t in fake.texts("markdown") if "Add documents" in t and "·" in t][0]
        self.assertIn("✅ Add documents", strip)
        self.assertIn("⬜ Index documents", strip)
        sidebar = " ".join(e.text for e in fake.in_sidebar())
        self.assertIn("Tap “Create embeddings and index”", sidebar)

    def test_asking_before_indexing_explains_what_to_do(self):
        self.runner.run(press=("Use sample documents",))
        fake = self.runner.run(chat=Q1)
        text = self.assistant_text(fake)
        self.assertIn("No indexed documents yet", text)
        self.assertIn("Documents tab", text)
        self.assertEqual(self.llm.answer_calls, [])

    def test_after_indexing_the_chat_is_ready_and_the_warning_is_gone(self):
        fake = self.prepare()
        chat_warnings = [e.text for e in fake.in_tab("💬 Chat") if e.kind == "warning" and "not indexed" in e.text]
        self.assertEqual(chat_warnings, [])

    def test_the_chat_input_is_disabled_when_chromadb_is_missing(self):
        with mock.patch.dict(sys.modules, {"chromadb": None, "chromadb.config": None}):
            self.runner.run(press=("Use sample documents",))
            fake = self.runner.run(chat=Q1)
        element = fake.all("chat_input")[0]
        self.assertTrue(element.kwargs["disabled"])
        self.assertEqual(fake.bubbles(), [])  # a disabled input cannot submit
        warnings = " ".join(e.text for e in fake.in_tab("💬 Chat") if e.kind == "warning")
        self.assertIn("ChromaDB", warnings)


# ====================================================================== follow-ups + memory
class FollowUpTests(UiCase):
    def test_a_follow_up_is_understood_through_the_history_and_still_searched(self):
        self.prepare()
        self.runner.run(chat=Q1)
        with mock.patch.object(Retriever, "retrieve", autospec=True, side_effect=Retriever.retrieve) as retrieve:
            fake = self.runner.run(chat=FOLLOW_UP)
        self.assertEqual(retrieve.call_args.args[1], REWRITE)  # retrieval ran on the rewritten question
        self.assertEqual(len(self.llm.rewrite_calls), 1)
        self.assertIn(Q1, self.llm.rewrite_calls[0][1]["content"])  # history was used to interpret
        self.assertIn(f"🔎 I understood your question as: {REWRITE}", self.assistant_text(fake))
        self.assertEqual([b["role"] for b in fake.bubbles()], ["user", "assistant", "user", "assistant"])
        self.assertEqual(fake.bubbles()[2]["text"], FOLLOW_UP)
        self.assertIn(f"📎 Sources: {FAQ}", self.assistant_text(fake))

    def test_earlier_answers_are_not_used_as_evidence_in_the_answer_prompt(self):
        self.prepare()
        earlier_answer = "The Starter plan costs USD 15 per agent per month [S1]."
        self.llm = TwoStepLLM(rewrite=REWRITE, answer=earlier_answer)
        self.runner.run(chat=Q1)
        self.runner.run(chat=FOLLOW_UP)
        self.assertEqual(len(self.llm.answer_calls), 2)
        final_prompt = self.llm.answer_calls[-1][1]["content"]
        self.assertNotIn(earlier_answer, final_prompt)  # the earlier answer is not in the answering prompt
        self.assertNotIn("<conversation", final_prompt)  # and neither is the chat history
        self.assertIn(REWRITE, final_prompt)  # only the standalone question + freshly retrieved passages

    def test_a_follow_up_without_context_asks_for_clarification_in_the_chat(self):
        self.prepare()
        fake = self.runner.run(chat=FOLLOW_UP)
        text = self.assistant_text(fake)
        self.assertIn("❓", text)
        self.assertIn("no earlier question in this chat", text)
        self.assertEqual((self.llm.rewrite_calls, self.llm.answer_calls), ([], []))

    def test_a_reply_that_is_only_search_results_when_there_is_no_key_still_shows_sources(self):
        self.prepare()
        self.llm_factory = lambda settings: (_ for _ in ()).throw(LLMUnavailableError("No API key was found."))
        fake = self.runner.run(chat=Q1)
        self.assertIn("Written answers are not available yet", self.assistant_text(fake))


class HistoryLimitTests(UiCase):
    env = {"CHAT_HISTORY_TURNS": "2"}

    def test_the_limit_is_explained_in_plain_language_in_the_chat_and_the_sidebar(self):
        fake = self.prepare()
        chat_caption = [e.text for e in fake.in_tab("💬 Chat") if e.kind == "caption" and "I remember" in e.text]
        side = [e.text for e in fake.in_sidebar() if e.kind == "caption" and "I remember" in e.text]
        self.assertTrue(chat_caption and "last 2 questions and answers" in chat_caption[0])
        self.assertTrue(side and "this chat only" in side[0])

    def test_only_the_last_two_exchanges_stay_in_the_conversation(self):
        self.prepare()
        for n in range(1, 5):
            self.runner.run(chat=f"{Q1} variant{n}")
        fake = self.runner.run()
        users = [b["text"] for b in fake.bubbles() if b["role"] == "user"]
        self.assertEqual([u.split()[-1] for u in users], ["variant3", "variant4"])
        self.assertIn("Remembered now: 2", " ".join(e.text for e in fake.in_sidebar() if e.kind == "caption"))


class MemoryOffTests(UiCase):
    env = {"CHAT_HISTORY_TURNS": "0"}

    def test_memory_off_is_explained_and_follow_ups_ask_for_clarification(self):
        self.prepare()
        self.runner.run(chat=Q1)
        fake = self.runner.run(chat=FOLLOW_UP)
        self.assertIn("Chat memory is off", " ".join(e.text for e in fake.in_sidebar() if e.kind == "caption"))
        self.assertIn("❓", self.assistant_text(fake))


# ====================================================================== E. rerun safety
class RerunTests(UiCase):
    def test_reruns_never_duplicate_messages_or_repeat_model_calls(self):
        self.prepare()
        first = self.runner.run(chat=Q1)
        calls = len(self.llm.answer_calls)
        for _ in range(3):  # Streamlit reruns the script on any interaction
            again = self.runner.run()
            self.assertEqual([b["role"] for b in again.bubbles()], ["user", "assistant"])
            self.assertEqual(again.bubbles()[0]["text"], first.bubbles()[0]["text"])
            self.assertIsNone(again.chat_input_returned)
        self.assertEqual(len(self.llm.answer_calls), calls)
        self.assertEqual(len(self.runner.state["chat"]), 1)

    def test_each_new_question_adds_exactly_one_exchange(self):
        self.prepare()
        self.runner.run(chat=Q1)
        fake = self.runner.run(chat=Q1 + " refund")
        self.assertEqual(len(fake.bubbles()), 4)
        self.assertEqual(len(self.runner.state["chat"]), 2)

    def test_the_input_is_the_native_chat_input_so_it_clears_itself(self):
        fake = self.prepare()
        self.assertEqual([e.kind for e in fake.elements if e.kind in ("chat_input", "text_input") and e.text.startswith("Ask a question")], ["chat_input"])


# ====================================================================== Clear chat
class ClearChatTests(UiCase):
    def asked_twice(self):
        self.prepare()
        self.runner.run(chat=Q1)
        return self.runner.run(chat=FOLLOW_UP)

    def test_clear_chat_empties_the_visible_conversation_immediately(self):
        self.asked_twice()
        fake = self.runner.run(press=("clear_chat_main",))
        self.assertEqual(fake.bubbles(), [])
        self.assertTrue(self.runner.state["chat"].is_empty)
        self.assertTrue(any("Chat cleared" in t for t in fake.texts("success")))

    def test_the_sidebar_button_does_the_same(self):
        self.asked_twice()
        fake = self.runner.run(press=("clear_chat_sidebar",))
        self.assertEqual(fake.bubbles(), [])
        self.assertTrue(any("Chat cleared" in t for t in fake.texts("success")))

    def test_clearing_keeps_the_documents_and_the_index(self):
        self.asked_twice()
        before = self.index_count()
        fake = self.runner.run(press=("clear_chat_main",))
        self.assertEqual(self.index_count(), before)
        self.assertEqual(before, 16)
        self.assertTrue(any("5" in t and "document(s) ready" in t for t in fake.texts("write")))
        answered = self.runner.run(chat=Q1)  # still searchable right away
        self.assertIn("The Starter plan costs USD 15", self.assistant_text(answered))

    def test_the_confirmation_is_shown_once_and_the_chat_stays_empty_after_a_rerun(self):
        self.asked_twice()
        self.runner.run(press=("clear_chat_main",))
        fake = self.runner.run()
        self.assertEqual(fake.bubbles(), [])
        self.assertEqual([t for t in fake.texts("success") if "Chat cleared" in t], [])

    def test_after_clearing_a_follow_up_has_no_context_any_more(self):
        self.asked_twice()
        self.runner.run(press=("clear_chat_main",))
        fake = self.runner.run(chat=FOLLOW_UP)
        self.assertIn("❓", self.assistant_text(fake))

    def test_the_button_is_disabled_when_there_is_nothing_to_clear(self):
        fake = self.prepare()
        buttons = [e for e in fake.all("button") if e.kwargs.get("key") in ("clear_chat_main", "clear_chat_sidebar")]
        self.assertEqual(len(buttons), 2)
        self.assertTrue(all(b.kwargs["disabled"] for b in buttons))
        self.runner.run(chat=Q1)
        fake = self.runner.run()
        buttons = [e for e in fake.all("button") if e.kwargs.get("key") in ("clear_chat_main", "clear_chat_sidebar")]
        self.assertFalse(any(b.kwargs["disabled"] for b in buttons))

    def test_a_cleared_error_message_also_disappears(self):
        self.prepare()
        self.llm = TwoStepLLM(answer_error=LLMError("The LLM provider is limiting requests."))
        self.runner.run(chat=Q1)
        fake = self.runner.run()
        self.assertTrue(fake.bubbles())  # the error stays visible on reruns until cleared
        fake = self.runner.run(press=("clear_chat_main",))
        self.assertEqual(fake.bubbles(), [])


# ====================================================================== session isolation
class SessionIsolationTests(UiCase):
    def test_two_visitors_never_see_each_others_conversation(self):
        alice, bob = self.runner, AppRunner()
        self.prepare(alice)
        self.prepare(bob)
        alice.run(chat="Alice private question about " + Q1)
        fake_bob = bob.run()
        self.assertEqual(fake_bob.bubbles(), [])
        self.assertNotIn("Alice private", fake_bob.everything())
        self.assertIsNot(alice.state["chat"], bob.state["chat"])

    def test_a_follow_up_in_a_new_session_cannot_use_another_sessions_history(self):
        alice, bob = self.runner, AppRunner()
        self.prepare(alice)
        self.prepare(bob)
        alice.run(chat=Q1)
        fake = bob.run(chat=FOLLOW_UP)
        self.assertIn("❓", self.assistant_text(fake))
        self.assertEqual(self.llm.rewrite_calls, [])  # nothing from Alice was used to interpret Bob's question

    def test_clearing_one_session_does_not_affect_another(self):
        alice, bob = self.runner, AppRunner()
        self.prepare(alice)
        self.prepare(bob)
        alice.run(chat=Q1)
        bob.run(chat=Q1 + " refund")
        alice.run(press=("clear_chat_main",))
        self.assertTrue(alice.state["chat"].is_empty)
        self.assertEqual(len(bob.run().bubbles()), 2)

    def test_each_session_only_searches_its_own_documents(self):
        alice, bob = self.runner, AppRunner()
        self.prepare(alice)
        bob.run(press=("Use sample documents",))  # Bob loaded the files but has not indexed them himself...
        bob.run(press=("Create embeddings and index",))
        fake = bob.run(chat=Q1)
        self.assertIn(f"📎 Sources: {FAQ}", self.assistant_text(fake))
        stranger = AppRunner()  # ...while a visitor with no documents gets nothing from the shared index
        fake = stranger.run(chat=Q1)
        self.assertEqual(fake.all("chat_input")[0].kwargs["disabled"], False)
        self.assertIn("No indexed documents yet", self.assistant_text(fake))


# ====================================================================== sensitive-key masking
class MaskingTests(UiCase):
    env = {"OPENAI_API_KEY": KEY}

    def test_a_configured_key_pasted_into_a_question_is_masked_everywhere(self):
        self.prepare()
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record.getMessage())
        logging.getLogger().addHandler(handler)
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.DEBUG)
        try:
            fake = self.runner.run(chat=f"{Q1} my key is {KEY}")
            self.runner.run(chat=FOLLOW_UP)
            later = self.runner.run()
        finally:
            logging.getLogger().removeHandler(handler)
            logging.getLogger().setLevel(old_level)
        for run in (fake, later):
            self.assertNotIn(KEY, run.everything())  # not drawn anywhere
        self.assertIn("[redacted]", fake.bubbles()[0]["text"])  # the user's bubble shows the masked text
        self.assertNotIn(KEY, repr(self.llm.answer_calls))  # not sent to the model for answering
        self.assertNotIn(KEY, repr(self.llm.rewrite_calls))  # nor for interpreting a follow-up
        self.assertNotIn(KEY, repr(self.runner.state["chat"].turns))  # not remembered
        self.assertNotIn(KEY, "\n".join(records))  # not logged

    def test_a_key_like_string_that_is_not_configured_is_masked_too(self):
        self.prepare()
        other = "sk-test-OTHER-9876543210fedcba"
        fake = self.runner.run(chat=f"{Q1} {other}")
        self.assertNotIn(other, fake.everything())
        self.assertNotIn(other, repr(self.llm.answer_calls))

    def test_a_key_echoed_by_the_model_is_masked_before_display(self):
        self.prepare()
        self.llm = TwoStepLLM(answer=f"The Starter plan costs USD 15 [S1]. (debug key {KEY})")
        fake = self.runner.run(chat=Q1)
        self.assertNotIn(KEY, fake.everything())
        self.assertIn("[redacted]", self.assistant_text(fake))

    def test_the_key_is_never_shown_in_the_sidebar_or_the_status(self):
        fake = self.prepare()
        self.assertNotIn(KEY, fake.everything())
        self.assertTrue(any("API key found" in t for t in fake.texts("success")))

    def test_the_key_never_appears_in_error_messages(self):
        self.prepare()
        self.llm = TwoStepLLM(answer_error=LLMError("The LLM provider rejected the API key."))
        fake = self.runner.run(chat=f"{Q1} {KEY}")
        self.assertNotIn(KEY, fake.everything())


# ====================================================================== Day 1-5 still works
class PreservedFeatureTests(UiCase):
    def test_the_documents_tab_keeps_upload_samples_chunk_preview_and_index_controls(self):
        fake = self.prepare()
        docs = fake.in_tab("📄 Documents")
        kinds = {e.kind for e in docs}
        self.assertIn("file_uploader", kinds)
        labels = [e.text for e in docs if e.kind == "button"]
        self.assertIn("Use sample documents", labels)
        self.assertIn("Create embeddings and index", labels)
        expanders = [e.text for e in docs if e.kind == "expander"]
        self.assertTrue(any(t.startswith("✅ ") for t in expanders))  # Day 1 document cards
        self.assertTrue(any(t.startswith("✂️ ") for t in expanders))  # Day 2 chunk previews
        self.assertIn("Manage the saved index", expanders)  # Day 3
        self.assertTrue(any("indexed" in t and "chunks saved" in t for t in fake.texts("success")))

    def test_uploading_files_still_reports_good_and_bad_files(self):
        self.prepare()
        fake = self.runner.run(uploads=[Upload("notes.txt", b"hello world"), Upload("bad.pdf", b"not a pdf"),
                                        Upload("sheet.xlsx", b"zz")])
        errors = " ".join(fake.texts("error"))
        self.assertIn("bad.pdf", errors)
        self.assertIn("sheet.xlsx", errors)
        self.assertTrue(any(t.startswith("✅ notes.txt") for t in fake.texts("expander")))

    def test_the_search_tab_still_returns_matching_passages(self):
        self.prepare()
        fake = self.runner.run(press=("Search",), text_inputs={"Type a question or keywords": Q1})
        search = fake.in_tab("🔎 Search")
        self.assertTrue(any("matching part(s) found" in e.text for e in search if e.kind == "success"))
        self.assertTrue(any(e.kind == "markdown" and FAQ in e.text for e in search))

    def test_the_sidebar_keeps_the_status_information(self):
        fake = self.prepare()
        side = " ".join(e.text for e in fake.in_sidebar())
        for expected in ("Max file size", "Supported files", "Chunks:", "Embedding model", "Answer model",
                         "Search index: ChromaDB · 16 chunks saved", "No API key yet", "How to use IntelliBot"):
            self.assertIn(expected, side)

    def test_removing_documents_from_the_index_still_works_and_then_chat_explains(self):
        self.prepare()
        self.runner.run(press=("Remove my documents from the index",))
        self.assertEqual(self.index_count(), 0)
        fake = self.runner.run(chat=Q1)
        self.assertIn("No indexed documents yet", self.assistant_text(fake))


# ====================================================================== mobile + safety structure
class MobileAndSafetyStructureTests(UiCase):
    """Structural proxies for 'works on a phone'. They are NOT real-device tests."""

    def test_no_wide_widgets_are_used(self):
        fake = self.prepare()
        self.runner.run(chat=Q1)
        fake = self.runner.last
        for kind in ("dataframe", "table", "data_editor", "pyplot", "plotly_chart"):
            self.assertEqual(fake.all(kind), [], kind)

    def test_buttons_fill_the_width_so_they_are_easy_to_tap(self):
        fake = self.prepare()
        for button in fake.all("button"):
            self.assertTrue(button.kwargs.get("use_container_width"), button.text)

    def test_the_static_stylesheet_is_injected_exactly_once(self):
        fake = self.prepare()
        styles = [e for e in fake.all("markdown") if e.text.startswith("<style>")]
        self.assertEqual(len(styles), 1)
        self.assertTrue(styles[0].kwargs["unsafe_allow_html"])
        from core.ui_helpers import CSS
        self.assertEqual(styles[0].text, f"<style>{CSS}</style>")

    def test_unsafe_html_is_never_used_for_anything_but_that_stylesheet(self):
        tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and any(k.arg == "unsafe_allow_html" for k in n.keywords)]
        self.assertEqual(len(uses), 1)
        self.assertIn("CSS", ast.dump(uses[0].args[0]))
        for path in list((PROJECT_ROOT / "core").glob("*.py")):
            self.assertNotIn("unsafe_allow_html", path.read_text(encoding="utf-8"), path.name)

    def test_user_text_with_html_is_drawn_as_plain_text_not_as_markup(self):
        self.prepare()
        fake = self.runner.run(chat=f"{Q1} <script>alert(1)</script>")
        flagged = [e for e in fake.elements if "<script>" in " ".join(str(a) for a in e.args) and e.kwargs.get("unsafe_allow_html")]
        self.assertEqual(flagged, [])


# ====================================================================== OPTIONAL: real Streamlit
@unittest.skipUnless(importlib.util.find_spec("streamlit") is not None,
                     "OPTIONAL: real streamlit is not installed here - real UI rendering UNVERIFIED")
class RealStreamlitAppTestTests(unittest.TestCase):
    """Runs app.py with Streamlit's own test runner. Never executed in the build environment."""

    def test_the_app_starts_without_errors(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(len(at.exception), 0)

    def test_the_title_is_shown(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertIn("IntelliBot", at.title[0].value)


if __name__ == "__main__":
    unittest.main()
