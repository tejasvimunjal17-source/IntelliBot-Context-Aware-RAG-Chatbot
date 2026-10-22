"""Tests for core/evaluation.py and scripts/run_evaluation.py.

REAL    : dataset checks against the real sample documents; validation, normalisation,
          percentile and report-rendering logic (pure code, no stand-ins).
MOCK    : retrieval/answer metrics run with a controlled embedder, mock ChromaDB
          (tests/fake_chroma.py), a stub conversation engine or a scripted LLM, so the
          expected numbers can be computed by hand. They test the evaluation CODE.
          They say NOTHING about IntelliBot's real retrieval quality, speed or RAGAS scores,
          which have never been measured (no real embedding model, ChromaDB or LLM here).
"""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.answering import ANSWERED, BLOCKED, NOT_IN_DOCUMENTS, UNVERIFIED, Answer, SourceReference
from core.chunking import ChunkingConfig
from core.config import PROJECT_ROOT, SAMPLE_DOCS_DIR
from core.conversation import ChatReply
from core.document_loader import load_path
from core.evaluation import (
    DEFAULT_EVAL_PATH, KIND_ATTACK, KIND_FOLLOW_UP, KIND_SUPPORTED, KIND_UNSUPPORTED, KINDS,
    EvalCase, EvaluationDataError, EvaluationRun, EvaluationUnavailableError, RetrievalEvalSummary,
    contains, execute_evaluation, load_eval_cases, normalize, render_not_run_report, render_report,
    run_answer_eval, run_retrieval_eval, validate_cases, _percentile,
)
from core.followup import REWRITE_CANARY
from core.memory import ChatSession
from core.prompting import CANARY
from core.ragas_eval import RagasFailure, RagasScores, RagasUnavailableError
from core.retrieval import Retriever
from core.vector_store import ChromaVectorStore
from tests import fake_chroma
from tests.helpers import FakeEmbedder, make_document
from tests.llm_helpers import ScriptedLLM

MODEL = "test/eval-model"


def sample_texts():
    return {p.name: load_path(p).document.text for p in sorted(SAMPLE_DOCS_DIR.iterdir())}


# ====================================================================== the shipped dataset (REAL)
class ShippedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_eval_cases()

    def test_the_dataset_matches_the_real_sample_documents(self):
        self.assertEqual(validate_cases(self.cases, sample_texts()), [])

    def test_it_covers_every_kind_of_question(self):
        counts = {k: sum(c.kind == k for c in self.cases) for k in KINDS}
        self.assertEqual(counts, {KIND_SUPPORTED: 14, KIND_UNSUPPORTED: 6, KIND_FOLLOW_UP: 2, KIND_ATTACK: 2})

    def test_every_sample_document_is_used_by_at_least_one_question(self):
        used = {s for c in self.cases for s in c.expected_sources}
        self.assertEqual(used, set(sample_texts()))

    def test_ids_are_unique_and_questions_are_not_empty(self):
        ids = [c.id for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(c.question.strip() for c in self.cases))

    def test_every_expected_fact_is_a_literal_phrase_of_its_document(self):
        texts = sample_texts()
        for case in self.cases:
            for fact in case.expected_facts:
                with self.subTest(case=case.id, fact=fact):
                    self.assertTrue(any(contains(texts[s], fact) for s in case.expected_sources))

    def test_unsupported_questions_really_are_not_answered_by_the_documents(self):
        """Safety net for the dataset itself: the 'trap' questions' key words must not be in the documents."""
        corpus = normalize(" ".join(sample_texts().values()))
        for case in self.cases:
            if case.kind == KIND_UNSUPPORTED:
                for word in ("salary", "france", "insurance", "mobile app", "football", "employees work"):
                    if word in normalize(case.question):
                        with self.subTest(case=case.id, word=word):
                            self.assertNotIn(word, corpus)

    def test_follow_up_cases_have_a_previous_question(self):
        for case in self.cases:
            if case.kind == KIND_FOLLOW_UP:
                self.assertTrue(case.previous_question.strip())


# ====================================================================== validation + loading (REAL)
def good_case(**changes):
    base = dict(id="X1", kind=KIND_SUPPORTED, question="How many leave days?",
                expected_sources=("a.txt",), expected_facts=("24 days",), key_terms=("24",),
                ground_truth="It is 24 days.")
    base.update(changes)
    return EvalCase(**base)


class ValidationTests(unittest.TestCase):
    DOCS = {"a.txt": "Employees get 24 days of leave.", "b.txt": "Other text."}

    def problems(self, *cases):
        return validate_cases(list(cases), self.DOCS)

    def test_a_good_case_has_no_problems(self):
        self.assertEqual(self.problems(good_case()), [])

    def test_each_kind_of_mistake_is_reported(self):
        bad = {
            "unknown source": good_case(expected_sources=("zzz.pdf",)),
            "fact not in source": good_case(expected_facts=("99 days",)),
            "fact only in another document": good_case(expected_facts=("Other text",)),
            "key term not in ground truth": good_case(key_terms=("77",)),
            "missing ground truth": good_case(ground_truth=""),
            "missing facts": good_case(expected_facts=()),
            "unknown kind": good_case(kind="weird"),
            "empty question": good_case(question="  "),
            "too long question": good_case(question="x" * 1001),
            "follow-up without previous question": good_case(kind=KIND_FOLLOW_UP),
            "unsupported with facts": good_case(kind=KIND_UNSUPPORTED, expected_sources=(), key_terms=(), ground_truth=""),
            "attack with sources": good_case(kind=KIND_ATTACK, expected_facts=(), key_terms=()),
        }
        for label, case in bad.items():
            with self.subTest(label):
                self.assertTrue(self.problems(case), label)

    def test_duplicate_ids_are_reported(self):
        self.assertTrue(any("duplicated" in p for p in self.problems(good_case(), good_case())))

    def test_problem_messages_name_the_case(self):
        self.assertTrue(self.problems(good_case(id="Q7", expected_facts=("nope",)))[0].startswith("[Q7]"))


class LoadingTests(unittest.TestCase):
    def write(self, content):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        path = Path(folder) / "cases.json"
        path.write_text(content, encoding="utf-8")
        return path

    def test_missing_file(self):
        with self.assertRaises(EvaluationDataError):
            load_eval_cases("/nonexistent/cases.json")

    def test_invalid_json(self):
        with self.assertRaises(EvaluationDataError):
            load_eval_cases(self.write("{not json"))

    def test_no_cases(self):
        for content in ('{"cases": []}', '{"cases": "x"}', "[]", "{}"):
            with self.subTest(content=content):
                with self.assertRaises(EvaluationDataError):
                    load_eval_cases(self.write(content))

    def test_malformed_case(self):
        with self.assertRaises(EvaluationDataError):
            load_eval_cases(self.write(json.dumps({"cases": [{"id": "a"}]})))

    def test_a_minimal_valid_file_loads_with_defaults(self):
        cases = load_eval_cases(self.write(json.dumps({"cases": [{"id": "U1", "kind": "unsupported", "question": "q?"}]})))
        self.assertEqual((cases[0].expected_sources, cases[0].previous_question), ((), ""))


class HelperTests(unittest.TestCase):
    def test_normalize_and_contains(self):
        self.assertEqual(normalize("  A\n  b\tC "), "a b c")
        self.assertTrue(contains("The  Probation\nperiod is 6 months.", "probation period is 6 months"))
        self.assertFalse(contains("24 days", "25 days"))
        self.assertEqual(normalize(None), "")  # type: ignore[arg-type]

    def test_percentile(self):
        self.assertEqual(_percentile([], 0.95), 0.0)
        self.assertEqual(_percentile([5.0], 0.95), 5.0)
        self.assertEqual(_percentile([1, 2, 3, 4, 100], 0.95), 100)
        self.assertEqual(_percentile([3, 1, 2], 0.0), 1)


# ====================================================================== retrieval metrics (MOCK services)
AX, AY, AZ = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


class ControlledEmbedder:
    """MOCK embedder with hand-chosen vectors: exact text -> vector (anything else -> Z)."""

    def __init__(self, table):
        self.model_name, self.table = MODEL, table

    def embed_documents(self, texts):
        return [list(self.table.get(t, AZ)) for t in texts]

    def embed_query(self, text):
        return list(self.table.get(text, AZ))


class RetrievalMetricTests(unittest.TestCase):
    TEXT_A = "Employees get 24 days of leave and 12 days of sick leave."
    TEXT_B = "The Business plan costs USD 39 per agent."

    def setUp(self):
        ctx = fake_chroma.installed()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.doc_a = make_document("a.txt", [self.TEXT_A], content_hash="aa" * 32)
        self.doc_b = make_document("b.txt", [self.TEXT_B], content_hash="bb" * 32)
        self.table = {self.TEXT_A: AX, self.TEXT_B: AY,
                      "Q-leave": AX,            # right document
                      "Q-price": AY,            # right document
                      "Q-wrong": AY,            # supported question that lands on the WRONG document
                      "Q-partial": AX,          # right document, but only one of two facts is in the text
                      "Q-trap-far": AZ,         # unrelated -> rejected at retrieval
                      "Q-trap-near": AX,        # unrelated question that still looks similar -> NOT rejected
                      "warm up": AX}
        self.embedder = ControlledEmbedder(self.table)
        self.store = ChromaVectorStore(Path(self.tmp) / "db", MODEL)
        from core.chunking import chunk_documents
        from core.indexing import index_chunks
        index_chunks(chunk_documents([self.doc_a, self.doc_b]), self.store, self.embedder)
        self.hashes = [self.doc_a.content_hash, self.doc_b.content_hash]
        self.retriever = Retriever(self.store, self.embedder, top_k=1, min_score=0.5)

    def sup(self, id, q, src, facts):
        return EvalCase(id, KIND_SUPPORTED, q, (src,), tuple(facts), ("x",), "gt")

    def cases(self):
        return [
            self.sup("S1", "Q-leave", "a.txt", ["24 days of leave"]),
            self.sup("S2", "Q-price", "b.txt", ["USD 39"]),
            self.sup("S3", "Q-wrong", "a.txt", ["24 days of leave"]),
            self.sup("S4", "Q-partial", "a.txt", ["24 days of leave", "99 days of holiday"]),
            EvalCase("U1", KIND_UNSUPPORTED, "Q-trap-far"),
            EvalCase("U2", KIND_UNSUPPORTED, "Q-trap-near"),
            EvalCase("A1", KIND_ATTACK, "Q-trap-far"),
            EvalCase("F1", KIND_FOLLOW_UP, "Q-leave", ("a.txt",), ("24 days",), ("24",), "gt", "prev"),
        ]

    def clock(self, step):
        ticks = iter(i * step for i in range(1000))
        return lambda: next(ticks)

    def run_eval(self, step=0.1):
        return run_retrieval_eval(self.cases(), self.retriever, self.hashes, clock=self.clock(step))

    def test_rates_are_computed_exactly(self):
        s = self.run_eval()
        self.assertEqual(s.n_supported, 4)
        # S1 hit, S2 hit, S3 lands on the wrong document (miss), S4 right document
        self.assertEqual([r.source_hit for r in s.rows if r.kind == KIND_SUPPORTED], [True, True, False, True])

    def test_source_hit_rate_value(self):
        self.assertAlmostEqual(self.run_eval().source_hit_rate, 0.75)

    def test_fact_recall_counts_partial_matches(self):
        s = self.run_eval()
        fractions = [r.fact_fraction for r in s.rows if r.kind == KIND_SUPPORTED]
        self.assertEqual(fractions, [1.0, 1.0, 0.0, 0.5])
        self.assertAlmostEqual(s.fact_recall, 0.625)
        self.assertAlmostEqual(s.all_facts_rate, 0.5)

    def test_unsupported_and_attack_questions_are_rejected_only_if_nothing_passes_the_threshold(self):
        s = self.run_eval()
        self.assertEqual(s.n_refuse, 3)
        self.assertEqual([r.rejected for r in s.rows if r.kind in (KIND_UNSUPPORTED, KIND_ATTACK)], [True, False, True])
        self.assertAlmostEqual(s.rejection_rate, 2 / 3)

    def test_follow_up_cases_are_skipped_because_they_need_an_llm(self):
        rows = self.run_eval().rows
        self.assertNotIn("F1", [r.case_id for r in rows])
        self.assertEqual(len(rows), 7)

    def test_the_first_query_is_timed_separately_and_excluded_from_the_statistics(self):
        s = self.run_eval(step=0.1)
        # warm-up uses ticks 0 and 0.1 -> 0.1 s; each of 7 queries then takes exactly 0.1 s
        self.assertAlmostEqual(s.cold_seconds, 0.1)
        self.assertTrue(all(abs(r.seconds - 0.1) < 1e-9 for r in s.rows))
        self.assertAlmostEqual(s.latency_mean, 0.1)
        self.assertAlmostEqual(s.latency_max, 0.1)

    def test_the_two_second_requirement_is_checked_against_the_slowest_query(self):
        self.assertTrue(self.run_eval(step=0.5).latency_ok)
        self.assertTrue(self.run_eval(step=1.99).latency_ok)
        self.assertFalse(self.run_eval(step=2.0).latency_ok)  # not strictly below 2 s
        self.assertFalse(self.run_eval(step=3.0).latency_ok)

    def test_settings_are_recorded(self):
        s = self.run_eval()
        self.assertEqual((s.top_k, s.min_score), (1, 0.5))

    def test_row_details(self):
        row = self.run_eval().rows[0]
        self.assertEqual((row.case_id, row.sources), ("S1", ["a.txt"]))
        self.assertAlmostEqual(row.best_score, 1.0, places=3)

    def test_an_empty_case_list_gives_zero_rates_not_errors(self):
        s = run_retrieval_eval([], self.retriever, self.hashes, clock=self.clock(0.1))
        self.assertEqual((s.n_supported, s.source_hit_rate, s.rejection_rate, s.latency_max), (0, 0.0, 0.0, 0.0))


# ====================================================================== answer metrics (MOCK engine)
def source(label, name, text="passage"):
    return SourceReference(label, name, 1, 1, 0, 1, 0.9, text, text, f"{name}-0", cited=True)


class StubEngine:
    """MOCK conversation engine: returns prepared Answers by question and records the calls."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []  # calls: (session OBJECT, question); objects are kept alive

    def ask(self, session, question, doc_hashes=None, **kwargs):
        self.calls.append((session, question))  # keeping the object avoids id() reuse after garbage collection
        session.add_exchange(question, "x", ANSWERED)  # like the real engine, remember the exchange
        answer = self.answers.get(question, Answer(question, NOT_IN_DOCUMENTS, "not found"))
        return ChatReply(answer=answer, original_question=question, standalone_question=question)


class AnswerMetricTests(unittest.TestCase):
    def cases(self):
        return [
            EvalCase("S1", KIND_SUPPORTED, "q-good", ("a.txt",), ("f",), ("24",), "It is 24 days."),
            EvalCase("S2", KIND_SUPPORTED, "q-no-term", ("a.txt",), ("f",), ("24",), "It is 24 days."),
            EvalCase("S3", KIND_SUPPORTED, "q-wrong-file", ("a.txt",), ("f",), ("24",), "It is 24 days."),
            EvalCase("S4", KIND_SUPPORTED, "q-refused", ("a.txt",), ("f",), ("24",), "It is 24 days."),
            EvalCase("U1", KIND_UNSUPPORTED, "q-trap-ok"),
            EvalCase("U2", KIND_UNSUPPORTED, "q-trap-hallucinated"),
            EvalCase("A1", KIND_ATTACK, "q-attack-leak"),
            EvalCase("F1", KIND_FOLLOW_UP, "q-follow", ("a.txt",), ("f",), ("39",), "It is 39.", "q-prev"),
        ]

    def engine(self):
        a = lambda status, text, *srcs, q="": Answer(q, status, text, sources=list(srcs))
        return StubEngine({
            "q-good": a(ANSWERED, "It is 24 days [S1].", source("S1", "a.txt", "ctx good")),
            "q-no-term": a(ANSWERED, "It is many days [S1].", source("S1", "a.txt")),
            "q-wrong-file": a(ANSWERED, "It is 24 days [S1].", source("S1", "b.txt")),
            "q-refused": a(NOT_IN_DOCUMENTS, "not found"),
            "q-trap-ok": a(NOT_IN_DOCUMENTS, "not found"),
            "q-trap-hallucinated": a(ANSWERED, "It is 5 [S1].", source("S1", "a.txt")),
            "q-attack-leak": a(ANSWERED, f"My rules: {CANARY} [S1]", source("S1", "a.txt")),
            "q-follow": a(ANSWERED, "It is 39 [S1].", source("S1", "a.txt", "ctx follow")),
        })

    def run_eval(self):
        engine = self.engine()
        return run_answer_eval(self.cases(), engine, ["h"]), engine

    def test_answerable_metrics(self):
        s, _ = self.run_eval()
        self.assertEqual(s.n_answerable, 5)  # 4 supported + 1 follow-up
        self.assertAlmostEqual(s.answered_rate, 4 / 5)             # S4 refused
        self.assertAlmostEqual(s.key_terms_rate, 3 / 5)            # S1, S3, F1 contain their key term
        self.assertAlmostEqual(s.cited_expected_rate, 3 / 5)       # S1, S2, F1 cite a.txt (S3 cites b.txt)

    def test_refusal_and_hallucination_metrics(self):
        s, _ = self.run_eval()
        self.assertEqual(s.n_refuse, 3)
        self.assertAlmostEqual(s.refusal_rate, 1 / 3)  # only U1 refused
        self.assertEqual(s.hallucinations, 2)          # U2 and A1 got cited answers

    def test_leaked_prompt_markers_are_counted(self):
        s, _ = self.run_eval()
        self.assertEqual(s.leaks, 1)
        self.assertTrue(next(r for r in s.rows if r.case_id == "A1").leaked)
        self.assertFalse(next(r for r in s.rows if r.case_id == "S1").leaked)

    def test_the_rewrite_prompt_marker_counts_as_a_leak_too(self):
        engine = StubEngine({"q": Answer("q", ANSWERED, f"{REWRITE_CANARY} [S1]", sources=[source("S1", "a.txt")])})
        s = run_answer_eval([EvalCase("A9", KIND_ATTACK, "q")], engine, ["h"])
        self.assertEqual(s.leaks, 1)

    def test_a_follow_up_asks_the_previous_question_first_in_the_same_fresh_chat(self):
        s, engine = self.run_eval()
        follow = [c for c in engine.calls if c[1] in ("q-prev", "q-follow")]
        self.assertEqual([c[1] for c in follow], ["q-prev", "q-follow"])
        self.assertIs(follow[0][0], follow[1][0])  # the same chat for the previous question and the follow-up
        others = [c[0] for c in engine.calls if c[1] not in ("q-prev", "q-follow")]
        self.assertEqual(len(others), 7)
        for i, first in enumerate(others):  # every other case has its OWN chat, and none is the follow-up's chat
            self.assertIsNot(first, follow[0][0])
            for second in others[i + 1:]:
                self.assertIsNot(first, second)

    def test_records_for_ragas_contain_only_answered_answerable_cases(self):
        s, _ = self.run_eval()
        ids = [r["case_id"] for r in s.records]
        self.assertEqual(ids, ["S1", "S2", "S3", "F1"])
        first = s.records[0]
        self.assertEqual((first["question"], first["ground_truth"]), ("q-good", "It is 24 days."))
        self.assertEqual(first["contexts"], ["ctx good"])

    def test_unverified_and_blocked_replies_count_as_not_answered(self):
        engine = StubEngine({"a": Answer("a", UNVERIFIED, "x"), "b": Answer("b", BLOCKED, "x")})
        s = run_answer_eval([EvalCase("U1", KIND_UNSUPPORTED, "a"), EvalCase("A1", KIND_ATTACK, "b")], engine, ["h"])
        self.assertEqual((s.refusal_rate, s.hallucinations), (1.0, 0))

    def test_no_cases(self):
        s = run_answer_eval([], StubEngine({}), ["h"])
        self.assertEqual((s.answered_rate, s.refusal_rate, s.hallucinations, s.records), (0.0, 0.0, 0, []))


# ====================================================================== full run with stand-ins (MOCK)
class ExecuteEvaluationTests(unittest.TestCase):
    def setUp(self):
        ctx = fake_chroma.installed()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = ChromaVectorStore(Path(self.tmp) / "db", MODEL)
        self.embedder = FakeEmbedder(MODEL)
        self.docs = [load_path(p).document for p in sorted(SAMPLE_DOCS_DIR.iterdir())]
        self.cases = load_eval_cases()

    def run_it(self, **kwargs):
        return execute_evaluation(self.cases, self.docs, self.embedder, self.store, min_score=0.0, **kwargs)

    def test_local_mode_with_the_shipped_dataset_and_sample_documents(self):
        run = self.run_it(mode="local")
        self.assertEqual((run.documents, run.chunks), (5, 16))
        self.assertEqual(self.store.count(), 16)
        self.assertEqual(run.retrieval.n_supported, 14)
        self.assertEqual(len(run.retrieval.rows), 22)  # the 2 follow-ups need an LLM
        self.assertIsNone(run.answers)
        self.assertIsNone(run.ragas)

    def test_a_stand_in_run_is_never_labelled_real(self):
        run = self.run_it(mode="local")
        self.assertFalse(run.real_services)
        self.assertEqual(run.embedding_model, MODEL)
        self.assertIn("STAND-IN RUN - NOT A QUALITY MEASUREMENT", render_report(run))

    def test_full_mode_needs_an_llm(self):
        with self.assertRaises(EvaluationUnavailableError):
            self.run_it(mode="full", llm=None)

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_it(mode="turbo")

    def test_full_mode_with_a_scripted_llm_runs_the_answer_evaluation(self):
        llm = ScriptedLLM(lambda messages: "NOT_IN_DOCUMENTS", model="scripted-model")
        run = self.run_it(mode="full", llm=llm)
        self.assertEqual(run.llm_model, "scripted-model")
        self.assertEqual(len(run.answers.rows), 24)
        self.assertEqual(run.answers.answered_rate, 0.0)  # a model that always refuses
        self.assertEqual(run.answers.hallucinations, 0)
        self.assertIsNone(run.ragas)

    def test_ragas_scores_from_the_runner_are_reported_as_given(self):
        scores = RagasScores(0.91, 0.82, 0.73, 12, "0.2.x")
        seen = []
        run = self.run_it(mode="full", llm=ScriptedLLM("Answer [S1]"), ragas_runner=lambda records: seen.append(records) or scores)
        self.assertIs(run.ragas, scores)
        self.assertTrue(seen)
        text = render_report(run)
        self.assertIn("| Faithfulness | 0.91 |", text)
        self.assertIn("| Answer relevancy | 0.82 |", text)
        self.assertIn("| Context precision | 0.73 |", text)

    def test_a_failing_ragas_run_is_reported_as_not_run_without_numbers(self):
        def failing(records):
            raise RagasUnavailableError("The 'ragas' package is not installed.")
        run = self.run_it(mode="full", llm=ScriptedLLM("Answer [S1]"), ragas_runner=failing)
        self.assertIsInstance(run.ragas, RagasFailure)
        text = render_report(run)
        self.assertIn("NOT RUN - RAGAS could not be used", text)
        self.assertNotIn("| Faithfulness |", text)

    def test_local_mode_never_calls_ragas(self):
        called = []
        self.run_it(mode="local", ragas_runner=lambda r: called.append(r))
        self.assertEqual(called, [])


# ====================================================================== report text (REAL, pure)
def summary(**changes):
    values = dict(rows=[], top_k=4, min_score=0.2, cold_seconds=3.21, n_supported=14, source_hit_rate=0.93,
                  fact_recall=0.9, all_facts_rate=0.86, n_refuse=8, rejection_rate=0.5, latency_mean=0.12,
                  latency_p95=0.2, latency_max=0.31)
    values.update(changes)
    return RetrievalEvalSummary(**values)


def make_run(**changes):
    values = dict(started="2026-01-01 10:00 UTC", mode="local", real_services=True, embedding_model="real-model",
                  documents=5, chunks=16, retrieval=summary())
    values.update(changes)
    return EvaluationRun(**values)


class ReportTests(unittest.TestCase):
    def test_a_real_run_has_no_warning_banner(self):
        text = render_report(make_run())
        self.assertNotIn("STAND-IN", text)
        self.assertIn("**Real services (embedding model + ChromaDB):** yes", text)

    def test_a_stand_in_run_has_the_banner_and_says_no(self):
        text = render_report(make_run(real_services=False))
        self.assertTrue(text.index("STAND-IN RUN") < text.index("## 1."))
        self.assertIn("NO (stand-ins)", text)
        self.assertIn("say nothing about the real model", text)

    def test_numbers_come_from_the_run(self):
        text = render_report(make_run())
        for expected in ("| Right document among the retrieved passages | 93% |", "| Mean | 0.120 |",
                         "| Slowest | 0.310 |", "| First query (includes loading the model, not counted below) | 3.21 |",
                         "top-K 4, minimum similarity 0.20"):
            self.assertIn(expected, text)

    def test_the_latency_verdict(self):
        self.assertIn("**YES**", render_report(make_run()))
        slow = render_report(make_run(retrieval=summary(latency_max=2.5)))
        self.assertIn("**NO**", slow)
        self.assertNotIn("**YES**", slow)

    def test_sections_that_were_not_run_say_so_and_contain_no_numbers(self):
        text = render_report(make_run(mode="local"))
        answers = text.split("## 2.")[1].split("## 3.")[0]
        ragas = text.split("## 3.")[1]
        self.assertIn("**NOT RUN**", answers)
        self.assertIn("**NOT RUN.**", ragas)
        self.assertNotRegex(answers + ragas, r"\d+%|\| \d\.\d\d \|")

    def test_retrieval_section_not_run(self):
        self.assertIn("**NOT RUN.**", render_report(make_run(retrieval=None)).split("## 1.")[1].split("## 2.")[0])

    def test_hallucinations_are_listed(self):
        from core.evaluation import AnswerEvalSummary, AnswerRow
        rows = [AnswerRow("U9", KIND_UNSUPPORTED, "What is the CEO salary?", ANSWERED, "x", ["a"], [])]
        answers = AnswerEvalSummary(rows=rows, n_refuse=1, hallucinations=1)
        text = render_report(make_run(mode="full", answers=answers, llm_model="m"))
        self.assertIn("**1**", text)
        self.assertIn("`U9` What is the CEO salary?", text)

    def test_missing_scores_are_shown_as_na_never_as_zero(self):
        run = make_run(mode="full", ragas=RagasScores(None, 0.5, None, 3))
        text = render_report(run)
        self.assertIn("| Faithfulness | n/a |", text)
        self.assertIn("| Context precision | n/a |", text)

    def test_the_not_run_report(self):
        text = render_not_run_report("no internet")
        self.assertIn("STATUS: NOT RUN", text)
        self.assertIn("No evaluation scores exist yet", text)
        self.assertIn("no internet", text)
        self.assertIn("python scripts/run_evaluation.py --mode local", text)
        self.assertIn("python scripts/run_evaluation.py --mode full", text)
        self.assertNotRegex(text, r"\d+%|faithfulness \|")

    def test_the_report_in_the_repository_is_never_a_stand_in_report(self):
        text = (PROJECT_ROOT / "docs" / "EVALUATION_REPORT.md").read_text(encoding="utf-8")
        self.assertNotIn("STAND-IN RUN", text)
        is_template = "STATUS: NOT RUN" in text
        is_real = "**Real services (embedding model + ChromaDB):** yes" in text
        self.assertTrue(is_template or is_real, "must be the NOT-RUN template or a report from real services")


# ====================================================================== the command-line script
def load_script():
    spec = importlib.util.spec_from_file_location("run_evaluation", PROJECT_ROOT / "scripts" / "run_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = Path(self.tmp) / "report.md"
        self.script = load_script()

    def test_missing_services_stop_the_run_without_writing_a_report(self):
        """REAL code path. Missing packages are forced so the result is the same on every machine."""
        with mock.patch.dict(sys.modules, {"chromadb": None, "chromadb.config": None}), \
             mock.patch("sys.stderr") as stderr:
            code = self.script.main(["--mode", "local", "--out", str(self.out)])
        self.assertEqual(code, 2)
        self.assertFalse(self.out.exists())
        printed = "".join(str(c.args[0]) for c in stderr.write.call_args_list)
        self.assertIn("Cannot run the evaluation", printed)
        self.assertIn("Nothing was written", printed)

    def test_an_existing_report_is_not_overwritten_when_the_run_fails(self):
        self.out.write_text("PREVIOUS REAL REPORT", encoding="utf-8")
        with mock.patch.dict(sys.modules, {"chromadb": None, "chromadb.config": None}), mock.patch("sys.stderr"):
            self.assertEqual(self.script.main(["--out", str(self.out)]), 2)
        self.assertEqual(self.out.read_text(encoding="utf-8"), "PREVIOUS REAL REPORT")

    def test_full_mode_without_an_api_key_stops_cleanly(self):
        def factory(settings, mode, work_dir):  # MOCK services except the LLM, which is the real (key-less) one
            from core.llm import create_llm_client
            store = ChromaVectorStore(Path(work_dir) / "idx", settings.embedding_model)
            return FakeEmbedder(settings.embedding_model), store, create_llm_client(settings), False
        with fake_chroma.installed(), mock.patch("sys.stderr"), \
             mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "LLM_PROVIDER": "openai"}):
            code = self.script.main(["--mode", "full", "--out", str(self.out)], service_factory=factory)
        self.assertEqual(code, 2)
        self.assertFalse(self.out.exists())

    def test_with_stand_in_services_a_report_is_written_and_labelled_as_stand_in(self):
        def factory(settings, mode, work_dir):
            store = ChromaVectorStore(Path(work_dir) / "idx", settings.embedding_model)
            return FakeEmbedder(settings.embedding_model), store, None, False
        with fake_chroma.installed(), mock.patch("builtins.print"):
            code = self.script.main(["--mode", "local", "--out", str(self.out)], service_factory=factory)
        self.assertEqual(code, 0)
        text = self.out.read_text(encoding="utf-8")
        self.assertIn("STAND-IN RUN - NOT A QUALITY MEASUREMENT", text)
        self.assertIn("## 1. Retrieval", text)

    def test_the_work_folder_is_removed_afterwards(self):
        created = []
        def factory(settings, mode, work_dir):
            created.append(work_dir)
            store = ChromaVectorStore(Path(work_dir) / "idx", settings.embedding_model)
            return FakeEmbedder(settings.embedding_model), store, None, False
        with fake_chroma.installed(), mock.patch("builtins.print"):
            self.script.main(["--out", str(self.out)], service_factory=factory)
        self.assertTrue(created)
        self.assertFalse(Path(created[0]).exists())

    def test_the_default_mode_is_local_and_the_default_output_is_the_docs_report(self):
        self.assertEqual(self.script.DEFAULT_OUT, PROJECT_ROOT / "docs" / "EVALUATION_REPORT.md")

    def test_the_shipped_dataset_path_exists(self):
        self.assertTrue(DEFAULT_EVAL_PATH.exists())


if __name__ == "__main__":
    unittest.main()
