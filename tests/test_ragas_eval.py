"""Tests for core/ragas_eval.py.

REAL : sample conversion, score extraction from named columns, and every "RAGAS is not
       available" path (the missing-key and missing-package paths are real code paths).
MOCK : a FAKE `ragas` module (below) stands in for the real library so the score-reading
       code can run. It is written from the author's knowledge of RAGAS v0.2+ and is NOT
       RAGAS. These tests do NOT prove that real RAGAS works with this adapter, and no real
       RAGAS score has ever been produced.
"""

import math
import sys
import types
import unittest
from unittest import mock

from core.ragas_eval import (
    RagasScores, RagasUnavailableError, build_samples, extract_scores, run_ragas,
)

KEY = "sk-test-FAKE-RAGAS-0123456789abcdef"
RECORD = {"case_id": "S1", "question": "How many days?", "answer": "24 days [S1].",
          "contexts": ["Employees receive 24 days."], "ground_truth": "24 days."}


class BuildSamplesTests(unittest.TestCase):
    def test_conversion_uses_the_ragas_v02_field_names(self):
        self.assertEqual(build_samples([RECORD]), [{
            "user_input": "How many days?", "response": "24 days [S1].",
            "retrieved_contexts": ["Employees receive 24 days."], "reference": "24 days."}])

    def test_incomplete_records_are_skipped_not_patched_up(self):
        incomplete = [
            {**RECORD, "answer": ""}, {**RECORD, "contexts": []}, {**RECORD, "contexts": ["  "]},
            {**RECORD, "ground_truth": ""}, {**RECORD, "question": " "}, {},
        ]
        self.assertEqual(build_samples(incomplete), [])
        self.assertEqual(len(build_samples(incomplete + [RECORD])), 1)

    def test_empty_context_pieces_are_dropped(self):
        sample = build_samples([{**RECORD, "contexts": ["a", "", "  ", "b"]}])[0]
        self.assertEqual(sample["retrieved_contexts"], ["a", "b"])

    def test_values_are_plain_strings(self):
        sample = build_samples([{**RECORD, "contexts": [123]}])[0]
        self.assertEqual(sample["retrieved_contexts"], ["123"])


class ExtractScoresTests(unittest.TestCase):
    def test_metrics_are_found_by_name_and_averaged(self):
        columns = {"user_input": ["q", "q"], "faithfulness": [1.0, 0.5], "answer_relevancy": [0.8, 0.6],
                   "llm_context_precision_with_reference": [1.0, 0.0]}
        scores = extract_scores(columns, 2, "0.2.9")
        self.assertAlmostEqual(scores.faithfulness, 0.75)
        self.assertAlmostEqual(scores.answer_relevancy, 0.7)
        self.assertAlmostEqual(scores.context_precision, 0.5)
        self.assertEqual((scores.n_samples, scores.library_version), (2, "0.2.9"))

    def test_other_common_column_names(self):
        columns = {"Faithfulness": [0.2], "response_relevancy": [0.4], "context_precision": [0.6]}
        scores = extract_scores(columns, 1)
        self.assertEqual((scores.faithfulness, scores.answer_relevancy, scores.context_precision), (0.2, 0.4, 0.6))

    def test_a_missing_metric_is_none_never_zero(self):
        scores = extract_scores({"faithfulness": [0.9]}, 1)
        self.assertEqual(scores.faithfulness, 0.9)
        self.assertIsNone(scores.answer_relevancy)
        self.assertIsNone(scores.context_precision)

    def test_nan_and_non_numbers_are_ignored(self):
        scores = extract_scores({"faithfulness": [float("nan"), 1.0, None, "x", True]}, 5)
        self.assertEqual(scores.faithfulness, 1.0)

    def test_a_column_with_only_nan_is_none(self):
        self.assertIsNone(extract_scores({"faithfulness": [math.nan, math.nan]}, 2).faithfulness)

    def test_no_columns_at_all(self):
        scores = extract_scores({}, 0)
        self.assertEqual((scores.faithfulness, scores.answer_relevancy, scores.context_precision), (None, None, None))


class UnavailablePathTests(unittest.TestCase):
    """REAL code paths: no stand-in is involved."""

    def test_no_answered_questions_to_score(self):
        with self.assertRaises(RagasUnavailableError) as ctx:
            run_ragas([], environ={"OPENAI_API_KEY": KEY})
        self.assertIn("no answered questions", str(ctx.exception))

    def test_a_missing_openai_key_is_explained_before_anything_else(self):
        for environ in ({}, {"OPENAI_API_KEY": ""}):
            with self.subTest(environ=environ):
                with self.assertRaises(RagasUnavailableError) as ctx:
                    run_ragas([RECORD], environ=environ)
                self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_a_missing_ragas_package_gives_the_install_command(self):
        # Setting the module to None makes the import fail on any machine, with or without ragas.
        with mock.patch.dict(sys.modules, {"ragas": None, "ragas.metrics": None}):
            with self.assertRaises(RagasUnavailableError) as ctx:
                run_ragas([RECORD], environ={"OPENAI_API_KEY": KEY})
        self.assertIn("pip install -r requirements-eval.txt", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))


# ---------------------------------------------------------------- MOCK: a pretend `ragas` package
class FakeFrame:
    def __init__(self, data):
        self._data = data
        self.columns = list(data)

    def __getitem__(self, name):
        column = self._data[name]
        return types.SimpleNamespace(tolist=lambda: list(column))


def fake_ragas(frame=None, error=None):
    """MOCK of the parts of `ragas` that the adapter uses."""
    module, metrics = types.ModuleType("ragas"), types.ModuleType("ragas.metrics")
    module.calls = {}

    class EvaluationDataset:
        @staticmethod
        def from_list(samples):
            module.calls["samples"] = samples
            return "dataset"

    def evaluate(dataset, metrics):
        module.calls["metrics"] = [type(m).__name__ for m in metrics]
        if error is not None:
            raise error
        return types.SimpleNamespace(to_pandas=lambda: frame)

    module.EvaluationDataset, module.evaluate = EvaluationDataset, evaluate
    for name in ("Faithfulness", "ResponseRelevancy", "LLMContextPrecisionWithReference"):
        setattr(metrics, name, type(name, (), {}))
    module.metrics = metrics
    return module, {"ragas": module, "ragas.metrics": metrics}


class MockRagasTests(unittest.TestCase):
    ENV = {"OPENAI_API_KEY": KEY}

    def test_scores_are_read_from_the_result_and_the_three_metrics_are_requested(self):
        frame = FakeFrame({"faithfulness": [1.0, 0.8], "answer_relevancy": [0.9, 0.7],
                           "llm_context_precision_with_reference": [0.5, 0.5]})
        module, modules = fake_ragas(frame)
        with mock.patch.dict(sys.modules, modules):
            scores = run_ragas([RECORD, {**RECORD, "case_id": "S2"}], environ=self.ENV)
        self.assertIsInstance(scores, RagasScores)
        self.assertEqual((scores.n_samples, round(scores.faithfulness, 2), round(scores.answer_relevancy, 2),
                          scores.context_precision), (2, 0.9, 0.8, 0.5))
        self.assertEqual(module.calls["metrics"], ["Faithfulness", "ResponseRelevancy", "LLMContextPrecisionWithReference"])
        self.assertEqual(module.calls["samples"][0]["user_input"], "How many days?")

    def test_only_usable_samples_are_sent_and_counted(self):
        module, modules = fake_ragas(FakeFrame({"faithfulness": [1.0]}))
        with mock.patch.dict(sys.modules, modules):
            scores = run_ragas([RECORD, {**RECORD, "answer": ""}], environ=self.ENV)
        self.assertEqual((len(module.calls["samples"]), scores.n_samples), (1, 1))

    def test_a_metric_ragas_did_not_return_stays_none(self):
        _, modules = fake_ragas(FakeFrame({"faithfulness": [1.0]}))
        with mock.patch.dict(sys.modules, modules):
            scores = run_ragas([RECORD], environ=self.ENV)
        self.assertIsNone(scores.answer_relevancy)
        self.assertIsNone(scores.context_precision)

    def test_any_failure_inside_ragas_becomes_a_friendly_message_without_internals_or_secrets(self):
        _, modules = fake_ragas(error=RuntimeError(f"HTTP 401 for key {KEY} Traceback (most recent call last)"))
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(RagasUnavailableError) as ctx:
                run_ragas([RECORD], environ=self.ENV)
        message = str(ctx.exception)
        self.assertIn("RuntimeError", message)  # only the exception TYPE
        self.assertNotIn(KEY, message)
        self.assertNotIn("Traceback", message)
        self.assertNotIn("401", message)

    def test_a_result_without_to_pandas_is_handled(self):
        module, modules = fake_ragas()
        module.evaluate = lambda dataset, metrics: object()
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(RagasUnavailableError):
                run_ragas([RECORD], environ=self.ENV)


if __name__ == "__main__":
    unittest.main()
