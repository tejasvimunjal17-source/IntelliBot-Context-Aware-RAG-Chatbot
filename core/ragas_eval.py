"""RAGAS evaluation adapter (Day 7).

RAGAS is the evaluation framework named in the official brief (faithfulness, answer
relevancy, context precision). It is an OPTIONAL, heavy dependency:

* it needs the ``ragas`` package (see requirements-eval.txt),
* it needs an LLM to act as the judge. This adapter uses RAGAS's default evaluator,
  which is OpenAI, so ``OPENAI_API_KEY`` must be set,
* it needs network access.

STATUS: written from RAGAS's documented v0.2+ API (``EvaluationDataset``, ``evaluate``,
``Faithfulness``, ``ResponseRelevancy``, ``LLMContextPrecisionWithReference``). It has NEVER
been run: ragas could not be installed in the build environment. RAGAS's API changes between
versions, so a first real run may need small adjustments.

Rules kept here
* ``build_samples`` (our side of the conversion) is pure and tested.
* Anything that goes wrong becomes a ``RagasUnavailableError`` with a friendly message;
  no score is ever estimated, defaulted or invented. A metric that cannot be read is ``None``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


class RagasUnavailableError(Exception):
    """RAGAS cannot be used here. The message is safe to show to the user."""


@dataclass(frozen=True)
class RagasScores:
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    n_samples: int
    library_version: str = ""


@dataclass(frozen=True)
class RagasFailure:
    reason: str


def build_samples(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Convert IntelliBot records to RAGAS v0.2 sample dictionaries.

    Only records that have an answer, at least one retrieved context and a reference
    (ground truth) are usable; others are skipped rather than patched up.
    """
    samples: List[Dict[str, Any]] = []
    for record in records:
        answer = str(record.get("answer", "")).strip()
        contexts = [str(c) for c in record.get("contexts", []) if str(c).strip()]
        reference = str(record.get("ground_truth", "")).strip()
        question = str(record.get("question", "")).strip()
        if not (question and answer and contexts and reference):
            continue
        samples.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        )
    return samples


def _mean(values: Sequence[Any]) -> Optional[float]:
    clean = [
        float(v) for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(float(v))
    ]
    return sum(clean) / len(clean) if clean else None


def extract_scores(columns: Mapping[str, Sequence[Any]], n_samples: int, version: str = "") -> RagasScores:
    """Average each metric column by name. A column that is missing gives None (never a guess)."""

    def pick(*keywords: str) -> Optional[float]:
        for name, values in columns.items():
            lowered = str(name).lower()
            if any(k in lowered for k in keywords):
                return _mean(list(values))
        return None

    return RagasScores(
        faithfulness=pick("faithful"),
        answer_relevancy=pick("relevan"),
        context_precision=pick("precision"),
        n_samples=n_samples,
        library_version=version,
    )


def run_ragas(
    records: Sequence[Mapping[str, Any]], environ: Optional[Mapping[str, str]] = None
) -> RagasScores:
    """Score IntelliBot's answers with RAGAS. Raises RagasUnavailableError if that is not possible."""
    environ = os.environ if environ is None else environ
    samples = build_samples(records)
    if not samples:
        raise RagasUnavailableError("There were no answered questions with contexts to score.")
    if not environ.get("OPENAI_API_KEY"):
        raise RagasUnavailableError(
            "RAGAS needs an OpenAI API key (OPENAI_API_KEY) to act as the judge, but none is set."
        )
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy
    except ImportError as exc:
        raise RagasUnavailableError(
            "The 'ragas' package is not installed. Install it with: pip install -r requirements-eval.txt"
        ) from exc
    try:
        dataset = EvaluationDataset.from_list(samples)
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()],
        )
        frame = result.to_pandas()
        columns = {str(name): frame[name].tolist() for name in frame.columns}
    except Exception as exc:  # RAGAS / LLM / network problems: never show internals
        raise RagasUnavailableError(
            f"RAGAS could not finish ({type(exc).__name__}). Check the API key, the network and "
            "that the installed ragas version matches requirements-eval.txt."
        ) from exc
    try:
        from importlib.metadata import version as package_version

        library_version = package_version("ragas")
    except Exception:
        library_version = ""
    return extract_scores(columns, len(samples), library_version)
