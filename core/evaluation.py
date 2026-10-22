"""Evaluation of IntelliBot (Day 7).

What this module measures, and what it deliberately does NOT claim
-------------------------------------------------------------------
* ``run_retrieval_eval``  - for questions whose answer IS in the sample documents: is the right
  document retrieved, and does the retrieved text contain the expected fact? Also: are questions
  whose answer is NOT in the documents rejected at retrieval? And how long does embedding +
  vector search take per query (the brief requires < 2 seconds)?  No LLM needed.
* ``run_answer_eval``     - needs an LLM. Does the bot answer supported questions with the right
  key facts and a citation of the right file? Does it refuse unsupported questions and attack
  attempts (a cited answer to an unsupported question counts as a hallucination)?
* ``render_report``       - writes the Markdown report. Every section says clearly whether it was
  RUN or NOT RUN, and whether real services or stand-ins were used. A report produced with
  stand-ins carries a warning banner. Numbers are only ever taken from actual runs.

The "key term" and "fact in context" checks are simple automatic proxies, not human grading.
The RAGAS metrics live in ragas_eval.py and are optional (they need a real LLM and the ragas
package). Nothing in this file invents or estimates a score.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

from .answering import ANSWERED, NOT_IN_DOCUMENTS, AnswerEngine
from .chunking import ChunkingConfig, chunk_documents
from .config import PROJECT_ROOT
from .conversation import ConversationEngine
from .document_loader import LoadedDocument
from .followup import REWRITE_CANARY
from .indexing import index_chunks
from .memory import ChatSession
from .prompting import CANARY
from .ragas_eval import RagasFailure, RagasScores, RagasUnavailableError
from .retrieval import STATUS_NO_RELEVANT, Retriever

DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "eval_cases.json"
LATENCY_LIMIT_SECONDS = 2.0  # from the official brief

KIND_SUPPORTED = "supported"
KIND_UNSUPPORTED = "unsupported"
KIND_FOLLOW_UP = "follow_up"
KIND_ATTACK = "attack"
KINDS = (KIND_SUPPORTED, KIND_UNSUPPORTED, KIND_FOLLOW_UP, KIND_ATTACK)
ANSWERABLE_KINDS = (KIND_SUPPORTED, KIND_FOLLOW_UP)  # the answer exists in the documents
REFUSE_KINDS = (KIND_UNSUPPORTED, KIND_ATTACK)  # the bot must not answer these


class EvaluationDataError(Exception):
    """The evaluation dataset is invalid."""


class EvaluationUnavailableError(Exception):
    """The evaluation cannot run (a required service is missing). Message is user-friendly."""


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    id: str
    kind: str
    question: str
    expected_sources: tuple = ()
    expected_facts: tuple = ()  # literal phrases that must appear in the retrieved text
    key_terms: tuple = ()  # short terms a correct ANSWER must contain (rough proxy)
    ground_truth: str = ""
    previous_question: str = ""  # follow-up cases: asked first, in the same chat


def normalize(text: str) -> str:
    """Lower-case and collapse whitespace so phrases can be compared reliably."""
    return " ".join((text or "").lower().split())


def contains(text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(text)


def load_eval_cases(path: Union[str, Path] = DEFAULT_EVAL_PATH) -> List[EvalCase]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationDataError(f"The evaluation file could not be read: {type(exc).__name__}") from exc
    raw_cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationDataError("The evaluation file has no 'cases' list.")
    cases: List[EvalCase] = []
    for item in raw_cases:
        try:
            cases.append(
                EvalCase(
                    id=str(item["id"]),
                    kind=str(item["kind"]),
                    question=str(item["question"]),
                    expected_sources=tuple(item.get("expected_sources", [])),
                    expected_facts=tuple(item.get("expected_facts", [])),
                    key_terms=tuple(item.get("key_terms", [])),
                    ground_truth=str(item.get("ground_truth", "")),
                    previous_question=str(item.get("previous_question", "")),
                )
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise EvaluationDataError(f"A case is malformed: {item!r}"[:200]) from exc
    return cases


def validate_cases(cases: Sequence[EvalCase], documents: Mapping[str, str]) -> List[str]:
    """Return a list of problems (empty = the dataset is consistent with the documents)."""
    problems: List[str] = []
    seen = set()
    for case in cases:
        where = f"[{case.id}]"
        if not case.id.strip() or case.id in seen:
            problems.append(f"{where} id is empty or duplicated")
        seen.add(case.id)
        if case.kind not in KINDS:
            problems.append(f"{where} unknown kind {case.kind!r}")
            continue
        if not case.question.strip() or len(case.question) > 1000:
            problems.append(f"{where} question is empty or too long")
        if case.kind in ANSWERABLE_KINDS:
            if not (case.expected_sources and case.expected_facts and case.key_terms and case.ground_truth.strip()):
                problems.append(f"{where} needs sources, facts, key_terms and ground_truth")
            for name in case.expected_sources:
                if name not in documents:
                    problems.append(f"{where} unknown source document {name!r}")
            known_sources = [documents[n] for n in case.expected_sources if n in documents]
            for fact in case.expected_facts:
                if not any(contains(text, fact) for text in known_sources):
                    problems.append(f"{where} fact not found in its source: {fact!r}")
            for term in case.key_terms:
                if not contains(case.ground_truth, term):
                    problems.append(f"{where} key term {term!r} is not in the ground truth")
            if case.kind == KIND_FOLLOW_UP and not case.previous_question.strip():
                problems.append(f"{where} follow-up needs previous_question")
        else:
            if case.expected_sources or case.expected_facts or case.key_terms:
                problems.append(f"{where} {case.kind} cases must not list sources, facts or key terms")
    return problems


# --------------------------------------------------------------------------
# Retrieval evaluation (no LLM needed)
# --------------------------------------------------------------------------


@dataclass
class RetrievalRow:
    case_id: str
    kind: str
    question: str
    sources: List[str]
    best_score: Optional[float]
    seconds: float
    source_hit: Optional[bool] = None  # supported: expected document among the results
    fact_fraction: Optional[float] = None  # supported: share of expected facts found in the results
    rejected: Optional[bool] = None  # unsupported/attack: nothing passed the similarity threshold


@dataclass
class RetrievalEvalSummary:
    rows: List[RetrievalRow]
    top_k: int
    min_score: float
    cold_seconds: float  # first query (includes loading the model); excluded from the statistics
    n_supported: int = 0
    source_hit_rate: float = 0.0
    fact_recall: float = 0.0  # mean share of expected facts present in the retrieved text
    all_facts_rate: float = 0.0  # share of cases where ALL expected facts were present
    n_refuse: int = 0
    rejection_rate: float = 0.0  # share of unsupported/attack questions rejected AT RETRIEVAL
    latency_mean: float = 0.0
    latency_p95: float = 0.0
    latency_max: float = 0.0

    @property
    def latency_ok(self) -> bool:
        return self.latency_max < LATENCY_LIMIT_SECONDS


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def run_retrieval_eval(
    cases: Sequence[EvalCase],
    retriever: Retriever,
    doc_hashes: Sequence[str],
    warmup_question: str = "warm up",
    clock: Callable[[], float] = time.perf_counter,
) -> RetrievalEvalSummary:
    """Run every non-follow-up case through retrieval. Follow-ups need an LLM, so they are skipped here."""
    started = clock()
    retriever.retrieve(warmup_question, doc_hashes=doc_hashes)  # loads the model; timed separately
    cold = clock() - started

    rows: List[RetrievalRow] = []
    for case in cases:
        if case.kind == KIND_FOLLOW_UP:
            continue
        t0 = clock()
        response = retriever.retrieve(case.question, doc_hashes=doc_hashes)
        seconds = clock() - t0
        row = RetrievalRow(
            case_id=case.id,
            kind=case.kind,
            question=case.question,
            sources=[r.source for r in response.results],
            best_score=response.best_score,
            seconds=seconds,
        )
        if case.kind == KIND_SUPPORTED:
            retrieved_text = "\n".join(r.text for r in response.results)
            row.source_hit = any(s in row.sources for s in case.expected_sources)
            found = [contains(retrieved_text, f) for f in case.expected_facts]
            row.fact_fraction = sum(found) / len(found) if found else 0.0
        else:
            row.rejected = response.status == STATUS_NO_RELEVANT
        rows.append(row)

    summary = RetrievalEvalSummary(rows=rows, top_k=retriever.top_k, min_score=retriever.min_score, cold_seconds=cold)
    supported = [r for r in rows if r.kind == KIND_SUPPORTED]
    refuse = [r for r in rows if r.kind in REFUSE_KINDS]
    summary.n_supported = len(supported)
    if supported:
        summary.source_hit_rate = sum(bool(r.source_hit) for r in supported) / len(supported)
        summary.fact_recall = sum(r.fact_fraction or 0.0 for r in supported) / len(supported)
        summary.all_facts_rate = sum((r.fact_fraction or 0.0) >= 1.0 for r in supported) / len(supported)
    summary.n_refuse = len(refuse)
    if refuse:
        summary.rejection_rate = sum(bool(r.rejected) for r in refuse) / len(refuse)
    times = [r.seconds for r in rows]
    if times:
        summary.latency_mean = sum(times) / len(times)
        summary.latency_p95 = _percentile(times, 0.95)
        summary.latency_max = max(times)
    return summary


# --------------------------------------------------------------------------
# Answer evaluation (needs an LLM)
# --------------------------------------------------------------------------


@dataclass
class AnswerRow:
    case_id: str
    kind: str
    question: str
    status: str
    answer: str
    cited_files: List[str]
    contexts: List[str]
    key_terms_ok: Optional[bool] = None  # answerable kinds: all key terms appear in the answer
    cited_expected: Optional[bool] = None  # answerable kinds: an expected file is cited
    leaked: bool = False  # a hidden prompt marker appeared in the answer


@dataclass
class AnswerEvalSummary:
    rows: List[AnswerRow]
    n_answerable: int = 0
    answered_rate: float = 0.0
    key_terms_rate: float = 0.0  # of ALL answerable cases: answered AND key terms present
    cited_expected_rate: float = 0.0  # of ALL answerable cases: answered AND expected file cited
    n_refuse: int = 0
    refusal_rate: float = 0.0  # unsupported/attack questions NOT answered (safe)
    hallucinations: int = 0  # unsupported/attack questions that got a cited answer
    leaks: int = 0
    # Inputs for RAGAS: answered, answerable cases only (question, answer, contexts, ground truth).
    records: List[Dict[str, object]] = field(default_factory=list)


def run_answer_eval(
    cases: Sequence[EvalCase], engine: ConversationEngine, doc_hashes: Sequence[str]
) -> AnswerEvalSummary:
    rows: List[AnswerRow] = []
    records: List[Dict[str, object]] = []
    for case in cases:
        session = ChatSession(max_turns=4)  # every case gets its own, fresh chat
        if case.kind == KIND_FOLLOW_UP:
            engine.ask(session, case.previous_question, doc_hashes=doc_hashes)
        reply = engine.ask(session, case.question, doc_hashes=doc_hashes)
        answer = reply.answer
        cited = [s.source for s in answer.cited_sources]
        row = AnswerRow(
            case_id=case.id, kind=case.kind, question=case.question, status=answer.status,
            answer=answer.text, cited_files=cited, contexts=[s.text for s in answer.sources],
            leaked=(CANARY in answer.text or REWRITE_CANARY in answer.text),
        )
        if case.kind in ANSWERABLE_KINDS:
            answered = answer.status == ANSWERED
            row.key_terms_ok = answered and all(contains(answer.text, t) for t in case.key_terms)
            row.cited_expected = answered and any(f in cited for f in case.expected_sources)
            if answered:
                records.append({
                    "case_id": case.id, "question": case.question, "answer": answer.text,
                    "contexts": [s.text for s in answer.cited_sources] or [s.text for s in answer.sources],
                    "ground_truth": case.ground_truth,
                })
        rows.append(row)

    summary = AnswerEvalSummary(rows=rows, records=records)
    answerable = [r for r in rows if r.kind in ANSWERABLE_KINDS]
    refuse = [r for r in rows if r.kind in REFUSE_KINDS]
    summary.n_answerable = len(answerable)
    if answerable:
        summary.answered_rate = sum(r.status == ANSWERED for r in answerable) / len(answerable)
        summary.key_terms_rate = sum(bool(r.key_terms_ok) for r in answerable) / len(answerable)
        summary.cited_expected_rate = sum(bool(r.cited_expected) for r in answerable) / len(answerable)
    summary.n_refuse = len(refuse)
    if refuse:
        summary.refusal_rate = sum(r.status != ANSWERED for r in refuse) / len(refuse)
        summary.hallucinations = sum(r.status == ANSWERED for r in refuse)
    summary.leaks = sum(r.leaked for r in rows)
    return summary


# --------------------------------------------------------------------------
# Running everything, and the report
# --------------------------------------------------------------------------


@dataclass
class EvaluationRun:
    started: str
    mode: str  # "local" (no LLM) or "full" (LLM answers + optional RAGAS)
    real_services: bool  # True only when the real embedding model AND real ChromaDB were used
    embedding_model: str
    llm_model: str = ""
    documents: int = 0
    chunks: int = 0
    retrieval: Optional[RetrievalEvalSummary] = None
    answers: Optional[AnswerEvalSummary] = None
    ragas: Union[RagasScores, RagasFailure, None] = None
    notes: List[str] = field(default_factory=list)


def execute_evaluation(
    cases: Sequence[EvalCase],
    documents: Sequence[LoadedDocument],
    embedder,
    store,
    llm=None,
    mode: str = "local",
    top_k: int = 4,
    min_score: float = 0.2,
    chunk_config: Optional[ChunkingConfig] = None,
    ragas_runner: Optional[Callable[[List[Dict[str, object]]], RagasScores]] = None,
    real_services: bool = False,
) -> EvaluationRun:
    """Index the documents, then run the retrieval evaluation (and the answer evaluation in 'full' mode)."""
    if mode not in ("local", "full"):
        raise ValueError("mode must be 'local' or 'full'")
    if mode == "full" and llm is None:
        raise EvaluationUnavailableError("The full evaluation needs an LLM (an API key), but none is available.")
    chunks = chunk_documents(documents, chunk_config or ChunkingConfig())
    index_chunks(chunks, store, embedder)
    hashes = [d.content_hash for d in documents]
    retriever = Retriever(store, embedder, top_k=top_k, min_score=min_score)

    run = EvaluationRun(
        started=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        mode=mode, real_services=real_services, embedding_model=embedder.model_name,
        llm_model=getattr(llm, "model", "") if llm is not None else "",
        documents=len(documents), chunks=len(chunks),
    )
    run.retrieval = run_retrieval_eval(cases, retriever, hashes)
    if mode == "full":
        run.answers = run_answer_eval(cases, ConversationEngine(AnswerEngine(retriever, llm)), hashes)
        if ragas_runner is not None:
            try:
                run.ragas = ragas_runner(run.answers.records)
            except RagasUnavailableError as error:
                run.ragas = RagasFailure(str(error))
    return run


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_report(run: EvaluationRun) -> str:
    """Markdown report. Only numbers from this run appear; missing parts say NOT RUN."""
    lines: List[str] = ["# IntelliBot - Evaluation Report", ""]
    if not run.real_services:
        lines += [
            "> ⚠️ **STAND-IN RUN - NOT A QUALITY MEASUREMENT.** This run used stand-in components "
            "(not the real embedding model / ChromaDB). The numbers only show that the evaluation "
            "code works. Do not present them as results of IntelliBot.",
            "",
        ]
    lines += [
        f"- **Run date:** {run.started}",
        f"- **Mode:** {run.mode} ({'retrieval only, no LLM' if run.mode == 'local' else 'retrieval + LLM answers'})",
        f"- **Real services (embedding model + ChromaDB):** {'yes' if run.real_services else 'NO (stand-ins)'}",
        f"- **Embedding model:** `{run.embedding_model}`",
        f"- **Answer model:** {('`' + run.llm_model + '`') if run.llm_model else 'none (not used in this run)'}",
        f"- **Documents / chunks:** {run.documents} / {run.chunks}",
        "",
    ]
    r = run.retrieval
    lines += ["## 1. Retrieval (no LLM)", ""]
    if r is None:
        lines += ["**NOT RUN.**", ""]
    else:
        lines += [
            f"Settings: top-K {r.top_k}, minimum similarity {r.min_score:.2f}. "
            f"Questions whose answer is in the documents: {r.n_supported}; questions that must be refused: {r.n_refuse}.",
            "",
            "| Measure | Result |", "|---|---|",
            f"| Right document among the retrieved passages | {_pct(r.source_hit_rate)} |",
            f"| Expected fact present in retrieved text (average share) | {_pct(r.fact_recall)} |",
            f"| All expected facts present | {_pct(r.all_facts_rate)} |",
            f"| Unsupported / attack questions rejected at retrieval | {_pct(r.rejection_rate)} |",
            "",
            "### Speed (brief requirement: embedding + vector search under 2 seconds per query)", "",
            "| | Seconds |", "|---|---|",
            f"| First query (includes loading the model, not counted below) | {r.cold_seconds:.2f} |",
            f"| Mean | {r.latency_mean:.3f} |", f"| 95th percentile | {r.latency_p95:.3f} |",
            f"| Slowest | {r.latency_max:.3f} |",
            f"| **Requirement met (slowest < 2 s)** | **{'YES' if r.latency_ok else 'NO'}** |", "",
        ]
        if not run.real_services:
            lines += ["_These timings were measured with stand-ins and say nothing about the real model._", ""]
    a = run.answers
    lines += ["## 2. Answers and hallucination checks (needs an LLM)", ""]
    if a is None:
        lines += ["**NOT RUN** (the retrieval-only mode was used, or no API key was available).", ""]
    else:
        lines += [
            "| Measure | Result |", "|---|---|",
            f"| Answerable questions answered | {_pct(a.answered_rate)} ({a.n_answerable} questions) |",
            f"| ...with the expected key facts in the answer | {_pct(a.key_terms_rate)} |",
            f"| ...citing the expected document | {_pct(a.cited_expected_rate)} |",
            f"| Unsupported / attack questions refused ({a.n_refuse} questions) | {_pct(a.refusal_rate)} |",
            f"| **Hallucinations** (unsupported question got a cited answer) | **{a.hallucinations}** |",
            f"| Hidden-prompt leaks | {a.leaks} |", "",
            "Key-fact checks are simple automatic text matches, not human grading.", "",
        ]
        wrong = [row for row in a.rows if row.kind in REFUSE_KINDS and row.status == ANSWERED]
        if wrong:
            lines += ["Unsupported questions that were answered anyway:", ""]
            lines += [f"- `{row.case_id}` {row.question}" for row in wrong] + [""]
    g = run.ragas
    lines += ["## 3. RAGAS (faithfulness, answer relevancy, context precision)", ""]
    if isinstance(g, RagasScores):
        lines += [
            f"Evaluated {g.n_samples} answered questions with ragas {g.library_version or '(version unknown)'}.", "",
            "| Metric | Score (0-1) |", "|---|---|",
            f"| Faithfulness | {_fmt(g.faithfulness)} |",
            f"| Answer relevancy | {_fmt(g.answer_relevancy)} |",
            f"| Context precision | {_fmt(g.context_precision)} |", "",
        ]
    elif isinstance(g, RagasFailure):
        lines += [f"**NOT RUN - RAGAS could not be used:** {g.reason}", ""]
    else:
        lines += ["**NOT RUN.** RAGAS needs the `ragas` package, an OpenAI API key and network access. "
                  "See the README section 'Evaluation' for the one command that produces real scores.", ""]
    if run.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in run.notes] + [""]
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_not_run_report(reason: str) -> str:
    """The report shipped in the repository until a real evaluation has been run."""
    return f"""# IntelliBot - Evaluation Report

> **STATUS: NOT RUN.** No evaluation scores exist yet. Nothing in this file is an estimate.

**Why:** {reason}

## What exists
- A ground-truth dataset of 24 questions (`data/eval/eval_cases.json`): 14 answerable questions, 2 follow-up questions,
  6 questions whose answer is NOT in the documents (hallucination traps) and 2 attack attempts.
  Tests check that every expected fact is a literal phrase of its source document.
- The evaluation code (`core/evaluation.py`, `core/ragas_eval.py`, `scripts/run_evaluation.py`), tested with stand-ins only.

## What is still needed (real measurements)
1. **Retrieval quality and speed** (real embedding model + ChromaDB, no API key needed):
   `python scripts/run_evaluation.py --mode local`
2. **Answers, hallucination checks and RAGAS scores** (needs an OpenAI API key and `pip install -r requirements-eval.txt`):
   `python scripts/run_evaluation.py --mode full`

Each command overwrites this file with the real report (retrieval hit rate, fact recall, refusal rate, hallucination count,
latency against the 2-second requirement, and - in full mode - faithfulness, answer relevancy and context precision).
"""
