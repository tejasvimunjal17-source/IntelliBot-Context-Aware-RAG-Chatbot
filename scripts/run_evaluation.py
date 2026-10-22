#!/usr/bin/env python3
"""Run the IntelliBot evaluation and write docs/EVALUATION_REPORT.md.

  python scripts/run_evaluation.py --mode local   # real embedding model + ChromaDB, NO API key
  python scripts/run_evaluation.py --mode full    # also real LLM answers, hallucination checks and
                                                  # RAGAS scores (needs OPENAI_API_KEY + requirements-eval.txt)

Only REAL services are used here. If the embedding model, ChromaDB or the LLM is not available,
the script prints what is missing and stops WITHOUT writing a report, so a failed run can never
overwrite the report or produce made-up numbers.

Exit codes: 0 = report written, 1 = dataset problem, 2 = a required service is not available.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import SAMPLE_DOCS_DIR, load_settings  # noqa: E402
from core.document_loader import load_path  # noqa: E402
from core.embeddings import EmbeddingError, SentenceTransformerEmbedder  # noqa: E402
from core.evaluation import (  # noqa: E402
    EvaluationDataError,
    EvaluationUnavailableError,
    execute_evaluation,
    load_eval_cases,
    render_report,
    validate_cases,
)
from core.llm import LLMUnavailableError, create_llm_client  # noqa: E402
from core.ragas_eval import run_ragas  # noqa: E402
from core.vector_store import ChromaVectorStore, VectorStoreError  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "EVALUATION_REPORT.md"


def real_services(settings, mode, work_dir):
    """Build the REAL embedder, ChromaDB store and (full mode) LLM client. Raises on anything missing."""
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
    store = ChromaVectorStore(Path(work_dir) / "eval_index", settings.embedding_model)
    llm = create_llm_client(settings) if mode == "full" else None
    return embedder, store, llm, True  # the last value says: these are real services


def main(argv=None, service_factory=real_services) -> int:
    parser = argparse.ArgumentParser(description="Evaluate IntelliBot and write the report.")
    parser.add_argument("--mode", choices=["local", "full"], default="local")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the Markdown report")
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        cases = load_eval_cases()
        documents = [load_path(p).document for p in sorted(SAMPLE_DOCS_DIR.iterdir())]
    except EvaluationDataError as error:
        print(f"Dataset problem: {error}", file=sys.stderr)
        return 1
    problems = validate_cases(cases, {d.name: d.text for d in documents})
    if problems:
        print("The evaluation dataset does not match the sample documents:", *problems, sep="\n  ", file=sys.stderr)
        return 1

    work_dir = tempfile.mkdtemp(prefix="intellibot_eval_")
    try:
        embedder, store, llm, is_real = service_factory(settings, args.mode, work_dir)
        run = execute_evaluation(
            cases, documents, embedder, store, llm=llm, mode=args.mode,
            top_k=settings.top_k, min_score=settings.min_similarity,
            ragas_runner=run_ragas if args.mode == "full" else None, real_services=is_real,
        )
    except (EmbeddingError, VectorStoreError, LLMUnavailableError, EvaluationUnavailableError) as error:
        print(f"Cannot run the evaluation: {error}", file=sys.stderr)
        print("Nothing was written. Install the requirements (and set your API key for --mode full) and try again.",
              file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(run), encoding="utf-8")
    r = run.retrieval
    print(f"Report written to {args.out}")
    print(f"Right document retrieved: {r.source_hit_rate:.0%} | slowest query: {r.latency_max:.3f} s "
          f"({'meets' if r.latency_ok else 'MISSES'} the 2-second requirement)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
