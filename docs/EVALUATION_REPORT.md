# IntelliBot - Evaluation Report

> **STATUS: NOT RUN.** No evaluation scores exist yet. Nothing in this file is an estimate.

**Why:** the build environment had no internet access and no API key, and neither the real embedding model (sentence-transformers), ChromaDB nor RAGAS could be installed there. Running the evaluation without them would only measure stand-ins, which would be misleading.

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
