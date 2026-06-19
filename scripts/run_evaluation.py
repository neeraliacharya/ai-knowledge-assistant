#!/usr/bin/env python
"""
Offline RAGAS Batch Evaluation Script
======================================

PURPOSE
--------
Runs the full RAG pipeline (retrieval → reranking → generation → RAGAS scoring)
against a curated test set of questions with known correct answers.

Use this script:
  - Before changing chunk size / overlap to verify quality doesn't regress
  - After swapping the embedding model to compare retrieval quality
  - In CI to gate deployments on a minimum faithfulness threshold
  - To produce a quality report for stakeholders

USAGE
------
  # Run from the project root:
  python scripts/run_evaluation.py

  # Custom test set:
  python scripts/run_evaluation.py --testset data/my_custom_testset.json

  # Fail CI if any score is below a threshold:
  python scripts/run_evaluation.py --min-faithfulness 0.7 --min-relevancy 0.75

OUTPUT
-------
  - Per-sample scores printed to stdout in a table
  - Full JSON records appended to logs/ragas_eval_log.jsonl
  - Summary averages printed at the end
  - Exit code 1 if any threshold is violated

WHAT EACH METRIC MEANS IN CONTEXT
------------------------------------
  faithfulness      Score 0–1. 1.0 means the LLM never added information
                    that wasn't in the retrieved chunks. Low scores mean
                    hallucination — the LLM is making things up.

  answer_relevancy  Score 0–1. 1.0 means the answer directly addresses the
                    question. Low scores mean the answer is correct but
                    off-topic (e.g. answering about pricing when asked about
                    competitors).

  context_precision Score 0–1. 1.0 means every retrieved chunk was useful.
                    Low scores mean the retriever is surfacing noisy chunks —
                    try tuning chunk_size, top_k, or the embedding model.

  context_recall    Score 0–1. 1.0 means the retriever found all information
                    needed to answer. Low scores mean relevant chunks exist
                    in the corpus but weren't retrieved — try increasing top_k
                    or improving chunking overlap.
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_rag_pipeline():
    """
    Import and initialise the RAG pipeline.

    We try the disk cache first (fast, ~1s). If no cache exists we do a full
    rebuild from the documents/ folder (slow, ~30s for large corpora).
    """
    print("Loading RAG pipeline...")
    import app.services.rag_pipeline as rag_pipeline
    if not rag_pipeline._load_from_disk():
        print("  No disk cache found — running full initialisation...")
        rag_pipeline.initialize_rag()
    else:
        print(f"  Loaded {len(rag_pipeline.chunks_store)} chunks from disk cache")
    return rag_pipeline


def _run_pipeline(question: str, rag_pipeline) -> tuple[str, list[str]]:
    """
    Run the full RAG pipeline for one question and return (answer, contexts).

    contexts is a list of raw chunk text strings — exactly what RAGAS needs
    to evaluate faithfulness and context coverage.
    """
    from app.services.embedding import embedding_model
    from app.services.llm_service import generate_rag_answer, reformulate_query
    from app.services.reranker import rerank_chunks
    from app.services.retrieval import retrieve_chunks

    search_query = reformulate_query(question, [])

    if rag_pipeline.vector_index is None:
        return "No documents indexed.", []

    retrieved = retrieve_chunks(
        search_query,
        embedding_model,
        rag_pipeline.vector_index,
        rag_pipeline.chunks_store,
        top_k=10,
    )
    reranked = rerank_chunks(search_query, retrieved)

    context = "\n\n".join(c["text"] for c in reranked[:4])
    answer = generate_rag_answer(question, context, [])
    contexts = [c["text"] for c in reranked]

    return answer, contexts


def main():
    parser = argparse.ArgumentParser(description="RAGAS offline batch evaluation")
    parser.add_argument(
        "--testset",
        default="data/eval_testset.json",
        help="Path to the JSON test set (default: data/eval_testset.json)",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=None,
        help="Fail with exit code 1 if avg faithfulness is below this threshold",
    )
    parser.add_argument(
        "--min-relevancy",
        type=float,
        default=None,
        help="Fail with exit code 1 if avg answer_relevancy is below this threshold",
    )
    args = parser.parse_args()

    # ── Validate prerequisites ─────────────────────────────────────────────
    if not os.path.exists(args.testset):
        print(f"ERROR: Test set not found at {args.testset}")
        print("Edit data/eval_testset.json with questions from your documents first.")
        sys.exit(1)

    import app.services.ragas_evaluator as ragas_eval
    if not ragas_eval.is_available():
        print(
            "ERROR: RAGAS dependencies not installed.\n"
            "Run: pip install ragas langchain-groq langchain-huggingface"
        )
        sys.exit(1)

    # ── Load test set ──────────────────────────────────────────────────────
    with open(args.testset) as f:
        testset = [item for item in json.load(f) if not item.get("_comment")]

    print(f"Test set: {len(testset)} samples from {args.testset}")

    # ── Load RAG pipeline ──────────────────────────────────────────────────
    rag = _load_rag_pipeline()
    if rag.vector_index is None:
        print("ERROR: No documents indexed. Upload PDFs via the API first.")
        sys.exit(1)

    # ── Evaluate each sample ───────────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"{'#':>3}  {'Question':<45}  {'Faith':>6}  {'Relev':>6}")
    print("─" * 72)

    for i, item in enumerate(testset, 1):
        question = item["question"]
        ground_truth: Optional[str] = item.get("ground_truth")

        # Run the full RAG pipeline to get the actual answer + retrieved chunks
        # This is what RAGAS will judge — the real pipeline output, not canned answers.
        t0 = time.time()
        answer, contexts = _run_pipeline(question, rag)
        pipeline_ms = int((time.time() - t0) * 1000)

        # Call RAGAS evaluation — this makes LLM API calls internally
        # and appends results to logs/ragas_eval_log.jsonl
        ragas_eval.evaluate_and_log(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        # Read back the last written score for display
        scores = _read_last_scores()
        faith = f"{scores.get('faithfulness', float('nan')):.3f}"
        relev = f"{scores.get('answer_relevancy', float('nan')):.3f}"

        q_short = question[:43] + ".." if len(question) > 45 else question
        print(f"{i:>3}. {q_short:<45}  {faith:>6}  {relev:>6}  ({pipeline_ms}ms)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("─" * 72)
    summary = ragas_eval.get_scores_summary()
    print("\nAVERAGE RAGAS SCORES ACROSS ALL SAMPLES")
    print("─" * 40)
    for key, val in summary.items():
        if key == "total_samples":
            print(f"  {'Samples evaluated':<30} {val}")
        else:
            metric = key.replace("avg_", "").replace("_", " ").title()
            bar = "█" * int(val * 20)
            print(f"  {metric:<30} {val:.4f}  {bar}")
    print()

    # ── Threshold enforcement (for CI) ─────────────────────────────────────
    failed = False
    if args.min_faithfulness is not None:
        avg_faith = summary.get("avg_faithfulness", 0.0)
        if avg_faith < args.min_faithfulness:
            print(
                f"THRESHOLD VIOLATION: avg_faithfulness {avg_faith:.4f} "
                f"< required {args.min_faithfulness}"
            )
            failed = True

    if args.min_relevancy is not None:
        avg_relev = summary.get("avg_answer_relevancy", 0.0)
        if avg_relev < args.min_relevancy:
            print(
                f"THRESHOLD VIOLATION: avg_answer_relevancy {avg_relev:.4f} "
                f"< required {args.min_relevancy}"
            )
            failed = True

    print(f"Full results saved to: logs/ragas_eval_log.jsonl")

    if failed:
        sys.exit(1)


def _read_last_scores() -> dict:
    """Read the most recently written RAGAS score from the log."""
    log_path = "logs/ragas_eval_log.jsonl"
    if not os.path.exists(log_path):
        return {}
    try:
        with open(log_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            return json.loads(lines[-1]).get("scores", {})
    except Exception:
        pass
    return {}


if __name__ == "__main__":
    main()
