"""
RAGAS Evaluator — AI Knowledge Assistant
=========================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS RAGAS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAGAS (Retrieval-Augmented Generation Assessment) is a Python framework
that measures the quality of your RAG pipeline objectively — without
requiring a human to manually read and score every answer.

The core insight: a RAG pipeline has two separable failure modes.

  RETRIEVAL FAILURE  — the right information was not retrieved at all
                       (the vector search missed the relevant chunks)

  GENERATION FAILURE — the right chunks were retrieved but the LLM
                       hallucinated, ignored context, or gave an
                       off-topic answer

RAGAS has separate metrics for each failure mode, so you can pinpoint
exactly where your pipeline degrades.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE FOUR METRICS USED HERE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────┬──────────────────────────────────────────────┐
  │ Metric              │ What it measures                             │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Faithfulness        │ Is every claim in the answer supported by    │
  │                     │ the retrieved context? Detects hallucination.│
  │                     │ Range 0–1. 1.0 = no hallucination.           │
  │                     │ Does NOT need ground truth.                  │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Answer Relevancy    │ Does the answer actually address what was    │
  │                     │ asked? A correct but off-topic answer scores │
  │                     │ low. Range 0–1.                              │
  │                     │ Does NOT need ground truth.                  │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Context Precision   │ Of the chunks we retrieved, how many were    │
  │                     │ actually useful? Measures retrieval signal-  │
  │                     │ to-noise. Range 0–1.                         │
  │                     │ Needs ground truth answer.                   │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Context Recall      │ Did retrieval fetch all the information      │
  │                     │ needed to answer? Low recall = the retriever │
  │                     │ missed crucial chunks. Range 0–1.            │
  │                     │ Needs ground truth answer.                   │
  └─────────────────────┴──────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW RAGAS COMPUTES EACH METRIC (step by step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FAITHFULNESS
  ─────────────
  1. RAGAS sends the answer to the judge LLM and asks it to decompose the
     answer into a list of atomic factual claims.
     e.g. "Alice is the CTO. She joined in 2019." → ["Alice is CTO",
          "Alice joined in 2019"]
  2. For each claim, RAGAS asks the judge LLM: "Can this claim be
     directly inferred from the retrieved context?"
  3. Score = verified_claims / total_claims
     If 3 out of 4 claims are grounded in context → faithfulness = 0.75

  ANSWER RELEVANCY
  ─────────────────
  1. RAGAS sends the answer to the judge LLM and asks it to generate N
     synthetic questions that the answer could be responding to.
     e.g. answer: "Alice joined the company in 2019 as CTO."
          synthetic questions: ["When did Alice join?", "What is Alice's role?"]
  2. Each synthetic question is embedded using the embeddings model.
  3. Score = mean cosine similarity between synthetic questions and the
     original user question.
     High similarity → the answer squarely addressed the question.
     Low similarity → the answer may be correct but off-topic.

  CONTEXT PRECISION  (needs ground_truth)
  ─────────────────────────────────────────
  1. For each retrieved chunk (in ranked order), the judge LLM is asked:
     "Given the reference answer, is this chunk useful for answering
     the question?"
  2. A weighted precision is computed that penalises irrelevant chunks
     appearing early in the ranked list more than those appearing late.
  3. Score near 1.0 = the retriever surfaced mostly useful chunks.
     Score near 0.0 = retrieved chunks are mostly noise.

  CONTEXT RECALL  (needs ground_truth)
  ──────────────────────────────────────
  1. The ground_truth answer is decomposed into sentences.
  2. For each sentence, the judge LLM is asked: "Can this sentence be
     attributed to any of the retrieved chunks?"
  3. Score = attributed_sentences / total_sentences
  4. Low recall means the retriever missed chunks that contained
     information present in the reference answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW RAGAS FITS INTO THIS PROJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ONLINE EVALUATION (live traffic)
  ──────────────────────────────────
  After every /ask response is returned to the user, FastAPI calls
  evaluate_and_log() as a BackgroundTask. The user's response is NOT
  delayed — evaluation happens in parallel. Scores are appended to
  logs/ragas_eval_log.jsonl. Only Faithfulness + AnswerRelevancy run
  here (no ground truth available in live traffic).

  OFFLINE EVALUATION (batch / CI)
  ─────────────────────────────────
  Run `python scripts/run_evaluation.py` against data/eval_testset.json.
  This runs the full pipeline (retrieval → reranking → generation →
  RAGAS) on a curated set of questions with known correct answers, so
  all four metrics (including ContextPrecision + ContextRecall) are
  computed. Use this before deploying a new embedding model or changing
  chunk parameters to verify quality didn't regress.

  THE JUDGE LLM
  ──────────────
  RAGAS itself calls an LLM to reason about faithfulness, relevance, and
  context coverage. We use the same Groq/Llama model we already have —
  no extra API key needed. LangchainLLMWrapper adapts Groq to the
  interface RAGAS expects.

  THE JUDGE EMBEDDINGS
  ─────────────────────
  AnswerRelevancy needs to embed synthetic questions. We reuse the same
  BAAI/bge-base-en-v1.5 model already loaded for the RAG pipeline via
  LangchainEmbeddingsWrapper — no extra download required.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from app.config import GROQ_API_KEY, LLM_MODEL
from app.services.logger import get_logger

log = get_logger(__name__)

EVAL_LOG_PATH = "logs/ragas_eval_log.jsonl"

# Lazy-initialised RAGAS components. Initialised on first call to is_available().
# Using lazy init means: if ragas is not installed, the app still starts
# normally — eval is silently skipped (graceful degradation).
_ragas_llm = None
_ragas_embeddings = None
_ragas_available: Optional[bool] = None  # None = not yet checked


def _init_ragas() -> None:
    """
    Import and initialise RAGAS judge LLM + embeddings on first use.

    WHY LAZY INITIALISATION?
    ─────────────────────────
    ragas, langchain-groq, and langchain-huggingface are optional extras.
    If they are not installed, the rest of the application keeps running
    and /ask continues to work — RAGAS eval is simply skipped. This avoids
    making RAGAS a hard startup dependency.

    WHY GROQ AS THE JUDGE?
    ───────────────────────
    RAGAS needs an LLM to reason about faithfulness and relevancy. We reuse
    the same Groq API key and Llama model already configured for generation,
    so no new credentials or cost centres are required. In production you
    might swap this for a stronger judge (e.g. claude-sonnet-4-6 via the
    ANTHROPIC_API_KEY already in .env) for higher evaluation accuracy.

    WHY BGE AS JUDGE EMBEDDINGS?
    ─────────────────────────────
    AnswerRelevancy embeds synthetic questions and compares them to the
    original question. We reuse the BGE model already downloaded for the RAG
    pipeline — no second model download needed.
    """
    global _ragas_llm, _ragas_embeddings, _ragas_available

    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper        # triggers nest_asyncio.apply()
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # ── Fix nest_asyncio + Python 3.12+ incompatibility ───────────────────
        # nest_asyncio (called at import time in ragas/executor.py) replaces
        # asyncio.Task with the pure-Python _PyTask. Python 3.12+ moved
        # asyncio.current_task() to a C built-in that tracks _CTask instances
        # in C-level storage — it cannot see _PyTask instances, so it always
        # returns None after nest_asyncio applies.  This causes asyncio.timeout()
        # (used inside RAGAS metric coroutines) to raise:
        #   RuntimeError: Timeout should be used inside a task
        # It also corrupts anyio's event loop in FastAPI tests because anyio
        # relies on current_task() for its own task-switching logic.
        #
        # Fix: replace current_task() with a Python version that reads from
        # asyncio.tasks._current_tasks — the Python dict that _PyTask.__step()
        # correctly updates (and that _CTask.__step() also updates, so the
        # patch is safe for non-nest_asyncio code paths too).
        import asyncio as _asyncio
        import asyncio.tasks as _asyncio_tasks
        if not getattr(_asyncio, '_current_task_patched', False):
            def _fixed_current_task(loop=None):
                if loop is None:
                    try:
                        loop = _asyncio.get_running_loop()
                    except RuntimeError:
                        return None
                return _asyncio_tasks._current_tasks.get(loop)

            _asyncio.current_task = _fixed_current_task
            _asyncio_tasks.current_task = _fixed_current_task
            _asyncio._current_task_patched = True

        # Wrap the Groq chat model so RAGAS can call it like a LangChain LLM.
        # temperature=0 ensures deterministic, reproducible evaluation scores.
        _ragas_llm = LangchainLLMWrapper(
            ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0)
        )

        # Wrap the same BGE embedding model used in the RAG pipeline.
        # RAGAS uses this to embed synthetic questions for AnswerRelevancy.
        _ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        )

        _ragas_available = True
        log.info("RAGAS evaluator initialised successfully")

    except ImportError as e:
        log.warning(
            "RAGAS dependencies not found — evaluation disabled. "
            "Install with: pip install ragas langchain-groq langchain-huggingface",
            extra={"error": str(e)},
        )
        _ragas_available = False
    except Exception as e:
        log.error(f"RAGAS initialisation failed: {e}")
        _ragas_available = False


def is_available() -> bool:
    """Return True if RAGAS is installed and the judge LLM is ready."""
    if _ragas_available is None:
        _init_ragas()
    return bool(_ragas_available)


def _append_to_log(record: dict) -> None:
    """Write one JSON-encoded evaluation result to the newline-delimited log."""
    os.makedirs("logs", exist_ok=True)
    with open(EVAL_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def evaluate_and_log(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """
    Evaluate one question/answer/context triple with RAGAS and log the scores.

    This is the central function called both by the live /ask background task
    (online mode) and by the offline batch evaluation script.

    Parameters
    ----------
    question     : The original user question.
    answer       : The answer generated by our RAG pipeline.
    contexts     : List of raw chunk texts passed to the LLM as context.
                   These are what RAGAS inspects for faithfulness and recall.
    ground_truth : Known-correct reference answer. When provided, RAGAS also
                   computes ContextPrecision and ContextRecall. Leave None for
                   live traffic where no ground truth exists.
    request_id   : Optional correlation ID for tracing this eval back to a
                   specific /ask request in the logs.

    Step-by-step flow
    -----------------
    1. Build a SingleTurnSample — the RAGAS data structure for one eval case.
    2. Wrap it in an EvaluationDataset (RAGAS evaluate() expects a dataset).
    3. Choose which metrics to run based on whether ground_truth is available.
    4. Call ragas.evaluate() — this makes multiple LLM API calls internally.
    5. Parse the scored DataFrame into a plain dict.
    6. Append the result to logs/ragas_eval_log.jsonl.
    """
    if not is_available():
        return

    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy

        # ── STEP 1: Build SingleTurnSample ────────────────────────────────────
        #
        # SingleTurnSample is the core data class in RAGAS for one evaluation
        # unit. The fields map directly to our RAG pipeline's outputs:
        #
        #   user_input        = the question the user typed into the chat UI
        #   response          = the answer our LLM generated (what we're judging)
        #   retrieved_contexts = the list of raw chunk text strings that were
        #                        passed to the LLM in the context window.
        #                        RAGAS reads these to check faithfulness and recall.
        #   reference         = a known-correct answer written by a human (optional).
        #                        Required for ContextPrecision and ContextRecall.
        #
        # Why list of strings for contexts?
        # RAGAS expects each retrieved passage as a separate string in the list,
        # NOT one big concatenated string. We pass [chunk["text"] for chunk in ...]
        # at the call sites, which preserves individual chunk boundaries.
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts if contexts else [""],
            reference=ground_truth,
        )

        # ── STEP 2: Wrap in EvaluationDataset ────────────────────────────────
        #
        # RAGAS's evaluate() function always takes a dataset (list of samples).
        # For per-request online eval we pass a single-item dataset.
        # For batch eval the caller passes multiple samples.
        dataset = EvaluationDataset(samples=[sample])

        # ── STEP 3: Choose metrics ────────────────────────────────────────────
        #
        # Without ground_truth → only reference-free metrics are possible.
        #   Faithfulness()    checks answer vs. contexts (no reference needed)
        #   AnswerRelevancy() checks answer vs. question (no reference needed)
        #
        # With ground_truth → add supervised metrics:
        #   ContextPrecision() checks retrieved chunks vs. reference answer
        #   ContextRecall()    checks retrieved chunks vs. reference answer
        #
        # We try to import the supervised metrics separately so that if only
        # the reference-free ones are needed, a missing import doesn't fail.
        metrics = [Faithfulness(), AnswerRelevancy()]

        if ground_truth:
            try:
                from ragas.metrics import ContextPrecision, ContextRecall
                metrics += [ContextPrecision(), ContextRecall()]
            except ImportError:
                log.warning("ContextPrecision/ContextRecall not available in this ragas version")

        # ── STEP 4: Run evaluation ────────────────────────────────────────────
        #
        # ragas.evaluate() orchestrates all the LLM calls needed for each metric:
        #
        #   Faithfulness:
        #     - LLM call 1: "Decompose this answer into atomic claims"
        #     - LLM call 2 (per claim): "Can this claim be inferred from context?"
        #
        #   AnswerRelevancy:
        #     - LLM call: "Generate N questions this answer could respond to"
        #     - Embedding call: embed synthetic questions + original question
        #     - Cosine similarity computed locally (no LLM call)
        #
        # This typically takes 5–15 seconds. That's why we run it in a
        # BackgroundTask — the user's /ask response is NOT delayed.
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=_ragas_llm,
            embeddings=_ragas_embeddings,
        )

        # ── STEP 5: Parse scores ──────────────────────────────────────────────
        #
        # result.to_pandas() returns a DataFrame with one row per sample and
        # one column per metric. Columns for metrics that weren't run (e.g.
        # ContextPrecision when no ground_truth) contain NaN.
        # We convert to a plain dict and drop NaN values (v == v is False for NaN).
        raw_scores = result.to_pandas().to_dict(orient="records")[0]
        scores = {
            k: round(float(v), 4)
            for k, v in raw_scores.items()
            if isinstance(v, (int, float)) and v == v  # filters NaN
        }

        # ── STEP 6: Write to log ──────────────────────────────────────────────
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "question": question,
            "answer": answer[:300],  # truncate for log readability
            "num_contexts": len(contexts),
            "scores": scores,
        }
        _append_to_log(record)
        log.info("RAGAS eval complete", extra={"scores": scores, "request_id": request_id})

    except Exception as e:
        # Evaluation failure must NEVER surface to the user or break /ask.
        log.error(f"RAGAS evaluation error: {e}", extra={"request_id": request_id})


def get_scores_summary() -> dict:
    """
    Read logs/ragas_eval_log.jsonl and return average scores across all logged
    evaluations. Useful for building a monitoring dashboard or CI threshold check.

    Returns a dict like:
      {
        "total_samples": 42,
        "avg_faithfulness": 0.87,
        "avg_answer_relevancy": 0.91,
        "avg_context_precision": 0.76,   # only if ground truth was provided
        "avg_context_recall": 0.82,
      }
    """
    if not os.path.exists(EVAL_LOG_PATH):
        return {"total_samples": 0}

    records: list[dict] = []
    with open(EVAL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return {"total_samples": 0}

    # Aggregate per-metric scores across all records that have that metric
    metric_totals: dict[str, list[float]] = {}
    for rec in records:
        for metric, score in rec.get("scores", {}).items():
            metric_totals.setdefault(metric, []).append(score)

    summary: dict = {"total_samples": len(records)}
    for metric, values in metric_totals.items():
        summary[f"avg_{metric}"] = round(sum(values) / len(values), 4)

    return summary
