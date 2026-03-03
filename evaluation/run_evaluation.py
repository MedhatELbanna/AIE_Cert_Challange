"""Run dual evaluation (LangSmith + RAGAS) against a pipeline.

Evaluation has two parts:
  Part A — LangSmith evaluate() with openevals LLM-as-judge evaluators
           (qa_correctness, helpfulness). Enables built-in comparison UI.
  Part B — RAGAS evaluate() with retrieval-quality metrics
           (LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy).

Usage:
  python -m evaluation.run_evaluation --pipeline basic --testset evaluation_results/testset.csv
  python -m evaluation.run_evaluation --pipeline advanced --langsmith-dataset "EDR-SDG-abc123"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)  # Critical: Windows empty env vars issue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval output parsing
# ---------------------------------------------------------------------------

def _parse_retrieval_output(tool_output: str) -> list[str]:
    """Parse the formatted retrieval tool output into individual context strings.

    Both basic (src/tools.py) and advanced (src/tools_advanced.py) use the format:
      --- Result 1 [source: ..., page: ..., type: ...] ---
      <content>

      --- Result 2 [source: ..., page: ..., type: ...] ---
      <content>

    Returns list of content strings (one per result).
    """
    if not tool_output or tool_output.startswith("ERROR:") or tool_output.startswith("No results"):
        return []

    parts = re.split(r"---\s*Result\s+\d+\s*\[.*?\]\s*---\s*\n", tool_output)
    # First element is empty (before first delimiter)
    contexts = [p.strip() for p in parts[1:] if p.strip()]
    return contexts


# ---------------------------------------------------------------------------
# Pipeline setup: index documents and return retrieval tools
# ---------------------------------------------------------------------------

def _setup_basic_pipeline():
    """Load and index documents for the basic pipeline.

    Returns the basic retrieve_documents tool.
    """
    from evaluation.config import PROPOSAL_PDF, SPEC_PDF
    from src.ingestion import chunk_documents, create_index, load_pdf

    logger.info("Setting up basic pipeline...")
    all_chunks = []
    for pdf_path, doc_type in [(SPEC_PDF, "spec"), (PROPOSAL_PDF, "proposal")]:
        if pdf_path.exists():
            docs = load_pdf(pdf_path)
            chunks = chunk_documents(docs, doc_type=doc_type)
            all_chunks.extend(chunks)
            logger.info("  %s: %d chunks from %s", doc_type, len(chunks), pdf_path.name)

    create_index(all_chunks)
    logger.info("  Basic index created: %d total chunks", len(all_chunks))

    from src.tools import retrieve_documents
    return retrieve_documents


def _setup_advanced_pipeline():
    """Load and index documents for the advanced pipeline.

    Returns the advanced retrieve_documents tool.
    """
    from evaluation.config import PROPOSAL_PDF, SPEC_PDF
    from src.ingestion import load_pdf
    from src.ingestion_advanced import (
        build_section_tree,
        chunk_hierarchical,
        create_index_advanced,
        load_pdf_with_layout,
    )

    logger.info("Setting up advanced pipeline...")
    all_chunks = []
    for pdf_path, doc_type in [(SPEC_PDF, "spec"), (PROPOSAL_PDF, "proposal")]:
        if pdf_path.exists():
            result = load_pdf_with_layout(pdf_path)
            sections = build_section_tree(result)
            parents, children = chunk_hierarchical(
                sections, doc_type=doc_type, source=pdf_path.name
            )
            all_chunks.extend(parents + children)
            logger.info(
                "  %s: %d parents + %d children from %s",
                doc_type,
                len(parents),
                len(children),
                pdf_path.name,
            )

    create_index_advanced(all_chunks)
    logger.info("  Advanced index created: %d total chunks", len(all_chunks))

    from src.tools_advanced import retrieve_documents
    return retrieve_documents


# ---------------------------------------------------------------------------
# RAG wrapper: builds both an LCEL chain and a raw rag_fn
# ---------------------------------------------------------------------------

def build_rag_chain(retrieval_tool, top_k: int = 5, max_context_chars: int = 0):
    """Build a RAG chain + low-level rag_fn from a retrieval tool.

    Args:
        retrieval_tool: The @tool retrieval function.
        top_k: Number of results to retrieve.
        max_context_chars: Max total context chars sent to LLM. 0 = no limit.
            Useful for normalizing context budgets across pipelines with
            different chunk sizes (e.g., basic flat chunks vs advanced parent chunks).

    Returns:
        (chain_invoke, rag_fn) tuple where:
          - chain_invoke: callable({"question": str}) -> str
            For LangSmith evaluate() — accepts dict, returns answer string.
          - rag_fn: callable(question: str) -> {"response": str, "contexts": list[str]}
            For RAGAS evaluate() — returns answer + retrieved contexts.
    """
    from langchain_openai import ChatOpenAI

    from evaluation.config import RAG_ANSWER_MODEL

    # Use OpenAI for RAG answer generation (avoids Anthropic credit issues).
    # If RAG_ANSWER_MODEL is None, default to gpt-4.1-mini.
    model_name = RAG_ANSWER_MODEL or "gpt-4.1-mini"
    llm = ChatOpenAI(model=model_name)

    RAG_SYSTEM_PROMPT = (
        "You are an engineering document review assistant. "
        "Answer the question based on the retrieved context from engineering "
        "specifications and proposals. "
        "If the context doesn't contain the answer, say so. "
        "Be specific and cite details from the context."
    )

    def _truncate_contexts(contexts: list[str], max_chars: int) -> list[str]:
        """Truncate context list to fit within max_chars total budget."""
        if max_chars <= 0:
            return contexts
        truncated = []
        remaining = max_chars
        for ctx in contexts:
            if remaining <= 0:
                break
            if len(ctx) <= remaining:
                truncated.append(ctx)
                remaining -= len(ctx)
            else:
                truncated.append(ctx[:remaining] + "... [truncated]")
                remaining = 0
        return truncated

    def rag_fn(question: str) -> dict:
        """Single-question RAG: retrieve + generate answer."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Retrieve
        raw_output = retrieval_tool.invoke(
            {"query": question, "doc_type": "all", "top_k": top_k}
        )
        contexts = _parse_retrieval_output(raw_output)

        # Apply context budget cap if configured
        if max_context_chars > 0:
            contexts = _truncate_contexts(contexts, max_context_chars)
            # Rebuild raw_output from truncated contexts for LLM prompt
            raw_output = "\n\n".join(
                f"--- Result {i+1} ---\n{ctx}" for i, ctx in enumerate(contexts)
            )

        # Generate answer
        response = llm.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Question: {question}\n\nContext:\n{raw_output}"
                ),
            ]
        )

        answer = response.content if hasattr(response, "content") else str(response)
        return {"response": answer, "contexts": contexts}

    def chain_invoke(inputs: dict) -> str:
        """LangSmith-compatible callable: {"question": str} -> str."""
        question = inputs.get("question", "")
        result = rag_fn(question)
        return result["response"]

    return chain_invoke, rag_fn


# ---------------------------------------------------------------------------
# LangSmith evaluators (openevals LLM-as-judge)
# ---------------------------------------------------------------------------

def create_langsmith_evaluators() -> list:
    """Create openevals LLM-as-judge evaluators for LangSmith.

    Returns list of evaluator callables.
    """
    from openevals.llm import create_llm_as_judge

    from evaluation.config import LANGSMITH_JUDGE_MODEL

    qa_evaluator = create_llm_as_judge(
        prompt=(
            "You are evaluating a QA system for engineering document review. "
            "Given the input question, assess whether the prediction correctly "
            "answers the question based on the reference answer.\n\n"
            "Input: {inputs}\n"
            "Prediction: {outputs}\n"
            "Reference answer: {reference_outputs}\n\n"
            "Is the prediction correct? Return 1 if correct, 0 if incorrect."
        ),
        feedback_key="qa_correctness",
        model=LANGSMITH_JUDGE_MODEL,
    )

    helpfulness_evaluator = create_llm_as_judge(
        prompt=(
            "You are assessing a submission based on the following criterion:\n\n"
            "helpfulness: Is this submission helpful to an engineering reviewer, "
            "taking into account the correct reference answer?\n\n"
            "Input: {inputs}\n"
            "Submission: {outputs}\n"
            "Reference answer: {reference_outputs}\n\n"
            "Does the submission meet the criterion? Return 1 if yes, 0 if no."
        ),
        feedback_key="helpfulness",
        model=LANGSMITH_JUDGE_MODEL,
    )

    return [qa_evaluator, helpfulness_evaluator]


# ---------------------------------------------------------------------------
# Part A: LangSmith evaluation
# ---------------------------------------------------------------------------

def run_langsmith_evaluation(
    chain_invoke,
    dataset_name: str,
    revision_id: str,
) -> dict:
    """Run LangSmith evaluate() with openevals LLM-as-judge evaluators.

    Args:
        chain_invoke: Callable({"question": str}) -> str.
        dataset_name: LangSmith dataset name to evaluate against.
        revision_id: Tag for this experiment (e.g., "basic_pipeline").

    Returns:
        Dict with experiment name and scores.
    """
    from langsmith.evaluation import evaluate

    evaluators = create_langsmith_evaluators()

    logger.info("Running LangSmith evaluation against dataset '%s' ...", dataset_name)
    logger.info("  revision_id: %s", revision_id)

    result = evaluate(
        chain_invoke,
        data=dataset_name,
        evaluators=evaluators,
        metadata={"revision_id": revision_id},
    )

    # Extract scores from ExperimentResults
    experiment_name = getattr(result, "experiment_name", revision_id)

    logger.info("  LangSmith experiment: %s", experiment_name)

    return {
        "experiment_name": experiment_name,
        "revision_id": revision_id,
        "dataset_name": dataset_name,
    }


# ---------------------------------------------------------------------------
# Part B: RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    rag_fn,
    testset_df: pd.DataFrame,
    evaluator_llm,
) -> dict:
    """Run RAGAS evaluate() with retrieval-quality metrics.

    Args:
        rag_fn: Callable(question: str) -> {"response": str, "contexts": list[str]}.
        testset_df: DataFrame with user_input, reference columns.
        evaluator_llm: RAGAS-wrapped LLM for metric scoring.

    Returns:
        Dict with aggregate scores and per-sample results.
    """
    from ragas import RunConfig, evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextRecall,
        ResponseRelevancy,
    )

    # Run RAG on each question
    results_rows = []
    total = len(testset_df)
    for idx, row in testset_df.iterrows():
        question = row.get("user_input", "")
        reference = row.get("reference", "")
        synthesizer = row.get("synthesizer_name", "")

        logger.info("  [%d/%d] %s", idx + 1, total, question[:80])
        try:
            rag_result = rag_fn(question)
            results_rows.append(
                {
                    "user_input": question,
                    "response": rag_result["response"],
                    "retrieved_contexts": rag_result["contexts"],
                    "reference": reference,
                    "synthesizer_name": synthesizer,
                }
            )
        except Exception as e:
            logger.warning("  FAILED question %d: %s", idx + 1, e)
            results_rows.append(
                {
                    "user_input": question,
                    "response": f"ERROR: {e}",
                    "retrieved_contexts": [],
                    "reference": reference,
                    "synthesizer_name": synthesizer,
                }
            )

    # Build EvaluationDataset
    samples = []
    for r in results_rows:
        samples.append(
            SingleTurnSample(
                user_input=r["user_input"],
                response=r["response"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["reference"],
            )
        )
    eval_dataset = EvaluationDataset(samples=samples)

    # Run RAGAS evaluate
    logger.info("Running RAGAS evaluation with 4 metrics...")
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
    ]

    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=RunConfig(timeout=360),
    )

    # Extract aggregate + per-sample scores from EvaluationResult
    aggregate = {}
    try:
        result_df = ragas_result.to_pandas()
        metric_cols = [
            c
            for c in result_df.columns
            if c not in ("user_input", "response", "retrieved_contexts", "reference")
        ]

        # Compute aggregate as mean of each metric column
        for col in metric_cols:
            values = pd.to_numeric(result_df[col], errors="coerce").dropna()
            if len(values) > 0:
                aggregate[col] = float(values.mean())

        # Attach per-sample scores
        for col in metric_cols:
            for i, val in enumerate(result_df[col]):
                if i < len(results_rows):
                    results_rows[i][col] = float(val) if pd.notna(val) else None
    except Exception as e:
        logger.warning("Could not extract scores from EvaluationResult: %s", e)
        # Fallback: try ragas_result.scores (list of dicts)
        if hasattr(ragas_result, "scores") and ragas_result.scores:
            from collections import defaultdict
            totals = defaultdict(list)
            for i, score_dict in enumerate(ragas_result.scores):
                for key, val in score_dict.items():
                    if isinstance(val, (int, float)):
                        totals[key].append(val)
                        if i < len(results_rows):
                            results_rows[i][key] = float(val)
            for key, vals in totals.items():
                aggregate[key] = sum(vals) / len(vals)

    return {
        "aggregate_scores": aggregate,
        "per_sample": results_rows,
    }


# ---------------------------------------------------------------------------
# Per-query-type breakdown
# ---------------------------------------------------------------------------

def _compute_query_type_breakdown(per_sample: list[dict]) -> dict:
    """Compute average metric scores broken down by query type.

    Maps synthesizer_name -> user-facing category via QUERY_TYPE_MAP,
    then averages each metric within each category.
    """
    from evaluation.config import QUERY_TYPE_MAP, RAGAS_METRIC_NAMES

    # Possible metric column names (lowercased versions of what RAGAS outputs)
    metric_cols = set()
    for sample in per_sample:
        for key in sample:
            if key not in (
                "user_input",
                "response",
                "retrieved_contexts",
                "reference",
                "synthesizer_name",
            ):
                metric_cols.add(key)

    # Group by query type
    by_type: dict[str, list[dict]] = {}
    for sample in per_sample:
        synth = sample.get("synthesizer_name", "")
        qtype = QUERY_TYPE_MAP.get(synth, "unknown")
        by_type.setdefault(qtype, []).append(sample)

    # Average metrics per type
    breakdown = {}
    for qtype, samples in by_type.items():
        breakdown[qtype] = {"count": len(samples)}
        for col in metric_cols:
            values = [s[col] for s in samples if s.get(col) is not None]
            if values:
                breakdown[qtype][col] = sum(values) / len(values)

    return breakdown


# ---------------------------------------------------------------------------
# Orchestrator: run both evaluations
# ---------------------------------------------------------------------------

def run_evaluation(config) -> Path:
    """Run full evaluation (LangSmith + RAGAS) on a pipeline.

    Steps:
      1. Set LangSmith project for tracing
      2. Set up pipeline (index documents)
      3. Build RAG chain + rag_fn
      4. Part A: LangSmith evaluate() if dataset name provided
      5. Part B: RAGAS evaluate()
      6. Compute per-query-type breakdown
      7. Save combined results JSON
      8. Print summary table

    Returns:
        Path to saved results JSON.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from evaluation.config import RAGAS_EVALUATOR_MODEL

    # Set LangSmith project for tracing
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
    logger.info("LangSmith project: %s", config.langsmith_project)

    # Step 1: Set up pipeline
    if config.pipeline_name == "advanced":
        retrieval_tool = _setup_advanced_pipeline()
    else:
        retrieval_tool = _setup_basic_pipeline()

    max_ctx = getattr(config, "max_context_chars", 0)
    chain_invoke, rag_fn = build_rag_chain(
        retrieval_tool, top_k=config.top_k, max_context_chars=max_ctx
    )
    if max_ctx:
        logger.info("  Context budget: %d chars max", max_ctx)

    # Step 2: Load testset
    logger.info("Loading testset from %s ...", config.testset_csv)
    testset_df = pd.read_csv(config.testset_csv)
    logger.info("  %d test samples loaded", len(testset_df))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Part A: LangSmith evaluation (if dataset name provided)
    langsmith_result = {}
    if config.langsmith_dataset_name:
        langsmith_result = run_langsmith_evaluation(
            chain_invoke,
            dataset_name=config.langsmith_dataset_name,
            revision_id=f"{config.pipeline_name}_pipeline",
        )

    # Part B: RAGAS evaluation
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=RAGAS_EVALUATOR_MODEL))
    ragas_result = run_ragas_evaluation(rag_fn, testset_df, evaluator_llm)

    # Compute per-query-type breakdown
    per_query_type = _compute_query_type_breakdown(ragas_result["per_sample"])

    # Build combined result
    result_dict = {
        "pipeline": config.pipeline_name,
        "timestamp": timestamp,
        "langsmith_project": config.langsmith_project,
        "langsmith_dataset": config.langsmith_dataset_name,
        "langsmith_experiment": langsmith_result.get("experiment_name", ""),
        "num_samples": len(testset_df),
        "ragas_scores": ragas_result["aggregate_scores"],
        "per_query_type": per_query_type,
        "per_sample": ragas_result["per_sample"],
    }

    # Save results JSON
    output_path = config.output_dir / f"eval_{config.pipeline_name}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # Print summary
    _print_summary(result_dict)

    return output_path


def _print_summary(result: dict) -> None:
    """Print evaluation results summary to console."""
    pipeline = result["pipeline"]
    ragas = result.get("ragas_scores", {})
    per_type = result.get("per_query_type", {})

    print(f"\n{'='*60}")
    print(f"  RAGAS Evaluation Results - {pipeline} pipeline")
    print(f"{'='*60}")

    if ragas:
        print(f"\n  {'Metric':<30s} {'Score':>8s}")
        print(f"  {'-'*38}")
        for metric, score in ragas.items():
            print(f"  {metric:<30s} {score:>8.4f}")

    if per_type:
        print(f"\n  Per Query Type:")
        print(f"  {'-'*50}")
        for qtype, scores in per_type.items():
            count = scores.pop("count", "?")
            print(f"\n  {qtype} (n={count}):")
            for metric, score in scores.items():
                print(f"    {metric:<28s} {score:>8.4f}")

    ls_exp = result.get("langsmith_experiment", "")
    if ls_exp:
        print(f"\n  LangSmith experiment: {ls_exp}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS + LangSmith evaluation on a pipeline"
    )
    parser.add_argument(
        "--pipeline",
        choices=["basic", "advanced"],
        default="basic",
        help="Pipeline variant to evaluate",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default=None,
        help="Path to testset CSV",
    )
    parser.add_argument(
        "--langsmith-dataset",
        type=str,
        default=None,
        help="LangSmith dataset name for LLM-as-judge evaluation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval top_k",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="Max total context chars sent to LLM (0 = no limit)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from evaluation.config import DEFAULT_TESTSET_CSV, EVAL_OUTPUT_DIR, EvalConfig

    config = EvalConfig(
        pipeline_name=args.pipeline,
        testset_csv=Path(args.testset) if args.testset else DEFAULT_TESTSET_CSV,
        output_dir=Path(args.output_dir) if args.output_dir else EVAL_OUTPUT_DIR,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        langsmith_dataset_name=args.langsmith_dataset or "",
    )

    run_evaluation(config)


if __name__ == "__main__":
    main()
