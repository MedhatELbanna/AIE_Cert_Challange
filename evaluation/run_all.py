"""Run the full RAGAS evaluation pipeline: generate, evaluate, compare.

Steps:
  1. Generate testset with Knowledge Graph + upload to LangSmith
  2. Evaluate basic pipeline (LangSmith + RAGAS)
  3. Evaluate advanced pipeline (LangSmith + RAGAS)
  4. Compare both and print report

Usage:
  python -m evaluation.run_all --testset-size 50
  python -m evaluation.run_all --testset evaluation_results/testset.csv --skip-generate
  python -m evaluation.run_all --testset-size 10 --skip-advanced
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run full RAGAS evaluation pipeline"
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=None,
        help="Number of test samples to generate",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default=None,
        help="Path to existing testset CSV (implies --skip-generate)",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default=None,
        help="Path to existing Knowledge Graph JSON",
    )
    parser.add_argument(
        "--langsmith-dataset",
        type=str,
        default=None,
        help="Existing LangSmith dataset name (skips upload)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip testset generation",
    )
    parser.add_argument(
        "--skip-basic",
        action="store_true",
        help="Skip basic pipeline evaluation",
    )
    parser.add_argument(
        "--skip-advanced",
        action="store_true",
        help="Skip advanced pipeline evaluation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval top_k for evaluation",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="Max total context chars sent to LLM (0 = no limit)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from evaluation.config import DEFAULT_TESTSET_CSV, DEFAULT_TESTSET_SIZE, EVAL_OUTPUT_DIR, EvalConfig

    testset_size = args.testset_size or DEFAULT_TESTSET_SIZE
    testset_path = Path(args.testset) if args.testset else DEFAULT_TESTSET_CSV
    langsmith_dataset_name = args.langsmith_dataset or ""

    if args.testset:
        args.skip_generate = True

    # ======================================================================
    # STEP 1: Generate testset
    # ======================================================================
    if not args.skip_generate:
        logger.info("=" * 60)
        logger.info("STEP 1: Generating testset with Knowledge Graph")
        logger.info("=" * 60)

        from evaluation.generate_testset import (
            generate_testset,
            load_documents,
            upload_to_langsmith,
        )

        docs = load_documents()
        if not docs:
            logger.error("No documents found. Aborting.")
            sys.exit(1)

        kg_path = Path(args.kg_path) if args.kg_path else None
        testset_path, df = generate_testset(
            docs,
            testset_size=testset_size,
            kg_path=kg_path,
        )

        # Upload to LangSmith
        if not langsmith_dataset_name:
            langsmith_dataset_name = upload_to_langsmith(df)
            logger.info("LangSmith dataset: %s", langsmith_dataset_name)
    else:
        logger.info("Using existing testset: %s", testset_path)
        if not testset_path.exists():
            logger.error("Testset not found at %s", testset_path)
            sys.exit(1)

    # ======================================================================
    # STEP 2: Evaluate basic pipeline
    # ======================================================================
    basic_result_path = None
    if not args.skip_basic:
        logger.info("=" * 60)
        logger.info("STEP 2: Evaluating basic pipeline")
        logger.info("=" * 60)

        from evaluation.run_evaluation import run_evaluation

        config = EvalConfig(
            pipeline_name="basic",
            testset_csv=testset_path,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            langsmith_dataset_name=langsmith_dataset_name,
        )
        basic_result_path = run_evaluation(config)

    # ======================================================================
    # STEP 3: Evaluate advanced pipeline
    # ======================================================================
    advanced_result_path = None
    if not args.skip_advanced:
        logger.info("=" * 60)
        logger.info("STEP 3: Evaluating advanced pipeline")
        logger.info("=" * 60)

        from evaluation.run_evaluation import run_evaluation

        config = EvalConfig(
            pipeline_name="advanced",
            testset_csv=testset_path,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            langsmith_dataset_name=langsmith_dataset_name,
        )
        advanced_result_path = run_evaluation(config)

    # ======================================================================
    # STEP 4: Compare
    # ======================================================================
    if basic_result_path and advanced_result_path:
        logger.info("=" * 60)
        logger.info("STEP 4: Comparing pipelines")
        logger.info("=" * 60)

        from evaluation.compare_pipelines import (
            compare,
            load_results,
            print_comparison,
            save_comparison,
        )

        result_a = load_results(basic_result_path)
        result_b = load_results(advanced_result_path)
        comparison = compare(result_a, result_b)
        print_comparison(comparison)

        comparison_path = EVAL_OUTPUT_DIR / "comparison.json"
        save_comparison(comparison, comparison_path)

        if langsmith_dataset_name:
            print(
                f"\n  View LangSmith comparison: "
                f"Open LangSmith > Datasets > '{langsmith_dataset_name}' > Compare\n"
            )
    elif basic_result_path or advanced_result_path:
        path = basic_result_path or advanced_result_path
        logger.info("Only one pipeline evaluated. Results at: %s", path)
    else:
        logger.info("No pipelines evaluated.")

    logger.info("Done!")


if __name__ == "__main__":
    main()
