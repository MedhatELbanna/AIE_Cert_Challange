"""Generate synthetic Q&A test data from BMS PDF documents using RAGAS.

Uses the unrolled SDG workflow:
  1. Load documents as LangChain Documents
  2. Build a Knowledge Graph (nodes + default transforms)
  3. Generate testset with custom query distribution (single-hop, multi-hop)
  4. Upload testset to LangSmith as a managed dataset
  5. Save testset CSV + KG JSON for reuse

Usage:
  python -m evaluation.generate_testset --size 50
  python -m evaluation.generate_testset --size 50 --kg-path evaluation_results/knowledge_graph.json
  python -m evaluation.generate_testset --estimate-cost
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from uuid import uuid4

import nltk
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)  # Critical: Windows empty env vars issue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK setup (required by RAGAS internals)
# ---------------------------------------------------------------------------

def _ensure_nltk():
    """Download NLTK data if not already present."""
    for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"taggers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents() -> list:
    """Load BMS PDF documents as LangChain Document objects.

    Reuses src.ingestion.load_pdf() for both spec and proposal.
    Returns combined list of Documents.
    """
    from evaluation.config import SPEC_PDF, PROPOSAL_PDF
    from src.ingestion import load_pdf

    docs = []
    for pdf_path in [SPEC_PDF, PROPOSAL_PDF]:
        if pdf_path.exists():
            logger.info("Loading %s ...", pdf_path.name)
            loaded = load_pdf(pdf_path)
            docs.extend(loaded)
            logger.info("  -> %d pages", len(loaded))
        else:
            logger.warning("PDF not found: %s", pdf_path)
    return docs


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(docs: list, testset_size: int) -> dict:
    """Estimate LLM cost for testset generation (heuristic).

    Heuristic breakdown:
      - KG transforms: ~500 input + ~200 output tokens per document page
      - Query synthesis: ~800 input + ~400 output tokens per generated question
    """
    from evaluation.config import COST_PER_1M_INPUT_TOKENS, COST_PER_1M_OUTPUT_TOKENS

    num_pages = len(docs)

    # KG transform phase
    kg_input = num_pages * 500
    kg_output = num_pages * 200

    # Query synthesis phase
    syn_input = testset_size * 800
    syn_output = testset_size * 400

    total_input = kg_input + syn_input
    total_output = kg_output + syn_output

    cost = (total_input / 1_000_000) * COST_PER_1M_INPUT_TOKENS + \
           (total_output / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS

    return {
        "num_pages": num_pages,
        "testset_size": testset_size,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": cost,
    }


# ---------------------------------------------------------------------------
# Knowledge Graph construction
# ---------------------------------------------------------------------------

def build_knowledge_graph(docs: list, generator_llm, generator_embeddings):
    """Build a RAGAS KnowledgeGraph from LangChain documents.

    Steps:
      1. Create empty KnowledgeGraph
      2. Add each document as a DOCUMENT node
      3. Apply default transforms:
         - HeadlinesExtractor -> HeadlineSplitter -> SummaryExtractor
         - CustomNodeFilter
         - [EmbeddingExtractor, ThemesExtractor, NERExtractor] (parallel)
         - [CosineSimilarityBuilder, OverlapScoreBuilder] (parallel - builds relationships)
      4. Return enriched KG with nodes + relationships

    The relationships (cosine similarity, entity overlap) enable multi-hop
    query generation by connecting related sections across documents.
    """
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms, default_transforms

    logger.info("Building Knowledge Graph from %d document pages...", len(docs))

    # Create empty KG and add document nodes
    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )
    logger.info("  Added %d document nodes", len(kg.nodes))

    # Apply default transforms (enriches nodes + builds relationships)
    logger.info("  Applying default transforms (this may take a few minutes)...")
    trans = default_transforms(
        documents=docs,
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )
    apply_transforms(kg, trans)

    logger.info(
        "  Knowledge Graph complete: %d nodes, %d relationships",
        len(kg.nodes),
        len(kg.relationships),
    )
    return kg


# ---------------------------------------------------------------------------
# Testset generation
# ---------------------------------------------------------------------------

def generate_testset(
    docs: list,
    testset_size: int = 50,
    output_path: Path | None = None,
    kg_path: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Generate a synthetic testset from documents.

    Two-step flow:
      1. Build (or load) Knowledge Graph
      2. Generate testset with custom query distribution

    Args:
        docs: LangChain Document objects.
        testset_size: Number of Q&A pairs to generate.
        output_path: Where to save CSV. Defaults to config.DEFAULT_TESTSET_CSV.
        kg_path: Path to existing KG JSON to load instead of rebuilding.

    Returns:
        (csv_path, dataframe) tuple.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    from evaluation.config import (
        DEFAULT_KG_PATH,
        DEFAULT_TESTSET_CSV,
        EMBEDDING_MODEL,
        EVAL_OUTPUT_DIR,
        GENERATOR_LLM_MODEL,
        QUERY_DISTRIBUTIONS,
    )

    _ensure_nltk()

    if output_path is None:
        output_path = DEFAULT_TESTSET_CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize RAGAS LLM + embeddings (using LangChain wrappers)
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=GENERATOR_LLM_MODEL))
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )

    # Step 1: Build or load Knowledge Graph
    if kg_path and Path(kg_path).exists():
        logger.info("Loading existing Knowledge Graph from %s ...", kg_path)
        kg = KnowledgeGraph.load(str(kg_path))
        logger.info(
            "  Loaded: %d nodes, %d relationships",
            len(kg.nodes),
            len(kg.relationships),
        )
    else:
        kg = build_knowledge_graph(docs, generator_llm, generator_embeddings)
        # Save KG for reuse
        save_path = kg_path if kg_path else DEFAULT_KG_PATH
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        kg.save(str(save_path))
        logger.info("  Saved Knowledge Graph to %s", save_path)

    # Step 2: Create generator with pre-built KG
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
    )

    # Step 3: Custom query distribution (simple 20%, reasoning 50%, multi_context 30%)
    query_distribution = [
        (
            SingleHopSpecificQuerySynthesizer(llm=generator_llm),
            QUERY_DISTRIBUTIONS["simple"],
        ),
        (
            MultiHopAbstractQuerySynthesizer(llm=generator_llm),
            QUERY_DISTRIBUTIONS["reasoning"],
        ),
        (
            MultiHopSpecificQuerySynthesizer(llm=generator_llm),
            QUERY_DISTRIBUTIONS["multi_context"],
        ),
    ]

    logger.info(
        "Generating %d test samples (simple=%.0f%%, reasoning=%.0f%%, multi_context=%.0f%%) ...",
        testset_size,
        QUERY_DISTRIBUTIONS["simple"] * 100,
        QUERY_DISTRIBUTIONS["reasoning"] * 100,
        QUERY_DISTRIBUTIONS["multi_context"] * 100,
    )
    testset = generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution,
    )

    # Step 4: Save to CSV
    df = testset.to_pandas()
    df.to_csv(output_path, index=False)
    logger.info("Saved %d samples to %s", len(df), output_path)

    # Log distribution breakdown
    if "synthesizer_name" in df.columns:
        logger.info("Query type distribution:")
        for synth_name, count in df["synthesizer_name"].value_counts().items():
            logger.info("  %s: %d", synth_name, count)

    return output_path, df


# ---------------------------------------------------------------------------
# LangSmith dataset upload
# ---------------------------------------------------------------------------

def upload_to_langsmith(df: pd.DataFrame, dataset_name: str | None = None) -> str:
    """Upload a testset DataFrame to LangSmith as a managed dataset.

    Creates a new dataset in LangSmith with each row as an example:
      - inputs: {"question": user_input}
      - outputs: {"answer": reference}
      - metadata: {"context": reference_contexts, "synthesizer": synthesizer_name}

    Args:
        df: Testset DataFrame with user_input, reference, reference_contexts, synthesizer_name columns.
        dataset_name: Name for the LangSmith dataset. Auto-generated if None.

    Returns:
        The dataset name (for use in langsmith.evaluation.evaluate(data=...)).
    """
    from langsmith import Client

    client = Client()

    if dataset_name is None:
        dataset_name = f"EDR-SDG-{uuid4().hex[:8]}"

    langsmith_dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="EDR synthetic test data generated by RAGAS SDG",
    )

    for _, row in df.iterrows():
        client.create_example(
            inputs={"question": row.get("user_input", "")},
            outputs={"answer": row.get("reference", "")},
            metadata={
                "context": row.get("reference_contexts", ""),
                "synthesizer": row.get("synthesizer_name", ""),
            },
            dataset_id=langsmith_dataset.id,
        )

    logger.info(
        "Uploaded %d examples to LangSmith dataset '%s'",
        len(df),
        dataset_name,
    )
    return dataset_name


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point: python -m evaluation.generate_testset"""
    parser = argparse.ArgumentParser(
        description="Generate RAGAS synthetic testset from BMS documents"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=f"Number of test samples (default: {50})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default=None,
        help="Path to existing Knowledge Graph JSON (skips KG rebuild)",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Print cost estimate and exit (no API calls)",
    )
    parser.add_argument(
        "--skip-langsmith-upload",
        action="store_true",
        help="Skip uploading testset to LangSmith",
    )
    parser.add_argument(
        "--langsmith-dataset-name",
        type=str,
        default=None,
        help="Custom name for LangSmith dataset",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from evaluation.config import DEFAULT_TESTSET_SIZE

    testset_size = args.size if args.size else DEFAULT_TESTSET_SIZE

    # Load documents
    docs = load_documents()
    logger.info("Loaded %d document pages total", len(docs))

    if not docs:
        logger.error("No documents found. Check docs/ directory.")
        sys.exit(1)

    # Cost estimation mode
    if args.estimate_cost:
        est = estimate_cost(docs, testset_size)
        print(f"\n{'='*50}")
        print(f"Cost Estimate for {est['testset_size']} samples")
        print(f"  Document pages:  {est['num_pages']}")
        print(f"  Input tokens:    ~{est['estimated_input_tokens']:,}")
        print(f"  Output tokens:   ~{est['estimated_output_tokens']:,}")
        print(f"  Estimated cost:  ~${est['estimated_cost_usd']:.3f}")
        print(f"{'='*50}\n")
        return

    # Confirm before proceeding
    est = estimate_cost(docs, testset_size)
    print(f"\nEstimated cost: ~${est['estimated_cost_usd']:.3f} for {testset_size} samples")
    confirm = input("Proceed? [Y/n]: ").strip().lower()
    if confirm and confirm != "y":
        print("Aborted.")
        return

    # Generate testset
    output_path = Path(args.output) if args.output else None
    kg_path = Path(args.kg_path) if args.kg_path else None

    csv_path, df = generate_testset(
        docs,
        testset_size=testset_size,
        output_path=output_path,
        kg_path=kg_path,
    )

    # Upload to LangSmith
    if not args.skip_langsmith_upload:
        dataset_name = upload_to_langsmith(df, args.langsmith_dataset_name)
        print(f"\nLangSmith dataset: {dataset_name}")

    print(f"Testset CSV: {csv_path}")
    print(f"Samples generated: {len(df)}")


if __name__ == "__main__":
    main()
