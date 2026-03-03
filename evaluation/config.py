"""Centralized configuration for the RAGAS evaluation module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)  # Critical: Windows empty env vars issue

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

SPEC_PDF = DOCS_DIR / "Bms-Specs.pdf"
PROPOSAL_PDF = DOCS_DIR / "BMS-Proposal.pdf"

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
GENERATOR_LLM_MODEL = "gpt-4o-mini"           # SDG generation
RAGAS_EVALUATOR_MODEL = "gpt-4.1-mini"         # RAGAS metric scoring
LANGSMITH_JUDGE_MODEL = "openai:gpt-4o"        # openevals LLM-as-judge
RAG_ANSWER_MODEL = None                        # None = use project's _get_llm()
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# SDG defaults
# ---------------------------------------------------------------------------
DEFAULT_TESTSET_SIZE = 50
DEFAULT_TESTSET_CSV = EVAL_OUTPUT_DIR / "testset.csv"
DEFAULT_KG_PATH = EVAL_OUTPUT_DIR / "knowledge_graph.json"

# Query distribution: user categories -> synthesizer weights
QUERY_DISTRIBUTIONS = {
    "simple": 0.2,          # SingleHopSpecificQuerySynthesizer
    "reasoning": 0.5,       # MultiHopAbstractQuerySynthesizer
    "multi_context": 0.3,   # MultiHopSpecificQuerySynthesizer
}

# Map RAGAS synthesizer names (output column) -> user-facing query types.
# Note: RAGAS has a typo in "specifc" (missing 'i').
QUERY_TYPE_MAP = {
    "single_hop_specifc_query_synthesizer": "simple",
    "single_hop_specific_query_synthesizer": "simple",       # in case they fix it
    "multi_hop_abstract_query_synthesizer": "reasoning",
    "multi_hop_specific_query_synthesizer": "multi_context",
}

# ---------------------------------------------------------------------------
# RAGAS metrics (from ragas.metrics)
# ---------------------------------------------------------------------------
RAGAS_METRIC_NAMES = [
    "LLMContextRecall",
    "Faithfulness",
    "FactualCorrectness",
    "ResponseRelevancy",
]

# ---------------------------------------------------------------------------
# LangSmith evaluator feedback keys
# ---------------------------------------------------------------------------
LANGSMITH_EVALUATOR_KEYS = ["qa_correctness", "helpfulness"]

# ---------------------------------------------------------------------------
# Cost estimates (per 1M tokens, approximate — gpt-4.1-nano)
# ---------------------------------------------------------------------------
COST_PER_1M_INPUT_TOKENS = 0.10     # gpt-4.1-nano input
COST_PER_1M_OUTPUT_TOKENS = 0.40    # gpt-4.1-nano output

# ---------------------------------------------------------------------------
# LangSmith project naming
# ---------------------------------------------------------------------------
LANGSMITH_PROJECT_PREFIX = "EDR"


def get_langsmith_project_name(pipeline_name: str) -> str:
    """Return LangSmith project name: EDR-{pipeline_name}-{timestamp}."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{LANGSMITH_PROJECT_PREFIX}-{pipeline_name}-{ts}"


# ---------------------------------------------------------------------------
# Runtime config dataclass
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Runtime configuration for an evaluation run."""

    pipeline_name: str = "basic"
    testset_csv: Path = DEFAULT_TESTSET_CSV
    output_dir: Path = EVAL_OUTPUT_DIR
    top_k: int = 5
    max_context_chars: int = 0
    langsmith_project: str = ""
    langsmith_dataset_name: str = ""

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.langsmith_project:
            self.langsmith_project = get_langsmith_project_name(self.pipeline_name)
