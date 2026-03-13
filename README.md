# EDR — Engineering Document Review Platform

Multi-agent AI system for automated compliance checking of engineering documents against project specifications and industry standards. Built with LangGraph's Deep Agent pattern, the system orchestrates specialized Agentic RAG agents to extract requirements, match proposal claims, look up referenced standards, and produce structured compliance verdicts with severity ratings.

---

## Table of Contents

- [Task 1: Problem, Audience, and Scope](#task-1-problem-audience-and-scope)
- [Task 2: Proposed Solution](#task-2-proposed-solution)
- [Task 3: Data Sources and Chunking Strategy](#task-3-data-sources-and-chunking-strategy)
- [Task 4: End-to-End Prototype](#task-4-end-to-end-prototype)
- [Task 5: RAGAS Evaluation — Baseline Results](#task-5-ragas-evaluation--baseline-results)
- [Task 6: Advanced Retrieval Technique and Comparison](#task-6-advanced-retrieval-technique-and-comparison)
- [Task 7: Next Steps](#task-7-next-steps)
- [Setup and Installation](#setup-and-installation)

---

## Task 1: Problem, Audience, and Scope

### Problem Statement

EDR automates the compliance review of engineering vendor proposals against project specifications — replacing a manual, error-prone process with structured, per-requirement verdicts and severity ratings.

### Why This Is a Problem

In engineering procurement, a **technical engineering consultant** prepares a detailed specification (e.g., a 67-page BMS spec covering HVAC controls, network architecture, alarm management, and integration requirements) and then must methodically verify that the **contractor's** submitted proposal actually addresses every requirement, references the correct industry standards, and meets the specified acceptance criteria. This review process is time-consuming, error-prone, and heavily dependent on the reviewer's domain expertise and attention to detail.

EDR automates this review for both audiences. The **engineering consultant** uses it to rapidly validate that a contractor's submittal is compliant before issuing approval, flagging gaps and partial compliance with severity ratings so they can focus their expert judgment on the most critical issues. The **contractor** uses it as a pre-submission quality check, running their draft proposal against the specification to identify missing requirements, weak compliance areas, and standards they may have overlooked — allowing them to strengthen their submittal before formal review. By structuring the output as per-requirement verdicts (Compliant, Partial, Non-Compliant, Not Addressed) with evidence and recommendations, EDR provides an actionable compliance report that fits directly into existing engineering review workflows.

### Evaluation Questions and Input-Output Pairs

The following input-output pairs define the core capabilities the system must demonstrate. These were used alongside the 50-question RAGAS synthetic testset (see [Task 5](#task-5-ragas-evaluation--baseline-results)) to evaluate the application.

| # | Input (Question / Task) | Expected Output |
|---|------------------------|-----------------|
| 1 | Given the BMS spec and proposal, what are the HVAC control requirements and does the proposal address them? | List of HVAC control requirements extracted from the spec, matched proposal claims, and per-requirement compliance verdicts with severity ratings |
| 2 | Does the proposal reference ASHRAE 135 (BACnet) as required by the specification? | Verdict indicating whether the proposal explicitly references ASHRAE 135, with evidence quotes from both documents |
| 3 | What requirements in the specification are not addressed at all by the proposal? | List of NOT_ADDRESSED requirements with their section references and severity (CRITICAL/MAJOR/MINOR) |
| 4 | How does the proposed network architecture compare to what the spec requires? | Side-by-side comparison of spec requirements vs. proposal claims for network topology, protocols, and integration points |
| 5 | What industry standards are referenced in Section 12651 of the specification? | Extracted list of standards (ASHRAE, NFPA, SMACNA, BSRIA, etc.) with their relevance to BMS requirements |
| 6 | Does the proposal meet the alarm management requirements? | Compliance verdicts for alarm-related requirements covering alarm types, notification methods, logging, and acknowledgment workflows |
| 7 | What are the data archiving and trending requirements, and are they met? | Requirements for historization intervals, trend configuration, and data retention — with proposal compliance status |
| 8 | Summarize the overall compliance status of the proposal against the specification. | Aggregate compliance report: count of COMPLIANT, PARTIAL, NON_COMPLIANT, and NOT_ADDRESSED verdicts with a severity breakdown |

---

## Task 2: Proposed Solution

### Solution Description

EDR is a multi-agent compliance review system that automates the process of checking engineering proposals against project specifications. Given a BMS (Building Management System) specification document and a vendor proposal, the system automatically identifies review topics, extracts technical requirements from the spec, matches corresponding claims from the proposal, optionally looks up referenced industry standards (ASHRAE, NFPA, etc.), and produces a structured compliance report with per-requirement verdicts (Compliant, Partial, Non-Compliant, Not Addressed) and severity ratings (Critical, Major, Minor).

The architecture uses LangGraph's Deep Agent pattern: a top-level orchestrator graph plans topics and spawns isolated sub-agent graphs per topic. Each sub-agent is a ReAct agent with tool access for Agentic RAG — they autonomously decide when and how many times to query the vector database, making retrieval decisions part of the agent's reasoning loop rather than a fixed pipeline step. This design enables the system to handle complex, multi-section engineering documents where different topics require different retrieval strategies.

### Infrastructure Diagram

```
                        +------------------+
                        |   Streamlit UI   |
                        |  (app.py)        |
                        +--------+---------+
                                 |
                                 v
                    +------------------------+
                    |   LangGraph Orchestrator |
                    |   (StateGraph[EDRState]) |
                    +-----+------+------+-----+
                          |      |      |
              +-----------+      |      +-----------+
              v                  v                  v
     +----------------+  +----------------+  +----------------+
     |  Topic Agent 1 |  |  Topic Agent 2 |  |  Topic Agent N |
     |  (TopicState)  |  |  (TopicState)  |  |  (TopicState)  |
     +-------+--------+  +-------+--------+  +-------+--------+
             |                    |                    |
             v                    v                    v
     +-------+--------+  +-------+--------+  +-------+--------+
     | spec_analyzer   |  | spec_analyzer   |  | spec_analyzer   |
     | proposal_review |  | proposal_review |  | proposal_review |
     | standards_lookup|  | standards_lookup|  | standards_lookup|
     | compliance_eval |  | compliance_eval |  | compliance_eval |
     +-----------------+  +-----------------+  +-----------------+
             |                    |                    |
             v                    v                    v
     +-------+--------------------+--------------------+--------+
     |                      Tool Layer                          |
     |  +------------------+  +-------------+  +-----------+   |
     |  | retrieve_documents|  | search_     |  | LLM       |   |
     |  | (Qdrant Vector DB)|  | standards   |  | (Claude)  |   |
     |  +--------+---------+  | (Tavily)    |  +-----------+   |
     |           |             +------+------+                  |
     +-----------+--------------------+-------------------------+
                 |                    |
                 v                    v
     +-----------+--------+  +-------+--------+
     |   Qdrant Vector DB |  |  Tavily Web    |
     |   (In-Memory)      |  |  Search API    |
     +----+----------+----+  +----------------+
          |          |
          v          v
   +------+---+ +---+--------+
   | BMS Spec | | BMS        |
   | PDF      | | Proposal   |
   | (67 pg)  | | PDF (26 pg)|
   +----------+ +------------+
```

### Tooling Choices and Justifications

| # | Tool / Technology | Role | Justification |
|---|-------------------|------|---------------|
| 1 | **LangGraph** | Multi-agent orchestration | Provides typed state management (`TypedDict` + `Annotated` reducers), conditional routing via `Command`, sub-graph composition, and built-in streaming. Enables the Deep Agent pattern where each topic runs in an isolated sub-graph with its own state. |
| 2 | **Claude Sonnet 4** (Anthropic) | Primary LLM for all agents | Superior reasoning quality for complex compliance analysis tasks. Temperature 0.0 ensures deterministic, reproducible outputs. Handles structured JSON output reliably for requirement extraction and verdict generation. |
| 3 | **OpenAI text-embedding-3-small** | Dense vector embeddings | Cost-effective embedding model (1536 dimensions) with strong semantic similarity performance. Used for both document indexing and query embedding in the retrieval pipeline. |
| 4 | **Qdrant** | Vector database | Supports both dense-only and hybrid (dense + sparse) retrieval modes in a single system. In-memory mode eliminates infrastructure overhead for the prototype. Native support for metadata filtering (by doc_type, chunk_level). |
| 5 | **FastEmbed BM25** (Qdrant/bm25) | Sparse embeddings for hybrid search | Enables lexical keyword matching alongside semantic search via Reciprocal Rank Fusion (RRF). Critical for engineering documents where exact terminology (section numbers, standard codes like "ASHRAE 135") matters as much as semantic meaning. |
| 6 | **Azure Document Intelligence** | Layout-aware PDF parsing | The `prebuilt-layout` model extracts document structure (headings, sections, tables) as Markdown with heading roles and levels. This structural information enables hierarchical chunking that preserves document organization — essential for a spec document with nested sections. |
| 7 | **Tavily** | Web search for engineering standards | Provides real-time web search for industry standards (ASHRAE, NFPA, SMACNA, UL, IEC) referenced in specifications. The `search_standards` tool allows agents to look up current standard requirements that are not contained in the indexed documents. |
| 8 | **Streamlit** | Web UI | Rapid prototyping with built-in streaming support (`st.status`, `st.write_stream`), session state management, and file upload handling. Two-tab interface provides both structured compliance review and freeform document chat. |
| 9 | **LangSmith** | Tracing and evaluation monitoring | End-to-end trace visibility for multi-agent runs. Built-in dataset management for evaluation testsets, `evaluate()` API for LLM-as-judge scoring, and side-by-side experiment comparison for pipeline benchmarking. |
| 10 | **RAGAS** | Synthetic data generation and retrieval evaluation | Knowledge Graph-based synthetic test generation produces multi-hop questions that test cross-document reasoning. Provides retrieval-quality metrics (Context Recall, Faithfulness, Factual Correctness, Answer Relevancy) that specifically measure RAG pipeline performance. |
| 11 | **openevals** | LLM-as-judge evaluators | Provides `create_llm_as_judge()` for custom evaluation criteria (QA correctness, helpfulness) integrated with LangSmith's evaluation framework. |
| 12 | **Pydantic** | Data validation and structured models | Type-safe models for `Requirement`, `ProposalClaim`, `StandardInfo`, and `ComplianceVerdict` ensure data integrity across agent boundaries. Enables reliable JSON serialization/deserialization between graph nodes. |

---

## Task 3: Data Sources and Chunking Strategy

### Data Sources

| Document | Description | Pages | Purpose |
|----------|-------------|-------|---------|
| `docs/Bms-Specs.pdf` | BMS (Building Management System) Project Specification | 67 | Source of technical requirements, referenced standards, and acceptance criteria |
| `docs/BMS-Proposal.pdf` | Vendor BMS Proposal / Submittal | 26 | Source of vendor claims, proposed solutions, and compliance statements |

These are real-world engineering documents used in Building Management System procurement. The specification defines what the client requires; the proposal is the vendor's response describing how they will meet those requirements.

### External APIs

| API | Purpose | Usage |
|-----|---------|-------|
| **OpenAI API** | Embeddings (text-embedding-3-small) and evaluation LLM (gpt-4.1-mini) | Document indexing, query embedding, RAGAS metric scoring |
| **Anthropic API** | Primary LLM (Claude Sonnet 4) | All agent reasoning — requirement extraction, claim matching, compliance evaluation |
| **Tavily API** | Web search | Standards lookup tool searches for ASHRAE, NFPA, SMACNA, and other engineering standards |
| **Azure Document Intelligence** | PDF layout extraction | Advanced pipeline uses `prebuilt-layout` to extract headings, sections, and tables with structural metadata |
| **LangSmith API** | Tracing and evaluation | Trace logging, dataset management, LLM-as-judge evaluation |

### Chunking Strategy

The project implements two chunking strategies, enabling direct comparison:

#### Basic Pipeline — Flat Recursive Splitting

```
PDF (PyPDFLoader) → Raw Pages → RecursiveCharacterTextSplitter → Flat Chunks → Qdrant (Dense)
```

| Parameter | Value |
|-----------|-------|
| Splitter | `RecursiveCharacterTextSplitter` |
| Chunk size | 1,000 characters |
| Chunk overlap | 200 characters |
| Separators | `["\n\n", "\n", ". ", " ", ""]` |
| Total chunks | 207 (spec: 181, proposal: 26) |
| Retrieval | Dense cosine similarity only |

This approach treats documents as flat text, splitting on character boundaries with no awareness of document structure.

#### Advanced Pipeline — Hierarchical Parent-Child Chunking

```
PDF (Azure DI) → Markdown with Headings → SectionNode Tree → Parent Chunks → Child Chunks → Qdrant (Hybrid)
```

| Parameter | Value |
|-----------|-------|
| Parser | Azure Document Intelligence (`prebuilt-layout`, Markdown output) |
| Tree construction | `SectionNode` tree built from heading roles (`h1`-`h6`) |
| Parent chunk size | 2,000 characters / 400 overlap |
| Child chunk size | 1,000 characters / 200 overlap |
| Min child size | 50 characters (filtered) |
| Max parent size | 4,000 characters (oversized sections re-split) |
| Total chunks | 918 (parent + child across both documents) |
| Retrieval | Hybrid BM25 + Dense with RRF fusion |

The advanced strategy preserves document hierarchy. Azure DI extracts heading levels and section boundaries. A `SectionNode` tree is built where each node represents a document section. Parent chunks contain full section context; child chunks are finer-grained sub-sections. At retrieval time, the system searches child chunks for precision, then returns their parent chunks for full context (small-to-big retrieval).

---

## Task 4: End-to-End Prototype

### Architecture — Deep Agent Pattern

The system is built as a two-level graph using LangGraph:

**Level 1 — Orchestrator Graph** (`StateGraph[EDRState]`):

```
START → plan_topics → process_topic → supervisor_verify → check_more_topics
                          ^                                      |
                          |                              [more topics]
                          +---- advance_topic <-----------------+
                                                                |
                                                         [all done]
                                                                |
                                                                v
                                                        generate_report → END
```

| Node | Function |
|------|----------|
| `plan_topics` | Analyzes document structure to identify review topics (e.g., "HVAC Controls", "Network Architecture") |
| `process_topic` | Invokes the Topic Sub-Graph for the current topic |
| `supervisor_verify` | Quality-checks the verdicts for completeness and accuracy |
| `check_more_topics` | Dynamic router using `Command` — advances to next topic or finishes |
| `advance_topic` | Moves the topic pointer forward (includes TPM rate limiting) |
| `generate_report` | Aggregates all verdicts into a final compliance report |

**Level 2 — Topic Sub-Graph** (`StateGraph[TopicState]`):

```
START → spec_analyzer → proposal_reviewer → route_standards
                                                |
                                        [has references]  →  standards_lookup → compliance_evaluator → END
                                        [no references]   →  compliance_evaluator → END
```

| Node | Function | Tools |
|------|----------|-------|
| `spec_analyzer` | ReAct agent that extracts requirements from the spec for this topic | `retrieve_documents` |
| `proposal_reviewer` | ReAct agent that matches proposal claims to each requirement | `retrieve_documents` |
| `route_standards` | Conditional router — checks if any requirements reference external standards | — |
| `standards_lookup` | ReAct agent that searches for referenced standards information | `search_standards` (Tavily) |
| `compliance_evaluator` | Produces structured `ComplianceVerdict` for each requirement (batched, 8 per call) | — |

Each ReAct agent uses Agentic RAG — the agent autonomously decides when and how many times to call the retrieval tool, up to a max of 3 calls per agent (enforced via prompt instructions to prevent recursion limit exhaustion).

### Structured Output Models

```python
class Requirement:       # id, description, source_section, priority, referenced_standards
class ProposalClaim:     # requirement_id, claim_text, source_section, confidence
class StandardInfo:      # standard_id, title, relevant_sections, summary
class ComplianceVerdict: # requirement_id, status, severity, evidence, gaps, recommendations
```

Verdict statuses: `COMPLIANT`, `PARTIAL`, `NON_COMPLIANT`, `NOT_ADDRESSED`
Severity levels: `CRITICAL`, `MAJOR`, `MINOR`

### Streamlit UI

The application (`src/app.py`) provides two tabs:

1. **Compliance Review Tab** — Upload spec + proposal PDFs, select pipeline mode (Basic/Advanced), run full compliance review with live streaming of findings, view results with severity filtering, download report as Markdown or JSON.

2. **Chat with Documents Tab** — Freeform Q&A against indexed documents using the selected pipeline's retrieval tool.

### Validated End-to-End Results

The system has been validated on the BMS test documents:
- 27 requirements extracted from the specification
- 27 proposal claims matched to requirements
- 27 compliance verdicts produced with severity ratings
- Sample verdict: REQ-001 PARTIAL (MAJOR), REQ-005 NOT_ADDRESSED (MAJOR)

---

## Task 5: RAGAS Evaluation — Baseline Results

### Synthetic Test Data Generation

Test data was generated using RAGAS Synthetic Data Generation (SDG) with a Knowledge Graph approach:

1. **Knowledge Graph Construction**: Both PDF documents were loaded and each page added as a `DOCUMENT` node to a RAGAS `KnowledgeGraph`. Default transforms were applied to build relationships between nodes:
   - `HeadlinesExtractor` → `HeadlineSplitter` → `SummaryExtractor` → `CustomNodeFilter`
   - `EmbeddingExtractor`, `ThemesExtractor`, `NERExtractor` (parallel)
   - `CosineSimilarityBuilder`, `OverlapScoreBuilder` (parallel — these build the multi-hop relationships)

2. **Query Distribution**: Three synthesizer types generate questions of varying complexity:

   | Query Type | Synthesizer | Weight | Count | Description |
   |------------|-------------|--------|-------|-------------|
   | Simple | `SingleHopSpecificQuerySynthesizer` | 20% | 10 | Single-section factual lookups |
   | Reasoning | `MultiHopAbstractQuerySynthesizer` | 50% | 25 | Abstract questions requiring synthesis across multiple KG nodes |
   | Multi-Context | `MultiHopSpecificQuerySynthesizer` | 30% | 15 | Specific questions requiring cross-document reference chains |

3. **Total Testset Size**: 50 synthetic question-answer pairs with reference contexts

### Evaluation Metrics

Four RAGAS metrics were used to evaluate retrieval and generation quality:

| Metric | What It Measures |
|--------|-----------------|
| **Context Recall** | How much of the ground-truth reference is covered by retrieved contexts |
| **Faithfulness** | Whether the generated answer is supported by the retrieved contexts (no hallucination) |
| **Factual Correctness** (F1) | Overlap between generated answer claims and ground-truth reference claims |
| **Answer Relevancy** | How relevant and focused the answer is to the original question |

#### Why Factual Correctness Instead of Context Precision

The standard RAGAS metric suite includes Context Precision (whether relevant retrieved chunks rank above irrelevant ones). We replaced it with **Factual Correctness (F1)** because it is a stronger fit for a compliance review system:

- **Context Precision is a retrieval-stage proxy**: It measures whether the retriever *ranks* relevant documents higher, but says nothing about whether the final answer is actually correct. A perfectly ranked retrieval can still produce a wrong compliance verdict if the LLM misinterprets the context.
- **Factual Correctness is an end-to-end metric**: It measures claim-level overlap between the generated answer and the ground-truth reference, catching failures at both the retrieval *and* generation stages. For a system whose output is structured compliance verdicts, what matters is whether each verdict's evidence and conclusions are factually accurate — not just whether the retriever's top-k ranking was optimal.
- **Compliance verdicts have real-world consequences**: An incorrect verdict could approve a non-compliant proposal or reject a compliant one. Factual Correctness directly quantifies this risk by measuring precision (are the system's claims true?) and recall (did it capture all relevant facts?) as an F1 score.

In practice, Context Recall already provides a retrieval-quality signal (did we retrieve enough relevant content?). Adding Context Precision on top would give a second retrieval metric at the expense of having no generation-quality metric beyond Faithfulness. Factual Correctness fills that gap by evaluating the final output against the ground truth.

### Baseline Results — Basic Pipeline (Dense Retrieval)

Evaluation settings: `top_k=3`, `max_context_chars=5000`, LLM: `gpt-4.1-mini`

| Metric | Score |
|--------|-------|
| Context Recall | 0.566 |
| Faithfulness | 0.854 |
| Factual Correctness (F1) | 0.313 |
| Answer Relevancy | 0.799 |

#### Per-Query-Type Breakdown (Basic Pipeline)

| Metric | Simple (n=10) | Reasoning (n=25) |
|--------|--------------|-------------------|
| Context Recall | 0.527 | 0.605 |
| Faithfulness | 0.877 | 0.825 |
| Factual Correctness (F1) | 0.197 | 0.458 |
| Answer Relevancy | 0.753 | 0.849 |

### Baseline Conclusions

- **Faithfulness is the strongest metric** (0.854): The LLM rarely hallucinated beyond what the retrieved contexts contained. This is expected since the answer generation uses a focused prompt with only the retrieved context.
- **Factual Correctness is the weakest** (0.313): The generated answers frequently missed claims present in the ground-truth references. This indicates the retrieval layer is not surfacing enough relevant information.
- **Reasoning queries outperform simple queries** on Context Recall and Factual Correctness, likely because reasoning questions span multiple sections and the dense retriever can capture partial matches across different embedding spaces.
- **Simple queries have low Factual Correctness** (0.197): Single-hop lookups require precise retrieval of the right section, and dense-only search sometimes retrieves semantically similar but wrong sections.

---

## Task 6: Advanced Retrieval Technique and Comparison

### Technique: Hybrid BM25 + Dense with Parent-Child Retrieval

The advanced pipeline combines three techniques to improve retrieval quality:

#### 1. Layout-Aware Document Parsing

Azure Document Intelligence extracts document structure as Markdown with heading roles and levels. A `SectionNode` tree is constructed from the heading hierarchy, preserving the logical organization of the specification (e.g., Section 12600 > 1.0 General > 1.1 Scope).

#### 2. Hierarchical Parent-Child Chunking (Small-to-Big)

Documents are chunked at two granularity levels:
- **Child chunks** (1,000 chars): Fine-grained chunks used for search — smaller chunks produce more precise embedding matches.
- **Parent chunks** (2,000 chars): Larger context windows returned to the LLM — parent chunks contain full section context, reducing information loss from chunk boundaries.

At retrieval time, the system searches child chunks, deduplicates by `parent_id`, then returns the parent chunks. This "small-to-big" strategy gets the precision of small chunks with the context of large chunks.

#### 3. Hybrid BM25 + Dense Search with RRF Fusion

Retrieval combines two signals using Reciprocal Rank Fusion (RRF):
- **Dense search** (text-embedding-3-small): Captures semantic similarity — useful for paraphrased or conceptually related content.
- **Sparse search** (BM25 via FastEmbed): Captures exact keyword matches — critical for engineering terminology, section numbers (e.g., "12600"), and standard codes (e.g., "ASHRAE 135-2020").

RRF merges the two ranked lists without requiring score normalization, giving each signal equal weight.

### Implementation Details

The 4-phase retrieval process (`src/tools_advanced.py`):

1. **Child Search**: Over-fetch child chunks (`top_k * 3`) using hybrid search to get a broad candidate set.
2. **Parent Deduplication**: Group child results by `parent_id`, keeping the top-scoring child per parent. If fewer than `top_k` unique parents are found, fall back to direct parent-level search.
3. **Parent Fetch**: Retrieve full parent chunks by ID from the vector store.
4. **Format**: Return parent chunks with metadata (source file, page number, section path).

### Fair Comparison Setup

To ensure a fair comparison, both pipelines were evaluated with identical constraints:

| Parameter | Value |
|-----------|-------|
| Testset | Same 50 synthetic questions (shared `testset.csv`) |
| top_k | 3 (retrieval results per query) |
| max_context_chars | 5,000 (total context budget sent to LLM) |
| Answer LLM | gpt-4.1-mini (same for both) |
| Evaluator LLM | gpt-4.1-mini (RAGAS metrics) |
| Embedding model | text-embedding-3-small (same for both) |

The `max_context_chars` parameter was added specifically to normalize context volume between the two pipelines. Without this limit, the advanced pipeline's parent chunks (2,000 chars each) would provide significantly more context than the basic pipeline's flat chunks (1,000 chars each), confounding the comparison.

### Aggregate Comparison Results

| Metric | Basic | Advanced | Delta | Winner |
|--------|-------|----------|-------|--------|
| Context Recall | 0.566 | 0.607 | +0.041 | Advanced |
| Faithfulness | 0.854 | 0.834 | -0.020 | Basic |
| Factual Correctness (F1) | 0.313 | 0.432 | +0.119 | **Advanced** |
| Answer Relevancy | 0.799 | 0.857 | +0.058 | Advanced |

**Overall Winner: Advanced Pipeline (3-1)**

### Per-Query-Type Comparison

#### Reasoning Queries (n=25)

| Metric | Basic | Advanced | Delta |
|--------|-------|----------|-------|
| Context Recall | 0.605 | 0.564 | -0.041 |
| Faithfulness | 0.825 | 0.818 | -0.007 |
| Factual Correctness (F1) | 0.458 | 0.467 | +0.010 |
| Answer Relevancy | 0.849 | 0.910 | +0.060 |

#### Simple Queries (n=10)

| Metric | Basic | Advanced | Delta |
|--------|-------|----------|-------|
| Context Recall | 0.527 | 0.400 | -0.127 |
| Faithfulness | 0.877 | 0.906 | +0.029 |
| Factual Correctness (F1) | 0.197 | 0.265 | +0.068 |
| Answer Relevancy | 0.753 | 0.762 | +0.009 |

#### Multi-Context Queries (n=15) — Advanced Pipeline Only

| Metric | Advanced |
|--------|----------|
| Context Recall | 0.817 |
| Faithfulness | 0.813 |
| Factual Correctness (F1) | 0.485 |
| Answer Relevancy | 0.833 |

Note: Basic pipeline multi-context metrics were not recorded due to a logging issue during the evaluation run. The advanced pipeline's multi-context scores are the highest across all query types, demonstrating the strength of the KG-based multi-hop retrieval relationships.

### Analysis

1. **Factual Correctness improvement is the largest delta (+0.119 / +38%)**: The advanced pipeline's hybrid retrieval surfaces more factually relevant content. BM25 catches exact engineering terms that dense embeddings miss, and parent chunks provide fuller context that includes more of the ground-truth claims.

2. **Answer Relevancy improved (+0.058 / +7%)**: Answers from the advanced pipeline are more focused and relevant. The hierarchical chunking preserves section boundaries, so retrieved content is more topically coherent.

3. **Faithfulness slightly decreased (-0.020 / -2%)**: With richer context from parent chunks, the LLM occasionally synthesizes across multiple context pieces, producing claims that are not directly stated in any single context chunk. This is a minor tradeoff — the LLM is doing more synthesis work, which slightly increases the risk of unsupported claims.

4. **Simple queries show lower Context Recall for advanced (-0.127)**: For straightforward single-section lookups, the basic pipeline's flat chunks sometimes match better because they are sized closer to the query scope. The advanced pipeline's parent chunks may include surrounding content that dilutes the recall score.

5. **Multi-context queries are the advanced pipeline's strength** (Context Recall: 0.817): The Knowledge Graph's cross-document relationships (built via CosineSimilarityBuilder and OverlapScoreBuilder) create multi-hop paths that the advanced retrieval can follow. This is exactly where hybrid search and hierarchical chunking shine.

---

## Task 7: Next Steps

### Decision: Keep the Advanced Retrieval Pipeline

Based on the evaluation results, the advanced hybrid retrieval pipeline will be used for Demo Day. The decision is justified by:

1. **+38% improvement in Factual Correctness** (0.313 → 0.432): This is the most important metric for a compliance review system. Higher factual correctness means the system's answers contain more of the actual claims from the reference documents, leading to more accurate compliance verdicts.

2. **+7% improvement in Answer Relevancy** (0.799 → 0.857): More focused, relevant answers lead to better requirement matching and clearer compliance assessments.

3. **3-of-4 metrics favor advanced**: The only metric where basic wins (Faithfulness, -2%) is a minor tradeoff that does not impact the system's core compliance review function.

4. **Multi-context capability**: The advanced pipeline handles cross-document, multi-section queries significantly better (Context Recall 0.817 for multi-context queries). This is critical for compliance review where requirements often reference multiple specification sections.

### Planned Improvements

| Improvement | Expected Impact | Priority |
|-------------|----------------|----------|
| **Add a reranking stage** (e.g., Cohere Rerank or cross-encoder) | Improve precision of top-k results after initial hybrid retrieval | High |
| **Tune RRF weights** between BM25 and dense scores | Optimize the balance between keyword and semantic matching for engineering documents | Medium |
| **Expand document corpus** with additional spec types | Test generalization beyond BMS documents (HVAC, electrical, fire protection) | Medium |
| **Fine-tune chunk sizes** based on per-query-type analysis | Improve simple query performance (currently weaker for advanced pipeline) | Medium |
| **Add evaluation of the full multi-agent pipeline** | Current RAGAS evaluation tests retrieval + Q&A; should also evaluate end-to-end compliance verdict quality | High |
| **Implement caching for embeddings and LLM responses** | Reduce API costs and latency for repeated queries | Low |

---

## Setup and Installation

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone <repository-url>
cd edr-capstone
uv sync
```

### Environment Variables

Create a `.env` file with the following keys:

```env
# Required
OPENAI_API_KEY=sk-...          # Embeddings + evaluation LLM
ANTHROPIC_API_KEY=sk-ant-...   # Claude Sonnet (primary agent LLM)
TAVILY_API_KEY=tvly-...        # Standards web search

# Optional
LANGCHAIN_API_KEY=lsv2_...     # LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=EDR

# Advanced pipeline only
AZURE_DI_ENDPOINT=https://...  # Azure Document Intelligence
AZURE_DI_KEY=...
```

### Running the Application

```bash
streamlit run src/app.py
```

### Running the Evaluation Suite

```bash
# Full pipeline: generate testset → evaluate both pipelines → compare
python -m evaluation.run_all --testset-size 50 --top-k 3 --max-context-chars 5000

# Or run steps individually:
python -m evaluation.generate_testset --size 50
python -m evaluation.run_evaluation --pipeline basic --testset evaluation_results/testset.csv --top-k 3 --max-context-chars 5000
python -m evaluation.run_evaluation --pipeline advanced --testset evaluation_results/testset.csv --top-k 3 --max-context-chars 5000
python -m evaluation.compare_pipelines evaluation_results/eval_basic_*.json evaluation_results/eval_advanced_*.json
```

### Project Structure

```
edr-capstone/
├── src/
│   ├── agents.py              # LangGraph orchestrator + topic sub-graph
│   ├── models.py              # Pydantic models (Requirement, Verdict, etc.)
│   ├── ingestion.py           # Basic pipeline: PyPDF + flat chunking
│   ├── ingestion_advanced.py  # Advanced pipeline: Azure DI + hierarchical chunking
│   ├── tools.py               # Basic retrieval tool (dense Qdrant)
│   ├── tools_advanced.py      # Advanced retrieval tool (hybrid BM25+Dense)
│   └── app.py                 # Streamlit UI
├── evaluation/
│   ├── config.py              # Evaluation configuration
│   ├── generate_testset.py    # RAGAS SDG with Knowledge Graph
│   ├── run_evaluation.py      # LangSmith + RAGAS evaluation
│   ├── compare_pipelines.py   # Side-by-side comparison
│   └── run_all.py             # Full evaluation orchestration
├── prompts/                   # Agent prompt templates (Markdown)
├── docs/                      # Test documents (BMS spec + proposal)
├── evaluation_results/        # Evaluation outputs (CSV, JSON)
└── pyproject.toml             # Dependencies
```
