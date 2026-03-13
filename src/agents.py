"""Deep Agent graph — Orchestrator + Topic sub-graphs with Agentic RAG agents."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from src.models import (
    ComplianceVerdict,
    EDRState,
    MatchQuality,
    Obligation,
    ProposalClaim,
    Requirement,
    Severity,
    StandardInfo,
    TopicState,
    Verdict,
)
from src.tools import search_standards

load_dotenv(override=True)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt template and substitute placeholders like {topic}."""
    path = PROMPTS_DIR / name
    text = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        text = text.replace(f"{{{key}}}", value)
    return text


def _make_trimmer(max_chars: int = 12_000):
    """Return a pre_model_hook for create_react_agent (LangGraph 1.0+).

    Called before every LLM turn inside the ReAct agent. The returned dict is a
    temporary override only for that model call — the actual graph state is unchanged.
    Uses character count as a token proxy (~4 chars/token), so 12,000 chars ≈ 3,000
    tokens — well within the 30 K TPM limit even with max_tokens=4096 reserved for
    the response.
    """
    _trimmer = trim_messages(
        max_tokens=max_chars,
        strategy="last",
        token_counter=lambda msgs: sum(len(getattr(m, "content", "") or "") for m in msgs),
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    def pre_model_hook(state: dict) -> dict:
        msgs = state.get("messages", [])
        total = sum(len(getattr(m, "content", "") or "") for m in msgs)
        if total <= max_chars:
            return {}
        return {"messages": _trimmer.invoke(msgs)}

    return pre_model_hook


class _TPMRateLimiter:
    """Sliding-window token-per-minute rate limiter.

    Records actual tokens from AIMessage.usage_metadata after each topic.
    wait_if_needed() blocks only when the 62-second window is nearly full —
    early topics fire immediately; pauses appear only when the budget tightens.
    """

    def __init__(self, limit: int = 26_000):
        self.limit = limit          # 4 K buffer under Anthropic's 30 K TPM
        self._window = 62.0
        self._usage: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def record(self, tokens: int) -> None:
        with self._lock:
            self._usage.append((time.monotonic(), tokens))

    def _prune_and_total(self) -> int:
        now = time.monotonic()
        with self._lock:
            while self._usage and now - self._usage[0][0] >= self._window:
                self._usage.popleft()
            return sum(t for _, t in self._usage)

    def wait_if_needed(self, budget: int = 8_000) -> None:
        """Block until there is enough headroom for the next topic."""
        while True:
            used = self._prune_and_total()
            if used + budget <= self.limit:
                return
            with self._lock:
                if self._usage:
                    wait_for = (self._usage[0][0] + self._window) - time.monotonic() + 1.0
                    if wait_for > 0:
                        time.sleep(min(wait_for, 30))
                        continue
            time.sleep(5)


_rate_limiter = _TPMRateLimiter()


def _get_llm(temperature: float = 0.0, max_tokens: int = 4096):
    """Return the primary LLM. Set LLM_PROVIDER=anthropic to use Claude, defaults to OpenAI.
    Set OPENAI_MODEL to override the OpenAI model (default: gpt-4o-mini).
    max_tokens controls output length; compliance_evaluator passes 8192 for batch output.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=6,
        )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=6,
    )


def _parse_json(text: Any) -> Any:
    """Extract and parse JSON from LLM response text.

    Handles three formats Claude may return:
      - Plain JSON string
      - Markdown code block  ```json ... ```
      - Prose + embedded JSON (most common with ReAct agents)
      - List of Anthropic content blocks [{"type":"text","text":"..."}]

    Uses json.JSONDecoder.raw_decode so it stops precisely at the end of
    the first valid JSON value — no greedy regex over-extension.
    """
    # ── Normalise content blocks to a single string ────────────────────────
    if isinstance(text, list):
        parts: list[str] = []
        for block in text:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or block.get("content", ""))
            else:
                parts.append(str(block))
        text = " ".join(parts)
    if not isinstance(text, str):
        text = str(text)

    # ── Strategy 1: direct parse ────────────────────────────────────────────
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Strategy 2: markdown code fence  ```json … ``` ─────────────────────
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Strategy 3: raw_decode from first [ or { ────────────────────────────
    # raw_decode stops exactly at the end of the first complete JSON value,
    # so trailing prose / extra brackets never corrupt the result.
    decoder = json.JSONDecoder()
    for start_char in ("[", "{"):
        pos = text.find(start_char)
        while pos >= 0:
            try:
                result, _ = decoder.raw_decode(text, pos)
                return result
            except json.JSONDecodeError:
                pos = text.find(start_char, pos + 1)

    return []


# ===========================================================================
# Topic Sub-Graph — processes a single topic with agentic RAG agents
# ===========================================================================

def _route_standards(state: TopicState) -> Command[Literal["standards_lookup", "compliance_evaluator"]]:
    """Route: if any requirements reference standards, look them up; else skip."""
    requirements = state.get("requirements", [])
    all_refs = []
    for req in requirements:
        all_refs.extend(req.standard_refs)
    unique_refs = list(set(all_refs))

    if unique_refs:
        return Command(
            goto="standards_lookup",
            update={"standard_refs": unique_refs},
        )
    return Command(
        goto="compliance_evaluator",
        update={"standard_refs": [], "standards_context": []},
    )


def _run_standards_lookup(state: TopicState) -> dict:
    """Agentic RAG: search web for referenced standards."""
    standard_refs = state.get("standard_refs", [])

    if not standard_refs:
        return {
            "standards_context": [],
            "messages": [AIMessage(content="[StandardsLookup] No standards to look up")],
        }

    refs_text = json.dumps(standard_refs)
    prompt = _load_prompt("standards_lookup.md", standard_refs=refs_text)

    agent = create_react_agent(
        model=_get_llm(),
        tools=[search_standards],
        prompt=prompt,
        pre_model_hook=_make_trimmer(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=f"Look up these standards: {', '.join(standard_refs)}")]},
        config={"recursion_limit": 25}
    )

    last_msg = result["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    parsed = _parse_json(content)

    standards = []
    for item in (parsed if isinstance(parsed, list) else []):
        try:
            std = StandardInfo(
                standard_ref=item.get("standard_ref", ""),
                summary=item.get("summary", ""),
                source_url=item.get("source_url", ""),
            )
            standards.append(std)
        except Exception:
            continue

    return {
        "standards_context": standards,
        "messages": [AIMessage(content=f"[StandardsLookup] Found info on {len(standards)} standards")],
    }


_EVAL_BATCH_SIZE = 8  # requirements per LLM call — keeps output well under 8 K tokens


def _run_compliance_evaluator(state: TopicState) -> dict:
    """Evaluate compliance for each requirement, processing in batches of 8.

    """
    topic = state["topic"]
    requirements = state.get("requirements", [])
    claims = state.get("claims", [])
    standards_context = state.get("standards_context", [])

    if not requirements:
        return {
            "verdicts": [],
            "messages": [AIMessage(content=f"[ComplianceEvaluator] No requirements to evaluate for '{topic}'")],
        }

    # Build a lookup so each batch gets only the claims relevant to that slice
    claims_by_req = {c.req_id: c for c in claims}
    std_json = json.dumps([s.model_dump() for s in standards_context], indent=2)
    llm = _get_llm(max_tokens=8192)

    all_verdicts: list[ComplianceVerdict] = []

    for batch_start in range(0, len(requirements), _EVAL_BATCH_SIZE):
        batch_reqs = requirements[batch_start: batch_start + _EVAL_BATCH_SIZE]
        batch_claims = [
            claims_by_req[r.req_id].model_dump()
            for r in batch_reqs
            if r.req_id in claims_by_req
        ]

        prompt = _load_prompt(
            "compliance_evaluator.md",
            topic=topic,
            requirements=json.dumps([r.model_dump() for r in batch_reqs], indent=2),
            claims=json.dumps(batch_claims, indent=2),
            standards_context=std_json,
        )

        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(
                content=(
                    f"Evaluate compliance for {len(batch_reqs)} requirements "
                    f"(batch {batch_start // _EVAL_BATCH_SIZE + 1}) on topic: {topic}"
                )
            ),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json(content)

        for item in (parsed if isinstance(parsed, list) else []):
            try:
                verdict = ComplianceVerdict(
                    req_id=item.get("req_id", ""),
                    verdict=Verdict(item.get("verdict", "NOT_ADDRESSED")),
                    reasoning=item.get("reasoning", ""),
                    severity=Severity(item.get("severity", "INFO")),
                    confidence=float(item.get("confidence", 0.5)),
                    requirement_text=item.get("requirement_text", ""),
                    proposal_claim=item.get("proposal_claim", ""),
                )
                all_verdicts.append(verdict)
            except Exception:
                continue

    return {
        "verdicts": all_verdicts,
        "messages": [AIMessage(content=f"[ComplianceEvaluator] Produced {len(all_verdicts)} verdicts for '{topic}'")],
    }


def build_topic_graph(retrieval_tool=None) -> StateGraph:
    """Build the topic sub-graph: spec → proposal → standards? → compliance.

    Args:
        retrieval_tool: The retrieval tool for agents. Defaults to basic pipeline's
                        retrieve_documents from src.tools.
    """
    if retrieval_tool is None:
        from src.tools import retrieve_documents as _rt
        retrieval_tool = _rt

    # ── Closures that capture retrieval_tool ──────────────────────────────

    def _run_spec_analyzer(state: TopicState) -> dict:
        """Agentic RAG: extract requirements from spec for this topic."""
        topic = state["topic"]
        prompt = _load_prompt("spec_analyzer.md", topic=topic)

        agent = create_react_agent(
            model=_get_llm(),
            tools=[retrieval_tool],
            prompt=prompt,
            pre_model_hook=_make_trimmer(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content=f"Extract all requirements for topic: {topic}")]},
            config={"recursion_limit": 25}
        )

        parsed: Any = []
        all_msgs = result.get("messages", [])
        for msg in reversed(all_msgs):
            if not isinstance(msg, AIMessage):
                continue
            content = msg.content if hasattr(msg, "content") else str(msg)
            candidate = _parse_json(content)
            if isinstance(candidate, list) and candidate:
                parsed = candidate
                break

        if not parsed:
            last_ai = next((m for m in reversed(all_msgs) if isinstance(m, AIMessage)), None)
            raw = (last_ai.content if last_ai and hasattr(last_ai, "content") else "") or ""
            if isinstance(raw, list):
                raw = str(raw)
            print(f"[DEBUG spec_analyzer '{topic}'] no JSON found. Last AI msg[:300]:\n{raw[:300]}\n")

        requirements = []
        for item in (parsed if isinstance(parsed, list) else []):
            try:
                req = Requirement(
                    req_id=item.get("req_id", f"REQ-{len(requirements)+1:03d}"),
                    text=item.get("text", ""),
                    obligation=Obligation(item.get("obligation", "NONE")),
                    page=item.get("page"),
                    section=item.get("section", ""),
                    standard_refs=item.get("standard_refs", []),
                )
                requirements.append(req)
            except Exception:
                continue

        return {
            "requirements": requirements,
            "spec_chunks": str(all_msgs[-1].content if all_msgs else ""),
            "messages": [AIMessage(content=f"[SpecAnalyzer] Extracted {len(requirements)} requirements for '{topic}'")],
        }

    def _run_proposal_reviewer(state: TopicState) -> dict:
        """Agentic RAG: match proposal claims to requirements."""
        topic = state["topic"]
        requirements = state.get("requirements", [])

        if not requirements:
            return {
                "claims": [],
                "proposal_chunks": "",
                "messages": [AIMessage(content=f"[ProposalReviewer] No requirements to check for '{topic}'")],
            }

        reqs_text = json.dumps([r.model_dump() for r in requirements], indent=2)
        prompt = _load_prompt("proposal_reviewer.md", topic=topic, requirements=reqs_text)

        agent = create_react_agent(
            model=_get_llm(),
            tools=[retrieval_tool],
            prompt=prompt,
            pre_model_hook=_make_trimmer(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content=f"Find proposal responses for {len(requirements)} requirements on topic: {topic}")]},
            config={"recursion_limit": 25}
        )

        parsed_claims: Any = []
        all_msgs = result.get("messages", [])
        for msg in reversed(all_msgs):
            if not isinstance(msg, AIMessage):
                continue
            content = msg.content if hasattr(msg, "content") else str(msg)
            candidate = _parse_json(content)
            if isinstance(candidate, list) and candidate:
                parsed_claims = candidate
                break

        if not parsed_claims:
            last_ai = next((m for m in reversed(all_msgs) if isinstance(m, AIMessage)), None)
            raw = (last_ai.content if last_ai and hasattr(last_ai, "content") else "") or ""
            if isinstance(raw, list):
                raw = str(raw)
            print(f"[DEBUG proposal_reviewer '{topic}'] no JSON found. Last AI msg[:300]:\n{raw[:300]}\n")

        claims = []
        for item in (parsed_claims if isinstance(parsed_claims, list) else []):
            try:
                claim = ProposalClaim(
                    req_id=item.get("req_id", ""),
                    claim_text=item.get("claim_text", ""),
                    proposal_page=item.get("proposal_page"),
                    match_quality=MatchQuality(item.get("match_quality", "NOT_FOUND")),
                )
                claims.append(claim)
            except Exception:
                continue

        return {
            "claims": claims,
            "proposal_chunks": str(all_msgs[-1].content if all_msgs else ""),
            "messages": [AIMessage(content=f"[ProposalReviewer] Found {len(claims)} proposal claims for '{topic}'")],
        }

    # ── Build graph ──────────────────────────────────────────────────────

    graph = StateGraph(TopicState)

    graph.add_node("spec_analyzer", _run_spec_analyzer)
    graph.add_node("proposal_reviewer", _run_proposal_reviewer)
    graph.add_node("route_standards", _route_standards)
    graph.add_node("standards_lookup", _run_standards_lookup)
    graph.add_node("compliance_evaluator", _run_compliance_evaluator)

    graph.add_edge(START, "spec_analyzer")
    graph.add_edge("spec_analyzer", "proposal_reviewer")
    graph.add_edge("proposal_reviewer", "route_standards")
    # route_standards uses Command for dynamic routing
    graph.add_edge("standards_lookup", "compliance_evaluator")
    graph.add_edge("compliance_evaluator", END)

    return graph


# ===========================================================================
# Orchestrator Graph — plans topics, spawns sub-graphs, generates report
# ===========================================================================

def _check_more_topics(state: EDRState) -> Command[Literal["advance_topic", "generate_report"]]:
    """Check if there are more topics to process."""
    idx = state.get("current_topic_index", 0)
    topics = state.get("topics", [])

    if idx + 1 < len(topics):
        return Command(goto="advance_topic")
    return Command(goto="generate_report")


def _advance_topic(state: EDRState) -> dict:
    """Move to the next topic, waiting only as long as the TPM budget requires."""
    _rate_limiter.wait_if_needed()
    idx = state.get("current_topic_index", 0)
    return {
        "current_topic_index": idx + 1,
        "messages": [AIMessage(content=f"[Orchestrator] Advancing to topic {idx + 2}")],
    }


def _generate_report(state: EDRState) -> dict:
    """Generate the final compliance report from all verdicts."""
    all_verdicts = state.get("all_verdicts", [])

    if not all_verdicts:
        return {
            "final_report": "# Compliance Report\n\nNo findings were generated.",
            "messages": [AIMessage(content="[Orchestrator] Generated empty report")],
        }

    prompt = _load_prompt(
        "report.md",
        verdicts=json.dumps([v.model_dump() for v in all_verdicts], indent=2),
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Generate a compliance report from {len(all_verdicts)} findings."),
    ])

    report = response.content if hasattr(response, "content") else str(response)

    return {
        "final_report": report,
        "messages": [AIMessage(content=f"[Orchestrator] Generated report with {len(all_verdicts)} findings")],
    }


def _supervisor_verify(state: EDRState) -> dict:
    """Supervisor reviews the current topic's verdicts for completeness and quality.

    This is the quality-control step in the pipeline: the same Supervisor that planned
    the topics verifies each topic's output before the graph advances to the next one.
    The verification note is stored in supervisor_notes (accumulated) and also emitted
    as an AIMessage for the streaming UI to display.
    """
    topic_idx = state.get("current_topic_index", 0)
    topics = state.get("topics", [])
    topic = topics[topic_idx] if topic_idx < len(topics) else "Unknown"
    verdicts = state.get("pending_topic_verdicts", [])

    if not verdicts:
        note = f"[SupervisorVerify] Topic '{topic}': no verdicts produced — may need re-review."
        return {
            "supervisor_notes": [note],
            "messages": [AIMessage(content=note)],
        }

    llm = _get_llm()
    verdicts_summary = json.dumps(
        [
            {
                "req_id": v.req_id,
                "verdict": v.verdict.value,
                "severity": v.severity.value,
                "reasoning": v.reasoning[:200],
            }
            for v in verdicts
        ],
        indent=2,
    )

    response = llm.invoke([
        SystemMessage(content=(
            "You are the compliance review supervisor. Your job is to briefly verify that "
            "the compliance verdicts for this topic are complete, logically consistent, and "
            "cover the key requirements. Flag any obvious gaps. Be concise (2-3 sentences)."
        )),
        HumanMessage(content=(
            f"Topic: {topic}\n\n"
            f"Verdicts ({len(verdicts)} total):\n{verdicts_summary}\n\n"
            "Verification: Are these verdicts complete and sound? Note any concerns."
        )),
    ])

    note = f"[SupervisorVerify] Topic '{topic}': {response.content}"
    return {
        "supervisor_notes": [note],
        "messages": [AIMessage(content=note)],
    }


def build_graph(retrieval_tool=None) -> StateGraph:
    """Build the orchestrator graph.

    Args:
        retrieval_tool: The retrieval tool for agents. Defaults to basic pipeline's
                        retrieve_documents from src.tools.

    Flow:
        START → plan_topics → process_topic → supervisor_verify
              → check_more_topics ──[more]──→ advance_topic → process_topic
                                  ──[done]──→ generate_report → END
    """
    if retrieval_tool is None:
        from src.tools import retrieve_documents as _rt
        retrieval_tool = _rt

    # ── Closures that capture retrieval_tool ──────────────────────────────

    def _plan_topics(state: EDRState) -> dict:
        """Plan which topics to review based on document structure."""
        prompt = _load_prompt("orchestrator.md")
        llm = _get_llm()

        structure = retrieval_tool.invoke({
            "query": "table of contents sections headings scope",
            "doc_type": "spec",
            "top_k": 5,
        })

        # If advanced pipeline, supplement with section outline for full coverage
        section_outline = ""
        try:
            from src.ingestion_advanced import get_section_outline
            outline = get_section_outline("spec")
            if outline:
                section_outline = f"\n\nDocument Section Outline:\n{outline}"
        except ImportError:
            pass

        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Here is the document structure:\n\n{structure}{section_outline}\n\nIdentify the key review topics."),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        topics = _parse_json(content)

        if not isinstance(topics, list) or not topics:
            topics = ["General BMS Requirements"]

        return {
            "topics": topics,
            "current_topic_index": 0,
            "messages": [AIMessage(content=f"[Orchestrator] Planned {len(topics)} topics: {topics}")],
        }

    def _process_topic(state: EDRState) -> dict:
        """Process the current topic using the topic sub-graph."""
        idx = state.get("current_topic_index", 0)
        topics = state.get("topics", [])

        if idx >= len(topics):
            return {"messages": [AIMessage(content="[Orchestrator] All topics processed")]}

        topic = topics[idx]

        # Build and compile the topic sub-graph with the same retrieval tool
        topic_graph = build_topic_graph(retrieval_tool=retrieval_tool).compile()

        topic_state: TopicState = {
            "topic": topic,
            "spec_chunks": "",
            "requirements": [],
            "proposal_chunks": "",
            "claims": [],
            "standard_refs": [],
            "standards_context": [],
            "verdicts": [],
            "messages": [],
        }

        result = topic_graph.invoke(topic_state)

        _topic_tokens = sum(
            (getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0)
            + (getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0)
            for m in result.get("messages", [])
        )
        if not _topic_tokens:
            _topic_tokens = sum(
                len(getattr(m, "content", "") or "") // 4
                for m in result.get("messages", [])
            )
        _rate_limiter.record(_topic_tokens)

        verdicts = result.get("verdicts", [])
        summary = f"Topic '{topic}': {len(verdicts)} verdicts"

        return {
            "all_verdicts": verdicts,
            "pending_topic_verdicts": verdicts,
            "topic_summaries": [summary],
            "messages": [AIMessage(content=f"[Orchestrator] {summary}")],
        }

    # ── Build graph ──────────────────────────────────────────────────────

    graph = StateGraph(EDRState)

    graph.add_node("plan_topics", _plan_topics)
    graph.add_node("process_topic", _process_topic)
    graph.add_node("supervisor_verify", _supervisor_verify)
    graph.add_node("check_more_topics", _check_more_topics)
    graph.add_node("advance_topic", _advance_topic)
    graph.add_node("generate_report", _generate_report)

    graph.add_edge(START, "plan_topics")
    graph.add_edge("plan_topics", "process_topic")
    graph.add_edge("process_topic", "supervisor_verify")      # quality-check after each topic
    graph.add_edge("supervisor_verify", "check_more_topics")  # then decide: more or done?
    # check_more_topics uses Command for dynamic routing
    graph.add_edge("advance_topic", "process_topic")
    graph.add_edge("generate_report", END)

    return graph


def compile_graph(retrieval_tool=None):
    """Compile and return the orchestrator graph ready to invoke.

    Args:
        retrieval_tool: The retrieval tool for agents. Defaults to basic pipeline's
                        retrieve_documents from src.tools.
    """
    return build_graph(retrieval_tool=retrieval_tool).compile()


def save_graph_image(path: str | Path = "graph.png") -> Path:
    """Render the orchestrator graph to a PNG image and save it.

    Uses LangGraph's built-in Mermaid rendering (calls mermaid.ink API).
    """
    path = Path(path)
    compiled = compile_graph()
    png_bytes = compiled.get_graph().draw_mermaid_png()
    path.write_bytes(png_bytes)
    return path


def save_topic_graph_image(path: str | Path = "topic_graph.png") -> Path:
    """Render the topic sub-graph to a PNG image and save it."""
    path = Path(path)
    compiled = build_topic_graph().compile()
    png_bytes = compiled.get_graph().draw_mermaid_png()
    path.write_bytes(png_bytes)
    return path
