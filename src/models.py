"""Data models and LangGraph state types for the EDR platform."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIAL = "PARTIAL"
    NOT_ADDRESSED = "NOT_ADDRESSED"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class Obligation(str, Enum):
    SHALL = "SHALL"
    MUST = "MUST"
    SHOULD = "SHOULD"
    MAY = "MAY"
    NONE = "NONE"


class MatchQuality(str, Enum):
    DIRECT = "DIRECT"
    PARTIAL = "PARTIAL"
    NOT_FOUND = "NOT_FOUND"


# ---------------------------------------------------------------------------
# Pydantic models — structured outputs from agents
# ---------------------------------------------------------------------------

class Requirement(BaseModel):
    """A single requirement extracted from the spec."""
    req_id: str = Field(description="Unique identifier, e.g. REQ-001")
    text: str = Field(description="Full requirement text")
    obligation: Obligation = Field(default=Obligation.NONE)
    page: int | None = Field(default=None, description="Source page number")
    section: str = Field(default="", description="Section heading")
    standard_refs: list[str] = Field(
        default_factory=list,
        description="Referenced standards, e.g. ['ASHRAE 90.1', 'NFPA 72']",
    )


class ProposalClaim(BaseModel):
    """A proposal's response to a requirement."""
    req_id: str = Field(description="Requirement this claim addresses")
    claim_text: str = Field(description="What the proposal claims")
    proposal_page: int | None = Field(default=None)
    match_quality: MatchQuality = Field(default=MatchQuality.NOT_FOUND)


class StandardInfo(BaseModel):
    """Information about a referenced standard."""
    standard_ref: str = Field(description="Standard identifier, e.g. ASHRAE 90.1")
    summary: str = Field(description="Relevant standard requirements summary")
    source_url: str = Field(default="")


class ComplianceVerdict(BaseModel):
    """Compliance verdict for a single requirement."""
    req_id: str
    verdict: Verdict
    reasoning: str
    severity: Severity = Field(default=Severity.INFO)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requirement_text: str = Field(default="")
    proposal_claim: str = Field(default="")


# ---------------------------------------------------------------------------
# LangGraph State Types
# ---------------------------------------------------------------------------

class TopicState(TypedDict):
    """State for a single topic sub-graph (isolated per topic)."""
    topic: str
    spec_chunks: str
    requirements: list[Requirement]
    proposal_chunks: str
    claims: list[ProposalClaim]
    standard_refs: list[str]
    standards_context: list[StandardInfo]
    verdicts: list[ComplianceVerdict]
    messages: Annotated[list, operator.add]


class EDRState(TypedDict):
    """State for the orchestrator graph."""
    # Input
    document_types: dict[str, str]  # filename -> doc_type
    review_request: str

    # Planning
    topics: list[str]
    current_topic_index: int

    # Accumulated results (reducer: append)
    all_verdicts: Annotated[list[ComplianceVerdict], operator.add]
    topic_summaries: Annotated[list[str], operator.add]

    # Current topic's verdicts pending supervisor review (plain field, overwritten each topic)
    pending_topic_verdicts: list[ComplianceVerdict]

    # Supervisor's per-topic verification notes (accumulated)
    supervisor_notes: Annotated[list[str], operator.add]

    # Final output
    final_report: str

    # Agent messages for tracing
    messages: Annotated[list, operator.add]
