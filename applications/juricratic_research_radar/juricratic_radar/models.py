"""Typed domain models for the Juricratic Research Radar."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CandidateKind(str, Enum):
    GITHUB_REPOSITORY = "github_repository"
    RESEARCH_PAPER = "research_paper"


class RunKind(str, Enum):
    GITHUB = "github"
    PAPERS = "papers"


class Stage(str, Enum):
    PLAN = "plan"
    DISCOVER = "discover"
    ACCESS = "access"
    ENRICH = "enrich"
    EXPAND = "expand"
    SCORE = "score"
    PERSIST = "persist"
    COMPLETE = "complete"


class EventLevel(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class AccessOutcome(str, Enum):
    ALLOWED = "allowed"
    API_ONLY = "api_only"
    BLOCKED = "blocked"
    REVIEW = "review"


class AccessDecision(BaseModel):
    model_config = ConfigDict(extra="allow")

    outcome: AccessOutcome
    code: str
    plain_english: str
    technical_detail: str = ""
    robots_allowed: bool | None = None
    proxy_escalation_allowed: bool = False
    checked_at: datetime = Field(default_factory=utc_now)


class FetchResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str
    ok: bool
    provider: str
    status_code: int | None = None
    content_type: str | None = None
    text: str = ""
    elapsed_ms: int = 0
    estimated_credits: float = 0.0
    access: AccessDecision | None = None
    error: str | None = None


class ArtifactExtraction(BaseModel):
    """Structured extraction produced by deterministic rules or ScrapeGraphAI."""

    model_config = ConfigDict(extra="allow")

    research_question: str = ""
    legal_domain: list[str] = Field(default_factory=list)
    jurisdiction: list[str] = Field(default_factory=list)
    unit_of_analysis: str = ""
    data_sources: list[str] = Field(default_factory=list)
    dataset_size: str = ""
    labels_or_targets: list[str] = Field(default_factory=list)
    factual_variables: list[str] = Field(default_factory=list)
    legal_variables: list[str] = Field(default_factory=list)
    behavioral_variables: list[str] = Field(default_factory=list)
    strategic_variables: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    validation_design: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    code_artifacts: list[str] = Field(default_factory=list)
    reusable_priors: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    juricratic_relevance: str = ""
    extraction_mode: str = "deterministic"


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="allow")

    overall: float = 0.0
    juricratic_fit: float = 0.0
    reproducibility: float = 0.0
    data_value: float = 0.0
    methodological_value: float = 0.0
    credibility: float = 0.0
    freshness: float = 0.0
    popularity: float = 0.0
    dimensions: dict[str, float] = Field(default_factory=dict)


class ResearchCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    kind: CandidateKind
    source: str
    title: str
    url: str
    canonical_url: str | None = None
    description: str = ""
    abstract: str = ""
    repository_full_name: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    publication_date: str | None = None
    doi: str | None = None
    jurisdiction: list[str] = Field(default_factory=list)
    legal_domains: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    language: str | None = None
    license: str | None = None
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    citation_count: int = 0
    updated_at: datetime | None = None
    created_at: datetime | None = None
    open_access: bool | None = None
    open_access_url: str | None = None
    readme_excerpt: str = ""
    artifact_paths: list[str] = Field(default_factory=list)
    extraction: ArtifactExtraction | None = None
    access: AccessDecision | None = None
    score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    discovered_by: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        parts = [
            self.title,
            self.description,
            self.abstract,
            self.readme_excerpt,
            " ".join(self.legal_domains),
            " ".join(self.topics),
            " ".join(self.methods),
            " ".join(self.datasets),
            " ".join(self.artifacts),
            " ".join(self.artifact_paths),
        ]
        if self.extraction:
            extraction = self.extraction
            parts.extend(
                [
                    extraction.research_question,
                    " ".join(extraction.legal_domain),
                    " ".join(extraction.jurisdiction),
                    extraction.unit_of_analysis,
                    " ".join(extraction.data_sources),
                    extraction.dataset_size,
                    " ".join(extraction.labels_or_targets),
                    " ".join(extraction.factual_variables),
                    " ".join(extraction.legal_variables),
                    " ".join(extraction.behavioral_variables),
                    " ".join(extraction.strategic_variables),
                    " ".join(extraction.methods),
                    " ".join(extraction.validation_design),
                    " ".join(extraction.metrics),
                    " ".join(extraction.datasets),
                    " ".join(extraction.code_artifacts),
                    " ".join(extraction.reusable_priors),
                    " ".join(extraction.limitations),
                    extraction.juricratic_relevance,
                ]
            )
        return "\n".join(part for part in parts if part)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: RunKind
    seed: str
    generated_queries: list[str] = Field(default_factory=list)
    families: dict[str, list[str]] = Field(default_factory=dict)
    lsi_terms: list[str] = Field(default_factory=list)
    budget: int = 0
    explanation: list[str] = Field(default_factory=list)


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    run_kind: RunKind
    timestamp: datetime = Field(default_factory=utc_now)
    stage: Stage
    level: EventLevel = EventLevel.INFO
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    stage_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    plain_english: str
    technical_detail: str = ""
    counters: dict[str, int | float | str] = Field(default_factory=dict)


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    run_kind: RunKind
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    status: str = "running"
    query_plan: QueryPlan | None = None
    candidates_found: int = 0
    candidates_saved: int = 0
    blocked_urls: int = 0
    errors: int = 0
    estimated_proxy_credits: float = 0.0
    top_score: float = 0.0
    notes: list[str] = Field(default_factory=list)
