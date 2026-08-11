"""Typed domain models used throughout the radar."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactType(StrEnum):
    PAPER = "paper"
    DATASET = "dataset"
    MODEL = "model"
    PIPELINE = "pipeline"
    BENCHMARK = "benchmark"
    ANNOTATION = "annotation"
    CITATION = "citation"
    NOTEBOOK = "notebook"
    FILE_SIGNAL = "file_signal"
    OTHER = "other"


class Artifact(BaseModel):
    artifact_type: ArtifactType
    name: str | None = None
    url: str | None = None
    identifier: str | None = None
    source: str
    path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def dedupe_key(self) -> tuple[str, str, str, str]:
        return (
            self.artifact_type.value,
            (self.identifier or "").casefold(),
            (self.url or "").casefold(),
            (self.path or "").casefold(),
        )


class RepositoryRecord(BaseModel):
    github_id: int
    full_name: str
    owner: str
    name: str
    html_url: str
    description: str | None = None
    language: str | None = None
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    size_kb: int = 0
    default_branch: str = "main"
    license_spdx: str | None = None
    topics: list[str] = Field(default_factory=list)
    is_fork: bool = False
    archived: bool = False
    disabled: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    discovered_at: datetime = Field(default_factory=utc_now)
    enriched_at: datetime | None = None
    source_query: str | None = None
    readme_text: str = ""
    tree_paths: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_github(cls, payload: dict[str, Any], *, source_query: str | None = None) -> "RepositoryRecord":
        owner_payload = payload.get("owner") or {}
        license_payload = payload.get("license") or {}
        return cls(
            github_id=int(payload["id"]),
            full_name=str(payload["full_name"]),
            owner=str(owner_payload.get("login") or str(payload["full_name"]).split("/", 1)[0]),
            name=str(payload["name"]),
            html_url=str(payload.get("html_url") or ""),
            description=payload.get("description"),
            language=payload.get("language"),
            stars=int(payload.get("stargazers_count") or 0),
            forks=int(payload.get("forks_count") or 0),
            open_issues=int(payload.get("open_issues_count") or 0),
            size_kb=int(payload.get("size") or 0),
            default_branch=str(payload.get("default_branch") or "main"),
            license_spdx=license_payload.get("spdx_id"),
            topics=list(payload.get("topics") or []),
            is_fork=bool(payload.get("fork")),
            archived=bool(payload.get("archived")),
            disabled=bool(payload.get("disabled")),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            pushed_at=payload.get("pushed_at"),
            source_query=source_query,
            raw=payload,
        )


class SearchShard(BaseModel):
    base_query: str
    query: str
    start_date: date
    end_date: date
    total_count: int
    star_qualifier: str | None = None
    overflow: bool = False


class ScoreBreakdown(BaseModel):
    total: float
    legal_domain: float = 0
    research_task: float = 0
    data_signal: float = 0
    method_signal: float = 0
    paper_signal: float = 0
    reproducibility: float = 0
    quality: float = 0
    penalties: float = 0
    classifications: list[str] = Field(default_factory=list)
    matched_terms: dict[str, list[str]] = Field(default_factory=dict)


class RunSummary(BaseModel):
    queries_attempted: int = 0
    shards_planned: int = 0
    repositories_seen: int = 0
    repositories_inserted_or_updated: int = 0
    repositories_enriched: int = 0
    errors: int = 0
    stopped_early: bool = False
