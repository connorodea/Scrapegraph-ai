"""Configuration loading and environment helpers."""

from __future__ import annotations

import os
from datetime import date
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class GitHubSettings(BaseModel):
    api_url: str = "https://api.github.com"
    concurrency: int = Field(default=6, ge=1, le=20)
    per_page: int = Field(default=100, ge=1, le=100)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=5, ge=0, le=10)
    user_agent: str = "juricratic-research-radar/0.1"


class DiscoverySettings(BaseModel):
    start_date: date = date(2008, 1, 1)
    end_date: date | None = None
    max_results_per_shard: int = Field(default=950, ge=1, le=1000)
    min_stars: int = Field(default=0, ge=0)
    include_forks: bool = False
    include_archived: bool = False
    max_queries: int | None = Field(default=None, ge=1)
    seed_repositories: list[str] = Field(default_factory=list)

    @field_validator("end_date", mode="after")
    @classmethod
    def default_end_date(cls, value: date | None) -> date:
        return value or date.today()


class SnapshotSettings(BaseModel):
    directory: str = "./snapshots"
    max_repo_mb: int = Field(default=200, ge=1)
    max_count: int = Field(default=25, ge=1)
    minimum_score: float = Field(default=65, ge=0, le=100)


class RadarConfig(BaseModel):
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    snapshot: SnapshotSettings = Field(default_factory=SnapshotSettings)
    queries: list[str] = Field(default_factory=list)
    ontology: dict[str, list[str]] = Field(default_factory=dict)
    score_weights: dict[str, float] = Field(default_factory=dict)


def default_config_path() -> Path:
    return Path(resource_files("juricratic_radar.data").joinpath("default.yaml"))


def load_config(path: str | Path | None = None) -> RadarConfig:
    target = Path(path).expanduser().resolve() if path else default_config_path()
    with target.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return RadarConfig.model_validate(raw)


def github_token() -> str | None:
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def default_db_path() -> Path:
    return Path(os.getenv("JURICRATIC_RADAR_DB", "./data/juricratic_radar.sqlite3")).expanduser()
