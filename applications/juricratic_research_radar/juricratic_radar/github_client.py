"""GitHub API-first discovery and repository enrichment."""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from urllib.parse import quote

from .config import Settings
from .http_client import RadarHttpError, ResilientHttpClient
from .models import (
    AccessDecision,
    AccessOutcome,
    CandidateKind,
    ResearchCandidate,
)
from .query_expansion import Taxonomy


@dataclass(slots=True)
class GitHubSearchBatch:
    query: str
    candidates: list[ResearchCandidate]
    total_count: int
    incomplete_results: bool
    rate_limit: dict[str, int | str] = field(default_factory=dict)


class GitHubResearchClient:
    API_ROOT = "https://api.github.com"
    URL_RE = re.compile(r"https?://[^\s)\]>\"']+")

    def __init__(
        self,
        settings: Settings,
        taxonomy: Taxonomy,
        http: ResilientHttpClient | None = None,
        on_wait: Callable[[float, str], None] | None = None,
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
        self.settings = settings
        self.taxonomy = taxonomy
        self.http = http or ResilientHttpClient(
            settings, default_headers=headers, on_wait=on_wait
        )
        self._own_http = http is None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _candidate_id(full_name: str) -> str:
        digest = hashlib.sha1(full_name.lower().encode("utf-8")).hexdigest()[:16]
        return f"gh:{digest}"

    def _from_repository(self, item: dict[str, Any], query: str) -> ResearchCandidate:
        full_name = str(item.get("full_name") or item.get("name") or "unknown")
        license_payload = item.get("license") or {}
        license_name = license_payload.get("spdx_id") or license_payload.get("name")
        topics = [str(topic) for topic in item.get("topics") or []]
        owner = item.get("owner") or {}
        return ResearchCandidate(
            candidate_id=self._candidate_id(full_name),
            kind=CandidateKind.GITHUB_REPOSITORY,
            source="GitHub REST API",
            title=full_name,
            url=str(item.get("html_url") or f"https://github.com/{full_name}"),
            canonical_url=str(item.get("html_url") or f"https://github.com/{full_name}"),
            description=str(item.get("description") or ""),
            repository_full_name=full_name,
            topics=topics,
            language=item.get("language"),
            license=license_name,
            stars=int(item.get("stargazers_count") or 0),
            forks=int(item.get("forks_count") or 0),
            watchers=int(item.get("watchers_count") or 0),
            updated_at=self._parse_datetime(item.get("updated_at")),
            created_at=self._parse_datetime(item.get("created_at")),
            access=AccessDecision(
                outcome=AccessOutcome.API_ONLY,
                code="github_api",
                plain_english="Public repository metadata was read through GitHub's official API.",
                technical_detail="No GitHub web-page scraping or private repository access was attempted.",
                proxy_escalation_allowed=False,
            ),
            discovered_by=[query],
            raw={
                "github_id": item.get("id"),
                "node_id": item.get("node_id"),
                "owner": owner.get("login"),
                "default_branch": item.get("default_branch") or "main",
                "archived": bool(item.get("archived")),
                "disabled": bool(item.get("disabled")),
                "fork": bool(item.get("fork")),
                "open_issues_count": int(item.get("open_issues_count") or 0),
                "size_kb": int(item.get("size") or 0),
                "homepage": item.get("homepage"),
                "has_wiki": bool(item.get("has_wiki")),
            },
        )

    def search_repositories(
        self,
        query: str,
        *,
        per_page: int = 30,
        pages: int = 1,
        sort: str = "updated",
    ) -> GitHubSearchBatch:
        per_page = max(1, min(per_page, 100))
        pages = max(1, min(pages, 10))
        candidates: list[ResearchCandidate] = []
        total_count = 0
        incomplete = False
        rate_limit: dict[str, int | str] = {}

        for page in range(1, pages + 1):
            payload, response = self.http.get_json(
                f"{self.API_ROOT}/search/repositories",
                params={
                    "q": query,
                    "sort": sort,
                    "order": "desc",
                    "per_page": per_page,
                    "page": page,
                },
            )
            if not isinstance(payload, dict):
                raise RadarHttpError("GitHub repository search returned an unexpected response.")
            total_count = int(payload.get("total_count") or 0)
            incomplete = bool(payload.get("incomplete_results"))
            rate_limit = response.rate_limit
            items = payload.get("items") or []
            if not items:
                break
            candidates.extend(self._from_repository(item, query) for item in items)
            if len(items) < per_page:
                break

        return GitHubSearchBatch(
            query=query,
            candidates=candidates,
            total_count=total_count,
            incomplete_results=incomplete,
            rate_limit=rate_limit,
        )

    def fetch_readme(self, full_name: str, *, max_chars: int = 60_000) -> str:
        payload, _ = self.http.get_json(
            f"{self.API_ROOT}/repos/{full_name}/readme",
            headers={"Accept": "application/vnd.github+json"},
            max_attempts=3,
        )
        if not isinstance(payload, dict):
            return ""
        encoded = payload.get("content") or ""
        if payload.get("encoding") == "base64" and encoded:
            try:
                text = base64.b64decode(encoded).decode("utf-8", errors="replace")
            except Exception:
                return ""
        else:
            text = str(encoded)
        return text[:max_chars]

    def fetch_languages(self, full_name: str) -> dict[str, int]:
        payload, _ = self.http.get_json(
            f"{self.API_ROOT}/repos/{full_name}/languages",
            max_attempts=3,
        )
        if not isinstance(payload, dict):
            return {}
        return {str(key): int(value) for key, value in payload.items()}

    def fetch_artifact_paths(
        self,
        full_name: str,
        default_branch: str,
        *,
        limit: int = 120,
    ) -> tuple[list[str], bool]:
        branch_ref = quote(default_branch, safe="")
        payload, _ = self.http.get_json(
            f"{self.API_ROOT}/repos/{full_name}/git/trees/{branch_ref}",
            params={"recursive": "1"},
            max_attempts=3,
        )
        if not isinstance(payload, dict):
            return [], False
        tree = payload.get("tree") or []
        signals = tuple(signal.lower() for signal in self.taxonomy.file_signals)
        paths: list[str] = []
        for entry in tree:
            if entry.get("type") != "blob":
                continue
            path = str(entry.get("path") or "")
            lower = path.lower()
            if any(
                lower.endswith(signal)
                or lower == signal
                or f"/{signal}" in lower
                for signal in signals
            ):
                paths.append(path)
            if len(paths) >= limit:
                break
        return paths, bool(payload.get("truncated"))

    def enrich_repository(
        self,
        candidate: ResearchCandidate,
        *,
        include_tree: bool = True,
        include_languages: bool = True,
    ) -> ResearchCandidate:
        full_name = candidate.repository_full_name
        if not full_name:
            return candidate
        readme = ""
        languages: dict[str, int] = {}
        artifact_paths: list[str] = []
        tree_truncated = False
        errors: list[str] = []

        try:
            readme = self.fetch_readme(full_name)
        except RadarHttpError as exc:
            errors.append(f"README: {exc}")
        if include_languages:
            try:
                languages = self.fetch_languages(full_name)
            except RadarHttpError as exc:
                errors.append(f"languages: {exc}")
        if include_tree:
            try:
                artifact_paths, tree_truncated = self.fetch_artifact_paths(
                    full_name,
                    str(candidate.raw.get("default_branch") or "main"),
                )
            except RadarHttpError as exc:
                errors.append(f"tree: {exc}")

        urls = [url.rstrip(".,;") for url in self.URL_RE.findall(readme)]
        update_raw = dict(candidate.raw)
        update_raw.update(
            {
                "languages": languages,
                "tree_truncated": tree_truncated,
                "readme_urls": urls[:100],
                "enrichment_errors": errors,
            }
        )
        return candidate.model_copy(
            update={
                "readme_excerpt": readme[:18_000],
                "artifact_paths": artifact_paths,
                "raw": update_raw,
            }
        )

    def close(self) -> None:
        if self._own_http:
            self.http.close()
