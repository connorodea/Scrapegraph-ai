"""End-to-end discovery, enrichment, vocabulary learning, and snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .config import RadarConfig
from .engine import ArtifactExtractor, GitHubClient, QueryPlanner, ResearchScorer, build_queries, derive_keywords
from .models import RepositoryRecord, RunSummary, SearchShard
from .storage import RadarDatabase

CITATION_PATH_RE = re.compile(r"(?:^|/)(?:citation\.cff|[^/]+\.bib)$", re.IGNORECASE)


class RadarPipeline:
    def __init__(self, config: RadarConfig, database: RadarDatabase, *, token: str | None = None):
        self.config = config
        self.database = database
        self.token = token
        self.extractor = ArtifactExtractor()
        self.scorer = ResearchScorer(config)
        self._database_lock = asyncio.Lock()

    async def discover(self, *, max_queries: int | None = None, max_repositories: int | None = None, shard_queries: bool = True) -> RunSummary:
        self.database.initialize()
        summary = RunSummary()
        dynamic_terms = self.database.top_derived_keywords(limit=25)
        queries = build_queries(self.config, dynamic_terms, limit=max_queries or self.config.discovery.max_queries)
        seen: set[int] = set()
        async with GitHubClient(self.config.github, self.token) as client:
            for full_name in self.config.discovery.seed_repositories:
                try:
                    repository = RepositoryRecord.from_github(await client.get_repository(full_name), source_query="seed_repository")
                    if self._allowed(repository):
                        self.database.upsert_repository(repository)
                        seen.add(repository.github_id)
                        summary.repositories_seen += 1
                        summary.repositories_inserted_or_updated += 1
                except Exception:
                    summary.errors += 1
            planner = QueryPlanner(client, max_results_per_shard=self.config.discovery.max_results_per_shard)
            for base_query in queries:
                summary.queries_attempted += 1
                try:
                    if shard_queries:
                        shards = await planner.plan(base_query, start_date=self.config.discovery.start_date, end_date=self.config.discovery.end_date)
                    else:
                        payload = await client.search_repositories(base_query, page=1, per_page=1)
                        total = int(payload.get("total_count") or 0)
                        shards = [SearchShard(base_query=base_query, query=base_query, start_date=self.config.discovery.start_date, end_date=self.config.discovery.end_date, total_count=total, overflow=total > 1000)] if total else []
                except Exception:
                    summary.errors += 1
                    continue
                summary.shards_planned += len(shards)
                for shard in shards:
                    run_id = self.database.start_query_run(base_query, shard.query, shard.total_count)
                    fetched = 0
                    try:
                        pages = min(10, max(1, math.ceil(min(shard.total_count, 1000) / self.config.github.per_page)))
                        for page in range(1, pages + 1):
                            payload = await client.search_repositories(shard.query, page=page, per_page=self.config.github.per_page)
                            items = list(payload.get("items") or [])
                            if not items:
                                break
                            fetched += len(items)
                            summary.repositories_seen += len(items)
                            for item in items:
                                repository = RepositoryRecord.from_github(item, source_query=shard.query)
                                if not self._allowed(repository):
                                    continue
                                self.database.upsert_repository(repository)
                                if repository.github_id not in seen:
                                    seen.add(repository.github_id)
                                    summary.repositories_inserted_or_updated += 1
                                if max_repositories and len(seen) >= max_repositories:
                                    summary.stopped_early = True
                                    break
                            if summary.stopped_early:
                                break
                        self.database.finish_query_run(run_id, fetched_count=fetched)
                    except Exception as exc:
                        summary.errors += 1
                        self.database.finish_query_run(run_id, fetched_count=fetched, status="failed", error=str(exc)[:2000])
                    if summary.stopped_early:
                        return summary
        return summary

    def _allowed(self, repository: RepositoryRecord) -> bool:
        settings = self.config.discovery
        return not (
            repository.stars < settings.min_stars
            or (repository.is_fork and not settings.include_forks)
            or (repository.archived and not settings.include_archived)
            or repository.disabled
        )

    async def enrich(self, *, limit: int = 100, only_unenriched: bool = True) -> RunSummary:
        self.database.initialize()
        rows = self.database.candidate_repositories(limit=limit, only_unenriched=only_unenriched)
        summary = RunSummary(repositories_seen=len(rows))
        semaphore = asyncio.Semaphore(self.config.github.concurrency)
        async with GitHubClient(self.config.github, self.token) as client:
            async def worker(row: dict[str, Any]) -> None:
                async with semaphore:
                    try:
                        await self._enrich_one(client, row)
                        summary.repositories_enriched += 1
                    except Exception:
                        summary.errors += 1
            await asyncio.gather(*(worker(row) for row in rows))
        return summary

    async def _enrich_one(self, client: GitHubClient, row: dict[str, Any]) -> None:
        full_name = str(row["full_name"])
        repository = RepositoryRecord.from_github(await client.get_repository(full_name), source_query=row.get("source_query"))
        readme, topics, license_spdx, (tree_paths, tree_truncated) = await asyncio.gather(
            client.get_readme(full_name),
            client.get_topics(full_name),
            client.get_license_spdx(full_name),
            client.get_tree(full_name, repository.default_branch),
        )
        repository.readme_text = readme[:2_500_000]
        repository.topics = topics
        repository.license_spdx = license_spdx or repository.license_spdx
        repository.tree_paths = tree_paths[:25_000]
        citation_paths = [path for path in repository.tree_paths if CITATION_PATH_RE.search(path)][:10]
        citation_files: dict[str, str] = {}
        if citation_paths:
            contents = await asyncio.gather(*(client.get_file_text(full_name, path, ref=repository.default_branch) for path in citation_paths), return_exceptions=True)
            for path, content in zip(citation_paths, contents, strict=True):
                if isinstance(content, str) and content.strip():
                    citation_files[path] = content
        artifacts = self.extractor.extract(readme_text=repository.readme_text, citation_files=citation_files, tree_paths=repository.tree_paths)
        score = self.scorer.score(repository, artifacts=artifacts)
        async with self._database_lock:
            self.database.update_enrichment(repository, tree_truncated=tree_truncated, artifacts=artifacts, score=score)

    def derive_and_store_keywords(self, *, minimum_score: float = 25, minimum_documents: int = 2, limit: int = 100):
        documents = self.database.documents(minimum_score=minimum_score)
        seeds = self.config.ontology.get("tasks", []) + self.config.ontology.get("argumentation", []) + self.config.ontology.get("jurisdictions", [])
        keywords = derive_keywords(documents, seeds, minimum_documents=minimum_documents, limit=limit)
        self.database.save_derived_keywords(keywords)
        return keywords

    async def snapshot(self, *, directory: str | Path | None = None, minimum_score: float | None = None, maximum_count: int | None = None, maximum_repo_mb: int | None = None) -> list[dict[str, Any]]:
        settings = self.config.snapshot
        destination = Path(directory or settings.directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        min_score = settings.minimum_score if minimum_score is None else minimum_score
        max_count = maximum_count or settings.max_count
        max_mb = maximum_repo_mb or settings.max_repo_mb
        candidates = self.database.candidate_repositories(limit=max_count * 4, only_unenriched=False, minimum_score=min_score)
        manifests: list[dict[str, Any]] = []
        async with GitHubClient(self.config.github, self.token) as client:
            for row in candidates:
                if len(manifests) >= max_count:
                    break
                estimated_mb = float(row.get("size_kb") or 0) / 1024
                if estimated_mb > max_mb:
                    continue
                full_name = str(row["full_name"])
                branch = str(row.get("default_branch") or "main")
                commit_sha = str((await client.get_commit(full_name, branch))["sha"])
                archive_path = destination / f"{full_name.replace('/', '__')}__{commit_sha[:12]}.tar.gz"
                await client.download_tarball(full_name, commit_sha, archive_path)
                digest = hashlib.sha256()
                with archive_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                manifest = {"github_id": row["github_id"], "full_name": full_name, "source_url": row["html_url"], "score": row["score"], "license_spdx": row.get("license_spdx"), "commit_sha": commit_sha, "archive_path": str(archive_path), "sha256": digest.hexdigest()}
                manifests.append(manifest)
                self.database.mark_snapshot(int(row["github_id"]), path=str(archive_path), sha256=digest.hexdigest(), commit_sha=commit_sha)
        with (destination / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            for manifest in manifests:
                handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        return manifests
