"""End-to-end discovery pipelines with visual progress events."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

from .config import Settings
from .extractor import ResearchExtractor
from .fetchers import PublicPageFetcher
from .github_client import GitHubResearchClient
from .http_client import RadarHttpError
from .models import (
    EventLevel,
    QueryPlan,
    ResearchCandidate,
    RunEvent,
    RunKind,
    RunSummary,
    Stage,
)
from .papers import ScholarlyResearchClient
from .query_expansion import LSIExpander, QueryPlanner, Taxonomy
from .ranking import CandidateRanker
from .storage import RadarStore


EventCallback = Callable[[RunEvent], None]


@dataclass(slots=True)
class GitHubRunOptions:
    seed: str = "legal judgment prediction litigation strategy"
    selected_families: list[str] = field(default_factory=list)
    max_queries: int = 24
    results_per_query: int = 20
    pages_per_query: int = 1
    enrich_top_n: int = 80
    include_repository_tree: bool = True
    include_languages: bool = True
    enable_lsi_second_pass: bool = True
    lsi_query_count: int = 8
    llm_enrich_top_n: int = 20


@dataclass(slots=True)
class PaperRunOptions:
    seed: str = "litigation prediction game theory behavioral law and economics"
    selected_families: list[str] = field(default_factory=list)
    providers: list[str] = field(
        default_factory=lambda: ["OpenAlex", "Crossref", "Semantic Scholar", "arXiv"]
    )
    max_queries: int = 18
    results_per_query: int = 15
    enrich_top_n: int = 160
    enable_lsi_second_pass: bool = True
    lsi_query_count: int = 8
    enrich_public_pages: bool = False
    public_page_top_n: int = 12
    user_approved_domains: list[str] = field(default_factory=list)
    llm_enrich_top_n: int = 25


@dataclass(slots=True)
class PipelineResult:
    summary: RunSummary
    candidates: list[ResearchCandidate]
    events: list[RunEvent]


class ResearchRadarPipeline:
    GITHUB_STAGE_OFFSETS = {
        Stage.PLAN: (0.00, 0.05),
        Stage.DISCOVER: (0.05, 0.43),
        Stage.EXPAND: (0.48, 0.10),
        Stage.ENRICH: (0.58, 0.22),
        Stage.SCORE: (0.80, 0.15),
        Stage.PERSIST: (0.95, 0.05),
        Stage.COMPLETE: (1.00, 0.00),
    }
    PAPER_STAGE_OFFSETS = {
        Stage.PLAN: (0.00, 0.05),
        Stage.DISCOVER: (0.05, 0.45),
        Stage.EXPAND: (0.50, 0.10),
        Stage.ACCESS: (0.60, 0.10),
        Stage.ENRICH: (0.70, 0.15),
        Stage.SCORE: (0.85, 0.10),
        Stage.PERSIST: (0.95, 0.05),
        Stage.COMPLETE: (1.00, 0.00),
    }

    def __init__(
        self,
        settings: Settings,
        taxonomy: Taxonomy | None = None,
        store: RadarStore | None = None,
    ) -> None:
        self.settings = settings
        self.taxonomy = taxonomy or Taxonomy.load(settings.taxonomy_path)
        self.store = store or RadarStore(settings.database_path)
        self.planner = QueryPlanner(self.taxonomy)
        self.lsi = LSIExpander()
        self.extractor = ResearchExtractor(settings, self.taxonomy)
        self.ranker = CandidateRanker(self.taxonomy)

    @staticmethod
    def _new_run_id(kind: RunKind) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{kind.value}-{timestamp}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _stage_progress(
        stage: Stage,
        stage_progress: float,
        offsets: dict[Stage, tuple[float, float]],
    ) -> float:
        start, width = offsets[stage]
        return min(1.0, start + width * max(0.0, min(1.0, stage_progress)))

    def _emitter(
        self,
        summary: RunSummary,
        offsets: dict[Stage, tuple[float, float]],
        callback: EventCallback | None,
        events: list[RunEvent],
    ) -> Callable[..., RunEvent]:
        def emit(
            stage: Stage,
            plain_english: str,
            *,
            stage_progress: float = 0.0,
            technical_detail: str = "",
            level: EventLevel = EventLevel.INFO,
            counters: dict[str, int | float | str] | None = None,
        ) -> RunEvent:
            event = RunEvent(
                run_id=summary.run_id,
                run_kind=summary.run_kind,
                stage=stage,
                level=level,
                progress=self._stage_progress(stage, stage_progress, offsets),
                stage_progress=max(0.0, min(1.0, stage_progress)),
                plain_english=plain_english,
                technical_detail=technical_detail,
                counters=counters or {},
            )
            events.append(event)
            self.store.save_event(event)
            if callback:
                callback(event)
            return event

        return emit

    @staticmethod
    def _merge_repositories(
        candidates: Iterable[ResearchCandidate],
    ) -> list[ResearchCandidate]:
        merged: dict[str, ResearchCandidate] = {}
        for candidate in candidates:
            key = (candidate.repository_full_name or candidate.url).lower()
            existing = merged.get(key)
            if existing is None:
                merged[key] = candidate
                continue
            preferred = candidate if len(candidate.description) > len(existing.description) else existing
            preferred = preferred.model_copy(
                update={
                    "stars": max(existing.stars, candidate.stars),
                    "forks": max(existing.forks, candidate.forks),
                    "watchers": max(existing.watchers, candidate.watchers),
                    "topics": list(dict.fromkeys(existing.topics + candidate.topics)),
                    "discovered_by": list(
                        dict.fromkeys(existing.discovered_by + candidate.discovered_by)
                    ),
                    "raw": {**existing.raw, **candidate.raw},
                }
            )
            merged[key] = preferred
        return list(merged.values())

    @staticmethod
    def _preselect_repositories(
        candidates: Iterable[ResearchCandidate], limit: int
    ) -> list[ResearchCandidate]:
        def quick_score(candidate: ResearchCandidate) -> float:
            query_hits = len(candidate.discovered_by)
            artifact_words = sum(
                word in candidate.searchable_text.lower()
                for word in (
                    "dataset",
                    "benchmark",
                    "model",
                    "pipeline",
                    "paper",
                    "legal",
                    "court",
                    "litigation",
                )
            )
            return query_hits * 20 + artifact_words * 5 + math.log1p(candidate.stars) * 3

        return sorted(candidates, key=quick_score, reverse=True)[: max(0, limit)]

    def run_github(
        self,
        options: GitHubRunOptions,
        *,
        callback: EventCallback | None = None,
    ) -> PipelineResult:
        run_id = self._new_run_id(RunKind.GITHUB)
        summary = RunSummary(run_id=run_id, run_kind=RunKind.GITHUB)
        self.store.create_run(summary)
        events: list[RunEvent] = []
        emit = self._emitter(
            summary, self.GITHUB_STAGE_OFFSETS, callback, events
        )
        github: GitHubResearchClient | None = None
        try:
            selected = options.selected_families or list(self.taxonomy.families)
            plan = self.planner.build_plan(
                RunKind.GITHUB,
                seed=options.seed,
                selected_families=selected,
                max_queries=options.max_queries,
            )
            summary = summary.model_copy(update={"query_plan": plan})
            self.store.update_run(summary)
            emit(
                Stage.PLAN,
                f"Built a bounded plan with {len(plan.generated_queries)} GitHub searches.",
                stage_progress=1.0,
                technical_detail="Queries combine legal research concepts with dataset/model/pipeline signals.",
                counters={"queries": len(plan.generated_queries)},
            )

            remote_context: dict[str, object] = {
                "stage": Stage.DISCOVER,
                "progress": 0.0,
            }

            def github_wait(seconds: float, reason: str) -> None:
                emit(
                    remote_context["stage"],
                    f"GitHub asked the radar to pause for {seconds:.1f} seconds; the run is waiting safely.",
                    stage_progress=float(remote_context["progress"]),
                    technical_detail=reason,
                    level=EventLevel.WARNING,
                    counters={"rate_limit_wait_seconds": round(seconds, 1)},
                )

            github = GitHubResearchClient(
                self.settings, self.taxonomy, on_wait=github_wait
            )
            discovered: list[ResearchCandidate] = []
            errors = 0
            for index, query in enumerate(plan.generated_queries, start=1):
                remote_context.update(
                    {
                        "stage": Stage.DISCOVER,
                        "progress": (index - 1) / max(len(plan.generated_queries), 1),
                    }
                )
                emit(
                    Stage.DISCOVER,
                    f"Searching GitHub query {index} of {len(plan.generated_queries)}.",
                    stage_progress=(index - 1) / max(len(plan.generated_queries), 1),
                    technical_detail=query,
                    counters={
                        "queries_complete": index - 1,
                        "candidates_seen": len(discovered),
                    },
                )
                try:
                    batch = github.search_repositories(
                        query,
                        per_page=options.results_per_query,
                        pages=options.pages_per_query,
                    )
                    discovered.extend(batch.candidates)
                    if batch.total_count > 1000:
                        emit(
                            Stage.DISCOVER,
                            "GitHub reported more than 1,000 matches for this query; the bounded run keeps the highest-ranked slice.",
                            stage_progress=index / max(len(plan.generated_queries), 1),
                            technical_detail=(
                                "GitHub search exposes at most 1,000 results per query. Use narrower families or date partitions for deeper harvesting."
                            ),
                            level=EventLevel.WARNING,
                            counters={"reported_matches": batch.total_count},
                        )
                except RadarHttpError as exc:
                    errors += 1
                    emit(
                        Stage.DISCOVER,
                        "One GitHub query failed; the run continued with the remaining searches.",
                        stage_progress=index / max(len(plan.generated_queries), 1),
                        technical_detail=str(exc),
                        level=EventLevel.WARNING,
                        counters={"errors": errors},
                    )
                if index < len(plan.generated_queries):
                    time.sleep(self.settings.github_search_pause_seconds)

            candidates = self._merge_repositories(discovered)
            emit(
                Stage.DISCOVER,
                f"Found {len(candidates):,} unique public repositories after deduplication.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={
                    "raw_hits": len(discovered),
                    "unique_repositories": len(candidates),
                    "errors": errors,
                },
            )

            lsi_terms: list[str] = []
            if options.enable_lsi_second_pass and candidates:
                corpus = [candidate.searchable_text for candidate in candidates]
                seed_terms = [options.seed] + list(self.taxonomy.reference_terms[:12])
                lsi_terms = self.lsi.expand(corpus, seed_terms, top_n=30)
                emit(
                    Stage.EXPAND,
                    f"Learned {len(lsi_terms)} related terms from the first-pass repository corpus.",
                    stage_progress=0.35,
                    technical_detail=", ".join(lsi_terms[:20]),
                    counters={"lsi_terms": len(lsi_terms)},
                )
                if lsi_terms and options.lsi_query_count > 0:
                    lsi_plan = self.planner.build_plan(
                        RunKind.GITHUB,
                        seed=options.seed,
                        selected_families=[],
                        max_queries=options.lsi_query_count,
                        lsi_terms=lsi_terms,
                        include_reference_profile=False,
                    )
                    existing_queries = set(plan.generated_queries)
                    extra_queries = [
                        query
                        for query in lsi_plan.generated_queries
                        if query not in existing_queries
                    ][: options.lsi_query_count]
                    for index, query in enumerate(extra_queries, start=1):
                        remote_context.update(
                            {
                                "stage": Stage.EXPAND,
                                "progress": 0.35
                                + 0.6 * (index - 1) / max(len(extra_queries), 1),
                            }
                        )
                        emit(
                            Stage.EXPAND,
                            f"Running semantic follow-up query {index} of {len(extra_queries)}.",
                            stage_progress=0.35
                            + 0.6 * (index - 1) / max(len(extra_queries), 1),
                            technical_detail=query,
                            counters={"semantic_queries_complete": index - 1},
                        )
                        try:
                            batch = github.search_repositories(
                                query,
                                per_page=options.results_per_query,
                                pages=options.pages_per_query,
                            )
                            discovered.extend(batch.candidates)
                        except RadarHttpError as exc:
                            errors += 1
                            emit(
                                Stage.EXPAND,
                                "A semantic follow-up query failed; the run continued.",
                                stage_progress=0.35
                                + 0.6 * index / max(len(extra_queries), 1),
                                technical_detail=str(exc),
                                level=EventLevel.WARNING,
                            )
                        if index < len(extra_queries):
                            time.sleep(self.settings.github_search_pause_seconds)
                    candidates = self._merge_repositories(discovered)
                    plan = plan.model_copy(
                        update={
                            "generated_queries": plan.generated_queries + extra_queries,
                            "lsi_terms": lsi_terms,
                            "budget": len(plan.generated_queries) + len(extra_queries),
                        }
                    )
                    summary = summary.model_copy(update={"query_plan": plan})
                    self.store.update_run(summary)
            emit(
                Stage.EXPAND,
                "Semantic expansion is complete.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={"unique_repositories": len(candidates)},
            )

            enrich_targets = self._preselect_repositories(
                candidates, options.enrich_top_n
            )
            enriched_by_id: dict[str, ResearchCandidate] = {
                candidate.candidate_id: candidate for candidate in candidates
            }
            for index, candidate in enumerate(enrich_targets, start=1):
                remote_context.update(
                    {
                        "stage": Stage.ENRICH,
                        "progress": 0.28
                        * (index - 1) / max(len(enrich_targets), 1),
                    }
                )
                emit(
                    Stage.ENRICH,
                    f"Reading public repository metadata {index} of {len(enrich_targets)}.",
                    stage_progress=(index - 1) / max(len(enrich_targets), 1),
                    technical_detail=candidate.repository_full_name or candidate.title,
                    counters={
                        "repositories_enriched": index - 1,
                        "repositories_selected": len(enrich_targets),
                    },
                )
                try:
                    enriched = github.enrich_repository(
                        candidate,
                        include_tree=options.include_repository_tree,
                        include_languages=options.include_languages,
                    )
                except Exception as exc:  # keep a long research run alive
                    errors += 1
                    enriched = candidate.model_copy(
                        update={"risks": candidate.risks + [f"Enrichment failed: {type(exc).__name__}"]}
                    )
                enriched_by_id[candidate.candidate_id] = enriched

            # Classify every repository deterministically. ScrapeGraphAI is reserved for
            # a bounded, preliminary top-ranked slice so an accidental broad run cannot
            # create an unbounded LLM bill or hide progress for the team.
            deterministic_candidates: list[ResearchCandidate] = []
            all_candidates = list(enriched_by_id.values())
            for index, candidate in enumerate(all_candidates, start=1):
                extracted = self.extractor.enrich_candidate(candidate, use_llm=False)
                deterministic_candidates.append(extracted)
                if index % 25 == 0 or index == len(all_candidates):
                    emit(
                        Stage.ENRICH,
                        f"Classified {index:,} of {len(all_candidates):,} repositories in plain research terms.",
                        stage_progress=0.72 * index / max(len(all_candidates), 1),
                        counters={"repositories_classified": index},
                    )

            preliminary_ranked = self.ranker.rank(deterministic_candidates)
            enriched_by_id = {
                candidate.candidate_id: candidate for candidate in deterministic_candidates
            }
            llm_target_count = (
                min(max(options.llm_enrich_top_n, 0), len(preliminary_ranked))
                if self.settings.enable_llm_enrichment
                else 0
            )
            for index, candidate in enumerate(
                preliminary_ranked[:llm_target_count], start=1
            ):
                emit(
                    Stage.ENRICH,
                    f"ScrapeGraphAI is structuring high-fit repository {index} of {llm_target_count}.",
                    stage_progress=0.72
                    + 0.26 * (index - 1) / max(llm_target_count, 1),
                    technical_detail=candidate.repository_full_name or candidate.title,
                    counters={"scrapegraphai_complete": index - 1},
                )
                enriched_by_id[candidate.candidate_id] = self.extractor.enrich_candidate(
                    candidate, use_llm=True
                )

            enriched_candidates = list(enriched_by_id.values())
            emit(
                Stage.ENRICH,
                "Repository enrichment and research classification are complete.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={
                    "repositories_classified": len(enriched_candidates),
                    "scrapegraphai_complete": llm_target_count,
                },
            )

            ranked = self.ranker.rank(enriched_candidates)
            emit(
                Stage.SCORE,
                "Ranked every repository by Juricratic fit, data value, methods, credibility, and reproducibility.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={
                    "ranked": len(ranked),
                    "top_score": ranked[0].score.overall if ranked else 0,
                },
            )

            saved = self.store.save_candidates(run_id, ranked)
            summary = summary.model_copy(
                update={
                    "completed_at": datetime.now(timezone.utc),
                    "status": "completed",
                    "candidates_found": len(candidates),
                    "candidates_saved": saved,
                    "errors": errors,
                    "top_score": ranked[0].score.overall if ranked else 0.0,
                    "notes": [
                        "GitHub was queried through its official REST API.",
                        "The run stores bounded public metadata and selected artifact paths, not a mirror of all GitHub content.",
                    ],
                }
            )
            self.store.update_run(summary)
            emit(
                Stage.PERSIST,
                f"Saved {saved:,} ranked repositories to the local research index.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={"saved": saved},
            )
            emit(
                Stage.COMPLETE,
                "GitHub research discovery is complete and ready for team review.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={"saved": saved, "errors": errors},
            )
            return PipelineResult(summary=summary, candidates=ranked, events=events)
        except Exception as exc:
            summary = summary.model_copy(
                update={
                    "completed_at": datetime.now(timezone.utc),
                    "status": "failed",
                    "errors": summary.errors + 1,
                    "notes": summary.notes + [str(exc)],
                }
            )
            self.store.update_run(summary)
            emit(
                Stage.COMPLETE,
                "The GitHub run stopped because of an unrecoverable error.",
                stage_progress=1.0,
                technical_detail=str(exc),
                level=EventLevel.ERROR,
            )
            raise
        finally:
            if github:
                github.close()

    @staticmethod
    def _paper_landing_url(candidate: ResearchCandidate) -> str | None:
        if candidate.open_access_url and not candidate.open_access_url.lower().endswith(
            ".pdf"
        ):
            return candidate.open_access_url
        if candidate.url and "doi.org" not in candidate.url:
            return candidate.url
        return candidate.url or None

    def run_papers(
        self,
        options: PaperRunOptions,
        *,
        callback: EventCallback | None = None,
    ) -> PipelineResult:
        run_id = self._new_run_id(RunKind.PAPERS)
        summary = RunSummary(run_id=run_id, run_kind=RunKind.PAPERS)
        self.store.create_run(summary)
        events: list[RunEvent] = []
        emit = self._emitter(summary, self.PAPER_STAGE_OFFSETS, callback, events)
        scholarly: ScholarlyResearchClient | None = None
        fetcher: PublicPageFetcher | None = None
        try:
            selected = options.selected_families or list(self.taxonomy.families)
            plan = self.planner.build_plan(
                RunKind.PAPERS,
                seed=options.seed,
                selected_families=selected,
                max_queries=options.max_queries,
            )
            summary = summary.model_copy(update={"query_plan": plan})
            self.store.update_run(summary)
            emit(
                Stage.PLAN,
                f"Built a broad paper plan with {len(plan.generated_queries)} concepts across {len(options.providers)} official sources.",
                stage_progress=1.0,
                counters={
                    "queries": len(plan.generated_queries),
                    "providers": len(options.providers),
                },
            )

            remote_context: dict[str, object] = {
                "stage": Stage.DISCOVER,
                "progress": 0.0,
            }

            def scholarly_wait(seconds: float, reason: str) -> None:
                emit(
                    remote_context["stage"],
                    f"A scholarly source asked the radar to pause for {seconds:.1f} seconds; the run is waiting safely.",
                    stage_progress=float(remote_context["progress"]),
                    technical_detail=reason,
                    level=EventLevel.WARNING,
                    counters={"rate_limit_wait_seconds": round(seconds, 1)},
                )

            scholarly = ScholarlyResearchClient(
                self.settings, on_wait=scholarly_wait
            )
            discovered: list[ResearchCandidate] = []
            errors = 0
            total_tasks = len(plan.generated_queries) * max(len(options.providers), 1)
            task_index = 0
            for query in plan.generated_queries:
                for provider in options.providers:
                    task_index += 1
                    remote_context.update(
                        {
                            "stage": Stage.DISCOVER,
                            "progress": (task_index - 1) / max(total_tasks, 1),
                        }
                    )
                    emit(
                        Stage.DISCOVER,
                        f"Searching {provider}: task {task_index} of {total_tasks}.",
                        stage_progress=(task_index - 1) / max(total_tasks, 1),
                        technical_detail=query,
                        counters={
                            "search_tasks_complete": task_index - 1,
                            "records_seen": len(discovered),
                        },
                    )
                    try:
                        batch = scholarly.search_provider(
                            provider, query, limit=options.results_per_query
                        )
                        discovered.extend(batch.candidates)
                    except Exception as exc:
                        errors += 1
                        emit(
                            Stage.DISCOVER,
                            f"{provider} could not complete one query; the other sources continued.",
                            stage_progress=task_index / max(total_tasks, 1),
                            technical_detail=str(exc),
                            level=EventLevel.WARNING,
                            counters={"errors": errors},
                        )
                    time.sleep(self.settings.provider_pause_seconds)

            candidates = scholarly.deduplicate(discovered)
            emit(
                Stage.DISCOVER,
                f"Found {len(candidates):,} unique papers after merging duplicate DOI/title records.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={
                    "raw_records": len(discovered),
                    "unique_papers": len(candidates),
                    "errors": errors,
                },
            )

            lsi_terms: list[str] = []
            if options.enable_lsi_second_pass and candidates:
                corpus = [candidate.searchable_text for candidate in candidates]
                lsi_terms = self.lsi.expand(
                    corpus,
                    [options.seed] + list(self.taxonomy.reference_terms[:12]),
                    top_n=30,
                )
                emit(
                    Stage.EXPAND,
                    f"Learned {len(lsi_terms)} related research phrases from titles and abstracts.",
                    stage_progress=0.4,
                    technical_detail=", ".join(lsi_terms[:20]),
                )
                if lsi_terms and options.lsi_query_count:
                    lsi_plan = self.planner.build_plan(
                        RunKind.PAPERS,
                        seed=options.seed,
                        selected_families=[],
                        max_queries=options.lsi_query_count,
                        lsi_terms=lsi_terms,
                        include_reference_profile=False,
                    )
                    extra_queries = [
                        query
                        for query in lsi_plan.generated_queries
                        if query not in set(plan.generated_queries)
                    ][: options.lsi_query_count]
                    extra_tasks = len(extra_queries) * max(len(options.providers), 1)
                    extra_index = 0
                    for query in extra_queries:
                        for provider in options.providers:
                            extra_index += 1
                            remote_context.update(
                                {
                                    "stage": Stage.EXPAND,
                                    "progress": 0.4
                                    + 0.55 * (extra_index - 1) / max(extra_tasks, 1),
                                }
                            )
                            emit(
                                Stage.EXPAND,
                                f"Semantic paper follow-up {extra_index} of {extra_tasks} via {provider}.",
                                stage_progress=0.4
                                + 0.55 * (extra_index - 1) / max(extra_tasks, 1),
                                technical_detail=query,
                            )
                            try:
                                batch = scholarly.search_provider(
                                    provider, query, limit=options.results_per_query
                                )
                                discovered.extend(batch.candidates)
                            except Exception as exc:
                                errors += 1
                                emit(
                                    Stage.EXPAND,
                                    "A semantic paper follow-up failed; the run continued.",
                                    stage_progress=0.4
                                    + 0.55 * extra_index / max(extra_tasks, 1),
                                    technical_detail=str(exc),
                                    level=EventLevel.WARNING,
                                )
                            time.sleep(self.settings.provider_pause_seconds)
                    candidates = scholarly.deduplicate(discovered)
                    plan = plan.model_copy(
                        update={
                            "generated_queries": plan.generated_queries + extra_queries,
                            "lsi_terms": lsi_terms,
                            "budget": len(plan.generated_queries) + len(extra_queries),
                        }
                    )
                    summary = summary.model_copy(update={"query_plan": plan})
                    self.store.update_run(summary)
            emit(
                Stage.EXPAND,
                "Paper query expansion is complete.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={"unique_papers": len(candidates)},
            )

            candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
            estimated_credits = 0.0
            blocked_urls = 0
            if options.enrich_public_pages and self.settings.allow_full_text_fetch:
                # Pre-rank metadata to avoid fetching every landing page.
                preliminary = self.ranker.rank(
                    self.extractor.enrich_candidate(candidate, use_llm=False)
                    for candidate in candidates
                )
                page_targets = preliminary[: options.public_page_top_n]

                current_access_progress = 0.0

                def audit(action: str, message: str, details: dict) -> None:
                    emit(
                        Stage.ACCESS,
                        message,
                        stage_progress=current_access_progress,
                        technical_detail=f"{action}: {details}",
                    )

                fetcher = PublicPageFetcher(self.settings, audit=audit)
                for index, candidate in enumerate(page_targets, start=1):
                    current_access_progress = (index - 1) / max(len(page_targets), 1)
                    url = self._paper_landing_url(candidate)
                    emit(
                        Stage.ACCESS,
                        f"Checking public paper page {index} of {len(page_targets)}.",
                        stage_progress=(index - 1) / max(len(page_targets), 1),
                        technical_detail=url or "No public landing URL",
                    )
                    if not url:
                        continue
                    result = fetcher.fetch(
                        url,
                        user_approved_domains=options.user_approved_domains,
                    )
                    estimated_credits += result.estimated_credits
                    if result.ok:
                        raw = dict(candidate.raw)
                        raw.update(
                            {
                                "landing_page_excerpt": result.text[:40_000],
                                "landing_page_provider": result.provider,
                                "landing_page_status": result.status_code,
                            }
                        )
                        candidate_map[candidate.candidate_id] = candidate.model_copy(
                            update={
                                "description": "\n".join(
                                    filter(
                                        None,
                                        [candidate.description, result.text[:20_000]],
                                    )
                                ),
                                "raw": raw,
                                "access": result.access,
                            }
                        )
                    elif result.access and result.access.outcome.value == "blocked":
                        blocked_urls += 1
                emit(
                    Stage.ACCESS,
                    "Public-page access checks are complete.",
                    stage_progress=1.0,
                    level=EventLevel.SUCCESS,
                    counters={
                        "pages_checked": len(page_targets),
                        "blocked": blocked_urls,
                        "estimated_proxy_credits": estimated_credits,
                    },
                )
            else:
                emit(
                    Stage.ACCESS,
                    "Full-text page fetching is off; the run used official paper metadata only.",
                    stage_progress=1.0,
                    level=EventLevel.SUCCESS,
                    technical_detail=(
                        "Enable both the run option and JURICRATIC_ALLOW_FULL_TEXT_FETCH to inspect approved public landing pages."
                    ),
                )

            deterministic_papers: list[ResearchCandidate] = []
            values = list(candidate_map.values())
            for index, candidate in enumerate(values, start=1):
                extracted = self.extractor.enrich_candidate(candidate, use_llm=False)
                deterministic_papers.append(extracted)
                if index % 25 == 0 or index == len(values):
                    emit(
                        Stage.ENRICH,
                        f"Classified {index:,} of {len(values):,} papers into reusable legal-research priors.",
                        stage_progress=0.72 * index / max(len(values), 1),
                        counters={"papers_classified": index},
                    )

            preliminary_ranked = self.ranker.rank(deterministic_papers)
            enriched_by_id = {
                candidate.candidate_id: candidate for candidate in deterministic_papers
            }
            llm_target_count = (
                min(max(options.llm_enrich_top_n, 0), len(preliminary_ranked))
                if self.settings.enable_llm_enrichment
                else 0
            )
            for index, candidate in enumerate(
                preliminary_ranked[:llm_target_count], start=1
            ):
                emit(
                    Stage.ENRICH,
                    f"ScrapeGraphAI is structuring high-fit paper {index} of {llm_target_count}.",
                    stage_progress=0.72
                    + 0.26 * (index - 1) / max(llm_target_count, 1),
                    technical_detail=candidate.title,
                    counters={"scrapegraphai_complete": index - 1},
                )
                enriched_by_id[candidate.candidate_id] = self.extractor.enrich_candidate(
                    candidate, use_llm=True
                )

            enriched = list(enriched_by_id.values())
            emit(
                Stage.ENRICH,
                "Paper classification is complete.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={
                    "papers_classified": len(enriched),
                    "scrapegraphai_complete": llm_target_count,
                },
            )

            ranked = self.ranker.rank(enriched)
            if options.enrich_top_n > 0:
                ranked = ranked[: max(options.enrich_top_n, 1)]
            emit(
                Stage.SCORE,
                "Ranked papers by Juricratic fit, methods, data access, validation, credibility, and reproducibility.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={
                    "ranked": len(ranked),
                    "top_score": ranked[0].score.overall if ranked else 0,
                },
            )

            saved = self.store.save_candidates(run_id, ranked)
            summary = summary.model_copy(
                update={
                    "completed_at": datetime.now(timezone.utc),
                    "status": "completed",
                    "candidates_found": len(candidates),
                    "candidates_saved": saved,
                    "blocked_urls": blocked_urls,
                    "errors": errors,
                    "estimated_proxy_credits": round(estimated_credits, 2),
                    "top_score": ranked[0].score.overall if ranked else 0.0,
                    "notes": [
                        "Paper discovery used official OpenAlex, Crossref, Semantic Scholar, and/or arXiv APIs.",
                        "Publisher authentication and paywalls are hard stops; public-page enrichment is opt-in.",
                    ],
                }
            )
            self.store.update_run(summary)
            emit(
                Stage.PERSIST,
                f"Saved {saved:,} ranked papers to the local research index.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={"saved": saved},
            )
            emit(
                Stage.COMPLETE,
                "Research-paper discovery is complete and ready for team review.",
                stage_progress=1.0,
                level=EventLevel.SUCCESS,
                counters={
                    "saved": saved,
                    "blocked": blocked_urls,
                    "errors": errors,
                },
            )
            return PipelineResult(summary=summary, candidates=ranked, events=events)
        except Exception as exc:
            summary = summary.model_copy(
                update={
                    "completed_at": datetime.now(timezone.utc),
                    "status": "failed",
                    "errors": summary.errors + 1,
                    "notes": summary.notes + [str(exc)],
                }
            )
            self.store.update_run(summary)
            emit(
                Stage.COMPLETE,
                "The paper run stopped because of an unrecoverable error.",
                stage_progress=1.0,
                technical_detail=str(exc),
                level=EventLevel.ERROR,
            )
            raise
        finally:
            if fetcher:
                fetcher.close()
            if scholarly:
                scholarly.close()
