"""Explainable Juricratic relevance and reproducibility scoring."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

from .models import CandidateKind, ResearchCandidate, ScoreBreakdown
from .query_expansion import Taxonomy


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _normalize_match_text(value: str) -> str:
    """Normalize identifiers and prose into the same matching surface."""

    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _coverage(text: str, terms: Iterable[str], *, saturation: int = 5) -> tuple[float, list[str]]:
    normalized_text = _normalize_match_text(text)
    matched = [
        term
        for term in terms
        if _normalize_match_text(term) in normalized_text
    ]
    matched = list(dict.fromkeys(matched))
    score = 100.0 * min(len(matched), saturation) / max(saturation, 1)
    return score, matched


class CandidateRanker:
    """Rank results for research value rather than raw GitHub popularity."""

    DIMENSION_FAMILIES = {
        "outcome_forecasting": ("legal_prediction", "causal_legal_analytics"),
        "matter_state_modeling": ("temporal_dynamic_models", "legal_argumentation"),
        "strategy_and_game_theory": (
            "game_theoretic_litigation",
            "litigation_economics",
            "multiagent_simulation",
        ),
        "behavior_and_actors": (
            "behavioral_law_economics",
            "judicial_and_counsel_behavior",
        ),
        "evidence_and_reasoning": (
            "evidence_and_credibility",
            "legal_argumentation",
            "legal_network_science",
        ),
        "data_and_infrastructure": (
            "datasets_and_benchmarks",
            "legal_nlp_models",
        ),
    }

    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy

    def _fit(self, candidate: ResearchCandidate) -> tuple[float, dict[str, float], list[str]]:
        text = candidate.searchable_text
        dimensions: dict[str, float] = {}
        matched_all: list[str] = []
        for dimension, family_names in self.DIMENSION_FAMILIES.items():
            weighted = 0.0
            weight_sum = 0.0
            for family_name in family_names:
                family = self.taxonomy.families.get(family_name)
                if not family:
                    continue
                coverage, matched = _coverage(text, family.phrases, saturation=4)
                weighted += coverage * family.weight
                weight_sum += family.weight
                matched_all.extend(matched)
            dimensions[dimension] = _clamp(weighted / weight_sum if weight_sum else 0.0)
        fit = sum(dimensions.values()) / max(len(dimensions), 1)
        # Reward cross-dimensional work because Juricratic combines law, behavior,
        # strategy, evidence, and computation rather than solving one narrow task.
        active = sum(1 for value in dimensions.values() if value >= 25)
        fit = _clamp(fit + max(0, active - 2) * 4.0)
        return fit, dimensions, list(dict.fromkeys(matched_all))

    def _reproducibility(self, candidate: ResearchCandidate) -> float:
        score = 0.0
        if candidate.license and candidate.license.lower() not in {"noassertion", "other"}:
            score += 22
        if candidate.readme_excerpt:
            score += 12
        if candidate.artifact_paths:
            score += min(28, 8 + len(candidate.artifact_paths) * 1.5)
        if candidate.extraction:
            if candidate.extraction.validation_design:
                score += 15
            if candidate.extraction.code_artifacts:
                score += 10
            if candidate.extraction.datasets:
                score += 8
        if candidate.kind == CandidateKind.RESEARCH_PAPER and candidate.open_access:
            score += 18
        if candidate.raw.get("archived"):
            score -= 25
        return _clamp(score)

    def _data_value(self, candidate: ResearchCandidate) -> float:
        text = candidate.searchable_text.lower()
        score = 0.0
        high_value = self.taxonomy.high_value_artifact_terms
        matched = [term for term in high_value if term.lower() in text]
        score += min(45, len(set(matched)) * 6.0)
        data_paths = [
            path
            for path in candidate.artifact_paths
            if path.lower().endswith(
                (".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx", ".ann", ".xml")
            )
        ]
        score += min(30, len(data_paths) * 4.0)
        if candidate.extraction:
            if candidate.extraction.dataset_size:
                score += 12
            score += min(15, len(candidate.extraction.datasets) * 4.0)
        if candidate.open_access_url:
            score += 8
        return _clamp(score)

    def _methodological_value(self, candidate: ResearchCandidate) -> float:
        extraction = candidate.extraction
        methods = candidate.methods + (extraction.methods if extraction else [])
        method_text = " ".join(methods).lower()
        groups_hit = 0
        total_matches = 0
        for group, terms in self.taxonomy.method_terms.items():
            matched = [term for term in terms if term.lower() in method_text]
            if matched:
                groups_hit += 1
                total_matches += len(matched)
        score = groups_hit * 18 + min(total_matches, 10) * 3
        if extraction and extraction.validation_design:
            score += 18
        if any(
            term in candidate.searchable_text.lower()
            for term in ("temporal validation", "external validation", "out of sample")
        ):
            score += 12
        return _clamp(score)

    @staticmethod
    def _credibility(candidate: ResearchCandidate) -> float:
        score = 20.0
        if candidate.kind == CandidateKind.RESEARCH_PAPER:
            score += min(40, math.log1p(candidate.citation_count) * 7.0)
            if candidate.doi:
                score += 12
            if candidate.authors:
                score += 8
            if candidate.abstract:
                score += 8
            if candidate.open_access:
                score += 8
        else:
            score += min(30, math.log1p(candidate.stars) * 5.0)
            if candidate.readme_excerpt:
                score += 12
            if candidate.raw.get("fork"):
                score -= 8
            if candidate.raw.get("disabled"):
                score -= 30
        if candidate.extraction and candidate.extraction.validation_design:
            score += 15
        return _clamp(score)

    @staticmethod
    def _freshness(candidate: ResearchCandidate) -> float:
        now = datetime.now(timezone.utc)
        if candidate.updated_at:
            age_years = max(0.0, (now - candidate.updated_at).days / 365.25)
        elif candidate.year:
            age_years = max(0.0, now.year - candidate.year)
        else:
            return 35.0
        return _clamp(100.0 * math.exp(-age_years / 6.0))

    @staticmethod
    def _popularity(candidate: ResearchCandidate) -> float:
        if candidate.kind == CandidateKind.GITHUB_REPOSITORY:
            raw = candidate.stars + candidate.forks * 1.5
        else:
            raw = candidate.citation_count
        return _clamp(math.log1p(max(raw, 0)) * 12.0)

    def score(self, candidate: ResearchCandidate) -> ResearchCandidate:
        fit, dimensions, matched = self._fit(candidate)
        reproducibility = self._reproducibility(candidate)
        data_value = self._data_value(candidate)
        methodology = self._methodological_value(candidate)
        credibility = self._credibility(candidate)
        freshness = self._freshness(candidate)
        popularity = self._popularity(candidate)
        overall = _clamp(
            fit * 0.32
            + reproducibility * 0.18
            + data_value * 0.16
            + methodology * 0.16
            + credibility * 0.10
            + freshness * 0.05
            + popularity * 0.03
        )

        reasons: list[str] = []
        top_dimensions = sorted(dimensions.items(), key=lambda item: item[1], reverse=True)
        for name, value in top_dimensions[:3]:
            if value >= 20:
                reasons.append(f"{name.replace('_', ' ').title()}: {value:.0f}/100")
        if reproducibility >= 65:
            reasons.append("Strong reproducibility signals: license, documentation, artifacts, or validation")
        if data_value >= 60:
            reasons.append("High-value dataset, annotation, model, or pipeline signals")
        if candidate.extraction and candidate.extraction.juricratic_relevance:
            reasons.append(candidate.extraction.juricratic_relevance)

        risks = list(candidate.risks)
        if candidate.kind == CandidateKind.GITHUB_REPOSITORY:
            if not candidate.license:
                risks.append("No explicit repository license detected")
            if candidate.raw.get("archived"):
                risks.append("Repository is archived")
            if not candidate.artifact_paths:
                risks.append("No high-value data/model artifact paths detected in the sampled tree")
        else:
            if candidate.open_access is False:
                risks.append("Paper is not marked open access")
            if not candidate.abstract:
                risks.append("No abstract returned by the selected metadata provider")
        if not candidate.extraction or not candidate.extraction.validation_design:
            risks.append("Validation design was not detected and should be reviewed manually")

        return candidate.model_copy(
            update={
                "score": ScoreBreakdown(
                    overall=round(overall, 2),
                    juricratic_fit=round(fit, 2),
                    reproducibility=round(reproducibility, 2),
                    data_value=round(data_value, 2),
                    methodological_value=round(methodology, 2),
                    credibility=round(credibility, 2),
                    freshness=round(freshness, 2),
                    popularity=round(popularity, 2),
                    dimensions={key: round(value, 2) for key, value in dimensions.items()},
                ),
                "matched_terms": list(dict.fromkeys(candidate.matched_terms + matched))[:80],
                "reasons": list(dict.fromkeys(candidate.reasons + reasons))[:12],
                "risks": list(dict.fromkeys(risks))[:12],
            }
        )

    def rank(self, candidates: Iterable[ResearchCandidate]) -> list[ResearchCandidate]:
        scored = [self.score(candidate) for candidate in candidates]
        return sorted(scored, key=lambda candidate: candidate.score.overall, reverse=True)
