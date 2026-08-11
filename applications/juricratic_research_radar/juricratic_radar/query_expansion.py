"""Semantic query planning and corpus-driven LSI expansion."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .models import QueryPlan, RunKind


_SPACE_RE = re.compile(r"\s+")


def normalize_phrase(value: str) -> str:
    return _SPACE_RE.sub(" ", re.sub(r"[^a-zA-Z0-9+.#/&' -]", " ", value)).strip()


@dataclass(frozen=True, slots=True)
class QueryFamily:
    name: str
    weight: float
    description: str
    phrases: tuple[str, ...]


class Taxonomy:
    def __init__(self, payload: dict):
        self.payload = payload
        self.reference_profile = payload.get("reference_profile", {})
        self.artifact_terms = payload.get("artifact_terms", {})
        self.method_terms = payload.get("method_terms", {})
        self.risk_terms = tuple(payload.get("risk_terms", []))
        self.jurisdiction_terms = tuple(payload.get("jurisdiction_terms", []))
        families: dict[str, QueryFamily] = {}
        for name, raw in payload.get("query_families", {}).items():
            families[name] = QueryFamily(
                name=name,
                weight=float(raw.get("weight", 1.0)),
                description=str(raw.get("description", "")),
                phrases=tuple(str(item) for item in raw.get("phrases", [])),
            )
        self.families = families

    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(payload)

    @property
    def reference_terms(self) -> tuple[str, ...]:
        profile = self.reference_profile
        values: list[str] = []
        for key in ("core_phrases", "methods", "reusable_structure"):
            values.extend(str(item) for item in profile.get(key, []))
        return tuple(values)

    @property
    def high_value_artifact_terms(self) -> tuple[str, ...]:
        return tuple(self.artifact_terms.get("high_value", []))

    @property
    def file_signals(self) -> tuple[str, ...]:
        return tuple(self.artifact_terms.get("file_signals", []))

    @property
    def all_method_terms(self) -> tuple[str, ...]:
        terms: list[str] = []
        for values in self.method_terms.values():
            terms.extend(str(item) for item in values)
        return tuple(dict.fromkeys(terms))

    @property
    def all_domain_terms(self) -> tuple[str, ...]:
        terms: list[str] = []
        for family in self.families.values():
            terms.extend(family.phrases)
        return tuple(dict.fromkeys(terms))


class LSIExpander:
    """Discover related terms from the descriptions/abstracts found in a first pass.

    This is genuine Latent Semantic Indexing: TF-IDF document-term vectors are reduced
    with truncated SVD, then terms nearest the seed centroid in latent space are returned.
    It is optional and degrades cleanly when the corpus is too small.
    """

    def __init__(self, max_features: int = 8000, n_components: int = 32):
        self.max_features = max_features
        self.n_components = n_components

    def expand(
        self,
        corpus: Iterable[str],
        seeds: Iterable[str],
        *,
        top_n: int = 30,
    ) -> list[str]:
        documents = [doc.strip() for doc in corpus if doc and doc.strip()]
        if len(documents) < 4:
            return []
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.decomposition import TruncatedSVD
        except ImportError:
            return []

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(documents)
        max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
        if max_components < 2:
            return []
        components = min(self.n_components, max_components)
        svd = TruncatedSVD(n_components=components, random_state=17)
        svd.fit(matrix)

        terms = vectorizer.get_feature_names_out()
        term_vectors = svd.components_.T * svd.singular_values_
        term_lookup = {term.lower(): index for index, term in enumerate(terms)}

        seed_indices: list[int] = []
        for raw_seed in seeds:
            seed = normalize_phrase(raw_seed).lower()
            if seed in term_lookup:
                seed_indices.append(term_lookup[seed])
                continue
            for token in seed.split():
                if token in term_lookup:
                    seed_indices.append(term_lookup[token])

        seed_indices = list(dict.fromkeys(seed_indices))
        if not seed_indices:
            return []
        centroid = np.mean(term_vectors[seed_indices], axis=0, keepdims=True)
        similarities = cosine_similarity(term_vectors, centroid).reshape(-1)
        ranked = np.argsort(similarities)[::-1]

        seed_set = {normalize_phrase(seed).lower() for seed in seeds}
        output: list[str] = []
        for index in ranked:
            term = str(terms[index])
            score = float(similarities[index])
            if score <= 0:
                break
            if term.lower() in seed_set or len(term) < 3:
                continue
            if term.isdigit() or math.isnan(score):
                continue
            output.append(term)
            if len(output) >= top_n:
                break
        return output


class QueryPlanner:
    """Build bounded, explainable query plans for GitHub and scholarly APIs."""

    GITHUB_ARTIFACT_HINTS = (
        "dataset",
        "benchmark",
        "pipeline",
        "model",
        "annotations",
        "notebook",
        "corpus",
    )

    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy

    def family_catalog(self) -> dict[str, str]:
        return {
            name: family.description for name, family in self.taxonomy.families.items()
        }

    def build_plan(
        self,
        mode: RunKind,
        *,
        seed: str,
        selected_families: Iterable[str] | None = None,
        max_queries: int = 40,
        lsi_terms: Iterable[str] = (),
        include_reference_profile: bool = True,
    ) -> QueryPlan:
        selected = (
            list(self.taxonomy.families.keys())
            if selected_families is None
            else list(selected_families)
        )
        selected = [name for name in selected if name in self.taxonomy.families]

        weighted_phrases: list[tuple[float, str, str]] = []
        for name in selected:
            family = self.taxonomy.families[name]
            for index, phrase in enumerate(family.phrases):
                position_discount = 1.0 / (1.0 + index * 0.025)
                weighted_phrases.append(
                    (family.weight * position_discount, phrase, name)
                )

        if include_reference_profile:
            for index, phrase in enumerate(self.taxonomy.reference_terms):
                weighted_phrases.append((1.08 - min(index * 0.005, 0.2), phrase, "reference"))
        for index, phrase in enumerate(lsi_terms):
            weighted_phrases.append((0.82 - min(index * 0.005, 0.2), phrase, "lsi"))
        if seed.strip():
            weighted_phrases.append((1.2, seed.strip(), "user_seed"))

        weighted_phrases.sort(key=lambda item: (-item[0], item[1].lower()))
        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for _, phrase, family in weighted_phrases:
            normalized = normalize_phrase(phrase)
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            deduped.append((normalized, family))

        queries: list[str] = []
        query_families: dict[str, list[str]] = {}
        if mode == RunKind.GITHUB:
            for index, (phrase, family) in enumerate(deduped):
                if len(queries) >= max_queries:
                    break
                quoted = f'"{phrase}"' if " " in phrase else phrase
                hint = self.GITHUB_ARTIFACT_HINTS[index % len(self.GITHUB_ARTIFACT_HINTS)]
                query = f"{quoted} {hint} in:name,description,readme archived:false"
                if len(query) > 250:
                    query = f"{quoted} in:name,description archived:false"
                queries.append(query)
                query_families.setdefault(family, []).append(query)
        else:
            for phrase, family in deduped:
                if len(queries) >= max_queries:
                    break
                query = phrase
                if seed.strip() and phrase.lower() != seed.strip().lower():
                    seed_words = set(normalize_phrase(seed).lower().split())
                    phrase_words = set(phrase.lower().split())
                    if seed_words and not seed_words.intersection(phrase_words):
                        query = f"{seed.strip()} {phrase}"
                queries.append(query)
                query_families.setdefault(family, []).append(query)

        explanation = [
            "The reference paper contributes court-outcome prediction, factual/legal labels, temporal validation, and explainable decision-rule terms.",
            "The ontology expands across legal NLP, litigation economics, game theory, behavioral economics, causal inference, evidence, agents, judges, and counsel.",
            "The query budget prevents an accidental unbounded crawl; results are deduplicated before enrichment.",
        ]
        if list(lsi_terms):
            explanation.append(
                "A second-pass LSI expansion adds terms learned from the first batch of repository descriptions and paper abstracts."
            )

        return QueryPlan(
            mode=mode,
            seed=seed,
            generated_queries=queries,
            families=query_families,
            lsi_terms=list(lsi_terms),
            budget=len(queries),
            explanation=explanation,
        )
