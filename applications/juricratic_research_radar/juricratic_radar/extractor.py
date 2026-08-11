"""Structured research extraction using rules plus optional ScrapeGraphAI."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .config import Settings
from .models import ArtifactExtraction, ResearchCandidate
from .query_expansion import Taxonomy


_SIZE_PATTERNS = (
    re.compile(r"\b(?:n\s*=\s*)?([\d,]{3,})\s+(?:cases|judgments|rulings|decisions|documents|opinions|samples|records|examples)\b", re.I),
    re.compile(r"\b(?:dataset|corpus|sample)\s+(?:of|with|contains?)\s+([\d,]{3,})\b", re.I),
)
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


class ResearchExtractor:
    """Turn repository/paper text into a comparable research-prior schema."""

    def __init__(self, settings: Settings, taxonomy: Taxonomy):
        self.settings = settings
        self.taxonomy = taxonomy

    @staticmethod
    def _matched_terms(text: str, terms: Iterable[str], *, limit: int = 40) -> list[str]:
        lower = text.lower()
        matches: list[str] = []
        for term in terms:
            normalized = str(term).strip()
            if normalized and normalized.lower() in lower:
                matches.append(normalized)
            if len(matches) >= limit:
                break
        return list(dict.fromkeys(matches))

    def deterministic_extract(self, text: str) -> ArtifactExtraction:
        text = text[:120_000]
        family_matches: dict[str, list[str]] = defaultdict(list)
        for name, family in self.taxonomy.families.items():
            family_matches[name] = self._matched_terms(text, family.phrases, limit=20)

        methods = self._matched_terms(text, self.taxonomy.all_method_terms, limit=50)
        jurisdictions = self._matched_terms(
            text, self.taxonomy.jurisdiction_terms, limit=20
        )
        datasets = self._matched_terms(
            text,
            (
                "CAIL2018",
                "LexGLUE",
                "LegalBench",
                "Pile of Law",
                "CUAD",
                "ECtHR",
                "ECHR",
                "SCOTUS",
                "CourtListener",
                "CENDOJ",
                "Caselaw Access Project",
                "Harvard Caselaw",
                "LEDGAR",
                "EURLEX",
                "MultiEURLEX",
                "CaseHOLD",
                "LexFiles",
                "ContractNLI",
                "COLIEE",
                "Brat",
            ),
            limit=30,
        )
        validations = self._matched_terms(
            text,
            self.taxonomy.method_terms.get("validation", []),
            limit=20,
        )
        limitations = self._matched_terms(text, self.taxonomy.risk_terms, limit=20)
        artifacts = self._matched_terms(
            text, self.taxonomy.high_value_artifact_terms, limit=30
        )

        size = ""
        for pattern in _SIZE_PATTERNS:
            match = pattern.search(text)
            if match:
                size = match.group(0)
                break

        factual = self._matched_terms(
            text,
            (
                "factual findings",
                "case facts",
                "fact pattern",
                "evidence strength",
                "procedural posture",
                "claim type",
                "case history",
                "damages",
                "citations",
                "precedent",
                "forum",
                "venue",
            ),
        )
        legal = self._matched_terms(
            text,
            (
                "legal principles",
                "legal grounds",
                "rule of law",
                "stare decisis",
                "res judicata",
                "burden of proof",
                "standard of proof",
                "statute",
                "doctrine",
                "holding",
                "ratio decidendi",
            ),
        )
        behavioral = self._matched_terms(
            text,
            self.taxonomy.families.get("behavioral_law_economics").phrases
            if "behavioral_law_economics" in self.taxonomy.families
            else [],
        )
        strategic = self._matched_terms(
            text,
            self.taxonomy.families.get("game_theoretic_litigation").phrases
            if "game_theoretic_litigation" in self.taxonomy.families
            else [],
        )

        urls = [url.rstrip(".,;") for url in _URL_RE.findall(text)][:50]
        priors: list[str] = []
        if family_matches.get("legal_prediction"):
            priors.append("Outcome labels and time-aware court-decision forecasting")
        if family_matches.get("legal_argumentation"):
            priors.append("Facts, issues, legal principles, and reasoning-chain extraction")
        if family_matches.get("game_theoretic_litigation"):
            priors.append("Strategic interaction, bargaining, signaling, and equilibrium features")
        if family_matches.get("behavioral_law_economics"):
            priors.append("Bias, incentives, risk preferences, and bounded-rationality features")
        if family_matches.get("temporal_dynamic_models"):
            priors.append("Procedural state transitions, event sequences, and hazard models")
        if family_matches.get("evidence_and_credibility"):
            priors.append("Evidence-support, contradiction, credibility, and burden-of-proof signals")
        if artifacts:
            priors.append("Reusable datasets, annotations, models, or evaluation artifacts")

        domain_names = [name for name, values in family_matches.items() if values]
        relevance = (
            "Useful to Juricratic because it contributes "
            + ", ".join(name.replace("_", " ") for name in domain_names[:6])
            if domain_names
            else "Potentially relevant; more repository or abstract content is needed for classification."
        )

        return ArtifactExtraction(
            research_question="",
            legal_domain=domain_names,
            jurisdiction=jurisdictions,
            unit_of_analysis=(
                "court decisions"
                if any(term in text.lower() for term in ("judgment", "ruling", "court decision"))
                else ""
            ),
            data_sources=urls,
            dataset_size=size,
            labels_or_targets=self._matched_terms(
                text,
                (
                    "outcome",
                    "decision",
                    "winner",
                    "sentence",
                    "charge",
                    "law article",
                    "damages",
                    "settlement",
                    "appeal",
                    "affirmed",
                    "reversed",
                ),
                limit=20,
            ),
            factual_variables=factual,
            legal_variables=legal,
            behavioral_variables=behavioral,
            strategic_variables=strategic,
            methods=methods,
            validation_design=validations,
            metrics=self._matched_terms(
                text,
                (
                    "accuracy",
                    "precision",
                    "recall",
                    "F1",
                    "AUC",
                    "ROC",
                    "calibration",
                    "specificity",
                    "sensitivity",
                    "mean absolute error",
                    "Brier score",
                    "concordance index",
                ),
                limit=20,
            ),
            datasets=datasets,
            code_artifacts=artifacts,
            reusable_priors=priors,
            limitations=limitations,
            juricratic_relevance=relevance,
            extraction_mode="deterministic",
        )

    def _scrapegraph_extract(self, text: str) -> ArtifactExtraction:
        from scrapegraphai.graphs import SmartScraperGraph

        llm_config: dict[str, object] = {
            "model": self.settings.llm_model,
            "format": "json",
        }
        if self.settings.llm_api_key:
            llm_config["api_key"] = self.settings.llm_api_key
        if self.settings.llm_base_url:
            llm_config["base_url"] = self.settings.llm_base_url
        graph_config = {
            "llm": llm_config,
            "verbose": False,
            "headless": True,
            "force": False,
            "html_mode": False,
            "reattempt": True,
        }
        prompt = """
Extract this legal/ML research artifact into the supplied schema. Be conservative:
only report information supported by the text. Focus on research question, jurisdiction,
unit of analysis, data sources and size, labels/targets, factual/legal/behavioral/strategic
variables, methods, validation design, metrics, downloadable datasets or code, reusable
priors for a litigation intelligence and simulation engine, and limitations. In
juricratic_relevance, explain concretely how the work could inform litigation state
modeling, outcome forecasting, evidence/argument graphs, counsel/judge behavior,
settlement/game theory, behavioral economics, causal inference, or multi-agent simulation.
Do not infer private access or claim a dataset is downloadable unless the text says so.
""".strip()
        graph = SmartScraperGraph(
            prompt=prompt,
            source=text[:80_000],
            config=graph_config,
            schema=ArtifactExtraction,
        )
        result = graph.run()
        if isinstance(result, ArtifactExtraction):
            return result.model_copy(update={"extraction_mode": "scrapegraphai"})
        if isinstance(result, dict):
            payload = result.get("answer", result)
            if isinstance(payload, ArtifactExtraction):
                return payload.model_copy(update={"extraction_mode": "scrapegraphai"})
            if isinstance(payload, dict):
                return ArtifactExtraction.model_validate(payload).model_copy(
                    update={"extraction_mode": "scrapegraphai"}
                )
        raise ValueError("ScrapeGraphAI returned an unsupported extraction result.")

    def extract(
        self,
        text: str,
        *,
        use_llm: bool | None = None,
    ) -> ArtifactExtraction:
        deterministic = self.deterministic_extract(text)
        llm_requested = (
            self.settings.enable_llm_enrichment if use_llm is None else use_llm
        )
        if not (
            llm_requested
            and self.settings.enable_llm_enrichment
            and self.settings.llm_configured
            and text.strip()
        ):
            return deterministic
        try:
            enriched = self._scrapegraph_extract(text)
        except Exception as exc:
            return deterministic.model_copy(
                update={
                    "limitations": list(
                        dict.fromkeys(
                            deterministic.limitations
                            + [f"ScrapeGraphAI enrichment failed: {type(exc).__name__}"]
                        )
                    )
                }
            )

        # Preserve deterministic matches that the LLM may omit.
        merged: dict[str, object] = enriched.model_dump()
        for field in (
            "legal_domain",
            "jurisdiction",
            "data_sources",
            "labels_or_targets",
            "factual_variables",
            "legal_variables",
            "behavioral_variables",
            "strategic_variables",
            "methods",
            "validation_design",
            "metrics",
            "datasets",
            "code_artifacts",
            "reusable_priors",
            "limitations",
        ):
            merged[field] = list(
                dict.fromkeys(
                    list(getattr(enriched, field)) + list(getattr(deterministic, field))
                )
            )
        if not merged.get("dataset_size"):
            merged["dataset_size"] = deterministic.dataset_size
        return ArtifactExtraction.model_validate(merged)

    def enrich_candidate(
        self,
        candidate: ResearchCandidate,
        *,
        use_llm: bool | None = None,
    ) -> ResearchCandidate:
        extraction = self.extract(candidate.searchable_text, use_llm=use_llm)
        matched_terms = list(
            dict.fromkeys(
                extraction.methods
                + extraction.legal_domain
                + extraction.reusable_priors
            )
        )
        methods = list(dict.fromkeys(candidate.methods + extraction.methods))
        datasets = list(dict.fromkeys(candidate.datasets + extraction.datasets))
        legal_domains = list(
            dict.fromkeys(candidate.legal_domains + extraction.legal_domain)
        )
        jurisdiction = list(
            dict.fromkeys(candidate.jurisdiction + extraction.jurisdiction)
        )
        return candidate.model_copy(
            update={
                "extraction": extraction,
                "methods": methods,
                "datasets": datasets,
                "legal_domains": legal_domains,
                "jurisdiction": jurisdiction,
                "matched_terms": matched_terms,
                "risks": list(
                    dict.fromkeys(candidate.risks + extraction.limitations)
                ),
            }
        )
