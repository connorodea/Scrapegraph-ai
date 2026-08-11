"""Official-API scholarly discovery for Juricratic research papers."""

from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .config import Settings
from .http_client import RadarHttpError, ResilientHttpClient
from .models import (
    AccessDecision,
    AccessOutcome,
    CandidateKind,
    ResearchCandidate,
)


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value))).strip()


def _candidate_id(prefix: str, stable_value: str) -> str:
    digest = hashlib.sha1(stable_value.lower().encode("utf-8")).hexdigest()[:18]
    return f"{prefix}:{digest}"


def _date_from_parts(parts: Any) -> str | None:
    try:
        values = list(parts[0])
    except (TypeError, IndexError):
        return None
    while len(values) < 3:
        values.append(1)
    return f"{int(values[0]):04d}-{int(values[1]):02d}-{int(values[2]):02d}"


def _reconstruct_openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for token, positions in index.items():
        for position in positions or []:
            try:
                positioned.append((int(position), str(token)))
            except (TypeError, ValueError):
                continue
    positioned.sort(key=lambda item: item[0])
    return " ".join(token for _, token in positioned)


@dataclass(slots=True)
class PaperSearchBatch:
    provider: str
    query: str
    candidates: list[ResearchCandidate]
    total_count: int | None = None
    rate_limit: dict[str, int | str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ScholarlyResearchClient:
    OPENALEX_URL = "https://api.openalex.org/works"
    CROSSREF_URL = "https://api.crossref.org/works"
    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        settings: Settings,
        *,
        on_wait: Callable[[float, str], None] | None = None,
    ):
        self.settings = settings
        self.http = ResilientHttpClient(settings, on_wait=on_wait)

    @staticmethod
    def available_providers() -> tuple[str, ...]:
        return ("OpenAlex", "Crossref", "Semantic Scholar", "arXiv")

    @staticmethod
    def _api_access(provider: str) -> AccessDecision:
        return AccessDecision(
            outcome=AccessOutcome.API_ONLY,
            code=f"{provider.lower().replace(' ', '_')}_api",
            plain_english=f"Paper metadata was collected from the official {provider} API.",
            technical_detail="No publisher login, paywall, or private full text was accessed.",
            proxy_escalation_allowed=False,
        )

    def search_openalex(self, query: str, *, limit: int = 25) -> PaperSearchBatch:
        params: dict[str, Any] = {
            "search": query,
            "per-page": max(1, min(limit, 100)),
        }
        if self.settings.openalex_mailto:
            params["mailto"] = self.settings.openalex_mailto
        payload, response = self.http.get_json(self.OPENALEX_URL, params=params)
        results = payload.get("results") if isinstance(payload, dict) else []
        candidates: list[ResearchCandidate] = []
        for item in results or []:
            title = str(item.get("display_name") or item.get("title") or "Untitled")
            doi = str(item.get("doi") or "").replace("https://doi.org/", "") or None
            authors = [
                str((authorship.get("author") or {}).get("display_name"))
                for authorship in item.get("authorships") or []
                if (authorship.get("author") or {}).get("display_name")
            ]
            primary = item.get("primary_location") or {}
            best_oa = item.get("best_oa_location") or {}
            access = item.get("open_access") or {}
            oa_url = (
                best_oa.get("pdf_url")
                or best_oa.get("landing_page_url")
                or access.get("oa_url")
            )
            url = (
                f"https://doi.org/{doi}"
                if doi
                else primary.get("landing_page_url")
                or item.get("id")
                or ""
            )
            keywords = [
                str(keyword.get("display_name"))
                for keyword in item.get("keywords") or []
                if keyword.get("display_name")
            ]
            concepts = [
                str(concept.get("display_name"))
                for concept in item.get("concepts") or []
                if concept.get("display_name") and float(concept.get("score") or 0) >= 0.3
            ]
            stable = doi or str(item.get("id") or title)
            candidates.append(
                ResearchCandidate(
                    candidate_id=_candidate_id("paper", stable),
                    kind=CandidateKind.RESEARCH_PAPER,
                    source="OpenAlex",
                    title=title,
                    url=url,
                    canonical_url=url,
                    abstract=_reconstruct_openalex_abstract(
                        item.get("abstract_inverted_index")
                    ),
                    authors=authors,
                    year=item.get("publication_year"),
                    publication_date=item.get("publication_date"),
                    doi=doi,
                    citation_count=int(item.get("cited_by_count") or 0),
                    open_access=bool(access.get("is_oa")),
                    open_access_url=oa_url,
                    topics=list(dict.fromkeys(keywords + concepts))[:40],
                    access=self._api_access("OpenAlex"),
                    discovered_by=[query],
                    raw={
                        "openalex_id": item.get("id"),
                        "type": item.get("type"),
                        "primary_source": (primary.get("source") or {}).get(
                            "display_name"
                        ),
                        "is_retracted": bool(item.get("is_retracted")),
                        "is_paratext": bool(item.get("is_paratext")),
                    },
                )
            )
        total = None
        if isinstance(payload, dict):
            total = int((payload.get("meta") or {}).get("count") or 0)
        return PaperSearchBatch(
            provider="OpenAlex",
            query=query,
            candidates=candidates,
            total_count=total,
            rate_limit=response.rate_limit,
        )

    def search_crossref(self, query: str, *, limit: int = 25) -> PaperSearchBatch:
        params: dict[str, Any] = {
            "query.bibliographic": query,
            "rows": max(1, min(limit, 100)),
            "select": (
                "DOI,title,abstract,author,published,created,URL,link,subject,"
                "publisher,type,is-referenced-by-count,reference-count,score"
            ),
        }
        if self.settings.openalex_mailto:
            params["mailto"] = self.settings.openalex_mailto
        payload, response = self.http.get_json(self.CROSSREF_URL, params=params)
        message = payload.get("message") if isinstance(payload, dict) else {}
        items = message.get("items") if isinstance(message, dict) else []
        candidates: list[ResearchCandidate] = []
        for item in items or []:
            titles = item.get("title") or []
            title = str(titles[0] if titles else "Untitled")
            doi = str(item.get("DOI") or "") or None
            authors = []
            for author in item.get("author") or []:
                name = " ".join(
                    part
                    for part in (author.get("given"), author.get("family"))
                    if part
                )
                if name:
                    authors.append(name)
            published = (item.get("published") or {}).get("date-parts")
            publication_date = _date_from_parts(published)
            year = None
            if publication_date:
                try:
                    year = int(publication_date[:4])
                except ValueError:
                    pass
            links = item.get("link") or []
            full_text_links = [
                link.get("URL")
                for link in links
                if link.get("URL") and "text" in str(link.get("content-type") or "")
            ]
            url = str(item.get("URL") or (f"https://doi.org/{doi}" if doi else ""))
            stable = doi or url or title
            candidates.append(
                ResearchCandidate(
                    candidate_id=_candidate_id("paper", stable),
                    kind=CandidateKind.RESEARCH_PAPER,
                    source="Crossref",
                    title=title,
                    url=url,
                    canonical_url=url,
                    abstract=_clean_text(item.get("abstract")),
                    authors=authors,
                    year=year,
                    publication_date=publication_date,
                    doi=doi,
                    citation_count=int(item.get("is-referenced-by-count") or 0),
                    open_access=None,
                    open_access_url=full_text_links[0] if full_text_links else None,
                    topics=[str(value) for value in item.get("subject") or []],
                    access=self._api_access("Crossref"),
                    discovered_by=[query],
                    raw={
                        "publisher": item.get("publisher"),
                        "type": item.get("type"),
                        "reference_count": int(item.get("reference-count") or 0),
                        "crossref_score": item.get("score"),
                        "full_text_links": full_text_links,
                    },
                )
            )
        total = None
        if isinstance(message, dict):
            total = int(message.get("total-results") or 0)
        return PaperSearchBatch(
            provider="Crossref",
            query=query,
            candidates=candidates,
            total_count=total,
            rate_limit=response.rate_limit,
        )

    def search_semantic_scholar(
        self, query: str, *, limit: int = 25
    ) -> PaperSearchBatch:
        headers: dict[str, str] = {}
        if self.settings.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.semantic_scholar_api_key
        fields = (
            "paperId,title,abstract,year,publicationDate,authors,url,externalIds,"
            "citationCount,influentialCitationCount,openAccessPdf,fieldsOfStudy,"
            "s2FieldsOfStudy,publicationTypes,journal,venue"
        )
        payload, response = self.http.get_json(
            self.SEMANTIC_SCHOLAR_URL,
            params={"query": query, "limit": max(1, min(limit, 100)), "fields": fields},
            headers=headers,
        )
        items = payload.get("data") if isinstance(payload, dict) else []
        candidates: list[ResearchCandidate] = []
        for item in items or []:
            external = item.get("externalIds") or {}
            doi = external.get("DOI")
            arxiv_id = external.get("ArXiv")
            oa_pdf = item.get("openAccessPdf") or {}
            url = item.get("url") or (
                f"https://doi.org/{doi}"
                if doi
                else f"https://arxiv.org/abs/{arxiv_id}"
                if arxiv_id
                else ""
            )
            fields_of_study = [str(value) for value in item.get("fieldsOfStudy") or []]
            s2_fields = [
                str(value.get("category"))
                for value in item.get("s2FieldsOfStudy") or []
                if value.get("category")
            ]
            paper_id = str(item.get("paperId") or doi or url or item.get("title"))
            candidates.append(
                ResearchCandidate(
                    candidate_id=_candidate_id("paper", paper_id),
                    kind=CandidateKind.RESEARCH_PAPER,
                    source="Semantic Scholar",
                    title=str(item.get("title") or "Untitled"),
                    url=str(url),
                    canonical_url=str(url),
                    abstract=str(item.get("abstract") or ""),
                    authors=[
                        str(author.get("name"))
                        for author in item.get("authors") or []
                        if author.get("name")
                    ],
                    year=item.get("year"),
                    publication_date=item.get("publicationDate"),
                    doi=str(doi) if doi else None,
                    citation_count=int(item.get("citationCount") or 0),
                    open_access=bool(oa_pdf.get("url")),
                    open_access_url=oa_pdf.get("url"),
                    topics=list(dict.fromkeys(fields_of_study + s2_fields)),
                    access=self._api_access("Semantic Scholar"),
                    discovered_by=[query],
                    raw={
                        "paper_id": item.get("paperId"),
                        "arxiv_id": arxiv_id,
                        "influential_citation_count": int(
                            item.get("influentialCitationCount") or 0
                        ),
                        "publication_types": item.get("publicationTypes") or [],
                        "venue": item.get("venue"),
                        "journal": item.get("journal") or {},
                    },
                )
            )
        total = int(payload.get("total") or 0) if isinstance(payload, dict) else None
        return PaperSearchBatch(
            provider="Semantic Scholar",
            query=query,
            candidates=candidates,
            total_count=total,
            rate_limit=response.rate_limit,
        )

    def search_arxiv(self, query: str, *, limit: int = 25) -> PaperSearchBatch:
        search_query = f'all:"{query.replace(chr(34), "")}"'
        text, response = self.http.get_text(
            self.ARXIV_URL,
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": max(1, min(limit, 100)),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise RadarHttpError(f"arXiv returned invalid Atom XML: {exc}") from exc
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        total_node = root.find("opensearch:totalResults", ns)
        total = int(total_node.text or 0) if total_node is not None else None
        candidates: list[ResearchCandidate] = []
        for entry in root.findall("atom:entry", ns):
            identifier = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            arxiv_id = identifier.rsplit("/", 1)[-1]
            title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
            abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
            published = entry.findtext("atom:published", default="", namespaces=ns)
            year = None
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    pass
            authors = [
                _clean_text(author.findtext("atom:name", default="", namespaces=ns))
                for author in entry.findall("atom:author", ns)
            ]
            links = [
                {
                    "href": link.attrib.get("href"),
                    "rel": link.attrib.get("rel"),
                    "type": link.attrib.get("type"),
                    "title": link.attrib.get("title"),
                }
                for link in entry.findall("atom:link", ns)
            ]
            pdf_url = next(
                (
                    link["href"]
                    for link in links
                    if link.get("title") == "pdf" or link.get("type") == "application/pdf"
                ),
                None,
            )
            categories = [
                category.attrib.get("term", "")
                for category in entry.findall("atom:category", ns)
                if category.attrib.get("term")
            ]
            candidates.append(
                ResearchCandidate(
                    candidate_id=_candidate_id("paper", arxiv_id or identifier or title),
                    kind=CandidateKind.RESEARCH_PAPER,
                    source="arXiv",
                    title=title or "Untitled",
                    url=identifier,
                    canonical_url=identifier,
                    abstract=abstract,
                    authors=[author for author in authors if author],
                    year=year,
                    publication_date=published[:10] if published else None,
                    doi=entry.findtext("arxiv:doi", default=None, namespaces=ns),
                    open_access=True,
                    open_access_url=pdf_url,
                    topics=categories,
                    access=self._api_access("arXiv"),
                    discovered_by=[query],
                    raw={
                        "arxiv_id": arxiv_id,
                        "links": links,
                        "primary_category": (
                            entry.find("arxiv:primary_category", ns).attrib.get("term")
                            if entry.find("arxiv:primary_category", ns) is not None
                            else None
                        ),
                    },
                )
            )
        return PaperSearchBatch(
            provider="arXiv",
            query=query,
            candidates=candidates,
            total_count=total,
            rate_limit=response.rate_limit,
        )

    def search_provider(
        self, provider: str, query: str, *, limit: int = 25
    ) -> PaperSearchBatch:
        normalized = provider.strip().lower()
        if normalized == "openalex":
            return self.search_openalex(query, limit=limit)
        if normalized == "crossref":
            return self.search_crossref(query, limit=limit)
        if normalized in {"semantic scholar", "semanticscholar", "s2"}:
            return self.search_semantic_scholar(query, limit=limit)
        if normalized == "arxiv":
            return self.search_arxiv(query, limit=limit)
        raise ValueError(f"Unknown scholarly provider: {provider}")

    @staticmethod
    def deduplicate(candidates: Iterable[ResearchCandidate]) -> list[ResearchCandidate]:
        by_key: dict[str, ResearchCandidate] = {}
        for candidate in candidates:
            title_key = re.sub(r"\W+", "", candidate.title.lower())
            key = (candidate.doi or title_key or candidate.url).lower()
            if not key:
                key = candidate.candidate_id
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = candidate
                continue
            merged_sources = list(dict.fromkeys(existing.discovered_by + candidate.discovered_by))
            merged_topics = list(dict.fromkeys(existing.topics + candidate.topics))
            preferred = existing
            if len(candidate.abstract) > len(existing.abstract):
                preferred = candidate
            preferred = preferred.model_copy(
                update={
                    "discovered_by": merged_sources,
                    "topics": merged_topics,
                    "citation_count": max(
                        existing.citation_count, candidate.citation_count
                    ),
                    "open_access": bool(existing.open_access or candidate.open_access),
                    "open_access_url": existing.open_access_url or candidate.open_access_url,
                    "raw": {
                        "merged_sources": list(
                            dict.fromkeys(
                                [existing.source, candidate.source]
                                + existing.raw.get("merged_sources", [])
                                + candidate.raw.get("merged_sources", [])
                            )
                        ),
                        "records": [existing.raw, candidate.raw],
                    },
                }
            )
            by_key[key] = preferred
        return list(by_key.values())

    def close(self) -> None:
        self.http.close()
