"""GitHub discovery, query sharding, artifact extraction, and transparent scoring."""

from __future__ import annotations

import asyncio
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from urllib.parse import quote, urlparse

import httpx
import yaml

from .config import GitHubSettings, RadarConfig
from .models import Artifact, ArtifactType, RepositoryRecord, ScoreBreakdown, SearchShard

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9_-]{2,}")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]{1,240})\]\((https?://[^)\s]+)\)")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_URL_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:\.pdf)?", re.IGNORECASE)
ARXIV_ID_RE = re.compile(r"\barxiv\s*:\s*([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})\b", re.IGNORECASE)
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{[^@]+?\n\}", re.IGNORECASE | re.DOTALL)
BIB_FIELD_RE = re.compile(r"\b(title|doi|url|booktitle|journal|year)\s*=\s*[{\"](.+?)[}\"]\s*,?", re.IGNORECASE | re.DOTALL)

STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "and", "any", "are",
    "because", "been", "before", "being", "between", "both", "but", "can", "could",
    "data", "does", "each", "from", "github", "have", "having", "into", "its", "more",
    "most", "not", "only", "other", "our", "out", "over", "project", "repository",
    "research", "should", "some", "such", "than", "that", "the", "their", "then",
    "there", "these", "they", "this", "through", "using", "very", "was", "were",
    "what", "when", "where", "which", "while", "with", "within", "would", "you",
    "your", "para", "como", "con", "del", "desde", "este", "esta", "las", "los",
    "por", "que", "una", "uno",
}
DATA_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".arrow", ".xml", ".ann", ".txt", ".conll", ".brat", ".sqlite", ".db", ".zip"}
MODEL_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx", ".h5"}
PIPELINE_NAMES = {"train.py", "training.py", "evaluate.py", "evaluation.py", "preprocess.py", "preprocessing.py", "inference.py", "predict.py", "dataloader.py", "dataset.py", "dockerfile", "makefile", "requirements.txt", "pyproject.toml", "environment.yml"}


@dataclass(frozen=True)
class DerivedKeyword:
    term: str
    score: float
    document_frequency: int
    relevant_document_frequency: int
    source: str = "corpus_cooccurrence"


def normalize_phrase(value: str) -> str:
    return SPACE_RE.sub(" ", value.casefold().strip())


def tokenize(text: str) -> list[str]:
    return [token.casefold().strip("-_") for token in TOKEN_RE.findall(text) if token.casefold().strip("-_") not in STOPWORDS]


def _ngrams(tokens: Sequence[str], max_n: int = 3) -> set[str]:
    values: set[str] = set()
    for n in range(1, max_n + 1):
        for index in range(0, len(tokens) - n + 1):
            gram = tokens[index:index + n]
            if not gram or gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                continue
            phrase = " ".join(gram)
            if len(phrase) <= 72 and not phrase.replace(" ", "").isdigit():
                values.add(phrase)
    return values


def derive_keywords(documents: Iterable[str], seed_terms: Iterable[str], *, minimum_documents: int = 2, limit: int = 100) -> list[DerivedKeyword]:
    docs = [normalize_phrase(document) for document in documents if document and document.strip()]
    if not docs:
        return []
    normalized_seeds = {normalize_phrase(term) for term in seed_terms if term.strip()}
    seed_tokens = {token for term in normalized_seeds for token in tokenize(term)}
    doc_terms: list[set[str]] = []
    relevant_flags: list[bool] = []
    for doc in docs:
        tokens = tokenize(doc)
        terms = _ngrams(tokens)
        doc_terms.append(terms)
        relevant_flags.append(any(seed in doc for seed in normalized_seeds) or bool(seed_tokens.intersection(tokens)))
    relevant_documents = max(1, sum(relevant_flags))
    total_documents = len(docs)
    df: Counter[str] = Counter()
    relevant_df: Counter[str] = Counter()
    for terms, relevant in zip(doc_terms, relevant_flags, strict=True):
        df.update(terms)
        if relevant:
            relevant_df.update(terms)
    result: list[DerivedKeyword] = []
    for term, frequency in df.items():
        rel_frequency = relevant_df.get(term, 0)
        if frequency < minimum_documents or rel_frequency < minimum_documents or term in normalized_seeds or len(term) < 4:
            continue
        precision = rel_frequency / frequency
        coverage = rel_frequency / relevant_documents
        idf = math.log((total_documents + 1) / (frequency + 1)) + 1
        score = precision * coverage * idf * math.log1p(rel_frequency) * 100
        result.append(DerivedKeyword(term, round(score, 6), frequency, rel_frequency))
    return sorted(result, key=lambda item: (item.score, item.relevant_document_frequency, len(item.term)), reverse=True)[:limit]


def build_queries(config: RadarConfig, dynamic_terms: Iterable[str] | None = None, *, limit: int | None = None) -> list[str]:
    queries = list(config.queries)
    for term in list(dynamic_terms or [])[:50]:
        normalized = normalize_phrase(term)
        queries.extend([f'"{normalized}" legal dataset in:name,description,readme', f'"{normalized}" court machine learning in:name,description,readme'])
    for task in config.ontology.get("tasks", [])[:24]:
        queries.extend([f'"{task}" dataset in:name,description,readme', f'"{task}" benchmark in:name,description,readme'])
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        compact = SPACE_RE.sub(" ", query.strip())
        key = compact.casefold()
        if not compact or key in seen:
            continue
        seen.add(key)
        unique.append(compact)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?)]}")


def _clean_identifier(value: str) -> str:
    return value.strip().rstrip(".,;:!?)]}").lower()


def _classify_link(label: str, url: str) -> ArtifactType:
    haystack = f"{label} {url}".casefold()
    host = urlparse(url).netloc.casefold()
    if "arxiv.org" in host or "doi.org" in host or "aclweb.org" in host:
        return ArtifactType.PAPER
    if any(term in haystack for term in ("dataset", "corpus", "benchmark", "data card")):
        return ArtifactType.DATASET
    if any(host.endswith(value) for value in ("huggingface.co", "zenodo.org", "kaggle.com", "figshare.com", "osf.io")):
        return ArtifactType.MODEL if "model" in haystack or "/models/" in url else ArtifactType.DATASET
    if any(term in haystack for term in ("model", "checkpoint", "weights")):
        return ArtifactType.MODEL
    if any(term in haystack for term in ("pipeline", "training code", "implementation", "code")):
        return ArtifactType.PIPELINE
    if any(term in haystack for term in ("paper", "publication", "article", "proceedings")):
        return ArtifactType.PAPER
    return ArtifactType.OTHER


class ArtifactExtractor:
    def extract(self, *, readme_text: str, citation_files: dict[str, str] | None = None, tree_paths: Iterable[str] = ()) -> list[Artifact]:
        artifacts: list[Artifact] = []
        citation_files = citation_files or {}
        combined_parts = [readme_text]
        for path, text in citation_files.items():
            combined_parts.append(text)
            lower = path.casefold()
            if lower.endswith("citation.cff"):
                try:
                    payload = yaml.safe_load(text)
                except yaml.YAMLError:
                    payload = None
                if isinstance(payload, dict) and any(payload.get(key) for key in ("title", "doi", "url")):
                    doi = payload.get("doi")
                    artifacts.append(Artifact(artifact_type=ArtifactType.PAPER, name=payload.get("title"), url=payload.get("url") or (f"https://doi.org/{doi}" if doi else None), identifier=_clean_identifier(str(doi)) if doi else None, source=path, metadata={"authors": payload.get("authors") or [], "version": payload.get("version")}))
            elif lower.endswith(".bib"):
                for entry in BIB_ENTRY_RE.findall(text):
                    fields = {key.casefold(): SPACE_RE.sub(" ", value).strip() for key, value in BIB_FIELD_RE.findall(entry)}
                    if fields:
                        doi = fields.get("doi")
                        artifacts.append(Artifact(artifact_type=ArtifactType.PAPER, name=fields.get("title"), url=fields.get("url") or (f"https://doi.org/{doi}" if doi else None), identifier=_clean_identifier(doi) if doi else None, source=path, metadata={key: value for key, value in fields.items() if key not in {"title", "doi", "url"}}))
        combined = "\n".join(combined_parts)
        for label, url in MARKDOWN_LINK_RE.findall(combined):
            cleaned = _clean_url(url)
            artifacts.append(Artifact(artifact_type=_classify_link(label, cleaned), name=SPACE_RE.sub(" ", label).strip() or None, url=cleaned, source="markdown_link"))
        for doi in DOI_RE.findall(combined):
            identifier = _clean_identifier(doi)
            artifacts.append(Artifact(artifact_type=ArtifactType.PAPER, url=f"https://doi.org/{identifier}", identifier=identifier, source="doi_regex"))
        for match in ARXIV_URL_RE.finditer(combined):
            identifier = _clean_identifier(match.group(1))
            artifacts.append(Artifact(artifact_type=ArtifactType.PAPER, url=f"https://arxiv.org/abs/{identifier}", identifier=f"arxiv:{identifier}", source="arxiv_url"))
        for identifier in ARXIV_ID_RE.findall(combined):
            clean = _clean_identifier(identifier)
            artifacts.append(Artifact(artifact_type=ArtifactType.PAPER, url=f"https://arxiv.org/abs/{clean}", identifier=f"arxiv:{clean}", source="arxiv_id"))
        artifacts.extend(self._file_signals(tree_paths))
        seen: set[tuple[str, str, str, str]] = set()
        output: list[Artifact] = []
        for artifact in artifacts:
            if artifact.dedupe_key in seen:
                continue
            seen.add(artifact.dedupe_key)
            output.append(artifact)
        return output

    @staticmethod
    def _file_signals(tree_paths: Iterable[str], limit: int = 80) -> list[Artifact]:
        result: list[Artifact] = []
        for raw_path in tree_paths:
            path = PurePosixPath(raw_path)
            lower = raw_path.casefold()
            name = path.name.casefold()
            suffix = path.suffix.casefold()
            artifact_type: ArtifactType | None = None
            if name == "citation.cff" or suffix == ".bib":
                artifact_type = ArtifactType.CITATION
            elif suffix == ".ipynb":
                artifact_type = ArtifactType.NOTEBOOK
            elif suffix in MODEL_EXTENSIONS or any(term in lower for term in ("checkpoint", "weights/", "models/")):
                artifact_type = ArtifactType.MODEL
            elif suffix in DATA_EXTENSIONS and any(term in lower for term in ("data", "dataset", "corpus", "benchmark", "annotation", "judgment")):
                artifact_type = ArtifactType.DATASET
            elif name in PIPELINE_NAMES or any(term in lower for term in ("pipeline", "preprocess", "training", "evaluation")):
                artifact_type = ArtifactType.PIPELINE
            if artifact_type:
                result.append(Artifact(artifact_type=artifact_type, name=path.name, path=raw_path, source="repository_tree"))
                if len(result) >= limit:
                    break
        return result


def _contains(text: str, term: str) -> bool:
    normalized = term.casefold().strip()
    if not normalized:
        return False
    if " " in normalized or "-" in normalized:
        return re.search(re.escape(normalized).replace(r"\ ", r"\s+"), text) is not None
    return re.search(rf"(?<![\w]){re.escape(normalized)}(?![\w])", text) is not None


def _component(weight: float, hits: int, saturation: float) -> float:
    return 0.0 if hits <= 0 else weight * (1 - math.exp(-hits / saturation))


class ResearchScorer:
    def __init__(self, config: RadarConfig):
        self.config = config
        self.weights = {"legal_domain": 22, "research_task": 22, "data_signal": 14, "method_signal": 14, "paper_signal": 14, "reproducibility": 9, "quality": 5, **config.score_weights}

    def score(self, repository: RepositoryRecord, *, artifacts: Iterable[Artifact] = ()) -> ScoreBreakdown:
        artifacts = list(artifacts)
        text = " ".join([repository.full_name, repository.description or "", repository.readme_text, " ".join(repository.topics), " ".join(repository.tree_paths)]).casefold()
        matched: dict[str, list[str]] = {}
        for facet, terms in self.config.ontology.items():
            if facet not in {"negative_terms", "coursework_terms"}:
                matched[facet] = sorted({term for term in terms if _contains(text, term)}, key=str.casefold)
        legal_hits = len(matched.get("legal_context", [])) + min(3, len(matched.get("jurisdictions", [])))
        task_hits = len(matched.get("tasks", [])) + min(3, len(matched.get("argumentation", [])))
        counts = Counter(artifact.artifact_type for artifact in artifacts)
        data_hits = len(matched.get("data_artifacts", [])) + counts[ArtifactType.DATASET] + counts[ArtifactType.BENCHMARK] + counts[ArtifactType.ANNOTATION]
        method_hits = len(matched.get("methods", [])) + counts[ArtifactType.MODEL] + counts[ArtifactType.PIPELINE] + counts[ArtifactType.NOTEBOOK]
        paper_hits = len(matched.get("research_markers", [])) + 2 * (counts[ArtifactType.PAPER] + counts[ArtifactType.CITATION])
        legal_domain = _component(self.weights["legal_domain"], legal_hits, 3)
        research_task = _component(self.weights["research_task"], task_hits, 3)
        data_signal = _component(self.weights["data_signal"], data_hits, 3)
        method_signal = _component(self.weights["method_signal"], method_hits, 4)
        paper_signal = _component(self.weights["paper_signal"], paper_hits, 3)
        paths = [path.casefold() for path in repository.tree_paths]
        reproducibility_hits = sum([
            any(path.endswith("citation.cff") or path.endswith(".bib") for path in paths),
            any(path.endswith(("requirements.txt", "pyproject.toml", "environment.yml")) for path in paths),
            any(path.endswith(("dockerfile", "docker-compose.yml")) for path in paths),
            any(path.endswith(".ipynb") for path in paths),
            any("train" in path or "training" in path for path in paths),
            any("test" in path or "evaluation" in path for path in paths),
            any("data" in path or "dataset" in path for path in paths),
            any(path.endswith(("license", "license.md", "license.txt")) for path in paths),
        ])
        reproducibility = _component(self.weights["reproducibility"], reproducibility_hits, 4)
        star_factor = min(1.0, math.log1p(repository.stars) / math.log(501))
        license_factor = 1.0 if repository.license_spdx and repository.license_spdx != "NOASSERTION" else 0.0
        quality = self.weights["quality"] * (0.7 * star_factor + 0.3 * license_factor)
        penalties = 15 * repository.archived + 20 * repository.disabled + 5 * repository.is_fork + (10 if not repository.description and not repository.readme_text.strip() else 0)
        penalties += sum(8 for term in self.config.ontology.get("negative_terms", []) if _contains(text, term))
        penalties += sum(4 for term in self.config.ontology.get("coursework_terms", []) if _contains(text, term))
        penalties = min(35.0, penalties)
        total = max(0.0, min(100.0, legal_domain + research_task + data_signal + method_signal + paper_signal + reproducibility + quality - penalties))
        classifications: list[str] = []
        task_terms = " ".join(matched.get("tasks", [])).casefold()
        if "prediction" in task_terms or "forecast" in task_terms:
            classifications.append("outcome_prediction")
        if matched.get("argumentation"):
            classifications.append("legal_argumentation")
        if data_hits:
            classifications.append("dataset_or_corpus")
        if method_hits:
            classifications.append("model_or_pipeline")
        if paper_hits:
            classifications.append("paper_linked")
        classifications.append("high_priority" if total >= 55 else "review" if total >= 35 else "low_priority")
        return ScoreBreakdown(total=round(total, 3), legal_domain=round(legal_domain, 3), research_task=round(research_task, 3), data_signal=round(data_signal, 3), method_signal=round(method_signal, 3), paper_signal=round(paper_signal, 3), reproducibility=round(reproducibility, 3), quality=round(quality, 3), penalties=round(penalties, 3), classifications=sorted(set(classifications)), matched_terms={key: value for key, value in matched.items() if value})


class GitHubError(RuntimeError):
    pass


class GitHubClient:
    def __init__(self, settings: GitHubSettings, token: str | None = None):
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.concurrency)
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": settings.user_agent}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(base_url=settings.api_url.rstrip("/"), headers=headers, timeout=settings.request_timeout_seconds, follow_redirects=True)

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, allow_not_found: bool = False) -> httpx.Response | None:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.request(method, path, params=params, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                await asyncio.sleep(min(30.0, 2 ** attempt + random.random()))
                continue
            if response.status_code == 404 and allow_not_found:
                return None
            if response.status_code < 400:
                return response
            if response.status_code in {403, 429} and attempt < self.settings.max_retries:
                retry_after = response.headers.get("Retry-After")
                reset = response.headers.get("X-RateLimit-Reset")
                remaining = response.headers.get("X-RateLimit-Remaining")
                wait = float(retry_after) if retry_after else max(1.0, float(reset) - time.time() + 1) if remaining == "0" and reset else min(60.0, 2 ** attempt + random.random())
                await asyncio.sleep(min(wait, 900.0))
                continue
            if response.status_code >= 500 and attempt < self.settings.max_retries:
                await asyncio.sleep(min(30.0, 2 ** attempt + random.random()))
                continue
            raise GitHubError(f"GitHub API {method} {path} failed ({response.status_code}): {response.text[:1000]}")
        raise GitHubError(f"GitHub request failed after retries: {last_error}")

    async def search_repositories(self, query: str, *, page: int = 1, per_page: int | None = None) -> dict[str, Any]:
        response = await self._request("GET", "/search/repositories", params={"q": query, "page": page, "per_page": per_page or self.settings.per_page, "sort": "updated", "order": "desc"})
        assert response is not None
        return response.json()

    async def get_repository(self, full_name: str) -> dict[str, Any]:
        response = await self._request("GET", f"/repos/{full_name}")
        assert response is not None
        return response.json()

    async def get_readme(self, full_name: str) -> str:
        response = await self._request("GET", f"/repos/{full_name}/readme", headers={"Accept": "application/vnd.github.raw+json"}, allow_not_found=True)
        return response.text if response else ""

    async def get_topics(self, full_name: str) -> list[str]:
        response = await self._request("GET", f"/repos/{full_name}/topics", allow_not_found=True)
        return list(response.json().get("names") or []) if response else []

    async def get_license_spdx(self, full_name: str) -> str | None:
        response = await self._request("GET", f"/repos/{full_name}/license", allow_not_found=True)
        return ((response.json().get("license") or {}).get("spdx_id")) if response else None

    async def get_tree(self, full_name: str, ref: str) -> tuple[list[str], bool]:
        response = await self._request("GET", f"/repos/{full_name}/git/trees/{quote(ref, safe='')}", params={"recursive": "1"}, allow_not_found=True)
        if not response:
            return [], False
        payload = response.json()
        return [str(item["path"]) for item in payload.get("tree") or [] if item.get("type") == "blob" and item.get("path")], bool(payload.get("truncated"))

    async def get_file_text(self, full_name: str, path: str, *, ref: str | None = None, max_bytes: int = 750_000) -> str | None:
        response = await self._request("GET", f"/repos/{full_name}/contents/{quote(path, safe='/')}", params={"ref": ref} if ref else None, headers={"Accept": "application/vnd.github.raw+json"}, allow_not_found=True)
        return response.content[:max_bytes].decode("utf-8", errors="replace") if response else None

    async def get_commit(self, full_name: str, ref: str) -> dict[str, Any]:
        response = await self._request("GET", f"/repos/{full_name}/commits/{quote(ref, safe='')}")
        assert response is not None
        return response.json()

    async def download_tarball(self, full_name: str, ref: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with self._semaphore:
            async with self._client.stream("GET", f"/repos/{full_name}/tarball/{quote(ref, safe='')}") as response:
                if response.status_code >= 400:
                    raise GitHubError(f"Archive download failed for {full_name}: {response.status_code}")
                with destination.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
        return destination


class QueryPlanner:
    STAR_BANDS = ("stars:0", "stars:1..4", "stars:5..19", "stars:20..99", "stars:100..499", "stars:>=500")

    def __init__(self, client: GitHubClient, *, max_results_per_shard: int = 950):
        self.client = client
        self.max_results_per_shard = max_results_per_shard
        self._count_cache: dict[str, int] = {}

    async def _count(self, query: str) -> int:
        if query not in self._count_cache:
            payload = await self.client.search_repositories(query, page=1, per_page=1)
            self._count_cache[query] = int(payload.get("total_count") or 0)
        return self._count_cache[query]

    @staticmethod
    def qualify(base_query: str, start_date: date, end_date: date, star_qualifier: str | None = None) -> str:
        return " ".join([base_query, f"created:{start_date.isoformat()}..{end_date.isoformat()}"] + ([star_qualifier] if star_qualifier else []))

    async def plan(self, base_query: str, *, start_date: date, end_date: date) -> list[SearchShard]:
        query = self.qualify(base_query, start_date, end_date)
        total = await self._count(query)
        if total == 0:
            return []
        if total <= self.max_results_per_shard:
            return [SearchShard(base_query=base_query, query=query, start_date=start_date, end_date=end_date, total_count=total)]
        if start_date < end_date:
            midpoint = start_date + timedelta(days=(end_date - start_date).days // 2)
            left = await self.plan(base_query, start_date=start_date, end_date=midpoint)
            right_start = midpoint + timedelta(days=1)
            right = await self.plan(base_query, start_date=right_start, end_date=end_date) if right_start <= end_date else []
            return left + right
        shards: list[SearchShard] = []
        for band in self.STAR_BANDS:
            star_query = self.qualify(base_query, start_date, end_date, band)
            star_total = await self._count(star_query)
            if star_total:
                shards.append(SearchShard(base_query=base_query, query=star_query, start_date=start_date, end_date=end_date, total_count=star_total, star_qualifier=band, overflow=star_total > self.max_results_per_shard))
        return shards
