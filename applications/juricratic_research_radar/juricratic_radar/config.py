"""Configuration and secret handling for the Juricratic Research Radar."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = APP_DIR / "data" / "juricratic_research_radar.sqlite3"
DEFAULT_TAXONOMY_PATH = APP_DIR / "config" / "query_taxonomy.yaml"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. Secrets are never persisted or included in logs."""

    github_token: str = ""
    scrapingbee_api_key: str = ""
    scrapedo_api_key: str = ""
    semantic_scholar_api_key: str = ""
    openalex_mailto: str = ""

    llm_model: str = "ollama/llama3.2"
    llm_api_key: str = ""
    llm_base_url: str = ""
    enable_llm_enrichment: bool = False

    database_path: Path = DEFAULT_DB_PATH
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH
    request_timeout_seconds: float = 30.0
    max_response_bytes: int = 8_000_000
    max_workers: int = 4
    github_search_pause_seconds: float = 2.1
    provider_pause_seconds: float = 0.35
    user_agent: str = (
        "JuricraticResearchRadar/0.1 "
        "(public research discovery; contact configured via OPENALEX_MAILTO)"
    )

    respect_robots_txt: bool = True
    allow_proxy_escalation: bool = False
    allow_full_text_fetch: bool = False
    allow_javascript_rendering: bool = False
    default_proxy_provider: str = "scrapingbee"
    proxy_approved_domains: tuple[str, ...] = (
        "arxiv.org",
        "export.arxiv.org",
        "zenodo.org",
        "figshare.com",
        "osf.io",
        "huggingface.co",
    )
    blocked_domains: tuple[str, ...] = (
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "169.254.169.254",
    )

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> "Settings":
        if env_file:
            load_dotenv(Path(env_file), override=False)
        else:
            load_dotenv(APP_DIR / ".env", override=False)

        settings = cls(
            github_token=os.getenv("GITHUB_TOKEN", ""),
            scrapingbee_api_key=os.getenv("SCRAPINGBEE_API_KEY", ""),
            scrapedo_api_key=os.getenv("SCRAPEDO_API_KEY", ""),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
            openalex_mailto=os.getenv("OPENALEX_MAILTO", ""),
            llm_model=os.getenv("JURICRATIC_LLM_MODEL", "ollama/llama3.2"),
            llm_api_key=os.getenv(
                "JURICRATIC_LLM_API_KEY",
                os.getenv("OPENAI_API_KEY", ""),
            ),
            llm_base_url=os.getenv("JURICRATIC_LLM_BASE_URL", ""),
            enable_llm_enrichment=_env_bool(
                "JURICRATIC_ENABLE_LLM_ENRICHMENT", False
            ),
            database_path=Path(
                os.getenv("JURICRATIC_RADAR_DB", str(DEFAULT_DB_PATH))
            ).expanduser(),
            taxonomy_path=Path(
                os.getenv("JURICRATIC_TAXONOMY_PATH", str(DEFAULT_TAXONOMY_PATH))
            ).expanduser(),
            request_timeout_seconds=_env_float(
                "JURICRATIC_REQUEST_TIMEOUT_SECONDS", 30.0
            ),
            max_response_bytes=_env_int("JURICRATIC_MAX_RESPONSE_BYTES", 8_000_000),
            max_workers=_env_int("JURICRATIC_MAX_WORKERS", 4),
            github_search_pause_seconds=_env_float(
                "JURICRATIC_GITHUB_SEARCH_PAUSE_SECONDS", 2.1
            ),
            provider_pause_seconds=_env_float(
                "JURICRATIC_PROVIDER_PAUSE_SECONDS", 0.35
            ),
            respect_robots_txt=_env_bool("JURICRATIC_RESPECT_ROBOTS", True),
            allow_proxy_escalation=_env_bool(
                "JURICRATIC_ALLOW_PROXY_ESCALATION", False
            ),
            allow_full_text_fetch=_env_bool(
                "JURICRATIC_ALLOW_FULL_TEXT_FETCH", False
            ),
            allow_javascript_rendering=_env_bool(
                "JURICRATIC_ALLOW_JS_RENDERING", False
            ),
            default_proxy_provider=os.getenv(
                "JURICRATIC_DEFAULT_PROXY_PROVIDER", "scrapingbee"
            ).lower(),
        )
        if overrides:
            valid = {key: value for key, value in overrides.items() if hasattr(settings, key)}
            settings = replace(settings, **valid)
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        return settings

    @property
    def github_authenticated(self) -> bool:
        return bool(self.github_token)

    @property
    def scrapingbee_configured(self) -> bool:
        return bool(self.scrapingbee_api_key)

    @property
    def scrapedo_configured(self) -> bool:
        return bool(self.scrapedo_api_key)

    @property
    def llm_configured(self) -> bool:
        if self.llm_model.startswith("ollama/"):
            return True
        return bool(self.llm_api_key)

    def secret_status(self) -> dict[str, bool]:
        return {
            "GitHub": self.github_authenticated,
            "ScrapingBee": self.scrapingbee_configured,
            "Scrape.do": self.scrapedo_configured,
            "Semantic Scholar": bool(self.semantic_scholar_api_key),
            "ScrapeGraphAI LLM": self.enable_llm_enrichment and self.llm_configured,
        }
