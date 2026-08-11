"""Conservative access policy for public research discovery."""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

from .config import Settings
from .models import AccessDecision, AccessOutcome


@dataclass(slots=True)
class RobotsCacheEntry:
    parser: RobotFileParser
    fetched_at: float


class AccessPolicy:
    """Approve public URLs and keep restricted sources outside the fetch pipeline."""

    RESTRICTION_MARKERS = (
        "sign in",
        "log in",
        "login required",
        "subscribe to continue",
        "subscription required",
        "purchase access",
        "institutional access",
        "access denied",
        "unauthorized",
        "forbidden",
        "paywall",
    )
    API_ONLY_HOSTS = {"github.com", "www.github.com", "api.github.com"}

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": settings.user_agent},
        )
        self._robots_cache: dict[str, RobotsCacheEntry] = {}
        self._robots_lock = threading.Lock()

    @staticmethod
    def _normalized_url(url: str) -> str:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        return urlunparse(
            (parsed.scheme.lower(), netloc, parsed.path or "/", "", parsed.query, "")
        )

    @staticmethod
    def _domain_matches(host: str, domains: Iterable[str]) -> bool:
        host = host.lower().rstrip(".")
        return any(
            host == domain.lower().rstrip(".")
            or host.endswith("." + domain.lower().rstrip("."))
            for domain in domains
        )

    @staticmethod
    def _is_private_ip(value: str) -> bool:
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            return False
        return any(
            (
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            )
        )

    def _host_resolves_private(self, host: str) -> bool:
        if self._is_private_ip(host):
            return True
        try:
            records = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, TimeoutError, OSError):
            return False
        return any(self._is_private_ip(record[4][0]) for record in records)

    def evaluate_url(
        self,
        url: str,
        *,
        api_source: bool = False,
        user_approved_domains: Iterable[str] = (),
    ) -> AccessDecision:
        try:
            raw = urlparse(url.strip())
            if raw.username or raw.password:
                return AccessDecision(
                    outcome=AccessOutcome.BLOCKED,
                    code="embedded_credentials",
                    plain_english="Addresses containing usernames or passwords are blocked.",
                    technical_detail="Credentials embedded in a URL can leak secrets.",
                )
            parsed = urlparse(self._normalized_url(url))
        except Exception as exc:
            return AccessDecision(
                outcome=AccessOutcome.BLOCKED,
                code="invalid_url",
                plain_english="This address is not a valid public web URL.",
                technical_detail=str(exc),
            )

        if parsed.scheme not in {"http", "https"}:
            return AccessDecision(
                outcome=AccessOutcome.BLOCKED,
                code="unsupported_scheme",
                plain_english="Only normal public HTTP and HTTPS pages are allowed.",
                technical_detail=f"Rejected URL scheme: {parsed.scheme or '<empty>'}",
            )
        host = (parsed.hostname or "").lower()
        if not host:
            return AccessDecision(
                outcome=AccessOutcome.BLOCKED,
                code="missing_host",
                plain_english="The address does not identify a public website.",
            )
        if self._domain_matches(host, self.settings.blocked_domains):
            return AccessDecision(
                outcome=AccessOutcome.BLOCKED,
                code="blocked_host",
                plain_english="Local, internal, and infrastructure addresses are blocked.",
                technical_detail=f"Blocked host: {host}",
            )
        if self._host_resolves_private(host):
            return AccessDecision(
                outcome=AccessOutcome.BLOCKED,
                code="private_network",
                plain_english="This address resolves to a private or internal network.",
                technical_detail="Network safety checks rejected a non-public address.",
            )
        if host in self.API_ONLY_HOSTS and not api_source:
            return AccessDecision(
                outcome=AccessOutcome.API_ONLY,
                code="github_api_only",
                plain_english="GitHub content is collected through GitHub's official API.",
                technical_detail="Use api.github.com for repository metadata and files.",
                proxy_escalation_allowed=False,
            )

        approved = tuple(self.settings.proxy_approved_domains) + tuple(
            user_approved_domains
        )
        alternate_transport_allowed = (
            self.settings.allow_proxy_escalation
            and self._domain_matches(host, approved)
        )
        return AccessDecision(
            outcome=AccessOutcome.ALLOWED,
            code="public_url",
            plain_english="The address is a public web page and passed the safety checks.",
            technical_detail=f"Approved public host: {host}",
            proxy_escalation_allowed=alternate_transport_allowed,
        )

    def robots_allowed(self, url: str) -> tuple[bool | None, str]:
        if not self.settings.respect_robots_txt:
            return True, "robots.txt checking is disabled by configuration"
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = f"{origin}/robots.txt"
        now = time.time()
        with self._robots_lock:
            cached = self._robots_cache.get(origin)
            if cached and now - cached.fetched_at < 3600:
                return (
                    cached.parser.can_fetch(self.settings.user_agent, url),
                    f"robots.txt cache: {robots_url}",
                )

        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            response = self.client.get(robots_url)
            if response.status_code == 404:
                return True, "No robots.txt file was published."
            response.raise_for_status()
            parser.parse(response.text.splitlines())
        except httpx.HTTPError as exc:
            return None, f"robots.txt could not be read directly: {exc}"
        with self._robots_lock:
            self._robots_cache[origin] = RobotsCacheEntry(parser, now)
        return parser.can_fetch(self.settings.user_agent, url), f"robots.txt evaluated: {robots_url}"

    def approve_fetch(
        self,
        url: str,
        *,
        api_source: bool = False,
        user_approved_domains: Iterable[str] = (),
    ) -> AccessDecision:
        decision = self.evaluate_url(
            url,
            api_source=api_source,
            user_approved_domains=user_approved_domains,
        )
        if decision.outcome != AccessOutcome.ALLOWED or api_source:
            return decision
        allowed, detail = self.robots_allowed(url)
        if allowed is False:
            return AccessDecision(
                outcome=AccessOutcome.BLOCKED,
                code="robots_disallow",
                plain_english="The website's robots.txt asks automated tools not to fetch this page.",
                technical_detail=detail,
                robots_allowed=False,
                proxy_escalation_allowed=False,
            )
        if allowed is None:
            return decision.model_copy(
                update={
                    "robots_allowed": None,
                    "proxy_escalation_allowed": False,
                    "technical_detail": (
                        f"{decision.technical_detail}; {detail}; "
                        "proxy fallback disabled because robots.txt status is unknown"
                    ),
                }
            )
        return decision.model_copy(
            update={
                "robots_allowed": True,
                "technical_detail": f"{decision.technical_detail}; {detail}",
            }
        )

    def may_escalate_after_response(
        self,
        decision: AccessDecision,
        status_code: int,
        body_preview: str,
    ) -> tuple[bool, str]:
        text = (body_preview or "").lower()
        if status_code in {401, 402}:
            return False, "Authentication or payment is required; no additional transport is used."
        if any(marker in text for marker in self.RESTRICTION_MARKERS):
            return False, "The response appears to require login, subscription, or authorization."
        if not decision.proxy_escalation_allowed:
            return False, "Alternate transport is disabled or the domain is not approved."
        if status_code in {403, 408, 425, 429, 500, 502, 503, 504}:
            return True, "The approved public page had a transient delivery failure."
        return False, f"HTTP {status_code} is not eligible for alternate transport."

    def close(self) -> None:
        self.client.close()
