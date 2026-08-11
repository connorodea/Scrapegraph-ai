"""Public-page fetchers with direct, ScrapingBee, and Scrape.do transports."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from .config import Settings
from .models import AccessDecision, AccessOutcome, FetchResult
from .policy import AccessPolicy


AuditCallback = Callable[[str, str, dict[str, Any]], None]


class PublicPageFetcher:
    """Fetch approved public pages without crossing authentication boundaries."""

    def __init__(
        self,
        settings: Settings,
        policy: AccessPolicy | None = None,
        audit: AuditCallback | None = None,
    ) -> None:
        self.settings = settings
        self.client = httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "text/html,text/plain,application/xhtml+xml,application/json;q=0.8,*/*;q=0.5",
            },
        )
        self.policy = policy or AccessPolicy(settings)
        self._own_policy = policy is None
        self.audit = audit

    def _emit(self, action: str, message: str, **details: Any) -> None:
        if self.audit:
            self.audit(action, message, details)

    def _redact(self, value: str) -> str:
        redacted = value or ""
        for secret in (
            self.settings.scrapingbee_api_key,
            self.settings.scrapedo_api_key,
        ):
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    def _validate_payload(
        self,
        response: httpx.Response,
        *,
        target_url: str,
        provider: str,
        decision: AccessDecision,
        started: float,
        credits: float,
    ) -> FetchResult:
        content = response.content
        elapsed = int((time.perf_counter() - started) * 1000)
        if len(content) > self.settings.max_response_bytes:
            return FetchResult(
                url=target_url,
                ok=False,
                provider=provider,
                status_code=response.status_code,
                elapsed_ms=elapsed,
                estimated_credits=credits,
                access=decision,
                error=f"Response exceeded the {self.settings.max_response_bytes:,}-byte safety limit.",
            )
        content_type = response.headers.get("content-type", "").lower()
        allowed_types = (
            "text/",
            "application/json",
            "application/xhtml+xml",
            "application/xml",
        )
        if response.status_code < 400 and not any(
            marker in content_type for marker in allowed_types
        ):
            return FetchResult(
                url=target_url,
                ok=False,
                provider=provider,
                status_code=response.status_code,
                content_type=content_type,
                elapsed_ms=elapsed,
                estimated_credits=credits,
                access=decision,
                error="The response was not a text or structured-data document.",
            )
        text = content.decode("utf-8", errors="replace")
        if response.status_code < 400 and self._looks_restricted(text[:8000]):
            blocked = decision.model_copy(
                update={
                    "outcome": AccessOutcome.BLOCKED,
                    "code": "restriction_page",
                    "plain_english": "The page appears to require login, payment, or special authorization.",
                    "technical_detail": "A restriction marker was found in the returned page; the content was not used.",
                    "proxy_escalation_allowed": False,
                }
            )
            return FetchResult(
                url=target_url,
                ok=False,
                provider=provider,
                status_code=response.status_code,
                content_type=content_type,
                elapsed_ms=elapsed,
                estimated_credits=credits,
                access=blocked,
                error=blocked.plain_english,
            )
        return FetchResult(
            url=target_url,
            ok=response.status_code < 400 and bool(text.strip()),
            provider=provider,
            status_code=response.status_code,
            content_type=content_type,
            text=text,
            elapsed_ms=elapsed,
            estimated_credits=credits,
            access=decision,
            error=None if response.status_code < 400 else f"HTTP {response.status_code}",
        )

    @staticmethod
    def _looks_restricted(text: str) -> bool:
        lowered = " ".join(text.lower().split())
        markers = (
            "sign in to continue",
            "log in to continue",
            "login required",
            "subscribe to continue",
            "subscription required",
            "purchase access",
            "institutional access required",
            "this content is only available to",
            "you are not authorized to access",
            "payment required",
        )
        return any(marker in lowered for marker in markers)

    def _direct(self, url: str, decision: AccessDecision) -> FetchResult:
        self._emit("direct_start", "Trying the public page directly.", provider="direct")
        started = time.perf_counter()
        try:
            response = self.client.get(url)
        except httpx.HTTPError as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            return FetchResult(
                url=url,
                ok=False,
                provider="direct",
                elapsed_ms=elapsed,
                access=decision,
                error=self._redact(str(exc)),
            )
        return self._validate_payload(
            response,
            target_url=url,
            provider="direct",
            decision=decision,
            started=started,
            credits=0.0,
        )

    def _scrapingbee(self, url: str, decision: AccessDecision) -> FetchResult:
        if not self.settings.scrapingbee_api_key:
            return FetchResult(
                url=url,
                ok=False,
                provider="ScrapingBee",
                access=decision,
                error="ScrapingBee is not configured.",
            )
        render = self.settings.allow_javascript_rendering
        self._emit(
            "provider_start",
            "Trying ScrapingBee on an approved public page.",
            provider="ScrapingBee",
            javascript=render,
        )
        started = time.perf_counter()
        try:
            response = self.client.get(
                "https://app.scrapingbee.com/api/v1",
                params={
                    "url": url,
                    "render_js": str(render).lower(),
                    "return_page_markdown": "true",
                    "block_resources": "true",
                },
                headers={
                    "Authorization": f"Bearer {self.settings.scrapingbee_api_key}",
                    "Accept": "text/plain,text/markdown,text/html,*/*;q=0.5",
                },
            )
        except httpx.HTTPError as exc:
            return FetchResult(
                url=url,
                ok=False,
                provider="ScrapingBee",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                estimated_credits=5.0 if render else 1.0,
                access=decision,
                error=self._redact(str(exc)),
            )
        return self._validate_payload(
            response,
            target_url=url,
            provider="ScrapingBee",
            decision=decision,
            started=started,
            credits=5.0 if render else 1.0,
        )

    def _scrapedo(self, url: str, decision: AccessDecision) -> FetchResult:
        if not self.settings.scrapedo_api_key:
            return FetchResult(
                url=url,
                ok=False,
                provider="Scrape.do",
                access=decision,
                error="Scrape.do is not configured.",
            )
        render = self.settings.allow_javascript_rendering
        self._emit(
            "provider_start",
            "Trying Scrape.do on an approved public page.",
            provider="Scrape.do",
            javascript=render,
        )
        started = time.perf_counter()
        try:
            response = self.client.get(
                "https://api.scrape.do/",
                params={
                    "token": self.settings.scrapedo_api_key,
                    "url": url,
                    "output": "markdown",
                    "render": str(render).lower(),
                    "super": "false",
                    "blockResources": "true",
                },
                headers={"Accept": "text/plain,text/markdown,text/html,*/*;q=0.5"},
            )
        except httpx.HTTPError as exc:
            return FetchResult(
                url=url,
                ok=False,
                provider="Scrape.do",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                estimated_credits=5.0 if render else 1.0,
                access=decision,
                error=self._redact(str(exc)),
            )
        return self._validate_payload(
            response,
            target_url=url,
            provider="Scrape.do",
            decision=decision,
            started=started,
            credits=5.0 if render else 1.0,
        )

    def fetch(
        self,
        url: str,
        *,
        user_approved_domains: Iterable[str] = (),
        provider_order: Iterable[str] | None = None,
    ) -> FetchResult:
        decision = self.policy.approve_fetch(
            url,
            user_approved_domains=user_approved_domains,
        )
        self._emit(
            "policy",
            decision.plain_english,
            outcome=decision.outcome.value,
            code=decision.code,
        )
        if decision.outcome != AccessOutcome.ALLOWED:
            return FetchResult(
                url=url,
                ok=False,
                provider="policy",
                access=decision,
                error=decision.plain_english,
            )

        direct = self._direct(url, decision)
        if direct.ok:
            self._emit("success", "The public page was read directly.", provider="direct")
            return direct

        status = direct.status_code or 503
        eligible, reason = self.policy.may_escalate_after_response(
            decision,
            status,
            direct.text[:4000],
        )
        self._emit("escalation_check", reason, eligible=eligible, status=status)
        if not eligible:
            return direct

        order = list(provider_order or [])
        if not order:
            first = self.settings.default_proxy_provider.lower()
            order = [first, "scrapedo" if first == "scrapingbee" else "scrapingbee"]
        attempts: list[FetchResult] = [direct]
        for provider in order:
            normalized = provider.strip().lower().replace(".", "")
            if normalized in {"scrapingbee", "bee"}:
                result = self._scrapingbee(url, decision)
            elif normalized in {"scrapedo", "scrape do", "do"}:
                result = self._scrapedo(url, decision)
            else:
                continue
            attempts.append(result)
            if result.ok:
                self._emit(
                    "success",
                    f"The approved public page was read with {result.provider}.",
                    provider=result.provider,
                    estimated_credits=result.estimated_credits,
                )
                return result

        errors = self._redact(
            "; ".join(
                f"{attempt.provider}: {attempt.error or attempt.status_code}"
                for attempt in attempts
            )
        )
        last = attempts[-1]
        return last.model_copy(update={"error": errors})

    def close(self) -> None:
        self.client.close()
        if self._own_policy:
            self.policy.close()
