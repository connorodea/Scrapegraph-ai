"""Small resilient HTTP client with bounded retries and rate-limit awareness."""

from __future__ import annotations

import email.utils
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from .config import Settings


class RadarHttpError(RuntimeError):
    """Raised when a remote API cannot be read safely or reliably."""


@dataclass(slots=True)
class HttpResult:
    status_code: int
    headers: dict[str, str]
    content: bytes
    elapsed_ms: int
    url: str
    rate_limit: dict[str, int | str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        import json

        return json.loads(self.text)


class ResilientHttpClient:
    """Synchronous client designed for Streamlit's straightforward execution model."""

    RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        settings: Settings,
        *,
        default_headers: dict[str, str] | None = None,
        on_wait: Callable[[float, str], None] | None = None,
    ) -> None:
        headers = {"User-Agent": settings.user_agent, "Accept": "application/json"}
        if default_headers:
            headers.update(default_headers)
        self.settings = settings
        self.on_wait = on_wait
        self.client = httpx.Client(
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
            headers=headers,
        )

    @staticmethod
    def _rate_limit(headers: httpx.Headers) -> dict[str, int | str]:
        result: dict[str, int | str] = {}
        for header, key in (
            ("x-ratelimit-limit", "limit"),
            ("x-ratelimit-remaining", "remaining"),
            ("x-ratelimit-used", "used"),
            ("x-ratelimit-reset", "reset"),
            ("x-ratelimit-resource", "resource"),
        ):
            value = headers.get(header)
            if value is None:
                continue
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
        return result

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                try:
                    parsed = email.utils.parsedate_to_datetime(raw)
                    return max(
                        0.0,
                        (parsed - datetime.now(timezone.utc)).total_seconds(),
                    )
                except (TypeError, ValueError):
                    pass
        remaining = response.headers.get("x-ratelimit-remaining")
        reset = response.headers.get("x-ratelimit-reset")
        if remaining == "0" and reset:
            try:
                return max(0.0, float(reset) - time.time())
            except ValueError:
                pass
        return None

    def _wait(self, seconds: float, reason: str) -> None:
        bounded = min(max(seconds, 0.0), 60.0)
        if self.on_wait:
            self.on_wait(bounded, reason)
        time.sleep(bounded)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_attempts: int = 4,
        expected_content: str | None = None,
    ) -> HttpResult:
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self.client.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                delay = min(2 ** (attempt - 1) + random.random(), 15.0)
                self._wait(delay, f"Network retry {attempt}/{max_attempts}")
                continue

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            content = response.content
            if len(content) > self.settings.max_response_bytes:
                raise RadarHttpError(
                    f"Response exceeded the {self.settings.max_response_bytes:,}-byte safety limit."
                )

            retry_after = self._retry_after_seconds(response)
            rate_limited_forbidden = (
                response.status_code == 403 and retry_after is not None
            )
            if (
                response.status_code in self.RETRYABLE_STATUS
                or rate_limited_forbidden
            ) and attempt < max_attempts:
                delay = retry_after
                if delay is None:
                    delay = min(2 ** (attempt - 1) + random.random(), 15.0)
                self._wait(delay, f"HTTP {response.status_code} retry {attempt}/{max_attempts}")
                continue

            if response.status_code >= 400:
                preview = content[:800].decode("utf-8", errors="replace")
                raise RadarHttpError(
                    f"HTTP {response.status_code} from {response.url.host}: {preview}"
                )

            if expected_content:
                content_type = response.headers.get("content-type", "").lower()
                if expected_content not in content_type:
                    raise RadarHttpError(
                        f"Unexpected content type '{content_type or 'unknown'}' from {response.url.host}."
                    )

            return HttpResult(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=content,
                elapsed_ms=elapsed_ms,
                url=str(response.url),
                rate_limit=self._rate_limit(response.headers),
            )

        raise RadarHttpError(f"Request failed after {max_attempts} attempts: {last_error}")

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_attempts: int = 4,
    ) -> tuple[Any, HttpResult]:
        result = self.request(
            "GET",
            url,
            params=params,
            headers=headers,
            max_attempts=max_attempts,
        )
        try:
            return result.json(), result
        except Exception as exc:
            raise RadarHttpError(f"Invalid JSON returned by {url}: {exc}") from exc

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_attempts: int = 4,
    ) -> tuple[str, HttpResult]:
        result = self.request(
            "GET",
            url,
            params=params,
            headers=headers,
            max_attempts=max_attempts,
        )
        return result.text, result

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ResilientHttpClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
