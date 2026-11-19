"""Politeness controls for web collection.

Enforces rate limiting, robots.txt compliance, honest User-Agent,
and backoff on 429/5xx responses.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# Project User-Agent: honest, no personal details
USER_AGENT = "WasaltResearchBot/1.0 (academic real-estate RAG project)"

# Disallowed paths from wasalt.sa robots.txt (checked 18 Nov 2026)
# User-agent: *
# Allow: /
# Disallow: /search
# Disallow: /cdn-cgi/rum
WASALT_DISALLOWED_PREFIXES = ("/search", "/cdn-cgi/rum")


def is_allowed_wasalt(path: str) -> bool:
    """Check if a path is allowed by wasalt.sa robots.txt."""
    for prefix in WASALT_DISALLOWED_PREFIXES:
        if path.startswith(prefix):
            return False
    return True


class RateLimiter:
    """Simple rate limiter: at most 1 request per `interval` seconds."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._last: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self._interval:
            delay = self._interval - elapsed
            time.sleep(delay)
        self._last = time.monotonic()


class BackoffHandler:
    """Exponential backoff on 429 and 5xx responses."""

    def __init__(
        self,
        initial_delay: float = 5.0,
        max_delay: float = 120.0,
        multiplier: float = 2.0,
        max_retries: int = 5,
    ) -> None:
        self._initial = initial_delay
        self._max_delay = max_delay
        self._multiplier = multiplier
        self._max_retries = max_retries

    def should_retry(self, status_code: int, attempt: int) -> float | None:
        """Return delay in seconds if should retry, None if should give up.

        Only retries on 429 and 5xx.
        """
        if status_code == 429 or status_code >= 500:
            if attempt >= self._max_retries:
                return None
            delay = min(self._initial * (self._multiplier ** attempt), self._max_delay)
            logger.warning(
                "HTTP %d on attempt %d, backing off %.1f s",
                status_code, attempt, delay,
            )
            return delay
        return None  # non-retryable status
