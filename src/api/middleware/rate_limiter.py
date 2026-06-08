"""
ModelMesh — Sliding Window Rate Limiter Middleware
---------------------------------------------------
Limits requests per API key (or IP fallback) using an in-memory
sliding window counter. For multi-replica deployments, swap
_counters dict with a Redis backend.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter.

    Each key tracks a deque of request timestamps within the window.
    Requests older than window_seconds are discarded on each check.
    O(1) amortized per request.
    """

    EXEMPT_PATHS = {"/health/live", "/health/ready", "/metrics", "/docs", "/redoc"}

    def __init__(
        self,
        app,
        max_requests: int = 1000,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # key → deque of timestamps
        self._counters: Dict[str, Deque[float]] = defaultdict(deque)

    def _get_client_key(self, request: Request) -> str:
        """Prefer API key header; fall back to client IP."""
        api_key = request.headers.get(settings.security.api_key_header)
        if api_key:
            return f"key:{api_key}"
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or getattr(request.client, "host", "unknown")
        )
        return f"ip:{client_ip}"

    def _is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Returns (allowed, remaining_requests).
        Evicts stale timestamps and checks against limit.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds
        timestamps = self._counters[key]

        # Evict expired entries from left side of deque
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        count = len(timestamps)
        if count >= self.max_requests:
            return False, 0

        timestamps.append(now)
        return True, self.max_requests - count - 1

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health/metrics endpoints
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        key = self._get_client_key(request)
        allowed, remaining = self._is_allowed(key)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                client_key=key[:20],  # truncate for privacy
                path=request.url.path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": self.window_seconds,
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
