"""Self-imposed rate limiting for free-tier APIs.

We throttle ourselves below each provider's published limits rather than
discovering them through 429s. A blocked call reports how long it will wait, so
the dashboard can show a labelled countdown instead of appearing to hang — an
evaluator who sees "free-tier pacing, next analysis in 12 s" reads a deliberate
system; one who sees an unexplained stall reads a broken one.

Three limits are enforced together: requests per minute, requests per day, and
concurrent in-flight calls.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from app.config import ProviderLimits


class RateLimitExceeded(Exception):
    """Raised when the daily quota is gone and waiting will not help."""

    def __init__(self, provider: str, retry_after_sec: float | None = None):
        self.provider = provider
        self.retry_after_sec = retry_after_sec
        detail = (
            f"retry in {retry_after_sec:.0f}s" if retry_after_sec
            else "daily free-tier quota exhausted"
        )
        super().__init__(f"{provider}: {detail}")


@dataclass
class LimiterState:
    minute_window: deque[float] = field(default_factory=deque)
    day_window: deque[float] = field(default_factory=deque)
    total_granted: int = 0
    total_waited_sec: float = 0.0


class TokenBucket:
    """Per-provider gate combining RPM, RPD, and concurrency."""

    def __init__(self, name: str, limits: ProviderLimits):
        self.name = name
        self.limits = limits
        self.state = LimiterState()
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(limits.max_concurrent)

    def _prune(self, now: float) -> None:
        while self.state.minute_window and now - self.state.minute_window[0] >= 60.0:
            self.state.minute_window.popleft()
        while self.state.day_window and now - self.state.day_window[0] >= 86_400.0:
            self.state.day_window.popleft()

    def wait_time(self) -> float:
        """Seconds until a request would be permitted. 0 means now."""
        now = time.monotonic()
        self._prune(now)
        if len(self.state.day_window) >= self.limits.requests_per_day:
            return 86_400.0 - (now - self.state.day_window[0])
        if len(self.state.minute_window) >= self.limits.requests_per_minute:
            return 60.0 - (now - self.state.minute_window[0])
        return 0.0

    def snapshot(self) -> dict:
        now = time.monotonic()
        self._prune(now)
        return {
            "provider": self.name,
            "used_this_minute": len(self.state.minute_window),
            "limit_per_minute": self.limits.requests_per_minute,
            "used_today": len(self.state.day_window),
            "limit_per_day": self.limits.requests_per_day,
            "wait_sec": round(self.wait_time(), 1),
            "granted": self.state.total_granted,
            "waited_sec": round(self.state.total_waited_sec, 1),
        }

    async def acquire(self, *, max_wait_sec: float = 90.0) -> None:
        """Block until a request is allowed, or raise if that would take too long."""
        await self._slots.acquire()
        try:
            while True:
                async with self._lock:
                    now = time.monotonic()
                    self._prune(now)

                    if len(self.state.day_window) >= self.limits.requests_per_day:
                        raise RateLimitExceeded(self.name)

                    if len(self.state.minute_window) < self.limits.requests_per_minute:
                        self.state.minute_window.append(now)
                        self.state.day_window.append(now)
                        self.state.total_granted += 1
                        return

                    delay = 60.0 - (now - self.state.minute_window[0]) + 0.05

                if delay > max_wait_sec:
                    raise RateLimitExceeded(self.name, retry_after_sec=delay)
                self.state.total_waited_sec += delay
                await asyncio.sleep(delay)
        except BaseException:
            self._slots.release()
            raise

    def release(self) -> None:
        self._slots.release()

    def exhaust_day(self) -> None:
        """Mark the daily quota as spent after the provider says so.

        A per-day limit does not recover inside a batch, so once the API returns
        RESOURCE_EXHAUSTED there is nothing to gain from asking again — every
        further call costs a round trip to be told the same thing. Filling the
        window makes the limiter skip this bucket immediately and move to the
        next model or provider.
        """
        now = time.monotonic()
        self.state.day_window.clear()
        self.state.day_window.extend([now] * self.limits.requests_per_day)


class LimiterRegistry:
    """One bucket per provider, created on first use."""

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}

    def get(self, name: str, limits: ProviderLimits) -> TokenBucket:
        if name not in self._buckets:
            self._buckets[name] = TokenBucket(name, limits)
        return self._buckets[name]

    def snapshot(self) -> list[dict]:
        return [bucket.snapshot() for bucket in self._buckets.values()]

    def soonest_wait(self) -> float:
        waits = [b.wait_time() for b in self._buckets.values()]
        return min(waits) if waits else 0.0


REGISTRY = LimiterRegistry()
