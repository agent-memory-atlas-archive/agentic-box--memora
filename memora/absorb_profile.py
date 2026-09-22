"""Per-phase wall time and DB request counts for one memory_absorb call.

absorb_memory opens one AbsorbProfile per call; code inside the call marks
phases with ``absorb_phase("name")``. Phases nest and are accounted
EXCLUSIVELY: while a child phase runs, its parent's clock and request
counter are paused, so the per-phase numbers sum to the call total (plus the
"other" bucket for time outside any named phase).

Request counting reads the connection's own counter:
  - D1Connection.request_count — one per HTTPS POST to the D1 query API.
  - any other conn exposing request_count (the offline FakeD1 test double).
  - plain sqlite3.Connection — a trace callback counts executed statements.
    Local statements are not network round-trips; the number is reported
    for comparison only.

Everything here is observational: a profile never changes what absorb does,
and code that runs outside an active profile (every non-absorb caller) pays
one ContextVar lookup per phase mark.
"""

from __future__ import annotations

import contextvars
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

_ACTIVE: contextvars.ContextVar[Optional["AbsorbProfile"]] = contextvars.ContextVar(
    "memora_absorb_profile", default=None
)


class _SqliteStatementCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, _statement: str) -> None:
        self.count += 1


class AbsorbProfile:
    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._sqlite_counter: Optional[_SqliteStatementCounter] = None
        if getattr(conn, "request_count", None) is None and isinstance(conn, sqlite3.Connection):
            self._sqlite_counter = _SqliteStatementCounter()
            conn.set_trace_callback(self._sqlite_counter)
        self.phases: Dict[str, Dict[str, float]] = {}
        self.counters: Dict[str, int] = {}
        # Stack of [name, started_at, requests_at_start] — only the top runs.
        self._stack: List[List[Any]] = []
        self._t0 = time.perf_counter()
        self._r0 = self._requests()
        self.total_seconds = 0.0
        self.total_requests = 0

    def _requests(self) -> int:
        if self._sqlite_counter is not None:
            return self._sqlite_counter.count
        return int(getattr(self._conn, "request_count", 0) or 0)

    def _charge(self, name: str, seconds: float, requests: int) -> None:
        slot = self.phases.setdefault(name, {"seconds": 0.0, "requests": 0, "calls": 0})
        slot["seconds"] += seconds
        slot["requests"] += requests

    def enter(self, name: str) -> None:
        now, req = time.perf_counter(), self._requests()
        if self._stack:
            top = self._stack[-1]
            self._charge(top[0], now - top[1], req - top[2])
        self._stack.append([name, now, req])
        self.phases.setdefault(name, {"seconds": 0.0, "requests": 0, "calls": 0})["calls"] += 1

    def exit(self) -> None:
        now, req = time.perf_counter(), self._requests()
        name, started, req0 = self._stack.pop()
        self._charge(name, now - started, req - req0)
        if self._stack:
            # Resume the parent's clock from now.
            self._stack[-1][1] = now
            self._stack[-1][2] = req

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def finish(self) -> Dict[str, Any]:
        while self._stack:
            self.exit()
        self.total_seconds = time.perf_counter() - self._t0
        self.total_requests = self._requests() - self._r0
        if self._sqlite_counter is not None:
            try:
                self._conn.set_trace_callback(None)
            except Exception:  # pragma: no cover - closed connection
                pass
        named_s = sum(p["seconds"] for p in self.phases.values())
        named_r = sum(p["requests"] for p in self.phases.values())
        phases = {
            name: {
                "seconds": round(p["seconds"], 4),
                "requests": int(p["requests"]),
                "calls": int(p["calls"]),
            }
            for name, p in self.phases.items()
        }
        phases["other"] = {
            "seconds": round(max(0.0, self.total_seconds - named_s), 4),
            "requests": int(self.total_requests - named_r),
            "calls": 0,
        }
        return {
            "request_unit": (
                "sqlite_statements" if self._sqlite_counter is not None else "d1_requests"
            ),
            "total_seconds": round(self.total_seconds, 4),
            "total_requests": int(self.total_requests),
            "phases": phases,
            "counters": dict(self.counters),
        }


@contextmanager
def absorb_profile(conn: Any) -> Iterator[AbsorbProfile]:
    profile = AbsorbProfile(conn)
    token = _ACTIVE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE.reset(token)


@contextmanager
def absorb_phase(name: str) -> Iterator[None]:
    profile = _ACTIVE.get()
    if profile is None:
        yield
        return
    profile.enter(name)
    try:
        yield
    finally:
        profile.exit()


def absorb_count(key: str, n: int = 1) -> None:
    profile = _ACTIVE.get()
    if profile is not None:
        profile.count(key, n)
