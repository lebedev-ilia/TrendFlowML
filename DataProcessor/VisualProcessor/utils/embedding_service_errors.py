"""
Dedicated errors for Embedding Service (semantic heads: brand, car, place, face, franchise).

Used when /health fails or the HTTP client cannot reach the service (connection refused,
timeout, etc.). Callers may catch ``EmbeddingServiceUnavailableError`` to degrade gracefully
when the service is optional; other failures can remain generic ``RuntimeError``.
"""

from __future__ import annotations

import errno
from urllib.parse import urlparse


class EmbeddingServiceUnavailableError(RuntimeError):
    """
    Embedding Service unreachable or failed the health check.

    ``str(exception)`` is a single short line for logs (no URL scheme, no long boilerplate).
    Attributes ``base_url`` and ``reason`` keep the raw details if needed programmatically.
    """

    def __init__(self, base_url: str, reason: str) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.reason = str(reason or "").strip()
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        loc = _embedding_service_display_loc(self.base_url)
        brief = _brief_unreachable_reason(self.reason)
        return f"[Embedding Service] {loc} — {brief}"


def _embedding_service_display_loc(base_url: str) -> str:
    """Host[:port] or netloc — omit scheme for compact logs."""
    u = urlparse(base_url)
    if u.netloc:
        return u.netloc
    s = base_url.replace("https://", "").replace("http://", "").strip("/")
    return s.split("/")[0] if s else base_url


def _brief_unreachable_reason(tip: str) -> str:
    t = tip.lower()
    if "connection refused" in t or "nothing listening" in t:
        return "not running"
    if "timed out" in t or "timeout" in t:
        return "timeout"
    if "unreachable" in t and "host" in t:
        return "network error"
    if "ssl" in t or "tls" in t:
        return "TLS error"
    if "http" in t and "health" in t:
        # e.g. "HTTP 503 on /health" → keep short
        return tip.replace(" on /health", "") if len(tip) < 80 else tip[:77] + "…"
    if len(tip) > 72:
        return tip[:69] + "…"
    return tip


def brief_request_exception_message(exc: BaseException) -> str:
    """
    One-line summary for failed HTTP requests (health checks, etc.).
    Avoids dumping urllib3 MaxRetryError / full connection chain into logs or tracebacks.
    """
    try:
        import requests
    except ImportError:  # pragma: no cover
        requests = None  # type: ignore[assignment]

    for sub in _exception_graph(exc):
        if requests is not None and isinstance(sub, requests.exceptions.Timeout):
            return "request timed out"
        if requests is not None and isinstance(sub, requests.exceptions.HTTPError):
            resp = getattr(sub, "response", None)
            if resp is not None:
                return f"HTTP {resp.status_code} on /health"
            return "HTTP error on /health"
        if requests is not None and isinstance(sub, requests.exceptions.SSLError):
            return "TLS/SSL error"
        if isinstance(sub, ConnectionRefusedError):
            return "connection refused (nothing listening on host:port)"
        if isinstance(sub, OSError) and getattr(sub, "errno", None) is not None:
            en = int(sub.errno)
            if en in (errno.ECONNREFUSED, 111):
                return "connection refused (nothing listening on host:port)"
            if en in (errno.ETIMEDOUT, errno.EHOSTUNREACH, 110, 113):
                return "network unreachable or timed out"

    msg = str(exc).strip().replace("\n", " ")
    if len(msg) > 120:
        return msg[:117] + "…"
    return msg


def _exception_graph(root: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    seen: set[int] = set()
    stack: list[BaseException | None] = [root]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        out.append(e)
        stack.append(e.__cause__)
        stack.append(getattr(e, "__context__", None))
    return out
