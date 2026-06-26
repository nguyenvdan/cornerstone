"""Polite, cached HTTP fetching.

Every URL is cached to ``data/raw/`` after the first successful fetch, so the
whole pipeline is reproducible offline once the cache is warm. Requests are
rate-limited and retried with exponential backoff to respect the source's
stated limits.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import config

_LAST_REQUEST_AT = 0.0


def _cache_path(url: str) -> Path:
    """Stable, human-readable cache filename for a URL."""
    parsed = urlparse(url)
    slug = (parsed.path.strip("/").replace("/", "_") or "index")
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    return config.RAW / f"{parsed.netloc}__{slug}__{digest}.html"


def _throttle() -> None:
    global _LAST_REQUEST_AT
    elapsed = time.time() - _LAST_REQUEST_AT
    wait = config.REQUEST_DELAY_SEC - elapsed
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST_AT = time.time()


def get(url: str, *, force: bool = False) -> str:
    """Return page HTML, using the on-disk cache when available.

    Args:
        url: Absolute URL to fetch.
        force: If True, bypass the cache and re-fetch.
    """
    cache = _cache_path(url)
    if cache.exists() and not force:
        return cache.read_text(encoding="utf-8", errors="replace")

    last_err: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        _throttle()
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": config.USER_AGENT},
                timeout=config.TIMEOUT_SEC,
            )
            if resp.status_code == 200:
                # BBRef/SRef serve UTF-8 but often omit charset, so requests
                # guesses Latin-1 and mangles accented names. Decode explicitly.
                text = resp.content.decode("utf-8", errors="replace")
                cache.write_text(text, encoding="utf-8")
                return text
            # 429/5xx -> back off and retry; 404 -> give up immediately.
            if resp.status_code == 404:
                raise FileNotFoundError(f"404 Not Found: {url}")
            last_err = RuntimeError(f"HTTP {resp.status_code} for {url}")
        except (requests.RequestException, RuntimeError) as exc:
            last_err = exc
        backoff = min(60, 2**attempt)
        time.sleep(backoff)

    raise RuntimeError(f"Failed to fetch {url} after {config.MAX_RETRIES} tries") from last_err


def is_cached(url: str) -> bool:
    return _cache_path(url).exists()
