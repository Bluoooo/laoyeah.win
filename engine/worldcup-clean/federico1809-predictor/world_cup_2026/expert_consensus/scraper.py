"""
scraper.py
----------
Fetch and parse article text from URLs for the expert consensus pipeline.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from loguru import logger
from newspaper import Article

from world_cup_2026.config import DATA_DIR

RAW_DIR = DATA_DIR / "expert_opinions" / "raw"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_MIN_WORDS = 200


class PaywallError(Exception):
    """Raised when extracted text is too short — likely paywalled or empty."""


def scrape_url(url: str) -> dict:
    """Fetch article text from a URL and save raw text to disk.

    Uses newspaper4k (imported as `newspaper`) which follows Google News
    redirects internally, resolving CBMi tracking URLs to real article URLs.

    Returns:
        dict with keys: url, source_domain, title, date, text, scraped_at

    Raises:
        PaywallError: if extracted text has fewer than 200 words.
        Exception: propagates newspaper/network errors to the pipeline.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    scraped_at = datetime.now(timezone.utc).isoformat()

    article = Article(url, headers=_HEADERS, language="en")
    article.download()
    article.parse()

    title = article.title or ""
    text = re.sub(r"\s+", " ", article.text or "").strip()

    word_count = len(text.split())
    if word_count < _MIN_WORDS:
        raise PaywallError(
            f"Text too short ({word_count} words) — likely paywalled: {url}"
        )

    # Prefer canonical_link, then newspaper's internally resolved URL, then original
    resolved_url = article.canonical_link or article.url or url
    # Guard: if canonical_link still points to Google News, skip it
    if "news.google.com" in resolved_url:
        resolved_url = article.url or url

    source_domain = urlparse(resolved_url).netloc.removeprefix("www.")

    if article.publish_date:
        date = article.publish_date.date().isoformat()
    else:
        soup = BeautifulSoup(article.html or "", "html.parser")
        date = _extract_date(soup)

    safe_domain = re.sub(r"[^\w.-]", "_", source_domain)
    raw_path = RAW_DIR / f"{safe_domain}_{date}.txt"
    raw_path.write_text(f"{title}\n\n{text}", encoding="utf-8")
    logger.info(f"Saved raw text → {raw_path}")

    return {
        "url": resolved_url,
        "source_domain": source_domain,
        "title": title,
        "date": date,
        "text": text,
        "scraped_at": scraped_at,
    }


def _extract_date(soup: BeautifulSoup) -> str:
    """Extract publication date from meta tags; falls back to today (UTC)."""
    for attr, prop in [
        ("property", "article:published_time"),
        ("name", "date"),
        ("itemprop", "datePublished"),
    ]:
        tag = soup.find("meta", attrs={attr: prop})
        if tag and tag.get("content"):
            try:
                return datetime.fromisoformat(tag["content"][:10]).date().isoformat()
            except ValueError:
                pass
    return datetime.now(timezone.utc).date().isoformat()
