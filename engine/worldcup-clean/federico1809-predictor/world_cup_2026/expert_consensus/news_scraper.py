"""
news_scraper.py
---------------
Automatically find relevant WC2026 prediction articles and populate
data/expert_opinions/urls.txt for the expert consensus pipeline.

Usage:
    python -m world_cup_2026.expert_consensus.news_scraper
    python -m world_cup_2026.expert_consensus.news_scraper --dry-run
"""
from __future__ import annotations

import datetime
from pathlib import Path

import typer
from dateutil import parser as dateutil_parser
from gnews import GNews
from loguru import logger

from world_cup_2026.config import DATA_DIR

app = typer.Typer(add_completion=False)

_URLS_FILE = DATA_DIR / "expert_opinions" / "urls.txt"
_START_DATE = (2025, 5, 1)
_MAX_RESULTS_PER_QUERY = 10

# ---------------------------------------------------------------------------
# Queries grouped by language → (language, country, queries)
# ---------------------------------------------------------------------------
_QUERY_GROUPS: list[tuple[str, str, list[str]]] = [
    ("en", "US", [
        "World Cup 2026 winner prediction",
        "who will win World Cup 2026",
        "FIFA World Cup 2026 favorites",
        "World Cup 2026 best teams analysis",
        "Argentina World Cup 2026 prediction",
        "France World Cup 2026 favorite",
        "Spain World Cup 2026 chances",
        "Brazil World Cup 2026 prediction",
        "England World Cup 2026 analysis",
    ]),
    ("es", "ES", [
        "pronostico campeón Mundial 2026",
        "quien va a ganar el Mundial 2026",
        "Copa Mundial 2026 favorito",
        "Mundial 2026 analisis favoritos",
        "Argentina Copa Mundial 2026",
        "Francia favorita Mundial 2026",
    ]),
    ("fr", "FR", [
        "pronostic vainqueur Coupe du Monde 2026",
        "favori Coupe du Monde 2026",
    ]),
]

BLOCKED_DOMAINS: list[str] = [
    "ticketmaster", "viagogo", "stubhub", "linkedin",
    "reddit", "twitter", "facebook", "instagram",
    "youtube", "wikipedia",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}


def _is_blocked(url: str) -> bool:
    return any(domain in url for domain in BLOCKED_DOMAINS)


def _parse_published_date(raw: str) -> datetime.datetime | None:
    try:
        return dateutil_parser.parse(raw, ignoretz=True)
    except (ValueError, OverflowError):
        return None


def _is_recent_enough(raw_date: str, cutoff: datetime.datetime) -> bool:
    parsed = _parse_published_date(raw_date)
    if parsed is None:
        return True  # can't verify — let it through, pipeline will handle it
    return parsed >= cutoff


def decode_google_news_url(url: str) -> str:
    """Resolve a Google News tracking URL to the real article URL.
    Returns the original URL unchanged if decoding fails.
    """
    if "news.google.com" not in url:
        return url
    try:
        from googlenewsdecoder import new_decoderv1
        result = new_decoderv1(url, interval=1)
        if result.get("status") == True and result.get("decoded_url"):  # noqa: E712
            return result["decoded_url"]
    except Exception as e:
        logger.warning(f"Could not decode Google News URL: {url} — {e}")
    return url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print URLs without writing to file."),
) -> None:
    _URLS_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = _load_existing_urls(_URLS_FILE)
    logger.info(f"Existing URLs in file: {len(seen)}")

    cutoff = datetime.datetime(*_START_DATE)
    today = datetime.datetime.now()
    new_urls: list[str] = []
    resolved_count = 0
    fallback_count = 0

    for language, country, queries in _QUERY_GROUPS:
        client = GNews(
            language=language,
            country=country,
            max_results=_MAX_RESULTS_PER_QUERY,
            start_date=_START_DATE,
            end_date=(today.year, today.month, today.day),
            exclude_websites=BLOCKED_DOMAINS,
        )

        for query in queries:
            try:
                results = client.get_news(query)
            except Exception as e:
                logger.warning(f"Query failed [{language}] '{query}': {e}")
                continue

            added_this_query = 0
            for article in results:
                raw_url = article.get("url", "").strip()
                if not raw_url:
                    continue
                url = decode_google_news_url(raw_url)
                if url != raw_url:
                    resolved_count += 1
                elif "news.google.com" in url:
                    fallback_count += 1
                if url in seen:
                    continue
                if _is_blocked(url):
                    logger.debug(f"Blocked domain — skipping: {url}")
                    continue
                if not _is_recent_enough(article.get("published date", ""), cutoff):
                    logger.debug(f"Too old — skipping: {url}")
                    continue

                seen.add(url)
                new_urls.append(url)
                added_this_query += 1

            logger.info(f"[{language}] '{query}' → {added_this_query} new URLs")

    logger.info(
        f"URL resolution: {resolved_count} resolved, "
        f"{fallback_count} fell back to Google News tracking URLs"
    )

    if not new_urls:
        logger.info("No new URLs found.")
        return

    if dry_run:
        logger.info(f"Dry run — {len(new_urls)} URLs would be added:")
        for url in new_urls:
            logger.info(f"  {url}")
        return

    with _URLS_FILE.open("a", encoding="utf-8") as fh:
        for url in new_urls:
            fh.write(url + "\n")

    logger.success(f"Added {len(new_urls)} new URLs → {_URLS_FILE}")


if __name__ == "__main__":
    app()
