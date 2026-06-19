"""
transfermarkt.py — One-shot scraper for WC2026 squad features.

Scrapes avg_age, squad_size, and continuity_pct (vs Qatar 2022) 
for all 48 WC2026 teams from Transfermarkt.

Usage:
    python -m world_cup_2026.data_ingestion.transfermarkt
"""

import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from loguru import logger

from world_cup_2026.config import RAW_DATA_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
BASE_URL = "https://www.transfermarkt.com"
SLEEP    = 1.5  # seconds between requests


def get_squad_names(slug, verein_id, season=None):
    """Return set of player names from a squad page."""
    if season:
        url = f"{BASE_URL}/{slug}/kader/verein/{verein_id}/saison_id/{season}"
    else:
        url = f"{BASE_URL}/{slug}/kader/verein/{verein_id}"

    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        logger.warning(f"HTTP {r.status_code} for {url}")
        return set()

    soup  = BeautifulSoup(r.content, "html.parser")
    table = soup.find("table", class_="items")
    if not table:
        logger.warning(f"No squad table found for {url}")
        return set()

    names = set()
    for row in table.find_all("tr"):
        name_td = row.find("td", class_="hauptlink")
        if name_td and name_td.find("a"):
            names.add(name_td.find("a").get_text(strip=True))
    return names


def get_squad_meta(slug, verein_id):
    """Return avg_age and squad_size from team header page."""
    url = f"{BASE_URL}/{slug}/startseite/verein/{verein_id}"
    r   = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        logger.warning(f"HTTP {r.status_code} for {url}")
        return None, None

    soup     = BeautifulSoup(r.content, "html.parser")
    avg_age  = None
    squad_sz = None

    for li in soup.find_all("li", class_="data-header__label"):
        text = li.get_text(strip=True)
        if "Average age" in text:
            try:
                avg_age = float(text.replace("Average age:", "").strip())
            except ValueError:
                pass
        if "Squad size" in text:
            try:
                squad_sz = int(text.replace("Squad size:", "").strip())
            except ValueError:
                pass

    return avg_age, squad_sz


def scrape_all(ids_path, wc2026_teams):
    """
    Scrape Transfermarkt for all WC2026 teams.
    Returns DataFrame with team, avg_age, squad_size, continuity_pct.
    """
    df_ids = pd.read_csv(ids_path)
    # Filter to WC2026 teams only
    df_ids = df_ids[df_ids["team"].isin(wc2026_teams)].reset_index(drop=True)
    logger.info(f"Scraping {len(df_ids)} teams...")

    rows = []
    for _, row in df_ids.iterrows():
        team      = row["team"]
        slug      = row["slug"]
        verein_id = row["verein_id"]
        logger.info(f"  {team}...")

        # Current squad meta
        avg_age, squad_size = get_squad_meta(slug, verein_id)
        time.sleep(SLEEP)

        # Current squad names
        current_squad = get_squad_names(slug, verein_id)
        time.sleep(SLEEP)

        # Qatar 2022 squad names
        qatar_squad = get_squad_names(slug, verein_id, season=2022)
        time.sleep(SLEEP)

        # Continuity
        if current_squad and qatar_squad:
            overlap         = len(current_squad & qatar_squad)
            continuity_pct  = round(overlap / len(current_squad) * 100, 1)
        else:
            continuity_pct  = None

        rows.append({
            "team":            team,
            "avg_age":         avg_age,
            "squad_size":      squad_size,
            "continuity_pct":  continuity_pct,
            "current_n":       len(current_squad),
            "qatar_n":         len(qatar_squad),
        })

        logger.info(
            f"    avg_age={avg_age}, squad_size={squad_size}, "
            f"continuity={continuity_pct}% ({len(current_squad & qatar_squad) if current_squad and qatar_squad else 0} shared)"
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from world_cup_2026.config import RAW_DATA_DIR

    ids_path = RAW_DATA_DIR / "transfermarkt" / "tm_team_ids.csv"
    out_path = RAW_DATA_DIR / "transfermarkt" / "tm_squad_features.csv"

    # WC2026 confirmed teams
    import pandas as pd
    df_teams   = pd.read_csv(RAW_DATA_DIR / "areezvisram12_fixture" / "teams.csv")
    wc2026     = df_teams[~df_teams["is_placeholder"]]["team_name"].tolist()

    df_results = scrape_all(ids_path, wc2026)
    df_results.to_csv(out_path, index=False)
    logger.success(f"Saved → {out_path}")
    logger.success(f"Shape: {df_results.shape}")
    print(df_results.to_string())