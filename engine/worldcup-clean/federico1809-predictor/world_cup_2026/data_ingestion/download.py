"""
download.py
-----------
Automated download of all public datasets required for the
FIFA World Cup 2026 prediction pipeline.

Datasets:
    1. martj42       - International Football Results 1872-2026
    2. patateriedata - All International Football Results (daily updates)
    3. lchikry       - International Football Match Features & Statistics
    4. joshfjelstul  - World Cup Database (relational)
    5. cashncarry    - FIFA World Rankings historical
    6. sarazahran1   - WC2026 Match Probability Baseline (Elo)
    7. areezvisram12 - FIFA World Cup 2026 Match Data (fixture)
    8. StatsBomb     - Open Data World Cup events (GitHub)

Usage:
    python -m world_cup_2026.data_ingestion.download
    python -m world_cup_2026.data_ingestion.download --source martj42
"""

import os
import subprocess
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

from world_cup_2026.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAGGLE_DATASETS = {
    "martj42": {
        "dataset": "martj42/international-football-results-from-1872-to-2017",
        "dest": RAW_DATA_DIR / "martj42_results",
        "description": "International football results 1872-2026 (base dataset)",
        "key_files": ["results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv"],
    },
    "patateriedata": {
        "dataset": "patateriedata/all-international-football-results",
        "dest": RAW_DATA_DIR / "patateriedata_results",
        "description": "All international results with daily updates — covers qualifiers 2026",
        "key_files": ["all_matches.csv", "countries_names.csv"],
    },
    "lchikry": {
        "dataset": "lchikry/international-football-match-features-and-statistics",
        "dest": RAW_DATA_DIR / "lchikry_features",
        "description": "Pre-calculated match features and team form up to Dec 2025",
        "key_files": ["teams_form.csv", "teams_match_features.csv", "player_aggregates.csv"],
    },
    "joshfjelstul": {
        "dataset": "joshfjelstul/world-cup-database",
        "dest": RAW_DATA_DIR / "joshfjelstul_worldcup",
        "description": "Comprehensive relational World Cup database 1930-2022",
        "key_files": ["matches.csv", "groups.csv", "goals.csv", "penalty_kicks.csv", "squads.csv"],
    },
    "cashncarry": {
        "dataset": "cashncarry/fifaworldranking",
        "dest": RAW_DATA_DIR / "cashncarry_rankings",
        "description": "FIFA World Rankings historical up to June 2024",
        "key_files": ["fifa_ranking-2024-06-20.csv"],
    },
    "sarazahran1": {
        "dataset": "sarazahran1/wc2026-match-probability-baseline-dataset",
        "dest": RAW_DATA_DIR / "sarazahran1_baseline",
        "description": "WC2026 Elo-based match probability baseline for model comparison",
        "key_files": ["future_match_probabilities_baseline.csv"],
    },
    "areezvisram12": {
        "dataset": "areezvisram12/fifa-world-cup-2026-match-data-unofficial",
        "dest": RAW_DATA_DIR / "areezvisram12_fixture",
        "description": "FIFA WC 2026 complete fixture — 104 matches, teams, stages",
        "key_files": ["matches.csv", "teams.csv", "tournament_stages.csv", "host_cities.csv"],
    },
}

STATSBOMB_FILES = {
    # competition_id=43 (FIFA World Cup), season_id=3 (2018), season_id=106 (2022)
    "wc_2018_matches": (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/3.json"
    ),
    "wc_2022_matches": (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/106.json"
    ),
    "competitions": (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_dest_populated(dest: Path) -> bool:
    """Return True if destination directory exists and has at least one file."""
    return dest.exists() and any(dest.iterdir())


def _run_kaggle_download(dataset_slug: str, dest: Path) -> None:
    """Run kaggle CLI download command via subprocess."""
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", str(dest),
        "--unzip",
    ]
    logger.debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Kaggle download failed:\n{result.stderr}")
        raise RuntimeError(f"Failed to download {dataset_slug}")

    logger.debug(result.stdout)


# ---------------------------------------------------------------------------
# Kaggle downloader
# ---------------------------------------------------------------------------

def download_kaggle_dataset(name: str) -> None:
    """Download and unzip a single Kaggle dataset.

    Args:
        name: Key from KAGGLE_DATASETS dictionary.

    Raises:
        KeyError: If name is not in KAGGLE_DATASETS.
        RuntimeError: If kaggle CLI fails.
    """
    if name not in KAGGLE_DATASETS:
        raise KeyError(f"Unknown dataset key: '{name}'. Valid keys: {list(KAGGLE_DATASETS)}")

    config = KAGGLE_DATASETS[name]
    dest: Path = config["dest"]

    if _is_dest_populated(dest):
        logger.info(f"[{name}] Already downloaded at {dest} — skipping.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{name}] Downloading: {config['description']}")
    _run_kaggle_download(config["dataset"], dest)
    logger.success(f"[{name}] Done → {dest}")


def download_all_kaggle() -> None:
    """Download all configured Kaggle datasets sequentially."""
    logger.info(f"Downloading {len(KAGGLE_DATASETS)} Kaggle datasets...")
    for name in tqdm(KAGGLE_DATASETS, desc="Kaggle datasets"):
        download_kaggle_dataset(name)
    logger.success("All Kaggle datasets downloaded.")


# ---------------------------------------------------------------------------
# StatsBomb downloader
# ---------------------------------------------------------------------------

def download_statsbomb_file(name: str, url: str) -> None:
    """Download a single JSON file from StatsBomb Open Data on GitHub.

    Args:
        name: Output filename (without .json extension).
        url: Raw GitHub URL of the file.
    """
    dest_dir = EXTERNAL_DATA_DIR / "statsbomb"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{name}.json"

    if dest_file.exists():
        logger.info(f"[statsbomb/{name}] Already exists — skipping.")
        return

    logger.info(f"[statsbomb/{name}] Downloading...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest_file.write_bytes(response.content)
    logger.success(f"[statsbomb/{name}] Saved → {dest_file}")


def download_all_statsbomb() -> None:
    """Download all configured StatsBomb JSON files."""
    logger.info(f"Downloading {len(STATSBOMB_FILES)} StatsBomb files...")
    for name, url in tqdm(STATSBOMB_FILES.items(), desc="StatsBomb files"):
        download_statsbomb_file(name, url)
    logger.success("All StatsBomb files downloaded.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_download_summary() -> None:
    """Print a summary of what was downloaded and where."""
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)

    for name, config in KAGGLE_DATASETS.items():
        dest = config["dest"]
        if dest.exists():
            files = list(dest.iterdir())
            status = f"✓ {len(files)} files" if files else "✗ empty"
        else:
            status = "✗ not found"
        logger.info(f"  [{name}] {status} → {dest}")

    statsbomb_dir = EXTERNAL_DATA_DIR / "statsbomb"
    for name in STATSBOMB_FILES:
        f = statsbomb_dir / f"{name}.json"
        status = "✓" if f.exists() else "✗ not found"
        logger.info(f"  [statsbomb/{name}] {status}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download_all() -> None:
    """Run full download pipeline for all data sources."""
    logger.info("=" * 60)
    logger.info("FIFA World Cup 2026 — Data Acquisition Pipeline")
    logger.info("=" * 60)

    download_all_kaggle()
    download_all_statsbomb()
    print_download_summary()

    logger.success("Pipeline complete.")
    logger.info(f"Raw data   → {RAW_DATA_DIR}")
    logger.info(f"External   → {EXTERNAL_DATA_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--source":
        # Selective download: python -m ... --source martj42
        source = sys.argv[2]
        download_kaggle_dataset(source)
    else:
        download_all()