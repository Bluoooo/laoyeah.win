"""
elo.py
------
Elo rating system for international football teams.

Calculates Elo ratings from scratch over the full historical match record,
producing pre-match Elo values for every game (no data leakage).

Key design decisions:
    - Initial rating: 1500 for all teams on debut
    - K-factor: dynamic, based on tournament importance and goal margin
    - Output: original DataFrame with elo_home, elo_away, elo_diff columns added
    - Elo stored is PRE-match (computed before updating) to avoid leakage

Usage:
    from world_cup_2026.features.elo import calculate_elo
    df_with_elo = calculate_elo(df_results)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from world_cup_2026.config import RANDOM_SEED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_ELO: float = 1500.0
ELO_SCALE: float = 400.0      # controls sensitivity of probability curve

# Tournament importance — K-factor base values
TOURNAMENT_K: dict[str, float] = {
    "FIFA World Cup":                        60.0,
    "FIFA World Cup qualification":          40.0,
    "UEFA Euro":                             45.0,
    "UEFA Euro qualification":               35.0,
    "Copa América":                          45.0,
    "Africa Cup of Nations":                 45.0,
    "Africa Cup of Nations qualification":   35.0,
    "AFC Asian Cup":                         45.0,
    "AFC Asian Cup qualification":           35.0,
    "CONCACAF Gold Cup":                     45.0,
    "UEFA Nations League":                   30.0,
    "CONCACAF Nations League":               30.0,
    "Friendly":                              20.0,
}
DEFAULT_K: float = 30.0   # fallback for tournaments not in the dict

# Goal margin multipliers (caps at 2.5)
def _goal_margin_multiplier(goal_diff: int) -> float:
    """Return K multiplier based on absolute goal difference.

    Args:
        goal_diff: Absolute difference in goals scored.

    Returns:
        Multiplier between 1.0 and 2.5.
    """
    abs_diff = abs(goal_diff)
    if abs_diff <= 1:
        return 1.0
    elif abs_diff == 2:
        return 1.5
    elif abs_diff == 3:
        return 1.75
    else:
        return min(1.75 + (abs_diff - 3) * 0.05, 2.5)


# ---------------------------------------------------------------------------
# Core Elo functions
# ---------------------------------------------------------------------------

def expected_score(elo_a: float, elo_b: float) -> float:
    """Calculate expected score for team A against team B.

    Args:
        elo_a: Elo rating of team A.
        elo_b: Elo rating of team B.

    Returns:
        Expected score for team A (between 0 and 1).
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_SCALE))


def update_elo(
    elo_a: float,
    elo_b: float,
    score_a: float,
    k_factor: float,
    goal_diff: int,
) -> tuple[float, float]:
    """Update Elo ratings for both teams after a match.

    Args:
        elo_a: Pre-match Elo of team A (home).
        elo_b: Pre-match Elo of team B (away).
        score_a: Actual result for team A (1=win, 0.5=draw, 0=loss).
        k_factor: Base K-factor for this match.
        goal_diff: home_score - away_score (signed).

    Returns:
        Tuple of (new_elo_a, new_elo_b).
    """
    e_a = expected_score(elo_a, elo_b)
    e_b = 1.0 - e_a
    score_b = 1.0 - score_a

    multiplier = _goal_margin_multiplier(goal_diff)
    k = k_factor * multiplier

    new_elo_a = elo_a + k * (score_a - e_a)
    new_elo_b = elo_b + k * (score_b - e_b)

    return new_elo_a, new_elo_b


def get_k_factor(tournament: str) -> float:
    """Return K-factor for a given tournament name.

    Args:
        tournament: Tournament name as it appears in df_results.

    Returns:
        K-factor float value.
    """
    # Exact match first
    if tournament in TOURNAMENT_K:
        return TOURNAMENT_K[tournament]

    # Partial match — handles variants like "UEFA Euro 2024 qualification"
    tournament_lower = tournament.lower()
    for key, k in TOURNAMENT_K.items():
        if key.lower() in tournament_lower:
            return k

    return DEFAULT_K


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def calculate_elo(
    df: pd.DataFrame,
    initial_elo: float = INITIAL_ELO,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    tournament_col: str = "tournament",
) -> pd.DataFrame:
    """Calculate pre-match Elo ratings for every match in the dataset.

    Iterates over all matches chronologically, maintaining a live Elo
    registry for each team. Stores PRE-match Elo to avoid data leakage.

    Args:
        df: DataFrame with match results, sorted or unsorted by date.
        initial_elo: Starting Elo for teams on their debut.
        date_col: Column name for match date.
        home_col: Column name for home team.
        away_col: Column name for away team.
        home_score_col: Column name for home goals scored.
        away_score_col: Column name for away goals scored.
        tournament_col: Column name for tournament name.

    Returns:
        Copy of df with added columns:
            - elo_pre_home   : home team Elo before this match
            - elo_pre_away   : away team Elo before this match
            - elo_diff       : elo_pre_home - elo_pre_away
            - elo_post_home  : home team Elo after this match
            - elo_post_away  : away team Elo after this match
            - win_prob_home  : expected win probability for home team
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)

    # Live Elo registry — updated after every match
    elo_registry: dict[str, float] = {}

    # Output arrays — pre-allocated for performance
    n = len(df)
    elo_pre_home  = np.zeros(n)
    elo_pre_away  = np.zeros(n)
    elo_post_home = np.zeros(n)
    elo_post_away = np.zeros(n)
    win_prob_home = np.zeros(n)

    logger.info(f"Calculating Elo for {n:,} matches...")

    for i, row in enumerate(df.itertuples(index=False)):
        home = getattr(row, home_col)
        away = getattr(row, away_col)
        h_score = getattr(row, home_score_col)
        a_score = getattr(row, away_score_col)
        tournament = getattr(row, tournament_col)

        # Get current Elo (debut = initial_elo)
        e_home = elo_registry.get(home, initial_elo)
        e_away = elo_registry.get(away, initial_elo)

        # Store PRE-match Elo
        elo_pre_home[i] = e_home
        elo_pre_away[i] = e_away
        win_prob_home[i] = expected_score(e_home, e_away)

        # Determine result from home team perspective
        if h_score > a_score:
            score_home = 1.0
        elif h_score < a_score:
            score_home = 0.0
        else:
            score_home = 0.5

        # Update Elo
        k = get_k_factor(tournament)
        goal_diff = int(h_score) - int(a_score)
        new_home, new_away = update_elo(e_home, e_away, score_home, k, goal_diff)

        # Store POST-match Elo
        elo_post_home[i] = new_home
        elo_post_away[i] = new_away

        # Update registry
        elo_registry[home] = new_home
        elo_registry[away] = new_away

    # Attach to DataFrame
    df["elo_pre_home"]  = elo_pre_home
    df["elo_pre_away"]  = elo_pre_away
    df["elo_diff"]      = elo_pre_home - elo_pre_away
    df["elo_post_home"] = elo_post_home
    df["elo_post_away"] = elo_post_away
    df["win_prob_home"] = win_prob_home

    logger.success(f"Elo calculation complete. Registry has {len(elo_registry)} teams.")
    return df


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_current_elo(df_with_elo: pd.DataFrame, team_col: str = "home_team") -> pd.Series:
    """Extract the most recent Elo rating for each team.

    Args:
        df_with_elo: Output of calculate_elo().
        team_col: Which team column to use as reference.

    Returns:
        Series indexed by team name with latest post-match Elo.
    """
    return (
        df_with_elo.sort_values("date")
        .groupby(team_col)["elo_post_home"]
        .last()
        .rename("current_elo")
    )


def get_elo_at_date(
    df_with_elo: pd.DataFrame,
    team: str,
    date: pd.Timestamp,
) -> float:
    """Get a team's Elo rating as of a specific date.

    Args:
        df_with_elo: Output of calculate_elo().
        team: Team name.
        date: Target date.

    Returns:
        Most recent post-match Elo before or on the given date.
        Returns INITIAL_ELO if team has no matches before that date.
    """
    mask = (
        ((df_with_elo["home_team"] == team) | (df_with_elo["away_team"] == team))
        & (df_with_elo["date"] <= date)
    )
    subset = df_with_elo[mask]

    if subset.empty:
        return INITIAL_ELO

    last = subset.sort_values("date").iloc[-1]
    if last["home_team"] == team:
        return float(last["elo_post_home"])
    else:
        return float(last["elo_post_away"])