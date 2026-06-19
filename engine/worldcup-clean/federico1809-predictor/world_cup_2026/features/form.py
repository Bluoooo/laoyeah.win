"""
form.py
-------
Recent form features for international football teams.
Computes rolling performance metrics for each team over configurable
match windows (default: 5, 10, 20 matches), strictly before each
match date to avoid data leakage.

Features computed per window:
    - win/draw/loss rates
    - goals scored/conceded averages
    - goal difference average
    - clean sheet rate
    - failed to score rate
    - form points (3/1/0 weighted average)
    - weighted form (exponential decay — most recent matches weighted more)

Performance: O(n log n) via groupby + rolling on team-perspective DataFrame.
Prior implementation was O(n²) due to per-row list comprehension over
growing history slices.

Usage:
    from world_cup_2026.features.form import FormCalculator
    calc = FormCalculator(df_results_norm, windows=[5, 10, 20])
    df_with_form = calc.compute_form_features(df_results_norm)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WINDOWS: list[int] = [5, 10, 20]
FORM_DECAY_LAMBDA: float = 0.1  # decay per match (not per month)

OFFICIAL_TOURNAMENTS: set[str] = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "Africa Cup of Nations",
    "Africa Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "CONCACAF Gold Cup",
    "UEFA Nations League",
    "CONCACAF Nations League",
    "FIFA Confederations Cup",
}

# Precompute decay weight arrays per window size at module load time.
# weights[w][i] = exp(-lambda * (w-1-i)) for i in 0..w-1, normalized.
# This avoids recomputing inside every rolling.apply() call.
_DECAY_WEIGHTS: dict[int, np.ndarray] = {}


def _get_decay_weights(window: int, decay_lambda: float) -> np.ndarray:
    """Return normalized exponential decay weights for a given window."""
    key = (window, decay_lambda)
    if key not in _DECAY_WEIGHTS:
        w = np.exp(-decay_lambda * np.arange(window - 1, -1, -1))
        _DECAY_WEIGHTS[key] = w / w.sum()
    return _DECAY_WEIGHTS[key]


# ---------------------------------------------------------------------------
# FormCalculator
# ---------------------------------------------------------------------------
class FormCalculator:
    """Compute recent form features for all teams over all matches.

    Args:
        df: Normalized match results DataFrame (output of normalize pipeline).
        windows: List of match window sizes to compute form over.
        decay_lambda: Exponential decay rate per match for weighted form.
        official_only_windows: If True, also compute form on official
                               matches only (excludes friendlies).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        windows: list[int] = DEFAULT_WINDOWS,
        decay_lambda: float = FORM_DECAY_LAMBDA,
        official_only_windows: bool = True,
    ) -> None:
        self.df = df.copy().sort_values("date").reset_index(drop=True)
        self.windows = windows
        self.decay_lambda = decay_lambda
        self.official_only = official_only_windows
        logger.info(
            f"FormCalculator initialized — {len(self.df):,} matches, "
            f"windows={windows}, decay_lambda={decay_lambda}"
        )

    # -----------------------------------------------------------------------
    # Internal: build team-perspective (long format) DataFrame
    # -----------------------------------------------------------------------
    def _build_long_df(self, source_df: pd.DataFrame) -> pd.DataFrame:
        """Convert match DataFrame to team-perspective long format.

        Each match produces two rows: one for the home team, one for away.
        The resulting DataFrame is sorted by (team, date) and contains a
        sequential match index per team — which is what rolling() needs.

        Uses the original match index + role ('home'/'away') as merge key,
        which handles teams playing two matches on the same date (common
        in early 20th century South American tournaments).

        Args:
            source_df: Normalized match DataFrame sorted by date.

        Returns:
            Long-format DataFrame with columns:
                _match_idx, _role, date, team, opponent,
                goals_for, goals_against, goal_diff,
                win, draw, loss, points,
                clean_sheet, failed_to_score, is_official
        """
        source = source_df.reset_index(drop=True)
        source["_match_idx"] = source.index

        home = source[["_match_idx", "date", "home_team", "away_team",
                    "home_score", "away_score", "tournament"]].copy()
        home.columns = ["_match_idx", "date", "team", "opponent",
                        "goals_for", "goals_against", "tournament"]
        home["_role"] = "home"

        away = source[["_match_idx", "date", "away_team", "home_team",
                    "away_score", "home_score", "tournament"]].copy()
        away.columns = ["_match_idx", "date", "team", "opponent",
                        "goals_for", "goals_against", "tournament"]
        away["_role"] = "away"

        long = pd.concat([home, away], ignore_index=True)
        long["goals_for"] = long["goals_for"].astype(int)
        long["goals_against"] = long["goals_against"].astype(int)
        long["goal_diff"] = long["goals_for"] - long["goals_against"]

        long["win"]   = (long["goal_diff"] > 0).astype(int)
        long["draw"]  = (long["goal_diff"] == 0).astype(int)
        long["loss"]  = (long["goal_diff"] < 0).astype(int)
        long["points"] = long["win"] * 3 + long["draw"] * 1
        long["clean_sheet"]     = (long["goals_against"] == 0).astype(int)
        long["failed_to_score"] = (long["goals_for"] == 0).astype(int)
        long["is_official"] = long["tournament"].isin(OFFICIAL_TOURNAMENTS).astype(int)

        long = long.sort_values(["team", "date"]).reset_index(drop=True)
        long.drop(columns=["tournament"], inplace=True)

        logger.debug(f"Long format built: {len(long):,} rows, {long['team'].nunique()} teams.")
        return long

    # -----------------------------------------------------------------------
    # Internal: compute rolling features on long DataFrame
    # -----------------------------------------------------------------------
    def _compute_rolling_features(self, long: pd.DataFrame) -> pd.DataFrame:
        """Compute all rolling window features on the long-format DataFrame.

        Uses .shift(1) before rolling to ensure strictly pre-match history
        (no leakage — the current match is never included in its own form).

        Args:
            long: Team-perspective DataFrame from _build_long_df().

        Returns:
            long with one column per (window, metric) combination added.
        """
        feature_cols = [
            "win", "draw", "loss",
            "goals_for", "goals_against", "goal_diff",
            "clean_sheet", "failed_to_score", "points",
        ]

        grouped = long.groupby("team", sort=False)

        for w in self.windows:
            logger.debug(f"  Computing rolling window={w}...")

            # Shift(1) within each group — current match excluded from its own form
            shifted = grouped[feature_cols].shift(1)

            # Simple rolling means — native C, very fast
            rolled = shifted.groupby(long["team"]).rolling(
                window=w, min_periods=1
            ).mean().reset_index(level=0, drop=True)

            long[f"form_{w}_win_rate"]           = rolled["win"]
            long[f"form_{w}_draw_rate"]          = rolled["draw"]
            long[f"form_{w}_loss_rate"]          = rolled["loss"]
            long[f"form_{w}_goals_scored_avg"]   = rolled["goals_for"]
            long[f"form_{w}_goals_conceded_avg"] = rolled["goals_against"]
            long[f"form_{w}_goal_diff_avg"]      = rolled["goal_diff"]
            long[f"form_{w}_clean_sheet_rate"]   = rolled["clean_sheet"]
            long[f"form_{w}_failed_score_rate"]  = rolled["failed_to_score"]
            long[f"form_{w}_points_avg"]         = rolled["points"]

            # Matches played in this window (capped at w)
            counts = shifted["win"].groupby(long["team"]).rolling(
                window=w, min_periods=1
            ).count().reset_index(level=0, drop=True)
            long[f"form_{w}_matches_played"] = counts.astype(int)

            # Weighted points — exponential decay, rolling.apply()
            decay_weights = _get_decay_weights(w, self.decay_lambda)

            def _weighted_points(arr: np.ndarray) -> float:
                n = len(arr)
                if n == 0:
                    return 1.0
                if n < len(decay_weights):
                    # Fewer matches than window: use last n weights, renormalize
                    w_slice = decay_weights[-n:]
                    w_norm = w_slice / w_slice.sum()
                else:
                    w_norm = decay_weights
                return float(np.dot(arr, w_norm))

            shifted_pts = shifted["points"]
            long[f"form_{w}_weighted_points"] = (
                shifted_pts
                .groupby(long["team"])
                .rolling(window=w, min_periods=1)
                .apply(_weighted_points, raw=True)
                .reset_index(level=0, drop=True)
            )

        return long

    # -----------------------------------------------------------------------
    # Internal: fill neutral stats for teams with no prior history
    # -----------------------------------------------------------------------
    def _fill_neutral(self, df_out: pd.DataFrame) -> pd.DataFrame:
        """Replace NaN form values with neutral priors.

        Called after the pivot/merge step. NaN appears when a team has
        zero prior matches in the source (e.g. very first international).

        Neutral priors:
            win/draw/loss rate: 0.333 each
            goals scored/conceded avg: 1.0
            goal diff avg: 0.0
            clean sheet rate: 0.2
            failed score rate: 0.2
            points avg: 1.0
            weighted points: 1.0
            matches played: 0
        """
        neutral = {
            "win_rate": 0.333,
            "draw_rate": 0.333,
            "loss_rate": 0.333,
            "goals_scored_avg": 1.0,
            "goals_conceded_avg": 1.0,
            "goal_diff_avg": 0.0,
            "clean_sheet_rate": 0.2,
            "failed_score_rate": 0.2,
            "points_avg": 1.0,
            "weighted_points": 1.0,
            "matches_played": 0,
        }
        for side in ("home", "away"):
            for w in self.windows:
                for metric, val in neutral.items():
                    col = f"{side}_form_{w}_{metric}"
                    if col in df_out.columns:
                        df_out[col] = df_out[col].fillna(val)
        return df_out

    # -----------------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------------
    def compute_form_features(
        self,
        df: pd.DataFrame,
        official_only: bool = False,
    ) -> pd.DataFrame:
        """Add form features to a match DataFrame.

        For each match row, computes form features for both home and
        away teams based on their history strictly before that match
        (no data leakage).

        Performance: O(n log n) via groupby + rolling.
        Prior implementation was O(n²).

        Args:
            df: Normalized match DataFrame sorted by date.
            official_only: If True, compute form only from official matches.

        Returns:
            Copy of df with form feature columns added for both teams.
            Column naming: home_form_{w}_{metric} / away_form_{w}_{metric}
        """
        suffix = "_off" if official_only else ""
        df_out = df.copy().sort_values("date").reset_index(drop=True)
        df_out["_match_idx"] = df_out.index

        source_df = self.df.copy()
        if official_only:
            source_df = source_df[
                source_df["tournament"].isin(OFFICIAL_TOURNAMENTS)
            ].copy()

        logger.info(
            f"Computing form features ({'official only' if official_only else 'all matches'}) "
            f"for {len(df_out):,} rows..."
        )

        # Step 1: build long format (includes _match_idx and _role)
        long = self._build_long_df(source_df)

        # Step 2: compute rolling features on long format
        long = self._compute_rolling_features(long)

        # Step 3: collect form column names
        form_cols = [c for c in long.columns if c.startswith("form_")]

        # Step 4: build lookup index on (_match_idx, _role)
        # This is guaranteed unique — one home row and one away row per match
        long_indexed = long.set_index(["_match_idx", "_role"])

        # Step 5: extract home and away form using match index
        home_form = (
            long_indexed.loc[
                list(zip(df_out["_match_idx"], ["home"] * len(df_out))),
                form_cols,
            ]
            .reset_index(drop=True)
        )
        home_form.columns = [f"home_{c}{suffix}" for c in form_cols]

        away_form = (
            long_indexed.loc[
                list(zip(df_out["_match_idx"], ["away"] * len(df_out))),
                form_cols,
            ]
            .reset_index(drop=True)
        )
        away_form.columns = [f"away_{c}{suffix}" for c in form_cols]

        # Step 6: concatenate
        df_out = pd.concat(
            [df_out.drop(columns=["_match_idx"]),
            home_form,
            away_form],
            axis=1,
        )

        # Step 7: fill NaN with neutral priors
        df_out = self._fill_neutral(df_out)

        n_cols = len(home_form.columns) + len(away_form.columns)
        logger.success(
            f"Form features computed. Added {n_cols} columns."
        )
        return df_out

    # -----------------------------------------------------------------------
    # Public utility: single team snapshot
    # -----------------------------------------------------------------------
    def get_team_current_form(
        self,
        team: str,
        as_of: pd.Timestamp,
        window: int = 10,
        official_only: bool = False,
    ) -> dict:
        """Get current form snapshot for a single team.

        Args:
            team: Team name (normalized).
            as_of: Reference date — only matches before this date.
            window: Number of recent matches.
            official_only: Use only official matches.

        Returns:
            Dict of form stats for this team at the given date.
        """
        source = self.df.copy()
        if official_only:
            source = source[source["tournament"].isin(OFFICIAL_TOURNAMENTS)]

        # Filter to this team's matches before as_of
        mask = (
            ((source["home_team"] == team) | (source["away_team"] == team))
            & (source["date"] < as_of)
        )
        team_matches = source[mask].sort_values("date").tail(window)

        if team_matches.empty:
            return self._neutral_snapshot(window)

        records = []
        for _, row in team_matches.iterrows():
            if row["home_team"] == team:
                gf, ga = int(row["home_score"]), int(row["away_score"])
            else:
                gf, ga = int(row["away_score"]), int(row["home_score"])
            gd = gf - ga
            records.append({
                "goals_for": gf,
                "goals_against": ga,
                "goal_diff": gd,
                "win": int(gd > 0),
                "draw": int(gd == 0),
                "loss": int(gd < 0),
                "points": 3 if gd > 0 else (1 if gd == 0 else 0),
                "clean_sheet": int(ga == 0),
                "failed_to_score": int(gf == 0),
            })

        n = len(records)
        arr = {k: np.array([r[k] for r in records]) for k in records[0]}
        weights = _get_decay_weights(n, self.decay_lambda)

        return {
            f"form_{window}_matches_played":     n,
            f"form_{window}_win_rate":           float(arr["win"].mean()),
            f"form_{window}_draw_rate":          float(arr["draw"].mean()),
            f"form_{window}_loss_rate":          float(arr["loss"].mean()),
            f"form_{window}_goals_scored_avg":   float(arr["goals_for"].mean()),
            f"form_{window}_goals_conceded_avg": float(arr["goals_against"].mean()),
            f"form_{window}_goal_diff_avg":      float(arr["goal_diff"].mean()),
            f"form_{window}_clean_sheet_rate":   float(arr["clean_sheet"].mean()),
            f"form_{window}_failed_score_rate":  float(arr["failed_to_score"].mean()),
            f"form_{window}_points_avg":         float(arr["points"].mean()),
            f"form_{window}_weighted_points":    float(np.dot(arr["points"], weights)),
        }

    def _neutral_snapshot(self, window: int) -> dict:
        """Neutral priors for a team with no history."""
        return {
            f"form_{window}_matches_played":     0,
            f"form_{window}_win_rate":           0.333,
            f"form_{window}_draw_rate":          0.333,
            f"form_{window}_loss_rate":          0.333,
            f"form_{window}_goals_scored_avg":   1.0,
            f"form_{window}_goals_conceded_avg": 1.0,
            f"form_{window}_goal_diff_avg":      0.0,
            f"form_{window}_clean_sheet_rate":   0.2,
            f"form_{window}_failed_score_rate":  0.2,
            f"form_{window}_points_avg":         1.0,
            f"form_{window}_weighted_points":    1.0,
        }