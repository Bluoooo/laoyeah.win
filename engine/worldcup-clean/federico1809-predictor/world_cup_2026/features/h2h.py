"""
h2h.py
------
Head-to-head and transitive rival analysis for international football.

Three-layer feature engineering module:
    1. Direct H2H: historical performance between two specific teams
       within a configurable time window, adjusted by Elo expectations.
    2. Transitive rival: performance against common opponents,
       used as a proxy when direct H2H matches are scarce.
    3. Squad composition decay: reduces H2H signal weight based on
       how much each team has changed since the last encounter.
       Currently implemented as temporal decay (fallback).
       Plug in Transfermarkt squad data for precise decay.

Usage:
    from world_cup_2026.features.h2h import H2HAnalyzer
    analyzer = H2HAnalyzer(df_with_elo, window_months=48)
    features = analyzer.get_matchup_features("Argentina", "France",
                                              as_of=pd.Timestamp("2026-06-01"))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_MONTHS: int = 48       # 4 years = 2 World Cup cycles
MIN_H2H_MATCHES: int = 2             # minimum H2H matches to trust the signal
MIN_COMMON_RIVALS: int = 3           # minimum common rivals for transitive signal
TEMPORAL_DECAY_LAMBDA: float = 0.02  # decay rate per month for H2H recency


# ---------------------------------------------------------------------------
# H2H Analyzer
# ---------------------------------------------------------------------------

class H2HAnalyzer:
    """Compute head-to-head and transitive rival features for any matchup.

    Args:
        df_with_elo: Output of calculate_elo() — must have elo_pre_home,
                     elo_pre_away, win_prob_home columns.
        window_months: Lookback window in months for all calculations.
        min_h2h_matches: Minimum direct H2H matches to report H2H features.
        min_common_rivals: Minimum common rivals for transitive signal.
        decay_lambda: Exponential decay rate per month for H2H recency weight.
    """

    def __init__(
        self,
        df_with_elo: pd.DataFrame,
        window_months: int = DEFAULT_WINDOW_MONTHS,
        min_h2h_matches: int = MIN_H2H_MATCHES,
        min_common_rivals: int = MIN_COMMON_RIVALS,
        decay_lambda: float = TEMPORAL_DECAY_LAMBDA,
    ) -> None:
        self.df = df_with_elo.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.window_months = window_months
        self.min_h2h_matches = min_h2h_matches
        self.min_common_rivals = min_common_rivals
        self.decay_lambda = decay_lambda
        logger.info(
            f"H2HAnalyzer initialized — {len(self.df):,} matches, "
            f"window={window_months}m, decay_lambda={decay_lambda}"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_window_start(self, as_of: pd.Timestamp) -> pd.Timestamp:
        """Return the start date of the lookback window."""
        return as_of - pd.DateOffset(months=self.window_months)

    def _get_h2h_matches(
        self,
        team_a: str,
        team_b: str,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return all matches between team_a and team_b before as_of."""
        window_start = self._get_window_start(as_of)
        mask = (
            (self.df["date"] < as_of)
            & (self.df["date"] >= window_start)
            & (
                ((self.df["home_team"] == team_a) & (self.df["away_team"] == team_b))
                | ((self.df["home_team"] == team_b) & (self.df["away_team"] == team_a))
            )
        )
        return self.df[mask].copy()

    def _get_team_matches(
        self,
        team: str,
        as_of: pd.Timestamp,
        exclude_team: str | None = None,
    ) -> pd.DataFrame:
        """Return all matches for a team before as_of within the window."""
        window_start = self._get_window_start(as_of)
        mask = (
            (self.df["date"] < as_of)
            & (self.df["date"] >= window_start)
            & (
                (self.df["home_team"] == team)
                | (self.df["away_team"] == team)
            )
        )
        df = self.df[mask].copy()
        if exclude_team:
            df = df[
                (df["home_team"] != exclude_team)
                & (df["away_team"] != exclude_team)
            ]
        return df

    def _get_result_for_team(self, row: pd.Series, team: str) -> float:
        """Return match result from team's perspective (1/0.5/0)."""
        if row["home_team"] == team:
            h, a = row["home_score"], row["away_score"]
        else:
            h, a = row["away_score"], row["home_score"]

        if h > a:
            return 1.0
        elif h < a:
            return 0.0
        return 0.5

    def _get_expected_for_team(self, row: pd.Series, team: str) -> float:
        """Return Elo-expected score from team's perspective."""
        if row["home_team"] == team:
            return float(row["win_prob_home"])
        return 1.0 - float(row["win_prob_home"])

    def _temporal_decay_weight(
        self,
        match_date: pd.Timestamp,
        as_of: pd.Timestamp,
    ) -> float:
        """Compute exponential decay weight based on months since match.

        Args:
            match_date: Date of the historical match.
            as_of: Reference date (match being predicted).

        Returns:
            Weight between 0 and 1 (1 = very recent, ~0 = very old).
        """
        months_ago = (as_of - match_date).days / 30.44
        return float(np.exp(-self.decay_lambda * months_ago))

    # -----------------------------------------------------------------------
    # Layer 1 — Direct H2H
    # -----------------------------------------------------------------------

    def get_h2h_features(
        self,
        team_a: str,
        team_b: str,
        as_of: pd.Timestamp,
        squad_change_data: dict | None = None,
    ) -> dict:
        """Compute direct H2H features between team_a and team_b.

        Args:
            team_a: First team (treated as "home" perspective).
            team_b: Second team.
            as_of: Match date — only use history before this date.
            squad_change_data: Optional dict with squad composition info.
                If None, uses temporal decay as fallback.
                Expected format:
                {
                    "team_a_change_pct": 0.35,  # 35% squad changed
                    "team_b_change_pct": 0.60,
                }

        Returns:
            Dict with H2H features for team_a vs team_b.
        """
        h2h_df = self._get_h2h_matches(team_a, team_b, as_of)

        if len(h2h_df) < self.min_h2h_matches:
            logger.debug(
                f"H2H {team_a} vs {team_b}: only {len(h2h_df)} matches "
                f"(min={self.min_h2h_matches}) — returning neutral features."
            )
            return {
                "h2h_matches": len(h2h_df),
                "h2h_win_rate_a": 0.5,
                "h2h_goal_diff_a": 0.0,
                "h2h_elo_edge_a": 0.0,
                "h2h_weighted_edge_a": 0.0,
                "h2h_decay_weight": 0.0,
                "h2h_reliable": False,
            }

        results_a, expected_a, weights = [], [], []

        for _, row in h2h_df.iterrows():
            res = self._get_result_for_team(row, team_a)
            exp = self._get_expected_for_team(row, team_a)

            # Decay weight — temporal or squad-based
            if squad_change_data:
                # Plug in Transfermarkt data here when available
                # For now: use temporal decay multiplied by inverse squad change
                base_w = self._temporal_decay_weight(row["date"], as_of)
                avg_change = (
                    squad_change_data.get("team_a_change_pct", 0.5)
                    + squad_change_data.get("team_b_change_pct", 0.5)
                ) / 2
                w = base_w * (1.0 - avg_change)
            else:
                w = self._temporal_decay_weight(row["date"], as_of)

            results_a.append(res)
            expected_a.append(exp)
            weights.append(w)

        weights_arr = np.array(weights)
        results_arr = np.array(results_a)
        expected_arr = np.array(expected_a)

        # Normalize weights
        w_sum = weights_arr.sum()
        if w_sum > 0:
            w_norm = weights_arr / w_sum
        else:
            w_norm = np.ones(len(weights_arr)) / len(weights_arr)

        # Goal difference from team_a perspective
        goal_diffs = []
        for _, row in h2h_df.iterrows():
            if row["home_team"] == team_a:
                goal_diffs.append(row["home_score"] - row["away_score"])
            else:
                goal_diffs.append(row["away_score"] - row["home_score"])

        h2h_win_rate = float(np.average(results_arr, weights=w_norm))
        h2h_expected = float(np.average(expected_arr, weights=w_norm))
        h2h_elo_edge = h2h_win_rate - h2h_expected
        h2h_goal_diff = float(np.average(goal_diffs, weights=w_norm))
        avg_decay = float(np.mean(weights_arr))

        return {
            "h2h_matches": len(h2h_df),
            "h2h_win_rate_a": h2h_win_rate,
            "h2h_goal_diff_a": h2h_goal_diff,
            "h2h_elo_edge_a": h2h_elo_edge,
            "h2h_weighted_edge_a": h2h_elo_edge * avg_decay,
            "h2h_decay_weight": avg_decay,
            "h2h_reliable": len(h2h_df) >= self.min_h2h_matches,
        }

    # -----------------------------------------------------------------------
    # Layer 2 — Transitive rival
    # -----------------------------------------------------------------------

    def get_transitive_features(
        self,
        team_a: str,
        team_b: str,
        as_of: pd.Timestamp,
    ) -> dict:
        """Compute transitive rival features via common opponents.

        For each common opponent Z that both team_a and team_b faced:
            - Calculate how much better/worse each team performed vs Z
              relative to Elo expectations
            - Aggregate across all common Z to get a transitive edge

        Args:
            team_a: First team.
            team_b: Second team.
            as_of: Reference date.

        Returns:
            Dict with transitive features.
        """
        matches_a = self._get_team_matches(team_a, as_of, exclude_team=team_b)
        matches_b = self._get_team_matches(team_b, as_of, exclude_team=team_a)

        # Get opponents faced by each team
        def get_opponents(df: pd.DataFrame, team: str) -> set:
            home_opp = set(df[df["home_team"] == team]["away_team"])
            away_opp = set(df[df["away_team"] == team]["home_team"])
            return home_opp | away_opp

        opponents_a = get_opponents(matches_a, team_a)
        opponents_b = get_opponents(matches_b, team_b)
        common_rivals = opponents_a & opponents_b

        if len(common_rivals) < self.min_common_rivals:
            logger.debug(
                f"Transitive {team_a} vs {team_b}: only {len(common_rivals)} "
                f"common rivals (min={self.min_common_rivals}) — returning neutral."
            )
            return {
                "transitive_common_rivals": len(common_rivals),
                "transitive_edge_a": 0.0,
                "transitive_goal_diff_edge": 0.0,
                "transitive_reliable": False,
            }

        edges, goal_edges, weights = [], [], []

        for rival in common_rivals:
            # Get team_a matches vs rival
            mask_a = (
                ((matches_a["home_team"] == team_a) & (matches_a["away_team"] == rival))
                | ((matches_a["home_team"] == rival) & (matches_a["away_team"] == team_a))
            )
            matches_a_vs_z = matches_a[mask_a]

            # Get team_b matches vs rival
            mask_b = (
                ((matches_b["home_team"] == team_b) & (matches_b["away_team"] == rival))
                | ((matches_b["home_team"] == rival) & (matches_b["away_team"] == team_b))
            )
            matches_b_vs_z = matches_b[mask_b]

            if matches_a_vs_z.empty or matches_b_vs_z.empty:
                continue

            # Performance of team_a vs rival (actual - expected)
            perf_a = np.mean([
                self._get_result_for_team(row, team_a) -
                self._get_expected_for_team(row, team_a)
                for _, row in matches_a_vs_z.iterrows()
            ])

            # Performance of team_b vs rival (actual - expected)
            perf_b = np.mean([
                self._get_result_for_team(row, team_b) -
                self._get_expected_for_team(row, team_b)
                for _, row in matches_b_vs_z.iterrows()
            ])

            # Goal diff edge
            def goal_diff_for_team(df, team):
                gd = []
                for _, row in df.iterrows():
                    if row["home_team"] == team:
                        gd.append(row["home_score"] - row["away_score"])
                    else:
                        gd.append(row["away_score"] - row["home_score"])
                return np.mean(gd)

            gd_a = goal_diff_for_team(matches_a_vs_z, team_a)
            gd_b = goal_diff_for_team(matches_b_vs_z, team_b)

            # Recency weight — average of last matches vs rival
            last_a = matches_a_vs_z["date"].max()
            last_b = matches_b_vs_z["date"].max()
            avg_date = last_a + (last_b - last_a) / 2
            w = self._temporal_decay_weight(avg_date, as_of)

            edges.append(perf_a - perf_b)
            goal_edges.append(gd_a - gd_b)
            weights.append(w)

        if not edges:
            return {
                "transitive_common_rivals": len(common_rivals),
                "transitive_edge_a": 0.0,
                "transitive_goal_diff_edge": 0.0,
                "transitive_reliable": False,
            }

        w_arr = np.array(weights)
        w_norm = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr

        return {
            "transitive_common_rivals": len(common_rivals),
            "transitive_edge_a": float(np.average(edges, weights=w_norm)),
            "transitive_goal_diff_edge": float(np.average(goal_edges, weights=w_norm)),
            "transitive_reliable": len(common_rivals) >= self.min_common_rivals,
        }

    # -----------------------------------------------------------------------
    # Combined features
    # -----------------------------------------------------------------------

    def get_matchup_features(
        self,
        team_a: str,
        team_b: str,
        as_of: pd.Timestamp,
        squad_change_data: dict | None = None,
    ) -> dict:
        """Get all H2H and transitive features for a matchup.

        Combines Layer 1 (H2H direct) and Layer 2 (transitive rival)
        into a single feature dict ready for model consumption.

        Args:
            team_a: First team (home perspective).
            team_b: Second team.
            as_of: Match date reference.
            squad_change_data: Optional squad composition info.

        Returns:
            Combined feature dict with all H2H and transitive features.
        """
        h2h = self.get_h2h_features(team_a, team_b, as_of, squad_change_data)
        transitive = self.get_transitive_features(team_a, team_b, as_of)

        combined = {**h2h, **transitive}
        combined["team_a"] = team_a
        combined["team_b"] = team_b
        combined["as_of"] = as_of

        return combined

    def build_feature_matrix(
        self,
        matchups: list[tuple[str, str, pd.Timestamp]],
    ) -> pd.DataFrame:
        """Build H2H feature matrix for a list of matchups.

        Args:
            matchups: List of (team_a, team_b, as_of_date) tuples.

        Returns:
            DataFrame with one row per matchup and all H2H features.
        """
        logger.info(f"Building H2H feature matrix for {len(matchups)} matchups...")
        rows = []
        for team_a, team_b, as_of in matchups:
            features = self.get_matchup_features(team_a, team_b, as_of)
            rows.append(features)
        df_out = pd.DataFrame(rows)
        logger.success(f"H2H feature matrix built: {df_out.shape}")
        return df_out