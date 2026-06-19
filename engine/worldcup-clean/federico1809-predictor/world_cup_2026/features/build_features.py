"""
build_features.py
-----------------
Standalone pipeline that replicates notebooks/02_features/02_feature_matrix.ipynb
and produces data/processed/master_features.parquet.

Usage:
    python -m world_cup_2026.features.build_features
"""

from __future__ import annotations

import time
from bisect import bisect_left

import numpy as np
import pandas as pd
from loguru import logger

from world_cup_2026.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_SEED
from world_cup_2026.data_ingestion.normalize import normalize_dataframe_teams
from world_cup_2026.features.elo import calculate_elo
from world_cup_2026.features.form import FormCalculator
from world_cup_2026.features.h2h import H2HAnalyzer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTOFF_DATE = pd.Timestamp("2016-01-01")

RANKING_NAME_MAP: dict[str, str] = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Congo DR":               "DR Congo",
    "IR Iran":                "Iran",
    "Korea Republic":         "South Korea",
}

KANCHANA_NAME_MAP: dict[str, str] = {
    "Czech Republic": "Czechia",
    "Cote d'Ivoire":  "Côte d'Ivoire",
    "Korea, South":   "South Korea",
    "Türkiye":        "Turkey",
    "United States":  "USA",
}

TOURNAMENT_TIERS: dict[str, int] = {
    "FIFA World Cup":                       3,
    "UEFA Euro":                            3,
    "Copa América":                         3,
    "African Cup of Nations":               3,
    "AFC Asian Cup":                        3,
    "Gold Cup":                             2,
    "UEFA Nations League":                  2,
    "CONCACAF Nations League":              2,
    "FIFA World Cup qualification":         1,
    "UEFA Euro qualification":              1,
    "African Cup of Nations qualification": 1,
    "AFC Asian Cup qualification":          1,
    "Friendly":                             0,
}

ELO_COLS = [
    "date", "home_team", "away_team", "tournament", "neutral",
    "elo_pre_home", "elo_pre_away", "elo_diff", "win_prob_home",
]

H2H_META_COLS = ["team_a", "team_b", "as_of"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_result(row: pd.Series) -> int:
    if row["home_score"] > row["away_score"]:
        return 2
    if row["home_score"] == row["away_score"]:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Step 1 — Base features: Elo + Form + H2H
# ---------------------------------------------------------------------------

def step1_base_features() -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)

    logger.info("Step 1 — Loading match results...")
    df_results = pd.read_csv(
        RAW_DATA_DIR / "martj42_results" / "results.csv",
        parse_dates=["date"],
    )
    df_results_norm = normalize_dataframe_teams(df_results, ["home_team", "away_team"])
    logger.info(f"Matches loaded: {len(df_results_norm):,}  "
                f"({df_results_norm['date'].min().date()} → {df_results_norm['date'].max().date()})")

    t0 = time.time()
    df_elo = calculate_elo(df_results_norm)
    logger.info(f"Elo done in {time.time() - t0:.1f}s")

    t0 = time.time()
    calc = FormCalculator(df_results_norm, windows=[5, 10, 20])
    df_form = calc.compute_form_features(df_results_norm)
    logger.info(f"Form done in {time.time() - t0:.1f}s — {df_form.shape[1]} cols")

    analyzer = H2HAnalyzer(df_elo)

    df_results_norm["target"] = df_results_norm.apply(_encode_result, axis=1)

    form_cols = [c for c in df_form.columns if c.startswith(("home_form_", "away_form_"))]

    df_master = df_elo[ELO_COLS].copy()
    df_master["target"] = df_results_norm["target"].values
    df_master = pd.concat(
        [df_master, df_form[form_cols].reset_index(drop=True)],
        axis=1,
    )

    df_master = df_master[df_master["date"] >= CUTOFF_DATE].reset_index(drop=True)
    logger.info(f"After {CUTOFF_DATE.date()} filter: {len(df_master):,} matches")

    logger.info("Computing H2H features (row-by-row, this takes a few minutes)...")
    t0 = time.time()
    h2h_records = []
    n = len(df_master)
    for idx, row in df_master.iterrows():
        h2h_records.append(
            analyzer.get_matchup_features(
                team_a=row["home_team"],
                team_b=row["away_team"],
                as_of=row["date"],
            )
        )
        if idx > 0 and idx % 5_000 == 0:
            elapsed = time.time() - t0
            logger.info(f"  H2H {idx:,}/{n:,} ({idx/n*100:.0f}%) — {elapsed:.1f}s elapsed")

    df_h2h = pd.DataFrame(h2h_records)
    df_master = pd.concat([df_master, df_h2h.reset_index(drop=True)], axis=1)
    df_master = df_master.drop(columns=H2H_META_COLS, errors="ignore")
    logger.info(f"H2H done in {time.time() - t0:.1f}s")

    out = PROCESSED_DATA_DIR / "master_features.parquet"
    df_master.to_parquet(out, index=False)
    logger.success(f"Step 1 saved → {out}  shape={df_master.shape}")
    return df_master


# ---------------------------------------------------------------------------
# Step 2 — FIFA rankings (as-of join, bisect O(log n))
# ---------------------------------------------------------------------------

def step2_rankings(df_master: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 2 — Adding FIFA rankings...")

    df_rank = pd.read_csv(
        RAW_DATA_DIR / "cashncarry_rankings" / "fifa_ranking-2024-06-20.csv"
    )
    df_rank["rank_date"] = pd.to_datetime(df_rank["rank_date"])
    df_rank["team"] = df_rank["country_full"].replace(RANKING_NAME_MAP)

    df_rank_clean = (
        df_rank[["rank_date", "team", "rank", "confederation"]]
        .dropna(subset=["rank"])
        .sort_values(["team", "rank_date"])
        .reset_index(drop=True)
    )

    # Build per-team sorted arrays for O(log n) strict-< lookups
    team_dates: dict[str, list] = {}
    team_ranks: dict[str, list] = {}
    team_confs: dict[str, list] = {}
    for team, grp in df_rank_clean.groupby("team", sort=False):
        team_dates[team] = grp["rank_date"].tolist()
        team_ranks[team] = grp["rank"].tolist()
        team_confs[team]  = grp["confederation"].tolist()

    def _lookup(team: str, date: pd.Timestamp) -> tuple:
        dates = team_dates.get(team)
        if dates is None:
            return None, None
        idx = bisect_left(dates, date) - 1  # strict <: exclude exact match
        if idx < 0:
            return None, None
        return team_ranks[team][idx], team_confs[team][idx]

    df_master = df_master.copy()
    df_master["date"] = pd.to_datetime(df_master["date"])

    home_results = [_lookup(t, d) for t, d in zip(df_master["home_team"], df_master["date"])]
    away_results = [_lookup(t, d) for t, d in zip(df_master["away_team"], df_master["date"])]

    df_master["ranking_home"]       = [r[0] for r in home_results]
    df_master["ranking_away"]       = [r[0] for r in away_results]
    df_master["confederation_home"] = [r[1] for r in home_results]
    df_master["confederation_away"] = [r[1] for r in away_results]

    median_rank = df_master["ranking_home"].median()
    logger.info(f"Median ranking (fill value for unknowns): {median_rank:.0f}")
    df_master["ranking_home"] = df_master["ranking_home"].fillna(median_rank)
    df_master["ranking_away"] = df_master["ranking_away"].fillna(median_rank)
    df_master["ranking_diff"] = df_master["ranking_home"] - df_master["ranking_away"]
    df_master["confederation_home"] = df_master["confederation_home"].fillna("Unknown")
    df_master["confederation_away"] = df_master["confederation_away"].fillna("Unknown")

    logger.info(f"Missing ranking_home after fill: {df_master['ranking_home'].isna().sum()}")
    logger.info(f"Missing ranking_away after fill: {df_master['ranking_away'].isna().sum()}")

    out = PROCESSED_DATA_DIR / "master_features.parquet"
    df_master.to_parquet(out, index=False)
    logger.success(f"Step 2 saved → {out}  shape={df_master.shape}")
    return df_master


# ---------------------------------------------------------------------------
# Step 3 — Squad value (Transfermarkt / kanchana1990)
# ---------------------------------------------------------------------------

def step3_squad_value(df_master: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 3 — Adding squad values...")

    kanchana_path = (
        RAW_DATA_DIR
        / "kanchana1990_transfermarkt"
        / "Football_Player_Market_Value_Trajectories"
        / "transfermarkt_player_values.csv"
    )
    df_k = pd.read_csv(kanchana_path)
    df_k["nationality_norm"] = df_k["nationality"].replace(KANCHANA_NAME_MAP)

    df_squad_value = (
        df_k.groupby("nationality_norm")["current_value_eur"]
        .sum()
        .reset_index()
        .rename(columns={"nationality_norm": "team", "current_value_eur": "squad_value_eur"})
    )

    full_lookup = dict(zip(df_squad_value["team"], df_squad_value["squad_value_eur"]))
    overall_median = df_squad_value["squad_value_eur"].median()
    logger.info(f"Overall median squad value: €{overall_median / 1e6:.1f}M")

    df_master = df_master.copy()
    df_master["squad_value_home"] = df_master["home_team"].map(full_lookup).fillna(overall_median)
    df_master["squad_value_away"] = df_master["away_team"].map(full_lookup).fillna(overall_median)
    df_master["squad_value_diff"] = df_master["squad_value_home"] - df_master["squad_value_away"]

    logger.info(f"Nulls squad_value_home after fill: {df_master['squad_value_home'].isna().sum()}")
    logger.info(f"Nulls squad_value_away after fill: {df_master['squad_value_away'].isna().sum()}")

    out = PROCESSED_DATA_DIR / "master_features.parquet"
    df_master.to_parquet(out, index=False)
    logger.success(f"Step 3 saved → {out}  shape={df_master.shape}")
    return df_master


# ---------------------------------------------------------------------------
# Step 4 — Rest days between matches
# ---------------------------------------------------------------------------

def step4_rest_days(df_master: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 4 — Adding rest days...")

    df_master = df_master.copy()
    df_master["date"] = pd.to_datetime(df_master["date"])

    df_home = df_master[["date", "home_team"]].rename(columns={"home_team": "team"})
    df_away = df_master[["date", "away_team"]].rename(columns={"away_team": "team"})
    df_all = (
        pd.concat([df_home, df_away])
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )

    df_all["prev_match_date"] = df_all.groupby("team")["date"].shift(1)
    df_all["rest_days"] = (df_all["date"] - df_all["prev_match_date"]).dt.days

    rest_lookup: dict[tuple, float] = {
        (row["team"], row["date"]): row["rest_days"]
        for _, row in df_all.iterrows()
    }

    df_master["rest_days_home"] = df_master.apply(
        lambda r: rest_lookup.get((r["home_team"], r["date"]), None), axis=1
    )
    df_master["rest_days_away"] = df_master.apply(
        lambda r: rest_lookup.get((r["away_team"], r["date"]), None), axis=1
    )

    median_rest = df_master["rest_days_home"].median()
    logger.info(f"Median rest days: {median_rest:.0f}")
    logger.info(f"NaN rest_days_home before fill: {df_master['rest_days_home'].isna().sum()}")
    logger.info(f"NaN rest_days_away before fill: {df_master['rest_days_away'].isna().sum()}")

    df_master["rest_days_home"] = df_master["rest_days_home"].fillna(median_rest)
    df_master["rest_days_away"] = df_master["rest_days_away"].fillna(median_rest)

    out = PROCESSED_DATA_DIR / "master_features.parquet"
    df_master.to_parquet(out, index=False)
    logger.success(f"Step 4 saved → {out}  shape={df_master.shape}")
    return df_master


# ---------------------------------------------------------------------------
# Step 5 — Match importance tier
# ---------------------------------------------------------------------------

def step5_match_importance(df_master: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step 5 — Adding match importance tiers...")

    df_master = df_master.copy()
    df_master["match_importance"] = df_master["tournament"].map(TOURNAMENT_TIERS).fillna(1)

    logger.info("match_importance distribution:\n"
                + df_master["match_importance"].value_counts().sort_index().to_string())

    out = PROCESSED_DATA_DIR / "master_features.parquet"
    df_master.to_parquet(out, index=False)
    logger.success(f"Step 5 saved → {out}  shape={df_master.shape}")
    return df_master


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("build_features.py — building master_features.parquet")
    logger.info("=" * 60)
    t_total = time.time()

    df_master = step1_base_features()
    df_master = step2_rankings(df_master)
    df_master = step3_squad_value(df_master)
    df_master = step4_rest_days(df_master)
    df_master = step5_match_importance(df_master)

    elapsed = time.time() - t_total
    logger.success(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"Final shape: {df_master.shape[0]:,} rows × {df_master.shape[1]} columns")
    logger.info(f"Last 5 columns: {df_master.columns.tolist()[-5:]}")


if __name__ == "__main__":
    main()
