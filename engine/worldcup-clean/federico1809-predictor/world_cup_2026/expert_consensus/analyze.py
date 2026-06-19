"""
analyze.py
----------
Compare expert consensus predictions vs XGBoost simulation results.

Usage:
    python -m world_cup_2026.expert_consensus.analyze
    python -m world_cup_2026.expert_consensus.analyze --predictions path/to/file.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from scipy.stats import spearmanr

from world_cup_2026.config import DATA_DIR, PREDICTIONS_DIR

app = typer.Typer(add_completion=False)

_DEFAULT_PREDICTIONS = DATA_DIR / "expert_opinions" / "processed" / "expert_predictions.csv"
_DEFAULT_SIMULATION = PREDICTIONS_DIR / "simulation_results.csv"

_CONFIDENCE_WEIGHTS = {
    "certain": 1.0,
    "likely": 0.75,
    "possible": 0.5,
    "unlikely": 0.25,
}


def compute_expert_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-team expert consensus metrics from the predictions DataFrame."""
    mention_count = df.groupby("team").size().rename("mention_count")

    champion_df = df[df["prediction_type"] == "champion"].copy()
    champion_df["weight"] = champion_df["confidence"].map(_CONFIDENCE_WEIGHTS).fillna(0.0)

    champion_mentions = champion_df.groupby("team").size().rename("champion_mentions")
    champion_score = champion_df.groupby("team")["weight"].sum().rename("champion_score")
    avg_sentiment = df.groupby("team")["sentiment"].mean().rename("avg_sentiment")

    consensus = pd.concat(
        [mention_count, champion_mentions, avg_sentiment, champion_score], axis=1
    ).fillna(0.0)

    total = consensus["champion_score"].sum()
    consensus["expert_p_champion"] = (
        consensus["champion_score"] / total if total > 0 else 0.0
    )

    return consensus.reset_index()


def build_comparison(consensus: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Merge expert consensus with simulation results and compute rank columns."""
    merged = sim[["team", "p_champion"]].merge(
        consensus[["team", "expert_p_champion"]], on="team", how="left"
    )
    merged["expert_p_champion"] = merged["expert_p_champion"].fillna(0.0)

    merged["rank_model"] = (
        merged["p_champion"].rank(ascending=False, method="min").astype(int)
    )
    merged["rank_expert"] = (
        merged["expert_p_champion"].rank(ascending=False, method="min").astype(int)
    )
    merged["rank_delta"] = merged["rank_model"] - merged["rank_expert"]

    return merged.sort_values("rank_model").reset_index(drop=True)


@app.command()
def main(
    predictions: Path = typer.Option(
        _DEFAULT_PREDICTIONS, "--predictions", help="Path to expert_predictions.csv."
    ),
    simulation: Path = typer.Option(
        _DEFAULT_SIMULATION, "--simulation", help="Path to simulation_results.csv."
    ),
) -> None:
    if not predictions.exists():
        logger.error(f"Predictions file not found: {predictions}")
        raise typer.Exit(1)
    if not simulation.exists():
        logger.error(f"Simulation results not found: {simulation}")
        raise typer.Exit(1)

    df_preds = pd.read_csv(predictions)
    df_sim = pd.read_csv(simulation)

    n_sources = df_preds["url"].nunique() if "url" in df_preds.columns else "?"
    logger.info(f"Loaded {len(df_preds)} predictions from {n_sources} sources")

    consensus = compute_expert_consensus(df_preds)
    comparison = build_comparison(consensus, df_sim)

    print("\n── Top 10 by Model p_champion ─────────────────────────────────")
    top_model = comparison.nsmallest(10, "rank_model")[["team", "p_champion", "rank_model"]]
    print(top_model.to_string(index=False))

    print("\n── Top 10 by Expert p_champion ────────────────────────────────")
    top_expert = comparison.nsmallest(10, "rank_expert")[
        ["team", "expert_p_champion", "rank_expert"]
    ]
    print(top_expert.to_string(index=False))

    print("\n── Full Comparison Table ───────────────────────────────────────")
    print(
        comparison[
            ["team", "p_champion", "expert_p_champion",
             "rank_model", "rank_expert", "rank_delta"]
        ].to_string(index=False)
    )

    corr, pvalue = spearmanr(comparison["rank_model"], comparison["rank_expert"])
    print(
        f"\nSpearman correlation (rank_model vs rank_expert): "
        f"ρ = {corr:.3f}  p = {pvalue:.4f}"
    )


if __name__ == "__main__":
    app()
