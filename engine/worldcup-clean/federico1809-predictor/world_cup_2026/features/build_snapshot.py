"""
build_snapshot.py
-----------------
Standalone pipeline that replicates notebooks/03_unsupervised/03_clustering.ipynb
and produces data/processed/team_snapshot_clustered.parquet.

Usage:
    python -m world_cup_2026.features.build_snapshot
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from world_cup_2026.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_SEED
from world_cup_2026.data_ingestion.normalize import (
    normalize_dataframe_teams,
    normalize_team_name,
)
from world_cup_2026.features.elo import calculate_elo
from world_cup_2026.features.form import FormCalculator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AS_OF = pd.Timestamp("2026-03-31")

CLUSTER_FEATURES: list[str] = [
    "elo",
    "form_5_win_rate",
    "form_10_win_rate",
    "form_20_win_rate",
    "form_5_goals_scored_avg",
    "form_10_goals_scored_avg",
    "form_5_goal_diff_avg",
    "form_10_goal_diff_avg",
    "form_5_points_avg",
    "form_10_points_avg",
    "form_20_points_avg",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_cluster_names(df: pd.DataFrame) -> dict[int, str]:
    """Map KMeans cluster integers to semantic names.

    Rules (applied after KMeans fit):
        Highest avg Elo  → "Elite"
        Lowest avg Elo   → "Underdogs"
        Middle two: higher avg form_10_win_rate → "Dynamic Mid-Tier"
                    lower  avg form_10_win_rate → "Consolidated Mid-Tier"
    """
    stats = (
        df.groupby("cluster")
        .agg(avg_elo=("elo", "mean"), avg_wr=("form_10_win_rate", "mean"))
        .sort_values("avg_elo", ascending=False)
    )
    ranked = stats.index.tolist()  # cluster IDs sorted high→low Elo

    names: dict[int, str] = {
        ranked[0]: "Elite",
        ranked[3]: "Underdogs",
    }
    mid_a, mid_b = ranked[1], ranked[2]
    if stats.loc[mid_a, "avg_wr"] >= stats.loc[mid_b, "avg_wr"]:
        names[mid_a] = "Dynamic Mid-Tier"
        names[mid_b] = "Consolidated Mid-Tier"
    else:
        names[mid_b] = "Dynamic Mid-Tier"
        names[mid_a] = "Consolidated Mid-Tier"

    for cid, name in sorted(names.items()):
        logger.info(
            f"  Cluster {cid} → {name!r:30s}"
            f"  avg_elo={stats.loc[cid,'avg_elo']:.1f}"
            f"  avg_wr={stats.loc[cid,'avg_wr']:.3f}"
        )
    return names


# ---------------------------------------------------------------------------
# Step 1 — Build team snapshot (Elo + Form)
# ---------------------------------------------------------------------------

def build_snapshot() -> pd.DataFrame:
    """Compute per-team Elo and form stats for the 48 WC2026 teams."""
    logger.info("Loading WC2026 teams...")
    df_teams = pd.read_csv(RAW_DATA_DIR / "areezvisram12_fixture" / "teams.csv")
    df_teams = df_teams[~df_teams["is_placeholder"]].copy()
    df_teams["team_norm"] = df_teams["team_name"].map(normalize_team_name)
    logger.info(f"WC2026 teams loaded: {len(df_teams)}")

    logger.info("Loading match results...")
    df_results = pd.read_csv(
        RAW_DATA_DIR / "martj42_results" / "results.csv",
        parse_dates=["date"],
    )
    df_results_norm = normalize_dataframe_teams(df_results, ["home_team", "away_team"])

    t0 = time.time()
    df_elo = calculate_elo(df_results_norm)
    logger.info(f"Elo computed in {time.time() - t0:.1f}s")

    logger.info("Initializing FormCalculator (windows=[5, 10, 20])...")
    calc = FormCalculator(df_results_norm, windows=[5, 10, 20])

    logger.info(f"Building team snapshots as of {AS_OF.date()}...")
    snapshots = []
    for _, team_row in df_teams.iterrows():
        team = team_row["team_norm"]
        group = team_row["group_letter"]

        # Elo: pre-match Elo of the team's last recorded match before AS_OF
        mask = (
            ((df_elo["home_team"] == team) | (df_elo["away_team"] == team))
            & (df_elo["date"] < AS_OF)
        )
        team_matches = df_elo[mask].sort_values("date")
        if team_matches.empty:
            elo = 1500.0
        else:
            last = team_matches.iloc[-1]
            elo = (
                last["elo_pre_home"]
                if last["home_team"] == team
                else last["elo_pre_away"]
            )

        form_5  = calc.get_team_current_form(team, as_of=AS_OF, window=5)
        form_10 = calc.get_team_current_form(team, as_of=AS_OF, window=10)
        form_20 = calc.get_team_current_form(team, as_of=AS_OF, window=20)

        snap: dict = {"team": team, "group": group, "elo": round(elo, 1)}
        snap.update(form_5)
        snap.update(form_10)
        snap.update(form_20)
        snapshots.append(snap)

    df_snapshot = pd.DataFrame(snapshots)
    logger.info(f"Snapshot built: {df_snapshot.shape}")
    return df_snapshot


# ---------------------------------------------------------------------------
# Step 2 — Cluster + PCA + anomaly detection
# ---------------------------------------------------------------------------

def cluster_snapshot(df_snapshot: pd.DataFrame) -> pd.DataFrame:
    """Run KMeans, PCA, and anomaly detection on the team snapshot."""
    missing = [c for c in CLUSTER_FEATURES if c not in df_snapshot.columns]
    if missing:
        raise ValueError(f"Missing cluster features: {missing}")

    X = df_snapshot[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info(f"Feature matrix scaled: {X_scaled.shape}")

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_snapshot = df_snapshot.copy()
    df_snapshot["cluster"] = cluster_labels
    logger.info(f"KMeans inertia: {kmeans.inertia_:.2f}")
    logger.info(
        "Cluster sizes (raw):\n"
        + df_snapshot["cluster"].value_counts().sort_index().to_string()
    )

    # Derive semantic cluster names from statistics
    logger.info("Deriving cluster names...")
    cluster_names = _derive_cluster_names(df_snapshot)
    df_snapshot["cluster_name"] = df_snapshot["cluster"].map(cluster_names)

    # PCA 2D projection (visualization only)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    df_snapshot["pc1"] = X_pca[:, 0]
    df_snapshot["pc2"] = X_pca[:, 1]
    var = pca.explained_variance_ratio_
    logger.info(
        f"PCA variance — PC1: {var[0]:.1%}, PC2: {var[1]:.1%}, "
        f"total: {sum(var):.1%}"
    )

    # Anomaly detection: Euclidean distance to own cluster centroid
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[cluster_labels], axis=1)
    df_snapshot["dist_to_centroid"] = distances
    threshold = distances.mean() + 1.5 * distances.std()
    df_snapshot["is_anomaly"] = distances > threshold
    logger.info(f"Anomaly threshold (mean + 1.5σ): {threshold:.3f}")
    logger.info(f"Anomalies detected: {df_snapshot['is_anomaly'].sum()}")

    return df_snapshot


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("build_snapshot.py — building team_snapshot_clustered.parquet")
    logger.info("=" * 60)
    np.random.seed(RANDOM_SEED)
    t_total = time.time()

    df_snapshot = build_snapshot()
    df_snapshot = cluster_snapshot(df_snapshot)

    out = PROCESSED_DATA_DIR / "team_snapshot_clustered.parquet"
    df_snapshot.to_parquet(out, index=False)
    logger.success(f"Saved → {out}  shape={df_snapshot.shape}")
    logger.info(
        "Cluster distribution:\n"
        + df_snapshot["cluster_name"].value_counts().to_string()
    )
    logger.info(f"Columns: {df_snapshot.columns.tolist()}")
    logger.success(f"Pipeline complete in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
