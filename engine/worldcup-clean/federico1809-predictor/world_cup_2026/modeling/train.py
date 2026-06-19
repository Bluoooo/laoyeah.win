"""
train.py — Production training script for WC2026 match predictor.

Reads master_features.parquet, trains XGBoost with known best_params,
serializes model + feature list to models/.

Usage:
    python -m world_cup_2026.modeling.train
    python -m world_cup_2026.modeling.train --parquet-path data/processed/master_features.parquet
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import typer
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from world_cup_2026.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED

app = typer.Typer()

# ── Hyperparameters (from Optuna run, notebook 04) ────────────────────────────
BEST_PARAMS = {
    "n_estimators":     414,
    "max_depth":        6,
    "learning_rate":    0.00960612367254347,
    "subsample":        0.7271351657620417,
    "colsample_bytree": 0.6981036422155107,
    "min_child_weight": 10,
    "reg_alpha":        0.6172115948107072,
    "reg_lambda":       0.7404472559987593,
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "tree_method":      "hist",
    "random_state":     RANDOM_SEED,
    "verbosity":        0,
}

# ── Feature groups ────────────────────────────────────────────────────────────
ELO_FEATURES = ["elo_pre_home", "elo_pre_away", "elo_diff", "win_prob_home"]
NEUTRAL_FEATURES = ["neutral"]
RANKING_FEATURES = ["ranking_home", "ranking_away", "ranking_diff"]
SQUAD_VALUE_FEATURES = ["squad_value_home", "squad_value_away", "squad_value_diff"]
FORM_FEATURES = (
    [f"home_form_{w}_{s}" for w in [5, 10, 20] for s in [
        "win_rate", "draw_rate", "loss_rate",
        "goals_scored_avg", "goals_conceded_avg", "goal_diff_avg",
        "clean_sheet_rate", "failed_score_rate", "points_avg",
        "matches_played", "weighted_points",
    ]]
    + [f"away_form_{w}_{s}" for w in [5, 10, 20] for s in [
        "win_rate", "draw_rate", "loss_rate",
        "goals_scored_avg", "goals_conceded_avg", "goal_diff_avg",
        "clean_sheet_rate", "failed_score_rate", "points_avg",
        "matches_played", "weighted_points",
    ]]
)
H2H_FEATURES = [
    "h2h_matches", "h2h_win_rate_a", "h2h_goal_diff_a",
    "h2h_elo_edge_a", "h2h_weighted_edge_a", "h2h_decay_weight",
    "h2h_reliable", "transitive_common_rivals", "transitive_edge_a",
    "transitive_goal_diff_edge", "transitive_reliable",
]
CLUSTER_FEATURES = ["home_cluster_enc", "away_cluster_enc"]

NUMERIC_FEATURES = (
    ELO_FEATURES + NEUTRAL_FEATURES + RANKING_FEATURES
    + SQUAD_VALUE_FEATURES + FORM_FEATURES + H2H_FEATURES
)
MODEL_FEATURES = NUMERIC_FEATURES + CLUSTER_FEATURES


@app.command()
def main(
    parquet_path: Path = PROCESSED_DATA_DIR / "master_features.parquet",
    model_out: Path = MODELS_DIR / "xgb_match_predictor.pkl",
    features_out: Path = MODELS_DIR / "model_features.json",
    train_cutoff: int = 2021,
):
    """Train XGBoost match predictor and serialize to disk."""

    np.random.seed(RANDOM_SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading features from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded: {df.shape}")

    # ── Cluster encoding ──────────────────────────────────────────────────────
    # Cluster labels not in master_features — join from team_snapshot_clustered
    df_clusters = pd.read_parquet(PROCESSED_DATA_DIR / "team_snapshot_clustered.parquet")
    cluster_map = df_clusters.set_index("team")["cluster_name"].to_dict()

    df["home_cluster"] = df["home_team"].map(cluster_map).fillna("Non-WC2026")
    df["away_cluster"] = df["away_team"].map(cluster_map).fillna("Non-WC2026")

    le_home = LabelEncoder()
    le_away = LabelEncoder()
    all_labels = ["Consolidated Mid-Tier", "Dynamic Mid-Tier", "Elite", "Non-WC2026", "Underdogs"]
    le_home.fit(all_labels)
    le_away.fit(all_labels)

    df["home_cluster_enc"] = le_home.transform(df["home_cluster"])
    df["away_cluster_enc"] = le_away.transform(df["away_cluster"])
    logger.info("Cluster encoding applied.")

    # ── Temporal split ────────────────────────────────────────────────────────
    df["year"] = df["date"].dt.year
    train = df[df["year"] <= train_cutoff].copy()
    val   = df[(df["year"] >= train_cutoff + 1) & (df["year"] <= train_cutoff + 2)].copy()

    logger.info(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

    # ── Validate features ─────────────────────────────────────────────────────
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        logger.error(f"Missing features in parquet: {missing}")
        raise SystemExit(1)

    X_train = train[MODEL_FEATURES].astype(float).values
    y_train = train["target"].values
    X_val   = val[MODEL_FEATURES].astype(float).values
    y_val   = val["target"].values

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = xgb.XGBClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    # ── Evaluate on val ───────────────────────────────────────────────────────
    from sklearn.metrics import f1_score, log_loss
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")
    ll = log_loss(y_val, y_prob)
    logger.info(f"Val — F1-macro: {f1:.4f} | Log-loss: {ll:.4f}")

    # ── Serialize ─────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    logger.success(f"Model saved → {model_out}")

    with open(features_out, "w") as f:
        json.dump(MODEL_FEATURES, f, indent=2)
    logger.success(f"Features saved → {features_out} ({len(MODEL_FEATURES)} features)")


if __name__ == "__main__":
    app()