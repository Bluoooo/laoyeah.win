"""
simulate.py — Production simulation script for WC2026 tournament.

Loads trained XGBoost model, runs N Monte Carlo simulations of the full
104-match tournament, saves probability table to outputs/predictions/.

Usage:
    python -m world_cup_2026.simulation.simulate
    python -m world_cup_2026.simulation.simulate --n-sims 1000
"""

from pathlib import Path
from collections import Counter
from itertools import combinations
import json

import numpy as np
import pandas as pd
import joblib
import typer
from loguru import logger
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from world_cup_2026.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
)

app = typer.Typer()

PHASES = ["Group", "R32", "R16", "QF", "SF", "Third", "Final", "Champion"]

FORM_WINDOWS = [5, 10, 20]
FORM_STATS = [
    "win_rate", "draw_rate", "loss_rate",
    "goals_scored_avg", "goals_conceded_avg", "goal_diff_avg",
    "clean_sheet_rate", "failed_score_rate", "points_avg",
    "matches_played", "weighted_points",
]


# ── Globals (populated in main) ───────────────────────────────────────────────
xgb_model   = None
MODEL_FEATURES = None
df_snapshot = None
df_teams    = None
all_teams   = None
MATCH_PROBS = None


def build_match_features(home_team, away_team, neutral=True):
    h = df_snapshot[df_snapshot["team"] == home_team].iloc[0]
    a = df_snapshot[df_snapshot["team"] == away_team].iloc[0]

    elo_pre_home  = h["elo"]
    elo_pre_away  = a["elo"]
    elo_diff      = elo_pre_home - elo_pre_away
    win_prob_home = 1 / (1 + 10 ** (-elo_diff / 400))

    features = {
        "elo_pre_home":     elo_pre_home,
        "elo_pre_away":     elo_pre_away,
        "elo_diff":         elo_diff,
        "win_prob_home":    win_prob_home,
        "neutral":          int(neutral),
        "ranking_home":     h["ranking"],
        "ranking_away":     a["ranking"],
        "ranking_diff":     h["ranking"] - a["ranking"],
        "squad_value_home": h["squad_value"],
        "squad_value_away": a["squad_value"],
        'squad_value_diff': h["squad_value"] - a["squad_value"],
        'rest_days_home':   30,
        'rest_days_away':   30,
        'match_importance': 3,  # World Cup = tier 3 (highest)
    }

    for w in FORM_WINDOWS:
        for stat in FORM_STATS:
            features[f"home_form_{w}_{stat}"] = h[f"form_{w}_{stat}"]
            features[f"away_form_{w}_{stat}"] = a[f"form_{w}_{stat}"]

    features.update({
        "h2h_matches":              0,
        "h2h_win_rate_a":           0.5,
        "h2h_goal_diff_a":          0.0,
        "h2h_elo_edge_a":           0.0,
        "h2h_weighted_edge_a":      0.0,
        "h2h_decay_weight":         0.0,
        "h2h_reliable":             0,
        "transitive_common_rivals": 0,
        "transitive_edge_a":        0.0,
        "transitive_goal_diff_edge": 0.0,
        "transitive_reliable":      0,
    })

    features["home_cluster_enc"] = h["cluster_enc"]
    features["away_cluster_enc"] = a["cluster_enc"]

    return np.array([features[f] for f in MODEL_FEATURES], dtype=float)

def precompute_match_probs(teams):
    """Precompute predict_proba for all possible matchups in batch."""
    from itertools import permutations
    pairs = list(permutations(teams, 2))
    matrix = np.array(
        [build_match_features(h, a) for h, a in pairs],
        dtype=float
    )
    probs = xgb_model.predict_proba(matrix)
    return {(h, a): probs[i] for i, (h, a) in enumerate(pairs)}


def simulate_match_fast(home_team, away_team, neutral=True):
    """Sample match result from precomputed probabilities."""
    probs = MATCH_PROBS[(home_team, away_team)]
    result = np.random.choice(["away", "draw", "home"], p=probs)
    if result == "home":
        return 3, 0, "home"
    elif result == "away":
        return 0, 3, "away"
    else:
        return 1, 1, "draw"


def simulate_knockout_fast(team_a, team_b):
    _, _, result = simulate_match_fast(team_a, team_b)
    if result == "draw":
        result = "home" if np.random.random() < 0.5 else "away"
    return team_a if result == "home" else team_b


def simulate_group_fast(teams):
    standings = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    h_snap = {t: df_snapshot[df_snapshot["team"] == t].iloc[0] for t in teams}
    for home, away in combinations(teams, 2):
        h_pts, a_pts, result = simulate_match_fast(home, away)
        standings[home]["pts"] += h_pts
        standings[away]["pts"] += a_pts
        h_gf = max(0, round(np.random.normal(h_snap[home]["form_10_goals_scored_avg"], 1.0)))
        a_gf = max(0, round(np.random.normal(h_snap[away]["form_10_goals_scored_avg"], 1.0)))
        standings[home]["gd"] += h_gf - a_gf
        standings[away]["gd"] += a_gf - h_gf
        standings[home]["gf"] += h_gf
        standings[away]["gf"] += a_gf
    return standings


def rank_group(standings):
    return sorted(
        standings.keys(),
        key=lambda t: (standings[t]["pts"], standings[t]["gd"], standings[t]["gf"]),
        reverse=True,
    )


def get_groups():
    groups = {}
    for _, row in df_teams.iterrows():
        g = row["group_letter"]
        if g not in groups:
            groups[g] = []
        groups[g].append(row["team_name"])
    return groups


def select_best_thirds(third_place_teams, third_place_standings):
    return sorted(
        third_place_teams,
        key=lambda t: (
            third_place_standings[t]["pts"],
            third_place_standings[t]["gd"],
            third_place_standings[t]["gf"],
        ),
        reverse=True,
    )[:8]


def resolve_third(thirds_sorted, groups_allowed, used_thirds):
    for team in thirds_sorted:
        if team in used_thirds:
            continue
        team_group = df_teams[df_teams["team_name"] == team]["group_letter"].values[0]
        if team_group in groups_allowed:
            used_thirds.add(team)
            return team
    for team in thirds_sorted:
        if team not in used_thirds:
            used_thirds.add(team)
            return team


def assemble_r32(group_winners, group_runners, best_thirds, third_standings):
    w = group_winners
    r = group_runners
    thirds_sorted = sorted(
        best_thirds,
        key=lambda t: (
            third_standings[t]["pts"],
            third_standings[t]["gd"],
            third_standings[t]["gf"],
        ),
        reverse=True,
    )
    used = set()
    return [
        (r["A"],  r["B"]),
        (w["C"],  r["F"]),
        (w["E"],  resolve_third(thirds_sorted, list("ABCDF"), used)),
        (w["F"],  r["C"]),
        (r["E"],  r["I"]),
        (w["I"],  resolve_third(thirds_sorted, list("CDFGH"), used)),
        (w["A"],  resolve_third(thirds_sorted, list("CEFHI"), used)),
        (w["L"],  resolve_third(thirds_sorted, list("EHIJK"), used)),
        (w["G"],  resolve_third(thirds_sorted, list("AEHIJ"), used)),
        (w["D"],  resolve_third(thirds_sorted, list("BEFIJ"), used)),
        (w["H"],  r["J"]),
        (r["K"],  r["L"]),
        (w["B"],  resolve_third(thirds_sorted, list("EFGIJ"), used)),
        (r["D"],  r["G"]),
        (w["J"],  r["H"]),
        (w["K"],  resolve_third(thirds_sorted, list("DEIJL"), used)),
    ]


def load_expert_signal(csv_path: Path) -> dict[str, float]:
    """Load expert consensus champion probabilities, normalized to sum to 1."""
    if not csv_path.exists():
        logger.warning(f"Expert CSV not found: {csv_path} — skipping blend.")
        return {}
    df = pd.read_csv(csv_path)
    champion_preds = df[df["prediction_type"] == "champion"]
    counts = champion_preds["team"].value_counts()
    total = counts.sum()
    if total == 0:
        return {}
    return (counts / total).to_dict()


def simulate_tournament_fast():
    groups          = get_groups()
    group_winners   = {}
    group_runners   = {}
    third_teams     = []
    third_standings = {}

    for letter, teams in sorted(groups.items()):
        standings = simulate_group_fast(teams)
        ranked    = rank_group(standings)
        group_winners[letter] = ranked[0]
        group_runners[letter] = ranked[1]
        third_teams.append(ranked[2])
        third_standings[ranked[2]] = standings[ranked[2]]

    best_thirds = select_best_thirds(third_teams, third_standings)
    phase = {t: "Group" for t in all_teams}

    for t in list(group_winners.values()) + list(group_runners.values()) + best_thirds:
        phase[t] = "R32"

    r32_matches = assemble_r32(group_winners, group_runners, best_thirds, third_standings)
    r32_winners = [simulate_knockout_fast(h, a) for h, a in r32_matches]
    for t in r32_winners:
        phase[t] = "R16"

    r16_pairs = [
        (r32_winners[0],  r32_winners[2]),
        (r32_winners[1],  r32_winners[4]),
        (r32_winners[3],  r32_winners[5]),
        (r32_winners[6],  r32_winners[7]),
        (r32_winners[10], r32_winners[11]),
        (r32_winners[8],  r32_winners[9]),
        (r32_winners[13], r32_winners[15]),
        (r32_winners[12], r32_winners[14]),
    ]
    r16_winners = [simulate_knockout_fast(h, a) for h, a in r16_pairs]
    for t in r16_winners:
        phase[t] = "QF"

    qf_pairs = [
        (r16_winners[0], r16_winners[1]),
        (r16_winners[4], r16_winners[5]),
        (r16_winners[2], r16_winners[3]),
        (r16_winners[6], r16_winners[7]),
    ]
    qf_winners = [simulate_knockout_fast(h, a) for h, a in qf_pairs]
    for t in qf_winners:
        phase[t] = "SF"

    sf_winners = []
    sf_losers  = []
    for a, b in [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]:
        winner = simulate_knockout_fast(a, b)
        loser  = b if winner == a else a
        sf_winners.append(winner)
        sf_losers.append(loser)
    for t in sf_winners:
        phase[t] = "Final"

    third = simulate_knockout_fast(sf_losers[0], sf_losers[1])
    phase[third] = "Third"

    champion = simulate_knockout_fast(sf_winners[0], sf_winners[1])
    phase[champion] = "Champion"

    return phase


@app.command()
def main(
    n_sims: int = 10_000,
    model_path: Path = Path("models/xgb_match_predictor.pkl"),
    features_path: Path = Path("models/model_features.json"),
    output_path: Path = Path("outputs/predictions/simulation_results.csv"),
    snapshot_date: str = "2026-03-31",
    alpha: float = typer.Option(0.3, "--alpha", help="Expert blend weight (0=model only, 1=expert only)."),
    expert_csv: Path = typer.Option(Path("data/expert_opinions/processed/expert_predictions.csv"), "--expert-csv"),
):
    """Run N Monte Carlo simulations of WC2026 and save probability table."""
    global xgb_model, MODEL_FEATURES, df_snapshot, df_teams, all_teams

    np.random.seed(RANDOM_SEED)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model from {model_path}")
    xgb_model = joblib.load(model_path)
    with open(features_path) as f:
        MODEL_FEATURES = json.load(f)
    logger.info(f"Model loaded. Features: {len(MODEL_FEATURES)}")

    # ── Load teams and fixture ────────────────────────────────────────────────
    df_teams = pd.read_csv(RAW_DATA_DIR / "areezvisram12_fixture" / "teams.csv")
    df_teams = df_teams[~df_teams["is_placeholder"]].reset_index(drop=True)
    all_teams = df_teams["team_name"].tolist()
    logger.info(f"Teams loaded: {len(all_teams)}")

    # ── Build snapshot ────────────────────────────────────────────────────────
    SNAPSHOT_DATE = pd.Timestamp(snapshot_date)
    df_master = pd.read_parquet(PROCESSED_DATA_DIR / "master_features.parquet")

    df_snapshot_raw = pd.read_parquet(PROCESSED_DATA_DIR / "team_snapshot_clustered.parquet")

    # Rankings
    latest_rankings = (
        df_master[df_master["date"] <= SNAPSHOT_DATE]
        .sort_values("date")
        .groupby("home_team")
        .last()["ranking_home"]
        .reset_index()
        .rename(columns={"home_team": "team", "ranking_home": "ranking"})
    )
    df_snapshot_raw = df_snapshot_raw.merge(latest_rankings, on="team", how="left")
    df_snapshot_raw["ranking"] = df_snapshot_raw["ranking"].fillna(
        latest_rankings["ranking"].median()
    )

    # Squad value
    latest_squad = (
        df_master[df_master["date"] <= SNAPSHOT_DATE]
        .sort_values("date")
        .groupby("home_team")
        .last()["squad_value_home"]
        .reset_index()
        .rename(columns={"home_team": "team", "squad_value_home": "squad_value"})
    )
    df_snapshot_raw = df_snapshot_raw.merge(latest_squad, on="team", how="left")
    df_snapshot_raw["squad_value"] = df_snapshot_raw["squad_value"].fillna(
        latest_squad["squad_value"].median()
    )

    # Cluster encoding
    le = LabelEncoder()
    le.fit(["Consolidated Mid-Tier", "Dynamic Mid-Tier", "Elite", "Non-WC2026", "Underdogs"])
    df_snapshot_raw["cluster_enc"] = le.transform(df_snapshot_raw["cluster_name"])

    df_snapshot = df_snapshot_raw
    logger.info(f"Snapshot ready: {df_snapshot.shape}")
    logger.info("Precomputing match probabilities...")
    global MATCH_PROBS
    MATCH_PROBS = precompute_match_probs(all_teams)
    logger.info(f"Precomputed {len(MATCH_PROBS):,} matchups")


    # ── Run simulations ───────────────────────────────────────────────────────
    counts = {t: {p: 0 for p in PHASES} for t in all_teams}
    logger.info(f"Running {n_sims:,} simulations...")

    for _ in tqdm(range(n_sims)):
        result = simulate_tournament_fast()
        for team, phase in result.items():
            if team in counts and phase in counts[team]:
                counts[team][phase] += 1

    # ── Build results table ───────────────────────────────────────────────────
    rows = []
    for team in all_teams:
        row = {"team": team}
        for phase in PHASES:
            row[f"p_{phase.lower()}"] = round(counts[team][phase] / n_sims, 4)
        rows.append(row)

    df_results = pd.DataFrame(rows).sort_values("p_champion", ascending=False)

    if alpha > 0:
        expert_signal = load_expert_signal(expert_csv)
        if expert_signal:
            df_results["expert_p_champion"] = df_results["team"].map(
                expert_signal
            ).fillna(0.0)
            df_results["p_champion_blended"] = (
                (1 - alpha) * df_results["p_champion"]
                + alpha * df_results["expert_p_champion"]
            ).round(4)
            df_results = df_results.sort_values(
                "p_champion_blended", ascending=False
            )
            logger.info(
                f"Expert blend applied — alpha={alpha}, "
                f"teams with expert signal: {(df_results['expert_p_champion'] > 0).sum()}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logger.success(f"Results saved → {output_path}")

    # ── Print top 10 ──────────────────────────────────────────────────────────
    if "p_champion_blended" in df_results.columns:
        print("\nTop 10 — P(Champion) blended:")
        print(
            df_results.head(10)[
                ["team", "p_champion_blended", "p_champion", "expert_p_champion"]
            ].to_string(index=False)
        )
    else:
        print("\nTop 10 — P(Champion):")
        print(
            df_results.head(10)[["team", "p_champion", "p_final", "p_sf", "p_r16"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    app()