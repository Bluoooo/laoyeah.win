# ⚽ FIFA World Cup 2026 Predictor
> End-to-end machine learning pipeline for predicting FIFA World Cup 2026 outcomes
> using ensemble models, Monte Carlo simulation, and unsupervised clustering.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

🔴 **Live dashboard:** [world-cup-2026-predictor-board.streamlit.app](https://world-cup-2026-predictor-board.streamlit.app)

---

## 🎯 What This Project Predicts

| Level | Output |
|-------|--------|
| Match | Win / Draw / Loss probability per match |
| Group stage | Final standings + classification probabilities |
| Knockout bracket | How R32/R16 fixture assembles from group results |
| Phase advancement | P(reach R32 / R16 / QF / SF / Final / Win) per team |
| Tournament winner | Full probability distribution for all 48 teams |

### Adaptive retraining strategy
The model is designed for real-time use during the tournament:
- Pre-tournament: predict all 104 matches end-to-end via Monte Carlo
- After group stage: retrain on actual group results, predict knockout phase
- After each round: incorporate new results, update remaining predictions

Final presentation: FiveThirtyEight-style probability table + Streamlit interactive
dashboard + static notebook visualizations.

---

## 🧠 Key Design Principles

- Recent form over historical reputation — Brazil's 1970 title does not predict 2026.
  All features computed with strict temporal awareness.
- No data leakage — features computed strictly as of each match date.
- Strict temporal split — train/val/test by date, never random.
- Calibrated probabilities — outputs calibrated for reliable Monte Carlo input.
- Unsupervised methods inform supervised — clustering of 48 qualified teams generates
  categorical features; PCA detects multicollinearity.
- Adaptive pipeline — model retrained mid-tournament with real results.

---

## 📁 Project Structure

    world-cup-2026-predictor/
    ├── configs/                  Global parameters (seeds, paths, hyperparameters)
    ├── data/
    │   ├── raw/                  Downloaded datasets (never modified)
    │   ├── interim/              Intermediate transformations
    │   ├── processed/            Model-ready feature matrices (*.parquet tracked)
    │   └── external/             StatsBomb events, third-party sources
    ├── models/                   Serialized trained models and encoders
    ├── notebooks/
    │   ├── 01_eda/               Exploratory Data Analysis
    │   ├── 02_features/          Feature Engineering (reference — use build_features.py)
    │   ├── 03_unsupervised/      Clustering and PCA (reference — use build_snapshot.py)
    │   ├── 04_modeling/          Supervised modeling and evaluation
    │   └── 05_simulation/        Monte Carlo tournament simulation
    ├── outputs/
    │   ├── figures/              Generated plots (gitignored)
    │   └── predictions/          Tournament simulation results
    ├── streamlit_app/            Interactive dashboard (app.py)
    ├── world_cup_2026/
    │   ├── data_ingestion/       Download pipeline and normalization
    │   ├── features/
    │   │   ├── build_features.py     Feature matrix pipeline (replaces notebook 02)
    │   │   ├── build_snapshot.py     Team snapshot + clustering (replaces notebook 03)
    │   │   ├── elo.py                Elo rating calculator
    │   │   ├── form.py               Recent form features
    │   │   └── h2h.py                Head-to-head + transitive features
    │   ├── modeling/
    │   │   └── train.py              XGBoost training script
    │   ├── simulation/
    │   │   └── simulate.py           Monte Carlo simulation engine
    │   └── expert_consensus/
    │       ├── news_scraper.py       Auto-discovers URLs via Google News (17 queries EN/ES/FR)
    │       ├── scraper.py            Fetches and parses article text (newspaper4k)
    │       ├── extractor.py          Structured extraction via Gemini 2.5-flash
    │       ├── pipeline.py           Orchestrates scraper + extractor, writes CSV
    │       └── analyze.py            Expert consensus vs model comparison
    └── tests/

---

## 📊 Datasets

| Source | Description | Period | Size |
|--------|-------------|--------|------|
| martj42 | All international results | 1872-2026 | 49,071 matches |
| patateriedata | Daily updated results incl. qualifiers | 1872-2026 | 51,384 matches |
| lchikry | Pre-calculated Elo, form, FIFA ratings | 1872-2025 | 43,364 x 35 features |
| joshfjelstul | Relational World Cup database | 1930-2022 | 900 matches |
| cashncarry | Monthly FIFA rankings | 1992-2024 | 67,472 records |
| sarazahran1 | WC2026 Elo probability baseline | 2026 | 72 matches |
| areezvisram12 | Complete 104-match fixture | 2026 | 104 matches |
| StatsBomb Open Data | Match events xG, passes, shots | 2018-2022 WC | JSON |

---

## 🔬 Feature Engineering

### Implemented (master_features.parquet — 9,796 × 98)

| Module | File | Description |
|--------|------|-------------|
| Elo rating | features/elo.py | Recalculated from 150yr history, dynamic K-factor (WC=60, Friendly=20) |
| H2H + Transitive | features/h2h.py | Direct H2H edge + transitive rival + temporal decay |
| Recent form | features/form.py | Win rate, goals, points over 5/10/20 matches + exp decay |
| FIFA Rankings | cashncarry dataset | ranking_home, ranking_away, ranking_diff — as-of join per match date |
| Neutral venue | results dataset | Binary flag — reduces home advantage ~3.5pp |
| Squad market value | Transfermarkt | squad_value_home, squad_value_away, squad_value_diff |
| Rest days | match dates | Days since last match per team — defaults to 30 for WC2026 |
| Match importance | tournament tier | Tier encoding 0–3 — World Cup fixed at 3 (highest) |
| Cluster label | build_snapshot.py | KMeans cluster assignment (Elite / Mid-Tier / Underdogs) |

### Planned

| Feature | Source | Signal strength |
|---------|--------|-----------------|
| Average squad age | Transfermarkt | HIGH |
| Coach tenure months | Transfermarkt | MEDIUM |
| Squad continuity since 2022 WC | Transfermarkt | HIGH |
| Key player injuries/suspensions | Press scraping | HIGH |
| Venue altitude | Sedes data | MEDIUM |
| Match day weather | Weather API | LOW-MEDIUM |

---

## 🤖 Modeling Pipeline

### Supervised models

| Model | Val Accuracy | Val F1-macro | Val Log-loss | Status |
|-------|-------------|--------------|--------------|--------|
| Logistic Regression (baseline) | 0.4113 | 0.3262 | 1.0948 | ✅ Done |
| XGBoost + Optuna (87 features) | 0.3969 | 0.3667 | 1.0886 | ✅ Done |
| Random Forest (87 features) | 0.3846 | 0.3566 | 1.0895 | ✅ Done |
| XGBoost + Optuna (93 features) | 0.3981 | 0.3710 | 1.0871 | ✅ Done — selected |
| MLP (PyTorch) | — | — | — | ⏳ Pending |
| Stacking Ensemble | — | — | — | ⏳ Pending |

**Selected model:** XGBoost — best F1-macro and log-loss on validation set.

**Feature set (93):** Elo (4) + Neutral (1) + FIFA Rankings (3) + Form 5/10/20 (66) + H2H (11) + Squad Value (3) + Rest Days (2) + Match Importance (1) + Cluster (2)

**Top features by gain:** neutral, elo_diff, h2h_win_rate_a, win_prob_home, ranking_diff

### Unsupervised methods

| Method | Output | Status |
|--------|--------|--------|
| K-Means (k=4) | Cluster labels for 48 WC2026 teams | ✅ Done |
| PCA 2D | Visualization + variance analysis (80.4% in 2 components) | ✅ Done |
| Anomaly detection | Distance to centroid — 3 teams flagged | ✅ Done |

**Cluster results:**

| Cluster | Name | n | Avg Elo | Form WR |
|---------|------|---|---------|---------|
| Elite | Elite | 11 | 2014 | 0.72 |
| Consolidated Mid-Tier | Consolidated Mid-Tier | 20 | 1809 | 0.47 |
| Dynamic Mid-Tier | Dynamic Mid-Tier | 7 | 1924 | 0.70 |
| Underdogs | Underdogs | 10 | 1775 | 0.34 |

### Monte Carlo simulation

- 10,000 full tournament simulations
- WC2026 structure: Groups → R32 → R16 → QF → SF → 3rd place match → Final
- XGBoost probabilities sampled per match, penalty shootout on knockout draws
- Official FIFA bracket respected (stage-by-stage fixture from areezvisram12 dataset)
- Adaptive: re-run after each round with real results

---

## 🏆 Current Tournament Predictions (10,000 simulations)

| # | Team | P(R16) | P(QF) | P(SF) | P(Final) | P(Champion) |
|---|------|--------|-------|-------|----------|-------------|
| 1 | Croatia | 14.85% | 12.74% | 2.29% | 3.56% | 4.33% |
| 2 | Argentina | 17.58% | 9.56% | 1.96% | 3.41% | 4.28% |
| 3 | Spain | 19.27% | 9.91% | 2.26% | 3.24% | 3.69% |
| 4 | Uruguay | 18.60% | 10.04% | 2.19% | 2.65% | 3.48% |
| 5 | Switzerland | 24.51% | 9.22% | 2.31% | 2.54% | 3.36% |
| 6 | France | 15.79% | 11.76% | 3.47% | 3.07% | 3.35% |
| 7 | Colombia | 15.19% | 9.06% | 2.33% | 2.71% | 3.22% |
| 8 | Portugal | 11.75% | 8.11% | 1.82% | 2.37% | 3.18% |
| 9 | England | 15.53% | 8.88% | 2.21% | 2.61% | 2.86% |
| 10 | Belgium | 20.01% | 8.53% | 1.45% | 2.36% | 2.82% |

*Full results in `outputs/predictions/simulation_results.csv`*

---
## 🗞️ Expert Consensus Module

Automatically discovers, scrapes, and extracts structured predictions 
from football journalists and analysts across EN/ES/FR sources, 
then blends the resulting signal into the Monte Carlo simulation.

### How it works
1. `news_scraper.py` queries Google News with 17 queries across 3 languages,
   resolves tracking URLs to real article URLs via `googlenewsdecoder`
2. `scraper.py` fetches and cleans article text using newspaper4k
3. `extractor.py` calls Gemini 2.5-flash to extract structured predictions
   (team, prediction_type, confidence, sentiment, quote)
4. `pipeline.py` orchestrates the above, writes incrementally to CSV
   (safely resumable — already-processed URLs are skipped)
5. `analyze.py` compares expert consensus vs XGBoost model output
6. `simulate.py` blends both signals: `p_blended = (1-α) × p_model + α × p_expert`

### Current status
- **978 predictions** extracted from **39 sources**
- **48 teams** covered
- Spearman ρ = 0.63 (model vs expert consensus)

### Expert consensus top 5 (p_champion)
| Team | Expert | Model | Blended (α=0.3) |
|------|--------|-------|-----------------|
| Spain | 23.2% | 4.1% | 9.8% |
| France | 11.6% | 3.4% | 5.9% |
| Argentina | 12.1% | 4.1% | 6.5% |
| England | 10.1% | 3.0% | 5.1% |
| Brazil | 9.2% | 2.1% | 4.2% |

### Usage
    # Step 1 — populate urls.txt (run once, or to refresh)
    python -m world_cup_2026.expert_consensus.news_scraper

    # Step 2 — extract predictions (resumable, ~20 URLs/day on free Gemini tier)
    python -m world_cup_2026.expert_consensus.pipeline

    # Step 3 — compare expert vs model
    python -m world_cup_2026.expert_consensus.analyze

    # Step 4 — simulate with blend (alpha=0 for model-only)
    python -m world_cup_2026.simulation.simulate --alpha 0.3

---

## 📈 EDA Key Findings

| Finding | Value |
|---------|-------|
| Elo diff correlation with goal diff | 0.515 |
| Home advantage — all internationals | 49% HW / 23% D / 28% AW |
| Home advantage — World Cup neutral | 45.5% HW / 22% D / 32.3% AW |
| Away win rate trend 2021-2026 | 28% to 33% |
| Top Elo WC2026 team | Spain (2195) |
| Brazil current form last 10 | 0.50 win rate |
| England current form last 10 | 0.90 win rate |

---

## 🚀 Quickstart

    git clone https://github.com/federico1809/world-cup-2026-predictor.git
    cd world-cup-2026-predictor
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -m world_cup_2026.data_ingestion.download

### Run the full pipeline (no notebooks required)

    python -m world_cup_2026.features.build_features   # ~17 min — builds master_features.parquet
    python -m world_cup_2026.features.build_snapshot   # ~11 sec — builds team_snapshot_clustered.parquet
    python -m world_cup_2026.modeling.train             # ~5 min  — trains XGBoost model
    python -m world_cup_2026.simulation.simulate        # ~14 min — runs 10k Monte Carlo simulations + expert blend

### Run the dashboard

    streamlit run streamlit_app/app.py

Or visit the live version: [world-cup-2026-predictor-board.streamlit.app](https://world-cup-2026-predictor-board.streamlit.app)

---

## 🗂️ Development Status

| Phase | Status |
|-------|--------|
| Project scaffold | ✅ Done |
| Data ingestion pipeline | ✅ Done |
| Team name normalization 42/42 | ✅ Done |
| Exploratory Data Analysis | ✅ Done |
| Elo calculator | ✅ Done |
| H2H + transitive rival features | ✅ Done |
| Recent form features 5/10/20 | ✅ Done |
| FIFA Rankings feature join | ✅ Done |
| Squad market value features | ✅ Done |
| Rest days + match importance features | ✅ Done |
| Unsupervised clustering (k=4) | ✅ Done |
| Supervised modeling (XGBoost, 93 features) | ✅ Done |
| Monte Carlo simulation (10,000 runs) | ✅ Done |
| Streamlit dashboard (deployed) | ✅ Done |
| Notebook-free pipeline (build_features + build_snapshot) | ✅ Done |
| Expert consensus module (news scraper + extractor + pipeline) | ✅ Done |
| Expert consensus blend in simulation (--alpha param) | ✅ Done |
| MLP / Stacking Ensemble | ⏳ Pending |
| Mid-tournament retraining | ⏳ Pending |

---

## 👤 Author

Federico Ceballos Torres — Data Scientist
GitHub: https://github.com/federico1809

---

## 📄 License

MIT — see LICENSE for details.