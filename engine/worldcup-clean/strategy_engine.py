#!/usr/bin/env python3
"""
strategy_engine.py — 世界杯投注策略引擎 v2.0

功能:
  1. 5模型集成概率
  2. 从Pinnacle(最 sharp 的博彩公司)获取实时赔率
  3. 支持: 胜平负、让球(spreads)、大小球(totals)
  4. 串关(parlay)优化组合
  5. Kelly公式 + 风控

使用:
  python strategy_engine.py --bankroll 1000
  python strategy_engine.py --bookmaker Pinnacle
"""
import os, sys, json, math, argparse, itertools
from datetime import datetime

try:
    import requests
except ImportError:
    sys.exit("需要安装 requests: pip install requests")

# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════
API_KEY = os.environ.get("ODDS_API_KEY", "e8b2bcb052c1919d9a75e67368637dcd")
BASE_URL = "https://api.the-odds-api.com/v4"
BOOKMAKER = "Pinnacle"  # 最 sharp, 返还率最高

KELLY_FRACTION = 0.5
MAX_SINGLE_BET = 0.10
MAX_TOTAL_EXPOSURE = 0.30
MAX_PARLAY_BET = 0.05
MIN_EV = 0.01

# ══════════════════════════════════════════════════════════════════════════
# Elo 数据库
# ══════════════════════════════════════════════════════════════════════════
ELO_DB = {
    # == TOP TIER (1900+) ==
    "Spain": 2129, "Argentina": 2128, "France": 2084, "England": 2055,
    "Colombia": 1998, "Brazil": 1978, "Portugal": 1967, "Netherlands": 1944,
    "Germany": 1939, "Norway": 1929, "Japan": 1910,
    # == STRONG (1850-1900) ==
    "Mexico": 1896, "Ecuador": 1890, "Switzerland": 1885, "Croatia": 1881,
    "Belgium": 1879, "Uruguay": 1870, "Austria": 1857, "Turkey": 1849,
    "Türkiye": 1849, "Morocco": 1840, "Australia": 1839, "Senegal": 1839,
    # == MID-TIER (1750-1800) ==
    "Scotland": 1794, "Paraguay": 1780, "USA": 1780, "United States": 1780,
    "Canada": 1777, "South Korea": 1771, "Algeria": 1759, "Iran": 1756,
    "Sweden": 1755,
    # == LOWER MID (1700-1750) ==
    "Ivory Coast": 1743, "Egypt": 1711, "Uzbekistan": 1698,
    "Czech Republic": 1696, "Czechia": 1696,
    "Panama": 1683, "DR Congo": 1674, "Jordan": 1653,
    "Cape Verde": 1606, "Cabo Verde": 1606,
    "Saudi Arabia": 1598, "Bosnia and Herzegovina": 1596, "Bosnia": 1596,
    "Iraq": 1592, "Tunisia": 1585,
    # == WEAK (1500-1585) ==
    "New Zealand": 1578, "Ghana": 1557, "South Africa": 1527, "Haiti": 1536,
    # == BOTTOM (<1500) ==
    "Qatar": 1437, "Curacao": 1427, "Curaçao": 1427,
}

FED_MAP = {
    "Canada": "CONCACAF", "USA": "CONCACAF", "United States": "CONCACAF",
    "Mexico": "CONCACAF", "Jamaica": "CONCACAF", "Panama": "CONCACAF",
    "Haiti": "CONCACAF", "Honduras": "CONCACAF", "Costa Rica": "CONCACAF",
    "Bosnia": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Germany": "UEFA", "France": "UEFA", "England": "UEFA",
    "Spain": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA",
    "Belgium": "UEFA", "Italy": "UEFA", "Croatia": "UEFA",
    "Switzerland": "UEFA", "Denmark": "UEFA", "Poland": "UEFA",
    "Serbia": "UEFA", "Wales": "UEFA", "Czech Republic": "UEFA",
    "Scotland": "UEFA", "Albania": "UEFA", "Austria": "UEFA",
    "Hungary": "UEFA", "Turkey": "UEFA", "Romania": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Georgia": "UEFA",
    "Ukraine": "UEFA",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL", "Chile": "CONMEBOL",
    "Japan": "AFC", "South Korea": "AFC", "Korea Republic": "AFC",
    "Australia": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC", "Qatar": "AFC",
    "Jordan": "AFC",
    "New Zealand": "OFC",
    "Morocco": "CAF", "Senegal": "CAF", "Tunisia": "CAF",
    "Ghana": "CAF", "Cameroon": "CAF", "Nigeria": "CAF",
    "Egypt": "CAF", "Ivory Coast": "CAF", "South Africa": "CAF",
    "DR Congo": "CAF",
    "Iraq": "AFC",
    "Jordan": "AFC",
    "Uzbekistan": "AFC",
    "Norway": "UEFA",
    "Sweden": "UEFA",
    "Algeria": "CAF",
    "Cabo Verde": "CAF",
    "Curacao": "CONCACAF",
}


# ══════════════════════════════════════════════════════════════════════════
# 5模型
# ══════════════════════════════════════════════════════════════════════════
def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    p = math.exp(-lam)
    for i in range(1, k + 1): p *= lam / i
    return p

def dc_tau(a, b, lam, mu, rho):
    if a == 0 and b == 0: return 1 - lam * mu * rho
    if a == 0 and b == 1: return 1 + lam * rho
    if a == 1 and b == 0: return 1 + mu * rho
    if a == 1 and b == 1: return 1 - rho
    return 1

def model_hicruben(home_elo, away_elo):
    DC_RHO = -0.13
    def eg(r, o): return max(0.3, min(3.5, 1.35 + (r - o) / 400))
    lam = eg(home_elo, away_elo); mu = eg(away_elo, home_elo)
    w = d = l = 0; scores = {}
    for a in range(9):
        for b in range(9):
            tau = dc_tau(a, b, lam, mu, DC_RHO)
            p = poisson_pmf(a, lam) * poisson_pmf(b, mu) * tau
            scores[(a, b)] = p
            if a > b: w += p
            elif a < b: l += p
            else: d += p
    t = w + d + l
    return w/t, d/t, l/t, scores

def model_mikobinbin(home_elo, away_elo):
    adj_h = home_elo * 0.88 + 200; adj_a = away_elo * 0.88 + 200
    lam_h = max(0.3, min(3.5, 1.3 + (adj_h - 1700) / 500))
    lam_a = max(0.3, min(3.5, 1.3 + (adj_a - 1700) / 500))
    w = d = l = 0; scores = {}
    for a in range(9):
        for b in range(9):
            p = poisson_pmf(a, lam_h) * poisson_pmf(b, lam_a)
            scores[(a, b)] = p
            if a > b: w += p
            elif a < b: l += p
            else: d += p
    t = w + d + l
    return w/t, d/t, l/t, scores

def model_andydu(home_elo, away_elo):
    diff = (home_elo - away_elo) / 400
    hw = 1 / (1 + 10 ** (-diff))
    draw_base = 0.25 * math.exp(-abs(diff) * 2)
    draw = max(0.08, min(0.30, draw_base))
    rem = 1 - draw; e_hw = hw * rem; e_aw = (1 - hw) * rem
    lam_h = max(0.3, min(3.5, 1.5 * (10 ** (diff / 2)) * 0.85))
    lam_a = max(0.3, min(3.5, 1.5 * (10 ** (-diff / 2)) * 0.85))
    dc_w = dc_d = dc_l = 0; scores = {}
    for a in range(9):
        for b in range(9):
            p = poisson_pmf(a, lam_h) * poisson_pmf(b, lam_a)
            scores[(a, b)] = p
            if a > b: dc_w += p
            elif a < b: dc_l += p
            else: dc_d += p
    t = dc_w + dc_d + dc_l; dc_w /= t; dc_d /= t; dc_l /= t
    fw = (e_hw + dc_w) / 2; fd = (draw + dc_d) / 2; fl = (e_aw + dc_l) / 2
    t2 = fw + fd + fl
    return fw/t2, fd/t2, fl/t2, scores

def predict_amir42(home_elo, away_elo, h_conf, a_conf):
    import joblib, pandas as pd
    obj = joblib.load(os.path.join(os.path.dirname(__file__), "amir42-predictor", "models", "xgb_wc2026.joblib"))
    xgb = obj["model"]; fcols = obj["feature_cols"]
    elo_diff = home_elo - away_elo
    feat = {
        "home_elo": home_elo, "away_elo": away_elo, "elo_diff": elo_diff,
        "home_win_rate_5": 0.55 if elo_diff > 0 else 0.45,
        "away_win_rate_5": 0.45 if elo_diff > 0 else 0.55,
        "home_gd_5": elo_diff / 200, "away_gd_5": -elo_diff / 200,
        "home_win_rate_10": 0.55 if elo_diff > 0 else 0.45,
        "away_win_rate_10": 0.45 if elo_diff > 0 else 0.55,
        "home_gd_10": elo_diff / 200, "away_gd_10": -elo_diff / 200,
        "h2h_n": 0, "h2h_home_wr": 0.5,
        "home_conf_elo": 1700, "away_conf_elo": 1700,
        "neutral": 1, "is_world_cup": 1,
    }
    X = pd.DataFrame([feat])
    for c in fcols:
        if c not in X.columns: X[c] = 0
    h_key = f"h_conf_{h_conf}"; a_key = f"a_conf_{a_conf}"
    if h_key in fcols: X[h_key] = 1
    if a_key in fcols: X[a_key] = 1
    X = X[fcols]
    probs = xgb.predict_proba(X)[0]
    return float(probs[0]), float(probs[1]), float(probs[2])

def predict_federico1809(home_elo, away_elo, home_team, away_team):
    import joblib, numpy as np, pandas as pd
    base = os.path.join(os.path.dirname(__file__), "federico1809-predictor")
    model = joblib.load(os.path.join(base, "models", "xgb_match_predictor.pkl"))
    with open(os.path.join(base, "models", "model_features.json")) as f: features = json.load(f)
    snapshot = pd.read_parquet(os.path.join(base, "data", "processed", "team_snapshot_clustered.parquet"))
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(["Consolidated Mid-Tier", "Dynamic Mid-Tier", "Elite", "Non-WC2026", "Underdogs"])
    elo_diff = home_elo - away_elo
    win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    def find(name):
        m = snapshot[snapshot["team"].str.contains(name, case=False, na=False)]
        return m.iloc[0] if len(m) > 0 else None
    h = find(home_team); a = find(away_team)
    def est_rank(elo):
        if elo >= 2000: return 1
        if elo >= 1900: return 5
        if elo >= 1800: return 15
        if elo >= 1700: return 30
        return 50
    feat = {
        "elo_pre_home": home_elo, "elo_pre_away": away_elo,
        "elo_diff": elo_diff, "win_prob_home": win_prob,
        "neutral": 1.0, "ranking_home": est_rank(home_elo), "ranking_away": est_rank(away_elo),
        "ranking_diff": est_rank(home_elo) - est_rank(away_elo),
        "squad_value_home": 500.0, "squad_value_away": 300.0, "squad_value_diff": 200.0,
        "rest_days_home": 30.0, "rest_days_away": 30.0, "match_importance": 3.0,
    }
    for stat in ["win_rate", "draw_rate", "loss_rate", "goals_scored_avg", "goals_conceded_avg",
                  "goal_diff_avg", "clean_sheet_rate", "failed_score_rate", "points_avg",
                  "matches_played", "weighted_points"]:
        for w in [5, 10, 20]:
            feat[f"home_form_{w}_{stat}"] = float(h[f"form_{w}_{stat}"]) if h is not None else 0.5
            feat[f"away_form_{w}_{stat}"] = float(a[f"form_{w}_{stat}"]) if a is not None else 0.5
    cn_h = h.get("cluster_name", "Dynamic Mid-Tier") if h is not None else "Dynamic Mid-Tier"
    cn_a = a.get("cluster_name", "Dynamic Mid-Tier") if a is not None else "Dynamic Mid-Tier"
    feat["home_cluster_enc"] = float(le.transform([cn_h])[0])
    feat["away_cluster_enc"] = float(le.transform([cn_a])[0])
    for c in ["h2h_matches", "h2h_win_rate_a", "h2h_goal_diff_a", "h2h_elo_edge_a",
              "h2h_weighted_edge_a", "h2h_decay_weight", "h2h_reliable",
              "transitive_common_rivals", "transitive_edge_a", "transitive_goal_diff_edge",
              "transitive_reliable"]:
        feat[c] = 0.0
    X1 = np.array([feat[f] for f in features], dtype=float).reshape(1, -1)
    p1 = model.predict_proba(X1)[0]
    feat2 = dict(feat)
    feat2["elo_pre_home"], feat2["elo_pre_away"] = away_elo, home_elo
    feat2["elo_diff"] = -elo_diff; feat2["win_prob_home"] = 1 - win_prob
    X2 = np.array([feat2[f] for f in features], dtype=float).reshape(1, -1)
    p2 = model.predict_proba(X2)[0]
    p_home = (float(p1[2]) + float(p2[0])) / 2
    p_draw = (float(p1[1]) + float(p2[1])) / 2
    p_away = (float(p1[0]) + float(p2[2])) / 2
    return p_home, p_draw, p_away


def get_ensemble(home_team, away_team):
    """运行5模型, 返回集成概率 + 比分分布"""
    h_elo = ELO_DB.get(home_team, 1700)
    a_elo = ELO_DB.get(away_team, 1700)
    h_conf = FED_MAP.get(home_team, "UEFA")
    a_conf = FED_MAP.get(away_team, "UEFA")

    results = {}  # name -> (hw, d, aw, scores)

    hw, d, aw, sc = model_hicruben(h_elo, a_elo)
    results["Hicruben"] = (hw, d, aw, sc)

    hw, d, aw, sc = model_mikobinbin(h_elo, a_elo)
    results["mikobinbin"] = (hw, d, aw, sc)

    hw, d, aw, sc = model_andydu(h_elo, a_elo)
    results["AndyDu"] = (hw, d, aw, sc)

    try:
        hw, d, aw = predict_amir42(h_elo, a_elo, h_conf, a_conf)
        # amir42没有scores分布, 用Elo差估算
        diff = (h_elo - a_elo) / 400
        xgh = max(0.3, min(3.5, 1.35 + diff * 0.8))
        xga = max(0.3, min(3.5, 1.35 - diff * 0.8))
        sc = {}
        for a2 in range(9):
            for b2 in range(9):
                sc[(a2, b2)] = poisson_pmf(a2, xgh) * poisson_pmf(b2, xga)
        t = sum(sc.values())
        sc = {k: v/t for k, v in sc.items()}
        results["amir42"] = (hw, d, aw, sc)
    except Exception as e:
        print(f"  [amir42] {e}, 使用平均值")
        n = len(results)
        avg_hw = sum(r[0] for r in results.values()) / n
        avg_d = sum(r[1] for r in results.values()) / n
        avg_aw = sum(r[2] for r in results.values()) / n
        avg_sc = {}
        for k2 in range(9):
            for b2 in range(9):
                avg_sc[(k2, b2)] = sum(r[3].get((k2, b2), 0) for r in results.values()) / n
        results["amir42"] = (avg_hw, avg_d, avg_aw, avg_sc)

    try:
        hw, d, aw = predict_federico1809(h_elo, a_elo, home_team, away_team)
        diff = (h_elo - a_elo) / 400
        xgh = max(0.3, min(3.5, 1.35 + diff * 0.8))
        xga = max(0.3, min(3.5, 1.35 - diff * 0.8))
        sc = {}
        for a2 in range(9):
            for b2 in range(9):
                sc[(a2, b2)] = poisson_pmf(a2, xgh) * poisson_pmf(b2, xga)
        t = sum(sc.values())
        sc = {k: v/t for k, v in sc.items()}
        results["federico1809"] = (hw, d, aw, sc)
    except Exception as e:
        print(f"  [federico1809] {e}, 使用平均值")
        n = len(results)
        avg_hw = sum(r[0] for r in results.values()) / n
        avg_d = sum(r[1] for r in results.values()) / n
        avg_aw = sum(r[2] for r in results.values()) / n
        avg_sc = {}
        for k2 in range(9):
            for b2 in range(9):
                avg_sc[(k2, b2)] = sum(r[3].get((k2, b2), 0) for r in results.values()) / n
        results["federico1809"] = (avg_hw, avg_d, avg_aw, avg_sc)

    # 集成: 平均概率 + 平均比分分布
    n = len(results)
    ens_hw = sum(r[0] for r in results.values()) / n
    ens_d = sum(r[1] for r in results.values()) / n
    ens_aw = sum(r[2] for r in results.values()) / n
    ens_sc = {}
    for k2 in range(9):
        for b2 in range(9):
            ens_sc[(k2, b2)] = sum(r[3].get((k2, b2), 0) for r in results.values()) / n
    # 归一化
    sc_total = sum(ens_sc.values())
    ens_sc = {k: v/sc_total for k, v in ens_sc.items()}

    return {
        "hw": ens_hw, "d": ens_d, "aw": ens_aw,
        "scores": ens_sc,
        "details": {name: {"hw": r[0], "d": r[1], "aw": r[2]} for name, r in results.items()},
        "elo": {"home": h_elo, "away": a_elo},
    }


# ══════════════════════════════════════════════════════════════════════════
# 赔率获取 (指定单一博彩公司)
# ══════════════════════════════════════════════════════════════════════════
def fetch_match_odds(home_team, away_team, bookmaker=BOOKMAKER):
    """获取指定比赛在指定博彩公司的 h2h + spreads + totals"""
    url = f"{BASE_URL}/sports/soccer_fifa_world_cup/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us,uk,eu",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  API剩余: {remaining} 次")
        events = resp.json()
        for event in events:
            h = event.get("home_team", "")
            a = event.get("away_team", "")
            if _match(home_team, h) and _match(away_team, a):
                return _parse_event(event, bookmaker, reversed_match=False)
            if _match(home_team, a) and _match(away_team, h):
                return _parse_event(event, bookmaker, reversed_match=True)
    except Exception as e:
        print(f"  [API] {e}")
    return None


def _match(kw, full):
    kw2 = kw.lower().replace("&", "and").replace("and", " ")
    full2 = full.lower().replace("&", "and").replace("and", " ")
    # 检查主要关键词 (第一个单词)
    kw_first = kw.split()[0].lower()
    return kw_first in full.lower() or kw.lower() in full.lower() or full.lower() in kw.lower()


def _parse_event(event, bookmaker, reversed_match):
    for bm in event.get("bookmakers", []):
        if bm["title"].lower().startswith(bookmaker.lower()):
            result = {"h2h": {}, "spreads": [], "totals": []}
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for o in market["outcomes"]:
                        name = o["name"]
                        if name == event.get("home_team"):
                            key = "away" if reversed_match else "home"
                        elif name == event.get("away_team"):
                            key = "home" if reversed_match else "away"
                        elif name == "Draw":
                            key = "draw"
                        else:
                            continue
                        result["h2h"][key] = o["price"]
                elif market["key"] == "spreads":
                    for o in market["outcomes"]:
                        name = o["name"]
                        point = o["point"]
                        if reversed_match:
                            point = -point
                            if name == event.get("home_team"):
                                name = event.get("away_team")
                            elif name == event.get("away_team"):
                                name = event.get("home_team")
                        result["spreads"].append({
                            "team": name, "point": point, "price": o["price"]
                        })
                elif market["key"] == "totals":
                    for o in market["outcomes"]:
                        result["totals"].append({
                            "name": o["name"], "point": o["point"], "price": o["price"]
                        })
            return result
    return None


# ══════════════════════════════════════════════════════════════════════════
# 概率计算 (从比分分布推导各种市场)
# ══════════════════════════════════════════════════════════════════════════
def calc_spread_prob(scores, handicap, team="home"):
    """计算让球概率
    handicap > 0: 主队让球 (如 -0.5 = 主队让半球)
    handicap < 0: 客队让球
    team: "home" or "away"
    """
    prob = 0
    for (hg, ag), p in scores.items():
        if team == "home":
            effective_diff = hg - ag + handicap  # handicap为负=让球
        else:
            effective_diff = ag - hg + handicap
        if effective_diff > 0:
            prob += p
        elif effective_diff == 0:
            prob += p * 0.5  # 走盘退款, 算半赢
    return prob


def calc_total_prob(scores, line, over=True):
    """计算大小球概率"""
    prob = 0
    for (hg, ag), p in scores.items():
        total = hg + ag
        if over:
            if total > line:
                prob += p
            elif total == line and line != int(line):
                pass  # 走盘
            elif total == line and line == int(line):
                prob += p * 0.5  # 整数盘走盘
        else:
            if total < line:
                prob += p
            elif total == line and line == int(line):
                prob += p * 0.5
    return prob


def kelly(prob, odds):
    if odds <= 1: return 0
    f = (prob * odds - 1) / (odds - 1)
    return max(0, f)


def ev(prob, odds):
    return prob * odds - 1


# ══════════════════════════════════════════════════════════════════════════
# 分析引擎
# ══════════════════════════════════════════════════════════════════════════
def analyze_match(match_name, ensemble, odds):
    """分析单场比赛所有市场, 返回所有正EV投注"""
    if not odds:
        print(f"\n  [!] {match_name}: 未获取到赔率")
        return []

    bets = []
    hw, d, aw = ensemble["hw"], ensemble["d"], ensemble["aw"]
    scores = ensemble["scores"]
    home_team = match_name.split(" vs ")[0].strip()

    # ── 1. 胜平负 ────────────────────────────────────────────────────────
    print(f"\n  +-- 胜平负 --------------------------------------------------------------------+")
    print(f"  | {'结果':<10} {'赔率':>8} {'隐含':>8} {'模型':>8} {'EV':>10} {'Kelly':>8} |")
    print(f"  |----------------------------------------------------------------------|")
    for label, prob_key, odds_key in [("主胜", "home", "home"), ("平局", "draw", "draw"), ("客胜", "away", "away")]:
        odds_val = odds["h2h"].get(odds_key, 0)
        if odds_val <= 0: continue
        model_p = {"home": hw, "draw": d, "away": aw}[odds_key]
        implied = 1 / odds_val
        ev_val = ev(model_p, odds_val)
        kf = kelly(model_p, odds_val)
        marker = " <<<" if ev_val > MIN_EV else ""
        print(f"  | {label:<10} {odds_val:>8.2f} {implied:>7.1%} {model_p:>7.1%} {ev_val:>+9.1%}{marker} {kf:>7.1%} |")
        if ev_val > MIN_EV:
            bets.append({"type": "胜平负", "match": match_name, "selection": label,
                         "odds": odds_val, "prob": model_p, "ev": ev_val, "kelly": kf})
    print(f"  +----------------------------------------------------------------------+")

    # ── 2. 让球 (spreads) ────────────────────────────────────────────────
    if odds.get("spreads"):
        print(f"\n  +-- 让球 ----------------------------------------------------------------------+")
        print(f"  | {'盘口':<16} {'赔率':>8} {'隐含':>8} {'模型':>8} {'EV':>10} {'Kelly':>8} |")
        print(f"  |----------------------------------------------------------------------|")
        for sp in odds["spreads"]:
            team = sp["team"]
            point = sp["point"]
            price = sp["price"]
            if price <= 0: continue

            is_home = _match(home_team, team)
            spread_team = "home" if is_home else "away"
            handicap = point  # point为负=让球

            model_p = calc_spread_prob(scores, handicap, spread_team)
            implied = 1 / price
            ev_val = ev(model_p, price)
            kf = kelly(model_p, price)

            # 盘口描述
            if is_home:
                if handicap < 0:
                    desc = f"主队{handicap}"
                elif handicap > 0:
                    desc = f"主队+{handicap}"
                else:
                    desc = "主队 0"
            else:
                if handicap < 0:
                    desc = f"客队{handicap}"
                elif handicap > 0:
                    desc = f"客队+{handicap}"
                else:
                    desc = "客队 0"

            marker = " <<<" if ev_val > MIN_EV else ""
            print(f"  | {desc:<16} {price:>8.2f} {implied:>7.1%} {model_p:>7.1%} {ev_val:>+9.1%}{marker} {kf:>7.1%} |")
            if ev_val > MIN_EV:
                bets.append({"type": "让球", "match": match_name, "selection": desc,
                             "odds": price, "prob": model_p, "ev": ev_val, "kelly": kf})
        print(f"  +----------------------------------------------------------------------+")

    # ── 3. 大小球 (totals) ───────────────────────────────────────────────
    if odds.get("totals"):
        print(f"\n  +-- 大小球 --------------------------------------------------------------------+")
        print(f"  | {'盘口':<16} {'赔率':>8} {'隐含':>8} {'模型':>8} {'EV':>10} {'Kelly':>8} |")
        print(f"  |----------------------------------------------------------------------|")
        for tot in odds["totals"]:
            name = tot["name"]  # "Over" or "Under"
            point = tot["point"]
            price = tot["price"]
            if price <= 0: continue

            is_over = name == "Over"
            model_p = calc_total_prob(scores, point, over=is_over)
            implied = 1 / price
            ev_val = ev(model_p, price)
            kf = kelly(model_p, price)

            desc = f"{'大' if is_over else '小'}{point}球"
            marker = " <<<" if ev_val > MIN_EV else ""
            print(f"  | {desc:<16} {price:>8.2f} {implied:>7.1%} {model_p:>7.1%} {ev_val:>+9.1%}{marker} {kf:>7.1%} |")
            if ev_val > MIN_EV:
                bets.append({"type": "大小球", "match": match_name, "selection": desc,
                             "odds": price, "prob": model_p, "ev": ev_val, "kelly": kf})
        print(f"  +----------------------------------------------------------------------+")

    # ── 模型共识 ─────────────────────────────────────────────────────────
    print(f"\n  +-- 5模型共识 ------------------------------------------------------------------+")
    print(f"  | {'模型':<16} {'主胜':>8} {'平局':>8} {'客胜':>8} {'预测':>6} |")
    print(f"  |----------------------------------------------------------------------|")
    labels = {"hw": "主胜", "d": "平局", "aw": "客胜"}
    for name, probs in ensemble["details"].items():
        pred = max(probs, key=probs.get)
        print(f"  | {name:<16} {probs['hw']:>7.1%} {probs['d']:>7.1%} {probs['aw']:>7.1%} {labels[pred]:>6} |")
    print(f"  |----------------------------------------------------------------------|")
    print(f"  | {'集成(平均)':<16} {hw:>7.1%} {d:>7.1%} {aw:>7.1%}        |")
    print(f"  +----------------------------------------------------------------------+")

    return bets


def build_parlays(single_bets, max_legs=3):
    """构建串关组合: 从正EV单注中选2-3个组合"""
    if len(single_bets) < 2:
        return []

    parlays = []

    # 2串1
    for combo in itertools.combinations(single_bets, 2):
        # 不同比赛才能串
        if combo[0]["match"] == combo[1]["match"]:
            # 同一比赛的不同市场可以串
            if combo[0]["type"] == combo[1]["type"]:
                continue  # 同类型不串
        combined_odds = combo[0]["odds"] * combo[1]["odds"]
        combined_prob = combo[0]["prob"] * combo[1]["prob"]
        parlay_ev = ev(comblay_odds if (comblay_ev := ev(combined_prob, combined_odds)) > 0 else 0, combined_odds)
        # 修正: 直接算
        parlay_ev = combined_prob * combined_odds - 1
        if parlay_ev > 0:
            kf = kelly(combined_prob, combined_odds)
            parlays.append({
                "type": "2串1",
                "legs": combo,
                "odds": combined_odds,
                "prob": combined_prob,
                "ev": parlay_ev,
                "kelly": kf,
            })

    # 3串1
    if len(single_bets) >= 3:
        for combo in itertools.combinations(single_bets, 3):
            combined_odds = combo[0]["odds"] * combo[1]["odds"] * combo[2]["odds"]
            combined_prob = combo[0]["prob"] * combo[1]["prob"] * combo[2]["prob"]
            parlay_ev = combined_prob * combined_odds - 1
            if parlay_ev > 0:
                kf = kelly(combined_prob, combined_odds)
                parlays.append({
                    "type": "3串1",
                    "legs": combo,
                    "odds": combined_odds,
                    "prob": combined_prob,
                    "ev": parlay_ev,
                    "kelly": kf,
                })

    # 按 EV 排序
    parlays.sort(key=lambda x: x["ev"], reverse=True)
    return parlays


def optimize_portfolio(all_bets, all_parlays, bankroll):
    """组合优化: 单注 + 串关"""
    portfolio = []

    # 单注: Half-Kelly, 按 EV 排序
    singles = sorted([b for b in all_bets], key=lambda x: x["ev"], reverse=True)
    for b in singles:
        frac = min(b["kelly"] * KELLY_FRACTION, MAX_SINGLE_BET)
        portfolio.append({**b, "fraction": frac, "category": "single"})

    # 串关: 按 EV 排序, 用更小的 Kelly
    parlays = sorted(all_parlays, key=lambda x: x["ev"], reverse=True)[:5]  # 最多5个串关
    for p in parlays:
        # 串关用 1/4 Kelly (风险更高)
        frac = min(p["kelly"] * 0.25, MAX_PARLAY_BET)
        legs_desc = " + ".join(f"{l['selection']}@{l['odds']:.2f}" for l in p["legs"])
        portfolio.append({
            "type": p["type"], "match": p["type"],
            "selection": legs_desc,
            "odds": p["odds"], "prob": p["prob"],
            "ev": p["ev"], "kelly": p["kelly"],
            "fraction": frac, "category": "parlay",
            "legs": p["legs"],
        })

    # 总暴露检查
    total = sum(item["fraction"] for item in portfolio)
    if total > MAX_TOTAL_EXPOSURE:
        scale = MAX_TOTAL_EXPOSURE / total
        for item in portfolio:
            item["fraction"] *= scale

    # 计算金额
    for item in portfolio:
        item["amount"] = round(bankroll * item["fraction"], 2)
        item["potential_win"] = round(item["amount"] * item["odds"], 2)
        item["expected_profit"] = round(item["amount"] * item["ev"], 2)

    return portfolio


def print_portfolio(portfolio, bankroll):
    """打印投资组合"""
    if not portfolio:
        print(f"\n  当前无正EV投注机会")
        return

    singles = [p for p in portfolio if p["category"] == "single"]
    parlays = [p for p in portfolio if p["category"] == "parlay"]

    print(f"\n{'=' * 100}")
    print(f"  最优投资组合  |  总资金: {bankroll}元  |  Pinnacle 赔率")
    print(f"{'=' * 100}")

    if singles:
        print(f"\n  +-- 单注 ----------------------------------------------------------------------+")
        print(f"  | {'类型':<8} {'比赛/选择':<28} {'赔率':>7} {'模型':>7} {'EV':>8} {'比例':>7} {'金额':>7} {'赢则得':>8} |")
        print(f"  |----------------------------------------------------------------------|")
        for b in singles:
            desc = f"{b['match'][:18]} {b['selection']}"
            print(f"  | {b['type']:<8} {desc:<28} {b['odds']:>7.2f} {b['prob']:>6.1%} {b['ev']:>+7.1%} {b['fraction']:>6.1%} {b['amount']:>6.0f}元 {b['potential_win']:>7.0f}元 |")
        print(f"  +----------------------------------------------------------------------+")

    if parlays:
        print(f"\n  +-- 串关 ----------------------------------------------------------------------+")
        print(f"  | {'类型':<8} {'组合':<40} {'赔率':>7} {'概率':>7} {'EV':>8} {'比例':>7} {'金额':>7} {'赢则得':>8} |")
        print(f"  |----------------------------------------------------------------------|")
        for p in parlays:
            desc = p["selection"][:40]
            print(f"  | {p['type']:<8} {desc:<40} {p['odds']:>7.2f} {p['prob']:>6.1%} {p['ev']:>+7.1%} {p['fraction']:>6.1%} {p['amount']:>6.0f}元 {p['potential_win']:>7.0f}元 |")
        print(f"  +----------------------------------------------------------------------+")

    # 汇总
    total_amount = sum(p["amount"] for p in portfolio)
    total_expected = sum(p["expected_profit"] for p in portfolio)
    print(f"\n  +-- 汇总 ----------------------------------------------------------------------+")
    print(f"  |  总投注: {total_amount:.0f}元 ({total_amount/bankroll:.1%} of bankroll)                                        |")
    print(f"  |  预期收益: {total_expected:+.1f}元 (ROI: {total_expected/max(total_amount,1):+.1%})                                      |")
    worst = -total_amount
    best = sum(p["potential_win"] for p in portfolio) - total_amount
    print(f"  |  最坏: {worst:.0f}元  |  最好: +{best:.0f}元                                              |")
    print(f"  +----------------------------------------------------------------------+")

    # 具体购买清单
    print(f"\n  +-- 购买清单 ------------------------------------------------------------------+")
    for p in portfolio:
        if p["category"] == "single":
            print(f"  |  [{p['type']}] {p['match']} → {p['selection']} @ {p['odds']:.2f} → 投 {p['amount']:.0f}元 |")
        else:
            legs_str = " + ".join(f"{l['selection']}" for l in p["legs"])
            print(f"  |  [{p['type']}] {legs_str} @ {p['odds']:.2f} → 投 {p['amount']:.0f}元 |")
    print(f"  +----------------------------------------------------------------------+")


# ══════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bankroll", type=float, default=1000)
    parser.add_argument("--bookmaker", type=str, default=BOOKMAKER)
    parser.add_argument("--matches", nargs="+")
    args = parser.parse_args()

    bankroll = args.bankroll
    bookmaker = args.bookmaker

    if args.matches:
        matches = [tuple(m.split(",")) for m in args.matches]
    else:
        matches = [
            ("Canada", "Bosnia and Herzegovina"),
            ("USA", "Paraguay"),
        ]

    print()
    print("=" * 100)
    print(f"  世界杯投注策略引擎 v2.0  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  总资金: {bankroll}元  |  博彩公司: {bookmaker}  |  策略: 5模型 + Kelly + 串关")
    print("=" * 100)

    all_bets = []

    for home, away in matches:
        match_name = f"{home} vs {away}"
        print(f"\n{'━' * 100}")
        print(f"  分析: {match_name}")
        print(f"{'━' * 100}")

        # 1. 模型
        print(f"  运行5模型...")
        ensemble = get_ensemble(home, away)

        # 2. 赔率
        print(f"  获取{bookmaker}赔率...")
        odds = fetch_match_odds(home, away, bookmaker)

        # 3. 分析
        bets = analyze_match(match_name, ensemble, odds)
        all_bets.extend(bets)

    # 4. 串关
    print(f"\n{'━' * 100}")
    print(f"  构建串关组合...")
    print(f"{'━' * 100}")
    parlays = build_parlays(all_bets)
    if parlays:
        print(f"  发现 {len(parlays)} 个正EV串关组合 (显示前5):")
        for i, p in enumerate(parlays[:5]):
            legs_str = " + ".join(f"{l['selection']}@{l['odds']:.2f}" for l in p["legs"])
            print(f"    {i+1}. [{p['type']}] {legs_str}")
            print(f"       综合赔率: {p['odds']:.2f}  概率: {p['prob']:.1%}  EV: {p['ev']:+.1%}")
    else:
        print(f"  未发现正EV串关组合")

    # 5. 组合优化
    portfolio = optimize_portfolio(all_bets, parlays, bankroll)
    print_portfolio(portfolio, bankroll)

    # 6. 策略说明
    print(f"""
{'=' * 100}
  策略说明
{'=' * 100}
  单注: Half-Kelly, 上限10%/注
  串关: 1/4 Kelly (风险更高), 上限5%/注
  总暴露: 不超过30%
  选择标准: EV > {MIN_EV:.0%} 的正期望值投注
  赔率源: {bookmaker} (最sharp的博彩公司, 返还率~97-98%)
""")
    return


# 修复 build_parlays 中的语法错误
def build_parlays(single_bets, max_legs=3):
    """构建串关组合"""
    if len(single_bets) < 2:
        return []

    parlays = []

    # 2串1
    for combo in itertools.combinations(single_bets, 2):
        if combo[0]["match"] == combo[1]["match"] and combo[0]["type"] == combo[1]["type"]:
            continue
        combined_odds = combo[0]["odds"] * combo[1]["odds"]
        combined_prob = combo[0]["prob"] * combo[1]["prob"]
        parlay_ev = combined_prob * combined_odds - 1
        if parlay_ev > 0:
            kf = kelly(combined_prob, combined_odds)
            parlays.append({
                "type": "2串1", "legs": combo,
                "odds": combined_odds, "prob": combined_prob,
                "ev": parlay_ev, "kelly": kf,
            })

    # 3串1
    if len(single_bets) >= 3:
        for combo in itertools.combinations(single_bets, 3):
            combined_odds = combo[0]["odds"] * combo[1]["odds"] * combo[2]["odds"]
            combined_prob = combo[0]["prob"] * combo[1]["prob"] * combo[2]["prob"]
            parlay_ev = combined_prob * combined_odds - 1
            if parlay_ev > 0:
                kf = kelly(combined_prob, combined_odds)
                parlays.append({
                    "type": "3串1", "legs": combo,
                    "odds": combined_odds, "prob": combined_prob,
                    "ev": parlay_ev, "kelly": kf,
                })

    parlays.sort(key=lambda x: x["ev"], reverse=True)
    return parlays


if __name__ == "__main__":
    main()
