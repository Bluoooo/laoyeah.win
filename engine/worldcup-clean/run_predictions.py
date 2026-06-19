#!/usr/bin/env python3
"""Run 5 models for today's matches and output in template format."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from strategy_engine import model_hicruben, model_mikobinbin, model_andydu, predict_amir42, predict_federico1809
from strategy_engine import poisson_pmf, FED_MAP

# Calibrated Elo from elo-calibrated.json
ELO = {
    "Argentina": 1976, "Algeria": 1704,
    "France": 2009, "Senegal": 1848,
    "Iraq": 1599, "Norway": 1880,
}

# Also map short names for federico1809 lookup
ELO_LOWER = {k.lower(): v for k, v in ELO.items()}

def score_dist_from_elo_diff(h_elo, a_elo):
    """Estimate score distribution from Elo diff (for models that don't produce scores)."""
    diff = (h_elo - a_elo) / 400
    xgh = max(0.3, min(3.5, 1.35 + diff * 0.8))
    xga = max(0.3, min(3.5, 1.35 - diff * 0.8))
    sc = {}
    for a in range(9):
        for b in range(9):
            sc[(a, b)] = poisson_pmf(a, xgh) * poisson_pmf(b, xga)
    t = sum(sc.values())
    return {k: v/t for k, v in sc.items()}


def run_match(home, away):
    h_elo = ELO[home]
    a_elo = ELO[away]
    h_conf = FED_MAP.get(home, "UEFA")
    a_conf = FED_MAP.get(away, "UEFA")

    results = {}

    # Hicruben
    hw, d, aw, sc = model_hicruben(h_elo, a_elo)
    results["Hicruben"] = {"hw": hw, "d": d, "aw": aw, "scores": sc}

    # mikobinbin
    hw, d, aw, sc = model_mikobinbin(h_elo, a_elo)
    results["mikobinbin"] = {"hw": hw, "d": d, "aw": aw, "scores": sc}

    # AndyDu
    hw, d, aw, sc = model_andydu(h_elo, a_elo)
    results["AndyDu"] = {"hw": hw, "d": d, "aw": aw, "scores": sc}

    # amir42
    try:
        hw, d, aw = predict_amir42(h_elo, a_elo, h_conf, a_conf)
        sc = score_dist_from_elo_diff(h_elo, a_elo)
        results["amir42"] = {"hw": hw, "d": d, "aw": aw, "scores": sc}
    except Exception as e:
        print(f"  [amir42 error: {e}, using average]")
        n = len(results)
        avg_hw = sum(r["hw"] for r in results.values()) / n
        avg_d = sum(r["d"] for r in results.values()) / n
        avg_aw = sum(r["aw"] for r in results.values()) / n
        results["amir42"] = {"hw": avg_hw, "d": avg_d, "aw": avg_aw, "scores": score_dist_from_elo_diff(h_elo, a_elo)}

    # federico1809
    try:
        hw, d, aw = predict_federico1809(h_elo, a_elo, home, away)
        sc = score_dist_from_elo_diff(h_elo, a_elo)
        results["federico1809"] = {"hw": hw, "d": d, "aw": aw, "scores": sc}
    except Exception as e:
        print(f"  [federico1809 error: {e}, using average]")
        n = len(results)
        avg_hw = sum(r["hw"] for r in results.values()) / n
        avg_d = sum(r["d"] for r in results.values()) / n
        avg_aw = sum(r["aw"] for r in results.values()) / n
        results["federico1809"] = {"hw": avg_hw, "d": avg_d, "aw": avg_aw, "scores": score_dist_from_elo_diff(h_elo, a_elo)}

    return results


def top_scores(results, n=10):
    """Get top N scores by average probability across all models."""
    all_scores = {}
    model_order = ["Hicruben", "mikobinbin", "AndyDu", "amir42", "federico1809"]

    # Collect all score keys
    all_keys = set()
    for name in model_order:
        if name in results:
            all_keys.update(results[name]["scores"].keys())

    # Calculate average for each score
    for key in all_keys:
        vals = []
        for name in model_order:
            if name in results:
                vals.append(results[name]["scores"].get(key, 0))
        all_scores[key] = sum(vals) / len(vals)

    # Sort by average probability descending
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:n]


def print_report(home, away, results):
    model_order_wdl = ["AndyDu", "amir42", "Hicruben", "mikobinbin", "federico1809"]
    model_order_score = ["Hicruben", "mikobinbin", "AndyDu", "amir42", "federico1809"]

    print(f"\n## {home} vs {away}")
    print(f"Elo: {home} {ELO[home]} / {away} {ELO[away]}  (差值: {ELO[home] - ELO[away]})")

    # 胜平负概率表
    print(f"\n### 胜平负概率表")
    print(f"| 模型 | {home}胜 | 平局 | {away}胜 |")
    print(f"| :--- | :--- | :--- | :--- |")

    avg_hw = avg_d = avg_aw = 0
    count = 0
    for name in model_order_wdl:
        if name in results:
            r = results[name]
            print(f"| {name} | {r['hw']*100:.1f}% | {r['d']*100:.1f}% | {r['aw']*100:.1f}% |")
            avg_hw += r['hw']
            avg_d += r['d']
            avg_aw += r['aw']
            count += 1

    avg_hw /= count
    avg_d /= count
    avg_aw /= count
    print(f"| **5模型平均** | **{avg_hw*100:.1f}%** | **{avg_d*100:.1f}%** | **{avg_aw*100:.1f}%** |")

    # 比分概率表
    print(f"\n### 比分概率表（Top10）")
    print(f"| 比分 | Hicruben | mikobinbin | AndyDu | amir42 | federico1809 | 平均 |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    top = top_scores(results, 10)
    for (a, b), avg_p in top:
        row = f"| {a}-{b} "
        for name in model_order_score:
            if name in results:
                p = results[name]["scores"].get((a, b), 0)
                row += f"| {p*100:.1f}% "
            else:
                row += "| - "
        row += f"| {avg_p*100:.1f}% |"
        print(row)

    # 推荐比分
    best_score = top[0]
    print(f"\n**推荐比分: {best_score[0][0]}-{best_score[0][1]}** (平均概率 {best_score[1]*100:.1f}%)")

    # 胜负推荐
    if avg_hw > avg_d and avg_hw > avg_aw:
        pred = f"{home}胜"
        prob = avg_hw
    elif avg_aw > avg_d and avg_aw > avg_hw:
        pred = f"{away}胜"
        prob = avg_aw
    else:
        pred = "平局"
        prob = avg_d
    print(f"**推荐结果: {pred}** (概率 {prob*100:.1f}%)")


if __name__ == "__main__":
    matches = [
        ("Argentina", "Algeria"),
        ("France", "Senegal"),
        ("Iraq", "Norway"),
    ]

    print("# 2026世界杯预测报告 — 2026年6月16日")
    print()

    for home, away in matches:
        print(f"\n{'='*60}")
        results = run_match(home, away)
        print_report(home, away, results)
