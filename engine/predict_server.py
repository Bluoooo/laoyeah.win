#!/usr/bin/env python3
"""
世界杯预测 API 服务器 v2.1
使用 optimized_engine.py v2 模型 + Football Match Deep Analysis Framework
"""
import os, sys, json, math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ENGINE_DIR = Path("/root/.openclaw/workspace/wc26-predict/extracted/worldcup-clean")
sys.path.insert(0, str(ENGINE_DIR))

from optimized_engine import (
    model_hicruben_v2, model_mikobinbin_v2, model_andydu_v2,
    calibrate_probs, get_adaptive_weights, score_dist_from_elo,
)
from strategy_engine import predict_amir42, predict_federico1809, poisson_pmf, ELO_DB, FED_MAP

# ── Entropy-based model confidence weights ──
MAX_ENTROPY = math.log2(3)  # log2(3) ≈ 1.585 for 3-outcome WDL

def model_confidence_weight(hw, d, aw):
    """Higher confidence when WDL is extreme (e.g. 80/15/5), lower when uncertain (e.g. 33/33/33)."""
    eps = 1e-10
    hw = max(eps, min(1-eps, hw))
    d  = max(eps, min(1-eps, d))
    aw = max(eps, min(1-eps, aw))
    entropy = -(hw * math.log2(hw) + d * math.log2(d) + aw * math.log2(aw))
    confidence = 1.0 - (entropy / MAX_ENTROPY)
    return max(eps, confidence)



# ── Team aliases ──
TEAM_ALIASES = {
    "Korea Republic": "South Korea", "United States": "USA",
    "Bosnia and Herzegovina": "Bosnia", "Turkey": "Türkiye", "Czechia": "Czech Republic",
}
for alias, real in TEAM_ALIASES.items():
    if alias not in ELO_DB and real in ELO_DB:
        ELO_DB[alias] = ELO_DB[real]

TEAM_CN = {
    "Argentina": "阿根廷", "Brazil": "巴西", "France": "法国", "Germany": "德国",
    "England": "英格兰", "Spain": "西班牙", "Portugal": "葡萄牙", "Netherlands": "荷兰",
    "Belgium": "比利时", "Italy": "意大利", "Croatia": "克罗地亚", "Uruguay": "乌拉圭",
    "Colombia": "哥伦比亚", "Japan": "日本", "Morocco": "摩洛哥", "Senegal": "塞内加尔",
    "Australia": "澳大利亚", "Ecuador": "厄瓜多尔", "Switzerland": "瑞士",
    "Denmark": "丹麦", "Poland": "波兰", "Serbia": "塞尔维亚", "Iran": "伊朗",
    "Tunisia": "突尼斯", "Saudi Arabia": "沙特阿拉伯", "Ghana": "加纳",
    "Cameroon": "喀麦隆", "Egypt": "埃及", "Ivory Coast": "科特迪瓦",
    "Panama": "巴拿马", "New Zealand": "新西兰", "Qatar": "卡塔尔",
    "Haiti": "海地", "Scotland": "苏格兰", "Austria": "奥地利",
    "Mexico": "墨西哥", "South Africa": "南非", "South Korea": "韩国",
    "Czech Republic": "捷克", "Canada": "加拿大", "USA": "美国",
    "Bosnia": "波黑", "Paraguay": "巴拉圭", "Türkiye": "土耳其",
    "Sweden": "瑞典", "Norway": "挪威", "Algeria": "阿尔及利亚",
    "Iraq": "伊拉克", "Jordan": "约旦", "DR Congo": "刚果(金)",
    "Uzbekistan": "乌兹别克斯坦", "Cabo Verde": "佛得角", "Curacao": "库拉索",
}
# Reverse CN → EN
TEAM_EN = {v: k for k, v in TEAM_CN.items()}

DATA_DIR = Path("/root/.openclaw/workspace/wc26-predict/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIXTURES_FILE = ENGINE_DIR / "2026-world-cup-predictor" / "data" / "match_cache.json"
MODEL_NAMES = ["Hicruben", "mikobinbin", "AndyDu", "amir42", "federico1809"]

V2_MODELS = {"Hicruben": model_hicruben_v2, "mikobinbin": model_mikobinbin_v2, "AndyDu": model_andydu_v2}

def normalize_team(name):
    if name in ELO_DB: return name
    if name in TEAM_ALIASES: return TEAM_ALIASES[name]
    return name

def aggregate_scores(model_results, results_wdl=None):
    """Aggregate score distributions using entropy-based model confidence weights.
    
    Models with more extreme (confident) WDL predictions get higher weight on scores.
    Models that are uncertain (WDL near 33/33/33) get lower weight.
    """
    # Calculate confidence weight for each model from its WDL
    conf_weights = {}
    total_cw = 0
    for name in MODEL_NAMES:
        if results_wdl and name in results_wdl:
            r = results_wdl[name]
            cw = model_confidence_weight(r["hw"], r["d"], r["aw"])
        else:
            cw = 1.0 / len(MODEL_NAMES)
        conf_weights[name] = cw
        total_cw += cw
    # Normalize
    for name in conf_weights:
        conf_weights[name] /= total_cw
    
    all_scores = {}
    for name, r in model_results.items():
        w = conf_weights.get(name, 0.20)
        for key, val in r["scores"].items():
            all_scores.setdefault(key, 0)
            all_scores[key] += val * w
    total = sum(all_scores.values())
    if total > 0:
        for k in all_scores: all_scores[k] /= total
    return all_scores

# ── Deep Analysis: tactical style classification ──
def classify_team_style(elo, elo_diff_abs):
    """Classify team style based on Elo and context."""
    if elo >= 1900:
        if elo_diff_abs > 200:
            return "超级巨星型强队", "拥有顶级球星和深厚阵容，具备个人能力和整体压制力"
        return "体系型强队", "整体战术成熟、控球能力强，但缺少绝对爆点"
    elif elo >= 1750:
        return "中上游劲旅", "具备一定实力，战术纪律好，对抗强队有韧性"
    elif elo >= 1650:
        return "黑马潜质队", "非传统强队但防守纪律好、反击有威胁，不能简单套用弱队模板"
    else:
        return "低位防守球队", "整体实力较弱，依赖防守纪律和反击偷鸡"

def classify_match_level(elo_diff_abs):
    if elo_diff_abs > 200:
        return "碾压局", "模型可信度最高，强弱分明，推荐出手"
    elif elo_diff_abs > 100:
        return "实力差距", "有实力差距但仍有悬念，需结合概率判断"
    elif elo_diff_abs > 50:
        return "稍有优势", "El o 差距可控，受让方具备一定拉力"
    else:
        return "旗鼓相当", "El o差<50，不碰胜平负（按规则），关注大小球"

def generate_enriched_analysis(home, away, h_elo, a_elo, results, weights):
    """Generate enriched analysis per Deep Analysis Skill framework."""
    elo_diff = h_elo - a_elo
    abs_diff = abs(elo_diff)
    ens = results["ensemble"]
    avg_hw, avg_d, avg_aw = ens["hw"], ens["d"], ens["aw"]

    # 1. Team style & level
    h_style, h_desc = classify_team_style(h_elo, abs_diff)
    a_style, a_desc = classify_team_style(a_elo, abs_diff)
    match_level, level_desc = classify_match_level(abs_diff)

    # 2. Model agreement analysis
    count_hw = sum(1 for n in MODEL_NAMES if results[n]["hw"] > results[n]["aw"] and results[n]["hw"] > results[n]["d"])
    count_aw = sum(1 for n in MODEL_NAMES if results[n]["aw"] > results[n]["hw"] and results[n]["aw"] > results[n]["d"])
    count_d  = sum(1 for n in MODEL_NAMES if results[n]["d"] > results[n]["hw"] and results[n]["d"] > results[n]["aw"])

    if count_hw >= 4 and count_aw == 0:
        agreement_type, agreement_desc = "高度一致", f"{count_hw}/5模型一致看好{TEAM_CN.get(home,home)}胜"
    elif count_aw >= 4 and count_hw == 0:
        agreement_type, agreement_desc = "高度一致", f"{count_aw}/5模型一致看好{TEAM_CN.get(away,away)}胜"
    elif count_hw >= 3 or count_aw >= 3:
        direction = TEAM_CN.get(home,home) if count_hw > count_aw else TEAM_CN.get(away,away)
        agreement_type, agreement_desc = "多数共识", f"{max(count_hw,count_aw)}/5模型倾向{direction}胜"
    else:
        agreement_type, agreement_desc = "存在分歧", f"模型之间预测分歧较大（H{count_hw}/D{count_d}/A{count_aw}），需谨慎判断"

    # 3. Federerico1809 draw signal — only meaningful when both high prob AND close match
    fed_d = results.get("federico1809", results["Hicruben"])["d"]
    abs_diff_local = abs(elo_diff)
    if fed_d >= 0.38 and abs_diff_local < 100:
        draw_signal = "强" if fed_d >= 0.42 else "中等"
    else:
        draw_signal = "无"

    # 4. Risk factors based on model data
    risks = []
    max_prob = max(avg_hw, avg_d, avg_aw) * 100
    
    if abs_diff < 100:
        risks.append({"factor": "El o 差距过小", "detail": "双方实力接近，任何方向都不具备统计优势", "severity": "high"})
    if max_prob < 40:
        risks.append({"factor": "概率低于40%", "detail": f"最高概率仅{max_prob:.0f}%，足球三结果基准线33%，仅高出{max_prob-33:.0f}pp，方向不明晰", "severity": "high"})
    if count_d >= 2:
        risks.append({"factor": "平局风险较高", "detail": f"{count_d}/5模型预测平局走势", "severity": "medium"})
    if abs_diff > 200 and avg_hw > 0.65:
        risks.append({"factor": "碾压局深盘", "detail": "强队赢球无悬念，但赢几球是主要问题，需关注让球盘口", "severity": "low"})
    if fed_d >= 0.35 and abs_diff < 100:
        risks.append({"factor": "平局冷门信号", "detail": "federico1809平局模型给出强平局信号（≥35%），结合Elo差<150，平局概率不容忽视", "severity": "medium"})

    # 5. Superstar factor
    superstar_teams = {"巴西", "法国", "阿根廷", "葡萄牙", "英格兰"}
    has_superstar = (home in superstar_teams or away in superstar_teams)
    superstar_factor = None
    if has_superstar and abs_diff > 100:
        superstar_factor = {
            "hasSuperstar": True,
            "detail": f"{'主队' if home in superstar_teams else '客队'}拥有超级巨星，面对密集防守时具备个人能力破局能力",
        }

    # 6. Goal analysis (Skill-adjusted xG)
    from skill_context import skill_context
    sc = skill_context(h_elo, a_elo, home, TEAM_CN.get(home, home), away, TEAM_CN.get(away, away))
    xg_h = sc["adjusted_xg_h"]
    xg_a = sc["adjusted_xg_a"]

    # Top scores — AndyDu 原始比分概率直接使用
    andydu_scores = results["AndyDu"]["scores"]
    total_andydu = sum(andydu_scores.values())
    top_scores = []
    if total_andydu > 0:
        for (h_goals, a_goals), p in sorted(andydu_scores.items(), key=lambda x: -x[1])[:5]:
            top_scores.append(((h_goals, a_goals), p/total_andydu))
    score_detail = [{"score": f"{k[0]}-{k[1]}", "prob": round(v*100, 1)} for k, v in top_scores]

    result = {
        "matchLevel": {"type": match_level, "desc": level_desc},
        "teamHome": {"name": TEAM_CN.get(home, home), "elo": h_elo, "style": h_style, "styleDesc": h_desc},
        "teamAway": {"name": TEAM_CN.get(away, away), "elo": a_elo, "style": a_style, "styleDesc": a_desc},
        "modelAgreement": {"type": agreement_type, "detail": agreement_desc, "votes": f"H{count_hw}-D{count_d}-A{count_aw}"},
        "drawSignal": {"level": draw_signal, "fedProb": round(fed_d*100, 1)},
        "expectedGoals": {"home": round(xg_h, 2), "away": round(xg_a, 2), "total": round(xg_h + xg_a, 2)},
        "riskFactors": risks,
        "topScores": score_detail,
        "styleDetail": {
            "offensive": sc["details"]["offensive_style"],
            "defensive": sc["details"]["defensive_style"],
            "dynamic": sc["details"]["match_dynamic"],
            "assessment": sc["details"].get("match_assessment", ""),
        },
    }
    if superstar_factor:
        result["superstarFactor"] = superstar_factor
    return result

def get_confidence(prob):
    if prob >= 65: return "高"
    if prob >= 50: return "中"
    if prob >= 40: return "中低"
    return "低"

def run_prediction(home, away, h_elo, a_elo):
    elo_diff = h_elo - a_elo
    weights = get_adaptive_weights(elo_diff)
    normal_home = normalize_team(home)
    normal_away = normalize_team(away)
    h_conf = FED_MAP.get(normal_home, "UEFA")
    a_conf = FED_MAP.get(normal_away, "UEFA")
    results = {}
    fallback_scores = score_dist_from_elo(h_elo, a_elo)

    for name, model_fn in V2_MODELS.items():
        try:
            hw, d, aw, sc = model_fn(h_elo, a_elo)
            hw, d, aw = calibrate_probs(hw, d, aw, elo_diff)
            results[name] = {"hw": hw, "d": d, "aw": aw, "scores": sc}
        except Exception as e:
            results[name] = {"hw": 0.34, "d": 0.33, "aw": 0.33, "scores": fallback_scores}

    for name, fn, needs_conf in [("amir42", predict_amir42, True), ("federico1809", predict_federico1809, False)]:
        try:
            if needs_conf:
                hw, d, aw = fn(h_elo, a_elo, h_conf, a_conf)
            else:
                hw, d, aw = fn(h_elo, a_elo, normal_home, normal_away)
            hw, d, aw = calibrate_probs(hw, d, aw, elo_diff)
            results[name] = {"hw": hw, "d": d, "aw": aw, "scores": fallback_scores}
        except:
            avg_hw = sum(r["hw"] for r in results.values()) / len(results)
            avg_d = sum(r["d"] for r in results.values()) / len(results)
            avg_aw = sum(r["aw"] for r in results.values()) / len(results)
            results[name] = {"hw": avg_hw, "d": avg_d, "aw": avg_aw, "scores": fallback_scores}

    w_avg_hw = sum(results[n]["hw"] * weights[n] for n in MODEL_NAMES)
    w_avg_d = sum(results[n]["d"] * weights[n] for n in MODEL_NAMES)
    w_avg_aw = sum(results[n]["aw"] * weights[n] for n in MODEL_NAMES)
    total = w_avg_hw + w_avg_d + w_avg_aw
    w_avg_hw /= total; w_avg_d /= total; w_avg_aw /= total
    agg_scores = aggregate_scores(results, results)
    results["ensemble"] = {"hw": w_avg_hw, "d": w_avg_d, "aw": w_avg_aw, "scores": agg_scores}
    return results, weights

def format_prediction_json(home, away, time_str, h_elo, a_elo, results, weights, score_a=None, score_b=None):
    ens = results["ensemble"]
    avg_hw, avg_d, avg_aw = ens["hw"], ens["d"], ens["aw"]
    elo_diff = h_elo - a_elo

    # WDL
    wdl = {}
    for name in MODEL_NAMES:
        r = results.get(name, results["Hicruben"])
        wdl[name] = {"home": round(r["hw"]*100, 1), "draw": round(r["d"]*100, 1), "away": round(r["aw"]*100, 1)}
    wdl["ensemble"] = {"home": round(avg_hw*100, 1), "draw": round(avg_d*100, 1), "away": round(avg_aw*100, 1)}

    # ── Score probabilities: AndyDu's 原始输出直接作为比分概率 ──
    from skill_context import skill_context
    sc = skill_context(h_elo, a_elo, home, TEAM_CN.get(home, home), away, TEAM_CN.get(away, away))
    
    andydu_scores = results["AndyDu"]["scores"]
    total_andydu = sum(andydu_scores.values())
    score_probs = []
    if total_andydu > 0:
        for (h_goals, a_goals), p in sorted(andydu_scores.items(), key=lambda x: -x[1])[:12]:
            probs = [round(results.get(n, results["Hicruben"])["scores"].get((h_goals, a_goals), 0)*100, 1) for n in MODEL_NAMES]
            score_probs.append({"score": f"{h_goals}-{a_goals}", "probs": probs, "avg": round((p/total_andydu)*100, 2)})

    skill_ctx_out = sc["summary"] + f" (boost:{sc['elo_boost']:.1f}x, star:{sc['star_factor']:.1f}, xg_add:{sc['xg_direct_add']})"

    # Recommendation — result direction from WDL 集成
    if avg_hw > avg_d and avg_hw > avg_aw:
        result = TEAM_CN.get(home, home) + "胜"
        result_prob = round(avg_hw*100, 1)
        result_dir = "H"
    elif avg_aw > avg_d:
        result = TEAM_CN.get(away, away) + "胜"
        result_prob = round(avg_aw*100, 1)
        result_dir = "A"
    else:
        result = "平局"
        result_prob = round(avg_d*100, 1)
        result_dir = "D"

    # Best score: score_probs[0] 就是综合概率最高的比分
    abs_diff = abs(elo_diff)
    best_score = score_probs[0]["score"] if score_probs else "1-1"
    best_score_prob = score_probs[0]["avg"] if score_probs else 0
    # 方向仍从 WDL 集成得来（反应整体胜率，不是单分概率）
    if avg_hw > avg_d and avg_hw > avg_aw:
        result = TEAM_CN.get(home, home) + "胜"
        result_prob = round(avg_hw*100, 1)
        result_dir = "H"
    elif avg_aw > avg_d:
        result = TEAM_CN.get(away, away) + "胜"
        result_prob = round(avg_aw*100, 1)
        result_dir = "A"
    else:
        result = "平局"
        result_prob = round(avg_d*100, 1)
        result_dir = "D"

    # Enriched Deep Analysis
    analysis = generate_enriched_analysis(home, away, h_elo, a_elo, results, weights)

    # Reasoning
    abs_diff = abs(elo_diff)
    _r = []  # build list, then add conditional blocks
    _r.append({"label": "实力对比", "value": analysis["matchLevel"]["type"],
         "detail": f"El o差距{int(abs_diff)}，{TEAM_CN.get(home,home)}({int(h_elo)}) vs {TEAM_CN.get(away,away)}({int(a_elo)})。{sc['details']['match_dynamic']}",
         "type": "green" if abs_diff > 200 else "gold" if abs_diff > 100 else "neutral"})
    _r.append({"label": "模型一致度", "value": analysis["modelAgreement"]["type"],
         "detail": analysis["modelAgreement"]["detail"], "type": "green" if "一致" in analysis["modelAgreement"]["type"] else "gold" if "共识" in analysis["modelAgreement"]["type"] else "neutral"})
    _r.append({"label": "概率分析", "value": f"{max(avg_hw, avg_d, avg_aw)*100:.0f}%",
         "detail": f"集成最高概率{max(avg_hw, avg_d, avg_aw)*100:.1f}%（三结果基准33%），{'远超33%基准，方向非常明确' if max(avg_hw, avg_d, avg_aw)*100 >= 65 else '大幅高于33%基准，方向明确' if max(avg_hw, avg_d, avg_aw)*100 >= 50 else '高于33%基准，有一定倾向' if max(avg_hw, avg_d, avg_aw)*100 >= 40 else '略高于33%基准，方向不明确'}",
         "type": "green" if max(avg_hw, avg_d, avg_aw)*100 >= 65 else "gold" if max(avg_hw, avg_d, avg_aw)*100 >= 50 else "neutral"})
    _r.append({"label": "对阵风格", "value": f"{TEAM_CN.get(home,home)}({analysis['teamHome']['style']}) vs {TEAM_CN.get(away,away)}({analysis['teamAway']['style']})",
         "detail": f"进攻：{sc['details']['offensive_style']}。防守：{sc['details']['defensive_style']}",
         "type": "neutral"})
    
    # 比赛评估（Skill v2 框架）
    _r.append({"label": "比赛评估", "value": analysis.get("matchLevel", {}).get("type", ""),
         "detail": sc["details"].get("match_assessment", ""),
         "type": "neutral"})
    
    # 巨星因素（有条件）
    _show_star = analysis.get("superstarFactor") or (abs_diff > 100 and (TEAM_CN.get(home,home) in ['巴西','法国','阿根廷','葡萄牙','英格兰'] or TEAM_CN.get(away,away) in ['巴西','法国','阿根廷','葡萄牙','英格兰']))
    if _show_star:
        star_team_cn = TEAM_CN.get(home, home) if TEAM_CN.get(home,home) in ['巴西','法国','阿根廷','葡萄牙','英格兰'] else TEAM_CN.get(away, away)
        _r.append({"label": "巨星因素", "value": f"{sc['star_factor']:.0%}增益" if sc['star_factor']>0 else "加成",
           "detail": f"{star_team_cn}拥有超级巨星，在密集防守面前具备个人能力破局能力，是改变比赛走向的关键变量。Skill将El o倍率提升至{sc['elo_boost']:.1f}倍并附加{sc['xg_direct_add']:.2f}xG直加" if sc['star_factor']>0 else f"{star_team_cn}拥有超级巨星，但本场对手并不弱，巨星效应可能受限",
           "type": "gold"})
    
    # 平局信号
    if analysis['drawSignal']['level'] != '无':
        _r.append({
            "label": "平局信号", "value": f"{analysis['drawSignal']['level']}({analysis['drawSignal']['fedProb']}%)",
            "detail": f"federico1809平局模型给出{analysis['drawSignal']['fedProb']}%平局概率，{'平局需重点防范' if analysis['drawSignal']['level'] == '强' else '平局有一定可能'}",
            "type": "gold",
        })
    _r.append({
        "label": "大小球", "value": "大球" if analysis["expectedGoals"]["total"] > 2.5 else "小球" if analysis["expectedGoals"]["total"] < 2.2 else "均衡",
         "detail": f"期望总进球{analysis['expectedGoals']['total']:.1f}，{TEAM_CN.get(home,home)}场均{analysis['expectedGoals']['home']:.2f}球 vs {TEAM_CN.get(away,away)}场均{analysis['expectedGoals']['away']:.2f}球。{'双方差距过大会拉大比分' if analysis['expectedGoals']['total'] > 3.0 else '有一定可能性打出大球' if analysis['expectedGoals']['total'] > 2.5 else '小球倾向明显' if analysis['expectedGoals']['total'] < 2.0 else ''}",
         "type": "gold" if analysis["expectedGoals"]["total"] > 2.5 or analysis["expectedGoals"]["total"] < 2.2 else "neutral"})
    
    # 风险信号（有条件）
    _risks = analysis.get("riskFactors", [])
    if _risks:
        high_count = sum(1 for r in _risks if r["severity"] == "high")
        med_count = sum(1 for r in _risks if r["severity"] == "medium") 
        risk_summary = f"{len(_risks)}项风险" + (f"（{high_count}高{med_count}中）" if high_count > 0 or med_count > 0 else "")
        risk_details = "；".join(f"【{'高' if r['severity']=='high' else '中' if r['severity']=='medium' else '低'}】{r['detail']}" for r in _risks[:3])
        _r.append({"label": "风险信号", "value": risk_summary,
           "detail": risk_details,
           "type": "gold" if high_count > 0 else "neutral"})
    
    reasoning = _r

    ret = {
        "home": TEAM_CN.get(home, home),
        "away": TEAM_CN.get(away, away),
        "time": time_str,
        "eloHome": round(h_elo, 1),
        "eloAway": round(a_elo, 1),
        "eloDiff": round(elo_diff, 1),
        "weights": {k: round(v, 2) for k, v in sorted(weights.items(), key=lambda x: -x[1])},
        "winDrawLoss": wdl,
        "scoreProb": score_probs,
        "recommendation": {
            "result": result, "resultProb": result_prob,
            "score": best_score, "scoreProb": best_score_prob,
            "confidence": get_confidence(result_prob),
        },
        "analysis": analysis,
        "skillContext": skill_ctx_out,
        "reasoning": reasoning,
    }
    if score_a is not None:
        ret["actualScore"] = f"{score_a}-{score_b}"
    return ret

def load_fixtures():
    if not FIXTURES_FILE.exists():
        print(f"❌ Not found: {FIXTURES_FILE}"); return []
    with open(FIXTURES_FILE) as f: data = json.load(f)
    return data.get("fixtures", [])

def generate_all_predictions():
    fixtures = load_fixtures()
    print(f"📋 Loaded {len(fixtures)} fixtures")
    date_groups = {}
    for f in fixtures:
        date = f["datetime"][:10]
        date_groups.setdefault(date, []).append(f)
    output = {"dates": [], "matches": {}}
    weekday_map = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    for date_str in sorted(date_groups.keys()):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        output["dates"].append({"date": date_str, "weekday": weekday_map[dt.weekday()],
                                "day": str(dt.day), "month": f"{dt.month}月"})
        output["matches"][date_str] = []
        date_groups[date_str].sort(key=lambda f: f["time"])
        for f in date_groups[date_str]:
            home, away = f["team_a"], f["team_b"]
            h_elo = ELO_DB.get(normalize_team(home), 1800)
            a_elo = ELO_DB.get(normalize_team(away), 1800)
            results, weights = run_prediction(home, away, h_elo, a_elo)
            pred = format_prediction_json(home, away, f["time"], h_elo, a_elo, results, weights,
                                          f.get("score_a"), f.get("score_b"))
            output["matches"][date_str].append(pred)
    path = DATA_DIR / "predictions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ v2.1 done! {len(output['matches'])} dates, {sum(len(v) for v in output['matches'].values())} matches")
    return path

# ═══════════ FastAPI ═══════════
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="世界杯预测 API v2.1", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

def load_predictions():
    p = DATA_DIR / "predictions.json"
    if not p.exists(): return None
    with open(p, encoding="utf-8") as f: return json.load(f)

@app.get("/api/predictions")
def get_predictions(date: Optional[str] = Query(None)):
    data = load_predictions()
    if not data: return JSONResponse({"error": "No data"}, status_code=404)
    if date: return JSONResponse({"dates": [d for d in data["dates"] if d["date"] == date],
                                  "matches": {date: data["matches"].get(date, [])}})
    return JSONResponse(data)

@app.get("/api/predictions/today")
def get_today():
    today = datetime.now().strftime("%Y-%m-%d")
    data = load_predictions()
    if not data: return JSONResponse({"error": "No data"}, status_code=404)
    return JSONResponse({"date": today, "count": len(data["matches"].get(today, [])),
                         "matches": data["matches"].get(today, [])})

@app.post("/api/refresh")
def refresh():
    try:
        path = generate_all_predictions()
        return JSONResponse({"status": "ok", "message": "v2.1 enriched predictions updated", "path": str(path)})
    except Exception as e:
        import traceback
        return JSONResponse({"status": "error", "message": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.get("/api/status")
def status():
    p = DATA_DIR / "predictions.json"
    if p.exists():
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        with open(p) as f: data = json.load(f)
        return JSONResponse({"status": "running", "version": "2.1",
            "lastUpdate": mtime.strftime("%Y-%m-%d %H:%M"),
            "matchCount": sum(len(v) for v in data["matches"].values()),
            "dateCount": len(data["dates"])})
    return JSONResponse({"status": "running", "version": "2.1"})

@app.get("/")
def root():
    return {"name": "世界杯预测 API v2.1", "version": "2.1.0"}

@app.get("/api")
def api_root():
    return {"name": "世界杯预测 API v2.1", "version": "2.1.0"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 v2.1 server starting...")
    if not (DATA_DIR / "predictions.json").exists():
        print("📊 First run: generating predictions...")
        generate_all_predictions()
    uvicorn.run(app, host="127.0.0.1", port=8080)
