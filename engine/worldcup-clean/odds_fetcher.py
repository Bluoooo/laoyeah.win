#!/usr/bin/env python3
"""
odds_fetcher.py — 赔率抓取 + 模型概率对比 + 最优投注分析

使用 The Odds API (https://the-odds-api.com/)
免费注册: 500次请求/月, 覆盖 bet365/Pinnacle/William Hill 等主流博彩公司

使用方法:
  1. 访问 https://the-odds-api.com/ 注册免费账号, 获取 API Key
  2. 设置环境变量: set ODDS_API_KEY=your_key_here
     或在下方直接填写 API_KEY
  3. python odds_fetcher.py
"""
import os
import json
import math
import sys

try:
    import requests
except ImportError:
    sys.exit("需要安装 requests: pip install requests")

# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════
API_KEY = os.environ.get("ODDS_API_KEY", "e8b2bcb052c1919d9a75e67368637dcd")
BASE_URL = "https://api.the-odds-api.com/v4"

# 要查询的比赛队伍关键词
MATCHES_TO_FIND = [
    ("Canada", "Bosnia"),
    ("USA", "Paraguay"),
]

# ══════════════════════════════════════════════════════════════════════════
# 模型概率 (来自我们的5模型预测)
# ══════════════════════════════════════════════════════════════════════════
MODEL_PROBS = {
    "Canada vs Bosnia": {
        "home": 0.537,  # Canada胜 (5模型平均)
        "draw": 0.258,
        "away": 0.205,  # Bosnia胜
    },
    "USA vs Paraguay": {
        "home": 0.369,  # USA胜
        "draw": 0.271,
        "away": 0.360,  # Paraguay胜
    },
}


def fetch_sports():
    """获取可用的体育项目列表"""
    url = f"{BASE_URL}/sports"
    params = {"apiKey": API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [错误] 获取体育列表失败: {e}")
        return []


def fetch_odds(sport="soccer_fifa_world_cup", regions="eu,us,uk", markets="h2h,spreads,totals"):
    """获取指定赛事的赔率
    sport: soccer_fifa_world_cup, soccer_epl, soccer_uefa_champs_league 等
    regions: us, uk, eu, au (可逗号分隔)
    markets: h2h (胜平负), spreads (让球), totals (大小球)
    """
    url = f"{BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        # 检查剩余请求次数
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        print(f"  API请求剩余: {remaining} 次 (已用: {used})")
        return resp.json()
    except requests.HTTPError as e:
        if resp.status_code == 401:
            print("  [错误] API Key无效, 请检查 ODDS_API_KEY")
        elif resp.status_code == 422:
            print(f"  [错误] 参数错误: {resp.text}")
        else:
            print(f"  [错误] HTTP {resp.status_code}: {e}")
        return []
    except Exception as e:
        print(f"  [错误] 请求失败: {e}")
        return []


def find_matches(events, keywords_list):
    """从赛事列表中找到目标比赛"""
    found = {}
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        for kw_home, kw_away in keywords_list:
            if (kw_home.lower() in home.lower() and kw_away.lower() in away.lower()):
                key = f"{kw_home} vs {kw_away}"
                found[key] = event
            elif (kw_away.lower() in home.lower() and kw_home.lower() in away.lower()):
                key = f"{kw_home} vs {kw_away}"
                # 主客反了, 记录下来
                event["_reversed"] = True
                found[key] = event
    return found


def extract_best_odds(event):
    """从赛事数据中提取各博彩公司的最优赔率"""
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    results = {
        "bookmakers": {},
        "best_home": 0, "best_draw": 0, "best_away": 0,
        "best_home_bm": "", "best_draw_bm": "", "best_away_bm": "",
        "avg_home": 0, "avg_draw": 0, "avg_away": 0,
        "over25": {}, "under25": {},
        "btts_yes": {}, "btts_no": {},
    }

    home_prices = []
    draw_prices = []
    away_prices = []

    for bm in bookmakers:
        bm_name = bm["title"]
        for market in bm.get("markets", []):
            if market["key"] == "h2h":
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                h = outcomes.get(event.get("home_team", ""), 0)
                d = outcomes.get("Draw", 0)
                a = outcomes.get(event.get("away_team", ""), 0)
                if event.get("_reversed"):
                    h, a = a, h
                results["bookmakers"][bm_name] = {"home": h, "draw": d, "away": a}
                if h: home_prices.append(h)
                if d: draw_prices.append(d)
                if a: away_prices.append(a)

            elif market["key"] == "totals":
                for o in market["outcomes"]:
                    if o.get("point") == 2.5:
                        if o["name"] == "Over":
                            results["over25"][bm_name] = o["price"]
                        else:
                            results["under25"][bm_name] = o["price"]

            elif market["key"] == "btts":
                for o in market["outcomes"]:
                    if o["name"] == "Yes":
                        results["btts_yes"][bm_name] = o["price"]
                    else:
                        results["btts_no"][bm_name] = o["price"]

    if home_prices:
        results["best_home"] = max(home_prices)
        results["avg_home"] = sum(home_prices) / len(home_prices)
        results["best_home_bm"] = [bm for bm, v in results["bookmakers"].items() if v["home"] == results["best_home"]][0] if results["bookmakers"] else ""
    if draw_prices:
        results["best_draw"] = max(draw_prices)
        results["avg_draw"] = sum(draw_prices) / len(draw_prices)
        results["best_draw_bm"] = [bm for bm, v in results["bookmakers"].items() if v["draw"] == results["best_draw"]][0] if results["bookmakers"] else ""
    if away_prices:
        results["best_away"] = max(away_prices)
        results["avg_away"] = sum(away_prices) / len(away_prices)
        results["best_away_bm"] = [bm for bm, v in results["bookmakers"].items() if v["away"] == results["best_away"]][0] if results["bookmakers"] else ""

    return results


def odds_to_implied(odds):
    """赔率转隐含概率"""
    if odds <= 0: return 0
    return 1.0 / odds


def kelly_fraction(prob, odds):
    """凯利公式: f* = (p * odds - 1) / (odds - 1)"""
    if odds <= 1: return 0
    f = (prob * odds - 1) / (odds - 1)
    return max(0, f)


def ev(prob, odds):
    """期望值: EV = prob * odds - 1"""
    return prob * odds - 1


def analyze_match(match_name, model_probs, odds_data):
    """分析单场比赛: 模型概率 vs 博彩赔率"""
    print(f"\n{'=' * 100}")
    print(f"  {match_name}")
    print(f"{'=' * 100}")

    if not odds_data:
        print("  未找到赔率数据")
        return

    p = model_probs
    o = odds_data

    # ── 各博彩公司赔率 ───────────────────────────────────────────────────
    print(f"\n  +-- 各博彩公司胜平负赔率 ------------------------------------------------------+")
    print(f"  | {'博彩公司':<18} {'主胜':>8} {'平局':>8} {'客胜':>8} {'返还率':>8} |")
    print(f"  |----------------------------------------------------------------------|")
    for bm_name, bm_odds in o["bookmakers"].items():
        h, d, a = bm_odds["home"], bm_odds["draw"], bm_odds["away"]
        if h > 0 and d > 0 and a > 0:
            margin = (1/h + 1/d + 1/a - 1) * 100
            payout = (1 - margin/100) * 100
            print(f"  | {bm_name:<18} {h:>8.2f} {d:>8.2f} {a:>8.2f} {payout:>7.1f}% |")
    print(f"  +----------------------------------------------------------------------+")

    # ── 最优赔率 vs 模型概率 ─────────────────────────────────────────────
    print(f"\n  +-- 最优赔率 vs 模型概率 (寻找价值投注) ----------------------------------------+")
    print(f"  | {'结果':<10} {'最优赔率':>10} {'来源':<14} {'隐含概率':>10} {'模型概率':>10} {'EV':>10} {'凯利比例':>10} |")
    print(f"  |----------------------------------------------------------------------|")

    analyses = [
        ("主胜", o["best_home"], o["best_home_bm"], p["home"]),
        ("平局", o["best_draw"], o["best_draw_bm"], p["draw"]),
        ("客胜", o["best_away"], o["best_away_bm"], p["away"]),
    ]

    for label, best_odds, bm, model_prob in analyses:
        implied = odds_to_implied(best_odds)
        ev_val = ev(model_prob, best_odds)
        kelly = kelly_fraction(model_prob, best_odds)
        ev_str = f"{ev_val:+.1%}"
        kelly_str = f"{kelly:.1%}" if kelly > 0 else "0%"
        marker = " <<<" if ev_val > 0 else ""
        print(f"  | {label:<10} {best_odds:>10.2f} {bm:<14} {implied:>9.1%} {model_prob:>9.1%} {ev_str:>10}{marker} {kelly_str:>10} |")
    print(f"  +----------------------------------------------------------------------+")

    # ── 价值投注判断 ─────────────────────────────────────────────────────
    print(f"\n  +-- 价值投注分析 --------------------------------------------------------------+")
    value_bets = []
    for label, best_odds, bm, model_prob in analyses:
        ev_val = ev(model_prob, best_odds)
        if ev_val > 0:
            kelly = kelly_fraction(model_prob, best_odds)
            value_bets.append((label, best_odds, bm, model_prob, ev_val, kelly))

    if value_bets:
        print(f"  |  发现 {len(value_bets)} 个价值投注机会!                                                |")
        print(f"  |----------------------------------------------------------------------|")
        for label, best_odds, bm, model_prob, ev_val, kelly in value_bets:
            print(f"  |  推荐: {label} @ {best_odds:.2f} ({bm})                                   |")
            print(f"  |    模型概率: {model_prob:.1%}  vs  隐含概率: {odds_to_implied(best_odds):.1%}  |")
            print(f"  |    期望值(EV): {ev_val:+.1%}  |  凯利比例: {kelly:.1%}                    |")
        print(f"  |----------------------------------------------------------------------|")
        print(f"  |  建议: 按凯利比例的半数(Half-Kelly)投注以控制风险                        |")
    else:
        print(f"  |  未发现正EV的价值投注机会                                                |")
        print(f"  |  博彩公司的赔率已经充分反映了各结果的概率                                |")
        print(f"  |  建议: 观望或选择模型置信度最高的结果小额投注                            |")
    print(f"  +----------------------------------------------------------------------+")

    # ── 大小球/双方进球 ──────────────────────────────────────────────────
    if o.get("over25") or o.get("btts_yes"):
        print(f"\n  +-- 大小球 & 双方进球 --------------------------------------------------------+")
        if o.get("over25"):
            best_over = max(o["over25"].values()) if o["over25"] else 0
            best_under = max(o["under25"].values()) if o.get("under25") else 0
            bm_over = max(o["over25"], key=o["over25"].get) if o["over25"] else ""
            print(f"  |  大2.5球: {best_over:.2f} ({bm_over})  |  小2.5球: {best_under:.2f}                   |")
        if o.get("btts_yes"):
            best_yes = max(o["btts_yes"].values()) if o["btts_yes"] else 0
            best_no = max(o["btts_no"].values()) if o.get("btts_no") else 0
            bm_yes = max(o["btts_yes"], key=o["btts_yes"].get) if o["btts_yes"] else ""
            print(f"  |  双方进球-是: {best_yes:.2f} ({bm_yes})  |  双方进球-否: {best_no:.2f}           |")
        print(f"  +----------------------------------------------------------------------+")

    return value_bets


def main():
    print()
    print("=" * 100)
    print("  赔率抓取 + 模型概率对比 + 最优投注分析")
    print("  数据源: The Odds API (the-odds-api.com)")
    print("=" * 100)

    if API_KEY == "YOUR_API_KEY_HERE":
        print()
        print("  [!] 请先设置 API Key:")
        print("      方法1: set ODDS_API_KEY=your_key_here  (Windows)")
        print("      方法2: export ODDS_API_KEY=your_key_here  (Linux/Mac)")
        print("      方法3: 直接修改本文件中的 API_KEY 变量")
        print()
        print("      免费注册: https://the-odds-api.com/")
        print("      免费额度: 500次请求/月")
        print()

        # 使用模拟数据做演示
        print("  使用模拟赔率数据进行演示分析...")
        demo_analysis()
        return

    # ── 获取赔率 ─────────────────────────────────────────────────────────
    print("\n  正在获取世界杯赔率...")

    # 尝试多个sport key
    sport_keys = [
        "soccer_fifa_world_cup",
        "soccer_fifa_world_cup_qualifiers",
        "soccer_international_friendly",
    ]

    all_events = []
    for sport in sport_keys:
        print(f"\n  查询: {sport}")
        events = fetch_odds(sport=sport, regions="eu,us,uk", markets="h2h,totals")
        if events:
            print(f"    找到 {len(events)} 场比赛")
            all_events.extend(events)

    if not all_events:
        print("\n  未找到任何赛事数据, 使用模拟数据做演示...")
        demo_analysis()
        return

    # ── 查找目标比赛 ─────────────────────────────────────────────────────
    found = find_matches(all_events, MATCHES_TO_FIND)

    if not found:
        print(f"\n  未找到目标比赛, 显示所有可用比赛:")
        for event in all_events[:20]:
            h = event.get("home_team", "?")
            a = event.get("away_team", "?")
            t = event.get("commence_time", "?")
            print(f"    {h} vs {a}  ({t})")
        print(f"\n  使用模拟数据做演示...")
        demo_analysis()
        return

    # ── 分析每场比赛 ─────────────────────────────────────────────────────
    all_value_bets = []
    for match_name, event in found.items():
        odds_data = extract_best_odds(event)
        model_probs = None
        for key in MODEL_PROBS:
            if any(kw.lower() in match_name.lower() for kw in key.split(" vs ")):
                model_probs = MODEL_PROBS[key]
                break
        if not model_probs:
            # 使用默认
            model_probs = {"home": 0.33, "draw": 0.33, "away": 0.33}

        value_bets = analyze_match(match_name, model_probs, odds_data)
        if value_bets:
            all_value_bets.extend([(match_name, *vb) for vb in value_bets])

    # ── 综合建议 ─────────────────────────────────────────────────────────
    if all_value_bets:
        print(f"\n{'=' * 100}")
        print(f"  综合投注建议")
        print(f"{'=' * 100}")
        print(f"\n  发现 {len(all_value_bets)} 个价值投注机会:")
        for match, label, odds, bm, prob, ev_val, kelly in all_value_bets:
            print(f"    {match}: {label} @ {odds:.2f} ({bm})  EV={ev_val:+.1%}  Kelly={kelly:.1%}")


def demo_analysis():
    """使用模拟赔率数据做演示分析"""
    # 模拟中国竞彩和国际博彩公司的典型赔率
    demo_odds = {
        "Canada vs Bosnia": {
            "bookmakers": {
                "竞彩(模拟)": {"home": 1.62, "draw": 3.40, "away": 4.80},
                "bet365": {"home": 1.70, "draw": 3.50, "away": 5.00},
                "Pinnacle": {"home": 1.72, "draw": 3.60, "away": 5.20},
                "William Hill": {"home": 1.65, "draw": 3.40, "away": 4.80},
                "Betfair": {"home": 1.75, "draw": 3.55, "away": 5.10},
            },
            "best_home": 1.75, "best_draw": 3.60, "best_away": 5.20,
            "best_home_bm": "Betfair", "best_draw_bm": "Pinnacle", "best_away_bm": "Pinnacle",
            "avg_home": 1.69, "avg_draw": 3.49, "avg_away": 4.98,
            "over25": {"bet365": 1.95, "Pinnacle": 1.98, "竞彩(模拟)": 1.90},
            "under25": {"bet365": 1.85, "Pinnacle": 1.88, "竞彩(模拟)": 1.82},
            "btts_yes": {"bet365": 1.90, "Pinnacle": 1.92},
            "btts_no": {"bet365": 1.90, "Pinnacle": 1.88},
        },
        "USA vs Paraguay": {
            "bookmakers": {
                "竞彩(模拟)": {"home": 2.35, "draw": 3.10, "away": 2.75},
                "bet365": {"home": 2.40, "draw": 3.20, "away": 2.80},
                "Pinnacle": {"home": 2.45, "draw": 3.25, "away": 2.85},
                "William Hill": {"home": 2.38, "draw": 3.15, "away": 2.78},
                "Betfair": {"home": 2.48, "draw": 3.22, "away": 2.88},
            },
            "best_home": 2.48, "best_draw": 3.25, "best_away": 2.88,
            "best_home_bm": "Betfair", "best_draw_bm": "Pinnacle", "best_away_bm": "Betfair",
            "avg_home": 2.41, "avg_draw": 3.18, "avg_away": 2.81,
            "over25": {"bet365": 2.05, "Pinnacle": 2.08, "竞彩(模拟)": 2.00},
            "under25": {"bet365": 1.78, "Pinnacle": 1.80, "竞彩(模拟)": 1.75},
            "btts_yes": {"bet365": 1.85, "Pinnacle": 1.88},
            "btts_no": {"bet365": 1.95, "Pinnacle": 1.92},
        },
    }

    print("\n  [注意] 以下使用模拟赔率数据, 非实时数据!")
    print("  真实赔率请设置 API Key 后获取\n")

    all_value_bets = []
    for match_name in ["Canada vs Bosnia", "USA vs Paraguay"]:
        model_probs = MODEL_PROBS.get(match_name, {"home": 0.33, "draw": 0.33, "away": 0.33})
        value_bets = analyze_match(match_name, model_probs, demo_odds[match_name])
        if value_bets:
            all_value_bets.extend([(match_name, *vb) for vb in value_bets])

    if all_value_bets:
        print(f"\n{'=' * 100}")
        print(f"  综合投注建议 (基于模拟赔率)")
        print(f"{'=' * 100}")
        total_kelly = 0
        for match, label, odds, bm, prob, ev_val, kelly in all_value_bets:
            print(f"\n  {match}: {label} @ {odds:.2f} ({bm})")
            print(f"    模型概率: {prob:.1%}  |  隐含概率: {odds_to_implied(odds):.1%}")
            print(f"    EV: {ev_val:+.1%}  |  Kelly: {kelly:.1%}  |  Half-Kelly: {kelly/2:.1%}")
            total_kelly += kelly

        print(f"\n  +----------------------------------------------------------------------+")
        print(f"  |  总凯利比例: {total_kelly:.1%}  |  总Half-Kelly: {total_kelly/2:.1%}                |")
        print(f"  |  如果总投注额为100元:                                               |")
        for match, label, odds, bm, prob, ev_val, kelly in all_value_bets:
            amount = 100 * (kelly / 2) / max(total_kelly / 2, 0.01)
            print(f"  |    {match} {label}: {amount:.0f}元                                         |")
        print(f"  +----------------------------------------------------------------------+")
    else:
        print(f"\n  模拟赔率下未发现价值投注, 博彩公司赔率与模型概率基本一致")


if __name__ == "__main__":
    main()
