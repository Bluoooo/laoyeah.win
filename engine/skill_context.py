"""
Skill Context — Deep Analysis Framework as Model Harness

Skill 不直接推比分。Skill 输出环境简报（信心度/差距感知/巨星因素/风险信号/score_boost），
调制模型的 xG + 比分概率。

影响范围：最高 50% 的模型输出（通过 blend_ratio 控制）
"""

import math

# ── 球队分类 (English names, same as fixture data) ──
SUPERSTAR_TEAMS = {"Brazil", "France", "Argentina", "Portugal", "England"}
SYSTEM_TEAMS = {"Germany", "Spain", "Netherlands", "Belgium", "Croatia", "Uruguay", "Colombia"}
DARK_HORSE_TEAMS = {
    "Turkey", "Morocco", "Senegal", "Japan", "South Korea", "Australia",
    "Canada", "USA", "Mexico", "Ecuador", "Paraguay", "Sweden", "Norway"
}
LOW_TIER_TEAMS = {
    "Haiti", "Curacao", "Cape Verde", "Jordan", "Uzbekistan", "Iraq",
    "Qatar", "Panama", "Saudi Arabia", "South Africa", "New Zealand", "Iran",
    "Algeria", "Tunisia", "Ghana", "DR Congo", "Czech Republic", "Austria",
    "Bosnia", "Switzerland", "Eswatini", "Egypt"
}


def classify_team(name):
    """返回 (分类标签, 超级巨星标志, 风格描述)"""
    if name in SUPERSTAR_TEAMS:
        return "superstar", True, "超级巨星型强队"
    elif name in SYSTEM_TEAMS:
        return "system", False, "体系型强队"
    elif name in DARK_HORSE_TEAMS:
        return "dark_horse", False, "黑马型中上游"
    else:
        return "low_tier", False, "低位防守/弱旅"


def skill_context(h_elo, a_elo, h_name, h_name_cn, a_name, a_name_cn):
    """
    返回 Skill 环境简报字典。

    输出字段:
      - overall_confidence: 0-10
      - elo_boost: 0.7-1.5, Elo 差距倍增系数
      - star_factor: 0-0.3, 超级巨星 xG 增益
      - xg_direct_add: 0-0.5, 直加 xG（不通过 Elo）
      - score_boost: 0-0.4, 从 1 球胜向 2+ 球胜抽概率
      - risk_penalty: 0-0.2
      - blend_ratio: 0-0.50
      - summary, details
    """
    elo_diff = h_elo - a_elo
    abs_diff = abs(elo_diff)

    h_style, h_star, h_label = classify_team(h_name)
    a_style, a_star, a_label = classify_team(a_name)

    # ── 1. 差距信心 ──
    gap_score = 0
    if abs_diff >= 350:
        gap_score = 10
    elif abs_diff >= 250:
        gap_score = 8
    elif abs_diff >= 150:
        gap_score = 5
    else:
        gap_score = 3

    strong_style = h_style if elo_diff > 0 else a_style
    weak_style = a_style if elo_diff > 0 else h_style

    if gap_score >= 8 and strong_style in ("superstar", "system") and weak_style == "low_tier":
        gap_score = min(10, gap_score + 1)

    # ── 2. 巨星增益 ──
    star_factor = 0.0
    if elo_diff > 0 and h_star and weak_style == "low_tier":
        star_factor = 0.35
    elif elo_diff < 0 and a_star and weak_style == "low_tier":
        star_factor = 0.35
    elif elo_diff > 0 and h_star:
        star_factor = 0.20
    elif elo_diff < 0 and a_star:
        star_factor = 0.20

    # ── 3. Elo 倍增系数 ──
    elo_boost = 1.0
    if gap_score >= 8 and strong_style in ("superstar", "system"):
        elo_boost = 1.35 + (star_factor * 0.4)
    elif gap_score >= 5 and strong_style in ("superstar", "system"):
        elo_boost = 1.15
    elif gap_score <= 3:
        elo_boost = 1.05  # 不再压缩，反而稍微放大

    # ── 4. 风险因素 ──
    risk_penalty = 0.0
    risks = []
    if strong_style == "dark_horse" and weak_style == "low_tier":
        risk_penalty = max(risk_penalty, 0.05)
        risks.append("黑马vs弱旅不如巨星队稳定")
    if 100 <= abs_diff <= 200 and strong_style == "dark_horse":
        risk_penalty = max(risk_penalty, 0.10)
        risks.append("实力接近+黑马热门，冷门风险")
    if strong_style == "superstar" and abs_diff > 300:
        risk_penalty = max(risk_penalty, 0.05)
        risks.append("巨星队可能留力")

    # ── 5. 综合信心 ──
    base_confidence = gap_score
    if elo_boost >= 1.3:
        base_confidence = min(10, base_confidence + 1)
    if risk_penalty > 0:
        base_confidence = max(1, base_confidence - 1)
    overall_confidence = base_confidence

    # ── 6. xG 直加 + score_boost（用已算好的 overall_confidence）──
    xg_direct_add = 0.0
    score_boost = 0.0
    if overall_confidence >= 8:
        xg_direct_add = 0.40 + (star_factor * 0.5)
        score_boost = 0.30
    elif overall_confidence >= 6:
        xg_direct_add = 0.30
        score_boost = 0.20
    elif overall_confidence >= 5:
        xg_direct_add = 0.0
        score_boost = 0.10
    else:
        xg_direct_add = 0.0
        score_boost = 0.0

    # ── 7. blend_ratio（最高 50%）──
    blend_ratio = (overall_confidence / 10) * 0.50
    blend_ratio = max(0, min(0.50, blend_ratio))

    # ── 8. 计算调整后 xG ──
    adjusted_diff = elo_diff * elo_boost
    xg_h = max(0.3, min(4.0, 1.35 + adjusted_diff / 400 * 0.8))
    xg_a = max(0.3, min(4.0, 1.35 - adjusted_diff / 400 * 0.8))

    # 巨星因子（通过 xG 直加）
    if h_star:
        xg_h += star_factor
    if a_star:
        xg_a += star_factor

    # xG 直加（独立于 Elo，附加给强队）
    if elo_diff > 0:
        xg_h += xg_direct_add
    else:
        xg_a += xg_direct_add

    # ── 9. 风格倾向描述（附加分析用）──
    # 基于球队分类和 Elo 差距，产生更具描述性的风格分析
    style_class_h, _, _ = classify_team(h_name)
    style_class_a, _, _ = classify_team(a_name)
    
    # 强队风格细分
    if elo_diff > 0:
        strong_class, weak_class = style_class_h, style_class_a
        strong_name_cn, weak_name_cn = h_name_cn, a_name_cn
    else:
        strong_class, weak_class = style_class_a, style_class_h
        strong_name_cn, weak_name_cn = a_name_cn, h_name_cn
    
    # 强队进攻风格推断
    # 强队进攻风格——Skill v2 框架：超级巨星型/体系型/黑马型/低效热门型
    if strong_class == "superstar":
        off_style = f"天赋型强队。{strong_name_cn}拥有顶级球星（梅西级别），个人能力足以撕破密集防守，是典型的天赋型进攻体系"
        off_style += "。进攻不依赖整体运转，而是核心球员的灵光一现。定位球战术也是重要破局武器"
        if abs_diff > 200:
            off_style += "。面对弱旅时经常打出大比分，但要注意赛程密集后的轮换"
    elif strong_class == "system":
        off_style = f"体系型强队。{strong_name_cn}以整体战术和体系运转取胜，不依赖个人能力，配合默契度高"
        if elo_diff > 0 and elo_diff < 300:
            off_style += "。控球渗透为主，通过传导拉扯防守创造机会"
        else:
            off_style += "。进攻端多点开花，具备持续施压能力"
        if abs_diff > 200:
            off_style += "。对弱旅有稳定的进球率，但大胜概率不如巨星队，要防范只赢一球的小胜剧本"
    elif strong_class == "dark_horse":
        if abs_diff > 150:
            off_style = f"黑马劲旅。{strong_name_cn}具备黑马成色，反击速度快、战术执行力强，擅长利用对手压上后的身后空间"
        else:
            off_style = f"中上游球队。{strong_name_cn}虽非传统豪门但已证明实力，整体打法成熟，具备与强队抗衡的能力"
        off_style += "。面对弱旅时控球率较高，需要靠阵地战解决问题"
    else:
        off_style = f"弱旅。{strong_name_cn}整体实力有限，进攻端缺乏稳定得分点，依赖定位球和反击偷鸡"
        if abs_diff > 200:
            off_style += "。预计摆大巴死守，能否守住前30分钟是比赛走势的关键"
    
    # 弱队防守风格
    if weak_class == "low_tier":
        if strong_class == "superstar" and abs_diff > 200:
            def_style = f"弱旅打强。{weak_name_cn}大概率摆大巴+严防死守，依赖密集中路防守和门将神扑。需要重点关注：开局能否守住30分钟不败、定位球防守是否有明显弱点、反击质量能否制造威胁"
        elif abs_diff > 200:
            def_style = f"弱旅守城。{weak_name_cn}大概率低位防守，靠纪律性和身体对抗消耗对手。防守重点是防边路传中和定位球"
        elif abs_diff > 100:
            def_style = f"弱旅守城。{weak_name_cn}有实力差距但不至于溃败，防守反击有可操作性"
        else:
            def_style = f"弱旅。{weak_name_cn}实力与对手在同一档，比赛节奏可能开放"
    elif weak_class == "dark_horse":
        def_style = f"中上游防守。{weak_name_cn}具备一定的整体防守能力，不至于溃败，反击速度和战术执行力是其最大武器"
        if abs_diff > 150:
            def_style += "。面对强队时可以通过中场绞杀扰乱对手节奏"
        else:
            def_style += "。同档次对决时防守稳定性决定比赛走向"
    elif weak_class == "system":
        if abs_diff > 100:
            def_style = f"体系队防守。{weak_name_cn}虽然是体系型强队但本场是相对强者，战术上不会死守，控球压迫是主要策略"
        else:
            def_style = f"体系队。{weak_name_cn}与对手同为体系型强队，防守端比拼的是整体纪律性和协防默契"
    else:
        def_style = f"同档较量。{weak_name_cn}实力与对手在同一档，比赛节奏可能开放"
    
    # 比赛局势预判
    if abs_diff > 250:
        match_dynamic = f"局势判断：强弱分明，{strong_name_cn}控球率预计65%+，{weak_name_cn}很难组织起有效进攻"
        if strong_class == "superstar":
            match_dynamic += "。关键看{strong_name_cn}多久打破僵局——前20分钟进球则比赛失去悬念，上半场闷平则下半场需要更多耐心。{weak_name_cn}能否守到70分钟后是冷门的关键窗口"
        elif strong_class == "system":
            match_dynamic += "。体系型强队需要耐心传导，打破密集防守的核心在于边路宽度和大范围转移。{weak_name_cn}的防守纪律性和门将状态是本场最大变量"
        else:
            match_dynamic += "。{strong_name_cn}虽然有实力优势但不具备绝对的破密防能力，需防范久攻不下后被反击失球"
    elif abs_diff > 150:
        match_dynamic = f"局有优势但不稳。{strong_name_cn}在纸面和控球率上占优，但{weak_name_cn}有足够的时间和空间组织有效反击。比赛可能由以下因素决定：第一个进球的时间点、定位球效率、以及强队能否在前60分钟打破僵局"
        if strong_class == "superstar":
            match_dynamic += "。超级巨星的个人发挥可能是打破平衡的唯一钥匙"
        elif strong_class == "dark_horse":
            match_dynamic += "。黑马冲击强队时，中场绞杀和反击效率是关键"
    elif abs_diff > 50:
        match_dynamic = f"实力接近但有倾向。{strong_name_cn}稍占上风，但{weak_name_cn}完全有能力拿分。比赛大概率由以下因素决定：中场控制权归属、定位球攻防效率、以及一方能否在70分钟后利用体能优势打破僵局。平局是大概率事件，但如果一方先破门，比赛节奏会快速开放"
    else:
        match_dynamic = f"势均力敌。双方实力几乎持平，比赛可能陷入中场绞杀。最可能的剧本是：平局或一球小胜，进球数预计2球以内。关键因素：谁的战术执行力更强、谁在关键时刻犯错的概率更低。这种盘口不建议重注，胜负很大程度上取决于临场发挥"
    
    # ── 比赛评估（Skill v2 框架：最可能剧本 + 关键因素 + 冷门窗口）──
    # 根据 Elo 差距和球队风格，生成这个比赛的分析摘要
    if overall_confidence >= 8:
        if strong_class == "superstar":
            core_story = f"超巨碾压局：{strong_name_cn}有望上半场就打破僵局，预计控球率65%+"
            if abs_diff > 300:
                core_story += "，大比分穿盘是大概率事件。"
            else:
                core_story += "，但需警惕一球小胜的剧本。"
            risk_focus = "最大不确定性：巨星状态是否在线、赛前是否有轮换信号、是否留力后续比赛。"
            key_factor = f"关键因素：{strong_name_cn}超级巨星的个人发挥、{weak_name_cn}的防守纪律性和门将表现"
        elif strong_class == "system":
            core_story = f"体系碾压局：{strong_name_cn}通过整体传导和控球消耗对手，预计控球率60%+"
            if abs_diff > 300:
                core_story += "，持续施压后应该能取得2球以上的胜利。"
            else:
                core_story += "，但体系队面对大巴有时会陷入久攻不下的困境。"
            risk_focus = "最大不确定性：体系队缺少绝对爆点，如果久攻不下可能被反击偷鸡。"
            key_factor = f"关键因素：{strong_name_cn}的进攻效率、{weak_name_cn}能否守住前60分钟"
        else:
            core_story = f"实力碾压局：{strong_name_cn}全面占优，预计控球率65%+，但进攻端可能不够锐利"
            risk_focus = "最大不确定性：缺少巨星破局，面对密集防守可能效率低下。"
            key_factor = f"关键因素：{strong_name_cn}的进攻效率、{weak_name_cn}的立足防守能力"
    elif overall_confidence >= 5:
        if strong_class in ("superstar", "system"):
            core_story = f"实力优势局：{strong_name_cn}有一定优势但对手并非毫无还手之力"
        else:
            core_story = f"难缠对局：{strong_name_cn}虽有优势但不稳，{weak_name_cn}有机会制造麻烦"
        risk_focus = "最大不确定性：这是最容易出现冷门的区间，优势不足以碾压，平局概率不容忽视。"
        key_factor = f"关键因素：谁能掌握中场控制权、定位球攻防效率、强队能否在前60分钟破局"
    else:
        if abs_diff < 50:
            core_story = f"镜像对局：双方实力几乎处于同一水平线，这场比赛是真正意义上的五五开"
            risk_focus = "最大不确定性：这种对局谁赢都正常，胜负很大程度上取决于临场发挥和运气。"
            key_factor = f"关键因素：后防线是否出现致命失误、关键球员的个人能力差距"
        else:
            strong_side = strong_name_cn
            weak_side = weak_name_cn
            core_story = f"谨慎对局：{strong_side}略占上风但不足以信服，{weak_side}完全有能力拿分"
            risk_focus = "最大不确定性：平局不能丢，胜负的判断过于敏感。"
            key_factor = f"关键因素：全场第一个进球的发生时间、定位球、防守稳定性"
    
    match_assessment = f"这场比赛的核心剧本：{core_story}。{key_factor}。{risk_focus}"
    
    # ── 10. 总结 ──
    if overall_confidence >= 8:
        if elo_diff > 0:
            summary = f"Skill信心高：{h_name_cn}对{a_name_cn}优势明显，可大胆预测大比分"
        else:
            summary = f"Skill信心高：{a_name_cn}对{h_name_cn}优势明显，可大胆预测大比分"
    elif overall_confidence >= 5:
        summary = "Skill信心中等：有一定优势，但不宜过度激进"
    else:
        summary = "Skill信心低：接近五五开，建议保守预测"

    return {
        "overall_confidence": overall_confidence,
        "blend_ratio": blend_ratio,
        "xg_direct_add": round(xg_direct_add, 2),
        "score_boost": round(score_boost, 2),
        "adjusted_xg_h": round(xg_h, 3),
        "adjusted_xg_a": round(xg_a, 3),
        "elo_boost": round(elo_boost, 2),
        "star_factor": round(star_factor, 2),
        "risk_penalty": round(risk_penalty, 2),
        "summary": summary,
        "details": {
            "gap_score": gap_score,
            "home_style": h_label,
            "away_style": a_label,
            "risks": risks,
            "confidence_label": "强" if overall_confidence >= 8 else ("中" if overall_confidence >= 5 else "弱"),
            "blend_pct": round(blend_ratio * 100),
            "offensive_style": off_style,
            "defensive_style": def_style,
            "match_dynamic": match_dynamic,
            "match_assessment": match_assessment,
        }
    }
