#!/usr/bin/env python3
"""
optimized_engine.py — v2 optimized models with draw calibration.

Changes from strategy_engine.py v1:
  1. Hicruben: DC_RHO -0.13 -> -0.25
  2. mikobinbin: added DC correction (RHO=-0.20)
  3. AndyDu: draw_base coeff 0.25->0.35, cap 0.30->0.40
  4. Post-hoc draw calibration layer
  5. Adaptive ensemble weighting by Elo diff
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))

from strategy_engine import poisson_pmf, dc_tau, predict_amir42, predict_federico1809, FED_MAP


# ══════════════════════════════════════════════════════════════════════════
# v2 Models
# ══════════════════════════════════════════════════════════════════════════

def model_hicruben_v2(home_elo, away_elo):
    """Hicruben with stronger DC correction for draws."""
    DC_RHO = -0.25  # was -0.13
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


def model_mikobinbin_v2(home_elo, away_elo):
    """mikobinbin with added DC correction."""
    DC_RHO = -0.20  # new: mikobinbin v1 had no DC correction
    adj_h = home_elo * 0.88 + 200; adj_a = away_elo * 0.88 + 200
    lam_h = max(0.3, min(3.5, 1.3 + (adj_h - 1700) / 500))
    lam_a = max(0.3, min(3.5, 1.3 + (adj_a - 1700) / 500))
    w = d = l = 0; scores = {}
    for a in range(9):
        for b in range(9):
            tau = dc_tau(a, b, lam_h, lam_a, DC_RHO)
            p = poisson_pmf(a, lam_h) * poisson_pmf(b, lam_a) * tau
            scores[(a, b)] = p
            if a > b: w += p
            elif a < b: l += p
            else: d += p
    t = w + d + l
    return w/t, d/t, l/t, scores


def model_andydu_v2(home_elo, away_elo):
    """AndyDu with stronger draw prior."""
    diff = (home_elo - away_elo) / 400
    hw = 1 / (1 + 10 ** (-diff))
    draw_base = 0.35 * math.exp(-abs(diff) * 2)  # was 0.25
    draw = max(0.08, min(0.40, draw_base))  # cap was 0.30, now 0.40
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


def score_dist_from_elo(h_elo, a_elo, elo_boost=1.0, star_factor_h=0.0, star_factor_a=0.0, xg_direct_add=0.0):
    """Estimate score distribution from Elo diff.
    
    elo_boost:      0.7-1.5, amplify or reduce the effective Elo gap
    star_factor:    0-0.3, add xG boost for superstar teams
    xg_direct_add:  0-0.5, direct xG addition (independent of Elo)
    """
    diff = (h_elo - a_elo) / 400 * elo_boost
    # xg_direct_add goes to the favorite (higher xG team)
    add_h = xg_direct_add if diff >= 0 else 0
    add_a = xg_direct_add if diff < 0 else 0
    xgh = max(0.3, min(4.0, 1.35 + diff * 0.8 + star_factor_h + add_h))
    xga = max(0.3, min(4.0, 1.35 - diff * 0.8 + star_factor_a + add_a))
    sc = {}
    for a in range(9):
        for b in range(9):
            sc[(a, b)] = poisson_pmf(a, xgh) * poisson_pmf(b, xga)
    t = sum(sc.values())
    return {k: v/t for k, v in sc.items()}


# ══════════════════════════════════════════════════════════════════════════
# Post-hoc calibration
# ══════════════════════════════════════════════════════════════════════════

def calibrate_probs(hw, d, aw, elo_diff):
    """Calibrate probabilities — very mild draw correction.

    Conservative: 5-10% relative boost to draw. Prevents over-fitting
    to the abnormally high 50% draw rate of the first 14 matches.
    """
    abs_diff = abs(elo_diff)

    # Very mild draw boost
    if abs_diff < 100:
        draw_boost = 1.08  # close match: boost draw by 8%
    elif abs_diff < 200:
        draw_boost = 1.05  # medium: boost by 5%
    else:
        draw_boost = 1.02  # mismatch: minimal boost

    d_cal = d * draw_boost

    # Redistribute: take from H and A proportionally
    excess = d_cal - d
    if hw + aw > 0:
        hw_cal = hw - excess * (hw / (hw + aw))
        aw_cal = aw - excess * (aw / (hw + aw))
    else:
        hw_cal = hw
        aw_cal = aw

    # Ensure non-negative
    hw_cal = max(0.02, hw_cal)
    d_cal = max(0.02, d_cal)
    aw_cal = max(0.02, aw_cal)

    # Normalize
    total = hw_cal + d_cal + aw_cal
    return hw_cal / total, d_cal / total, aw_cal / total


def predict_with_threshold(hw, d, aw, elo_diff):
    """Ensemble prediction: max probability wins.

    Note on draw prediction: extensive backtesting shows that Poisson-based
    models produce overlapping draw probability distributions for draw vs
    non-draw matches (25-34% for both). No threshold cleanly separates them.
    The v2 optimization improves probability quality (RPS, LogLoss) even
    without improving draw classification accuracy.

    For draw-aware predictions, use federico1809 standalone (42.9% draw recall).
    """
    if hw >= d and hw >= aw:
        return "H"
    elif aw >= d and aw >= hw:
        return "A"
    else:
        return "D"


# ══════════════════════════════════════════════════════════════════════════
# Adaptive ensemble
# ══════════════════════════════════════════════════════════════════════════

def get_adaptive_weights(elo_diff):
    """federico1809 is the only model that predicts draws, but it's biased.
    Give it moderate weight — enough to influence, not enough to dominate.
    """
    abs_diff = abs(elo_diff)
    if abs_diff < 150:
        # Close match: federico1809 gets 20%, others share 80%
        return {
            "Hicruben": 0.20, "mikobinbin": 0.20,
            "AndyDu": 0.20, "amir42": 0.20,
            "federico1809": 0.20,
        }
    elif abs_diff < 250:
        # Medium: federico1809 slightly less
        return {
            "Hicruben": 0.21, "mikobinbin": 0.21,
            "AndyDu": 0.21, "amir42": 0.21,
            "federico1809": 0.16,
        }
    else:
        # Mismatch: federico1809 minimal
        return {
            "Hicruben": 0.23, "mikobinbin": 0.23,
            "AndyDu": 0.23, "amir42": 0.23,
            "federico1809": 0.08,
        }


def run_model_v2(name, home, away, h_elo, a_elo):
    """Run a single model (v2 version)."""
    if name == "Hicruben":
        return model_hicruben_v2(h_elo, a_elo)
    elif name == "mikobinbin":
        return model_mikobinbin_v2(h_elo, a_elo)
    elif name == "AndyDu":
        return model_andydu_v2(h_elo, a_elo)
    elif name == "amir42":
        h_conf = FED_MAP.get(home, "UEFA")
        a_conf = FED_MAP.get(away, "UEFA")
        try:
            hw, d, aw = predict_amir42(h_elo, a_elo, h_conf, a_conf)
        except:
            return None
        sc = score_dist_from_elo(h_elo, a_elo)
        return hw, d, aw, sc
    elif name == "federico1809":
        try:
            hw, d, aw = predict_federico1809(h_elo, a_elo, home, away)
        except:
            return None
        sc = score_dist_from_elo(h_elo, a_elo)
        return hw, d, aw, sc
    return None


def get_ensemble_v2(home, away, h_elo, a_elo, calibrate=True):
    """Run all 5 v2 models with adaptive weighting and calibration."""
    elo_diff = h_elo - a_elo
    weights = get_adaptive_weights(elo_diff)

    results = {}
    for name in ["Hicruben", "mikobinbin", "AndyDu", "amir42", "federico1809"]:
        r = run_model_v2(name, home, away, h_elo, a_elo)
        if r:
            hw, d, aw, sc = r
            if calibrate:
                hw, d, aw = calibrate_probs(hw, d, aw, elo_diff)
            results[name] = {"hw": hw, "d": d, "aw": aw, "scores": sc}

    if not results:
        return None

    # Weighted average
    total_w = sum(weights.get(n, 0) for n in results)
    ens_hw = sum(results[n]["hw"] * weights.get(n, 0) for n in results) / total_w
    ens_d = sum(results[n]["d"] * weights.get(n, 0) for n in results) / total_w
    ens_aw = sum(results[n]["aw"] * weights.get(n, 0) for n in results) / total_w

    # Normalize
    t = ens_hw + ens_d + ens_aw
    return {
        "hw": ens_hw / t, "d": ens_d / t, "aw": ens_aw / t,
        "details": results, "weights": weights,
    }
