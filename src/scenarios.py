"""Bull / base / bear scenarios.

Each scenario is a coherent set of assumption changes (growth, margins, terminal
growth) plus a target forward multiple. The 12-month price target is
target_fwd_pe x fwd_eps (a re-rating thesis); the DCF intrinsic value is also
computed per scenario for context. Outputs a probability-weighted target and the
implied risk/reward.
"""
from __future__ import annotations

import copy

import pandas as pd

import dcf


def run_scenarios(cfg: dict, base_revenue: float, wacc: float,
                  net_debt: float, shares: float, price: float) -> tuple[pd.DataFrame, dict]:
    rows = {}
    for name, sc in cfg["scenarios"].items():
        scfg = copy.deepcopy(cfg)
        scfg["valuation"]["projections"]["revenue_growth"] = sc["revenue_growth"]
        scfg["valuation"]["projections"]["ebit_margin"] = sc["ebit_margin"]
        scfg["valuation"]["terminal"]["growth"] = sc["terminal_growth"]

        proj = dcf.project(scfg, base_revenue)
        val = dcf.value(scfg, proj, wacc, net_debt, shares, price)
        pt = sc["target_fwd_pe"] * sc["fwd_eps"]
        rows[name] = {
            "probability": sc["probability"],
            "fwd_eps": sc["fwd_eps"],
            "target_fwd_pe": sc["target_fwd_pe"],
            "price_target": pt,
            "upside": pt / price - 1.0,
            "dcf_gordon": val["gordon"]["per_share"],
            "dcf_exit": val["exit"]["per_share"],
            "rev_cagr": (proj["revenue"].iloc[-1] / base_revenue) ** (1 / len(proj)) - 1,
            "terminal_margin": sc["ebit_margin"][-1],
        }
    df = pd.DataFrame(rows).T
    df = df.reindex(["bull", "base", "bear"])

    pw_target = float((df["price_target"] * df["probability"]).sum())
    summary = {
        "prob_weighted_target": pw_target,
        "prob_weighted_upside": pw_target / price - 1.0,
        "bull_pt": df.loc["bull", "price_target"],
        "base_pt": df.loc["base", "price_target"],
        "bear_pt": df.loc["bear", "price_target"],
        "reward_to_bull": df.loc["bull", "price_target"] / price - 1.0,
        "risk_to_bear": df.loc["bear", "price_target"] / price - 1.0,
        "price": price,
    }
    # reward/risk ratio (upside to bull vs downside to bear)
    down = price - df.loc["bear", "price_target"]
    up = df.loc["bull", "price_target"] - price
    summary["reward_risk_ratio"] = up / down if down > 0 else float("inf")
    return df, summary
