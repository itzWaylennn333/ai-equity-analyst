"""Auto-derive valuation assumptions from a company's own history.

Any `null` left in the merged config (i.e. not hand-specified in companies/<T>.yaml)
is filled here from historical financials + the live snapshot, so the platform can
produce a sensible, fully-populated valuation for ANY ticker — which the user then
tunes via the interface. Companies with explicit assumptions (e.g. PYPL) are untouched.

Everything is clamped to defensible ranges; nothing is invented beyond the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(x, lo, hi, default):
    try:
        x = float(x)
        if not np.isfinite(x):
            return default
        return float(np.clip(x, lo, hi))
    except (TypeError, ValueError):
        return default


def _recent_mean(series: pd.Series, n: int = 3):
    s = series.dropna()
    return s.tail(n).mean() if len(s) else np.nan


def _derive_scenarios(projections: dict, terminal: dict, info: dict) -> dict:
    fwd_eps = info.get("forwardEps") or info.get("trailingEps") or 1.0
    fwd_pe = info.get("forwardPE") or info.get("trailingPE") or 15.0
    fwd_pe = _clip(fwd_pe, 4, 40, 15.0)
    bg, bm = projections["revenue_growth"], projections["ebit_margin"]
    cap = terminal["growth_cap"]

    def shift(lst, d):
        return [round(max(v + d, -0.05), 4) for v in lst]

    return {
        "bull": {"probability": 0.25, "revenue_growth": shift(bg, 0.03), "ebit_margin": shift(bm, 0.03),
                 "terminal_growth": round(min(terminal["growth"] + 0.005, cap), 4),
                 "fwd_eps": round(fwd_eps * 1.05, 2), "target_fwd_pe": round(fwd_pe * 1.30, 1),
                 "notes": "Auto: faster growth, margin expansion, modest multiple re-rating."},
        "base": {"probability": 0.50, "revenue_growth": list(bg), "ebit_margin": list(bm),
                 "terminal_growth": round(terminal["growth"], 4),
                 "fwd_eps": round(fwd_eps, 2), "target_fwd_pe": round(fwd_pe, 1),
                 "notes": "Auto: current trajectory at the current multiple (neutral)."},
        "bear": {"probability": 0.25, "revenue_growth": shift(bg, -0.03), "ebit_margin": shift(bm, -0.03),
                 "terminal_growth": round(max(terminal["growth"] - 0.01, 0.0), 4),
                 "fwd_eps": round(fwd_eps * 0.95, 2), "target_fwd_pe": round(fwd_pe * 0.75, 1),
                 "notes": "Auto: slower growth, margin pressure, multiple de-rating."},
    }


def ensure_assumptions(cfg: dict, H: pd.DataFrame, info: dict,
                       peer_ev_ebitda: float | None = None) -> dict:
    """Fill any null valuation inputs from history. Returns the same cfg (mutated)."""
    v = cfg["valuation"]
    erp = cfg.get("erp", {})
    horizon = v["horizon_years"]

    rev = H["revenue"].dropna()
    hist_cagr = ((rev.iloc[-1] / rev.iloc[0]) ** (1 / (len(rev) - 1)) - 1) if len(rev) >= 2 else 0.03
    hist_cagr = _clip(hist_cagr, -0.02, 0.20, 0.03)
    base_margin = _clip(_recent_mean(H["operating_income"] / H["revenue"]), 0.02, 0.45, 0.12)
    base_tax = _clip(_recent_mean(H["tax_provision"] / H["pretax_income"]), 0.12, 0.30, 0.23)
    da_pct = _clip(_recent_mean(H["da"] / H["revenue"]), 0.003, 0.12, 0.03)
    capex_pct = _clip(_recent_mean(H["capex"].abs() / H["revenue"]), 0.003, 0.15, 0.03)

    # Terminal
    term = v["terminal"]
    if term.get("growth") is None:
        term["growth"] = round(min(0.025, term["growth_cap"]), 4)
    if term.get("exit_ev_ebitda") is None:
        term["exit_ev_ebitda"] = _clip(peer_ev_ebitda, 5, 15, 10.0)

    # WACC inputs
    w = v["wacc"]
    if w.get("beta") is None:
        w["beta"] = _clip(info.get("beta"), 0.4, 2.5, 1.1)
    if w.get("equity_risk_premium") is None:
        w["equity_risk_premium"] = erp.get("default", 0.045)
    if w.get("erp_source") is None:
        w["erp_source"] = erp.get("source", "Damodaran ERP (default)")
    if w.get("cost_of_debt") is None:
        td, ie = H["total_debt"].dropna(), H["interest_expense"].dropna()
        cod = (ie.iloc[-1] / td.iloc[-1]) if len(td) and len(ie) and td.iloc[-1] > 0 else 0.05
        w["cost_of_debt"] = _clip(cod, 0.02, 0.09, 0.05)

    # Taxes
    if v.get("tax_rate") is None:
        v["tax_rate"] = round(base_tax, 4)
    if v.get("marginal_tax_rate") is None:
        v["marginal_tax_rate"] = 0.24

    # Projections
    p = v["projections"]
    if p.get("revenue_growth") is None:
        g0, gT = hist_cagr, term["growth"]
        p["revenue_growth"] = ([round(g0 + (gT - g0) * i / (horizon - 1), 4) for i in range(horizon)]
                               if horizon > 1 else [round(g0, 4)])
    if p.get("ebit_margin") is None:
        p["ebit_margin"] = [round(base_margin, 4)] * horizon
    if p.get("da_pct_revenue") is None:
        p["da_pct_revenue"] = round(da_pct, 4)
    if p.get("capex_pct_revenue") is None:
        p["capex_pct_revenue"] = round(capex_pct, 4)
    if p.get("nwc_pct_revenue") is None:
        p["nwc_pct_revenue"] = 0.01

    # Scenarios
    if cfg.get("scenarios") is None:
        cfg["scenarios"] = _derive_scenarios(p, term, info)

    return cfg
