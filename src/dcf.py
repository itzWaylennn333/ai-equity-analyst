"""Discounted cash flow: FCFF projection, terminal value (Gordon + exit multiple),
enterprise -> equity -> per-share bridge, and a WACC x terminal-growth sensitivity grid.

FCFF_t = EBIT_t*(1-tax) + D&A_t - CapEx_t - dNWC_t   (SBC already expensed in EBIT)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def project(cfg: dict, base_revenue: float) -> pd.DataFrame:
    """Build the explicit FCFF projection off config base-case drivers."""
    v = cfg["valuation"]
    p = v["projections"]
    n = v["horizon_years"]
    g = p["revenue_growth"]
    m = p["ebit_margin"]
    tax = v["tax_rate"]
    da_pct, capex_pct, nwc_pct = p["da_pct_revenue"], p["capex_pct_revenue"], p["nwc_pct_revenue"]

    rows = []
    rev = base_revenue
    base_year = 2025
    for t in range(1, n + 1):
        rev = rev * (1 + g[t - 1])
        ebit = rev * m[t - 1]
        nopat = ebit * (1 - tax)
        da = rev * da_pct
        capex = rev * capex_pct
        dnwc = rev * nwc_pct
        fcff = nopat + da - capex - dnwc
        ebitda = ebit + da
        rows.append({
            "year": base_year + t, "t": t, "revenue": rev, "rev_growth": g[t - 1],
            "ebit_margin": m[t - 1], "ebit": ebit, "nopat": nopat, "da": da,
            "capex": capex, "dnwc": dnwc, "fcff": fcff, "ebitda": ebitda,
        })
    return pd.DataFrame(rows).set_index("year")


def value(cfg: dict, proj: pd.DataFrame, wacc: float, net_debt: float, shares: float,
          price: float, g: float | None = None, exit_mult: float | None = None) -> dict:
    v = cfg["valuation"]
    g = v["terminal"]["growth"] if g is None else g
    exit_mult = v["terminal"]["exit_ev_ebitda"] if exit_mult is None else exit_mult
    n = v["horizon_years"]

    disc = 1.0 / (1.0 + wacc) ** proj["t"]
    pv_fcff = (proj["fcff"] * disc).sum()

    fcff_n = proj["fcff"].iloc[-1]
    ebitda_n = proj["ebitda"].iloc[-1]
    disc_n = 1.0 / (1.0 + wacc) ** n

    tv_gordon = fcff_n * (1 + g) / (wacc - g)
    tv_exit = ebitda_n * exit_mult
    pv_tv_gordon = tv_gordon * disc_n
    pv_tv_exit = tv_exit * disc_n

    out = {"pv_fcff": pv_fcff, "wacc": wacc, "g": g, "exit_mult": exit_mult,
           "net_debt": net_debt, "shares": shares, "price": price}
    for tag, pv_tv, tv in (("gordon", pv_tv_gordon, tv_gordon), ("exit", pv_tv_exit, tv_exit)):
        ev = pv_fcff + pv_tv
        equity = ev - net_debt
        per_share = equity / shares
        out[tag] = {
            "tv": tv, "pv_tv": pv_tv, "ev": ev, "equity": equity,
            "per_share": per_share, "upside": per_share / price - 1.0,
            "terminal_pct_of_ev": pv_tv / ev,
        }
    return out


def sensitivity(cfg: dict, proj: pd.DataFrame, net_debt: float, shares: float,
                wacc_range, g_range) -> pd.DataFrame:
    """Per-share intrinsic value (Gordon TV) across WACC (rows) x terminal g (cols)."""
    grid = pd.DataFrame(index=[round(w, 4) for w in wacc_range],
                        columns=[round(g, 4) for g in g_range], dtype=float)
    for w in wacc_range:
        disc = 1.0 / (1.0 + w) ** proj["t"]
        pv_fcff = (proj["fcff"] * disc).sum()
        fcff_n = proj["fcff"].iloc[-1]
        disc_n = 1.0 / (1.0 + w) ** cfg["valuation"]["horizon_years"]
        for g in g_range:
            tv = fcff_n * (1 + g) / (w - g)
            ev = pv_fcff + tv * disc_n
            grid.loc[round(w, 4), round(g, 4)] = (ev - net_debt) / shares
    return grid.astype(float)
