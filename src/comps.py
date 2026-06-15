"""Comparable-company analysis.

Pulls a named peer set, builds a multiples table (EV/Revenue, EV/EBITDA, trailing &
forward P/E), takes the CORE-peer medians (card networks are shown for context only,
excluded from the median), and applies them to PayPal's metrics for an implied range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_loader as dl

_FIELDS = ["shortName", "marketCap", "enterpriseValue", "enterpriseToRevenue",
           "enterpriseToEbitda", "trailingPE", "forwardPE", "trailingEps",
           "forwardEps", "revenueGrowth", "ebitda", "totalRevenue"]


def _peer_row(cfg, ticker, name, group):
    info = dl.get_info(cfg, ticker)
    row = {"ticker": ticker, "name": name, "group": group}
    for f in _FIELDS:
        row[f] = info.get(f)
    # PEG = trailing P/E / (forward earnings growth %), when both available
    g = info.get("earningsGrowth") or info.get("revenueGrowth")
    pe = info.get("trailingPE")
    row["peg"] = (pe / (g * 100)) if (pe and g and g > 0) else np.nan
    return row


def build_comps(cfg: dict) -> pd.DataFrame:
    rows = []
    for grp in ("core", "anchors"):
        for p in cfg["peers"][grp]:
            rows.append(_peer_row(cfg, p["ticker"], p["name"], grp))
    df = pd.DataFrame(rows).set_index("ticker")
    return df


def peer_medians(comps: pd.DataFrame) -> dict:
    core = comps[comps["group"] == "core"]
    return {
        "ev_revenue": core["enterpriseToRevenue"].median(),
        "ev_ebitda": core["enterpriseToEbitda"].median(),
        "trailing_pe": core["trailingPE"].median(),
        "forward_pe": core["forwardPE"].median(),
        "ev_revenue_range": (core["enterpriseToRevenue"].min(), core["enterpriseToRevenue"].max()),
        "ev_ebitda_range": (core["enterpriseToEbitda"].min(), core["enterpriseToEbitda"].max()),
        "trailing_pe_range": (core["trailingPE"].min(), core["trailingPE"].max()),
    }


def implied_valuation(cfg: dict, pypl_info: dict, medians: dict,
                      net_debt: float, shares: float) -> pd.DataFrame:
    """Apply peer medians to PYPL metrics -> implied per-share values."""
    rev = pypl_info["totalRevenue"]
    ebitda = pypl_info["ebitda"]
    eps = pypl_info.get("trailingEps")
    feps = pypl_info.get("forwardEps")
    price = pypl_info["currentPrice"]

    out = []

    def add(method, ev=None, equity=None):
        if equity is None and ev is not None:
            equity = ev - net_debt
        ps = equity / shares
        out.append({"method": method, "implied_ev": ev,
                    "per_share": ps, "upside": ps / price - 1.0})

    add("EV/Revenue (peer median)", ev=medians["ev_revenue"] * rev)
    add("EV/EBITDA (peer median)", ev=medians["ev_ebitda"] * ebitda)
    if eps:
        out.append({"method": "Trailing P/E (peer median)", "implied_ev": np.nan,
                    "per_share": medians["trailing_pe"] * eps,
                    "upside": (medians["trailing_pe"] * eps) / price - 1.0})
    if feps and medians.get("forward_pe") == medians.get("forward_pe"):
        out.append({"method": "Forward P/E (peer median)", "implied_ev": np.nan,
                    "per_share": medians["forward_pe"] * feps,
                    "upside": (medians["forward_pe"] * feps) / price - 1.0})
    return pd.DataFrame(out)
