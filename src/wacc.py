"""Cost of capital (WACC).

CAPM cost of equity with a LIVE risk-free rate (10Y UST via ^TNX), Damodaran's
published ERP, and a levered beta. Cost of debt from the company's own interest
expense / total debt. Weights from market value of equity and (book≈market) debt.
"""
from __future__ import annotations

import yfinance as yf

import utils


def get_live_risk_free(cfg: dict) -> tuple[float, str]:
    """Current US 10Y Treasury yield from ^TNX (quoted in %). Falls back to config."""
    try:
        s = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
        if len(s):
            return float(s.iloc[-1]) / 100.0, "live ^TNX (10Y UST)"
    except Exception:
        pass
    return float(cfg["valuation"]["wacc"]["risk_free_fallback"]), "config fallback (10Y UST)"


def compute_wacc(cfg: dict, market_cap: float, total_debt: float,
                 risk_free: float | None = None) -> dict:
    w = cfg["valuation"]["wacc"]
    if risk_free is None:
        risk_free, rf_src = get_live_risk_free(cfg)
    else:
        rf_src = "supplied"

    beta = w["beta"]
    erp = w["equity_risk_premium"]
    rd = w["cost_of_debt"]
    tax_marginal = cfg["valuation"]["marginal_tax_rate"]

    cost_equity = risk_free + beta * erp
    after_tax_rd = rd * (1 - tax_marginal)

    V = market_cap + total_debt
    we, wd = market_cap / V, total_debt / V
    wacc = we * cost_equity + wd * after_tax_rd

    return {
        "risk_free": risk_free,
        "risk_free_source": rf_src,
        "beta": beta,
        "erp": erp,
        "erp_source": w["erp_source"],
        "cost_of_equity": cost_equity,
        "cost_of_debt_pretax": rd,
        "cost_of_debt_aftertax": after_tax_rd,
        "marginal_tax_rate": tax_marginal,
        "weight_equity": we,
        "weight_debt": wd,
        "market_cap": market_cap,
        "total_debt": total_debt,
        "wacc": wacc,
    }


def format_wacc(w: dict) -> str:
    return (
        f"Risk-free (Rf):        {w['risk_free']*100:5.2f}%   [{w['risk_free_source']}]\n"
        f"Beta (levered):        {w['beta']:5.2f}\n"
        f"Equity risk premium:   {w['erp']*100:5.2f}%   [{w['erp_source']}]\n"
        f"Cost of equity (CAPM): {w['cost_of_equity']*100:5.2f}%\n"
        f"Cost of debt (pre-tax):{w['cost_of_debt_pretax']*100:5.2f}%\n"
        f"Cost of debt (after-t):{w['cost_of_debt_aftertax']*100:5.2f}%   (marginal tax {w['marginal_tax_rate']*100:.0f}%)\n"
        f"Weights  E/V={w['weight_equity']*100:.1f}%  D/V={w['weight_debt']*100:.1f}%\n"
        f"-----------------------------------\n"
        f"WACC:                  {w['wacc']*100:5.2f}%"
    )
