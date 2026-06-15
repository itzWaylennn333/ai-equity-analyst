"""Historical financial analysis: assemble a clean, validated per-year table.

Sourcing decisions (documented because they matter for defensibility):
- Revenue, gross profit (= PayPal's "transaction margin"), D&A, CapEx, OCF, FCF,
  SBC, debt, cash, shares: yfinance statements.
- GAAP **operating income**: taken from SEC EDGAR (`OperatingIncomeLoss`). yfinance's
  "Operating Income" row is a reclassification that does NOT match the filing
  (e.g. FY2025 yfinance $6,396M vs filed $6,065M = revenue $33,172M - opex $27,107M).
  We use the filing figure.
- Operating metrics (TPV, take rate, transactions, active accounts): hand-sourced
  from the 10-K MD&A (see data/processed/operating_metrics.csv with citations).

FCFF here treats stock-based comp as a REAL expense (config sbc_treatment="expense"):
it is embedded in NOPAT and NOT added back, which is the conservative, defensible
choice for a tech/payments name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils
import data_loader as dl


def _year_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [pd.to_datetime(c).year for c in df.columns]
    return df.loc[:, sorted(df.columns)]


def _row(df: pd.DataFrame, label: str) -> pd.Series:
    return df.loc[label] if label in df.index else pd.Series(dtype="float64")


def build_historical(cfg: dict, yahoo: dict, edgar_facts: dict | None = None) -> pd.DataFrame:
    inc = _year_cols(yahoo["income_annual"])
    bs = _year_cols(yahoo["balance_annual"])
    cf = _year_cols(yahoo["cashflow_annual"])
    years = sorted(set(inc.columns) | set(cf.columns))

    H = pd.DataFrame(index=years)
    H.index.name = "fy"

    # --- Income statement ---
    H["revenue"] = _row(inc, "Total Revenue")
    H["gross_profit"] = _row(inc, "Gross Profit")          # == transaction margin dollars
    H["ebitda"] = _row(inc, "EBITDA")
    H["net_income"] = _row(inc, "Net Income")
    H["pretax_income"] = _row(inc, "Pretax Income")
    H["tax_provision"] = _row(inc, "Tax Provision")
    H["interest_expense"] = _row(inc, "Interest Expense")
    H["diluted_shares"] = _row(inc, "Diluted Average Shares")
    H["da"] = _row(inc, "Reconciled Depreciation")

    # --- GAAP operating income from EDGAR (validated; overrides yfinance) ---
    if edgar_facts is not None:
        op = dl.edgar_annual_series(edgar_facts, ["OperatingIncomeLoss"])
        H["operating_income"] = pd.Series(op).reindex(H.index)
    else:
        H["operating_income"] = _row(inc, "Operating Income")

    # --- Cash flow ---
    H["ocf"] = _row(cf, "Operating Cash Flow")
    H["capex"] = _row(cf, "Capital Expenditure").abs() * -1  # store as negative
    H["fcf_reported"] = _row(cf, "Free Cash Flow")
    H["sbc"] = _row(cf, "Stock Based Compensation")
    H["chg_nwc"] = _row(cf, "Change In Working Capital")     # cash-flow signed
    H["buybacks"] = _row(cf, "Repurchase Of Capital Stock").abs()

    # --- Balance sheet ---
    H["total_debt"] = _row(bs, "Total Debt")
    H["cash"] = _row(bs, "Cash And Cash Equivalents")
    H["st_investments"] = _row(bs, "Other Short Term Investments")
    H["loans_receivable"] = _row(bs, "Loans Receivable")
    H["equity"] = _row(bs, "Stockholders Equity")
    H["shares_outstanding"] = _row(bs, "Ordinary Shares Number")

    # --- Derived ratios ---
    H["revenue_growth"] = H["revenue"].pct_change()
    H["gross_margin"] = H["gross_profit"] / H["revenue"]
    H["operating_margin"] = H["operating_income"] / H["revenue"]
    H["ebitda_margin"] = H["ebitda"] / H["revenue"]
    H["net_margin"] = H["net_income"] / H["revenue"]
    H["fcf_margin"] = H["fcf_reported"] / H["revenue"]
    H["sbc_pct_rev"] = H["sbc"] / H["revenue"]
    H["eff_tax_rate"] = H["tax_provision"] / H["pretax_income"]
    H["diluted_eps"] = H["net_income"] / H["diluted_shares"]

    # Net debt: conservative (debt - cash) and broad (also netting ST investments).
    # Customer funds and loans receivable are deliberately EXCLUDED (pass-through /
    # operating assets of the credit business, not corporate capital).
    H["net_debt"] = H["total_debt"] - H["cash"]
    H["net_debt_incl_sti"] = H["total_debt"] - H["cash"] - H["st_investments"]

    # Unlevered economics. NOPAT uses GAAP operating income (SBC already expensed).
    H["nopat"] = H["operating_income"] * (1 - H["eff_tax_rate"])
    H["fcff"] = H["nopat"] + H["da"] + H["capex"] + H["chg_nwc"]   # capex/chg_nwc already signed
    H["roic"] = H["nopat"] / (H["total_debt"] + H["equity"])       # approx invested capital

    # --- Operating metrics (filing-sourced) ---
    om = pd.read_csv(utils.resolve("data/processed/operating_metrics.csv"))
    om = om.set_index("year")
    H["tpv_usd_b"] = om["tpv_usd_b"].reindex(H.index)
    H["payment_transactions_b"] = om["payment_transactions_b"].reindex(H.index)
    H["active_accounts_m"] = om["active_accounts_m"].reindex(H.index)
    H["txn_expense_rate_pct"] = om["txn_expense_rate_pct"].reindex(H.index)
    H["total_take_rate"] = H["revenue"] / (H["tpv_usd_b"] * 1e9)   # revenue / TPV

    return H


def build_and_save(cfg: dict) -> pd.DataFrame:
    ticker = cfg["company"]["ticker"]
    yahoo = dl.get_yahoo(cfg, ticker)
    facts = dl.get_edgar_facts(cfg, cfg["company"]["cik"])
    H = build_historical(cfg, yahoo, facts)
    out = utils.resolve(cfg["data"]["processed_dir"]) / "historical_financials.csv"
    H.to_csv(out)
    return H
