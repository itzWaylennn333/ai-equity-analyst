"""Data loading: yfinance (prices, statements, live snapshot) + SEC EDGAR (XBRL).

Everything is cached under ``data/raw/`` and logged to the provenance manifest.
EDGAR is used both for a long, clean annual history and to VALIDATE the yfinance
figures (revenue, net income, shares, cash, debt) against the actual filings --
data APIs have gaps and errors, so filings are ground truth.
"""
from __future__ import annotations

import re
import time
import datetime as _dt

import pandas as pd
import requests
import yfinance as yf

import utils

SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


# --------------------------------------------------------------------------- #
# yfinance
# --------------------------------------------------------------------------- #
_YAHOO_STATEMENTS = {
    "income_annual": "income_stmt",
    "balance_annual": "balance_sheet",
    "cashflow_annual": "cashflow",
    "income_quarterly": "quarterly_income_stmt",
    "balance_quarterly": "quarterly_balance_sheet",
    "cashflow_quarterly": "quarterly_cashflow",
}


def get_yahoo(cfg: dict, ticker: str, force: bool = False) -> dict:
    """Return a dict of {info, prices, <six statements>} for `ticker`, cached.

    Re-pulls only if a cached file is missing or older than ``cache_max_age_days``.
    """
    dcfg = cfg["data"]
    base = f"{dcfg['cache_dir']}/yahoo/{ticker}"
    hist_years = dcfg.get("price_history_years", 6)

    paths = {k: f"{base}/{k}.csv" for k in _YAHOO_STATEMENTS}
    paths["prices"] = f"{base}/prices.csv"
    info_path = f"{base}/info.json"

    fresh = all(utils.cache_is_fresh(p, dcfg["cache_max_age_days"]) for p in paths.values())
    fresh = fresh and utils.cache_is_fresh(info_path, dcfg["cache_max_age_days"])

    if fresh and not force:
        out = {k: utils.load_df(p) for k, p in paths.items() if k != "prices"}
        out["prices"] = utils.load_df(paths["prices"], parse_dates=True)
        out["info"] = utils.load_json(info_path)
        return out

    t = yf.Ticker(ticker)
    out: dict = {}
    for key, attr in _YAHOO_STATEMENTS.items():
        df = getattr(t, attr)
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        utils.save_df(df, paths[key])
        out[key] = df

    start = (_dt.date.today() - _dt.timedelta(days=int(hist_years * 365.25))).isoformat()
    prices = t.history(start=start, auto_adjust=True)
    utils.save_df(prices, paths["prices"])
    out["prices"] = prices

    info = dict(t.info)
    utils.save_json(info, info_path)
    out["info"] = info

    utils.record_provenance(
        dcfg["manifest"], "yahoo", ticker,
        url=f"https://finance.yahoo.com/quote/{ticker}",
        files=list(paths.values()) + [info_path],
    )
    return out


# --------------------------------------------------------------------------- #
# SEC EDGAR
# --------------------------------------------------------------------------- #
def get_edgar_facts(cfg: dict, cik: str, force: bool = False) -> dict:
    """Pull and cache the full XBRL companyfacts JSON for a CIK."""
    dcfg = cfg["data"]
    cik = str(cik).zfill(10)
    path = f"{dcfg['cache_dir']}/edgar/CIK{cik}_facts.json"
    url = SEC_FACTS_URL.format(cik=cik)

    if utils.cache_is_fresh(path, dcfg["cache_max_age_days"]) and not force:
        return utils.load_json(path)

    headers = {"User-Agent": dcfg["sec_user_agent"]}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    facts = r.json()
    utils.save_json(facts, path)
    utils.record_provenance(dcfg["manifest"], "edgar", cik, url=url, files=[path])
    time.sleep(0.2)  # be polite to SEC (<10 req/s)
    return facts


def get_edgar_submissions(cfg: dict, cik: str, force: bool = False) -> dict:
    dcfg = cfg["data"]
    cik = str(cik).zfill(10)
    path = f"{dcfg['cache_dir']}/edgar/CIK{cik}_submissions.json"
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    if utils.cache_is_fresh(path, dcfg["cache_max_age_days"]) and not force:
        return utils.load_json(path)
    headers = {"User-Agent": dcfg["sec_user_agent"]}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    subs = r.json()
    utils.save_json(subs, path)
    time.sleep(0.2)
    return subs


_ANNUAL_FRAME = re.compile(r"CY(\d{4})(Q4I)?$")  # CY2024 (flow) or CY2024Q4I (instant year-end)


def edgar_annual_series(facts: dict, concepts, taxonomy: str = "us-gaap", unit: str = "USD") -> dict:
    """Return {fiscal_year: value} of annual values for the first concept that resolves.

    `concepts` may be a single XBRL concept name or a list of fallbacks (PayPal has
    switched some tags across years). Uses XBRL annual `frame`s so we get clean,
    de-duplicated calendar-year figures straight from the filings.
    """
    if isinstance(concepts, str):
        concepts = [concepts]
    facts_node = facts.get("facts", {}).get(taxonomy, {})
    for concept in concepts:
        node = facts_node.get(concept)
        if not node:
            continue
        units = node.get("units", {})
        series = units.get(unit) or (next(iter(units.values())) if units else None)
        if not series:
            continue
        out: dict[int, float] = {}
        for e in series:
            m = _ANNUAL_FRAME.fullmatch(e.get("frame", ""))
            if m:
                out[int(m.group(1))] = e["val"]
        if out:
            return dict(sorted(out.items()))
    return {}


# Concepts we cross-check against yfinance (with tag fallbacks across vintages).
EDGAR_VALIDATION_CONCEPTS = {
    "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"], "USD"),
    "net_income": (["NetIncomeLoss"], "USD"),
    "operating_income": (["OperatingIncomeLoss"], "USD"),
    "cash_and_equiv": (["CashAndCashEquivalentsAtCarryingValue"], "USD"),
    "total_assets": (["Assets"], "USD"),
    "long_term_debt": (["LongTermDebtNoncurrent", "LongTermDebt"], "USD"),
    "diluted_shares": (["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares"),
    "diluted_eps": (["EarningsPerShareDiluted"], "USD/shares"),
}


def edgar_validation_table(facts: dict) -> pd.DataFrame:
    """Tidy DataFrame of key GAAP concepts (rows) by fiscal year (cols) from EDGAR."""
    rows = {}
    for label, (concepts, unit) in EDGAR_VALIDATION_CONCEPTS.items():
        rows[label] = edgar_annual_series(facts, concepts, unit=unit)
    df = pd.DataFrame(rows).T
    df = df.reindex(sorted(df.columns), axis=1)
    return df
