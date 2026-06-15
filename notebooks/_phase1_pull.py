"""Phase 1 driver: pull + cache PYPL (yfinance + EDGAR) and validate.

Run: python notebooks/_phase1_pull.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import utils
import data_loader as dl

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.0f}")

cfg = utils.load_config()
ticker = cfg["company"]["ticker"]
cik = cfg["company"]["cik"]

print("Pulling yfinance ...")
y = dl.get_yahoo(cfg, ticker)
print("Pulling EDGAR facts ...")
facts = dl.get_edgar_facts(cfg, cik)

# --- EDGAR annual GAAP series (ground truth) ---
edgar = dl.edgar_validation_table(facts)
print("\n" + "=" * 78)
print("EDGAR XBRL annual figures (USD, shares in raw units):")
print("=" * 78)
print(edgar.to_string())

# --- yfinance annual income statement key rows ---
inc = y["income_annual"]
inc.columns = [str(pd.to_datetime(c).date()) for c in inc.columns]
keep = ["Total Revenue", "Gross Profit", "Operating Income", "EBIT", "EBITDA",
        "Net Income", "Diluted Average Shares", "Diluted EPS", "Tax Provision",
        "Pretax Income", "Interest Expense", "Reconciled Depreciation"]
print("\n" + "=" * 78)
print("yfinance annual income statement (selected rows):")
print("=" * 78)
print(inc.reindex(keep).to_string())

# --- yfinance balance sheet key rows ---
bs = y["balance_annual"]
bs.columns = [str(pd.to_datetime(c).date()) for c in bs.columns]
bkeep = ["Total Debt", "Long Term Debt", "Current Debt", "Net Debt",
         "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
         "Other Short Term Investments", "Loans Receivable", "Investments And Advances",
         "Total Assets", "Stockholders Equity", "Ordinary Shares Number"]
print("\n" + "=" * 78)
print("yfinance annual balance sheet (selected rows):")
print("=" * 78)
print(bs.reindex(bkeep).to_string())

# --- yfinance cash flow key rows ---
cf = y["cashflow_annual"]
cf.columns = [str(pd.to_datetime(c).date()) for c in cf.columns]
ckeep = ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow",
         "Stock Based Compensation", "Depreciation And Amortization",
         "Change In Working Capital", "Repurchase Of Capital Stock"]
print("\n" + "=" * 78)
print("yfinance annual cash flow (selected rows):")
print("=" * 78)
print(cf.reindex(ckeep).to_string())

# --- snapshot ---
info = y["info"]
print("\n" + "=" * 78)
print("Live snapshot (yfinance .info):")
print("=" * 78)
for k in ["currentPrice", "marketCap", "sharesOutstanding", "beta", "totalDebt",
          "totalCash", "enterpriseValue", "trailingPE", "forwardPE", "trailingEps",
          "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "totalRevenue", "ebitda"]:
    print(f"  {k:22}: {info.get(k)}")
print(f"\nPrices cached rows: {len(y['prices'])}  range: {y['prices'].index.min()} -> {y['prices'].index.max()}")
