"""Build + print the historical financials table (Phase 1 sanity check)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import utils, financials

pd.set_option("display.width", 200)

cfg = utils.load_config()
H = financials.build_and_save(cfg)

dollars_m = ["revenue", "gross_profit", "operating_income", "ebitda", "net_income",
             "ocf", "capex", "fcf_reported", "sbc", "chg_nwc", "buybacks", "nopat",
             "fcff", "total_debt", "cash", "st_investments", "net_debt", "net_debt_incl_sti"]
pcts = ["revenue_growth", "gross_margin", "operating_margin", "ebitda_margin",
        "net_margin", "fcf_margin", "sbc_pct_rev", "eff_tax_rate", "roic", "total_take_rate"]
other = ["diluted_shares", "shares_outstanding", "diluted_eps", "tpv_usd_b",
         "payment_transactions_b", "active_accounts_m", "txn_expense_rate_pct"]

T = H.T
print("\n===== DOLLARS ($M) =====")
print(T.loc[dollars_m].applymap(lambda x: f"{x/1e6:,.0f}" if pd.notna(x) else "").to_string())
print("\n===== RATIOS (%) =====")
print(T.loc[pcts].applymap(lambda x: f"{x*100:,.1f}%" if pd.notna(x) else "").to_string())
print("\n===== OTHER =====")
print(T.loc[other].applymap(lambda x: f"{x:,.1f}" if pd.notna(x) else "").to_string())
print(f"\nSaved -> data/processed/historical_financials.csv")
