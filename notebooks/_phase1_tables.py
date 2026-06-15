"""Write a formatted historical summary table (markdown + csv) for the note."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import utils, data_loader as dl, financials

cfg = utils.load_config()
yahoo = dl.get_yahoo(cfg, cfg["company"]["ticker"])
facts = dl.get_edgar_facts(cfg, cfg["company"]["cik"])
H = financials.build_historical(cfg, yahoo, facts)
d = H.loc[[2022, 2023, 2024, 2025]]

def money(s):  return [f"${v/1e9:,.1f}B" if pd.notna(v) else "n/a" for v in s]
def pct(s):    return [f"{v*100:,.1f}%" if pd.notna(v) else "n/a" for v in s]
def num(s, f): return [f.format(v) if pd.notna(v) else "n/a" for v in s]

rows = [
    ("Net revenue", money(d["revenue"])),
    ("  YoY growth", pct(d["revenue_growth"])),
    ("Transaction margin $ (gross profit)", money(d["gross_profit"])),
    ("Operating income (GAAP)", money(d["operating_income"])),
    ("  Operating margin", pct(d["operating_margin"])),
    ("EBITDA", money(d["ebitda"])),
    ("Net income", money(d["net_income"])),
    ("  Net margin", pct(d["net_margin"])),
    ("Diluted EPS", num(d["diluted_eps"], "${:,.2f}")),
    ("Effective tax rate", pct(d["eff_tax_rate"])),
    ("Reported FCF (OCF - capex)", money(d["fcf_reported"])),
    ("FCFF (SBC expensed, unlevered)", money(d["fcff"])),
    ("Stock-based comp", money(d["sbc"])),
    ("  SBC % of revenue", pct(d["sbc_pct_rev"])),
    ("Buybacks", money(d["buybacks"])),
    ("Diluted shares (M)", num(d["diluted_shares"]/1e6, "{:,.0f}")),
    ("ROIC (approx.)", pct(d["roic"])),
    ("Total debt", money(d["total_debt"])),
    ("Cash & equivalents", money(d["cash"])),
    ("Net debt (strict)", money(d["net_debt"])),
    ("Net debt incl. ST investments", money(d["net_debt_incl_sti"])),
    ("TPV", num(d["tpv_usd_b"]/1000, "${:,.2f}T")),
    ("Total take rate", pct(d["total_take_rate"])),
    ("Payment transactions (B)", num(d["payment_transactions_b"], "{:,.1f}")),
    ("Active accounts (M)", num(d["active_accounts_m"], "{:,.0f}")),
]
table = pd.DataFrame({lbl: vals for lbl, vals in rows}, index=[2022, 2023, 2024, 2025]).T
table.columns = [f"FY{c}" for c in table.columns]

md_path = utils.resolve(cfg["output"]["tables_dir"]) / "historical_summary.md"
md_path.parent.mkdir(parents=True, exist_ok=True)
md_path.write_text("# PayPal (PYPL) - Historical Summary (FY2022-FY2025)\n\n" +
                   table.to_markdown() + "\n\n_Source: SEC 10-K filings & yfinance; EDGAR-validated. Author analysis._\n",
                   encoding="utf-8")
table.to_csv(utils.resolve(cfg["output"]["tables_dir"]) / "historical_summary.csv")
print(table.to_string())
print(f"\nSaved -> {md_path.relative_to(ROOT)}")
