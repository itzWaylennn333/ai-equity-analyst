"""Phase 3 full run: WACC + DCF + comps + football field + charts + saved tables."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import utils, data_loader as dl, financials, wacc as wacc_mod, dcf, comps, charts

cfg = utils.load_config()
yahoo = dl.get_yahoo(cfg, cfg["company"]["ticker"])
facts = dl.get_edgar_facts(cfg, cfg["company"]["cik"])
H = financials.build_historical(cfg, yahoo, facts)
info = yahoo["info"]
charts.setup_style()

fy = 2025
total_debt, cash = H.loc[fy, "total_debt"], H.loc[fy, "cash"]
net_debt = total_debt - cash
base_rev = H.loc[fy, "revenue"]
shares, price, mktcap = info["sharesOutstanding"], info["currentPrice"], info["marketCap"]
ebitda_ttm, fwd_eps = info["ebitda"], info.get("forwardEps")

# WACC + DCF
W = wacc_mod.compute_wacc(cfg, mktcap, total_debt)
proj = dcf.project(cfg, base_rev)
val = dcf.value(cfg, proj, W["wacc"], net_debt, shares, price)
dcf_gordon, dcf_exit = val["gordon"]["per_share"], val["exit"]["per_share"]

# Comps (retry Fiserv once)
try: dl.get_info(cfg, "FI", force=True)
except Exception as e: print("FI retry failed:", e)
C = comps.build_comps(cfg)
med = comps.peer_medians(C)
pypl_ev_ebitda = info.get("enterpriseToEbitda")
pypl_fwd_pe = info.get("forwardPE")

# Implied per-share helpers
def ev_ebitda_ps(mult): return (mult * ebitda_ttm - net_debt) / shares
def fwd_pe_ps(mult):     return mult * fwd_eps

# Sell-side targets
at = {k: info.get(k) for k in ("targetLowPrice", "targetMeanPrice", "targetMedianPrice",
                               "targetHighPrice", "numberOfAnalystOpinions", "recommendationKey")}
print("Sell-side:", at)

# --- proposed 12-month PT anchors (re-rating, NOT full intrinsic value) ---
anchors = {
    "DCF (Gordon, intrinsic)": dcf_gordon,
    "DCF (exit 9x EBITDA)": dcf_exit,
    "Comps fwd P/E (peer median %.1fx)" % med["forward_pe"]: fwd_pe_ps(med["forward_pe"]),
    "Comps EV/EBITDA @8.5x (PYPL-specific)": ev_ebitda_ps(8.5),
    "Sell-side mean": at["targetMeanPrice"],
}
print("\nPT anchors:")
for k, v in anchors.items():
    print(f"  {k:42}: ${v:,.2f}  ({v/price-1:+.0%})")

# Football field ranges (low, high)
ff = [
    ("DCF (WACC/g sensitivity)", dcf_exit, dcf_gordon),
    ("Comps: EV/EBITDA (7x-10x)", ev_ebitda_ps(7), ev_ebitda_ps(10)),
    ("Comps: Forward P/E (9x-12x)", fwd_pe_ps(9), fwd_pe_ps(12)),
    ("Sell-side target range", at["targetLowPrice"], at["targetHighPrice"]),
    ("52-week trading range", info["fiftyTwoWeekLow"], info["fiftyTwoWeekHigh"]),
]
# proposed base-case target = blend (median of the credible re-rating anchors)
pt_blend = float(np.median([dcf_exit, fwd_pe_ps(med["forward_pe"]), ev_ebitda_ps(8.5), at["targetMeanPrice"]]))
print(f"\nProposed base-case 12m target (median of re-rating anchors): ${pt_blend:,.0f} ({pt_blend/price-1:+.0%})")

# --- charts ---
waccs = [W["wacc"] + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
gs = [0.015, 0.02, 0.025, 0.03, 0.035]
grid = dcf.sensitivity(cfg, proj, net_debt, shares, waccs, gs)
p1 = charts.chart_sensitivity_heatmap(cfg, grid, round(W["wacc"], 4), 0.025, price)
p2 = charts.chart_football_field(cfg, ff, price, pt_blend)
p3 = charts.chart_peer_multiples(cfg, C, pypl_ev_ebitda, pypl_fwd_pe)
print("\nCharts:", [Path(p).name for p in (p1, p2, p3)])

# --- save valuation summary table ---
vs = pd.DataFrame({
    "metric": ["WACC", "DCF Gordon $/sh", "DCF exit $/sh", "Comps fwd P/E $/sh",
               "Comps EV/EBITDA @8.5x", "Sell-side mean", "Proposed base PT", "Current price"],
    "value": [f"{W['wacc']*100:.2f}%", f"${dcf_gordon:,.2f}", f"${dcf_exit:,.2f}",
              f"${fwd_pe_ps(med['forward_pe']):,.2f}", f"${ev_ebitda_ps(8.5):,.2f}",
              f"${at['targetMeanPrice']:,.2f}", f"${pt_blend:,.0f}", f"${price:,.2f}"],
})
vs.to_csv(utils.resolve(cfg["output"]["tables_dir"]) / "valuation_summary.csv", index=False)
proj.to_csv(utils.resolve(cfg["data"]["processed_dir"]) / "dcf_projection.csv")
print("\nSaved valuation_summary.csv + dcf_projection.csv")
